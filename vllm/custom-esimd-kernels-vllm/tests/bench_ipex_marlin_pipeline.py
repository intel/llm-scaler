"""
Estimate the performance of an end-to-end MoE prefill pipeline built by
composing ipex marlin grouped-gemm with torch-side scatter/silu_mul/gather.

No kernel changes needed — this measures what Option 3 (lean on ipex marlin
for the GEMM work, keep everything else in torch/our small kernels) would buy
us compared with our current fully-ESIMD moe_prefill_full_int4 path.

Breakdown measured:
  scatter   : index_select x by expert_tokens
  up_gemm   : ipex marlin gate_up (x_perm -> gate_up_perm)
  silu_mul  : fused silu(gate) * up
  down_gemm : ipex marlin down (inter_perm -> down_perm)
  accumulate: weighted sum over top_k slots (use our accumulate kernel)
  total     : sum of above

Compared against:
  our_up + our_down + our_accumulate (skip scatter because our up kernel
  embeds scatter implicitly via expert_tokens gather).

Shape: Qwen3.5-122B-A10B TP=4 per-rank (H=3072, I=256, E=64, TK=8).
M in {512, 2048, 8192}.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import intel_extension_for_pytorch  # noqa: F401

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_moe_prefill_int4 import (
    quantize_experts_ipex, ref_gather, GROUP_SIZE,
)

DEVICE = "xpu"
WARMUP = 5
ITERS  = 20

CONFIGS = [
    dict(name="122B-TP4, M=512",  M=512,  H=3072, I=256, E=64, top_k=8),
    dict(name="122B-TP4, M=2048", M=2048, H=3072, I=256, E=64, top_k=8),
    dict(name="122B-TP4, M=8192", M=8192, H=3072, I=256, E=64, top_k=8),
]


def bench(fn, n_warmup=WARMUP, n_iters=ITERS):
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6


def make_inputs(cfg):
    torch.manual_seed(0)
    M, H, I, E, TK = cfg["M"], cfg["H"], cfg["I"], cfg["E"], cfg["top_k"]

    x = torch.randn(M, H, dtype=torch.float16) * 0.1
    W13 = torch.randn(E, 2 * I, H, dtype=torch.float16) * 0.02
    W2  = torch.randn(E, H, I,   dtype=torch.float16) * 0.02
    logits = torch.randn(M, E, dtype=torch.float16)

    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q,  W2_s,  _ = quantize_experts_ipex(W2,  GROUP_SIZE)

    probs = F.softmax(logits.float(), dim=-1)
    tw, ti = torch.topk(probs, TK, dim=-1)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).half()
    ti = ti.to(torch.int32)
    return dict(
        x=x.to(DEVICE), logits=logits.to(DEVICE),
        W13_q=W13_q.to(DEVICE), W13_s=W13_s.to(DEVICE),
        W2_q=W2_q.to(DEVICE),  W2_s=W2_s.to(DEVICE),
        topk_weights=tw.to(DEVICE), topk_idx=ti.to(DEVICE),
        M=M, H=H, I=I, E=E, top_k=TK,
    )


def setup_permutation(d):
    """Use our gather kernel to get expert_offsets + expert_tokens +
    pair_to_perm (inverse permutation), then derive rows_for_experts on-device.
    """
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    off, tok, pair_to_perm = ops.moe_prefill_gather_forward_v2(d["topk_idx"], d["E"])
    total = d["M"] * d["top_k"]
    # ipex group_mm_int4_out_marlin reinterprets rows_for_experts as int32*,
    # so keep the tensor in int32 (int64 silently produces wrong results).
    rows = torch.empty(d["E"], dtype=torch.int32, device=DEVICE)
    rows[:-1] = off[1:] - off[:-1]
    rows[-1]  = int(total) - off[-1]
    return off, tok, pair_to_perm, rows


def bench_ipex_pipeline(d, off, tok, pair_to_perm, rows):
    """Compose: scatter(x) -> ipex_marlin(W13) -> silu_mul -> ipex_marlin(W2)
               -> fused accumulate_permuted (ours, single HBM pass).

    The fused-permuted accumulate kernel reads `down_perm` directly using
    `pair_to_perm` as an inverse-permutation index, so there is no separate
    torch.index_select pass between the second GEMM and the weighted sum.
    """
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    M, H, I, E, TK = d["M"], d["H"], d["I"], d["E"], d["top_k"]
    total = M * TK
    two_I = 2 * I

    # Preallocate once (the real serving path would also reuse these buffers)
    x_perm       = torch.empty(total, H,    dtype=torch.float16, device=DEVICE)
    gate_up_perm = torch.empty(total, two_I, dtype=torch.float16, device=DEVICE)
    inter_perm   = torch.empty(total, I,    dtype=torch.float16, device=DEVICE)
    down_perm    = torch.empty(total, H,    dtype=torch.float16, device=DEVICE)

    # Derive the token row for each pair_idx slot (=pair_idx // TK). Precompute.
    pair_to_token = (tok.to(torch.int64) // TK)   # [total]

    def stage_scatter():
        torch.index_select(d["x"], 0, pair_to_token, out=x_perm)

    def stage_up_gemm():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            gate_up_perm, x_perm, d["W13_q"], d["W13_s"],
            None, rows, None, GROUP_SIZE)

    def stage_silu_mul():
        gate = gate_up_perm[:, :I]
        up   = gate_up_perm[:, I:]
        inter_perm.copy_(F.silu(gate) * up)

    def stage_down_gemm():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            down_perm, inter_perm, d["W2_q"], d["W2_s"],
            None, rows, None, GROUP_SIZE)

    def stage_acc_fused():
        # One HBM pass, fused inverse-permutation + weighted sum.
        ops.moe_prefill_accumulate_permuted_forward_v2(
            down_perm, d["topk_weights"], pair_to_perm)

    us_scatter = bench(stage_scatter)
    us_up_gemm = bench(lambda: (stage_scatter(), stage_up_gemm())[1])
    us_up_gemm -= us_scatter
    us_silu    = bench(lambda: (stage_scatter(), stage_up_gemm(), stage_silu_mul())[2])
    us_silu   -= us_scatter + us_up_gemm
    us_down_gemm = bench(lambda: (stage_scatter(), stage_up_gemm(), stage_silu_mul(), stage_down_gemm())[3])
    us_down_gemm -= us_scatter + us_up_gemm + us_silu
    us_acc      = bench(lambda: (stage_scatter(), stage_up_gemm(), stage_silu_mul(),
                                 stage_down_gemm(), stage_acc_fused())[4])
    us_acc     -= us_scatter + us_up_gemm + us_silu + us_down_gemm

    def full():
        stage_scatter()
        stage_up_gemm()
        stage_silu_mul()
        stage_down_gemm()
        stage_acc_fused()
    us_total = bench(full)
    return dict(scatter=us_scatter, up_gemm=us_up_gemm, silu=us_silu,
                down_gemm=us_down_gemm, acc=us_acc, total=us_total)


def bench_our_pipeline(d, off, tok):
    """Run our existing up+down+accumulate kernels as an end-to-end baseline."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    M, H, I, E, TK = d["M"], d["H"], d["I"], d["E"], d["top_k"]
    total = M * TK

    inter = torch.empty(total, I, dtype=torch.float16, device=DEVICE)
    exp_out = torch.empty(total, H, dtype=torch.float16, device=DEVICE)
    final = torch.empty(M, H, dtype=torch.float16, device=DEVICE)

    def once():
        inter = ops.moe_prefill_up_forward_v2(
            d["x"], d["W13_q"], d["W13_s"], off, tok, TK)
        exp_out = ops.moe_prefill_down_forward_v2(
            inter, d["W2_q"], d["W2_s"], off, tok)
        final = ops.moe_prefill_accumulate_forward_v2(exp_out, d["topk_weights"])
        return final

    us_up   = bench(lambda: ops.moe_prefill_up_forward_v2(
        d["x"], d["W13_q"], d["W13_s"], off, tok, TK))
    # Down needs a real intermediate:
    _inter = ops.moe_prefill_up_forward_v2(d["x"], d["W13_q"], d["W13_s"], off, tok, TK)
    us_down = bench(lambda: ops.moe_prefill_down_forward_v2(
        _inter, d["W2_q"], d["W2_s"], off, tok))
    _expo  = ops.moe_prefill_down_forward_v2(_inter, d["W2_q"], d["W2_s"], off, tok)
    us_acc  = bench(lambda: ops.moe_prefill_accumulate_forward_v2(_expo, d["topk_weights"]))
    us_total = bench(once)
    return dict(up=us_up, down=us_down, acc=us_acc, total=us_total)


def run():
    for cfg in CONFIGS:
        d = make_inputs(cfg)
        off, tok, pair_to_perm, rows = setup_permutation(d)

        ours = bench_our_pipeline(d, off, tok)
        ipex = bench_ipex_pipeline(d, off, tok, pair_to_perm, rows)

        print(f"\n=== {cfg['name']} ===")
        print(f"  ours: up={ours['up']:8.1f}  down={ours['down']:8.1f}  "
              f"acc={ours['acc']:6.1f}  total={ours['total']:8.1f} us")
        print(f"  ipex: scatter={ipex['scatter']:6.1f}  up_gemm={ipex['up_gemm']:7.1f}  "
              f"silu={ipex['silu']:6.1f}  down_gemm={ipex['down_gemm']:7.1f}  "
              f"acc={ipex['acc']:6.1f}  total={ipex['total']:8.1f} us")
        speedup = ours['total'] / ipex['total']
        print(f"  speedup (ours/ipex): {speedup:.2f}x")


if __name__ == "__main__":
    run()
