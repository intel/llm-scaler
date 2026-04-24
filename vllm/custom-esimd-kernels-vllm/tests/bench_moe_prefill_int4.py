"""
Prefill MoE kernel benchmark + comparison with IPEX marlin grouped gemm.

What we bench:
  - moe_int4_prefill_ops.moe_prefill_up_forward_v2       (ours)
  - moe_int4_prefill_ops.moe_prefill_down_forward_v2     (ours)
  - moe_int4_prefill_ops.moe_prefill_full_int4           (ours, end-to-end)
  - torch.ops.torch_ipex.group_mm_int4_out_marlin        (ipex baseline,
      per-expert grouped gemm, K-major marlin layout, same as our kernels)

The ipex op does not include silu+mul or scatter/gather. To keep the
comparison fair we measure just the int4 GEMM portion:

  our_up_gemm_equiv_us  ≈ moe_prefill_up_forward_v2 - silu_mul_cost
  ipex_up_us            ≈ 2 * group_mm_int4_out_marlin (gate + up, same shapes)

For down there is no silu+mul so our_down_us ≈ ipex_down_us + scatter_cost.

Shapes come from Qwen3.5-122B-A10B per-rank (TP=4):
  H = 3072, I = 256, E = 64, top_k = 8
We sweep M ∈ {512, 2048, 8192}.

Usage:
  python tests/bench_moe_prefill_int4.py
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
# Importing ipex registers torch.ops.torch_ipex.* ops (group_mm_int4_out_marlin etc).
import intel_extension_for_pytorch  # noqa: F401

# reuse layout helpers from the accuracy test (IPEX K-major + marlin)
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


def _xpu(t):
    return t.to(DEVICE)


def bench(fn, n_warmup=WARMUP, n_iters=ITERS):
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6  # μs


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

    return dict(x=_xpu(x), logits=_xpu(logits),
                W13_q=_xpu(W13_q), W13_s=_xpu(W13_s),
                W2_q=_xpu(W2_q),   W2_s=_xpu(W2_s),
                topk_weights=_xpu(tw), topk_idx=_xpu(ti),
                M=M, H=H, I=I, E=E, top_k=TK)


def bench_our_up(d, off, tok):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    us = bench(lambda: ops.moe_prefill_up_forward_v2(
        d["x"], d["W13_q"], d["W13_s"], off, tok, d["top_k"]))
    return us


def bench_our_down(d, off, tok, intermediate):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    us = bench(lambda: ops.moe_prefill_down_forward_v2(
        intermediate, d["W2_q"], d["W2_s"], off, tok))
    return us


def bench_our_full(d):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    us = bench(lambda: ops.moe_prefill_full_int4(
        d["x"], d["logits"],
        d["W13_q"], d["W13_s"],
        d["W2_q"],  d["W2_s"],
        d["top_k"], d["E"]))
    return us


# ── IPEX marlin comparison ────────────────────────────────────────────────────
# ipex.group_mm_int4_out_marlin takes (out, input_permuted_by_expert, weight,
# scale, None, rows_for_experts, None, group_size). The input must already be
# row-reordered so expert 0's rows come first, then expert 1's, etc.

def _reorder_input_by_expert(x, topk_idx, top_k, E):
    """Build [M*top_k, H] permuted so expert e's rows are contiguous, matching
    what ipex expects. Returns permuted_x and a rows_for_experts [E] tensor.
    Mirrors what moe_scatter would do in the real MoE pipeline.
    """
    M, H = x.shape
    off, tok = ref_gather(topk_idx.cpu(), E)
    off = off.tolist()
    permuted = torch.empty(M * top_k, H, dtype=x.dtype, device=x.device)
    tok_cpu = tok.cpu().tolist()
    tokens = [p // top_k for p in tok_cpu]
    permuted[:] = x[torch.tensor(tokens, device=x.device)]
    counts = []
    total = M * top_k
    for e in range(E):
        t0 = off[e]
        t1 = off[e + 1] if e + 1 < E else total
        counts.append(t1 - t0)
    rows = torch.tensor(counts, dtype=torch.int64, device=x.device)
    return permuted, rows


def bench_ipex_up(d, perm_x, rows):
    M, H, I, E = d["M"], d["H"], d["I"], d["E"]
    TK = d["top_k"]
    W13_q, W13_s = d["W13_q"], d["W13_s"]
    total = perm_x.size(0)

    out_gate_up = torch.empty(total, 2 * I, dtype=torch.float16, device=DEVICE)

    def once():
        # One grouped gemm produces [total, 2*I] by using W13 as-is.
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out_gate_up,               # out
            perm_x,                    # input
            W13_q,                     # weight [E, K_packed, 2*I]
            W13_s,                     # scale  [E, K_groups, 2*I]
            None,                      # bias
            rows,                      # rows_for_experts
            None,                      # zp
            GROUP_SIZE)
        # Apply silu_mul to mimic the fused part in our kernel
        gate = out_gate_up[:, :I]
        up   = out_gate_up[:, I:]
        _    = (gate / (1 + torch.exp(-gate.float()))).half() * up
    return bench(once)


def bench_ipex_down(d, perm_inter, rows):
    H = d["H"]
    W2_q, W2_s = d["W2_q"], d["W2_s"]
    total = perm_inter.size(0)
    out = torch.empty(total, H, dtype=torch.float16, device=DEVICE)

    def once():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out,
            perm_inter,
            W2_q,                      # [E, K_packed=I/8, H]
            W2_s,                      # [E, K_groups, H]
            None, rows, None, GROUP_SIZE)
    return bench(once)


# ── FLOPs / bandwidth helpers (for context) ───────────────────────────────────

def up_tflops(cfg, us):
    # 2 matmuls (gate + up), each [M*TK, H] * [H, I] int4
    total = cfg["M"] * cfg["top_k"]
    flops = 2 * 2 * total * cfg["I"] * cfg["H"]
    return flops / (us * 1e-6) / 1e12


def down_tflops(cfg, us):
    total = cfg["M"] * cfg["top_k"]
    flops = 2 * total * cfg["H"] * cfg["I"]
    return flops / (us * 1e-6) / 1e12


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"{'config':<22} {'our_up':>12} {'ipex_up':>12} {'ratio':>7}  "
          f"{'our_down':>12} {'ipex_down':>12} {'ratio':>7}  "
          f"{'our_full':>12}")
    print(f"{'':<22} {'us / TFLOPS':>12} {'us / TFLOPS':>12} {'':>7}  "
          f"{'us / TFLOPS':>12} {'us / TFLOPS':>12} {'':>7}  {'us':>12}")
    print("-" * 130)

    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    for cfg in CONFIGS:
        d = make_inputs(cfg)

        off, tok, _pp_unused = ops.moe_prefill_gather_forward_v2(d["topk_idx"], d["E"])
        inter = ops.moe_prefill_up_forward_v2(
            d["x"], d["W13_q"], d["W13_s"], off, tok, d["top_k"])

        perm_x, rows = _reorder_input_by_expert(d["x"], d["topk_idx"], d["top_k"], d["E"])
        # For down: permuted intermediate already has the expert-sorted order,
        # because our up kernel writes intermediate[pair_idx, :]. Reuse our tok
        # vector to permute it:
        tok_cpu = tok.cpu().long()
        perm_inter = inter.index_select(0, tok_cpu.to(DEVICE))

        us_our_up   = bench_our_up(d, off, tok)
        us_our_down = bench_our_down(d, off, tok, inter)
        # moe_prefill_full_int4 currently only has a moe_topk_v2 specialization
        # for (E=256, top_k=8); skip it when we run with E=64 local experts.
        us_our_full = float("nan")

        try:
            us_ipex_up   = bench_ipex_up(d, perm_x, rows)
        except Exception as e:
            print(f"  ipex up failed: {type(e).__name__}: {e}")
            us_ipex_up = float("nan")
        try:
            us_ipex_down = bench_ipex_down(d, perm_inter, rows)
        except Exception as e:
            print(f"  ipex down failed: {type(e).__name__}: {e}")
            us_ipex_down = float("nan")

        r_up   = us_our_up   / us_ipex_up   if us_ipex_up   == us_ipex_up   else float("nan")
        r_down = us_our_down / us_ipex_down if us_ipex_down == us_ipex_down else float("nan")

        print(f"{cfg['name']:<22} "
              f"{us_our_up:7.1f}/{up_tflops(cfg, us_our_up):5.2f} "
              f"{us_ipex_up:7.1f}/{up_tflops(cfg, us_ipex_up):5.2f} "
              f"{r_up:6.2f}x  "
              f"{us_our_down:7.1f}/{down_tflops(cfg, us_our_down):5.2f} "
              f"{us_ipex_down:7.1f}/{down_tflops(cfg, us_ipex_down):5.2f} "
              f"{r_down:6.2f}x  "
              f"{us_our_full:7.1f}")

    print("\nUnits: us / TFLOPS (int4 matmul effective throughput)")
    print("ratio = our_us / ipex_us (higher means we are slower)")


if __name__ == "__main__":
    run()
