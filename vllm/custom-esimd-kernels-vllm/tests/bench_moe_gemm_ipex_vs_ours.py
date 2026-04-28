"""GEMM-only bench: PR384 up/down kernels vs ipex group_mm_int4_out_marlin.

Decides the Phase-2 direction: can we drop ipex grouped-GEMM in favour of
our own moe_prefill_up/down_forward_v2 kernels? The two grouped-GEMMs are
the only ipex ops left in the MoE prefill path. If our kernels are within
~15% of ipex, we can switch. Otherwise we need to tune them first.

Both sides consume the same IPEX K-major + marlin-shuffled weights, and
operate over an expert-sorted routing. Accounting is as follows:

  ipex W13 path : group_mm_int4_out_marlin on x_perm -> gate_up_perm
                  (no silu, no gather; pure GEMM)
  ours W13 path : moe_prefill_up_forward_v2 on x    -> intermediate
                  (embeds gather + INT4 DPAS + silu*up + scatter-by-pair_idx)

The comparison is not apples-to-apples for W13 because ours includes silu*up
and scatter, which ipex's GEMM doesn't. To make it fair we also compare:

  ipex W13 + our silu_mul + scatter-equivalent cost
        vs.
  ours moe_prefill_up_forward_v2

so we're asking: "given the same *semantic* stage (expert-sorted input ->
intermediate ready for W2), which is faster?"

W2 is cleaner: ipex GEMM produces down_perm [M*TK, H] and ours
moe_prefill_down_forward_v2 produces expert_output [M*TK, H] indexed by
pair_idx. Both are pure INT4 marlin GEMM.

TFLOPS: counts int4 GEMM MACs (1 MAC = 2 flops, int4 is still reported as
dequant fp16 GEMM TFLOPS so the numbers are comparable).
  W13 GEMM : 2 * total * 2*I * H flops
  W2  GEMM : 2 * total * H * I flops

Configs:
  122B-TP4 : H=3072, I=1024, E=256, TK=8
  35B-TP4  : H=2048, I=512,  E=256, TK=8
  M ∈ {512, 2048, 8192, 16384}
"""
import sys
import time
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import intel_extension_for_pytorch  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_moe_prefill_int4 import quantize_experts_ipex, GROUP_SIZE

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP = 5
ITERS  = 20

CONFIGS = [
    dict(name="122B-TP4", H=3072, I=1024, E=256, TK=8),
    dict(name="35B-TP4",  H=2048, I=512,  E=256, TK=8),
]
MS = [512, 2048, 8192, 16384]


def bench(fn, n_warmup=WARMUP, n_iters=ITERS):
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6


def make_weights(H, I, E):
    """Allocate weights once per config. Quantise on CPU then move to XPU to
    avoid hitting the XPU allocator with the fp32 reference tensors."""
    torch.manual_seed(0)

    W13 = (torch.randn(E, 2 * I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W2  = (torch.randn(E, H, I,   dtype=torch.float32) * 0.02).to(DTYPE)

    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q,  W2_s,  _ = quantize_experts_ipex(W2,  GROUP_SIZE)
    return (
        W13_q.to(DEVICE), W13_s.to(DEVICE),
        W2_q.to(DEVICE),  W2_s.to(DEVICE),
    )


def prepare_routing(M, E, TK):
    """Route each pair_idx to a random expert and build expert-sorted helpers.

    Returns the pair_idx-sorted token tensor (for our kernels) and the
    rows_for_experts + gather output (for ipex GEMM + scatter setup)."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(1)
    x      = (torch.randn(M, 1, dtype=torch.float32) * 0.1)  # placeholder shape
    logits = (torch.randn(M, E, dtype=torch.float32) * 1.0).to(DTYPE).to(DEVICE)

    # Real topk
    tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, pair_to_perm, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    return dict(topk_weights=tw, topk_ids=ti,
                expert_offsets=off, expert_tokens=tok,
                pair_to_perm=pair_to_perm, rows_for_experts=rows)


# ─────────── W13 GEMM stage ────────────────────────────────────────────────
def bench_ipex_w13(x_perm, W13_q, W13_s, rows, two_I):
    total = x_perm.size(0)
    out = torch.empty(total, two_I, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out, x_perm, W13_q, W13_s, None, rows, None, GROUP_SIZE)
    return bench(_fn)


def bench_ours_up(x, W13_q, W13_s, off, tok, TK):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    def _fn():
        return ops.moe_prefill_up_forward_v2(x, W13_q, W13_s, off, tok, TK)
    return bench(_fn)


# ─────────── W2 GEMM stage ─────────────────────────────────────────────────
def bench_ipex_w2(inter_perm, W2_q, W2_s, rows, H):
    total = inter_perm.size(0)
    out = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out, inter_perm, W2_q, W2_s, None, rows, None, GROUP_SIZE)
    return bench(_fn)


def bench_ours_down(inter, W2_q, W2_s, off, tok):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    def _fn():
        return ops.moe_prefill_down_forward_v2(inter, W2_q, W2_s, off, tok)
    return bench(_fn)


# ─────────── TFLOPS helper ─────────────────────────────────────────────────
def tflops(flops, us):
    # us = microseconds; flops count is for one call
    if us <= 0:
        return 0.0
    return flops / (us * 1e-6) / 1e12


# ─────────── Driver ────────────────────────────────────────────────────────
def run_one_cfg(cfg):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    print(f"\n== {cfg['name']}  H={cfg['H']}  I={cfg['I']}  "
          f"E={cfg['E']}  TK={cfg['TK']} ==")
    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]
    two_I = 2 * I

    W13_q, W13_s, W2_q, W2_s = make_weights(H, I, E)

    print(f"  {'M':>6}  {'stage':<12} {'ipex(us)':>10} {'ipex(TFLOPs)':>13} "
          f"{'ours(us)':>10} {'ours(TFLOPs)':>13}  {'ours/ipex':>10}")

    for M in MS:
        total = M * TK

        # Build inputs
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        routing = prepare_routing(M, E, TK)
        off  = routing["expert_offsets"]
        tok  = routing["expert_tokens"]
        rows = routing["rows_for_experts"]

        # Produce x_perm for the ipex side (our kernel builds this implicitly)
        x_perm = ops.moe_prefill_scatter_x_forward(x, tok, TK)

        # ── W13 GEMM ──
        us_ipex_w13 = bench_ipex_w13(x_perm, W13_q, W13_s, rows, two_I)
        us_ours_up  = bench_ours_up(x, W13_q, W13_s, off, tok, TK)
        # Note: ours includes gather+silu*up+scatter, ipex is pure GEMM.
        flops_w13 = 2.0 * total * two_I * H
        print(f"  {M:>6}  {'W13 (GEMM)':<12} "
              f"{us_ipex_w13:>10.1f} {tflops(flops_w13, us_ipex_w13):>13.2f} "
              f"{us_ours_up:>10.1f} {tflops(flops_w13, us_ours_up):>13.2f} "
              f"{us_ours_up/us_ipex_w13:>9.3f}x")

        # ── W2 GEMM (apples-to-apples: both take [total, I] and produce [total, H]) ──
        # For ipex we need a real inter_perm buffer. Compute it via ipex W13 then silu_mul.
        gate_up_perm = torch.empty(total, two_I, dtype=DTYPE, device=DEVICE)
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            gate_up_perm, x_perm, W13_q, W13_s, None, rows, None, GROUP_SIZE)
        inter_perm = ops.moe_prefill_silu_mul_forward(gate_up_perm)
        us_ipex_w2 = bench_ipex_w2(inter_perm, W2_q, W2_s, rows, H)

        # For ours we need an intermediate in pair_idx order (the up kernel's output).
        inter = ops.moe_prefill_up_forward_v2(x, W13_q, W13_s, off, tok, TK)
        us_ours_down = bench_ours_down(inter, W2_q, W2_s, off, tok)

        flops_w2 = 2.0 * total * H * I
        print(f"  {M:>6}  {'W2  (GEMM)':<12} "
              f"{us_ipex_w2:>10.1f} {tflops(flops_w2, us_ipex_w2):>13.2f} "
              f"{us_ours_down:>10.1f} {tflops(flops_w2, us_ours_down):>13.2f} "
              f"{us_ours_down/us_ipex_w2:>9.3f}x")

        # ── Combined W13 + W2 (what the pipeline actually pays) ──
        us_ipex_sum = us_ipex_w13 + us_ipex_w2
        us_ours_sum = us_ours_up + us_ours_down
        print(f"  {M:>6}  {'W13+W2 sum':<12} "
              f"{us_ipex_sum:>10.1f} {'':>13} "
              f"{us_ours_sum:>10.1f} {'':>13} "
              f"{us_ours_sum/us_ipex_sum:>9.3f}x")

        del x, x_perm, gate_up_perm, inter_perm, inter
        torch.xpu.empty_cache()

    del W13_q, W13_s, W2_q, W2_s
    torch.xpu.empty_cache()


def main():
    for cfg in CONFIGS:
        run_one_cfg(cfg)


if __name__ == "__main__":
    main()
