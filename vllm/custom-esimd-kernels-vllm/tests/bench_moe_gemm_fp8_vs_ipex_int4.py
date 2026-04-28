"""MoE grouped GEMM bench: FP8 ESIMD V3 vs IPEX INT4 marlin.

Measures our FP8 kernel's throughput at the same (N, K, E, M·TK) shapes
as the INT4 path. FP8 is the upper bound we're aiming for when we rewrite
the INT4 kernel following the same architectural pattern (2D surface +
N-tile-per-WG merge + M-chunk per-WG + MT_MAX accumulators).

The two kernels have different weight / scale formats and the FP8 uses
per-N fp32 scale while INT4 uses per-group fp16 scale — but the
arithmetic intensity in TFLOPS is directly comparable because both do the
same MoE grouped GEMM work (2 * total_tokens * N * K flops).

Shapes:
  W13 GEMM  N=2I  K=H
  W2  GEMM  N=H   K=I

Configs:
  Qwen3.5-122B-A10B TP=4 : H=3072, I=1024, E=256, TK=8
  Qwen3.5-35B-A3B  TP=4 : H=2048, I=512,  E=256, TK=8
  M ∈ {512, 2048, 8192, 16384}
"""
import sys
import time
import pathlib
import numpy as np
import torch
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


def make_int4_weights(H, I, E):
    """IPEX K-major + marlin-shuffled INT4 weights for both W13 and W2."""
    torch.manual_seed(0)
    W13 = (torch.randn(E, 2 * I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W2  = (torch.randn(E, H, I,   dtype=torch.float32) * 0.02).to(DTYPE)
    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q,  W2_s,  _ = quantize_experts_ipex(W2,  GROUP_SIZE)
    return (
        W13_q.to(DEVICE), W13_s.to(DEVICE),
        W2_q.to(DEVICE),  W2_s.to(DEVICE),
    )


def make_fp8_weights(H, I, E):
    """FP8 E5M2 weights + per-N fp32 scale. Weight layout: [E, N, K] uint8.

    Generate from a real fp32 distribution and cast to fp8_e4m3fn on XPU
    (consistent with how upstream FP8 tests build weights), then reinterpret
    the storage as uint8 — avoids random NaN/inf bytes that degrade DPAS
    throughput measurements.
    """
    torch.manual_seed(0)
    W13_f = (torch.randn(E, 2 * I, H, dtype=torch.float32) * 0.3).clamp(-8, 8)
    W2_f  = (torch.randn(E, H, I,    dtype=torch.float32) * 0.3).clamp(-8, 8)
    W13_fp8 = W13_f.to(DEVICE).to(torch.float8_e5m2)
    W2_fp8  = W2_f.to(DEVICE).to(torch.float8_e5m2)
    W13_bytes = W13_fp8.view(torch.uint8)
    W2_bytes  = W2_fp8.view(torch.uint8)
    W13_s = (torch.rand(E, 2 * I, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005)
    W2_s  = (torch.rand(E, H,     dtype=torch.float32, device=DEVICE) * 0.02 + 0.005)
    return W13_bytes, W13_s, W2_bytes, W2_s


def prepare_routing(M, E, TK):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    torch.manual_seed(1)
    logits = (torch.randn(M, E, dtype=torch.float32) * 1.0).to(DTYPE).to(DEVICE)
    _tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, _p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    return off, tok, rows


def expert_idx_for_fp8(off, total_rows):
    """FP8 kernel wants expert_idx [E+1] uint32 — prefix-sum of rows with total
    appended. We already have off [E] int32 = expert_offsets (exclusive
    prefix sum over rows) and total, so just concat.
    """
    E = off.numel()
    out = torch.empty(E + 1, dtype=torch.uint32, device=off.device)
    out[:E] = off.to(torch.uint32)
    out[E]  = torch.tensor(total_rows, dtype=torch.uint32, device=off.device)
    return out


# ─────────── GEMM runners ──────────────────────────────────────────────────
def bench_ipex_w13(x_perm, W13_q, W13_s, rows, two_I):
    total = x_perm.size(0)
    out = torch.empty(total, two_I, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out, x_perm, W13_q, W13_s, None, rows, None, GROUP_SIZE)
    return bench(_fn)


def bench_ipex_w2(inter_perm, W2_q, W2_s, rows, H):
    total = inter_perm.size(0)
    out = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out, inter_perm, W2_q, W2_s, None, rows, None, GROUP_SIZE)
    return bench(_fn)


def bench_fp8_gemm(input_fp16, weight_bytes, scale_fp32, expert_idx,
                   N, K, E, max_m_per_expert):
    total = input_fp16.size(0)
    out = torch.empty(total, N, dtype=DTYPE, device=DEVICE)
    from custom_esimd_kernels_vllm.ops import esimd_moe_gemm_fp8
    def _fn():
        esimd_moe_gemm_fp8(input_fp16, weight_bytes, scale_fp32, out,
                           expert_idx, N, K, E, max_m_per_expert)
    return bench(_fn)


def tflops(flops, us):
    if us <= 0:
        return 0.0
    return flops / (us * 1e-6) / 1e12


# ─────────── Drivers ──────────────────────────────────────────────────────
def run_ipex_phase(cfg):
    """Bench only the IPEX INT4 side; FP8 weights not allocated."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]
    two_I = 2 * I

    W13_q, W13_s, W2_q, W2_s = make_int4_weights(H, I, E)
    results = {}

    for M in MS:
        total = M * TK
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        off, tok, rows = prepare_routing(M, E, TK)
        x_perm = ops.moe_prefill_scatter_x_forward(x, tok, TK)

        us_w13 = bench_ipex_w13(x_perm, W13_q, W13_s, rows, two_I)

        gate_up_perm = torch.empty(total, two_I, dtype=DTYPE, device=DEVICE)
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            gate_up_perm, x_perm, W13_q, W13_s, None, rows, None, GROUP_SIZE)
        inter_perm = ops.moe_prefill_silu_mul_forward(gate_up_perm)

        us_w2 = bench_ipex_w2(inter_perm, W2_q, W2_s, rows, H)

        results[M] = (us_w13, us_w2)

        del x, x_perm, gate_up_perm, inter_perm, off, tok, rows
        torch.xpu.empty_cache()

    del W13_q, W13_s, W2_q, W2_s
    torch.xpu.empty_cache()
    return results


def run_fp8_phase(cfg):
    """Bench only the FP8 side; INT4 weights not allocated."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]
    two_I = 2 * I

    W13_fp8, W13_s_fp8, W2_fp8, W2_s_fp8 = make_fp8_weights(H, I, E)
    results = {}

    for M in MS:
        total = M * TK
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        off, tok, rows = prepare_routing(M, E, TK)
        x_perm = ops.moe_prefill_scatter_x_forward(x, tok, TK)

        expert_idx = expert_idx_for_fp8(off, total)
        max_m = int(rows.max().item())

        us_w13 = bench_fp8_gemm(x_perm, W13_fp8, W13_s_fp8, expert_idx,
                                two_I, H, E, max_m)

        # inter_perm [total, I] fp16: produce via FP8 W13 then silu_mul so the
        # W2 input is consistent with FP8 path.
        gate_up_perm = torch.empty(total, two_I, dtype=DTYPE, device=DEVICE)
        from custom_esimd_kernels_vllm.ops import esimd_moe_gemm_fp8
        esimd_moe_gemm_fp8(x_perm, W13_fp8, W13_s_fp8, gate_up_perm,
                           expert_idx, two_I, H, E, max_m)
        inter_perm = ops.moe_prefill_silu_mul_forward(gate_up_perm)

        us_w2 = bench_fp8_gemm(inter_perm, W2_fp8, W2_s_fp8, expert_idx,
                               H, I, E, max_m)

        results[M] = (us_w13, us_w2)

        del x, x_perm, gate_up_perm, inter_perm, expert_idx, off, tok, rows
        torch.xpu.empty_cache()

    del W13_fp8, W13_s_fp8, W2_fp8, W2_s_fp8
    torch.xpu.empty_cache()
    return results


def run_one_cfg(cfg):
    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]
    two_I = 2 * I

    print(f"\n== {cfg['name']}  H={cfg['H']}  I={cfg['I']}  "
          f"E={cfg['E']}  TK={cfg['TK']} ==")

    ipex_r = run_ipex_phase(cfg)
    fp8_r  = run_fp8_phase(cfg)

    print(f"  {'M':>6}  {'stage':<12} "
          f"{'ipex-int4(us)':>14} {'ipex(TFLOPs)':>13} "
          f"{'ours-fp8(us)':>14} {'fp8(TFLOPs)':>13}  "
          f"{'fp8/ipex':>9}")

    for M in MS:
        total = M * TK
        us_ipex_w13, us_ipex_w2 = ipex_r[M]
        us_fp8_w13,  us_fp8_w2  = fp8_r[M]
        flops_w13 = 2.0 * total * two_I * H
        flops_w2  = 2.0 * total * H * I

        print(f"  {M:>6}  {'W13 (GEMM)':<12} "
              f"{us_ipex_w13:>14.1f} {tflops(flops_w13, us_ipex_w13):>13.2f} "
              f"{us_fp8_w13:>14.1f} {tflops(flops_w13, us_fp8_w13):>13.2f}  "
              f"{us_fp8_w13/us_ipex_w13:>8.3f}x")
        print(f"  {M:>6}  {'W2  (GEMM)':<12} "
              f"{us_ipex_w2:>14.1f} {tflops(flops_w2, us_ipex_w2):>13.2f} "
              f"{us_fp8_w2:>14.1f} {tflops(flops_w2, us_fp8_w2):>13.2f}  "
              f"{us_fp8_w2/us_ipex_w2:>8.3f}x")
        us_ipex_sum = us_ipex_w13 + us_ipex_w2
        us_fp8_sum  = us_fp8_w13  + us_fp8_w2
        print(f"  {M:>6}  {'W13+W2 sum':<12} "
              f"{us_ipex_sum:>14.1f} {'':>13} "
              f"{us_fp8_sum:>14.1f} {'':>13}  "
              f"{us_fp8_sum/us_ipex_sum:>8.3f}x")


def main():
    for cfg in CONFIGS:
        run_one_cfg(cfg)


if __name__ == "__main__":
    main()
