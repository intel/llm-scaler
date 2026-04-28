"""Three-way MoE INT4 GEMM bench: IPEX marlin vs vllm-xpu-kernels CUTLASS vs our ESIMD.

Covers both prefill (large M) and decode (small M) shapes to find the best
grouped-GEMM backend for each regime.

Paths:
  A. ipex: group_mm_int4_out_marlin
     Weight: [E, K/8, N] int32 K-major marlin-shuffled, scale: [E, K/GS, N] fp16
  B. vllm-xpu-kernels: cutlass_grouped_gemm_xe2(is_B_int4=True)
     Weight: [E, N, K/2] uint8 (signed int4 after implement_zp), scale: [E, N, K/GS] fp16
  C. our ESIMD: moe_prefill_up/down_forward_v2 (PR384 kernels)
     Weight: [E, K/8, N] int32 K-major marlin-shuffled (same as ipex)

All three paths do the same computation: grouped INT4 GEMM with per-group
fp16 scales and symmetric zero-point=8.

Configs:
  Qwen3.5-122B-A10B TP=4 : H=3072, I=1024, E=256, TK=8
  Qwen3.5-35B-A3B  TP=4 : H=2048, I=512,  E=256, TK=8

M values (per-expert average): prefill {512, 2048, 8192} + decode {1, 4, 8, 16}
  total_tokens = M * TK

Usage: ZE_AFFINITY_MASK=0 python tests/bench_moe_gemm_three_way.py
"""
import gc
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

# M = number of tokens (before ×TK expansion).
# Covers decode (small M) and prefill (large M).
MS_DECODE  = [1, 4, 8, 16]
MS_PREFILL = [512, 2048, 8192]


def bench(fn, n_warmup=WARMUP, n_iters=ITERS):
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6


def tflops(flops, us):
    if us <= 0:
        return 0.0
    return flops / (us * 1e-6) / 1e12


# ─────────────── Weight factories ──────────────────────────────────────────

def make_ipex_weights(H, I, E):
    """IPEX K-major marlin: [E, K/8, N] int32, [E, K/GS, N] fp16"""
    torch.manual_seed(0)
    W13 = (torch.randn(E, 2*I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W2  = (torch.randn(E, H,  I,  dtype=torch.float32) * 0.02).to(DTYPE)
    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q,  W2_s,  _ = quantize_experts_ipex(W2,  GROUP_SIZE)
    return W13_q.to(DEVICE), W13_s.to(DEVICE), W2_q.to(DEVICE), W2_s.to(DEVICE)


def _implement_zp(qweight_uint8):
    """u4 nibbles -> signed int4 packed (copied from vllm-xpu-kernels)."""
    high_u4 = (qweight_uint8 >> 4) & 0x0F
    low_u4  = qweight_uint8 & 0x0F
    high_s8 = high_u4.to(torch.int8) - 8
    low_s8  = low_u4.to(torch.int8) - 8
    def _pack(a, b):
        def _proc(x):
            sign = (x < 0).to(torch.uint8)
            abs3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs3
        return (_proc(a) << 4) | _proc(b)
    return _pack(high_s8, low_s8)


def make_cutlass_weights(H, I, E):
    """CUTLASS format: [E, N, K/2] uint8 (signed int4), [E, N, K/GS] fp16"""
    torch.manual_seed(0)
    GS = GROUP_SIZE
    two_I = 2 * I

    # W13: [E, 2I, H/2] uint8 random, scale [E, 2I, H/GS] fp16
    W13_u8  = torch.randint(0, 0xFF, (E, two_I, H // 2),
                            dtype=torch.uint8, device=DEVICE)
    W13_s   = (torch.rand(E, two_I, H // GS, dtype=torch.float32, device=DEVICE)
               * 0.04 + 0.002).to(DTYPE)

    # W2:  [E, H, I/2] uint8, scale [E, H, I/GS] fp16
    W2_u8   = torch.randint(0, 0xFF, (E, H, I // 2),
                            dtype=torch.uint8, device=DEVICE)
    W2_s    = (torch.rand(E, H, I // GS, dtype=torch.float32, device=DEVICE)
               * 0.04 + 0.002).to(DTYPE)

    # Apply implement_zp per expert (u4 -> s4 packed)
    W13_s4 = torch.empty_like(W13_u8)
    W2_s4  = torch.empty_like(W2_u8)
    for e in range(E):
        W13_s4[e] = _implement_zp(W13_u8[e])
        W2_s4[e]  = _implement_zp(W2_u8[e])

    return W13_s4, W13_s, W2_s4, W2_s


# ─────────────── Routing helpers ───────────────────────────────────────────

def prepare_routing(M, E, TK):
    """Returns (expert_offsets, expert_tokens, rows_for_experts) for IPEX/ESIMD,
    and (num_rows_per_expert, expert_first_token_offset) for CUTLASS."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(1 + M)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    _tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, _p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)

    total = M * TK
    # CUTLASS needs [E+1] int64 prefix sum
    efto = torch.zeros(E + 1, dtype=torch.int64, device=DEVICE)
    efto[:E] = off.to(torch.int64)
    efto[E]  = total

    return dict(off=off, tok=tok, rows=rows, total=total,
                num_rows_per_expert=rows, efto=efto,
                max_m=int(rows.max().item()))


# ─────────────── Bench runners ─────────────────────────────────────────────

def bench_ipex(x_perm, Wq, Ws, rows, N):
    total = x_perm.size(0)
    out = torch.empty(total, N, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out, x_perm, Wq, Ws, None, rows, None, GROUP_SIZE)
    return bench(_fn)


def bench_cutlass(x_sorted, Wq_s4, Ws, efto, N, K, E):
    total = x_sorted.size(0)
    out = torch.empty(total, N, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=x_sorted, ptr_B=Wq_s4, ptr_scales=Ws,
            ptr_bias=None, ptr_D=out,
            expert_first_token_offset=efto,
            N=N, K=K, num_experts=E,
            is_B_int4=True, is_B_mxfp4=False)
    return bench(_fn)


def bench_esimd_up(x, Wq, Ws, off, tok, TK):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    def _fn():
        return ops.moe_prefill_up_forward_v2(x, Wq, Ws, off, tok, TK)
    return bench(_fn)


def bench_esimd_down(inter, Wq, Ws, off, tok):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    def _fn():
        return ops.moe_prefill_down_forward_v2(inter, Wq, Ws, off, tok)
    return bench(_fn)


# ─────────────── Main ──────────────────────────────────────────────────────

def run_one_cfg(cfg):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]
    two_I = 2 * I

    print(f"\n{'='*80}")
    print(f"== {cfg['name']}  H={H}  I={I}  E={E}  TK={TK}")
    print(f"{'='*80}")

    # Check CUTLASS availability
    has_cutlass = hasattr(torch.ops, '_xpu_C') and hasattr(torch.ops._xpu_C, 'cutlass_grouped_gemm_interface')

    # ──── Phase: IPEX INT4 marlin ────
    print("\n  [Allocating IPEX weights...]")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights(H, I, E)
    ipex_results = {}
    esimd_results = {}

    for M in MS_DECODE + MS_PREFILL:
        total = M * TK
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        rt = prepare_routing(M, E, TK)

        # x_perm for IPEX
        x_perm = ops.moe_prefill_scatter_x_forward(x, rt["tok"], TK)

        us_w13 = bench_ipex(x_perm, W13_q, W13_s, rt["rows"], two_I)
        # W2 needs intermediate. For bench purposes just use x_perm truncated.
        inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE) * 0.1
        us_w2  = bench_ipex(inter, W2_q, W2_s, rt["rows"], H)
        ipex_results[M] = (us_w13, us_w2)

        # ESIMD up/down (same weights)
        us_up   = bench_esimd_up(x, W13_q, W13_s, rt["off"], rt["tok"], TK)
        us_down = bench_esimd_down(inter, W2_q, W2_s, rt["off"], rt["tok"])
        esimd_results[M] = (us_up, us_down)

        del x, x_perm, inter
        torch.xpu.empty_cache()

    del W13_q, W13_s, W2_q, W2_s
    torch.xpu.empty_cache(); gc.collect()

    # ──── Phase: CUTLASS INT4 ────
    cutlass_results = {}
    if has_cutlass:
        print("  [Allocating CUTLASS weights...]")
        cW13_s4, cW13_s, cW2_s4, cW2_s = make_cutlass_weights(H, I, E)

        for M in MS_DECODE + MS_PREFILL:
            total = M * TK
            torch.manual_seed(42 + M)
            x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
            rt = prepare_routing(M, E, TK)
            x_perm = ops.moe_prefill_scatter_x_forward(x, rt["tok"], TK)

            try:
                us_w13 = bench_cutlass(x_perm, cW13_s4, cW13_s, rt["efto"], two_I, H, E)
            except Exception as e:
                us_w13 = float('nan')
                print(f"    [CUTLASS W13 M={M} FAIL: {e}]")

            inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE) * 0.1
            try:
                us_w2 = bench_cutlass(inter, cW2_s4, cW2_s, rt["efto"], H, I, E)
            except Exception as e:
                us_w2 = float('nan')
                print(f"    [CUTLASS W2 M={M} FAIL: {e}]")

            cutlass_results[M] = (us_w13, us_w2)
            del x, x_perm, inter
            torch.xpu.empty_cache()

        del cW13_s4, cW13_s, cW2_s4, cW2_s
        torch.xpu.empty_cache(); gc.collect()
    else:
        print("  [CUTLASS not available — skipping]")

    # ──── Print results ────
    for label, ms_list in [("DECODE", MS_DECODE), ("PREFILL", MS_PREFILL)]:
        print(f"\n  --- {label} ---")
        hdr = (f"  {'M':>6} {'stage':<6} "
               f"{'ipex(us)':>10} {'ipex(TF)':>9} "
               f"{'cutlass(us)':>12} {'cut(TF)':>9} {'cut/ipex':>9} "
               f"{'esimd(us)':>11} {'esm(TF)':>9} {'esm/ipex':>9}")
        print(hdr)

        for M in ms_list:
            total = M * TK
            flops_w13 = 2.0 * total * two_I * H
            flops_w2  = 2.0 * total * H * I

            for stage, idx, flops in [("W13", 0, flops_w13), ("W2", 1, flops_w2)]:
                us_i = ipex_results[M][idx]
                us_c = cutlass_results.get(M, (float('nan'), float('nan')))[idx]
                us_e = esimd_results[M][idx]

                c_ratio = f"{us_c/us_i:.3f}x" if us_c == us_c and us_i > 0 else "N/A"
                e_ratio = f"{us_e/us_i:.3f}x" if us_e == us_e and us_i > 0 else "N/A"

                print(f"  {M:>6} {stage:<6} "
                      f"{us_i:>10.1f} {tflops(flops, us_i):>9.2f} "
                      f"{us_c:>12.1f} {tflops(flops, us_c):>9.2f} {c_ratio:>9} "
                      f"{us_e:>11.1f} {tflops(flops, us_e):>9.2f} {e_ratio:>9}")


def main():
    # Force-load _moe_C for remap_hidden_states / moe_gather (may not auto-load)
    try:
        import vllm_xpu_kernels._moe_C  # noqa: F401
    except ImportError:
        pass
    try:
        torch.ops.load_library(
            "/usr/local/lib/python3.12/dist-packages/vllm_xpu_kernels/_moe_C.abi3.so")
    except Exception:
        pass

    for cfg in CONFIGS:
        run_one_cfg(cfg)


if __name__ == "__main__":
    main()
