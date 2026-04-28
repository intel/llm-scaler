"""122B-TP4 full MoE INT4 bench: IPEX vs CUTLASS vs ESIMD decode kernel.

Covers:
  A. Prefill — 32k input, batch 1/4/8/12 (total_tokens = batch * 32768 * TK)
     - IPEX group_mm_int4_out_marlin (W13 + W2)
     - CUTLASS cutlass_grouped_gemm_interface(is_B_int4=True) (W13 + W2)

  B. Decode — batch 1/4/8/16/32 (total_tokens = batch * TK)
     - IPEX group_mm_int4_out_marlin (W13 + W2)
     - CUTLASS cutlass_grouped_gemm_interface(is_B_int4=True) (W13 + W2)
     - ESIMD moe_forward_full_int4 (end-to-end decode kernel)

Config: Qwen3.5-122B-A10B TP=4 — H=3072, I=1024, E=256, TK=8, GS=128

Usage: ZE_AFFINITY_MASK=2 python tests/bench_122b_full.py
"""
import gc
import sys
import time
import pathlib
import numpy as np
import torch
import intel_extension_for_pytorch  # noqa: F401

# Ensure vllm-xpu-kernels ops are registered
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_moe_prefill_int4 import quantize_experts_ipex, GROUP_SIZE

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP = 5
ITERS = 20

# 122B-TP4 shape
H, I, E, TK = 3072, 1024, 256, 8
TWO_I = 2 * I


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
    if us <= 0: return 0.0
    return flops / (us * 1e-6) / 1e12


# ─────────── Weights ───────────────────────────────────────────────────────

def make_ipex_weights():
    torch.manual_seed(0)
    W13 = (torch.randn(E, TWO_I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W2  = (torch.randn(E, H, I, dtype=torch.float32) * 0.02).to(DTYPE)
    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q, W2_s, _ = quantize_experts_ipex(W2, GROUP_SIZE)
    return W13_q.to(DEVICE), W13_s.to(DEVICE), W2_q.to(DEVICE), W2_s.to(DEVICE)


def _implement_zp(qw):
    hi = (qw >> 4) & 0x0F
    lo = qw & 0x0F
    hi_s = hi.to(torch.int8) - 8
    lo_s = lo.to(torch.int8) - 8
    def _p(x):
        s = (x < 0).to(torch.uint8)
        a = (x.view(torch.uint8) & 0x7).to(torch.uint8)
        return (s << 3) | a
    return (_p(hi_s) << 4) | _p(lo_s)


def make_cutlass_weights():
    torch.manual_seed(0)
    W13_u8 = torch.randint(0, 0xFF, (E, TWO_I, H // 2), dtype=torch.uint8, device=DEVICE)
    W13_s  = (torch.rand(E, TWO_I, H // GROUP_SIZE, dtype=torch.float32, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
    W2_u8  = torch.randint(0, 0xFF, (E, H, I // 2), dtype=torch.uint8, device=DEVICE)
    W2_s   = (torch.rand(E, H, I // GROUP_SIZE, dtype=torch.float32, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
    W13_s4 = torch.empty_like(W13_u8)
    W2_s4  = torch.empty_like(W2_u8)
    for e in range(E):
        W13_s4[e] = _implement_zp(W13_u8[e])
        W2_s4[e]  = _implement_zp(W2_u8[e])
    return W13_s4, W13_s, W2_s4, W2_s


# ─────────── Routing ───────────────────────────────────────────────────────

def prepare_routing(M):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    torch.manual_seed(1 + M)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, _p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    total = M * TK
    efto = torch.zeros(E + 1, dtype=torch.int64, device=DEVICE)
    efto[:E] = off.to(torch.int64)
    efto[E] = total
    return dict(tw=tw, ti=ti, off=off, tok=tok, rows=rows, total=total,
                efto=efto, logits=logits)


# ─────────── Bench helpers ─────────────────────────────────────────────────

def bench_ipex_gemm(x_sorted, Wq, Ws, rows, N):
    total = x_sorted.size(0)
    out = torch.empty(total, N, dtype=DTYPE, device=DEVICE)
    def _fn():
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            out, x_sorted, Wq, Ws, None, rows, None, GROUP_SIZE)
    return bench(_fn)


def bench_cutlass_gemm(x_sorted, Wq_s4, Ws, efto, N, K):
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


def bench_esimd_decode(x, logits, W13_q, W13_s, W2_q, W2_s):
    """ESIMD moe_forward_full_int4 — end-to-end decode MoE kernel.
    Needs dummy shared expert weights."""
    from custom_esimd_kernels_vllm.ops import moe_forward_full_int4
    # Shared expert: fp16 (not int4), dummy
    sgu = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
    sd  = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
    sg_dummy = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
    # Scales for shared expert (ignored for fp16 shared)
    sgus = torch.empty(0, dtype=DTYPE, device=DEVICE)
    sds  = torch.empty(0, dtype=DTYPE, device=DEVICE)

    def _fn():
        return moe_forward_full_int4(
            x, logits,
            W13_q, W13_s,
            sgu, sgus,
            W2_q, W2_s,
            sd, sds,
            sg_dummy,
            TK, 1, E, False)  # use_ggml_layout=False (IPEX K-major)
    return bench(_fn)


# ─────────── Main ──────────────────────────────────────────────────────────

def run_prefill():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    print("\n" + "=" * 90)
    print("PREFILL — 122B-TP4  H=3072 I=1024 E=256 TK=8  (input_len=32768)")
    print("=" * 90)

    # Prefill batches: limited by single-card memory for activations.
    # batch=1 32k -> x_perm = 262144 * 3072 * 2B = 1.5 GB (fits)
    # batch=4 32k -> 6 GB activations (won't fit on a card with 1.25GB weights)
    # Use shorter seq_len for larger batches to keep total_tokens manageable.
    # total_tokens capped at ~300k to stay within ~2 GB activation budget.
    PREFILL_CASES = [
        (1, 32768),   # batch=1, seq=32k -> M=32768
        (4, 8192),    # batch=4, seq=8k  -> M=32768
        (8, 4096),    # batch=8, seq=4k  -> M=32768
        (12, 2048),   # batch=12, seq=2k -> M=24576
    ]

    # --- IPEX phase ---
    print("  [IPEX weights...]")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    ipex_r = {}
    for batch, seq_len in PREFILL_CASES:
        M = batch * seq_len
        total = M * TK
        torch.manual_seed(42 + batch)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        rt = prepare_routing(M)
        x_perm = ops.moe_prefill_scatter_x_forward(x, rt["tok"], TK)
        del x; torch.xpu.empty_cache()
        # Bench W13
        w13_us = bench_ipex_gemm(x_perm, W13_q, W13_s, rt["rows"], TWO_I)
        # Reuse x_perm memory for inter (same total rows, just different width)
        del x_perm; torch.xpu.empty_cache()
        inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE) * 0.1
        w2_us  = bench_ipex_gemm(inter, W2_q, W2_s, rt["rows"], H)
        ipex_r[batch] = (w13_us, w2_us)
        del inter, rt; torch.xpu.empty_cache()
    del W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # --- CUTLASS phase ---
    print("  [CUTLASS weights...]")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()
    cut_r = {}
    for batch, seq_len in PREFILL_CASES:
        M = batch * seq_len
        total = M * TK
        torch.manual_seed(42 + batch)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        rt = prepare_routing(M)
        x_perm = ops.moe_prefill_scatter_x_forward(x, rt["tok"], TK)
        del x; torch.xpu.empty_cache()
        try:
            w13_us = bench_cutlass_gemm(x_perm, cW13, cW13s, rt["efto"], TWO_I, H)
        except Exception as e:
            w13_us = float('nan'); print(f"    CUTLASS W13 batch={batch}: {e}")
        del x_perm; torch.xpu.empty_cache()
        inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE) * 0.1
        try:
            w2_us = bench_cutlass_gemm(inter, cW2, cW2s, rt["efto"], H, I)
        except Exception as e:
            w2_us = float('nan'); print(f"    CUTLASS W2 batch={batch}: {e}")
        cut_r[batch] = (w13_us, w2_us)
        del inter, rt; torch.xpu.empty_cache()
    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    # --- Print ---
    print(f"\n  {'batch':>5} {'M':>8} {'stage':<6} "
          f"{'ipex(us)':>10} {'ipex(TF)':>9} "
          f"{'cutlass(us)':>12} {'cut(TF)':>9} {'cut/ipex':>9}")
    for batch, seq_len in PREFILL_CASES:
        M = batch * seq_len
        total = M * TK
        for stage, idx, flops in [("W13", 0, 2.0*total*TWO_I*H),
                                  ("W2",  1, 2.0*total*H*I)]:
            ui = ipex_r[batch][idx]
            uc = cut_r[batch][idx]
            cr = f"{uc/ui:.3f}x" if uc == uc else "N/A"
            print(f"  {batch:>5} {M:>8} {stage:<6} "
                  f"{ui:>10.1f} {tflops(flops,ui):>9.2f} "
                  f"{uc:>12.1f} {tflops(flops,uc):>9.2f} {cr:>9}")


def run_decode():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    print("\n" + "=" * 90)
    print("DECODE — 122B-TP4  H=3072 I=1024 E=256 TK=8")
    print("=" * 90)

    # --- IPEX + ESIMD decode phase (same weights) ---
    print("  [IPEX weights (shared with ESIMD decode)...]")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    ipex_r = {}
    esimd_r = {}
    for batch in [1, 4, 8, 16, 32]:
        M = batch  # decode: M = batch_size tokens
        total = M * TK
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        rt = prepare_routing(M)
        x_perm = ops.moe_prefill_scatter_x_forward(x, rt["tok"], TK)
        inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE) * 0.1

        # IPEX GEMM only
        w13_us = bench_ipex_gemm(x_perm, W13_q, W13_s, rt["rows"], TWO_I)
        w2_us  = bench_ipex_gemm(inter, W2_q, W2_s, rt["rows"], H)
        ipex_r[batch] = (w13_us, w2_us)

        # ESIMD decode (end-to-end: topk + scatter + up + silu + down + accumulate)
        try:
            esimd_us = bench_esimd_decode(x, rt["logits"], W13_q, W13_s, W2_q, W2_s)
        except Exception as e:
            esimd_us = float('nan')
            print(f"    ESIMD decode batch={batch}: {e}")
        esimd_r[batch] = esimd_us

        del x, x_perm, inter, rt; torch.xpu.empty_cache()
    del W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # --- CUTLASS phase ---
    print("  [CUTLASS weights...]")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()
    cut_r = {}
    for batch in [1, 4, 8, 16, 32]:
        M = batch
        total = M * TK
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        rt = prepare_routing(M)
        x_perm = ops.moe_prefill_scatter_x_forward(x, rt["tok"], TK)
        inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE) * 0.1
        try:
            w13_us = bench_cutlass_gemm(x_perm, cW13, cW13s, rt["efto"], TWO_I, H)
        except Exception as e:
            w13_us = float('nan'); print(f"    CUTLASS W13 batch={batch}: {e}")
        try:
            w2_us = bench_cutlass_gemm(inter, cW2, cW2s, rt["efto"], H, I)
        except Exception as e:
            w2_us = float('nan'); print(f"    CUTLASS W2 batch={batch}: {e}")
        cut_r[batch] = (w13_us, w2_us)
        del x, x_perm, inter, rt; torch.xpu.empty_cache()
    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    # --- Print ---
    print(f"\n  {'batch':>5} {'total':>6} {'stage':<8} "
          f"{'ipex(us)':>10} {'ipex(TF)':>9} "
          f"{'cutlass(us)':>12} {'cut(TF)':>9} {'cut/ipex':>9} "
          f"{'esimd_e2e(us)':>14} {'esm/ipex':>9}")
    for batch in [1, 4, 8, 16, 32]:
        M = batch
        total = M * TK
        for stage, idx, flops in [("W13", 0, 2.0*total*TWO_I*H),
                                  ("W2",  1, 2.0*total*H*I)]:
            ui = ipex_r[batch][idx]
            uc = cut_r[batch][idx]
            cr = f"{uc/ui:.3f}x" if uc == uc else "N/A"
            # ESIMD is e2e, only show on W13 row
            if stage == "W13":
                ue = esimd_r[batch]
                er = f"{ue/(ipex_r[batch][0]+ipex_r[batch][1]):.3f}x" if ue == ue else "N/A"
                print(f"  {batch:>5} {total:>6} {stage:<8} "
                      f"{ui:>10.1f} {tflops(flops,ui):>9.2f} "
                      f"{uc:>12.1f} {tflops(flops,uc):>9.2f} {cr:>9} "
                      f"{ue:>14.1f} {er:>9}")
            else:
                print(f"  {batch:>5} {total:>6} {stage:<8} "
                      f"{ui:>10.1f} {tflops(flops,ui):>9.2f} "
                      f"{uc:>12.1f} {tflops(flops,uc):>9.2f} {cr:>9} "
                      f"{'':>14} {'':>9}")
        # Summary row: ipex W13+W2 vs cutlass W13+W2 vs esimd e2e
        ui_sum = ipex_r[batch][0] + ipex_r[batch][1]
        uc_sum = cut_r[batch][0] + cut_r[batch][1]
        ue = esimd_r[batch]
        cr_sum = f"{uc_sum/ui_sum:.3f}x" if uc_sum == uc_sum else "N/A"
        er_sum = f"{ue/ui_sum:.3f}x" if ue == ue else "N/A"
        print(f"  {batch:>5} {total:>6} {'SUM':<8} "
              f"{ui_sum:>10.1f} {'':>9} "
              f"{uc_sum:>12.1f} {'':>9} {cr_sum:>9} "
              f"{ue:>14.1f} {er_sum:>9}")
        print()


def main():
    run_prefill()
    run_decode()


if __name__ == "__main__":
    main()
