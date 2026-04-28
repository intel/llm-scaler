"""122B-TP4 E2E MoE INT4 bench: four paths compared.

PREFILL paths (M = total_tokens for one MoE layer):
  A. IPEX         : ipex.GatedMLPMOE.forward (baseline)
  B. IPEX+ESIMD   : ipex marlin GEMM + our phase-1 ESIMD non-GEMM kernels
  C. CUTLASS(vllm-xpu-kernels): xpu_fused_moe(is_int4=True)

DECODE paths (batch 1/4/8/16/32):
  A. ESIMD decode : moe_int4_ops.moe_forward_full_int4 (current qwen3_next path)
  B. IPEX         : ipex.GatedMLPMOE.forward
  C. CUTLASS      : xpu_fused_moe(is_int4=True)

Config: Qwen3.5-122B-A10B TP=4 — H=3072, I=1024, E=256, TK=8, GS=128

Usage: ZE_AFFINITY_MASK=2 python tests/bench_122b_e2e.py
"""
import gc
import os
import sys
import time
import pathlib
import numpy as np
import torch
import torch.nn.functional as F

import intel_extension_for_pytorch as ipex

# vllm-xpu-kernels
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe, implement_zp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_moe_prefill_int4 import quantize_experts_ipex, GROUP_SIZE

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP = 5
ITERS  = 20

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


# ═══════════════════ Weight factories ═══════════════════════════════════════

def make_ipex_weights():
    """IPEX K-major marlin: [E, K/8, N] int32, [E, K/GS, N] fp16.
    Used by: IPEX GatedMLPMOE, our phase-1 ESIMD path, ESIMD decode kernel."""
    torch.manual_seed(0)
    W13 = (torch.randn(E, TWO_I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W2  = (torch.randn(E, H, I, dtype=torch.float32) * 0.02).to(DTYPE)
    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q, W2_s, _ = quantize_experts_ipex(W2, GROUP_SIZE)
    return W13_q.to(DEVICE), W13_s.to(DEVICE), W2_q.to(DEVICE), W2_s.to(DEVICE)


def make_cutlass_weights():
    """CUTLASS format: [E, N, K/2] uint8 (s4 via implement_zp), [E, N, K/GS] fp16."""
    torch.manual_seed(0)
    W13_u8 = torch.randint(0, 0xFF, (E, TWO_I, H // 2), dtype=torch.uint8, device=DEVICE)
    W13_s  = (torch.rand(E, TWO_I, H // GROUP_SIZE, dtype=torch.float32, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
    W2_u8  = torch.randint(0, 0xFF, (E, H, I // 2), dtype=torch.uint8, device=DEVICE)
    W2_s   = (torch.rand(E, H, I // GROUP_SIZE, dtype=torch.float32, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
    W13_s4 = torch.empty_like(W13_u8)
    W2_s4  = torch.empty_like(W2_u8)
    for e in range(E):
        W13_s4[e] = implement_zp(W13_u8[e])
        W2_s4[e]  = implement_zp(W2_u8[e])
    W13_s4.xpu_fused_moe = True  # skip re-conversion in xpu_fused_moe
    return W13_s4, W13_s, W2_s4, W2_s


def make_ipex_fusion(W13_q, W13_s, W2_q, W2_s):
    """Build ipex.GatedMLPMOE with N-major raw weights (ipex prepacks on first forward).
    We use separate random weights for ipex's prepack — the bench measures throughput, not accuracy."""
    from intel_extension_for_pytorch.llm.modules import GatedMLPMOE
    W13_n = torch.randint(0, 1 << 30, (E, TWO_I, H // 8), dtype=torch.int32, device=DEVICE)
    W2_n  = torch.randint(0, 1 << 30, (E, H, I // 8), dtype=torch.int32, device=DEVICE)
    W13_s_n = torch.randn(E, TWO_I, H // GROUP_SIZE, dtype=DTYPE, device=DEVICE) * 0.02
    W2_s_n  = torch.randn(E, H, I // GROUP_SIZE, dtype=DTYPE, device=DEVICE) * 0.02
    fusion = GatedMLPMOE(W13_n, W2_n, w1_scale_inv=W13_s_n, w2_scale_inv=W2_s_n, is_int4=True)
    return fusion


# ═══════════════════ Routing ═══════════════════════════════════════════════

def make_routing(M):
    """Build topk_weights [M, TK] fp16, topk_ids [M, TK] int32."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    torch.manual_seed(1 + M)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    tw, ti = ops.moe_topk_softmax(logits, TK, E)
    return logits, tw, ti


# ═══════════════════ Path runners ═══════════════════════════════════════════

# ─── Path A: IPEX GatedMLPMOE e2e ───
def bench_ipex_e2e(fusion, x, logits):
    def _fn():
        return fusion(x, False, TK, logits, True,
                      topk_group=None, num_expert_group=None,
                      custom_routing_function=None, scoring_func="softmax",
                      activation="silu")
    return bench(_fn)


# ─── Path B: IPEX GEMM + ESIMD non-GEMM (phase-1) ───
def bench_ipex_esimd_e2e(x, logits, W13_q, W13_s, W2_q, W2_s):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    M_loc = x.size(0)
    total = M_loc * TK

    def _fn():
        tw, ti = ops.moe_topk_softmax(logits, TK, E)
        off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
        x_perm = ops.moe_prefill_scatter_x_forward(x, tok, TK)
        gate_up_perm = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            gate_up_perm, x_perm, W13_q, W13_s, None, rows, None, GROUP_SIZE)
        inter_perm = ops.moe_prefill_silu_mul_forward(gate_up_perm)
        down_perm = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            down_perm, inter_perm, W2_q, W2_s, None, rows, None, GROUP_SIZE)
        return ops.moe_prefill_accumulate_permuted_forward_v2(down_perm, tw, p2p)
    return bench(_fn)


# ─── Path C: CUTLASS xpu_fused_moe e2e ───
def bench_cutlass_e2e(x, tw, ti, cW13, cW13s, cW2, cW2s):
    def _fn():
        return xpu_fused_moe(
            x, cW13, cW13s, None,
            cW2, cW2s, None,
            tw, ti.to(torch.int64), TK,
            "silu", E, is_int4=True)
    return bench(_fn)


# ─── Path D: ESIMD decode e2e (moe_forward_full_int4) ───
def bench_esimd_decode_e2e(x, logits, W13_q, W13_s, W2_q, W2_s):
    from custom_esimd_kernels_vllm.ops import moe_forward_full_int4
    sgu = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
    sd  = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
    sg  = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
    sgus = torch.empty(0, dtype=DTYPE, device=DEVICE)
    sds  = torch.empty(0, dtype=DTYPE, device=DEVICE)

    def _fn():
        return moe_forward_full_int4(
            x, logits, W13_q, W13_s, sgu, sgus,
            W2_q, W2_s, sd, sds, sg,
            TK, 1, E, False)
    return bench(_fn)


# ═══════════════════ PREFILL ═══════════════════════════════════════════════

def run_prefill():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    # total_tokens capped to fit single-card memory
    PREFILL_CASES = [
        (1, 32768),   # batch=1, seq=32k
        (4, 8192),    # batch=4, seq=8k -> same M=32768
        (8, 4096),    # batch=8, seq=4k -> same M=32768
        (12, 2048),   # batch=12, seq=2k -> M=24576
    ]

    print("\n" + "=" * 100)
    print("PREFILL — 122B-TP4  H=3072 I=1024 E=256 TK=8")
    print("=" * 100)

    # ─── IPEX e2e + IPEX+ESIMD e2e (share IPEX weights) ───
    print("  [IPEX weights...]")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    fusion = make_ipex_fusion(W13_q, W13_s, W2_q, W2_s)

    ipex_r = {}
    esimd_r = {}
    for batch, seq_len in PREFILL_CASES:
        M = batch * seq_len
        torch.manual_seed(42 + batch)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        logits_raw = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

        ipex_r[batch] = bench_ipex_e2e(fusion, x, logits_raw)
        esimd_r[batch] = bench_ipex_esimd_e2e(x, logits_raw, W13_q, W13_s, W2_q, W2_s)

        del x, logits_raw; torch.xpu.empty_cache()

    del fusion, W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # ─── CUTLASS e2e ───
    print("  [CUTLASS weights...]")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()

    cut_r = {}
    for batch, seq_len in PREFILL_CASES:
        M = batch * seq_len
        torch.manual_seed(42 + batch)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        _, tw, ti = make_routing(M)

        try:
            cut_r[batch] = bench_cutlass_e2e(x, tw, ti, cW13, cW13s, cW2, cW2s)
        except Exception as e:
            cut_r[batch] = float('nan')
            print(f"    CUTLASS batch={batch}: {e}")

        del x, tw, ti; torch.xpu.empty_cache()

    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    # ─── Print ───
    print(f"\n  {'batch':>5} {'seq':>6} {'M':>8}  "
          f"{'IPEX(us)':>10} {'IPEX+ESIMD(us)':>15} {'ie/ipex':>8} "
          f"{'CUTLASS(us)':>12} {'cut/ipex':>9}")
    for batch, seq_len in PREFILL_CASES:
        M = batch * seq_len
        ui = ipex_r[batch]
        ue = esimd_r[batch]
        uc = cut_r[batch]
        ie_ratio = f"{ue/ui:.3f}x"
        cr = f"{uc/ui:.3f}x" if uc == uc else "N/A"
        print(f"  {batch:>5} {seq_len:>6} {M:>8}  "
              f"{ui:>10.1f} {ue:>15.1f} {ie_ratio:>8} "
              f"{uc:>12.1f} {cr:>9}")


# ═══════════════════ DECODE ════════════════════════════════════════════════

def run_decode():
    DECODE_BATCHES = [1, 4, 8, 16, 32]

    print("\n" + "=" * 100)
    print("DECODE — 122B-TP4  H=3072 I=1024 E=256 TK=8")
    print("=" * 100)

    # ─── IPEX e2e + ESIMD decode e2e (share IPEX weights) ───
    print("  [IPEX weights...]")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    fusion = make_ipex_fusion(W13_q, W13_s, W2_q, W2_s)

    ipex_r = {}
    esimd_dec_r = {}
    for batch in DECODE_BATCHES:
        M = batch
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        logits_raw = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

        ipex_r[batch] = bench_ipex_e2e(fusion, x, logits_raw)
        try:
            esimd_dec_r[batch] = bench_esimd_decode_e2e(x, logits_raw, W13_q, W13_s, W2_q, W2_s)
        except Exception as e:
            esimd_dec_r[batch] = float('nan')
            print(f"    ESIMD decode batch={batch}: {e}")

        del x, logits_raw; torch.xpu.empty_cache()

    del fusion, W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # ─── CUTLASS e2e ───
    print("  [CUTLASS weights...]")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()

    cut_r = {}
    for batch in DECODE_BATCHES:
        M = batch
        torch.manual_seed(42 + M)
        x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
        _, tw, ti = make_routing(M)

        try:
            cut_r[batch] = bench_cutlass_e2e(x, tw, ti, cW13, cW13s, cW2, cW2s)
        except Exception as e:
            cut_r[batch] = float('nan')
            print(f"    CUTLASS decode batch={batch}: {e}")

        del x, tw, ti; torch.xpu.empty_cache()

    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    # ─── Print ───
    print(f"\n  {'batch':>5} {'total':>6}  "
          f"{'ESIMD_dec(us)':>14} {'IPEX(us)':>10} {'esm/ipex':>9} "
          f"{'CUTLASS(us)':>12} {'cut/ipex':>9}")
    for batch in DECODE_BATCHES:
        total = batch * TK
        ue = esimd_dec_r[batch]
        ui = ipex_r[batch]
        uc = cut_r[batch]
        er = f"{ue/ui:.3f}x" if ue == ue else "N/A"
        cr = f"{uc/ui:.3f}x" if uc == uc else "N/A"
        print(f"  {batch:>5} {total:>6}  "
              f"{ue:>14.1f} {ui:>10.1f} {er:>9} "
              f"{uc:>12.1f} {cr:>9}")


def main():
    run_prefill()
    run_decode()


if __name__ == "__main__":
    main()
