"""End-to-end MoE prefill bench: ipex.GatedMLPMOE vs. our Phase-1 all-kernel path.

Compares the two paths that sym_int4.py chooses between based on
USE_ESIMD_MOE_PREFILL:

  A. ipex baseline (`ipex.llm.modules.GatedMLPMOE.forward`)
     Built-in fused pipeline:
       topk_softmax -> moe_rows_counts -> moe_scatter
       -> group_mm_int4_out_marlin(W13) -> silu_and_mul
       -> group_mm_int4_out_marlin(W2)  -> moe_gather

  B. our Phase-1 path (`_esimd_prefill_moe_apply`):
       moe_topk_softmax -> moe_prefill_gather_forward_v2
       -> moe_prefill_scatter_x_forward
       -> group_mm_int4_out_marlin(W13)
       -> moe_prefill_silu_mul_forward
       -> group_mm_int4_out_marlin(W2)
       -> moe_prefill_accumulate_permuted_forward_v2

Both paths share the same IPEX K-major + marlin-shuffled weights, so the
two grouped GEMMs are identical. Any delta is from the non-GEMM stages.

Note: ipex.GatedMLPMOE's forward mutates `.data` of the W13/W2 tensors on
first call (transpose + marlin_shuffle + cat). We therefore construct an
ipex fusion with independently-allocated weights so it doesn't disturb the
tensors used by path B.

Configs:
  Qwen3.5-122B-A10B TP=4 : H=3072, I=1024, E=256, TK=8
  Qwen3.5-35B-A3B  TP=4 : H=2048, I=512,  E=256, TK=8
  M in {512, 2048, 8192, 16384}
"""
import os
import sys
import time
import pathlib
import numpy as np
import torch
import torch.nn.functional as F

import intel_extension_for_pytorch as ipex  # noqa: F401

# Reuse quantize helpers from the accuracy test file (same dir)
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
    """Build all per-config weights (K-major marlin for our path + N-major
    raw INT4 for ipex's GatedMLPMOE prepack). Only allocated once per config
    to avoid repeated large allocations causing OOM / device-lost."""
    torch.manual_seed(0)

    # Per-expert fp16 weights — [E, 2I, H] and [E, H, I].
    W13 = (torch.randn(E, 2 * I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W2  = (torch.randn(E, H, I,   dtype=torch.float32) * 0.02).to(DTYPE)

    # Our path: IPEX K-major + marlin-shuffled, on XPU.
    W13_q, W13_s, _ = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q,  W2_s,  _ = quantize_experts_ipex(W2,  GROUP_SIZE)
    W13_q = W13_q.to(DEVICE); W13_s = W13_s.to(DEVICE)
    W2_q  = W2_q.to(DEVICE);  W2_s  = W2_s.to(DEVICE)

    # ipex path: raw N-major int32 packed weights — ipex's GatedMLPMOE
    # prepack transposes + marlin-shuffles these on the first forward.
    W13_n = torch.randint(0, 1 << 30, (E, 2 * I, H // 8),
                          dtype=torch.int32, device=DEVICE)
    W2_n  = torch.randint(0, 1 << 30, (E, H, I // 8),
                          dtype=torch.int32, device=DEVICE)
    W13_s_n = torch.randn(E, 2 * I, H // GROUP_SIZE, dtype=DTYPE, device=DEVICE) * 0.02
    W2_s_n  = torch.randn(E, H, I // GROUP_SIZE,     dtype=DTYPE, device=DEVICE) * 0.02

    return dict(
        W13_q=W13_q, W13_s=W13_s,
        W2_q=W2_q,   W2_s=W2_s,
        W13_n=W13_n, W2_n=W2_n,
        W13_s_n=W13_s_n, W2_s_n=W2_s_n,
        H=H, I=I, E=E,
    )


def make_activations(M, H, E):
    torch.manual_seed(1)
    x      = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    logits = (torch.randn(M, E, dtype=torch.float32) * 1.0).to(DTYPE).to(DEVICE)
    return x, logits


# ───────────────────── Path A: ipex baseline ──────────────────────────────
def build_ipex_fusion(W):
    """One-shot GatedMLPMOE construction (the prepack runs inside forward)."""
    from intel_extension_for_pytorch.llm.modules import GatedMLPMOE
    return GatedMLPMOE(
        W["W13_n"], W["W2_n"],
        w1_scale_inv=W["W13_s_n"], w2_scale_inv=W["W2_s_n"],
        is_int4=True,
    )


def run_ipex_baseline(fusion, x, logits, TK):
    def _once():
        return fusion(
            x, False, TK, logits, True,
            topk_group=None, num_expert_group=None,
            custom_routing_function=None, scoring_func="softmax",
            activation="silu",
        )
    return bench(_once)


# ───────────────────── Path B: Phase-1 kernelized ─────────────────────────
def run_phase1_kernelized(W, x, logits, TK):
    """Replays _esimd_prefill_moe_apply directly (no sym_int4 layer wrapper)."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    M, H, I, E = x.size(0), W["H"], W["I"], W["E"]
    two_I = 2 * I
    total = M * TK

    def _once():
        # 1. topk
        tw, ti = ops.moe_topk_softmax(logits, TK, E)
        # 2. gather
        off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
        # 3. scatter_x
        x_perm = ops.moe_prefill_scatter_x_forward(x, tok, TK)
        # 4. W13 GEMM
        gate_up_perm = torch.empty(total, two_I, dtype=DTYPE, device=DEVICE)
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            gate_up_perm, x_perm, W["W13_q"], W["W13_s"],
            None, rows, None, GROUP_SIZE)
        # 5. silu * up
        inter_perm = ops.moe_prefill_silu_mul_forward(gate_up_perm)
        # 6. W2 GEMM
        down_perm = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
        torch.ops.torch_ipex.group_mm_int4_out_marlin(
            down_perm, inter_perm, W["W2_q"], W["W2_s"],
            None, rows, None, GROUP_SIZE)
        # 7. accumulate (fused inverse-permute + weighted sum)
        return ops.moe_prefill_accumulate_permuted_forward_v2(down_perm, tw, p2p)

    return bench(_once)


# ───────────────────── Driver ─────────────────────────────────────────────
def run_one_cfg(cfg):
    print(f"\n== {cfg['name']}  H={cfg['H']}  I={cfg['I']}  E={cfg['E']}  TK={cfg['TK']} ==")
    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]

    W = make_weights(H, I, E)
    fusion = build_ipex_fusion(W)

    for M in MS:
        x, logits = make_activations(M, H, E)

        try:
            ipex_us = run_ipex_baseline(fusion, x, logits, TK)
        except Exception as e:
            ipex_us = float('nan')
            print(f"  M={M:>6}  [ipex baseline FAIL] {type(e).__name__}: {e}")

        try:
            ours_us = run_phase1_kernelized(W, x, logits, TK)
        except Exception as e:
            ours_us = float('nan')
            print(f"  M={M:>6}  [phase1 kernelized FAIL] {type(e).__name__}: {e}")

        del x, logits
        torch.xpu.empty_cache()

        if ipex_us == ipex_us and ours_us == ours_us:
            delta = (ours_us - ipex_us) / ipex_us * 100
            flag = "faster" if delta < 0 else "slower"
            print(f"  M={M:>6}    ipex={ipex_us:>8.1f}us    ours={ours_us:>8.1f}us    "
                  f"ours/ipex={ours_us/ipex_us:.3f}x ({delta:+.1f}%, {flag})")
        else:
            print(f"  M={M:>6}    ipex={ipex_us}    ours={ours_us}")

    # Free per-config weights before next config.
    del W, fusion
    torch.xpu.empty_cache()


def main():
    for cfg in CONFIGS:
        run_one_cfg(cfg)


if __name__ == "__main__":
    main()
