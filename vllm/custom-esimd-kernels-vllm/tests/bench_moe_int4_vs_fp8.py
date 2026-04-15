"""
Performance benchmark: INT4 vs FP8 ESIMD MOE kernels.
Weights in IPEX K-major + marlin shuffled format.

Configs:
  - 1k-512:  hidden=1024, intermediate=512, experts=64
  - 32k-2k:  hidden=32768(?), actually model-realistic dims

Usage:
    python tests/bench_moe_int4_vs_fp8.py
"""

import torch
import time
import statistics
import numpy as np

DEVICE = "xpu"
GROUP_SIZE = 128
PACK_FACTOR = 8

WARMUP = 50
RUNS = 200
BATCH_SIZES = [1, 2, 4, 6, 8, 12, 16, 32]

# Two configs to benchmark
CONFIGS = [
    {
        "name": "Qwen3.5-35B-A3B TP4",
        "hidden_size": 2048,
        "intermediate_size": 256,   # d_ff=512, TP=4: 128→round_up(256)
        "shared_intermediate_size": 128,
        "num_experts": 256,
        "top_k": 8,
    },
    {
        "name": "Qwen3.5-122B-A10B TP4",
        "hidden_size": 3072,
        "intermediate_size": 256,   # d_ff=1024, TP=4: 256→round_up(256)
        "shared_intermediate_size": 256,
        "num_experts": 256,
        "top_k": 8,
    },
]

NUM_SHARED_EXPERTS = 1


def marlin_shuffle_np(qw_np):
    """Simulate IPEX marlin shuffle on numpy uint32 array."""
    shuffled_idx = np.array([0, 4, 1, 5, 2, 6, 3, 7])
    result = np.zeros_like(qw_np)
    for new_pos in range(8):
        old_pos = shuffled_idx[new_pos]
        nibbles = (qw_np >> np.uint32(old_pos * 4)) & np.uint32(0xF)
        result |= nibbles << np.uint32(new_pos * 4)
    return result


def make_int4_ipex_weight(E, N, K):
    """Create INT4 weight in IPEX K-major + marlin shuffled format on XPU."""
    qweight = torch.randint(0, 2**31, (E, N, K // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scales = (torch.randn(E, N, K // GROUP_SIZE, dtype=torch.float16, device=DEVICE) * 0.01).abs() + 0.001

    qw_t = qweight.permute(0, 2, 1).contiguous()
    sc_t = scales.permute(0, 2, 1).contiguous()

    qw_np = qw_t.cpu().numpy().view(np.uint32)
    qw_shuffled = marlin_shuffle_np(qw_np)
    qw_t = torch.from_numpy(qw_shuffled.view(np.int32)).to(DEVICE)

    return qw_t, sc_t


def make_fp8_weight(N, K):
    weight = torch.randn(N, K, dtype=torch.float16, device=DEVICE).to(torch.float8_e5m2)
    scale = torch.tensor([0.01], dtype=torch.float32, device=DEVICE)
    return weight, scale


def bench_config(cfg):
    from custom_esimd_kernels_vllm import moe_int4_ops, moe_ops

    H = cfg["hidden_size"]
    D = cfg["intermediate_size"]
    D_S = cfg["shared_intermediate_size"]
    E = cfg["num_experts"]
    TK = cfg["top_k"]

    print(f"\n{'=' * 75}")
    print(f"Config: {cfg['name']}  (H={H}, D={D}, E={E}, top_k={TK})")
    print(f"{'=' * 75}")

    # ── Router benchmark ──
    print(f"\n  Router (moe_router_forward)")
    print(f"  {'BS':>4s}  {'FP8 (us)':>10s}  {'INT4 (us)':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*36}")

    int4_qw = torch.randint(0, 2**31, (E, H // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    int4_sc = (torch.randn(H // GROUP_SIZE, E, dtype=torch.float16, device=DEVICE) * 0.01).abs()
    fp8_w, fp8_s = make_fp8_weight(E, H)

    for bs in BATCH_SIZES:
        x = torch.randn(bs, H, dtype=torch.float16, device=DEVICE)

        for _ in range(WARMUP):
            moe_ops.moe_router_forward(x, fp8_w, fp8_s)
        torch.xpu.synchronize()
        fp8_t = []
        for _ in range(RUNS):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            moe_ops.moe_router_forward(x, fp8_w, fp8_s)
            torch.xpu.synchronize()
            fp8_t.append((time.perf_counter() - t0) * 1e6)

        for _ in range(WARMUP):
            moe_int4_ops.moe_router_forward_int4(x, int4_qw, int4_sc)
        torch.xpu.synchronize()
        int4_t = []
        for _ in range(RUNS):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            moe_int4_ops.moe_router_forward_int4(x, int4_qw, int4_sc)
            torch.xpu.synchronize()
            int4_t.append((time.perf_counter() - t0) * 1e6)

        fp8_med = statistics.median(fp8_t)
        int4_med = statistics.median(int4_t)
        sp = fp8_med / int4_med if int4_med > 0 else 0
        print(f"  {bs:>4d}  {fp8_med:>10.1f}  {int4_med:>10.1f}  {sp:>7.2f}x")

    # ── Full MoE benchmark ──
    print(f"\n  Full MoE (moe_forward_full)")
    print(f"  {'BS':>4s}  {'FP8 (us)':>10s}  {'INT4 (us)':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*36}")

    w13_qw, w13_sc = make_int4_ipex_weight(E, 2 * D, H)
    w2_qw, w2_sc = make_int4_ipex_weight(E, H, D)
    shared_gu = torch.randn(2 * D_S, H, dtype=torch.float16, device=DEVICE) * 0.02
    shared_d = torch.randn(H, D_S, dtype=torch.float16, device=DEVICE) * 0.02
    shared_gw = torch.randn(1, H, dtype=torch.float16, device=DEVICE) * 0.02

    fp8_w13 = torch.randn(E, H, 2 * D, dtype=torch.float16, device=DEVICE).to(torch.float8_e5m2)
    fp8_w13_s = torch.ones(E, dtype=torch.float32, device=DEVICE) * 0.01
    fp8_w2 = torch.randn(E, D, H, dtype=torch.float16, device=DEVICE).to(torch.float8_e5m2)
    fp8_w2_s = torch.ones(E, dtype=torch.float32, device=DEVICE) * 0.01
    fp8_sgu = torch.randn(1, 2 * D_S, H, dtype=torch.float16, device=DEVICE).to(torch.float8_e5m2)
    fp8_sgu_s = torch.ones(1, dtype=torch.float32, device=DEVICE) * 0.01
    fp8_sd = torch.randn(1, H, D_S, dtype=torch.float16, device=DEVICE).to(torch.float8_e5m2)
    fp8_sd_s = torch.ones(1, dtype=torch.float32, device=DEVICE) * 0.01

    for bs in BATCH_SIZES:
        x = torch.randn(bs, H, dtype=torch.float16, device=DEVICE) * 0.1
        logits = torch.randn(bs, E, dtype=torch.float16, device=DEVICE) * 0.1

        for _ in range(WARMUP):
            moe_ops.moe_forward_full(x, logits, fp8_w13, fp8_w13_s,
                fp8_sgu, fp8_sgu_s, fp8_w2, fp8_w2_s,
                fp8_sd, fp8_sd_s, shared_gw, TK, NUM_SHARED_EXPERTS, E)
        torch.xpu.synchronize()
        fp8_t = []
        for _ in range(RUNS):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            moe_ops.moe_forward_full(x, logits, fp8_w13, fp8_w13_s,
                fp8_sgu, fp8_sgu_s, fp8_w2, fp8_w2_s,
                fp8_sd, fp8_sd_s, shared_gw, TK, NUM_SHARED_EXPERTS, E)
            torch.xpu.synchronize()
            fp8_t.append((time.perf_counter() - t0) * 1e6)

        for _ in range(WARMUP):
            moe_int4_ops.moe_forward_full_int4(x, logits,
                w13_qw, w13_sc, shared_gu, w2_qw, w2_sc, shared_d, shared_gw,
                TK, NUM_SHARED_EXPERTS, E)
        torch.xpu.synchronize()
        int4_t = []
        for _ in range(RUNS):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            moe_int4_ops.moe_forward_full_int4(x, logits,
                w13_qw, w13_sc, shared_gu, w2_qw, w2_sc, shared_d, shared_gw,
                TK, NUM_SHARED_EXPERTS, E)
            torch.xpu.synchronize()
            int4_t.append((time.perf_counter() - t0) * 1e6)

        fp8_med = statistics.median(fp8_t)
        int4_med = statistics.median(int4_t)
        sp = fp8_med / int4_med if int4_med > 0 else 0
        print(f"  {bs:>4d}  {fp8_med:>10.1f}  {int4_med:>10.1f}  {sp:>7.2f}x")


if __name__ == "__main__":
    for cfg in CONFIGS:
        bench_config(cfg)
    print("\nBenchmark complete.")
