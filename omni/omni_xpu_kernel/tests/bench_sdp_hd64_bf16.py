#!/usr/bin/env python3
"""Benchmark ESIMD SDP kernel vs PyTorch SDPA for head_dim=64 (FP16 + BF16)."""
import time
import torch
import torch.nn.functional as F
from omni_xpu_kernel import sdp

device = torch.device("xpu:0")

configs = [
    # name,         B, H,    S,   D, dtype
    ("SD3.5 1024",  1, 24, 1024,  64, torch.float16),
    ("SD3.5 4096",  1, 24, 4096,  64, torch.float16),
    ("LTX  4096",   1, 32, 4096,  64, torch.float16),
    ("LTX 16384",   1, 32, 16384, 64, torch.float16),
    ("z-img 1024",  1, 24, 1024,  64, torch.bfloat16),
    ("z-img 4096",  1, 24, 4096,  64, torch.bfloat16),
    ("z-img 16384", 1, 24, 16384, 64, torch.bfloat16),
]


def bench_kernel(fn, N):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N


def run():
    dt_str = lambda d: "fp16" if d == torch.float16 else "bf16"
    header = f"{'Config':16s} {'H':>3s} {'S':>6s} {'D':>4s} {'dtype':>5s}  {'SDPA ms':>8s} {'SDPA T':>7s}  {'ESIMD ms':>9s} {'ESIMD T':>7s}  {'Speedup':>7s}"
    print(header)
    print("-" * len(header))

    for name, B, H, S, D, dtype in configs:
        # BLHD layout for ESIMD kernel
        q_blhd = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k_blhd = torch.randn(B, S, H, D, device=device, dtype=dtype)
        v_blhd = torch.randn(B, S, H, D, device=device, dtype=dtype)

        # BHLD layout for PyTorch SDPA
        q_bhld = q_blhd.permute(0, 2, 1, 3).contiguous()
        k_bhld = k_blhd.permute(0, 2, 1, 3).contiguous()
        v_bhld = v_blhd.permute(0, 2, 1, 3).contiguous()

        N = 20 if S <= 4096 else 5
        flops = 4 * B * H * S * S * D

        # PyTorch SDPA
        t_sdpa = bench_kernel(lambda: F.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld), N)
        tflops_sdpa = flops / t_sdpa / 1e12

        # ESIMD kernel
        t_esimd = bench_kernel(lambda: sdp.sdp(q_blhd, k_blhd, v_blhd), N)
        tflops_esimd = flops / t_esimd / 1e12

        speedup = t_sdpa / t_esimd
        print(
            f"{name:16s} {H:3d} {S:6d} {D:4d} {dt_str(dtype):>5s}  "
            f"{t_sdpa*1000:7.2f}ms {tflops_sdpa:6.1f}T  "
            f"{t_esimd*1000:7.2f}ms  {tflops_esimd:6.1f}T  "
            f"{speedup:6.2f}x"
        )

        del q_blhd, k_blhd, v_blhd, q_bhld, k_bhld, v_bhld
        torch.xpu.empty_cache()


if __name__ == "__main__":
    run()
