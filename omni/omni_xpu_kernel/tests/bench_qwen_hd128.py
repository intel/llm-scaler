#!/usr/bin/env python3
"""Benchmark ESIMD SDP vs PyTorch SDPA for Qwen Image HD=128 shapes."""
import time
import torch
import torch.nn.functional as F
from omni_xpu_kernel import sdp

device = torch.device("xpu:0")

configs = [
    ("Qwen 1024 bf16", 1, 24, 1024, 128, torch.bfloat16),
    ("Qwen 4096 bf16", 1, 24, 4096, 128, torch.bfloat16),
    ("Qwen 7099 bf16", 1, 24, 7099, 128, torch.bfloat16),
    ("Flux 4608 fp16",  1, 24, 4608, 128, torch.float16),
    ("Flux 4608 fp16 fast", 1, 24, 4608, 128, torch.float16),
]

def bench(fn, N):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N

print(f"{'Config':22s} {'SDPA ms':>8s} {'SDPA T':>7s}  {'ESIMD ms':>9s} {'ESIMD T':>7s}  {'Speedup':>7s}")
print("-" * 78)

for name, B, H, S, D, dtype in configs:
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    qb = q.permute(0, 2, 1, 3).contiguous()
    kb = k.permute(0, 2, 1, 3).contiguous()
    vb = v.permute(0, 2, 1, 3).contiguous()

    N = 20 if S <= 4096 else 5
    flops = 4 * B * H * S * S * D

    t_sdpa = bench(lambda: F.scaled_dot_product_attention(qb, kb, vb), N)
    t_esimd = bench(lambda: sdp.sdp(q, k, v), N)

    tflops_sdpa = flops / t_sdpa / 1e12
    tflops_esimd = flops / t_esimd / 1e12
    speedup = t_sdpa / t_esimd

    dt = "fp16" if dtype == torch.float16 else "bf16"
    print(f"{name:22s} {t_sdpa*1000:7.2f}ms {tflops_sdpa:6.1f}T  {t_esimd*1000:7.2f}ms  {tflops_esimd:6.1f}T  {speedup:6.2f}x")

    del q, k, v, qb, kb, vb
    torch.xpu.empty_cache()
