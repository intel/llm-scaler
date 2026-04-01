#!/usr/bin/env python3
"""Benchmark ESIMD SDP vs PyTorch SDPA for LTX-2 video shapes (H=32, HD=64)."""
import time
import torch
import torch.nn.functional as F
from omni_xpu_kernel import sdp

device = torch.device("xpu:0")

# LTX-2: H=32, HD=64
# First pass (640x360, 121 frames): seq ~3600-54000 depending on compression
# Second pass (1280x720, after spatial 2x upscale): seq up to ~360000
configs = [
    # name,                B, H,    S,    D, dtype
    ("bf16 S=1024",        1, 32, 1024,   64, torch.bfloat16),
    ("bf16 S=2048",        1, 32, 2048,   64, torch.bfloat16),
    ("bf16 S=4096",        1, 32, 4096,   64, torch.bfloat16),
    ("bf16 S=8192",        1, 32, 8192,   64, torch.bfloat16),
    ("bf16 S=16384",       1, 32, 16384,  64, torch.bfloat16),
    ("bf16 S=32768",       1, 32, 32768,  64, torch.bfloat16),
    ("fp16 S=1024",        1, 32, 1024,   64, torch.float16),
    ("fp16 S=4096",        1, 32, 4096,   64, torch.float16),
    ("fp16 S=16384",       1, 32, 16384,  64, torch.float16),
    ("fp16 S=32768",       1, 32, 32768,  64, torch.float16),
]


def bench(fn, N):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N


print(f"{'Config':22s} {'SDPA ms':>9s} {'SDPA T':>7s}  {'ESIMD ms':>9s} {'ESIMD T':>7s}  {'Speedup':>7s}")
print("-" * 78)

for name, B, H, S, D, dtype in configs:
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    qb = q.permute(0, 2, 1, 3).contiguous()
    kb = k.permute(0, 2, 1, 3).contiguous()
    vb = v.permute(0, 2, 1, 3).contiguous()

    N = 10 if S <= 8192 else 3
    flops = 4 * B * H * S * S * D

    t_sdpa = bench(lambda: F.scaled_dot_product_attention(qb, kb, vb), N)
    t_esimd = bench(lambda: sdp.sdp(q, k, v), N)
    speedup = t_sdpa / t_esimd
    tflops_sdpa = flops / t_sdpa / 1e12
    tflops_esimd = flops / t_esimd / 1e12

    print(
        f"{name:22s} {t_sdpa*1000:8.2f}ms {tflops_sdpa:6.1f}T  "
        f"{t_esimd*1000:8.2f}ms  {tflops_esimd:6.1f}T  "
        f"{speedup:6.2f}x"
    )

    del q, k, v, qb, kb, vb
    torch.xpu.empty_cache()
