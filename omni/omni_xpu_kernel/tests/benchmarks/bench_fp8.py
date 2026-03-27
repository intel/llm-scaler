"""
Performance benchmarks for FP8 GEMM kernels

Compares oneDNN FP8 kernels against PyTorch native (dequant + F.linear) implementations.

Usage:
    python -m benchmarks.bench_fp8
"""

import time
import torch
from typing import Optional


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def benchmark_fp8_linear(m: int, n: int, k: int, dtype: torch.dtype,
                         has_bias: bool = False, warmup: int = 10, iters: int = 100):
    """Benchmark onednn_w8a16_fp8 vs PyTorch native (dequant + linear)."""
    from omni_xpu_kernel import linear
    
    device = "xpu"
    x = torch.randn(m, k, device=device, dtype=dtype)
    
    # Create FP8 weight
    weight_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)
    scales = (weight_fp32.abs().max(dim=1).values / 448.0).to(torch.float32)
    scales = torch.clamp(scales, min=1e-12)
    qweight = (weight_fp32 / scales.unsqueeze(1)).to(torch.float8_e4m3fn)
    
    bias = None
    if has_bias:
        bias = torch.randn(n, device=device, dtype=dtype)
    
    # =========================================================================
    # Benchmark oneDNN FP8 kernel
    # =========================================================================
    for _ in range(warmup):
        _ = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
    torch.xpu.synchronize()
    kernel_time = (time.perf_counter() - start) / iters * 1000
    
    # =========================================================================
    # Benchmark PyTorch native (dequant + linear)
    # =========================================================================
    for _ in range(warmup):
        weight_dequant = qweight.to(dtype) * scales.unsqueeze(1).to(dtype)
        _ = torch.nn.functional.linear(x, weight_dequant, bias)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        weight_dequant = qweight.to(dtype) * scales.unsqueeze(1).to(dtype)
        _ = torch.nn.functional.linear(x, weight_dequant, bias)
    torch.xpu.synchronize()
    pytorch_time = (time.perf_counter() - start) / iters * 1000
    
    # Standard GEMM-only FLOPs: 2 * M * N * K
    flops = 2 * m * n * k
    # Convert from milliseconds to seconds: time_ms / 1000
    # TFLOPS = (FLOPs / (time_s)) / 1e12 = (FLOPs / (time_ms / 1000)) / 1e12
    # TFLOPS = (FLOPs * 1000) / (time_ms * 1e12) = FLOPs / (time_ms * 1e9)
    kernel_tflops = flops / (kernel_time * 1e9) if kernel_time > 0 else 0
    pytorch_tflops = flops / (pytorch_time * 1e9) if pytorch_time > 0 else 0
    
    speedup = pytorch_time / kernel_time if kernel_time > 0 else 0
    return {
        "kernel_time_ms": kernel_time,
        "pytorch_time_ms": pytorch_time,
        "kernel_tflops": kernel_tflops,
        "pytorch_tflops": pytorch_tflops,
        "speedup": speedup,
    }


def run_fp8_benchmarks():
    """Run FP8 GEMM benchmarks."""
    print("=" * 140)
    print("FP8 GEMM Benchmark: oneDNN onednn_w8a16_fp8 vs PyTorch Native (Dequant + Linear)")
    print("=" * 140)
    print(f"{'M':>6} {'N':>6} {'K':>6} {'Bias':>6} {'Dtype':>10} │ "
          f"{'oneDNN(ms)':>12} {'PyTorch(ms)':>12} {'oneDNN(TF)':>12} {'PyTorch(TF)':>12} {'Speedup':>10}")
    print(f"{'':-<6} {'':-<6} {'':-<6} {'':-<6} {'':-<10} ┼ {'':-<12} {'':-<12} {'':-<12} {'':-<12} {'':-<10}")
    
    # Common shapes in LLM/Diffusion (e.g. attention/FFN)
    for (m, n, k) in [(1, 4096, 4096), (16, 4096, 4096), (1, 12288, 4096), (1, 4096, 12288)]:
        for dtype in [torch.float16, torch.bfloat16]:
            for has_bias in [False, True]:
                result = benchmark_fp8_linear(m, n, k, dtype, has_bias=has_bias)
                dtype_str = str(dtype).split('.')[-1]
                bias_str = "Yes" if has_bias else "No"
                print(f"{m:>6} {n:>6} {k:>6} {bias_str:>6} {dtype_str:>10} │ "
                      f"{result['kernel_time_ms']:>12.4f} {result['pytorch_time_ms']:>12.4f} "
                      f"{result['kernel_tflops']:>12.4f} {result['pytorch_tflops']:>12.4f} "
                      f"{result['speedup']:>9.2f}x")

    print("\nKnown bad-shape focus:")
    for (m, n, k) in [
        (4096, 12288, 4096),
        (4608, 16384, 4096),
        (4096, 24576, 4096),
        (4096, 4096, 12288),
        (4608, 36864, 4096),
        (4608, 4096, 16384),
    ]:
        for dtype in [torch.float16, torch.bfloat16]:
            for has_bias in [False, True]:
                result = benchmark_fp8_linear(m, n, k, dtype, has_bias=has_bias, warmup=3, iters=10)
                dtype_str = str(dtype).split('.')[-1]
                bias_str = "Yes" if has_bias else "No"
                print(f"{m:>6} {n:>6} {k:>6} {bias_str:>6} {dtype_str:>10} │ "
                      f"{result['kernel_time_ms']:>12.4f} {result['pytorch_time_ms']:>12.4f} "
                      f"{result['kernel_tflops']:>12.4f} {result['pytorch_tflops']:>12.4f} "
                      f"{result['speedup']:>9.2f}x")


def main():
    if not has_xpu():
        print("XPU not available, skipping benchmarks")
        return
    
    run_fp8_benchmarks()


if __name__ == "__main__":
    main()
