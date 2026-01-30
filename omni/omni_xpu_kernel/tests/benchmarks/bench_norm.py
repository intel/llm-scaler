"""
Performance benchmarks for normalization kernels

Compares ESIMD kernels against PyTorch native implementations.

Usage:
    python -m benchmarks.bench_norm
"""

import time
import torch


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def benchmark_rms_norm(batch_size: int, hidden_size: int, dtype: torch.dtype,
                       warmup: int = 10, iters: int = 100):
    """Benchmark RMSNorm: ESIMD vs PyTorch native."""
    from omni_xpu_kernel import norm
    
    input = torch.randn(batch_size, hidden_size, device="xpu", dtype=dtype)
    weight = torch.ones(hidden_size, device="xpu", dtype=dtype)
    eps = 1e-6
    
    # =========================================================================
    # Benchmark ESIMD kernel
    # =========================================================================
    for _ in range(warmup):
        _ = norm.rms_norm(weight, input, eps=eps)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = norm.rms_norm(weight, input, eps=eps)
    torch.xpu.synchronize()
    esimd_time = (time.perf_counter() - start) / iters * 1000
    
    # =========================================================================
    # Benchmark PyTorch native
    # =========================================================================
    for _ in range(warmup):
        rms = torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + eps)
        _ = (input / rms) * weight
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        rms = torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + eps)
        _ = (input / rms) * weight
    torch.xpu.synchronize()
    pytorch_time = (time.perf_counter() - start) / iters * 1000
    
    speedup = pytorch_time / esimd_time if esimd_time > 0 else 0
    return {
        "esimd_time_ms": esimd_time,
        "pytorch_time_ms": pytorch_time,
        "speedup": speedup,
    }


def benchmark_layer_norm(batch_size: int, hidden_size: int, dtype: torch.dtype,
                         warmup: int = 10, iters: int = 100):
    """Benchmark LayerNorm: ESIMD vs PyTorch native."""
    from omni_xpu_kernel import norm
    
    input = torch.randn(batch_size, hidden_size, device="xpu", dtype=dtype)
    weight = torch.ones(hidden_size, device="xpu", dtype=dtype)
    bias = torch.zeros(hidden_size, device="xpu", dtype=dtype)
    eps = 1e-5
    
    # =========================================================================
    # Benchmark ESIMD kernel
    # =========================================================================
    for _ in range(warmup):
        _ = norm.layer_norm(input, weight, bias, eps=eps)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = norm.layer_norm(input, weight, bias, eps=eps)
    torch.xpu.synchronize()
    esimd_time = (time.perf_counter() - start) / iters * 1000
    
    # =========================================================================
    # Benchmark PyTorch native
    # =========================================================================
    for _ in range(warmup):
        _ = torch.nn.functional.layer_norm(input, (hidden_size,), weight, bias, eps=eps)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.nn.functional.layer_norm(input, (hidden_size,), weight, bias, eps=eps)
    torch.xpu.synchronize()
    pytorch_time = (time.perf_counter() - start) / iters * 1000
    
    speedup = pytorch_time / esimd_time if esimd_time > 0 else 0
    return {
        "esimd_time_ms": esimd_time,
        "pytorch_time_ms": pytorch_time,
        "speedup": speedup,
    }


def run_rms_norm_benchmarks():
    """Run RMSNorm benchmarks."""
    print("=" * 90)
    print("RMSNorm Benchmark: ESIMD vs PyTorch Native")
    print("=" * 90)
    print(f"{'Batch':>8} {'Hidden':>8} {'Dtype':>10} │ "
          f"{'ESIMD(ms)':>12} {'PyTorch(ms)':>12} {'Speedup':>10}")
    print(f"{'':-<8} {'':-<8} {'':-<10} ┼ {'':-<12} {'':-<12} {'':-<10}")
    
    for bs in [1, 8, 32, 128]:
        for hs in [2048, 4096, 8192]:
            for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                result = benchmark_rms_norm(bs, hs, dtype)
                dtype_str = str(dtype).split('.')[-1]
                print(f"{bs:>8} {hs:>8} {dtype_str:>10} │ "
                      f"{result['esimd_time_ms']:>12.4f} {result['pytorch_time_ms']:>12.4f} "
                      f"{result['speedup']:>9.2f}x")


def run_layer_norm_benchmarks():
    """Run LayerNorm benchmarks."""
    print("=" * 90)
    print("LayerNorm Benchmark: ESIMD vs PyTorch Native")
    print("=" * 90)
    print(f"{'Batch':>8} {'Hidden':>8} {'Dtype':>10} │ "
          f"{'ESIMD(ms)':>12} {'PyTorch(ms)':>12} {'Speedup':>10}")
    print(f"{'':-<8} {'':-<8} {'':-<10} ┼ {'':-<12} {'':-<12} {'':-<10}")
    
    for bs in [1, 8, 32, 128]:
        for hs in [2048, 4096, 8192]:
            for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                result = benchmark_layer_norm(bs, hs, dtype)
                dtype_str = str(dtype).split('.')[-1]
                print(f"{bs:>8} {hs:>8} {dtype_str:>10} │ "
                      f"{result['esimd_time_ms']:>12.4f} {result['pytorch_time_ms']:>12.4f} "
                      f"{result['speedup']:>9.2f}x")


def run_benchmarks():
    """Run all normalization benchmarks."""
    run_rms_norm_benchmarks()
    print()
    run_layer_norm_benchmarks()


def main():
    if not has_xpu():
        print("XPU not available, skipping benchmarks")
        return
    
    run_benchmarks()


if __name__ == "__main__":
    main()
