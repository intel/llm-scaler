"""
Performance benchmarks for GGUF dequantization kernels

Compares ESIMD kernel against PyTorch native implementation.

Usage:
    python -m benchmarks.bench_gguf
"""

import time
import torch
import numpy as np


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def create_q4_0_data(n_blocks: int, device: str = "xpu"):
    """Create Q4_0 quantized data with valid FP16 scales."""
    data = []
    for _ in range(n_blocks):
        scale = np.random.uniform(-10.0, 10.0)
        scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
        packed = np.random.randint(0, 256, 16, dtype=np.uint8)
        data.extend(scale_bytes.tolist())
        data.extend(packed.tolist())
    return torch.tensor(data, dtype=torch.uint8, device=device)


def torch_dequantize_q4_0(data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    PyTorch native implementation of Q4_0 dequantization.
    Uses vectorized operations for better performance.
    """
    n_blocks = data.numel() // 18
    
    # Reshape data to [n_blocks, 18]
    data_reshaped = data.view(n_blocks, 18)
    
    # Extract scales (first 2 bytes of each block)
    scales = data_reshaped[:, :2].contiguous().view(torch.float16).view(n_blocks, 1)
    
    # Extract packed data (remaining 16 bytes)
    packed = data_reshaped[:, 2:].to(torch.int32)  # [n_blocks, 16]
    
    # Unpack low and high nibbles
    low = (packed & 0x0F) - 8   # [n_blocks, 16]
    high = (packed >> 4) - 8    # [n_blocks, 16]
    
    # Interleave low and high values: [low0, high0, low1, high1, ...]
    output = torch.stack([low, high], dim=2).view(n_blocks, 32)  # [n_blocks, 32]
    
    # Apply scale and convert to target dtype
    output = (output.to(scales.dtype) * scales).to(dtype)
    
    return output.view(-1)


def benchmark_gguf_dequant(n_blocks: int, dtype: torch.dtype, warmup: int = 10, iters: int = 100):
    """Benchmark GGUF Q4_0 dequantization: ESIMD vs PyTorch native."""
    from omni_xpu_kernel import gguf
    
    data = create_q4_0_data(n_blocks, "xpu")
    data_bytes = n_blocks * 18
    output_elements = n_blocks * 32
    
    # =========================================================================
    # Benchmark ESIMD kernel
    # =========================================================================
    for _ in range(warmup):
        _ = gguf.dequantize_q4_0(data, dtype)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = gguf.dequantize_q4_0(data, dtype)
    torch.xpu.synchronize()
    esimd_elapsed = time.perf_counter() - start
    esimd_time_ms = esimd_elapsed / iters * 1000
    
    # =========================================================================
    # Benchmark PyTorch native
    # =========================================================================
    for _ in range(warmup):
        _ = torch_dequantize_q4_0(data, dtype)
    torch.xpu.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch_dequantize_q4_0(data, dtype)
    torch.xpu.synchronize()
    pytorch_elapsed = time.perf_counter() - start
    pytorch_time_ms = pytorch_elapsed / iters * 1000
    
    # =========================================================================
    # Calculate metrics
    # =========================================================================
    speedup = pytorch_time_ms / esimd_time_ms if esimd_time_ms > 0 else 0
    esimd_bandwidth = (data_bytes + output_elements * dtype.itemsize) / (esimd_elapsed / iters) / 1e9
    pytorch_bandwidth = (data_bytes + output_elements * dtype.itemsize) / (pytorch_elapsed / iters) / 1e9
    
    return {
        "esimd_time_ms": esimd_time_ms,
        "pytorch_time_ms": pytorch_time_ms,
        "speedup": speedup,
        "esimd_bandwidth_gb": esimd_bandwidth,
        "pytorch_bandwidth_gb": pytorch_bandwidth,
    }


def verify_correctness(n_blocks: int = 1000, dtype: torch.dtype = torch.float16):
    """Verify that ESIMD and PyTorch implementations produce same results."""
    from omni_xpu_kernel import gguf
    
    data = create_q4_0_data(n_blocks, "xpu")
    
    esimd_output = gguf.dequantize_q4_0(data, dtype)
    pytorch_output = torch_dequantize_q4_0(data, dtype)
    
    try:
        torch.testing.assert_close(esimd_output, pytorch_output, rtol=1e-3, atol=1e-3)
        print("✓ Correctness verified: ESIMD and PyTorch outputs match")
        return True
    except AssertionError as e:
        print(f"✗ Correctness check failed: {e}")
        return False


def run_benchmarks():
    """Run all GGUF benchmarks."""
    print("=" * 100)
    print("GGUF Q4_0 Dequantization Benchmark: ESIMD vs PyTorch Native")
    print("=" * 100)
    
    # Verify correctness first
    print("\nVerifying correctness...")
    if not verify_correctness():
        print("Aborting benchmark due to correctness failure")
        return
    
    print()
    print(f"{'Blocks':>12} {'Elements':>12} {'Dtype':>10} │ "
          f"{'ESIMD(ms)':>12} {'PyTorch(ms)':>12} {'Speedup':>10} │ "
          f"{'ESIMD BW':>12} {'PyTorch BW':>12}")
    print(f"{'':-<12} {'':-<12} {'':-<10} ┼ "
          f"{'':-<12} {'':-<12} {'':-<10} ┼ "
          f"{'':-<12} {'':-<12}")
    
    for n_blocks in [1000, 10000, 100000, 1000000]:
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            result = benchmark_gguf_dequant(n_blocks, dtype)
            dtype_str = str(dtype).split('.')[-1]
            elements = n_blocks * 32
            print(f"{n_blocks:>12} {elements:>12} {dtype_str:>10} │ "
                  f"{result['esimd_time_ms']:>12.4f} {result['pytorch_time_ms']:>12.4f} "
                  f"{result['speedup']:>9.2f}x │ "
                  f"{result['esimd_bandwidth_gb']:>10.2f}GB/s {result['pytorch_bandwidth_gb']:>10.2f}GB/s")


def main():
    if not has_xpu():
        print("XPU not available, skipping benchmarks")
        return
    
    run_benchmarks()


if __name__ == "__main__":
    main()
