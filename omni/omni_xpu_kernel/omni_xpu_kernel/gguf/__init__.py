"""
GGUF Quantization Kernels

High-performance dequantization kernels for GGUF quantized models.
Supports Q4_0 format with both interleaved and sequential output layouts.

Example:
    import torch
    from omni_xpu_kernel import gguf
    
    # Standard GGUF dequantization
    output = gguf.dequantize_q4_0(quantized_data, torch.float16)
    
    # ComfyUI-compatible dequantization  
    output = gguf.dequantize_q4_0_comfyui(quantized_data, torch.float16)
"""

import torch
from typing import Optional


def _get_native():
    """Get the native GGUF module."""
    from .. import _load_extension
    return _load_extension().gguf


def dequantize_q4_0(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q4_0 quantized tensor using ESIMD optimization.
    
    This uses the standard GGUF interleaved output format where:
    - output[2i] = low nibble
    - output[2i+1] = high nibble
    
    Args:
        input: Quantized tensor (uint8, contiguous)
        dtype: Output dtype (float16, bfloat16, or float32)
    
    Returns:
        Dequantized tensor on the same device
    
    Note:
        Q4_0 format: 18 bytes per 32 elements
        - 2 bytes: FP16 scale
        - 16 bytes: 32 x 4-bit values
    """
    return _get_native().dequantize_q4_0(input, dtype)


def dequantize_q4_0_comfyui(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q4_0 quantized tensor with ComfyUI-compatible output layout.
    
    Uses sequential output format where:
    - output[0:15] = low nibbles
    - output[16:31] = high nibbles
    
    This matches the memory layout expected by ComfyUI-GGUF.
    
    Args:
        input: Quantized tensor (uint8, contiguous)
        dtype: Output dtype (float16, bfloat16, or float32)
    
    Returns:
        Dequantized tensor on the same device
    """
    return _get_native().dequantize_q4_0_comfyui(input, dtype)


def dequantize_q4_0_layout(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    sequential_layout: bool = False
) -> torch.Tensor:
    """
    Dequantize Q4_0 with configurable output layout.
    
    Args:
        input: Quantized tensor (uint8, contiguous)
        dtype: Output dtype (float16, bfloat16, or float32)
        sequential_layout: If True, use ComfyUI format; if False, use standard GGUF
    
    Returns:
        Dequantized tensor on the same device
    """
    return _get_native().dequantize_q4_0_layout(input, dtype, sequential_layout)


def benchmark(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    warmup_iters: int = 10,
    bench_iters: int = 100
) -> float:
    """
    Benchmark Q4_0 dequantization performance.
    
    Args:
        input: Quantized tensor for benchmarking
        dtype: Output dtype
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
    
    Returns:
        Average time per iteration in milliseconds
    """
    return _get_native().benchmark(input, dtype, warmup_iters, bench_iters)


__all__ = [
    "dequantize_q4_0",
    "dequantize_q4_0_comfyui",
    "dequantize_q4_0_layout",
    "benchmark",
]
