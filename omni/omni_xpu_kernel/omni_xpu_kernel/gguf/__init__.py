"""
GGUF Quantization Kernels - Intel XPU ESIMD Optimized

High-performance dequantization kernels for GGUF quantized models.
Supports Q4_0, Q8_0, Q4_K, Q6_K formats (ComfyUI-GGUF compatible).

Example:
    import torch
    from omni_xpu_kernel import gguf
    
    output = gguf.dequantize_q4_0(quantized_data, torch.float16)
    output = gguf.dequantize_q8_0(quantized_data, torch.float16)
    output = gguf.dequantize_q4_k(quantized_data, torch.float16)
    output = gguf.dequantize_q6_k(quantized_data, torch.float16)
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
    Dequantize Q4_0 quantized tensor.
    
    Q4_0 format: 18 bytes per 32 elements
    - 2 bytes: FP16 scale
    - 16 bytes: packed 4-bit values (2 per byte)
    
    Output layout (ComfyUI compatible):
    - output[0:16] = low nibbles
    - output[16:32] = high nibbles
    """
    return _get_native().dequantize_q4_0(input, dtype)


def dequantize_q8_0(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q8_0 quantized tensor.
    
    Q8_0 format: 34 bytes per 32 elements
    - 2 bytes: FP16 scale
    - 32 bytes: int8 values
    """
    return _get_native().dequantize_q8_0(input, dtype)


def dequantize_q4_k(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q4_K quantized tensor.
    
    Q4_K format: 144 bytes per 256 elements
    - 2 bytes: FP16 d (scale)
    - 2 bytes: FP16 dmin (min scale)
    - 12 bytes: packed scales
    - 128 bytes: packed 4-bit values
    """
    return _get_native().dequantize_q4_k(input, dtype)


def dequantize_q6_k(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q6_K quantized tensor.
    
    Q6_K format: 210 bytes per 256 elements
    - 128 bytes: ql (low 4 bits)
    - 64 bytes: qh (high 2 bits)
    - 16 bytes: int8 scales
    - 2 bytes: FP16 d (scale)
    """
    return _get_native().dequantize_q6_k(input, dtype)


__all__ = [
    'dequantize_q4_0',
    'dequantize_q8_0',
    'dequantize_q4_k',
    'dequantize_q6_k',
]
