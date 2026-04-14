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
    
    Output layout (ComfyUI compatible, Python slice notation):
    - output[0:16] contains low nibbles (indices 0–15)
    - output[16:32] contains high nibbles (indices 16–31)
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


def dequantize_batch(
    inputs: list,
    formats: list,
    dtype: torch.dtype = torch.float16
) -> list:
    """
    Batch dequantize multiple tensors with fewer kernel launches.

    Groups tensors by format, concatenates data, launches one kernel per
    format type, then splits outputs back. For N tensors of the same format,
    this reduces N kernel submissions to 1.

    Args:
        inputs: List of uint8 tensors (quantized data on XPU)
        formats: List of format strings ('q4_0', 'q8_0', 'q4_k', 'q6_k')
        dtype: Output dtype (default: torch.float16)

    Returns:
        List of dequantized tensors in same order as inputs

    Example:
        outputs = gguf.dequantize_batch(
            [tensor1, tensor2, tensor3],
            ['q4_0', 'q4_0', 'q8_0'],
            torch.float16
        )
    """
    return _get_native().dequantize_batch(inputs, formats, dtype)


__all__ = [
    'dequantize_q4_0',
    'dequantize_q8_0',
    'dequantize_q4_k',
    'dequantize_q6_k',
    'dequantize_batch',
]
