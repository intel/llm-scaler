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


def _get_native():
    """Get the native GGUF module."""
    from .. import _load_extension

    return _load_extension().gguf


def dequantize_q4_0(
    input: torch.Tensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q4_0 quantized tensor.

    Q4_0 format: 18 bytes per 32 elements
    - 2 bytes: FP16 scale
    - 16 bytes: packed 4-bit values (2 per byte)

    Output layout is interleaved by packed byte: low0, high0, low1,
    high1, ... . Use :func:`dequantize_q4_0_comfyui` for the sequential
    low-half/high-half layout.
    """
    sequential = _get_native().dequantize_q4_0(input, dtype)
    return sequential.reshape(-1, 2, 16).transpose(1, 2).reshape_as(sequential)


def dequantize_q4_0_comfyui(
    input: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize Q4_0 to ComfyUI's low-16 then high-16 layout."""
    return _get_native().dequantize_q4_0(input, dtype)


def dequantize_q8_0(
    input: torch.Tensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize Q8_0 quantized tensor.

    Q8_0 format: 34 bytes per 32 elements
    - 2 bytes: FP16 scale
    - 32 bytes: int8 values
    """
    return _get_native().dequantize_q8_0(input, dtype)


def dequantize_q4_k(
    input: torch.Tensor, dtype: torch.dtype = torch.float16
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
    input: torch.Tensor, dtype: torch.dtype = torch.float16
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
    inputs: list, formats: list, dtype: torch.dtype = torch.float16
) -> list:
    """
    Batch dequantize multiple tensors with platform-selected dispatch.

    BMG groups tensors by format and uses one concatenated launch per format.
    PTL-H launches directly from each original allocation because avoiding the
    packed-input concatenation is faster on that platform.

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
    outputs = _get_native().dequantize_batch(inputs, formats, dtype)
    return [
        output.reshape(-1, 2, 16).transpose(1, 2).reshape_as(output)
        if format_name == "q4_0"
        else output
        for output, format_name in zip(outputs, formats)
    ]


__all__ = [
    "dequantize_q4_0",
    "dequantize_q4_0_comfyui",
    "dequantize_q8_0",
    "dequantize_q4_k",
    "dequantize_q6_k",
    "dequantize_batch",
]
