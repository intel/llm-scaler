"""
Linear Operations Kernels

High-performance FP8 GEMM using Intel oneDNN.

Example:
    import torch
    from omni_xpu_kernel import linear
    
    # FP8 GEMM
    output = linear.onednn_w8a16_fp8(x, weight, scales, bias=None)
"""

import torch
from typing import Optional


def _get_native():
    """Get the native linear module."""
    from .. import _load_extension
    return _load_extension().linear


def onednn_w8a16_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    FP8 GEMM (Weight-only 8-bit, Activation 16-bit) using oneDNN optimization.
    
    Args:
        x: Input tensor of shape [..., K] (fp16/bf16)
        weight: Weight tensor of shape [N, K] (fp8_e4m3fn)
        scales: Scale tensor for weight quantization
        bias: Optional bias tensor of shape [N]
    
    Returns:
        Output tensor of shape [..., N]
    """
    return _get_native().onednn_w8a16_fp8(x, weight, scales, bias)


def fp8_cache_clear() -> None:
    """Clear cached oneDNN FP8 primitive state."""
    _get_native().fp8_cache_clear()


def fp8_cache_stats() -> dict[str, int]:
    """Return FP8 primitive cache counters and size."""
    return _get_native().fp8_cache_stats()


__all__ = [
    "onednn_w8a16_fp8",
    "fp8_cache_clear",
    "fp8_cache_stats",
]
