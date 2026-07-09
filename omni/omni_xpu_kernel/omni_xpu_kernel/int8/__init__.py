"""
INT8 Quantization and Linear Operations for Intel XPU.

Provides high-performance INT8 inference kernels matching comfy-kitchen's INT8 API:
- quantize_int8_tensorwise: Tensor-wise INT8 quantization (single scale)
- quantize_int8_rowwise: Per-row INT8 quantization (activation path)
- dequantize_int8_simple: INT8 dequantization with scale
- dequantize_int8_simple_dtype: INT8 dequantization with output dtype
- int8_linear: Dynamic INT8 linear layer (quant activation + INT8 GEMM + rescale)
- mm_int8: Raw INT8 matrix multiplication (s8×s8→s32)
- quantize_int8_convrot_weight: Offline ConvRot weight rotation + quantization
- dequantize_int8_convrot_weight: ConvRot dequantization with inverse rotation

Performance path:
    Native C++ (oneDNN s8 matmul + ESIMD fusion) > Python reference fallback

Example:
    from omni_xpu_kernel import int8

    # Quantize weight offline
    w_int8, w_scale = int8.quantize_int8_tensorwise(weight)

    # INT8 linear inference
    output = int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

    # With ConvRot
    output = int8.int8_linear(x, w_int8, w_scale, convrot=True, convrot_groupsize=256)
"""

import torch
from typing import Optional, Tuple

from ._reference import (
    quantize_int8_tensorwise as _ref_quantize_int8_tensorwise,
    quantize_int8_rowwise as _ref_quantize_int8_rowwise,
    dequantize_int8_simple as _ref_dequantize_int8_simple,
    dequantize_int8_simple_dtype as _ref_dequantize_int8_simple_dtype,
    mm_int8 as _ref_mm_int8,
    int8_linear as _ref_int8_linear,
    quantize_int8_convrot_weight as _ref_quantize_int8_convrot_weight,
    dequantize_int8_convrot_weight as _ref_dequantize_int8_convrot_weight,
)


def _get_native():
    """Get the native INT8 module (returns None if unavailable)."""
    try:
        from .. import _load_extension
        mod = _load_extension()
        return getattr(mod, 'int8', None)
    except (ImportError, AttributeError):
        return None


# =============================================================================
# Public API — dispatch to native or fallback to reference
# =============================================================================


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale.

    Args:
        x: Input tensor of any shape.
        scale: Optional pre-computed scale. If None, computes absmax/127.
        stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.

    Returns:
        Tuple of (quantized_int8, scale):
            - quantized_int8: INT8 tensor with same shape
            - scale: Scalar float32 tensor
    """
    native = _get_native()
    if native is not None and hasattr(native, 'quantize_int8_tensorwise'):
        return native.quantize_int8_tensorwise(x, scale, stochastic_rounding)
    return _ref_quantize_int8_tensorwise(x, scale, stochastic_rounding)


def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations).

    Args:
        x: Input tensor [..., K] where quantization is per-row.
        stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.

    Returns:
        Tuple of (quantized_int8, scales):
            - quantized_int8: INT8 tensor with same shape
            - scales: Float32 tensor [..., 1] with per-row scales
    """
    native = _get_native()
    if native is not None and hasattr(native, 'quantize_int8_rowwise'):
        return native.quantize_int8_rowwise(x, stochastic_rounding)
    return _ref_quantize_int8_rowwise(x, stochastic_rounding)


def dequantize_int8_simple(
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT8 tensor with scale.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).

    Returns:
        Dequantized float32 tensor.
    """
    native = _get_native()
    if native is not None and hasattr(native, 'dequantize_int8_simple'):
        return native.dequantize_int8_simple(q, scale)
    return _ref_dequantize_int8_simple(q, scale)


def dequantize_int8_simple_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize INT8 tensor with scale into specified output dtype.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).
        out_dtype: Output dtype (float32, float16, or bfloat16).

    Returns:
        Dequantized tensor in specified dtype.
    """
    native = _get_native()
    if native is not None and hasattr(native, 'dequantize_int8_simple_dtype'):
        dtype_code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[out_dtype]
        return native.dequantize_int8_simple_dtype(q, scale, dtype_code)
    return _ref_dequantize_int8_simple_dtype(q, scale, out_dtype)


def mm_int8(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """INT8 matrix multiplication: C[M,N] = A[M,K] @ B[K,N].

    Uses oneDNN s8×s8→s32 GEMM on XPU for maximum throughput.

    Args:
        a: INT8 tensor [M, K].
        b: INT8 tensor [K, N].

    Returns:
        INT32 tensor [M, N] with accumulated dot products.
    """
    native = _get_native()
    if native is not None and hasattr(native, 'mm_int8'):
        return native.mm_int8(a, b)
    return _ref_mm_int8(a, b)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 linear layer with dynamic activation quantization.

    Quantizes activation per-row, performs INT8 GEMM, rescales output.
    Optionally applies ConvRot (online Hadamard rotation) for improved accuracy.

    Args:
        x: Input tensor [..., K] (fp16/bf16).
        weight: INT8 weight tensor [N, K].
        weight_scale: Weight scale (scalar or per-channel [N] or [N,1]).
        bias: Optional bias tensor [N].
        out_dtype: Output dtype (defaults to x.dtype).
        convrot: If True, apply online activation rotation before quantization.
        convrot_groupsize: Group size for Hadamard rotation (must be power of 4).

    Returns:
        Result tensor [..., N] in out_dtype.
    """
    if out_dtype is None:
        out_dtype = x.dtype
    native = _get_native()
    if native is not None and hasattr(native, 'int8_linear'):
        # Handle ConvRot rotation in Python before native GEMM
        # (native ESIMD rotation kernel will replace this later)
        if convrot:
            from ._reference import _build_hadamard, _rotate_activation
            if x.shape[-1] % convrot_groupsize != 0:
                raise ValueError(
                    f"ConvRot group size {convrot_groupsize} does not divide "
                    f"input features {x.shape[-1]}"
                )
            h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
            x = _rotate_activation(x, h, convrot_groupsize)
        dtype_code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}.get(out_dtype, 2)
        return native.int8_linear(x, weight, weight_scale, bias, dtype_code,
                                  False, convrot_groupsize)
    return _ref_int8_linear(x, weight, weight_scale, bias, out_dtype,
                            convrot, convrot_groupsize)


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int = 256,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation followed by per-row INT8 quantization.

    Args:
        weight: Weight tensor [N, K].
        group_size: Hadamard rotation group size (must be power of 4).
        stochastic_rounding: Seed for stochastic rounding.

    Returns:
        Tuple of (rotated_quantized_weight_int8, per_row_scales).
    """
    native = _get_native()
    if native is not None and hasattr(native, 'quantize_int8_convrot_weight'):
        return native.quantize_int8_convrot_weight(weight, group_size, stochastic_rounding)
    return _ref_quantize_int8_convrot_weight(weight, group_size, stochastic_rounding)


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 256,
) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights and rotate back to original basis.

    Args:
        q: Quantized INT8 weight tensor [N, K].
        scale: Per-row scales [N, 1].
        group_size: Hadamard rotation group size.

    Returns:
        Dequantized weight tensor in float32.
    """
    native = _get_native()
    if native is not None and hasattr(native, 'dequantize_int8_convrot_weight'):
        return native.dequantize_int8_convrot_weight(q, scale, group_size)
    return _ref_dequantize_int8_convrot_weight(q, scale, group_size)


def int8_cache_clear() -> None:
    """Clear cached oneDNN INT8 primitive state."""
    native = _get_native()
    if native is not None and hasattr(native, 'int8_cache_clear'):
        native.int8_cache_clear()


def int8_cache_stats() -> dict:
    """Return INT8 primitive cache counters and size."""
    native = _get_native()
    if native is not None and hasattr(native, 'int8_cache_stats'):
        hits, misses, size = native.int8_cache_stats()
        return {"hits": hits, "misses": misses, "size": size}
    return {"hits": 0, "misses": 0, "size": 0}


__all__ = [
    "quantize_int8_tensorwise",
    "quantize_int8_rowwise",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "mm_int8",
    "int8_linear",
    "quantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight",
    "int8_cache_clear",
    "int8_cache_stats",
]
