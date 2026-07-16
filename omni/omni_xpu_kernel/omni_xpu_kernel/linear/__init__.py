"""
Linear Operations Kernels

High-performance FP8 GEMM using Intel oneDNN.

Example:
    import torch
    from omni_xpu_kernel import linear
    
    # FP8 GEMM
    output = linear.onednn_w8a16_fp8(x, weight, scales, bias=None)
"""

import logging
import threading
from typing import Optional

import torch


log = logging.getLogger("omni_xpu_kernel.fp8")

_UNSUPPORTED_MARKER = "OMNI_FP8_PRIMITIVE_UNSUPPORTED:"
_FAILED_KEYS = set()
_FAILED_KEYS_LOCK = threading.Lock()
_PYTHON_NEGATIVE_HITS = 0


def _get_native():
    """Get the native linear module."""
    from .. import _load_extension
    return _load_extension().linear


def _failure_key(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple:
    """Build the Python negative-cache key matching the native descriptor key."""
    return (
        x.device.type,
        x.device.index,
        x.dtype,
        weight.dtype,
        tuple(x.shape),
        tuple(weight.shape),
        bias is not None,
    )


def _cached_failure_error(key: tuple) -> RuntimeError:
    return RuntimeError(f"{_UNSUPPORTED_MARKER}cached: key={key}")


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
    global _PYTHON_NEGATIVE_HITS

    key = _failure_key(x, weight, bias)
    with _FAILED_KEYS_LOCK:
        if key in _FAILED_KEYS:
            _PYTHON_NEGATIVE_HITS += 1
            raise _cached_failure_error(key)

    try:
        return _get_native().onednn_w8a16_fp8(x, weight, scales, bias)
    except RuntimeError as exc:
        message = str(exc)
        if _UNSUPPORTED_MARKER not in message:
            raise

        with _FAILED_KEYS_LOCK:
            first_failure = key not in _FAILED_KEYS
            _FAILED_KEYS.add(key)

        if first_failure:
            log.warning(
                "oneDNN W8A16 primitive unavailable for device=%s input=%s "
                "weight=%s input_shape=%s weight_shape=%s bias=%s; "
                "caching unsupported shape",
                x.device,
                x.dtype,
                weight.dtype,
                tuple(x.shape),
                tuple(weight.shape),
                bias is not None,
            )
        raise


def try_onednn_w8a16_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return ``None`` when oneDNN cannot create this FP8 primitive.

    The first unsupported descriptor logs one warning and populates the native
    and Python negative caches. Later calls bypass native primitive creation.
    Validation errors and other runtime failures are not suppressed.
    """
    try:
        return onednn_w8a16_fp8(x, weight, scales, bias)
    except RuntimeError as exc:
        if _UNSUPPORTED_MARKER in str(exc):
            return None
        raise


def fp8_cache_clear() -> None:
    """Clear cached successful and failed oneDNN FP8 primitive state."""
    global _PYTHON_NEGATIVE_HITS

    _get_native().fp8_cache_clear()
    with _FAILED_KEYS_LOCK:
        _FAILED_KEYS.clear()
        _PYTHON_NEGATIVE_HITS = 0


def fp8_cache_stats() -> dict[str, int]:
    """Return FP8 primitive cache counters and size."""
    hits, misses, size = _get_native().fp8_cache_stats()
    return {"hits": hits, "misses": misses, "size": size}


def fp8_failure_cache_stats() -> dict[str, int]:
    """Return failed primitive insertions, negative hits, and cache size."""
    failures, native_hits, size = _get_native().fp8_failure_cache_stats()
    with _FAILED_KEYS_LOCK:
        python_hits = _PYTHON_NEGATIVE_HITS
    return {
        "failures": failures,
        "negative_hits": native_hits + python_hits,
        "size": size,
    }


__all__ = [
    "onednn_w8a16_fp8",
    "try_onednn_w8a16_fp8",
    "fp8_cache_clear",
    "fp8_cache_stats",
    "fp8_failure_cache_stats",
]
