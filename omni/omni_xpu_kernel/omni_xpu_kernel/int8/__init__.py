"""
INT8 Quantization and Linear Operations for Intel XPU.

Provides high-performance INT8 inference kernels matching comfy-kitchen's INT8 API:
- quantize_int8_tensorwise: Tensor-wise INT8 quantization (single scale)
- quantize_int8_rowwise: Per-row INT8 quantization (activation path)
- dequantize_int8_simple: INT8 dequantization with scale
- dequantize_int8_simple_dtype: INT8 dequantization with output dtype
- int8_linear: Dynamic INT8 linear layer (quant activation + INT8 GEMM + rescale)
- int8_linear_prequantized: Scaled INT8 linear for a prequantized activation
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

    # Reuse one activation quantization across one or more linear calls
    x_int8, x_scale = int8.quantize_int8_rowwise(x)
    output = int8.int8_linear_prequantized(
        x_int8, x_scale, w_int8, w_scale, bias=bias,
        out_dtype=torch.bfloat16,
    )

    # With ConvRot
    output = int8.int8_linear(x, w_int8, w_scale, convrot=True, convrot_groupsize=256)
"""

import os
import threading
from typing import Optional, Tuple

import torch

from ._reference import (
    quantize_int8_tensorwise as _ref_quantize_int8_tensorwise,
    quantize_int8_rowwise as _ref_quantize_int8_rowwise,
    fused_silu_mul_quantize_rowwise as _ref_fused_silu_mul_quantize_rowwise,
    fused_silu_mul as _ref_fused_silu_mul,
    dequantize_int8_simple as _ref_dequantize_int8_simple,
    dequantize_int8_simple_dtype as _ref_dequantize_int8_simple_dtype,
    mm_int8 as _ref_mm_int8,
    int8_linear as _ref_int8_linear,
    int8_linear_prequantized as _ref_int8_linear_prequantized,
    int8_linear_shared_input as _ref_int8_linear_shared_input,
    quantize_int8_convrot_weight as _ref_quantize_int8_convrot_weight,
    dequantize_int8_convrot_weight as _ref_dequantize_int8_convrot_weight,
)


def _get_native():
    """Get the native INT8 module (returns None if unavailable)."""
    try:
        from .. import _load_extension

        mod = _load_extension()
        return getattr(mod, "int8", None)
    except (ImportError, AttributeError):
        return None


def _apply_input_act(
    x: torch.Tensor, input_act: Optional[str]
) -> torch.Tensor:
    if input_act in (None, "none"):
        return x
    if input_act == "gelu_tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    if input_act == "swiglu":
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate).mul_(up)
    raise ValueError(
        f"unsupported input_act: {input_act!r} "
        "(expected one of ['gelu_tanh', 'none', 'swiglu'])"
    )


_h3_swiglu_trace_logged = False


def _is_supported_h3_swiglu_target() -> bool:
    """Require the BMG package/core pair used for H3 tuning."""
    try:
        from .. import __xpu_target__, core_aot_target

        return __xpu_target__ == "bmg" and core_aot_target() == "bmg"
    except (ImportError, RuntimeError):
        return False


def _can_fuse_h3_swiglu(
    x: torch.Tensor,
    native,
    weight: torch.Tensor,
    convrot: bool,
    convrot_groupsize: int,
    input_act: Optional[str],
) -> bool:
    """Match the structural BF16 H3 SwiGLU-to-G256 boundary."""
    return bool(
        input_act == "swiglu"
        and convrot
        and convrot_groupsize == 256
        and _is_supported_h3_swiglu_target()
        and native is not None
        and hasattr(native, "fused_silu_mul_exact_bf16")
        and isinstance(x, torch.Tensor)
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] > 0
        and x.shape[1] % 2 == 0
        and (x.shape[1] // 2) % convrot_groupsize == 0
        and x.is_contiguous()
        and not x.requires_grad
        and isinstance(weight, torch.Tensor)
        and weight.device == x.device
        and weight.dtype == torch.int8
        and weight.ndim == 2
        and weight.shape[1] == x.shape[1] // 2
    )


def _apply_h3_swiglu_exact(x: torch.Tensor, native) -> torch.Tensor:
    """Preserve PyTorch's BF16 SiLU and product materialization boundaries."""
    global _h3_swiglu_trace_logged

    gate, up = x.chunk(2, dim=-1)
    output = native.fused_silu_mul_exact_bf16(gate, up)
    if (
        not _h3_swiglu_trace_logged
        and os.environ.get("OMNIXPU_H3_SWIGLU_TRACE") == "1"
    ):
        print(
            "[OmniXPU] H3 exact SwiGLU route: "
            f"input={tuple(x.shape)} input_stride={tuple(x.stride())} "
            f"gate_stride={tuple(gate.stride())}",
            flush=True,
        )
        _h3_swiglu_trace_logged = True
    return output


def _can_fuse_gelu_tanh_quantize(
    x: torch.Tensor,
    native,
) -> bool:
    """Use only the BMG contracts where fused GELU beats materialization."""
    if (
        x.dtype not in (torch.float16, torch.bfloat16)
        or x.ndim < 2
        or x.shape[-1] <= 0
        or x.shape[-1] > 16384
        or not hasattr(native, "fused_gelu_tanh_quantize_rowwise")
        or not hasattr(native, "int8_linear_prequantized")
    ):
        return False
    try:
        from .. import __xpu_target__, core_aot_target

        if __xpu_target__ != "bmg" or core_aot_target() != "bmg":
            return False
    except (ImportError, RuntimeError):
        return False
    rows = x.numel() // x.shape[-1]
    return rows < 128


def _can_quantize_int8_convrot_g16_bmg(
    x: torch.Tensor,
    native,
    convrot: bool,
    convrot_groupsize: int,
) -> bool:
    """Return whether the captured BMG Boogu ConvRot route is exact."""
    if (
        native is None
        or not hasattr(native, "quantize_int8_convrot_g16_bmg")
        or not hasattr(native, "int8_linear_prequantized")
        or not convrot
        or convrot_groupsize != 16
        or x.device.type != "xpu"
        or x.dtype != torch.float16
        or x.ndim < 1
        or x.shape[-1] != 3360
        or not x.is_contiguous()
        or x.requires_grad
    ):
        return False
    rows = x.numel() // x.shape[-1]
    return rows in (109, 110, 4096, 4205, 4206)


def _can_pair_int8_convrot_g16_bmg(
    x: torch.Tensor,
    native,
    weight1: torch.Tensor,
    weight_scale1: torch.Tensor,
    weight2: torch.Tensor,
    weight_scale2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> bool:
    """Return whether the exact Boogu shared-up pair route applies."""
    if (
        not _can_quantize_int8_convrot_g16_bmg(
            x, native, convrot, convrot_groupsize
        )
        or not hasattr(native, "int8_linear_pair_prequantized")
        or out_dtype != torch.float16
        or bias1 is not None
        or bias2 is not None
    ):
        return False
    for weight, scale in (
        (weight1, weight_scale1),
        (weight2, weight_scale2),
    ):
        if (
            not isinstance(weight, torch.Tensor)
            or weight.device != x.device
            or weight.dtype != torch.int8
            or weight.shape != (13568, 3360)
            or not weight.is_contiguous()
            or not isinstance(scale, torch.Tensor)
            or scale.device != x.device
            or scale.numel() != 13568
        ):
            return False
    return True


_BMG_QKV_OUTPUT_FEATURES = (840, 3360)
_bmg_qkv_activation_cache = threading.local()


def _clear_bmg_qkv_activation_cache() -> None:
    """Drop the current thread's exact Boogu QKV activation cache."""
    _bmg_qkv_activation_cache.key = None
    _bmg_qkv_activation_cache.input = None
    _bmg_qkv_activation_cache.quantized = None
    _bmg_qkv_activation_cache.scale = None
    _bmg_qkv_activation_cache.remaining_hits = 0


def _can_cache_int8_convrot_g16_bmg(
    x: torch.Tensor,
    native,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> bool:
    """Return whether the exact Boogu image Q/K/V cache contract applies."""
    if (
        not _can_quantize_int8_convrot_g16_bmg(
            x, native, convrot, convrot_groupsize
        )
        or out_dtype != torch.float16
        or bias is not None
        or not isinstance(weight, torch.Tensor)
        or weight.device != x.device
        or weight.dtype != torch.int8
        or weight.ndim != 2
        or weight.shape[1] != 3360
        or weight.shape[0] not in _BMG_QKV_OUTPUT_FEATURES
        or not weight.is_contiguous()
        or not isinstance(weight_scale, torch.Tensor)
        or weight_scale.device != x.device
        or weight_scale.numel() != weight.shape[0]
    ):
        return False
    return True


def _bmg_qkv_input_key(x: torch.Tensor) -> tuple:
    """Identify one activation on one XPU stream without synchronizing."""
    stream = torch.xpu.current_stream(x.device)
    is_inference = torch.is_inference(x)
    version = None if is_inference else x._version
    return (
        x.device.index,
        stream.stream_id,
        x.data_ptr(),
        x.storage_offset(),
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        is_inference,
        version,
    )


def _quantize_int8_convrot_g16_bmg_qkv(
    x: torch.Tensor,
    native,
    output_features: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reuse the Q activation quantization for the following K and V calls.

    The exact Boogu attention sequence is one 3360-wide Q projection followed
    immediately by two 840-wide K/V projections from the same activation.
    Keeping the input alive prevents allocator pointer reuse; normal tensors
    also carry their mutation version in the key. Inference tensors do not
    expose a version counter, so their cache is deliberately limited to the
    two consecutive K/V hits and is cleared by every non-matching linear call.
    """
    key = _bmg_qkv_input_key(x)
    if (
        output_features == 840
        and getattr(_bmg_qkv_activation_cache, "key", None) == key
        and getattr(_bmg_qkv_activation_cache, "remaining_hits", 0) > 0
        and getattr(_bmg_qkv_activation_cache, "quantized", None) is not None
        and getattr(_bmg_qkv_activation_cache, "scale", None) is not None
    ):
        x_int8 = _bmg_qkv_activation_cache.quantized
        x_scale = _bmg_qkv_activation_cache.scale
        _bmg_qkv_activation_cache.remaining_hits -= 1
        if _bmg_qkv_activation_cache.remaining_hits == 0:
            _clear_bmg_qkv_activation_cache()
        return x_int8, x_scale

    _clear_bmg_qkv_activation_cache()
    x_int8, x_scale = native.quantize_int8_convrot_g16_bmg(x)
    if output_features == 3360:
        _bmg_qkv_activation_cache.key = key
        _bmg_qkv_activation_cache.input = x
        _bmg_qkv_activation_cache.quantized = x_int8
        _bmg_qkv_activation_cache.scale = x_scale
        _bmg_qkv_activation_cache.remaining_hits = 2
    return x_int8, x_scale


_KREA2_INPUT_SHAPE = (1, 4192, 6144)
_KREA2_OUTPUT_FEATURES = (1536, 6144, 16384)
_KREA2_MAXIMUM_CACHE_HITS = 3
_krea2_activation_cache = threading.local()


def _clear_krea2_activation_cache() -> None:
    """Drop the current thread's target-aligned Krea2 projection cache."""
    _krea2_activation_cache.key = None
    _krea2_activation_cache.input = None
    _krea2_activation_cache.quantized = None
    _krea2_activation_cache.scale = None
    _krea2_activation_cache.remaining_hits = 0


def _is_supported_krea2_cache_target() -> bool:
    """Require aligned package/core metadata for a validated XPU target."""
    try:
        from .. import __xpu_target__, core_aot_target

        return (
            __xpu_target__ in {"bmg", "ptl-h"}
            and core_aot_target() == __xpu_target__
        )
    except (ImportError, RuntimeError):
        return False


def _can_cache_krea2_int8_convrot(
    x: torch.Tensor,
    native,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    convrot: bool,
    convrot_groupsize: int,
) -> bool:
    """Return whether the trace-proven Krea2 projection contract applies."""
    if (
        not _is_supported_krea2_cache_target()
        or native is None
        or not hasattr(native, "rotate_convrot")
        or not hasattr(native, "quantize_int8_rowwise_fused")
        or not hasattr(native, "int8_linear_prequantized")
        or not convrot
        or convrot_groupsize != 256
        or not isinstance(x, torch.Tensor)
        or x.device.type != "xpu"
        or x.dtype != torch.bfloat16
        or tuple(x.shape) != _KREA2_INPUT_SHAPE
        or not x.is_contiguous()
        or x.requires_grad
        or bias is not None
        or out_dtype != torch.bfloat16
        or not isinstance(weight, torch.Tensor)
        or weight.device != x.device
        or weight.dtype != torch.int8
        or weight.ndim != 2
        or weight.shape[1] != _KREA2_INPUT_SHAPE[-1]
        or weight.shape[0] not in _KREA2_OUTPUT_FEATURES
        or not weight.is_contiguous()
        or not isinstance(weight_scale, torch.Tensor)
        or weight_scale.device != x.device
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != weight.shape[0]
        or not weight_scale.is_contiguous()
    ):
        return False
    return True


def _quantize_krea2_int8_convrot(
    x: torch.Tensor,
    native,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reuse the exact public G256 rotation and rowwise quantization.

    The canonical Krea2 workflow projects the same BF16 activation up to four
    immediately consecutive times.  The cache retains that activation, keys
    it by XPU stream and tensor identity, and permits at most three hits.
    Every non-matching public linear call clears the entry.
    """
    key = _bmg_qkv_input_key(x)
    if (
        getattr(_krea2_activation_cache, "key", None) == key
        and getattr(_krea2_activation_cache, "remaining_hits", 0) > 0
        and getattr(_krea2_activation_cache, "quantized", None) is not None
        and getattr(_krea2_activation_cache, "scale", None) is not None
    ):
        x_int8 = _krea2_activation_cache.quantized
        x_scale = _krea2_activation_cache.scale
        _krea2_activation_cache.remaining_hits -= 1
        if _krea2_activation_cache.remaining_hits == 0:
            _clear_krea2_activation_cache()
        return x_int8, x_scale

    _clear_krea2_activation_cache()
    rotated = native.rotate_convrot(x, 256)
    x_int8, x_scale = native.quantize_int8_rowwise_fused(rotated)
    _krea2_activation_cache.key = key
    _krea2_activation_cache.input = x
    _krea2_activation_cache.quantized = x_int8
    _krea2_activation_cache.scale = x_scale
    _krea2_activation_cache.remaining_hits = _KREA2_MAXIMUM_CACHE_HITS
    return x_int8, x_scale


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
    if native is not None and hasattr(native, "quantize_int8_tensorwise"):
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
    if native is not None:
        # The fused hot path covers deterministic FP32/FP16/BF16 rowwise input.
        # Preserve the generic native API for explicit stochastic rounding.
        if (
            stochastic_rounding <= 0
            and x.ndim >= 1
            and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and hasattr(native, "quantize_int8_rowwise_fused")
        ):
            return native.quantize_int8_rowwise_fused(x)
        if hasattr(native, "quantize_int8_rowwise"):
            return native.quantize_int8_rowwise(x, stochastic_rounding)
    return _ref_quantize_int8_rowwise(x, stochastic_rounding)


def fused_silu_mul_quantize_rowwise(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``SiLU(x1) * x2`` with deterministic rowwise INT8 quantization.

    The floating SwiGLU result is not materialized by the native path. The
    returned quantized tensor and row scales can be passed directly to
    :func:`int8_linear_prequantized`.
    """
    native = _get_native()
    if native is not None and hasattr(native, "fused_silu_mul_quantize_rowwise"):
        return native.fused_silu_mul_quantize_rowwise(x1, x2)
    return _ref_fused_silu_mul_quantize_rowwise(x1, x2)


def fused_swiglu_quantize_rowwise(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse concatenated ``[gate | up]`` SwiGLU with rowwise INT8 quantization."""
    if input.shape[-1] <= 0 or input.shape[-1] % 2:
        raise ValueError("SwiGLU input last dimension must be positive and even")
    native = _get_native()
    if native is not None and hasattr(native, "fused_swiglu_quantize_rowwise"):
        return native.fused_swiglu_quantize_rowwise(input)
    gate, up = input.chunk(2, dim=-1)
    return _ref_fused_silu_mul_quantize_rowwise(gate, up)


def fused_gelu_tanh_quantize_rowwise(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse tanh-approximate GELU with rowwise INT8 quantization."""
    if input.shape[-1] <= 0:
        raise ValueError("GELU input last dimension must be positive")
    native = _get_native()
    if native is not None and hasattr(native, "fused_gelu_tanh_quantize_rowwise"):
        return native.fused_gelu_tanh_quantize_rowwise(input)
    activated = torch.nn.functional.gelu(input, approximate="tanh")
    return quantize_int8_rowwise(activated)


def fused_silu_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> torch.Tensor:
    """Fuse ``SiLU(x1) * x2`` while retaining one floating output tensor.

    This boundary is useful before a required floating transform such as
    ConvRot: it removes the separate SiLU allocation while preserving the
    existing optimized transform implementation.
    """
    native = _get_native()
    if native is not None and hasattr(native, "fused_silu_mul"):
        return native.fused_silu_mul(x1, x2)
    return _ref_fused_silu_mul(x1, x2)


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
    if native is not None and hasattr(native, "dequantize_int8_simple"):
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
    if native is not None and hasattr(native, "dequantize_int8_simple_dtype"):
        _dtype_to_code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
        if out_dtype not in _dtype_to_code:
            raise ValueError(
                f"Unsupported out_dtype: {out_dtype}. Supported: float32, float16, bfloat16"
            )
        return native.dequantize_int8_simple_dtype(q, scale, _dtype_to_code[out_dtype])
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
    if native is not None and hasattr(native, "mm_int8"):
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
    input_act: Optional[str] = None,
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
        input_act: Optional ``gelu_tanh`` or ``swiglu`` activation applied before
            activation quantization. SwiGLU consumes concatenated ``[gate | up]``
            halves and therefore halves the input width.

    Returns:
        Result tensor [..., N] in out_dtype.
    """
    if out_dtype is None:
        out_dtype = x.dtype
    width = 2 if input_act == "swiglu" else 1
    if input_act not in (None, "none", "gelu_tanh", "swiglu"):
        _apply_input_act(x, input_act)
    if x.shape[-1] != weight.shape[-1] * width:
        raise ValueError(
            "Input and weight inner dimensions must match after input_act, "
            f"got {x.shape[-1]} and {weight.shape[-1]} with input_act={input_act!r}"
        )
    native = _get_native()
    if native is not None and hasattr(native, "int8_linear"):
        dtype_code = {
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
        }.get(out_dtype, 2)
        if (
            input_act == "swiglu"
            and not convrot
            and x.dtype in (torch.float16, torch.bfloat16)
            and hasattr(native, "fused_swiglu_quantize_rowwise")
            and hasattr(native, "int8_linear_prequantized")
        ):
            _clear_krea2_activation_cache()
            _clear_bmg_qkv_activation_cache()
            x_int8, x_scale = native.fused_swiglu_quantize_rowwise(x)
            return native.int8_linear_prequantized(
                x_int8,
                x_scale,
                weight,
                weight_scale,
                bias,
                dtype_code,
            )
        if (
            input_act == "gelu_tanh"
            and not convrot
            and _can_fuse_gelu_tanh_quantize(x, native)
        ):
            _clear_krea2_activation_cache()
            _clear_bmg_qkv_activation_cache()
            x_int8, x_scale = native.fused_gelu_tanh_quantize_rowwise(x)
            return native.int8_linear_prequantized(
                x_int8,
                x_scale,
                weight,
                weight_scale,
                bias,
                dtype_code,
            )
        if _can_fuse_h3_swiglu(
            x,
            native,
            weight,
            convrot,
            convrot_groupsize,
            input_act,
        ):
            x = _apply_h3_swiglu_exact(x, native)
        else:
            x = _apply_input_act(x, input_act)
        if _can_cache_krea2_int8_convrot(
            x,
            native,
            weight,
            weight_scale,
            bias,
            out_dtype,
            convrot,
            convrot_groupsize,
        ):
            _clear_bmg_qkv_activation_cache()
            x_int8, x_scale = _quantize_krea2_int8_convrot(x, native)
            return native.int8_linear_prequantized(
                x_int8,
                x_scale,
                weight,
                weight_scale,
                bias,
                dtype_code,
            )
        _clear_krea2_activation_cache()
        if _can_cache_int8_convrot_g16_bmg(
            x,
            native,
            weight,
            weight_scale,
            bias,
            out_dtype,
            convrot,
            convrot_groupsize,
        ):
            x_int8, x_scale = _quantize_int8_convrot_g16_bmg_qkv(
                x, native, weight.shape[0]
            )
            return native.int8_linear_prequantized(
                x_int8,
                x_scale,
                weight,
                weight_scale,
                bias,
                dtype_code,
            )
        _clear_bmg_qkv_activation_cache()
        if _can_quantize_int8_convrot_g16_bmg(
            x, native, convrot, convrot_groupsize
        ):
            x_int8, x_scale = native.quantize_int8_convrot_g16_bmg(x)
            return native.int8_linear_prequantized(
                x_int8,
                x_scale,
                weight,
                weight_scale,
                bias,
                dtype_code,
            )
        # Rotate through the native cached Hadamard-matrix implementation.
        if convrot:
            if x.shape[-1] % convrot_groupsize != 0:
                raise ValueError(
                    f"ConvRot group size {convrot_groupsize} does not divide "
                    f"input features {x.shape[-1]}"
                )
            if hasattr(native, "rotate_convrot"):
                x = native.rotate_convrot(x, convrot_groupsize)
            else:
                from ._reference import _build_hadamard, _rotate_activation

                h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
                x = _rotate_activation(x, h, convrot_groupsize)
        return native.int8_linear(
            x, weight, weight_scale, bias, dtype_code, False, convrot_groupsize
        )
    _clear_krea2_activation_cache()
    _clear_bmg_qkv_activation_cache()
    x = _apply_input_act(x, input_act)
    return _ref_int8_linear(
        x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
    )


def int8_linear_prequantized(
    x_int8: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """INT8 linear layer for an already rowwise-quantized activation.

    This API does not quantize the activation. It is intended for sharing one
    activation quantization across multiple Linear calls and for consuming the
    output of a fused producer such as SwiGLU-plus-quantize.

    Args:
        x_int8: Rowwise-quantized activation tensor [..., K] in INT8.
        x_scale: One activation scale per flattened input row.
        weight: INT8 weight tensor [N, K].
        weight_scale: Weight scale (scalar or per-channel [N] or [N,1]).
        bias: Optional bias tensor [N].
        out_dtype: Output dtype (float32, float16, or bfloat16).

    Returns:
        Result tensor [..., N] in out_dtype.
    """
    dtype_code = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }.get(out_dtype)
    if dtype_code is None:
        raise ValueError(
            f"Unsupported out_dtype: {out_dtype}. Supported: "
            "float32, float16, bfloat16"
        )

    _clear_krea2_activation_cache()
    _clear_bmg_qkv_activation_cache()
    native = _get_native()
    if native is not None and hasattr(native, "int8_linear_prequantized"):
        return native.int8_linear_prequantized(
            x_int8,
            x_scale,
            weight,
            weight_scale,
            bias,
            dtype_code,
        )
    return _ref_int8_linear_prequantized(
        x_int8,
        x_scale,
        weight,
        weight_scale,
        bias=bias,
        out_dtype=out_dtype,
    )


def int8_linear_shared_input(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight_scale1: torch.Tensor,
    weight2: torch.Tensor,
    weight_scale2: torch.Tensor,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run two INT8 linear projections with one activation quantization.

    ConvRot, when requested, is also applied once and therefore must be shared
    by both weights.
    """
    _clear_krea2_activation_cache()
    _clear_bmg_qkv_activation_cache()
    if out_dtype is None:
        out_dtype = x.dtype
    dtype_code = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }.get(out_dtype)
    if dtype_code is None:
        raise ValueError(
            f"Unsupported out_dtype: {out_dtype}. Supported: "
            "float32, float16, bfloat16"
        )

    native = _get_native()
    if native is not None and hasattr(native, "int8_linear_shared_input"):
        if _can_pair_int8_convrot_g16_bmg(
            x,
            native,
            weight1,
            weight_scale1,
            weight2,
            weight_scale2,
            bias1,
            bias2,
            out_dtype,
            convrot,
            convrot_groupsize,
        ):
            x_int8, x_scale = native.quantize_int8_convrot_g16_bmg(x)
            return native.int8_linear_pair_prequantized(
                x_int8,
                x_scale,
                weight1,
                weight_scale1,
                weight2,
                weight_scale2,
                dtype_code,
            )
        if convrot:
            if x.shape[-1] % convrot_groupsize != 0:
                raise ValueError(
                    f"ConvRot group size {convrot_groupsize} does not divide "
                    f"input features {x.shape[-1]}"
                )
            if hasattr(native, "rotate_convrot"):
                x = native.rotate_convrot(x, convrot_groupsize)
            else:
                from ._reference import _build_hadamard, _rotate_activation

                h = _build_hadamard(
                    convrot_groupsize, device=x.device, dtype=x.dtype
                )
                x = _rotate_activation(x, h, convrot_groupsize)
        return native.int8_linear_shared_input(
            x,
            weight1,
            weight_scale1,
            weight2,
            weight_scale2,
            bias1,
            bias2,
            dtype_code,
        )

    return _ref_int8_linear_shared_input(
        x,
        weight1,
        weight_scale1,
        weight2,
        weight_scale2,
        bias1=bias1,
        bias2=bias2,
        out_dtype=out_dtype,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
    )


def rotate_convrot(
    x: torch.Tensor,
    group_size: int = 256,
) -> torch.Tensor:
    """Apply the online groupwise Hadamard activation rotation."""
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"features {x.shape[-1]} not divisible by group_size {group_size}"
        )
    native = _get_native()
    if native is not None and hasattr(native, "rotate_convrot"):
        return native.rotate_convrot(x, group_size)

    from ._reference import _build_hadamard, _rotate_activation

    h = _build_hadamard(group_size, device=x.device, dtype=x.dtype)
    return _rotate_activation(x, h, group_size)


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
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"input features {weight.shape[-1]} not divisible by group_size {group_size}"
        )
    native = _get_native()
    if native is not None and hasattr(native, "quantize_int8_convrot_weight"):
        return native.quantize_int8_convrot_weight(
            weight, group_size, stochastic_rounding
        )
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
    if native is not None and hasattr(native, "dequantize_int8_convrot_weight"):
        return native.dequantize_int8_convrot_weight(q, scale, group_size)
    return _ref_dequantize_int8_convrot_weight(q, scale, group_size)


def int8_cache_clear() -> None:
    """Clear cached oneDNN INT8 primitive state."""
    _clear_krea2_activation_cache()
    _clear_bmg_qkv_activation_cache()
    native = _get_native()
    if native is not None and hasattr(native, "int8_cache_clear"):
        native.int8_cache_clear()


def int8_cache_stats() -> dict:
    """Return INT8 primitive cache counters and size."""
    native = _get_native()
    if native is not None and hasattr(native, "int8_cache_stats"):
        hits, misses, size = native.int8_cache_stats()
        return {"hits": hits, "misses": misses, "size": size}
    return {"hits": 0, "misses": 0, "size": 0}


__all__ = [
    "quantize_int8_tensorwise",
    "quantize_int8_rowwise",
    "fused_silu_mul_quantize_rowwise",
    "fused_swiglu_quantize_rowwise",
    "fused_gelu_tanh_quantize_rowwise",
    "fused_silu_mul",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "mm_int8",
    "int8_linear",
    "int8_linear_prequantized",
    "int8_linear_shared_input",
    "rotate_convrot",
    "quantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight",
    "int8_cache_clear",
    "int8_cache_stats",
]
