"""Native operators for Comfy Kitchen on Intel XPU."""

from __future__ import annotations

import torch

from omni_xpu_kernel import _load_extension
from .. import _compile_meta as _meta
from .._compile_ops import compile_op


def supports_decode_attention() -> bool:
    try:
        native = _load_extension().kitchen
        return all(hasattr(native, name) for name in ("decode_attention", "decode_step_merge"))
    except (AttributeError, ImportError):
        return False


def supports_int8_decode() -> bool:
    try:
        native = _load_extension().kitchen
        return all(hasattr(native, name) for name in ("int8_decode_update", "int8_decode"))
    except (AttributeError, ImportError):
        return False


def supports_gated_delta_deferred() -> bool:
    try:
        native = _load_extension().kitchen
        return all(hasattr(native, name) for name in ("deltanet_conv_deferred", "gated_delta_decode_deferred"))
    except (AttributeError, ImportError):
        return False


@compile_op("kitchen_deltanet_conv_deferred", _meta.void,
            mutates_args=("conv_state", "proj_buf", "qkv_buf"))
def deltanet_conv_deferred(proj: torch.Tensor, conv_state: torch.Tensor,
                           conv_w: torch.Tensor, conv_b: torch.Tensor | None,
                           proj_buf: torch.Tensor, qkv_buf: torch.Tensor, ctl: torch.Tensor) -> None:
    """Commit the previous accepted projections; verify current conv into side buffers."""
    args = (proj.contiguous(),conv_state,conv_w.contiguous(),
            None if conv_b is None else conv_b.contiguous(),proj_buf,qkv_buf,ctl)
    if torch.compiler.is_compiling():
        torch.ops.omni_xpu.kitchen_deltanet_conv_deferred(*args)
    else:
        _load_extension().kitchen.deltanet_conv_deferred(*args)


@compile_op("kitchen_gated_delta_decode_deferred", _meta.kitchen_deferred_delta,
            mutates_args=("state", "gates_buf", "sumsq_buf"))
def gated_delta_decode_deferred(x: torch.Tensor, w_a: torch.Tensor, w_b: torch.Tensor,
    dt_bias: torch.Tensor, g_decay: torch.Tensor, state: torch.Tensor, key_dim: int,
    key_heads: int, scale: float, z: torch.Tensor, norm_weight: torch.Tensor, eps: float,
    qkv_buf: torch.Tensor, gates_buf: torch.Tensor, sumsq_buf: torch.Tensor, ctl: torch.Tensor) -> torch.Tensor:
    """Commit the previous accepted prefix and verify current rows without committing them."""
    args = (x.contiguous(),w_a.contiguous(),w_b.contiguous(),dt_bias,g_decay,state,
            key_dim,key_heads,scale,z.contiguous(),norm_weight.contiguous(),eps,
            qkv_buf,gates_buf,sumsq_buf,ctl)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_gated_delta_decode_deferred(*args)
    return _load_extension().kitchen.gated_delta_decode_deferred(*args)


@compile_op("kitchen_int8_decode_update", _meta.void,
            mutates_args=("key", "value", "key_scale", "value_scale"))
def int8_decode_update(source_key: torch.Tensor, source_value: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       key_scale: torch.Tensor, value_scale: torch.Tensor,
                       length: torch.Tensor, capacity: int, initialize: bool = False) -> None:
    """Quantize the growing prefix or refresh its last two pages on device."""
    args = (source_key,source_value,key,value,key_scale,value_scale,length,capacity,initialize)
    if torch.compiler.is_compiling():
        torch.ops.omni_xpu.kitchen_int8_decode_update(*args)
    else:
        _load_extension().kitchen.int8_decode_update(*args)


@compile_op("kitchen_int8_decode", _meta.kitchen_int8_decode)
def int8_decode(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_scale: torch.Tensor, value_scale: torch.Tensor,
                length: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend a signed INT8 committed prefix, including query quantization."""
    args = (query,key,value,key_scale,value_scale,length,capacity)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_int8_decode(*args)
    out, lse = _load_extension().kitchen.int8_decode(*args)
    return out, lse


@compile_op("kitchen_decode_attention", _meta.kitchen_decode)
def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     lengths: torch.Tensor, causal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 D128/256 fixed-cache decode, returning output and natural-log LSE."""
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_decode_attention(q, k, v, lengths, causal)
    out, lse = _load_extension().kitchen.decode_attention(q, k, v, lengths, causal)
    return out, lse


@compile_op("kitchen_decode_step_merge", _meta.void, mutates_args=("merged",))
def decode_step_merge(out: torch.Tensor, lse: torch.Tensor, q: torch.Tensor,
                      k: torch.Tensor, v: torch.Tensor, merged: torch.Tensor) -> None:
    """Merge current causal rows into prefix attention; merged may alias out."""
    if torch.compiler.is_compiling():
        torch.ops.omni_xpu.kitchen_decode_step_merge(out, lse, q, k, v, merged)
    else:
        _load_extension().kitchen.decode_step_merge(out, lse, q, k, v, merged)


def supports_deltanet_conv_step() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "deltanet_conv_step")
    except (AttributeError, ImportError):
        return False


def supports_gated_delta_decode_fused() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "gated_delta_decode_fused")
    except (AttributeError, ImportError):
        return False


def supports_group_norm_silu_pad3d() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "group_norm_silu_pad3d")
    except (AttributeError, ImportError):
        return False


def supports_gemv_awq_w4a16() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "gemv_awq_w4a16")
    except (AttributeError, ImportError):
        return False


def supports_group_norm_silu_pad3d_out() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "group_norm_silu_pad3d_out")
    except (AttributeError, ImportError):
        return False


def supports_fp16_linear() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "fp16_linear")
    except (AttributeError, ImportError):
        return False


def supports_fp16_conv3d() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "fp16_conv3d")
    except (AttributeError, ImportError):
        return False


def supports_fp16_conv3d_out() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "fp16_conv3d_out")
    except (AttributeError, ImportError):
        return False


def supports_rms_norm_for_int8() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "rms_norm_for_int8")
    except (AttributeError, ImportError):
        return False


def supports_rms_norm_quantize_int8() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "rms_norm_quantize_int8")
    except (AttributeError, ImportError):
        return False


def supports_rms_norm_convrot_quantize_int8() -> bool:
    try:
        return hasattr(
            _load_extension().kitchen, "rms_norm_convrot_quantize_int8"
        )
    except (AttributeError, ImportError):
        return False


def supports_gated_quantize_int8() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "gated_quantize_int8")
    except (AttributeError, ImportError):
        return False


def supports_scaled_residual() -> bool:
    try:
        return hasattr(_load_extension().kitchen, "scaled_residual")
    except (AttributeError, ImportError):
        return False


@compile_op("kitchen_deltanet_conv_step", _meta.kitchen_delta_conv,
            mutates_args=("conv_state", "snapshots"))
def deltanet_conv_step(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None = None,
    snapshots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute up to eight causal decode steps and update ``conv_state`` in place."""
    args = (proj.contiguous(), conv_state, conv_w.contiguous(),
            None if conv_b is None else conv_b.contiguous(), snapshots)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_deltanet_conv_step(*args)
    return _load_extension().kitchen.deltanet_conv_step(*args)


@compile_op("kitchen_gated_delta_decode_fused", _meta.kitchen_gated_delta,
            mutates_args=("state", "snapshots"))
def gated_delta_decode_fused(
    mixed_qkv: torch.Tensor,
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    dt_bias: torch.Tensor,
    g_decay: torch.Tensor,
    state: torch.Tensor,
    key_dim: int,
    key_heads: int,
    scale: float,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    snapshots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode at most eight steps while updating ``state`` in place."""
    args = (mixed_qkv.contiguous(), x.contiguous(), w_a.contiguous(),
            w_b.contiguous(), dt_bias, g_decay, state, key_dim, key_heads,
            scale, z.contiguous(), norm_weight.contiguous(), eps, snapshots)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_gated_delta_decode_fused(*args)
    return _load_extension().kitchen.gated_delta_decode_fused(*args)


@compile_op("kitchen_gemv_awq_w4a16", _meta.kitchen_awq)
def gemv_awq_w4a16(
    input: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """Run native Kitchen AWQ W4A16 GEMV or dequantized GEMM on XPU."""
    args = (input.contiguous(), packed.contiguous(),
            scales.contiguous(), zeros.contiguous(),
            None if bias is None else bias.contiguous(), group_size)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_gemv_awq_w4a16(*args)
    return _load_extension().kitchen.gemv_awq_w4a16(*args)


@compile_op("kitchen_group_norm_silu_pad3d", _meta.kitchen_group_norm_pad)
def group_norm_silu_pad3d(
    input: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    groups: int = 32,
    eps: float = 1e-6,
    pad: list[int] | None = None,
    silu: bool = True,
    zero_pad: bool = False,
) -> torch.Tensor:
    """Run per-frame GroupNorm, optional SiLU, and causal 3D padding."""
    args = (input, weight, bias, groups, eps,
            [0, 0, 0, 0, 0] if pad is None else list(pad), silu,
            zero_pad)
    if not supports_group_norm_silu_pad3d_out():
        if zero_pad:
            raise RuntimeError("Omni XPU zero_pad requires the updated native core")
        args = args[:-1]
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_group_norm_silu_pad3d(*args)
    return _load_extension().kitchen.group_norm_silu_pad3d(*args)


@compile_op(
    "kitchen_group_norm_silu_pad3d_out", _meta.kitchen_group_norm_pad_out,
    mutates_args=("out",),
)
def group_norm_silu_pad3d_out(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
    zero_pad: bool,
    out: torch.Tensor,
) -> None:
    """Write native per-frame GroupNorm, SiLU and padding into out."""
    args = (input, weight, bias, groups, eps, list(pad), silu, zero_pad, out)
    if torch.compiler.is_compiling():
        torch.ops.omni_xpu.kitchen_group_norm_silu_pad3d_out(*args)
    else:
        _load_extension().kitchen.group_norm_silu_pad3d_out(*args)


@compile_op("kitchen_fp16_linear", _meta.kitchen_linear)
def fp16_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a native FP16 XPU projection and optional scaled residual."""
    args = (input.contiguous(), weight.contiguous(),
            None if bias is None else bias.contiguous(),
            None if residual is None else residual.contiguous(),
            None if residual_scale is None else residual_scale.contiguous())
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_fp16_linear(*args)
    return _load_extension().kitchen.fp16_linear(*args)


@compile_op("kitchen_fp16_conv3d", _meta.kitchen_conv3d)
def fp16_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    stride: list[int] | None = None,
) -> torch.Tensor:
    """Run an XPU Conv3D with optional bias and residual."""
    native_input = input if supports_fp16_conv3d_out() else input.contiguous()
    args = (native_input, weight.contiguous(),
            None if bias is None else bias.contiguous(),
            None if residual is None else residual.contiguous(),
            [1, 1, 1] if stride is None else list(stride))
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_fp16_conv3d(*args)
    return _load_extension().kitchen.fp16_conv3d(*args)


@compile_op(
    "kitchen_fp16_conv3d_out", _meta.kitchen_conv3d_out,
    mutates_args=("out",),
)
def fp16_conv3d_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: list[int],
    out: torch.Tensor,
) -> None:
    """Write native FP16 Conv3D into out, including a strided output view."""
    args = (input, weight.contiguous(),
            None if bias is None else bias.contiguous(),
            None if residual is None else residual.contiguous(),
            list(stride), out)
    if torch.compiler.is_compiling():
        torch.ops.omni_xpu.kitchen_fp16_conv3d_out(*args)
    else:
        _load_extension().kitchen.fp16_conv3d_out(*args)


@compile_op("kitchen_rms_norm_for_int8", _meta.unchanged)
def rms_norm_for_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize Kitchen INT8 input on XPU before native quantization."""
    args = (input.contiguous(), weight.contiguous(), eps)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_rms_norm_for_int8(*args)
    return _load_extension().kitchen.rms_norm_for_int8(*args)


@compile_op("kitchen_rms_norm_quantize_int8", _meta.kitchen_rms_norm_quantize)
def rms_norm_quantize_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Kitchen RMSNorm and rowwise INT8 quantization on XPU."""
    args = (input.contiguous(), weight.contiguous(), eps)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_rms_norm_quantize_int8(*args)
    return _load_extension().kitchen.rms_norm_quantize_int8(*args)


@compile_op(
    "kitchen_rms_norm_convrot_quantize_int8",
    _meta.kitchen_rms_norm_convrot_quantize,
)
def rms_norm_convrot_quantize_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Kitchen RMSNorm, ConvRot and rowwise INT8 quantization."""
    args = (input.contiguous(), weight.contiguous(), eps, group_size)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_rms_norm_convrot_quantize_int8(*args)
    return _load_extension().kitchen.rms_norm_convrot_quantize_int8(*args)


@compile_op("kitchen_scaled_residual", _meta.unchanged)
def scaled_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    residual_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply Kitchen's scaled residual using a native XPU kernel."""
    args = (input.contiguous(), residual.contiguous(),
            residual_scale.contiguous())
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_scaled_residual(*args)
    return _load_extension().kitchen.scaled_residual(*args)


@compile_op("kitchen_gated_quantize_int8", _meta.kitchen_gated_quantize)
def gated_quantize_int8(
    input: torch.Tensor,
    input_act: str,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiTU/SwiGLU, optional ConvRot and INT8 quantization for K<=8192."""
    args = (input.contiguous(), input_act, group_size)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.kitchen_gated_quantize_int8(*args)
    return _load_extension().kitchen.gated_quantize_int8(*args)


__all__ = [
    "gated_quantize_int8", "supports_gated_quantize_int8",
    "deltanet_conv_deferred", "gated_delta_decode_deferred", "supports_gated_delta_deferred",
    "int8_decode_update", "int8_decode", "supports_int8_decode",
    "decode_attention", "decode_step_merge", "supports_decode_attention",
    "deltanet_conv_step",
    "gated_delta_decode_fused",
    "group_norm_silu_pad3d",
    "group_norm_silu_pad3d_out",
    "fp16_linear",
    "fp16_conv3d",
    "gemv_awq_w4a16",
    "fp16_conv3d_out",
    "rms_norm_for_int8",
    "rms_norm_quantize_int8",
    "rms_norm_convrot_quantize_int8",
    "scaled_residual",
    "supports_deltanet_conv_step",
    "supports_gated_delta_decode_fused",
    "supports_group_norm_silu_pad3d",
    "supports_group_norm_silu_pad3d_out",
    "supports_fp16_linear",
    "supports_fp16_conv3d",
    "supports_gemv_awq_w4a16",
    "supports_fp16_conv3d_out",
    "supports_rms_norm_for_int8",
    "supports_rms_norm_quantize_int8",
    "supports_rms_norm_convrot_quantize_int8",
    "supports_scaled_residual",
]
