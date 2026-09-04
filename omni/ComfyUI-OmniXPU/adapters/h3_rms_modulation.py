"""Route MiniMax H3 segmented RMSNorm modulation without copying model logic."""

from __future__ import annotations

import hashlib
import inspect
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import torch

from ..patches.debug import log_debug_event, trace_patch


log = logging.getLogger("ComfyUI-OmniXPU")

_FORWARD_PATCH_MARKER = "__omnixpu_h3_rms_modulation_context__"
_MODULATE_PATCH_MARKER = "__omnixpu_h3_rms_modulation_original__"
_NORM_PATCH_MARKER = "__omnixpu_h3_deferred_rms_norm_original__"
_EXPECTED_FORWARD_SHA256 = (
    "a117b068b48abfc4e3b6e0a92fdf6b964043028f9846212562ec192a7a9136e5"
)
_EXPECTED_MODULATE_SHA256 = (
    "796cae2505e9742d971327bf446c03dc76e3f7c0d97f97fffbc37e37ce6a8e64"
)
_HIDDEN_SIZE = 5376
_MODULATION_EXPAND = 6
_MAX_SEGMENTS = 8
_active_h3_block: ContextVar[Any | None] = ContextVar(
    "omnixpu_active_h3_rms_modulation_block", default=None
)
_omni_norm = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}


@dataclass(frozen=True)
class _DeferredRmsNorm:
    layer: Any
    original_forward: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @property
    def input(self) -> Any:
        return self.args[0]

    def fallback(self) -> Any:
        return self.original_forward(self.layer, *self.args, **self.kwargs)


def _record_fallback(reason: str) -> None:
    global _fallback_calls
    _fallback_calls += 1
    _fallback_reasons[reason] = _fallback_reasons.get(reason, 0) + 1


def get_stats() -> dict[str, Any]:
    return {
        "routed": _routed_calls,
        "fallback": _fallback_calls,
        "reasons": dict(_fallback_reasons),
    }


def _source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _segment_reason(segments: Any, rows: int, modulation_rows: int) -> str:
    if not isinstance(segments, (tuple, list)):
        return "segment_type"
    if not 1 <= len(segments) <= _MAX_SEGMENTS:
        return "segment_count"
    previous_stop = 0
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) != 3:
            return "segment_shape"
        if any(type(value) is not int for value in segment):
            return "segment_value_type"
        start, stop, modulation_row = segment
        if start != previous_stop or stop <= start:
            return "segment_coverage"
        if not 0 <= modulation_row < modulation_rows:
            return "segment_modulation_row"
        previous_stop = stop
    if previous_stop != rows:
        return "segment_coverage"
    return ""


def _input_reason(layer: Any, x: Any, segments: Any) -> str:
    if not isinstance(x, torch.Tensor):
        return "input_type"
    if x.dtype != torch.bfloat16:
        return "input_dtype"
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != _HIDDEN_SIZE:
        return "input_shape"
    if not x.is_contiguous():
        return "input_layout"
    if x.requires_grad:
        return "requires_grad"
    normalized_shape = getattr(layer, "normalized_shape", None)
    try:
        normalized_shape = tuple(normalized_shape)
    except TypeError:
        return "norm_shape"
    if normalized_shape != (_HIDDEN_SIZE,):
        return "norm_shape"
    weight = getattr(layer, "weight", None)
    if weight is None or tuple(weight.shape) != (_HIDDEN_SIZE,):
        return "norm_weight"
    return _segment_reason(segments, x.shape[0], 1 << 30)


def _modulation_reason(
    shift: Any,
    scale: Any,
    x: torch.Tensor,
    segments: Any,
) -> str:
    expected_stride = (_MODULATION_EXPAND * _HIDDEN_SIZE, 1)
    for value in (shift, scale):
        if not isinstance(value, torch.Tensor):
            return "modulation_type"
        if value.device != x.device or value.dtype != torch.bfloat16:
            return "modulation_device_dtype"
        if value.ndim != 2 or value.shape[1] != _HIDDEN_SIZE:
            return "modulation_shape"
        if tuple(value.stride()) != expected_stride:
            return "modulation_layout"
        if value.requires_grad:
            return "modulation_requires_grad"
    if shift.shape != scale.shape:
        return "modulation_rows"
    return _segment_reason(segments, x.shape[0], scale.shape[0])


def _weight_reason(weight: Any, bias: Any, x: torch.Tensor) -> str:
    if bias is not None:
        return "norm_bias"
    if not isinstance(weight, torch.Tensor):
        return "norm_weight_type"
    if weight.device != x.device or weight.dtype != torch.bfloat16:
        return "norm_weight_device_dtype"
    if weight.ndim != 1 or weight.numel() != _HIDDEN_SIZE:
        return "norm_weight_shape"
    if not weight.is_contiguous():
        return "norm_weight_layout"
    return ""


def _run_fused(
    comfy_ops: Any,
    layer: Any,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    segments: list[tuple[int, int, int]],
) -> tuple[torch.Tensor | None, str]:
    comfy_ops.run_every_op()
    context = comfy_ops.CastBiasWeightContext(
        layer if layer.weight is not None else None,
        x,
        offloadable=True,
    )
    with context as (weight, bias):
        reason = _weight_reason(weight, bias, x)
        if reason:
            return None, reason
        eps = layer.eps if layer.eps is not None else 1e-6
        output = _omni_norm.rms_norm_segmented_modulation(
            weight,
            x,
            scale,
            shift,
            segments,
            eps=eps,
        )
    return output, ""


def _install_norm_forward_hook(layer: Any) -> bool:
    layer_type = type(layer)
    original = getattr(layer_type, "forward", None)
    if not callable(original):
        return False
    if hasattr(original, _NORM_PATCH_MARKER):
        return True

    @wraps(original)
    def deferred(layer_self, *args, **kwargs):
        block = _active_h3_block.get()
        is_block_norm = block is not None and (
            layer_self is getattr(block, "norm1", None)
            or layer_self is getattr(block, "norm2", None)
        )
        if is_block_norm and len(args) == 1 and not kwargs:
            return _DeferredRmsNorm(layer_self, original, args, kwargs)
        return original(layer_self, *args, **kwargs)

    setattr(deferred, _NORM_PATCH_MARKER, original)
    layer_type.forward = deferred
    return True


def _fallback(
    original_modulate: Callable[..., Any],
    pending: _DeferredRmsNorm,
    shift: Any,
    scale: Any,
    segments: Any,
    reason: str,
) -> Any:
    _record_fallback(reason)
    return original_modulate(pending.fallback(), shift, scale, segments)


def apply():
    global _omni_norm

    try:
        from omni_xpu_kernel import norm as candidate
    except ImportError:
        return False, "omni_xpu_kernel segmented RMS modulation is unavailable"
    capability = getattr(
        candidate, "supports_rms_norm_segmented_modulation", None
    )
    if not callable(capability) or not capability():
        return False, "native segmented RMS modulation is unavailable"
    policy_supported = getattr(
        candidate, "rms_norm_segmented_modulation_supported", None
    )
    if not callable(policy_supported):
        return False, "native segmented RMS modulation policy is unavailable"

    try:
        import comfy.model_management
        import comfy.ops as comfy_ops
        import comfy.ldm.minimax.model as h3_model
    except ImportError:
        return False, "MiniMax H3 ComfyUI model is unavailable"
    if not hasattr(comfy_ops, "CastBiasWeightContext") or not hasattr(
        comfy_ops, "run_every_op"
    ):
        return False, "ComfyUI cast/offload helpers are unavailable"

    target = getattr(h3_model, "DiTBlock", None)
    original_modulate = getattr(h3_model, "_mod_scale_shift", None)
    if target is None or not callable(getattr(target, "forward", None)):
        return False, "MiniMax H3 DiTBlock.forward is unavailable"
    if not callable(original_modulate):
        return False, "MiniMax H3 modulation helper is unavailable"

    original_forward = target.forward
    forward_patched = hasattr(original_forward, _FORWARD_PATCH_MARKER)
    modulate_patched = hasattr(original_modulate, _MODULATE_PATCH_MARKER)
    if forward_patched or modulate_patched:
        if forward_patched and modulate_patched:
            return True, "already patched"
        return False, "MiniMax H3 RMS modulation patch state is inconsistent"

    try:
        forward_hash = _source_sha256(original_forward)
        modulate_hash = _source_sha256(original_modulate)
    except (OSError, TypeError):
        return False, "MiniMax H3 source is unavailable"
    if forward_hash != _EXPECTED_FORWARD_SHA256:
        return False, "MiniMax H3 DiTBlock.forward source changed"
    if modulate_hash != _EXPECTED_MODULATE_SHA256:
        return False, "MiniMax H3 modulation helper source changed"

    _omni_norm = candidate

    @wraps(original_modulate)
    def routed_modulate(h, shift, scale, segments):
        global _routed_calls

        if not isinstance(h, _DeferredRmsNorm):
            return original_modulate(h, shift, scale, segments)
        x = h.input
        reason = _input_reason(h.layer, x, segments)
        if reason or comfy.model_management.in_training:
            return _fallback(
                original_modulate,
                h,
                shift,
                scale,
                segments,
                reason or "training",
            )
        reason = _modulation_reason(shift, scale, x, segments)
        if reason:
            return _fallback(
                original_modulate, h, shift, scale, segments, reason
            )
        output, reason = _run_fused(
            comfy_ops, h.layer, x, scale, shift, segments
        )
        if output is None:
            return _fallback(
                original_modulate, h, shift, scale, segments, reason
            )

        _routed_calls += 1
        log_debug_event(
            "kernel",
            "h3_rms_norm_segmented_modulation",
            {"input": x, "output": output},
            details={
                "backend": "bmg_sycl",
                "segments": len(segments),
            },
        )
        return output

    @wraps(original_forward)
    def forward_with_rms_context(self, *args, **kwargs):
        x = args[0] if args else kwargs.get("x")
        if not policy_supported(x):
            return original_forward(self, *args, **kwargs)
        if not _install_norm_forward_hook(getattr(self, "norm1", None)):
            return original_forward(self, *args, **kwargs)
        if not _install_norm_forward_hook(getattr(self, "norm2", None)):
            return original_forward(self, *args, **kwargs)

        token = _active_h3_block.set(self)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _active_h3_block.reset(token)

    forward_with_rms_context = trace_patch(
        "norm.H3DiTBlock.rms_modulation",
        ("self", "x", "t_emb", "mod_segments", "rope_freqs"),
        stage="dispatch",
        verbose_only=True,
    )(forward_with_rms_context)
    setattr(
        forward_with_rms_context, _FORWARD_PATCH_MARKER, original_forward
    )
    setattr(routed_modulate, _MODULATE_PATCH_MARKER, original_modulate)
    h3_model._mod_scale_shift = routed_modulate
    target.forward = forward_with_rms_context
    log.info("[OmniXPU] norm: patched MiniMax H3 segmented RMS modulation")
    return True, ""


__all__ = ["apply", "get_stats"]
