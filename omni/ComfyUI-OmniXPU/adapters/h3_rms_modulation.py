"""Route the exact MiniMax H3 RMSNorm/modulation boundary to Omni XPU."""

from __future__ import annotations

import hashlib
import inspect
import logging
from functools import wraps
from typing import Any, Callable

import torch

from ..patches.debug import log_debug_event, trace_patch


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_h3_rms_modulation_original__"
_EXPECTED_FORWARD_SHA256 = (
    "a117b068b48abfc4e3b6e0a92fdf6b964043028f9846212562ec192a7a9136e5"
)
_HIDDEN_SIZE = 5376
_MODULATION_EXPAND = 6
_MAX_SEGMENTS = 8
_omni_norm = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}


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


def _block_input_reason(block: Any, x: Any, segments: Any) -> str:
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
    for name in ("norm1", "norm2"):
        layer = getattr(block, name, None)
        if layer is None:
            return f"{name}_missing"
        normalized_shape = getattr(layer, "normalized_shape", None)
        try:
            normalized_shape = tuple(normalized_shape)
        except TypeError:
            return f"{name}_shape"
        if normalized_shape != (_HIDDEN_SIZE,):
            return f"{name}_shape"
        weight = getattr(layer, "weight", None)
        if weight is None or tuple(weight.shape) != (_HIDDEN_SIZE,):
            return f"{name}_weight"
    return _segment_reason(segments, x.shape[0], 1 << 30)


def _modulation_reason(values: Any, x: torch.Tensor, segments: Any) -> str:
    if not isinstance(values, (tuple, list)) or len(values) != _MODULATION_EXPAND:
        return "adaln_outputs"
    expected_stride = (_MODULATION_EXPAND * _HIDDEN_SIZE, 1)
    rows = None
    for value in values:
        if not isinstance(value, torch.Tensor):
            return "modulation_type"
        if value.device != x.device or value.dtype not in (
            torch.bfloat16, torch.float32
        ):
            return "modulation_device_dtype"
        if value.ndim != 2 or value.shape[1] != _HIDDEN_SIZE:
            return "modulation_shape"
        if tuple(value.stride()) != expected_stride:
            return "modulation_layout"
        if value.requires_grad:
            return "modulation_requires_grad"
        if rows is None:
            rows = value.shape[0]
        elif value.shape[0] != rows:
            return "modulation_rows"
    return _segment_reason(segments, x.shape[0], int(rows))


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
        # Curve-based H3 checkpoints produce FP32 modulation. Match the
        # original _mod_scale_shift: cast before 1 + scale and before shift
        # addition, while leaving the original gate path untouched.
        output = _omni_norm.rms_norm_segmented_modulation(
            weight,
            x,
            scale.to(dtype=x.dtype),
            shift.to(dtype=x.dtype),
            segments,
            eps=eps,
        )
    return output, ""


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
    if target is None or not callable(getattr(target, "forward", None)):
        return False, "MiniMax H3 DiTBlock.forward is unavailable"
    original = target.forward
    if hasattr(original, _PATCH_MARKER):
        return True, "already patched"
    try:
        source_hash = _source_sha256(original)
    except (OSError, TypeError):
        return False, "MiniMax H3 DiTBlock.forward source is unavailable"
    if source_hash != _EXPECTED_FORWARD_SHA256:
        return False, "MiniMax H3 DiTBlock.forward source changed"

    _omni_norm = candidate

    @wraps(original)
    def patched(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        global _routed_calls

        if not policy_supported(x):
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )

        reason = _block_input_reason(self, x, mod_segments)
        if reason or comfy.model_management.in_training:
            _record_fallback(reason or "training")
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )

        modulation = self.adaln_proj(t_emb)
        reason = _modulation_reason(modulation, x, mod_segments)
        if reason:
            _record_fallback(reason)
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation

        h, reason = _run_fused(
            comfy_ops,
            self.norm1,
            x,
            scale_msa,
            shift_msa,
            mod_segments,
        )
        if h is None:
            _record_fallback(reason)
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )

        x = h3_model._mod_gate(
            x,
            gate_msa,
            self.attn(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            ),
            mod_segments,
        )
        h, reason = _run_fused(
            comfy_ops,
            self.norm2,
            x,
            scale_mlp,
            shift_mlp,
            mod_segments,
        )
        if h is None:
            raise RuntimeError(
                "H3 norm2 modulation contract changed after routing: "
                f"{reason}"
            )
        output = h3_model._mod_gate(
            x,
            gate_mlp,
            self.mlp(h),
            mod_segments,
        )

        _routed_calls += 2
        log_debug_event(
            "kernel",
            "h3_rms_norm_segmented_modulation",
            {"input": x, "output": output},
            details={
                "backend": "bmg_sycl",
                "segments": len(mod_segments),
                "fused_calls": 2,
            },
        )
        return output

    patched = trace_patch(
        "norm.H3DiTBlock.rms_modulation",
        ("self", "x", "t_emb", "mod_segments", "rope_freqs"),
        stage="dispatch",
        verbose_only=True,
    )(patched)
    setattr(patched, _PATCH_MARKER, original)
    target.forward = patched
    log.info("[OmniXPU] norm: patched MiniMax H3 segmented RMS modulation")
    return True, ""


__all__ = ["apply", "get_stats"]
