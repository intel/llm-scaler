"""Route supported Lumina/Z-Image INT8 FFNs through fused Omni XPU kernels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_int8_ffn_original__"
_omni_int8 = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}


@dataclass(frozen=True)
class _Weight:
    qdata: torch.Tensor
    scale: torch.Tensor
    convrot: bool
    convrot_groupsize: int


def _module_weight(module: Any, x: torch.Tensor) -> tuple[_Weight | None, str]:
    """Extract an already-resident TensorWise INT8 weight without moving it."""
    weight = getattr(module, "weight", None)
    if weight is None:
        return None, "missing_weight"
    if getattr(module, "quant_format", None) != "int8_tensorwise":
        return None, "quant_format"
    if getattr(module, "layout_type", None) != "TensorWiseINT8Layout":
        return None, "layout"
    if getattr(module, "_full_precision_mm", False):
        return None, "full_precision_mm"
    if getattr(module, "comfy_force_cast_weights", False):
        return None, "force_cast_weights"
    if len(getattr(module, "weight_function", ())) != 0:
        return None, "weight_function"
    if len(getattr(module, "bias_function", ())) != 0:
        return None, "bias_function"
    if getattr(module, "bias", None) is not None:
        return None, "bias"
    if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
        return None, "weight_layout"

    qdata = getattr(weight, "_qdata", None)
    params = getattr(weight, "_params", None)
    scale = getattr(params, "scale", None)
    if not isinstance(qdata, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return None, "weight_storage"
    if qdata.dtype != torch.int8 or qdata.ndim != 2:
        return None, "weight_storage"
    if qdata.device != x.device or scale.device != x.device:
        return None, "offloaded_weight"
    if getattr(params, "orig_dtype", None) != x.dtype:
        return None, "logical_dtype"
    if getattr(params, "transposed", False):
        return None, "transposed_weight"

    return (
        _Weight(
            qdata=qdata,
            scale=scale,
            convrot=bool(getattr(params, "convrot", False)),
            convrot_groupsize=int(getattr(params, "convrot_groupsize", 256)),
        ),
        "",
    )


def _route_inputs(
    module: Any, x: Any
) -> tuple[tuple[_Weight, _Weight, _Weight] | None, str]:
    if not isinstance(x, torch.Tensor):
        return None, "input_type"
    if x.device.type != "xpu":
        return None, "device"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return None, "input_dtype"
    if x.ndim not in (2, 3) or x.shape[-1] == 0:
        return None, "input_shape"
    if x.requires_grad:
        return None, "requires_grad"

    weights = []
    for name in ("w1", "w3", "w2"):
        linear = getattr(module, name, None)
        if linear is None:
            return None, f"missing_{name}"
        extracted, reason = _module_weight(linear, x)
        if extracted is None:
            return None, f"{name}_{reason}"
        weights.append(extracted)

    w1, w3, w2 = weights
    input_features = x.shape[-1]
    if w1.qdata.shape[1] != input_features or w3.qdata.shape[1] != input_features:
        return None, "up_input_shape"
    if w1.qdata.shape[0] != w3.qdata.shape[0]:
        return None, "up_output_shape"
    if w2.qdata.shape != (input_features, w1.qdata.shape[0]):
        return None, "down_shape"
    if (w1.convrot, w1.convrot_groupsize) != (
        w3.convrot,
        w3.convrot_groupsize,
    ):
        return None, "up_convrot_mismatch"
    for name, weight in (("up", w1), ("down", w2)):
        if weight.convrot:
            size = weight.convrot_groupsize
            remaining = size
            while remaining > 1 and remaining % 4 == 0:
                remaining //= 4
            if size < 4 or remaining != 1:
                return None, f"{name}_convrot_groupsize"
            if weight.qdata.shape[1] % size != 0:
                return None, f"{name}_convrot_shape"

    return (w1, w3, w2), ""


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


def apply():
    global _omni_int8

    try:
        from omni_xpu_kernel import int8 as _candidate
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"

    required = (
        "int8_linear_shared_input",
        "fused_silu_mul",
        "fused_silu_mul_quantize_rowwise",
        "rotate_convrot",
        "quantize_int8_rowwise",
        "int8_linear_prequantized",
    )
    missing = [name for name in required if not hasattr(_candidate, name)]
    if missing:
        return False, f"omni_xpu_kernel.int8 missing {', '.join(missing)}"

    try:
        import comfy.ops as comfy_ops
        import comfy.ldm.lumina.model as lumina_model
    except ImportError as exc:
        return False, f"Lumina FeedForward unavailable ({exc})"

    feed_forward = getattr(lumina_model, "FeedForward", None)
    if feed_forward is None or not hasattr(feed_forward, "forward"):
        return False, "Lumina FeedForward.forward not found"
    if hasattr(feed_forward.forward, _PATCH_MARKER):
        return True, ""

    _omni_int8 = _candidate
    original_forward = feed_forward.forward

    def _forward(self, x):
        global _routed_calls

        weights, reason = _route_inputs(self, x)
        if weights is None:
            _record_fallback(reason)
            log_debug_event(
                "dispatch",
                "lumina.FeedForward",
                {"input": x},
                details={"route": "comfy", "reason": reason},
                verbose_only=True,
            )
            return original_forward(self, x)

        w1, w3, w2 = weights
        comfy_ops.run_every_op()
        up1, up3 = _omni_int8.int8_linear_shared_input(
            x,
            w1.qdata,
            w1.scale,
            w3.qdata,
            w3.scale,
            out_dtype=x.dtype,
            convrot=w1.convrot,
            convrot_groupsize=w1.convrot_groupsize,
        )
        if w2.convrot:
            gated = _omni_int8.fused_silu_mul(up1, up3)
            del up1, up3
            rotated = _omni_int8.rotate_convrot(
                gated, w2.convrot_groupsize
            )
            del gated
            gated_q, gated_scale = _omni_int8.quantize_int8_rowwise(rotated)
            del rotated
            route = "shared_up+fused_swiglu+convrot+quant+prequant_down"
        else:
            gated_q, gated_scale = _omni_int8.fused_silu_mul_quantize_rowwise(
                up1, up3
            )
            del up1, up3
            route = "shared_up+fused_swiglu_quant+prequant_down"
        output = _omni_int8.int8_linear_prequantized(
            gated_q,
            gated_scale,
            w2.qdata,
            w2.scale,
            out_dtype=x.dtype,
        )
        _routed_calls += 1
        log_debug_event(
            "kernel",
            "int8_swiglu_mlp",
            {
                "input": x,
                "up_weight": w1.qdata,
                "gate_weight": w3.qdata,
                "down_weight": w2.qdata,
                "output": output,
            },
            details={
                "backend": "omni_xpu",
                "route": route,
                "up_convrot": w1.convrot,
                "down_convrot": w2.convrot,
            },
        )
        return output

    setattr(_forward, _PATCH_MARKER, original_forward)
    feed_forward.forward = _forward
    log.info(
        "[OmniXPU] INT8 FFN: routed eligible Lumina FeedForward through fused kernels"
    )
    return True, ""


__all__ = ["apply", "get_stats"]
