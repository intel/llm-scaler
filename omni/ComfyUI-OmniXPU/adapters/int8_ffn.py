"""Route supported INT8 FFNs and exact Krea2 SwiGLU through fused XPU kernels."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..patches.debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_int8_ffn_original__"
_KREA_PATCH_MARKER = "__omnixpu_krea2_swiglu_original__"
_KREA_INPUT_SHAPE = (1, 4192, 6144)
_KREA_OUTPUT_SHAPE = (1, 4192, 16384)
_omni_int8 = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}
_krea_routed_calls = 0
_krea_fallback_calls = 0
_krea_fallback_reasons: dict[str, int] = {}


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

    projection_names = (
        ("w1", "w3", "w2")
        if all(hasattr(module, name) for name in ("w1", "w3", "w2"))
        else ("linear_1", "linear_3", "linear_2")
    )
    weights = []
    for role, name in zip(("w1", "w3", "w2"), projection_names):
        linear = getattr(module, name, None)
        if linear is None:
            return None, f"missing_{role}"
        extracted, reason = _module_weight(linear, x)
        if extracted is None:
            return None, f"{role}_{reason}"
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


def _record_krea_fallback(reason: str) -> None:
    global _krea_fallback_calls
    _krea_fallback_calls += 1
    _krea_fallback_reasons[reason] = (
        _krea_fallback_reasons.get(reason, 0) + 1
    )


def get_krea_stats() -> dict[str, Any]:
    return {
        "routed": _krea_routed_calls,
        "fallback": _krea_fallback_calls,
        "reasons": dict(_krea_fallback_reasons),
    }


def _can_route_krea_input(x: Any) -> tuple[bool, str]:
    if not isinstance(x, torch.Tensor):
        return False, "input_type"
    if x.device.type != "xpu":
        return False, "device"
    if x.dtype != torch.bfloat16:
        return False, "input_dtype"
    if tuple(x.shape) != _KREA_INPUT_SHAPE:
        return False, "input_shape"
    if not x.is_contiguous():
        return False, "input_layout"
    if x.requires_grad:
        return False, "requires_grad"
    return True, ""


def _krea_output_reason(gate: Any, up: Any) -> str:
    if not isinstance(gate, torch.Tensor) or not isinstance(up, torch.Tensor):
        return "output_type"
    if gate.device.type != "xpu" or up.device != gate.device:
        return "output_device"
    if gate.dtype != torch.bfloat16 or up.dtype != gate.dtype:
        return "output_dtype"
    if tuple(gate.shape) != _KREA_OUTPUT_SHAPE or up.shape != gate.shape:
        return "output_shape"
    if not gate.is_contiguous() or not up.is_contiguous():
        return "output_layout"
    if gate.requires_grad or up.requires_grad:
        return "output_requires_grad"
    return ""


def apply():
    global _omni_int8

    try:
        import omni_xpu_kernel as _omni_package
        from omni_xpu_kernel import int8 as _candidate
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"

    # The fused route starts with int8_linear_shared_input, which invokes the
    # same native dynamic rowwise quantizer exposed by the Kitchen XPU backend.
    # Skipping this wrapper lets the original Lumina path reach that safe eager
    # fallback instead of entering an uncatchable native process fault.
    target = getattr(_omni_package, "__xpu_target__", "")
    get_core_aot_target = getattr(_omni_package, "core_aot_target", None)
    core_aot_target = (
        get_core_aot_target() if callable(get_core_aot_target) else ""
    )
    if target == "ptl-h" and core_aot_target != target:
        return False, "PTL-H native dynamic rowwise INT8 quantization is guarded"

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
    except ImportError as exc:
        return False, f"ComfyUI operations unavailable ({exc})"

    targets = []
    try:
        import comfy.ldm.lumina.model as lumina_model

        targets.append(
            (
                "lumina.FeedForward",
                getattr(lumina_model, "FeedForward", None),
            )
        )
    except ImportError:
        pass
    try:
        import comfy.ldm.omnigen.omnigen2 as omnigen2_model

        targets.append(
            (
                "omnigen2.LuminaFeedForward",
                getattr(omnigen2_model, "LuminaFeedForward", None),
            )
        )
    except ImportError:
        pass

    krea_target = None
    krea_enabled = (
        target == "bmg"
        and os.environ.get("OMNIXPU_KREA2_SWIGLU", "1") != "0"
    )
    if krea_enabled:
        try:
            import comfy.ldm.krea2.model as krea2_model

            candidate_target = getattr(krea2_model, "SwiGLU", None)
            if candidate_target is not None and hasattr(
                candidate_target, "forward"
            ):
                krea_target = candidate_target
        except ImportError:
            pass

    targets = [
        (name, target)
        for name, target in targets
        if target is not None and hasattr(target, "forward")
    ]
    if not targets and krea_target is None:
        return False, "supported feed-forward modules unavailable"

    _omni_int8 = _candidate

    def make_forward(original_forward, target_name):
        def _forward(self, x):
            global _routed_calls

            weights, reason = _route_inputs(self, x)
            if weights is None:
                _record_fallback(reason)
                log_debug_event(
                    "dispatch",
                    target_name,
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
                gated_q, gated_scale = _omni_int8.quantize_int8_rowwise(
                    rotated
                )
                del rotated
                route = "shared_up+fused_swiglu+convrot+quant+prequant_down"
            else:
                gated_q, gated_scale = (
                    _omni_int8.fused_silu_mul_quantize_rowwise(up1, up3)
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
                    "module": target_name,
                    "route": route,
                    "up_convrot": w1.convrot,
                    "down_convrot": w2.convrot,
                },
            )
            return output

        setattr(_forward, _PATCH_MARKER, original_forward)
        return _forward

    patched_names = []
    for target_name, target in targets:
        if not hasattr(target.forward, _PATCH_MARKER):
            target.forward = make_forward(target.forward, target_name)
        patched_names.append(target_name)

    if krea_target is not None:
        original_forward = krea_target.forward
        if not hasattr(original_forward, _KREA_PATCH_MARKER):
            def _krea_forward(self, x):
                global _krea_routed_calls

                eligible, reason = _can_route_krea_input(x)
                if not eligible:
                    _record_krea_fallback(reason)
                    return original_forward(self, x)

                gate = self.gate(x)
                up = self.up(x)
                reason = _krea_output_reason(gate, up)
                if reason:
                    _record_krea_fallback(reason)
                    gated = F.silu(gate).mul_(up)
                else:
                    gated = _omni_int8.fused_silu_mul(gate, up)
                    _krea_routed_calls += 1
                    log_debug_event(
                        "kernel",
                        "krea2_swiglu",
                        {
                            "input": x,
                            "gate": gate,
                            "up": up,
                            "output": gated,
                        },
                        details={
                            "backend": "omni_xpu",
                            "module": "krea2.SwiGLU",
                            "route": "fused_silu_mul",
                        },
                    )
                return self.down(gated)

            setattr(_krea_forward, _KREA_PATCH_MARKER, original_forward)
            krea_target.forward = _krea_forward
        patched_names.append("krea2.SwiGLU")

    log.info(
        "[OmniXPU] INT8 FFN: routed eligible %s through fused kernels",
        ", ".join(patched_names),
    )
    return True, ""


__all__ = ["apply", "get_krea_stats", "get_stats"]
