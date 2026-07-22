"""Fuse the validated PTL-H Z-Image RMSNorm/gate/residual boundary."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .debug import log_debug_event


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_zimage_rms_gate_original__"
_omni_norm = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}


def _torch_major_minor():
    try:
        components = torch.__version__.split("+", 1)[0].split(".")
        return int(components[0]), int(components[1])
    except (AttributeError, IndexError, ValueError):
        return None


def _route_input(block: Any, x: Any, adaln_input: Any, timestep_zero_index):
    if not isinstance(x, torch.Tensor) or not isinstance(
        adaln_input, torch.Tensor
    ):
        return False, "input_type"
    if timestep_zero_index is not None:
        return False, "timestep_slices"
    if x.device.type != "xpu" or adaln_input.device != x.device:
        return False, "device"
    if x.dtype != torch.bfloat16 or adaln_input.dtype != torch.bfloat16:
        return False, "dtype"
    if x.requires_grad:
        return False, "requires_grad"
    if x.ndim != 3 or not x.is_contiguous():
        return False, "layout"
    if x.shape[0] != 1 or x.shape[1] not in (64, 1024, 1088):
        return False, "shape"
    if x.shape[2] != 3840 or getattr(block, "dim", None) != 3840:
        return False, "hidden_size"
    if not getattr(block, "modulation", False):
        return False, "modulation"

    modulation = getattr(block, "adaLN_modulation", None)
    try:
        linear = modulation[0]
    except (IndexError, KeyError, TypeError):
        return False, "adaln_layout"
    if len(modulation) != 1 or getattr(linear, "in_features", None) != 256:
        return False, "zimage_modulation"
    return True, ""


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


def _run_fused(comfy_ops, norm, residual, gate, value):
    """Return (result, reason); an empty reason means the native route ran."""
    casted = (
        getattr(norm, "comfy_cast_weights", False)
        or len(getattr(norm, "weight_function", ())) > 0
        or len(getattr(norm, "bias_function", ())) > 0
    )
    if casted:
        weight, bias, offload_stream = comfy_ops.cast_bias_weight(
            norm, value, offloadable=True
        )
    else:
        weight = getattr(norm, "weight", None)
        bias = getattr(norm, "bias", None)
        offload_stream = None

    reason = ""
    if not isinstance(weight, torch.Tensor):
        reason = "weight"
    elif bias is not None:
        reason = "bias"
    elif not (
        weight.device == value.device == residual.device == gate.device
    ):
        reason = "weight_device"
    elif not (
        weight.dtype == value.dtype == residual.dtype == gate.dtype
        == torch.bfloat16
    ):
        reason = "operand_dtype"
    elif not (
        weight.is_contiguous()
        and value.is_contiguous()
        and residual.is_contiguous()
        and gate.is_contiguous()
    ):
        reason = "operand_layout"
    elif weight.shape != (3840,) or gate.shape != (1, 3840):
        reason = "modulation_shape"
    elif value.shape != residual.shape or value.shape[-1] != 3840:
        reason = "value_shape"

    if reason:
        if casted:
            comfy_ops.uncast_bias_weight(
                norm, weight, bias, offload_stream
            )
        return None, reason

    eps = norm.eps if norm.eps is not None else 1e-6
    try:
        log_debug_event(
            "kernel",
            "rms_norm_gate_residual",
            {
                "input": value,
                "weight": weight,
                "gate": gate,
                "residual": residual,
            },
            details={"backend": "esimd", "route": "ptl_zimage"},
        )
        result = _omni_norm.rms_norm_gate_residual(
            weight,
            value.reshape(-1, 3840),
            gate.reshape(-1),
            residual.reshape(-1, 3840),
            eps,
        ).reshape(residual.shape)
    finally:
        if casted:
            comfy_ops.uncast_bias_weight(
                norm, weight, bias, offload_stream
            )
    return result, ""


def apply():
    global _omni_norm

    try:
        import omni_xpu_kernel as omni_package
    except ImportError:
        return False, "omni_xpu_kernel not available"
    if getattr(omni_package, "__xpu_target__", "") != "ptl-h":
        return False, "validated only for PTL-H"
    if _torch_major_minor() != (2, 11):
        return False, "validated only for Torch 2.11"

    import sys

    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    candidate = getattr(probe, "norm", None)
    if candidate is None:
        return False, "omni_xpu_kernel norm not available"
    try:
        native = candidate._get_native()
    except (AttributeError, ImportError, RuntimeError) as exc:
        return False, f"native norm unavailable ({exc})"
    if not hasattr(native, "rms_norm_gate_residual"):
        return False, "native norm missing rms_norm_gate_residual"

    try:
        import comfy.ops as comfy_ops
        import comfy.ldm.lumina.model as lumina_model
    except ImportError as exc:
        return False, f"Lumina block unavailable ({exc})"

    block = getattr(lumina_model, "JointTransformerBlock", None)
    if block is None or not hasattr(block, "forward"):
        return False, "Lumina JointTransformerBlock.forward not found"
    if hasattr(block.forward, _PATCH_MARKER):
        return True, ""

    _omni_norm = candidate
    original_forward = block.forward

    def _forward(
        self,
        x,
        x_mask,
        freqs_cis,
        adaln_input=None,
        timestep_zero_index=None,
        transformer_options={},
    ):
        global _routed_calls

        eligible, reason = _route_input(
            self, x, adaln_input, timestep_zero_index
        )
        if not eligible:
            _record_fallback(reason)
            return original_forward(
                self,
                x,
                x_mask,
                freqs_cis,
                adaln_input=adaln_input,
                timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options,
            )

        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
            adaln_input
        ).chunk(4, dim=1)
        attention_value = lumina_model.clamp_fp16(
            self.attention(
                lumina_model.modulate(self.attention_norm1(x), scale_msa),
                x_mask,
                freqs_cis,
                transformer_options=transformer_options,
            )
        )
        fused, reason = _run_fused(
            comfy_ops,
            self.attention_norm2,
            x,
            gate_msa.tanh(),
            attention_value,
        )
        if fused is None:
            _record_fallback(f"attention_{reason}")
            x = x + lumina_model.apply_gate(
                gate_msa.unsqueeze(1).tanh(),
                self.attention_norm2(attention_value),
            )
        else:
            x = fused
            _routed_calls += 1

        ffn_value = lumina_model.clamp_fp16(
            self.feed_forward(
                lumina_model.modulate(self.ffn_norm1(x), scale_mlp)
            )
        )
        fused, reason = _run_fused(
            comfy_ops,
            self.ffn_norm2,
            x,
            gate_mlp.tanh(),
            ffn_value,
        )
        if fused is None:
            _record_fallback(f"ffn_{reason}")
            x = x + lumina_model.apply_gate(
                gate_mlp.unsqueeze(1).tanh(),
                self.ffn_norm2(ffn_value),
            )
        else:
            x = fused
            _routed_calls += 1
        return x

    setattr(_forward, _PATCH_MARKER, original_forward)
    block.forward = _forward
    log.info(
        "[OmniXPU] Z-Image PTL RMSNorm+gate+residual route installed"
    )
    return True, ""


__all__ = ["apply", "get_stats"]
