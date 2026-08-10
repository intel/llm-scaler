"""Route the LTX split-half cos/sin operator boundary to Omni XPU."""

from __future__ import annotations

import logging
import sys

from ..patches.debug import log_debug_event
from .errors import is_fatal_accelerator_error

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_ltx_split_rope_original__"
_omni_rotary = None
_routed_calls = 0
_fallback_calls = 0
_failed_contracts = set()
_logged_route_shapes = set()


def _contract(input_tensor, cos, sin):
    try:
        return (
            tuple(input_tensor.shape),
            tuple(input_tensor.stride()),
            str(input_tensor.dtype),
            str(input_tensor.device),
            tuple(cos.shape),
            tuple(cos.stride()),
            tuple(sin.shape),
            tuple(sin.stride()),
        )
    except (AttributeError, TypeError):
        return None


def get_stats():
    return {
        "routed": _routed_calls,
        "fallback": _fallback_calls,
        "quarantined_contracts": len(_failed_contracts),
    }


def apply():
    global _omni_rotary

    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe is None or probe.rotary is None:
        return False, "omni_xpu_kernel rotary not available"
    _omni_rotary = probe.rotary

    try:
        import omni_xpu_kernel as omni_package
    except ImportError:
        return False, "omni_xpu_kernel package not available"
    if getattr(omni_package, "__xpu_target__", "") != "bmg":
        return False, "direct LTX split-half RoPE is BMG-only"

    capability = getattr(
        _omni_rotary, "supports_ltx_split_rope_direct", None
    )
    if not callable(capability) or not capability():
        return False, "direct LTX split-half RoPE not available"

    try:
        import comfy.ldm.lightricks.model as ltx_model
    except ImportError:
        return False, "ComfyUI LTX model module not available"
    original = getattr(ltx_model, "apply_split_rotary_emb", None)
    if not callable(original):
        return False, "LTX split-half RoPE function not available"
    if hasattr(original, _PATCH_MARKER):
        return True, "already patched"

    def patched(input_tensor, cos, sin):
        global _fallback_calls, _routed_calls

        log_debug_event(
            "dispatch",
            "ltx_split_rope_direct",
            {"input": input_tensor, "cos": cos, "sin": sin},
            details={"backend": "bmg_sycl"},
            verbose_only=True,
        )
        contract = _contract(input_tensor, cos, sin)
        supported = getattr(
            _omni_rotary, "ltx_split_rope_direct_supported", None
        )
        if (
            contract is not None
            and contract not in _failed_contracts
            and callable(supported)
            and supported(input_tensor, cos, sin)
        ):
            try:
                output = _omni_rotary.apply_ltx_split_rope_direct(
                    input_tensor, cos, sin
                )
                _routed_calls += 1
                route_shape = (
                    tuple(input_tensor.shape),
                    tuple(cos.shape),
                    str(input_tensor.dtype),
                )
                if route_shape not in _logged_route_shapes:
                    log.info(
                        "[OmniXPU] rotary: LTX direct split-half route "
                        "input=%s cos=%s dtype=%s",
                        tuple(input_tensor.shape),
                        tuple(cos.shape),
                        input_tensor.dtype,
                    )
                    _logged_route_shapes.add(route_shape)
                log_debug_event(
                    "kernel",
                    "ltx_split_rope_direct",
                    {
                        "input": input_tensor,
                        "cos": cos,
                        "sin": sin,
                        "output": output,
                    },
                    details={"backend": "bmg_sycl"},
                )
                return output
            except RuntimeError as error:
                if is_fatal_accelerator_error(error):
                    raise
                _failed_contracts.add(contract)
                log.warning(
                    "[OmniXPU] rotary: direct LTX split-half route failed "
                    "for input=%s cos=%s; falling back to ComfyUI and "
                    "quarantining this contract (%s). Set "
                    "OMNIXPU_ROTARY=0 before startup to disable the adapter.",
                    tuple(input_tensor.shape),
                    tuple(cos.shape),
                    error,
                )

        _fallback_calls += 1
        return original(input_tensor, cos, sin)

    setattr(patched, _PATCH_MARKER, original)
    ltx_model.apply_split_rotary_emb = patched
    return True, ""


__all__ = ["apply", "get_stats"]
