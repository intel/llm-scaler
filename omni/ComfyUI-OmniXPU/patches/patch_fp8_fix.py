import logging

import torch

from .debug import log_debug_event, trace_patch

log = logging.getLogger("ComfyUI-OmniXPU")

_original_fn = None


def apply():
    global _original_fn
    try:
        import comfy.float as comfy_float
    except ImportError:
        return False, "comfy.float not found"

    if not hasattr(comfy_float, "manual_stochastic_round_to_float8"):
        return False, "manual_stochastic_round_to_float8 not found"

    _original_fn = comfy_float.manual_stochastic_round_to_float8

    @trace_patch(
        "fp8_neg_zero_fix",
        ("x", "dtype", "generator"),
        stage="dispatch",
        verbose_only=True,
    )
    def _patched(x, dtype, generator=None):
        result = _original_fn(x, dtype, generator=generator)
        # Fix: on XPU, -0.0 → float8 produces NaN
        if result.device.type == "xpu":
            is_neg_zero = (result == 0) & torch.signbit(result)
            if is_neg_zero.any():
                log_debug_event(
                    "kernel",
                    "fp8_neg_zero_fix",
                    {"input": x, "result": result},
                    details={"backend": "torch_xpu"},
                )
                result = torch.where(is_neg_zero, torch.zeros_like(result), result)
        return result

    comfy_float.manual_stochastic_round_to_float8 = _patched
    return True, None
