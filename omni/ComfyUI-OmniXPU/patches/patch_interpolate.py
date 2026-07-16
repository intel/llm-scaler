import functools
import logging

import torch.nn.functional as F

from .debug import log_debug_event, trace_patch

log = logging.getLogger("ComfyUI-OmniXPU")

_original_interpolate = None


def apply():
    global _original_interpolate
    _original_interpolate = F.interpolate

    @functools.wraps(_original_interpolate)
    @trace_patch(
        "interpolate",
        ("input",),
        stage="dispatch",
        verbose_only=True,
    )
    def _xpu_interpolate(input_tensor, *args, **kwargs):
        if input_tensor.device.type == "xpu":
            log_debug_event(
                "kernel",
                "interpolate_fix",
                {"input": input_tensor},
                details={"backend": "cpu_fallback"},
            )
            dev = input_tensor.device
            result = _original_interpolate(input_tensor.to("cpu"), *args, **kwargs)
            return result.to(dev)
        return _original_interpolate(input_tensor, *args, **kwargs)

    F.interpolate = _xpu_interpolate
    return True, None
