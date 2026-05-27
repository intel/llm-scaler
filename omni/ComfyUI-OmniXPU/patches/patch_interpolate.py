import functools
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-OmniXPU")

_original_interpolate = None


def apply():
    global _original_interpolate
    _original_interpolate = F.interpolate

    @functools.wraps(_original_interpolate)
    def _xpu_interpolate(input_tensor, *args, **kwargs):
        if input_tensor.device.type == "xpu":
            dev = input_tensor.device
            result = _original_interpolate(input_tensor.to("cpu"), *args, **kwargs)
            return result.to(dev)
        return _original_interpolate(input_tensor, *args, **kwargs)

    F.interpolate = _xpu_interpolate
    return True, None
