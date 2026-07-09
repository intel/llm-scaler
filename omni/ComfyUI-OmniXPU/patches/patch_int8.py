"""INT8 linear acceleration for XPU via omni_xpu_kernel oneDNN s8 GEMM."""
import logging
import torch

log = logging.getLogger("ComfyUI-OmniXPU")

def apply():
    try:
        from omni_xpu_kernel import int8 as _omni_int8
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"
    try:
        from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
    except ImportError:
        return False, "comfy_kitchen not available"
    if not hasattr(torch.ops, "comfy_kitchen") or not hasattr(torch.ops.comfy_kitchen, "int8_linear"):
        return False, "comfy_kitchen::int8_linear not registered"

    @torch.library.impl("comfy_kitchen::int8_linear", "XPU")
    def _xpu_impl(x, weight, weight_scale, bias, output_dtype_code, convrot=False, convrot_groupsize=256):
        out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
        return _omni_int8.int8_linear(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize)

    log.info("[OmniXPU] INT8: registered XPU impl for comfy_kitchen::int8_linear")
    return True, ""
