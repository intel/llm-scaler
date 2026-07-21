"""INT8 linear acceleration for XPU via omni_xpu_kernel oneDNN s8 GEMM."""
import logging
from typing import Any

import torch

from .debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_native_target = ""
_native_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}
_first_shape_logged = False


def _tensor_metadata(tensor):
    if tensor is None:
        return None
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
    }


def _log_first_shape(
    x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
):
    global _first_shape_logged
    if _first_shape_logged:
        return
    _first_shape_logged = True
    k = x.shape[-1] if x.ndim else 0
    m = x.numel() // k if k else 0
    n = weight.shape[0] if weight.ndim else 0
    log.info(
        "[OmniXPU] INT8 call shape: M=%d K=%d N=%d x=%s weight=%s "
        "weight_scale=%s bias=%s out_dtype=%s convrot=%s convrot_groupsize=%d",
        m,
        k,
        n,
        _tensor_metadata(x),
        _tensor_metadata(weight),
        _tensor_metadata(weight_scale),
        _tensor_metadata(bias),
        out_dtype,
        bool(convrot),
        int(convrot_groupsize),
    )


def _native_guard_reason() -> str:
    # The current PTL-H AOT build's dynamic rowwise quantizer terminates the
    # process even for a minimal [1, 256] BF16 input.  int8_linear always calls
    # that kernel, so no M/K/N subset is safe to admit.  Keep the oneDNN
    # prequantized APIs available to callers that provide their own known-good
    # activation quantization, while this dispatcher uses Comfy Kitchen eager.
    if _native_target == "ptl-h":
        return "ptl_h_dynamic_rowwise_quantize"
    return ""


def _record_fallback(reason: str) -> None:
    global _fallback_calls
    _fallback_calls += 1
    _fallback_reasons[reason] = _fallback_reasons.get(reason, 0) + 1


def get_stats() -> dict[str, Any]:
    return {
        "native": _native_calls,
        "fallback": _fallback_calls,
        "reasons": dict(_fallback_reasons),
    }


def apply():
    global _native_target

    try:
        import omni_xpu_kernel as _omni_package
        from omni_xpu_kernel import int8 as _omni_int8
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"
    try:
        from comfy_kitchen.backends.eager.quantization import (
            DTYPE_CODE_TO_DTYPE,
            int8_linear as _eager_int8_linear,
        )
    except ImportError:
        return False, "comfy_kitchen not available"
    if not hasattr(torch.ops, "comfy_kitchen") or not hasattr(
        torch.ops.comfy_kitchen, "int8_linear"
    ):
        return False, "comfy_kitchen::int8_linear not registered"

    _native_target = getattr(_omni_package, "__xpu_target__", "")

    @torch.library.impl("comfy_kitchen::int8_linear", "XPU")
    def _xpu_impl(
        x,
        weight,
        weight_scale,
        bias,
        output_dtype_code,
        convrot=False,
        convrot_groupsize=256,
    ):
        global _native_calls
        out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
        _log_first_shape(
            x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
        )
        reason = _native_guard_reason()
        if reason:
            _record_fallback(reason)
            if _fallback_reasons[reason] <= 3:
                log.warning(
                    "[OmniXPU] INT8 native fallback: reason=%s target=%s "
                    "shape=(%s, %s, %s)",
                    reason,
                    _native_target,
                    x.numel() // x.shape[-1],
                    x.shape[-1],
                    weight.shape[0],
                )
            log_debug_event(
                "dispatch",
                "int8_linear",
                {
                    "x": x,
                    "weight": weight,
                    "weight_scale": weight_scale,
                    "bias": bias,
                },
                details={"route": "comfy_kitchen_eager", "reason": reason},
                verbose_only=True,
            )
            return _eager_int8_linear(
                x,
                weight,
                weight_scale,
                bias,
                out_dtype,
                convrot,
                convrot_groupsize,
            )
        _native_calls += 1
        log_debug_event(
            "kernel",
            "int8_linear",
            {
                "x": x,
                "weight": weight,
                "weight_scale": weight_scale,
                "bias": bias,
            },
            details={"backend": "omni_xpu"},
        )
        return _omni_int8.int8_linear(
            x,
            weight,
            weight_scale,
            bias,
            out_dtype,
            convrot,
            convrot_groupsize,
        )

    log.info("[OmniXPU] INT8: registered XPU impl for comfy_kitchen::int8_linear")
    return True, ""


__all__ = ["apply", "get_stats"]
