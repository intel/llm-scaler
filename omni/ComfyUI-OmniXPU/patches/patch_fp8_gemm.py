"""Patch comfy.ops fp8_linear and mixed_precision_ops to use
omni_xpu_kernel's oneDNN W8A16 FP8 GEMM when running on XPU.

Exactly mirrors the logic from comfyui_for_multi_arc.patch.
"""

import logging

import torch
import comfy.model_management

from .debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_fp8_linear = None
_logged_first_use = False


def _log_first(msg):
    global _logged_first_use
    if not _logged_first_use:
        _logged_first_use = True
        log.info("[OmniXPU] fp8_gemm first use: %s", msg)


def _dispatch_details(module):
    weight = getattr(module, "weight", None)
    layout = getattr(module, "layout_type", None)
    if layout is None:
        layout = getattr(weight, "_layout_cls", None)
    return {
        "quant_format": getattr(module, "quant_format", None),
        "layout": layout,
    }


def apply():
    global _omni_fp8_linear
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.linear_fp8 is None:
        return False, "omni_xpu_kernel linear_fp8 not available"
    _omni_fp8_linear = probe.linear_fp8

    import comfy.ops as comfy_ops

    # --- Patch fp8_linear module-level function ---
    if hasattr(comfy_ops, "fp8_linear"):
        _orig_fp8_linear = comfy_ops.fp8_linear

        def _patched_fp8_linear(self, input):
            log_debug_event(
                "dispatch",
                "fp8_linear",
                {"input": input},
                details=_dispatch_details(self),
                verbose_only=True,
            )
            dtype = self.weight.dtype
            if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                return None

            input_dtype = input.dtype
            input_shape = input.shape
            tensor_3d = input.ndim == 3
            if tensor_3d:
                input = input.reshape(-1, input_shape[2])
            if input.ndim != 2:
                return None

            lora_compute_dtype = comfy.model_management.lora_compute_dtype(input.device)
            w, bias, offload_stream = comfy_ops.cast_bias_weight(
                self, input, dtype=dtype, bias_dtype=input_dtype,
                offloadable=True, compute_dtype=lora_compute_dtype, want_requant=True,
            )

            # --- omni_xpu_kernel oneDNN FP8 GEMM (fast path for XPU) ---
            if _omni_fp8_linear is not None and input.is_xpu:
                _log_first(f"input={list(input.shape)} weight={list(w.shape)} dtype={dtype}")
                scale_weight = self.scale_weight if hasattr(self, 'scale_weight') and self.scale_weight is not None else torch.ones((), device=input.device, dtype=torch.float32)
                try:
                    o = _omni_fp8_linear(input, w, scale_weight, bias)
                    log_debug_event(
                        "kernel",
                        "fp8_linear",
                        {"input": input, "weight": w, "weight_scale": scale_weight, "bias": bias},
                        details={"backend": "omni_xpu", "format": dtype},
                    )
                    comfy_ops.uncast_bias_weight(self, w, bias, offload_stream)
                    if tensor_3d:
                        o = o.reshape((input_shape[0], input_shape[1], w.shape[0]))
                    return o
                except Exception as e:
                    _log_first(f"failed, falling back: {e}")

            # --- Original path: QuantizedTensor dispatch ---
            comfy_ops.uncast_bias_weight(self, w, bias, offload_stream)
            return _orig_fp8_linear(self, input)

        comfy_ops.fp8_linear = _patched_fp8_linear

    # --- Patch mixed_precision_ops Linear ---
    if hasattr(comfy_ops, "mixed_precision_ops"):
        _orig_mixed = comfy_ops.mixed_precision_ops
        QuantizedTensor = getattr(comfy_ops, "QuantizedTensor", None)

        def _patched_mixed(*args, **kwargs):
            klass = _orig_mixed(*args, **kwargs)

            _orig_fwd = klass.Linear.forward
            _orig_inner_fwd = klass.Linear._forward

            # -- Intercept 1: _forward(input, weight, bias) --
            # Called from forward_comfy_cast_weights after cast_bias_weight.
            def _mp_inner_forward(self, input, weight, bias):
                if (_omni_fp8_linear is not None and input.is_xpu and input.ndim == 2 and
                        hasattr(weight, 'dtype') and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)):
                    _log_first(f"input={list(input.shape)} weight={list(weight.shape)} dtype={weight.dtype}")
                    try:
                        scale_w = getattr(self, 'scale_weight', None)
                        if scale_w is None:
                            p = getattr(self.weight, 'params', None) or getattr(self.weight, '_layout_params', None)
                            scale_w = getattr(p, 'scale', None) if p else None
                        if scale_w is None:
                            scale_w = torch.ones((), device=input.device, dtype=torch.float32)
                        output = _omni_fp8_linear(input, weight, scale_w, bias)
                        log_debug_event(
                            "kernel",
                            "fp8_linear",
                            {"input": input, "weight": weight, "weight_scale": scale_w, "bias": bias},
                            details={"backend": "omni_xpu", "format": weight.dtype},
                        )
                        return output
                    except Exception as e:
                        _log_first(f"_forward failed, falling back: {e}")
                return _orig_inner_fwd(self, input, weight, bias)

            # -- Intercept 2: forward() --
            # Intercepts before comfy_kitchen QuantizedTensor dispatch.
            def _mp_forward(self, input, *fwd_args, **fwd_kwargs):
                log_debug_event(
                    "dispatch",
                    "mixed_precision.Linear",
                    {"input": input},
                    details=_dispatch_details(self),
                    verbose_only=True,
                )
                if (_omni_fp8_linear is not None and input.is_xpu and
                        getattr(self, 'quant_format', None) in ('float8_e4m3fn', 'float8_e5m2') and
                        len(self.weight_function) == 0 and len(self.bias_function) == 0):
                    input_shape = input.shape
                    input_2d = input.reshape(-1, input_shape[-1]) if input.ndim == 3 else input
                    if input_2d.ndim == 2:
                        try:
                            w = self.weight
                            fp8_dtype = torch.float8_e4m3fn if self.quant_format == 'float8_e4m3fn' else torch.float8_e5m2
                            if QuantizedTensor is not None and isinstance(w, QuantizedTensor):
                                w_fp8 = w._qdata
                                scale_w = getattr(w.params, 'scale', None)
                            else:
                                w_fp8 = w if w.dtype == fp8_dtype else w.view(fp8_dtype)
                                scale_w = getattr(self, 'scale_weight', None)
                            if scale_w is None:
                                scale_w = torch.ones((), device=input.device, dtype=torch.float32)
                            scale_w = comfy.model_management.cast_to_device(scale_w, input.device, torch.float32)
                            w_fp8 = comfy.model_management.cast_to_device(w_fp8, input.device, None)
                            bias = (comfy.model_management.cast_to_device(self.bias, input.device, input.dtype)
                                    if self.bias is not None else None)

                            _log_first(f"input={list(input_2d.shape)} weight={list(w_fp8.shape)} "
                                       f"dtype={w_fp8.dtype} format={self.quant_format}")

                            o = _omni_fp8_linear(input_2d, w_fp8, scale_w, bias)
                            log_debug_event(
                                "kernel",
                                "fp8_linear",
                                {"input": input_2d, "weight": w_fp8, "weight_scale": scale_w, "bias": bias},
                                details={"backend": "omni_xpu", "format": self.quant_format},
                            )
                            if input.ndim == 3:
                                o = o.reshape(input_shape[0], input_shape[1], -1)
                            return o
                        except Exception as e:
                            _log_first(f"forward failed, falling back: {e}")

                return _orig_fwd(self, input, *fwd_args, **fwd_kwargs)

            klass.Linear.forward = _mp_forward
            klass.Linear._forward = _mp_inner_forward
            return klass

        comfy_ops.mixed_precision_ops = _patched_mixed

    return True, None
