"""Enable H3 INT8 CR projections while keeping FP32 transformer boundaries."""

from functools import wraps
import inspect

import torch

_MARKER = "_omnixpu_h3_te_int8_cr_original"


def _operations(ops, quant_config, dtype):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor

    # Other operators and quantization formats retain SDClipModel's policy.
    upstream = ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)

    class Linear(upstream.Linear):
        # The module owns the FP32->BF16->INT8->FP32 boundary. ComfyUI's
        # combined/linear_input_act shortcuts must call this module so they
        # don't bypass its conversion, FP32 activations or projection hooks.
        comfy_force_forward = True

        def _load_from_state_dict(self, *args, **kwargs):
            # Reapply the policy on reload too. The upstream loader still owns
            # file-level full precision and disabled-format decisions.
            self._full_precision_mm = False
            super()._load_from_state_dict(*args, **kwargs)
            if not (self.quant_format == "int8_tensorwise"
                    and isinstance(self.weight, QuantizedTensor)
                    and self.weight._params.convrot):
                self._full_precision_mm = True

        def _forward(self, input, weight, bias):
            quantized = (isinstance(weight, QuantizedTensor)
                         and weight._layout_cls == "TensorWiseINT8Layout")
            if (quantized and weight._params.convrot
                    and not weight._params.transposed
                    and input.device.type == "xpu"
                    and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
                    and input.ndim >= 2 and input.numel() > 0
                    and not input.requires_grad and not self.training
                    and not self._full_precision_mm
                    and self.can_use_quantized_matmul(
                        ops.get_disabled_quant_formats(input.device))
                    and not getattr(self, "comfy_force_cast_weights", False)
                    and not self.weight_function and not self.bias_function):
                # Existing half-input inference needs no FP32 bridge. Keep
                # the same device, training, patch and backend policy gates.
                if input.dtype != torch.float32:
                    return super()._forward(input, weight, bias)
                return ck.int8_linear(
                    input.to(torch.bfloat16), weight._qdata.contiguous(),
                    weight._params.scale, bias, out_dtype=input.dtype,
                    convrot=True,
                    convrot_groupsize=weight._params.convrot_groupsize,
                )
            # Keep full-precision semantics for non-XPU, training, patches and
            # explicitly disabled formats/backends; don't dispatch QTensor MM.
            if quantized:
                weight = weight.dequantize()
            return super()._forward(input, weight, bias)

    class Operations(upstream):
        pass

    Operations.Linear = Linear
    return Operations


def _wrap_init(original, ops, model_class):
    signature = inspect.signature(original)

    @wraps(original)
    def initialize(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments["model_options"]
        quant_config = options.get("quantization_metadata")
        if (bound.arguments["model_class"] is model_class
                and quant_config is not None
                and options.get("custom_operations") is None):
            options = dict(options)
            options["custom_operations"] = _operations(
                ops, quant_config, bound.arguments["dtype"])
            bound.arguments["model_options"] = options
        return original(*bound.args, **bound.kwargs)

    setattr(initialize, _MARKER, original)
    return initialize


def apply():
    try:
        import comfy.ops as ops
        from comfy.text_encoders import minimax

        original = minimax.MiniMaxH3ClipModel.__init__
        if hasattr(original, _MARKER):
            return True, "already patched"
        parameters = inspect.signature(original).parameters
        if not {"dtype", "model_options", "model_class"} <= parameters.keys():
            return False, "H3 text encoder no longer exposes dtype/model_options"
        if not callable(getattr(ops, "mixed_precision_ops", None)):
            return False, "ComfyUI mixed-precision operations are unavailable"
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        return False, f"H3 INT8 CR text encoder adapter is unavailable: {error}"

    minimax.MiniMaxH3ClipModel.__init__ = _wrap_init(
        original, ops, minimax.MiniMaxQwen3VL)
    return True, None
