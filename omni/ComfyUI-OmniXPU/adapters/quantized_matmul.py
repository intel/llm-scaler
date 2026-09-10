"""Expose Kitchen's native INT8 format to ComfyUI's model eligibility check."""
from functools import wraps
import inspect

import torch

_MARKER = '__omnixpu_quantized_formats_original__'
_INT8_OPERATIONS = (
    'int8_linear', 'quantize_int8_rowwise', 'quantize_int8_tensorwise',
    'dequantize_int8_simple_dtype', 'quantize_int8_convrot_weight',
    'dequantize_int8_convrot_weight_dtype',
)


def _wrap(original, default_device, registry, backend_error):
    @wraps(original)
    def disabled_formats(device=None):
        disabled = set(original(device))
        target = default_device() if device is None else torch.device(device)
        if target.type != 'xpu' or not registry.is_available('xpu'):
            return disabled
        try:
            native = all(registry.get_capable_backend(name) == 'xpu'
                         for name in _INT8_OPERATIONS)
        except backend_error:
            native = False
        if native:
            # This format includes scalar/per-channel weights and ConvRot.
            # Kernel dispatch and input constraints remain owned by Kitchen.
            disabled.discard('int8_tensorwise')
        return disabled

    setattr(disabled_formats, _MARKER, original)
    return disabled_formats


def apply():
    try:
        import comfy.ops as ops
        import comfy.model_management as management
        from comfy_kitchen.registry import registry
        from comfy_kitchen.exceptions import BackendError
        original = ops.get_disabled_quant_formats
        if hasattr(original, _MARKER):
            return True, 'already patched'
        parameters = list(inspect.signature(original).parameters.values())
        if (len(parameters) != 1 or parameters[0].name != 'device'
                or parameters[0].default is not None
                or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return False, 'unsupported ComfyUI quantized-format eligibility signature'
        if not registry.is_available('xpu'):
            return False, 'Kitchen XPU backend is unavailable or disabled'
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        return False, f'quantized-format eligibility is unavailable: {error}'
    ops.get_disabled_quant_formats = _wrap(
        original, management.get_torch_device, registry, BackendError)
    return True, ''
