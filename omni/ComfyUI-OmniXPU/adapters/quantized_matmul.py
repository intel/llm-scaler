"""Expose available Kitchen INT8 formats to ComfyUI's model eligibility check."""
from functools import wraps
import inspect

import torch

_MARKER = '__omnixpu_quantized_formats_original__'
_INT8_OPERATIONS = (
    'int8_linear', 'quantize_int8_rowwise', 'quantize_int8_tensorwise',
    'dequantize_int8_simple_dtype', 'quantize_int8_convrot_weight',
    'dequantize_int8_convrot_weight_dtype',
)


def _wrap(original, default_device, registry, backend_error, packed_w4_available=lambda: False):
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
        if packed_w4_available():
            try:
                w4_native = registry.get_capable_backend('w4a8_int8_linear') == 'xpu'
            except backend_error:
                w4_native = False
            if w4_native:
                disabled.discard('asym_w4a8_int8')
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
        original, management.get_torch_device, registry, BackendError,
        _packed_w4_available)
    return True, ''


def _packed_w4_available():
    from omni_xpu_kernel import int8

    available = getattr(int8, 'supports_w4a8_linear_prequantized', None)
    return callable(available) and available()
