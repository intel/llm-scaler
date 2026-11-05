"""Share Qwen 2.1 INT8 activation preparation within one attention call.

Share ConvRot and rowwise quantization when the native APIs are available.
Keep ComfyUI's per-projection dynamic-weight context and never cache activations
between calls. Enable explicitly with OMNIXPU_QWEN_IMAGE21_QKV=1.
"""
from __future__ import annotations

from collections import Counter
import functools
import inspect
import textwrap

import torch

_stats = Counter()
_MARKER = "__omnixpu_shared_qkv_original__"


def get_stats():
    return dict(_stats)


@functools.lru_cache(maxsize=1)
def _default_projection_methods(factory):
    # Each ComfyUI factory call creates a fresh class. Its function code objects
    # are shared across calls, unlike the class and its closure configuration.
    linear = factory().Linear
    return {name: getattr(linear, name) for name in (
        "forward", "forward_comfy_cast_weights", "_forward")}


def _uses_default_projection(layer, methods):
    for name, expected in methods.items():
        method = getattr(layer, name, None)
        function = getattr(method, "__func__", None)
        if (name in vars(layer) or getattr(method, "__self__", None) is not layer
                or getattr(function, "__code__", None) is not expected.__code__
                or getattr(function, "__globals__", None) is not expected.__globals__):
            return False
    return True


def _valid_group_size(group):
    if type(group) is not int or group < 4:
        return False
    while group % 4 == 0:
        group //= 4
    return group == 1


def project_qkv(module, x):
    import comfy.model_management as mm
    import comfy.ops as ops
    from comfy_kitchen.backends import xpu as xpu_backend
    from comfy_kitchen.exceptions import NoCapableBackendError
    from comfy_kitchen.registry import registry
    from omni_xpu_kernel import int8

    layers = (module.to_q, module.to_k, module.to_v)

    def fallback(reason):
        _stats["fallback:" + reason] += 1
        return tuple(layer(x) for layer in layers)

    if (not isinstance(x, torch.Tensor) or x.device.type != "xpu"
            or x.dtype != torch.bfloat16 or x.ndim != 3
            or x.shape[-1] <= 0 or not x.is_contiguous()
            or x.requires_grad or torch.is_grad_enabled() or mm.in_training
            or torch.compiler.is_compiling()):
        return fallback("input")
    # Calling the native projections bypasses Module.__call__, including hooks
    # registered globally rather than on the individual projection modules.
    module_hooks = torch.nn.modules.module
    if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
        return fallback("module_contract")
    # Respect explicit overrides, disabled backends and priority before loading
    # native code or casting weights. Validate the actual cast operands below.
    try:
        if registry.get_capable_backend("int8_linear") != "xpu":
            return fallback("backend")
    except NoCapableBackendError:
        return fallback("backend")
    native = int8._get_native()
    if native is None or not all(hasattr(native, name) for name in (
            "rotate_convrot", "quantize_int8_rowwise_fused",
            "int8_linear_prequantized")):
        return fallback("native")
    methods = _default_projection_methods(ops.mixed_precision_ops)
    group_size = None
    # A fresh default class distinguishes inherited ComfyUI implementations from
    # class-level overrides as well as instance rebinding. Check the helper
    # methods too: the shared path bypasses all three of them.
    for layer in layers:
        weight = getattr(layer, "weight", None)
        params = getattr(weight, "_params", None)
        group = getattr(params, "convrot_groupsize", None)
        if (getattr(layer, "quant_format", None) != "int8_tensorwise"
                or getattr(layer, "layout_type", None) != "TensorWiseINT8Layout"
                or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout"
                or len(getattr(weight, "shape", ())) != 2
                or weight.shape[1] != x.shape[-1]
                or getattr(layer, "bias", None) is not None
                or getattr(layer, "_full_precision_mm", False)
                or getattr(layer, "comfy_force_cast_weights", False)
                or getattr(layer, "weight_function", ())
                or getattr(layer, "bias_function", ())
                or getattr(layer, "pre_quant_scale", None) is not None
                or getattr(layer, "input_scale", None) is not None
                or layer._forward_hooks or layer._forward_pre_hooks
                or not _uses_default_projection(layer, methods)
                or not getattr(params, "convrot", False)
                or not _valid_group_size(group) or x.shape[-1] % group != 0
                or (group_size is not None and group != group_size)
                or getattr(params, "transposed", False)
                or getattr(params, "orig_dtype", None) != x.dtype):
            return fallback("module_contract")
        group_size = group

    prepared = None
    outputs = []
    for layer in layers:
        ops.run_every_op()
        # Match weight_only_quant in MixedPrecisionOps.Linear. Backend constraints
        # must see these cast operands, not the CPU/offloaded weight metadata.
        with ops.CastBiasWeightContext(
                layer, input=None, dtype=layer.weight.dtype, device=x.device,
                bias_dtype=x.dtype, offloadable=True, compute_dtype=x.dtype,
                want_requant=True) as (weight, bias):
            data, ws = weight._qdata.contiguous(), weight._params.scale
            if data.device != x.device or ws.device != x.device or bias is not None:
                raise RuntimeError("Qwen shared QKV cast violated the admitted contract")
            kwargs = dict(x=x, weight=data, weight_scale=ws, bias=bias,
                          out_dtype=x.dtype, convrot=True,
                          convrot_groupsize=group_size, input_act=None,
                          input_act_weight=None, input_act_eps=0.0,
                          residual=None, residual_scale=None)
            implementation = registry.get_implementation("int8_linear", kwargs=kwargs)
            if implementation is not getattr(xpu_backend, "int8_linear", None):
                # Constraints may choose another backend for a particular
                # projection. Execute that choice with the original activation.
                outputs.append(implementation(**kwargs))
                _stats["fallback:backend_projection"] += 1
                continue
            if prepared is None:
                rotated = int8.rotate_convrot(x, group_size)
                prepared = int8.quantize_int8_rowwise(rotated)
                del rotated
            outputs.append(int8.int8_linear_prequantized(
                *prepared, data, ws, out_dtype=x.dtype))
    if prepared is not None:
        _stats["shared_calls"] += 1
        _stats["shared_rows:" + str(x.shape[1])] += 1
    return tuple(outputs)


def apply():
    import comfy.ldm.qwen_image21.model as qwen
    original = qwen.Attention.forward
    if hasattr(original, _MARKER):
        return True, "already applied"
    old = """q = self.to_q(x).view(B, N, self.heads, -1)
    k = self.to_k(x).view(B, N, self.heads, -1)
    v = self.to_v(x).view(B, N, self.heads, -1)"""
    new = """q, k, v = _omnixpu_project_qkv(self, x)
    q = q.view(B, N, self.heads, -1)
    k = k.view(B, N, self.heads, -1)
    v = v.view(B, N, self.heads, -1)"""
    source = textwrap.dedent(inspect.getsource(original))
    if original.__closure__ or source.count(old) != 1:
        raise RuntimeError("unsupported Qwen Image 2.1 attention source")
    namespace = dict(original.__globals__, _omnixpu_project_qkv=project_qkv)
    exec(compile(source.replace(old, new), inspect.getfile(original), "exec"), namespace)
    replacement = functools.update_wrapper(namespace[original.__name__], original)
    setattr(replacement, _MARKER, original)
    qwen.Attention.forward = replacement
    return True, ""
