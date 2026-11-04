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


def project_qkv(module, x):
    import comfy.model_management as mm
    import comfy.ops as ops
    from omni_xpu_kernel import int8

    layers = (module.to_q, module.to_k, module.to_v)

    def fallback(reason):
        _stats["fallback:" + reason] += 1
        return tuple(layer(x) for layer in layers)

    if (not isinstance(x, torch.Tensor) or x.device.type != "xpu"
            or x.dtype != torch.bfloat16 or x.ndim != 3
            or x.shape[-1] <= 0 or x.shape[-1] % 256 != 0 or not x.is_contiguous()
            or x.requires_grad or torch.is_grad_enabled() or mm.in_training
            or torch.compiler.is_compiling()):
        return fallback("input")
    native = int8._get_native()
    if native is None or not all(hasattr(native, name) for name in (
            "rotate_convrot", "quantize_int8_rowwise_fused",
            "int8_linear_prequantized")):
        return fallback("native")
    # Validate before preparing anything or entering a weight context. Original
    # module calls retain instance forward overrides, hooks, smoothing and training.
    for layer in layers:
        weight = getattr(layer, "weight", None)
        params = getattr(weight, "_params", None)
        if (getattr(layer, "quant_format", None) != "int8_tensorwise"
                or getattr(layer, "layout_type", None) != "TensorWiseINT8Layout"
                or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout"
                or len(weight.shape) != 2 or weight.shape[1] != x.shape[-1]
                or getattr(layer, "bias", None) is not None
                or getattr(layer, "_full_precision_mm", False)
                or getattr(layer, "comfy_force_cast_weights", False)
                or getattr(layer, "weight_function", ())
                or getattr(layer, "bias_function", ())
                or getattr(layer, "pre_quant_scale", None) is not None
                or getattr(layer, "input_scale", None) is not None
                or layer._forward_hooks or layer._forward_pre_hooks
                or getattr(layer.forward, "__func__", None) is not type(layer).forward
                or not getattr(params, "convrot", False)
                or getattr(params, "convrot_groupsize", None) != 256
                or getattr(params, "transposed", False)
                or getattr(params, "orig_dtype", None) != x.dtype):
            return fallback("module_contract")

    rotated = int8.rotate_convrot(x, 256)
    codes, scale = int8.quantize_int8_rowwise(rotated)
    del rotated
    outputs = []
    for layer in layers:
        ops.run_every_op()
        # Match weight_only_quant in MixedPrecisionOps.Linear. Each projection
        # releases its own stream/offload lifetime before the next one starts.
        with ops.CastBiasWeightContext(
                layer, input=None, dtype=layer.weight.dtype, device=x.device,
                bias_dtype=x.dtype, offloadable=True, compute_dtype=x.dtype,
                want_requant=True) as (weight, bias):
            data, ws = weight._qdata, weight._params.scale
            if data.device != x.device or ws.device != x.device or bias is not None:
                raise RuntimeError("Qwen shared QKV cast violated the admitted contract")
            outputs.append(int8.int8_linear_prequantized(
                codes, scale, data, ws, out_dtype=x.dtype))
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
