"""Dispatcher boundaries for native inference APIs used inside torch.compile.

Eager callers retain the original Python/native dispatch. Compiled callers see
an opaque operator with an explicit FakeTensor output contract; Dynamo and AOT
must never execute a pybind kernel on a FakeTensor. These inference operators
do not implement backward. Mutable boundaries name their modified arguments and
return None; public Python wrappers return the original inputs when required.
"""

import torch


_OPERATORS = {}
_EFFECT_HANDLES = []


def compile_op(name, fake, *, mutates_args=(), schema=None, ordered=False):
    """Register an inference API; its compile branch calls torch.ops directly.

    Keep each public function's own code object. A shared dispatch closure would
    make unrelated APIs share Dynamo's per-code recompile limit.
    """
    def decorate(function):
        operator = torch.library.custom_op(
            "omni_xpu::" + name, function, mutates_args=mutates_args, schema=schema
        )
        operator.register_fake(fake)
        if ordered:
            # These APIs expose runtime cache state. Effect tokens preserve
            # call order and calls whose tensor result is otherwise unused.
            from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
            _EFFECT_HANDLES.append(_register_effectful_op(getattr(torch.ops.omni_xpu, name).default, _EffectType.ORDERED))
        _OPERATORS[name] = operator
        return function
    return decorate


def fake_rms_norm(weight, input, eps=1e-6):
    return input.new_empty(input.shape)


def fake_layer_norm(input, weight=None, bias=None, eps=1e-5):
    return input.new_empty(input.shape)


def fake_rotate_convrot(x, group_size=256):
    return x.new_empty(x.shape)


def fake_silu_mul(x1, x2):
    return x1.new_empty(x1.shape)


def fake_rowwise(x, stochastic_rounding=0):
    # Stochastic TensorIterator output retains dense input layouts; the fused
    # deterministic path is contiguous. Both have [..., 1] FP32 scales.
    quantized = (torch.empty_like(x, dtype=torch.int8) if stochastic_rounding > 0
                 else x.new_empty(x.shape, dtype=torch.int8))
    return (quantized,
            x.new_empty((*x.shape[:-1], 1), dtype=torch.float32))


def fake_silu_mul_rowwise(x1, x2):
    return fake_rowwise(x1)


def fake_swiglu_rowwise(input):
    return (input.new_empty((*input.shape[:-1], input.shape[-1] // 2), dtype=torch.int8),
            input.new_empty((*input.shape[:-1], 1), dtype=torch.float32))


def fake_gelu_rowwise(input):
    return fake_rowwise(input)


def fake_int8_linear(x, weight, weight_scale, bias=None, out_dtype=None,
                     convrot=False, convrot_groupsize=256, input_act=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]),
                       dtype=x.dtype if out_dtype is None else out_dtype)


def fake_prequantized(x_int8, x_scale, weight, weight_scale, bias=None,
                      out_dtype=torch.bfloat16):
    return x_int8.new_empty((*x_int8.shape[:-1], weight.shape[0]), dtype=out_dtype)


def fake_shared_input(x, weight1, weight_scale1, weight2, weight_scale2,
                      bias1=None, bias2=None, out_dtype=None, convrot=False,
                      convrot_groupsize=256):
    return (fake_int8_linear(x, weight1, weight_scale1, bias1, out_dtype),
            fake_int8_linear(x, weight2, weight_scale2, bias2, out_dtype))
