"""Storage-free metadata for the allocating and mutable native APIs.

Only metadata and Torch operations are used here. Native capability checks and
kernel execution stay in the runtime implementation, never in a fake kernel.
"""

import torch


def unchanged(input, *args, **kwargs):
    return input.new_empty(input.shape)


def void(*args, **kwargs):
    return None


def norm_projection(input, norm_weight, proj_weight, eps=1e-6):
    return input.new_empty((*input.shape[:-1], proj_weight.shape[0]))


def norm_segmented(weight, input, scale, shift, starts, stops, rows, eps=1e-6):
    return input.new_empty(input.shape)


def norm_gate(weight, input, gate, residual, eps=1e-6):
    return input.new_empty(input.shape)


def group_norm(input, num_groups, weight, bias, eps=1e-6):
    return input.new_empty(input.shape)


def fp8_quantize(x, scale, out_dtype=torch.float8_e4m3fn):
    return x.new_empty(x.shape, dtype=out_dtype)


def fp8_dequantize(x, scale, out_dtype=torch.bfloat16):
    return x.new_empty(x.shape, dtype=out_dtype)


def fp8_round(x, rng, out_dtype=torch.float8_e4m3fn):
    return x.new_empty(x.shape, dtype=out_dtype)


def linear_fp8(x, weight, scales, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def tensorwise(x, stochastic_rounding=0):
    q = (torch.empty_like(x, dtype=torch.int8) if stochastic_rounding > 0
         else x.new_empty(x.shape, dtype=torch.int8))
    return q, x.new_empty((), dtype=torch.float32)


def tensorwise_scaled(x, scale, stochastic_rounding=0):
    return tensorwise(x, stochastic_rounding)[0]


def int8_dequantize(q, scale):
    return q.new_empty(torch.broadcast_shapes(q.shape, scale.shape), dtype=torch.float32)


def int8_dequantize_dtype(q, scale, out_dtype=torch.bfloat16):
    return q.new_empty(torch.broadcast_shapes(q.shape, scale.shape), dtype=out_dtype)


def int8_mm(a, b):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.int32)


def convrot_quantize(weight, group_size=256, stochastic_rounding=0):
    return (weight.new_empty(weight.shape, dtype=torch.int8),
            weight.new_empty((*weight.shape[:-1], 1), dtype=torch.float32))


def convrot_dequantize(q, scale, group_size=256):
    return q.new_empty(q.shape, dtype=torch.float32)


def gguf(input, dtype, block_bytes, block_elements):
    return input.new_empty((input.numel() // block_bytes * block_elements,), dtype=dtype)


def gguf_q4_0(input, dtype=torch.float16):
    return gguf(input, dtype, 18, 32)


def gguf_q4_1(input, dtype=torch.float16):
    return gguf(input, dtype, 20, 32)


def gguf_q8_0(input, dtype=torch.float16):
    return gguf(input, dtype, 34, 32)


def gguf_q4_k(input, dtype=torch.float16):
    return gguf(input, dtype, 144, 256)


def gguf_q6_k(input, dtype=torch.float16):
    return gguf(input, dtype, 210, 256)


def gguf_batch(inputs, formats, dtype=torch.float16):
    functions = {"q4_0": gguf_q4_0, "q4_1": gguf_q4_1,
                 "q8_0": gguf_q8_0, "q4_k": gguf_q4_k, "q6_k": gguf_q6_k}
    return [functions[format](input, dtype) for input, format in zip(inputs, formats)]


def svdq_dequantize(packed, scales, out_dtype=torch.bfloat16):
    return packed.new_empty((packed.shape[0], packed.shape[1] * 2), dtype=out_dtype)


def svdq_unpack(packed, signed=True):
    return packed.new_empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int8)


def svdq_quantize(input, group_size=64):
    rows, width = input.shape
    return (input.new_empty((rows, width // 2), dtype=torch.uint8),
            input.new_empty((width // group_size, rows)))


def svdq_gemm(act, packed, scales):
    return act.new_empty((act.shape[0], packed.shape[0]))


def svdq_smooth(x, factor):
    return x.new_empty(x.shape, dtype=torch.float16)


def cat_pad(prefix, input, spatial_pad=1):
    b, c, t, h, w = input.shape
    return input.new_empty((b, c, t + prefix.shape[2],
                            h + 2 * spatial_pad, w + 2 * spatial_pad))


def kitchen_rope1(x, freqs_cis, *, split_half=False):
    # The fast kernel consumes contiguous 4D input/frequencies and returns a
    # contiguous tensor. The fallback below mirrors only the native Torch
    # expressions' metadata, including non-contiguous broadcast layouts.
    if x.ndim == 4 and x.is_contiguous() and freqs_cis.is_contiguous():
        return x.new_empty(x.shape)
    if split_half:
        split = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs_cis.dtype)
        output = freqs_cis[..., 0] * split[..., 0] + freqs_cis[..., 1] * split[..., 1]
        return output.movedim(-1, -2).reshape(x.shape).to(x.dtype)
    paired = x.to(freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    freqs = freqs_cis
    if paired.shape[2] != 1 and freqs.shape[2] != 1 and paired.shape[2] != freqs.shape[2]:
        freqs = freqs[:, :, :paired.shape[2]]
    output = freqs[..., 0] * paired[..., 0]
    return output.reshape(x.shape).to(x.dtype)


def kitchen_rope(q, k, freqs_cis):
    return kitchen_rope1(q, freqs_cis), kitchen_rope1(k, freqs_cis)


def kitchen_split1(x, freqs_cis):
    return kitchen_rope1(x, freqs_cis, split_half=True)


def kitchen_split(q, k, freqs_cis):
    return kitchen_split1(q, freqs_cis), kitchen_split1(k, freqs_cis)


def rms_rope1(x, freqs_cis, scale, epsilon, split_half, rot_dim):
    return x.new_empty(x.shape)


def rms_rope(q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, rot_dim):
    return q.new_empty(q.shape), k.new_empty(k.shape)
