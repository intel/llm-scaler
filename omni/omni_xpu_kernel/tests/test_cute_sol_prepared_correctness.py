"""Independent XPU reference for the internal prepared CUTE boundary.

This validates the carrier mainloop, not Kitchen's preparation/selection API.
Every numerical input, reference operation and error reduction stays on XPU.
"""
import math

import pytest
import torch


def prepared_op():
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    return torch.ops.omni_xpu_sol_attn.forward_cute_prepared


def reference(q, k, v, qs, ks, vs, routes, tail, scale):
    tokens = q.shape[1]
    query_blocks = torch.arange(tokens, device=q.device) // 64
    key_blocks = torch.arange(tokens, device=q.device) // 64
    mask = routes.bool().index_select(2, query_blocks).index_select(3, key_blocks)
    mask = mask & (ks > 0).unsqueeze(-2)
    scores = torch.matmul(q.permute(0, 2, 1, 3).float(), k.permute(0, 2, 3, 1).float())
    scores *= qs.unsqueeze(-1) * ks.unsqueeze(-2) * (scale * math.log2(math.e))
    scores.masked_fill_(~mask, float("-inf"))
    state = tail.index_select(2, query_blocks)
    tail_max = torch.where(state[..., 1] > 0, state[..., 0], float("-inf"))
    maximum = torch.maximum(scores.amax(-1), tail_max)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.exp2(scores - maximum.unsqueeze(-1))
    tail_factor = torch.where(state[..., 1] > 0, torch.exp2(state[..., 0] - maximum), 0)
    numerator = torch.matmul(weights, v.permute(0, 2, 1, 3).float())
    numerator += tail_factor.unsqueeze(-1) * state[..., 2:]
    denominator = weights.sum(-1) + tail_factor * state[..., 1]
    result = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    return (result * vs.unsqueeze(-2)).permute(0, 2, 1, 3).to(torch.bfloat16)


@pytest.mark.parametrize("tokens,layout", [(1, "bthd"), (63, "bhld"), (64, "bthd"),
                                          (65, "packed"), (257, "packed"), (513, "bhld")])
@pytest.mark.parametrize("route_mode", ["all", "sparse", "none"])
def test_prepared_carriers_and_tail(tokens, layout, route_mode):
    generator = torch.Generator(device="xpu").manual_seed(19071 + tokens)
    batch, heads, dim = 2, 3, 128
    shape = (batch, tokens, heads, dim)
    if layout == "packed":
        packed = torch.randint(-127, 128, (batch, tokens, 3, heads, dim),
                               generator=generator, dtype=torch.int8, device="xpu")
        q, k, v = packed.unbind(2)
    else:
        allocation = shape if layout == "bthd" else (batch, heads, tokens, dim)
        values = [torch.randint(-127, 128, allocation, generator=generator,
                                dtype=torch.int8, device="xpu") for _ in range(3)]
        q, k, v = values if layout == "bthd" else [value.permute(0, 2, 1, 3) for value in values]
    qs = torch.rand((batch, heads, tokens), generator=generator, device="xpu") * 0.025 + 0.001
    ks = torch.rand((batch, heads, tokens), generator=generator, device="xpu") * 0.025 + 0.001
    vs = torch.rand((batch, heads, dim), generator=generator, device="xpu") * 0.1 + 0.001
    blocks = (tokens + 63) // 64
    routes = (torch.rand((batch, heads, blocks, blocks), generator=generator, device="xpu") < 0.3).to(torch.uint8)
    if route_mode != "sparse":
        routes.fill_(int(route_mode == "all"))
    tail = torch.randn((batch, heads, blocks, 130), generator=generator, device="xpu")
    tail[..., 0] *= 5
    tail[..., 1] = tail[..., 1].abs() + 0.1
    # Mixed empty tails include fully empty rows when exact routing is off.
    tail[:, 0, ::2] = 0
    actual = prepared_op()(q, k, v, qs, ks, vs, routes, tail, dim ** -0.5)
    expected = reference(q, k, v, qs, ks, vs, routes, tail, dim ** -0.5)
    assert actual.shape == shape and actual.dtype == torch.bfloat16 and actual.is_contiguous()
    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    delta = actual.float() - expected.float()
    assert (delta.square().sum() / expected.float().square().sum().clamp_min(1)).sqrt().item() < 0.01


@pytest.mark.parametrize("tail_only", [False, True])
@pytest.mark.parametrize("channel_base", [-128, 0])
def test_exact_channel_mapping(tail_only, channel_base):
    q = torch.zeros((2, 64, 3, 128), dtype=torch.int8, device="xpu")
    k = torch.zeros_like(q)
    channels = torch.arange(channel_base, channel_base + 128, dtype=torch.int8, device="xpu")
    v = channels.expand_as(q).contiguous()
    qs = torch.ones((2, 3, 64), device="xpu")
    ks = torch.ones_like(qs)
    vs = torch.exp2(torch.arange(128, device="xpu").float() % 5 - 3).expand(2, 3, 128).contiguous()
    tail = torch.zeros((2, 3, 1, 130), device="xpu")
    if tail_only:
        tail[..., 1] = 1
        tail[..., 2:] = channels.float()
    routes = torch.full((2, 3, 1, 1), int(not tail_only), dtype=torch.uint8, device="xpu")
    actual = prepared_op()(q, k, v, qs, ks, vs, routes, tail, 0.125)
    expected = (channels.float() * vs).unsqueeze(1).expand_as(q).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("strong_source", ["first_block", "last_block", "tail"])
def test_weaker_tile_mass_survives_probability_quantization(strong_source):
    # A global-max U8 softmax would round the weak tile to zero. Sol's
    # per-tile scale must retain its contribution, including after a tail.
    tokens = 64 if strong_source == "tail" else 128
    q = torch.zeros((1, tokens, 1, 128), dtype=torch.int8, device="xpu")
    q[..., 0] = 1
    k = torch.zeros_like(q)
    v = torch.full_like(q, 127)
    if strong_source != "tail":
        strong = slice(0, 64) if strong_source == "first_block" else slice(64, 128)
        k[:, strong, :, 0] = 10
        v[:, strong] = 0
    qs = torch.ones((1, 1, tokens), device="xpu")
    ks = torch.ones_like(qs)
    vs = torch.ones((1, 1, 128), device="xpu")
    blocks = tokens // 64
    routes = torch.ones((1, 1, blocks, blocks), dtype=torch.uint8, device="xpu")
    tail = torch.zeros((1, 1, blocks, 130), device="xpu")
    if strong_source == "tail":
        tail[..., 0] = 10
        tail[..., 1] = 64
    actual = prepared_op()(q, k, v, qs, ks, vs, routes, tail, math.log(2))
    expected = reference(q, k, v, qs, ks, vs, routes, tail, math.log(2))
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
    assert actual.float().amin().item() > 0.1
