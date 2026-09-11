"""SLA/VSA controls and chunked composition against independent quantized XPU math.

These bounded operator fixtures need no model weights. They do not establish
the output quality or performance of a trained SLA or FastH3-VSA workflow.
"""
import math

import pytest
import torch

from test_cute_sol_chunked_correctness import norm_rope_reference


def runtime():
    from omni_xpu_kernel.cute import sol_attn_v2
    from comfy_kitchen.backends.eager.sol_attn import sol_attn as reference

    assert torch.xpu.is_available(), "these tests require the admitted XPU environment"
    assert sol_attn_v2.is_available(), "complete native Sol API is required"
    return sol_attn_v2, reference


def controls(mode, tokens, heads, generator, ratio):
    kwargs = dict(topk_ratio=ratio, token_aug=0, sink_blocks=[0, 1], sink_q=[0, 1])
    live = torch.ones(tokens, device="xpu", dtype=torch.bool)
    if mode == "vsa":
        lengths = torch.randint(
            17, 65, ((tokens + 63) // 64,), device="xpu", generator=generator,
            dtype=torch.int32,
        )
        lengths[-1] = min(int(lengths[-1]), (tokens - 1) % 64 + 1)
        positions = torch.arange(tokens, device="xpu")
        live = positions % 64 < lengths[positions // 64]
        gate = torch.randn((1, tokens, heads, 128), device="xpu",
                           dtype=torch.bfloat16, generator=generator) * 0.25
        kwargs.update(tail=False, block_len=lengths, coarse_gate=gate)
    else:
        kwargs["tail"] = True
    return kwargs, live


def quantized_reference(q, k, v, kwargs, kmean=None, vscale=None):
    from test_cute_sol_carriers_correctness import reference as carriers
    from test_cute_sol_producer_correctness import producer_reference
    from test_cute_sol_pooled_correctness import route_reference
    from test_cute_sol_prepared_correctness import reference as attention

    tokens = q.shape[1]
    lengths = kwargs.get("block_len", torch.empty(0, device="xpu", dtype=torch.int32))
    scale, tau = 128**-0.5, 1.0
    if kmean is None:
        c = carriers(q, k, v, scale, tau, lengths)
    else:
        c = producer_reference(q, k, v, kmean, vscale, scale, tau, lengths)
    scores = (c["cq"].float() @ c["ck"].float().transpose(-1, -2)) * (
        c["cqs"].unsqueeze(-1) * (scale * math.log2(math.e))) * c["cks"].unsqueeze(-2)
    sinks = (*kwargs["sink_blocks"], *kwargs["sink_q"])
    selectable = scores.shape[-1] - (sinks[1] - sinks[0])
    count = max(0, min(selectable - 1, max(1, round(kwargs["topk_ratio"] * selectable))))
    routes, state, *_ = route_reference(scores, c["threshold"], c["vsum"], c["vs"],
        lengths, tokens, sinks, count, kwargs["tail"], False)
    expected = attention(*(c[name] for name in ("q", "k", "v", "qs", "ks", "vs")),
                         routes, state, scale)
    if "coarse_gate" in kwargs:
        ids = torch.arange(tokens, device="xpu") // 64
        live = (tokens - 64 * torch.arange(scores.shape[-1], device="xpu")).clamp_max(64)
        live = torch.minimum(live, lengths.clamp_min(1))
        coarse = torch.softmax(c["qmean"] @ c["kblocks"].transpose(-1, -2) * scale, -1) @ (
            c["vsum"].float() / live.reshape(1, 1, -1, 1))
        expected = (expected.float() + kwargs["coarse_gate"].float() *
            coarse.index_select(2, ids).permute(0, 2, 1, 3)).to(q.dtype)
    return expected


def record_full_precision_difference(record_property, actual, expected, live):
    # Quantized centroid top-k can select a different block near a rank boundary.
    # Preserve the full-precision discrepancy as a diagnostic; correctness below
    # uses independent math for the actual INT8 carrier/routing contract.
    a, b = actual[:, live].float().flatten(), expected[:, live].float().flatten()
    record_property("full_precision_relative_l2", float(
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-8)))


def close_on_live_rows(actual, expected, live, relative_tolerance=0.06):
    assert actual.device.type == expected.device.type == "xpu"
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.is_contiguous() and bool(torch.isfinite(actual).all())
    a, b = actual[:, live].float().flatten(), expected[:, live].float().flatten()
    # The caller declares its oracle: independent quantized math for operator
    # tests, full-precision math or full/chunked composition for node tests.
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
    relative_l2 = torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-8)
    assert float(cosine) > 0.995, float(cosine)
    assert float(relative_l2) < relative_tolerance, float(relative_l2)


@pytest.mark.parametrize("mode", ["sla", "vsa"])
@pytest.mark.parametrize("tokens", [257, 577])
@pytest.mark.parametrize("ratio", [0.1, 0.4])
def test_sparse_modes_match_independent_quantized_math_on_xpu(mode, tokens, ratio, record_property):
    native, reference = runtime()
    generator = torch.Generator(device="xpu").manual_seed(40117 + tokens)
    q, k, v = torch.randn(
        (1, tokens, 3, 3, 128), device="xpu", dtype=torch.bfloat16,
        generator=generator,
    ).unbind(2)
    q, k = q * 0.5, k * 0.5
    kwargs, live = controls(mode, tokens, 3, generator, ratio)
    if mode == "vsa":
        # Padding must not become keys, block means or coarse-branch values.
        for tensor in (q, k, v):
            tensor[:, ~live] = 64
    before = [tensor.clone() for tensor in (q, k, v)]
    actual = native.sol_attn(q, k, v, **kwargs)
    record_full_precision_difference(record_property, actual, reference(q, k, v, **kwargs), live)
    expected = quantized_reference(q, k, v, kwargs)
    close_on_live_rows(actual, expected, live, relative_tolerance=0.01)
    for tensor, snapshot in zip((q, k, v), before):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["sla", "vsa"])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("ratio", [0.1, 0.4])
def test_chunked_sparse_modes_match_xpu_reference_and_preserve_inputs(mode, replay, ratio, record_property):
    native, reference = runtime()
    tokens, heads = 577, 3
    generator = torch.Generator(device="xpu").manual_seed(40231)
    packed = torch.randn((tokens, 3 * heads * 128), device="xpu",
                         dtype=torch.bfloat16, generator=generator) * 0.5
    angles = torch.randn((1, tokens, 1, 48), device="xpu", generator=generator)
    co, si = angles.cos(), angles.sin()
    freqs = torch.stack((co, -si, si, co), -1).reshape(1, tokens, 1, 48, 2, 2).to(torch.bfloat16)
    weights = tuple(torch.ones(128, device="xpu", dtype=torch.bfloat16) for _ in range(2))
    kwargs, live = controls(mode, tokens, heads, generator, ratio)
    snapshot = packed.clone()
    invocations = []

    def factory():
        invocations.append(True)
        return (packed[start:start + 192] for start in range(0, tokens, 192))

    kmean = vscale = None
    if replay:
        _, kmean, vscale = native.sol_attn_chunked(
            factory, tokens, heads, freqs, weights, **kwargs,
        )
    invocations.clear()
    stats_before = None if kmean is None else (kmean.clone(), vscale.clone())
    actual, next_k, next_v = native.sol_attn_chunked(
        factory, tokens, heads, freqs, weights, kmean=kmean, vscale=vscale, **kwargs,
    )
    assert len(invocations) == (1 if replay else 2)
    from omni_xpu_kernel import rotary
    q, k, v = packed.clone().view(1, tokens, 3, heads, 128).unbind(2)
    qr, kr = norm_rope_reference(q, k, freqs, weights, 96)
    rotary.rms_kitchen_rope_split_half_(q, k, freqs, *weights, rot_dim=96)
    torch.testing.assert_close(q, qr, rtol=0.02, atol=0.02)
    torch.testing.assert_close(k, kr, rtol=0.02, atol=0.02)
    expected_k = k[:, live].float().mean(1)[0]
    expected_v = (v[:, live].float().abs().amax(1)[0] / 127 * 1.1).clamp_min(1e-8)
    expected = quantized_reference(q, k, v, kwargs,
        expected_k if kmean is None else kmean, expected_v if vscale is None else vscale)
    close_on_live_rows(actual, expected, live, relative_tolerance=0.01)
    record_full_precision_difference(record_property, actual, reference(qr, kr, v, **kwargs), live)
    torch.testing.assert_close(packed, snapshot, rtol=0, atol=0)
    if stats_before is not None:
        for actual_stats, before in zip((kmean, vscale), stats_before):
            torch.testing.assert_close(actual_stats, before, rtol=0, atol=0)
    torch.testing.assert_close(next_k, expected_k, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(next_v, expected_v, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bad", ["length_shape", "length_dtype", "gate_shape", "gate_dtype", "ratio"])
def test_chunked_sparse_modes_reject_invalid_control_metadata(bad):
    native, _ = runtime()
    tokens, heads = 65, 2
    packed = torch.zeros((tokens, 3 * heads * 128), device="xpu", dtype=torch.bfloat16)
    freqs = torch.eye(2, device="xpu").expand(1, tokens, 1, 48, 2, 2).contiguous()
    weights = (torch.ones(128, device="xpu"),) * 2
    kwargs = dict(topk_ratio=0.1, tail=False)
    field = "block_len" if bad.startswith("length") else "coarse_gate"
    if bad == "length_shape":
        kwargs[field] = torch.ones(1, device="xpu", dtype=torch.int32)
    elif bad == "length_dtype":
        kwargs[field] = torch.ones(2, device="xpu", dtype=torch.int64)
    elif bad == "gate_shape":
        kwargs[field] = torch.zeros((1, tokens, heads, 64), device="xpu")
    elif bad == "gate_dtype":
        kwargs[field] = torch.zeros((1, tokens, heads, 128), device="xpu", dtype=torch.int32)
    else:
        field = "topk_ratio"
        kwargs[field] = 1.0
    with pytest.raises(ValueError, match=field):
        native.sol_attn_chunked(lambda: iter([packed]), tokens, heads, freqs, weights, **kwargs)


@pytest.mark.parametrize("mode", ["sla", "vsa"])
@pytest.mark.parametrize("pattern", ["zero", "tied_nonzero_values"])
def test_sparse_modes_zero_and_tied_scores(mode, pattern):
    native, eager = runtime()
    tokens, heads = 257, 2
    generator = torch.Generator(device="xpu").manual_seed(40601)
    q = torch.zeros((1, tokens, heads, 128), device="xpu", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    kwargs, live = controls(mode, tokens, heads, generator, 0.1)
    if pattern == "tied_nonzero_values":
        v.copy_(torch.arange(128, device="xpu").remainder(7).to(torch.bfloat16))
    actual = native.sol_attn(q, k, v, **kwargs)
    expected = eager(q, k, v, **kwargs)
    torch.testing.assert_close(actual[:, live], expected[:, live], rtol=0.01, atol=0.01)
    if pattern == "zero":
        assert int(torch.count_nonzero(actual)) == 0
