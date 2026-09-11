"""Native Sol preparation against independent XPU-only tensor reductions."""
import math

import pytest
import torch
import torch.nn.functional as F


NAMES = ("q", "k", "v", "qs", "ks", "vs", "cq", "cqs", "ck", "cks",
         "qmean", "kblocks", "kmean", "kvar", "vsum", "threshold")


def quantize(value, minimum):
    scale = (value.abs().amax(-1) / 127).clamp_min(minimum)
    return (value * scale.reciprocal().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8), scale


def reference(q, k, v, scale, tau, lengths):
    batch, tokens, heads, dim = q.shape
    blocks = (tokens + 63) // 64
    block = torch.arange(blocks, device=q.device)
    live_lengths = (tokens - 64 * block).clamp_max(64)
    if lengths.numel():
        live_lengths = torch.minimum(live_lengths, lengths.clamp_min(1))
    token = torch.arange(tokens, device=q.device)
    valid = token % 64 < live_lengths[token // 64]
    qf, kf, vf = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    def pool(value):
        padded = F.pad(value * valid.reshape(1, 1, tokens, 1), (0, 0, 0, blocks * 64 - tokens))
        return padded.reshape(batch, heads, blocks, 64, dim).sum(-2)
    qm = pool(qf) / live_lengths.reshape(1, 1, blocks, 1)
    km = pool(kf) / live_lengths.reshape(1, 1, blocks, 1)
    vsum = pool(vf).to(torch.bfloat16)
    mean = km.mean(-2)
    var = (km.square().mean(-2) - mean.square()).clamp_min(0)
    cq, cqs = quantize(qm, 1e-8)
    ck, cks = quantize(km - mean.unsqueeze(-2), 1e-12)
    threshold = tau * ((qm.square() * var.unsqueeze(-2)).sum(-1) * (scale * math.log2(math.e))**2 + 1e-6).sqrt()
    qr, qs = quantize(qf, 1e-8)
    kr, ks = quantize(kf - mean.unsqueeze(-2), 1e-8)
    qr.masked_fill_(~valid.reshape(1, 1, tokens, 1), 0)
    kr.masked_fill_(~valid.reshape(1, 1, tokens, 1), 0)
    qs.masked_fill_(~valid.reshape(1, 1, tokens), 0)
    ks.masked_fill_(~valid.reshape(1, 1, tokens), 0)
    vf = vf * valid.reshape(1, 1, tokens, 1)
    vs = (vf.abs().amax(-2) / 127).clamp_min(1e-8)
    vr = (vf * vs.reciprocal().unsqueeze(-2)).round().clamp(-127, 127).to(torch.int8)
    result = dict(zip(NAMES, (qr.permute(0, 2, 1, 3), kr.permute(0, 2, 1, 3), vr.permute(0, 2, 1, 3),
        qs, ks, vs, cq, cqs, ck, cks, qm, km, mean, var, vsum, threshold)))
    result["_scaled"] = {
        "q": (qf * qs.clamp_min(1e-8).reciprocal().unsqueeze(-1)).permute(0, 2, 1, 3),
        "k": ((kf - mean.unsqueeze(-2)) * ks.clamp_min(1e-8).reciprocal().unsqueeze(-1)).permute(0, 2, 1, 3),
        "v": (vf * vs.reciprocal().unsqueeze(-2)).permute(0, 2, 1, 3),
        "cq": qm * cqs.reciprocal().unsqueeze(-1),
        "ck": (km - mean.unsqueeze(-2)) * cks.reciprocal().unsqueeze(-1),
    }
    return result


@pytest.mark.parametrize("tokens,layout", [(1, "bthd"), (63, "bhld"), (64, "bthd"),
                                          (65, "packed"), (257, "packed")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("zeros", [False, True])
def test_native_carriers(tokens, layout, dtype, masked, zeros):
    generator = torch.Generator(device="xpu").manual_seed(9201 + tokens)
    batch, heads = 2, 3
    if layout == "packed":
        packed = torch.randn((batch, tokens, 3, heads, 128), device="xpu", dtype=dtype, generator=generator)
        q, k, v = packed.unbind(2)
    else:
        shape = (batch, tokens, heads, 128) if layout == "bthd" else (batch, heads, tokens, 128)
        values = [torch.randn(shape, device="xpu", dtype=dtype, generator=generator) for _ in range(3)]
        q, k, v = values if layout == "bthd" else [x.permute(0, 2, 1, 3) for x in values]
    if zeros:
        q.zero_(); k.zero_(); v.zero_()
    blocks = (tokens + 63) // 64
    lengths = torch.arange(blocks, dtype=torch.int32, device="xpu") * 31 - 1 if masked else torch.empty(0, dtype=torch.int32, device="xpu")
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    scale, tau = 128**-0.5, 1.3
    actual = dict(zip(NAMES, torch.ops.omni_xpu_sol_attn.prepare_carriers(q, k, v, scale, tau, lengths)))
    expected = reference(q, k, v, scale, tau, lengths)
    for name in NAMES:
        assert actual[name].device.type == "xpu" and actual[name].shape == expected[name].shape
        assert actual[name].dtype == expected[name].dtype
        if actual[name].dtype == torch.int8:
            # Permit an adjacent byte only when the independent FP32 scaled
            # value lies within roundoff of a half-integer. A mismatch-count
            # percentage does not establish that per-element property.
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1)
            mismatch = actual[name] != expected[name]
            scaled = expected["_scaled"][name]
            distance = (0.5 - (scaled - scaled.round()).abs()).abs()
            budget = 8 * torch.finfo(torch.float32).eps * scaled.abs().clamp_min(1)
            assert bool((~mismatch | (distance <= budget)).all().item()), name
        else:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=1e-7)
    # Zero K scale is the prepared mainloop's dead-key contract. Compare its
    # composition on valid rows using an unrelated full FP32 attention oracle.
    if masked and not zeros and dtype == torch.bfloat16:
        from test_cute_sol_prepared_correctness import reference as attention_reference
        routes = torch.ones((batch, heads, blocks, blocks), dtype=torch.uint8, device="xpu")
        tail = torch.zeros((batch, heads, blocks, 130), device="xpu")
        args = [actual[name] for name in ("q", "k", "v", "qs", "ks", "vs")]
        output = torch.ops.omni_xpu_sol_attn.forward_cute_prepared(*args, routes, tail, scale)
        wanted = attention_reference(*args, routes, tail, scale)
        torch.testing.assert_close(output, wanted, rtol=0.02, atol=0.02)
