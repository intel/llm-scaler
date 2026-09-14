"""MTP verify snapshots and accepted-prefix commit, at 27B TP-local shapes."""

import pytest
import torch
import torch.nn.functional as F

if not torch.xpu.is_available():
    pytest.skip("requires an Intel XPU", allow_module_level=True)

import custom_esimd_kernels_sglang  # noqa: F401,E402


@pytest.mark.parametrize("batch,steps", [(1, 2), (4, 2), (1, 4), (4, 4)])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_verify_snapshots_and_rejected_suffix(batch, steps, state_dtype):
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(20260911)
    hk, hv, dim = 8, 24, 128
    width = (2 * hk + hv) * dim
    slots = batch + 2

    def rand(shape, scale=0.2, dtype=torch.float16):
        return (torch.randn(shape, generator=g) * scale).to(dtype)

    # Match the real row-strided packed QKV and BA projections.
    backing = rand((batch * steps, width + hv * dim))
    packed = backing[:, :width]
    ba = rand((batch * steps, 2 * hv))
    a, b = ba[:, hv:], ba[:, :hv]
    alog, bias = rand((hv,), dtype=torch.float32), rand((hv,), dtype=torch.float32)
    state = rand((slots, hv, dim, dim), 0.02, state_dtype)
    cache_ids = torch.arange(batch, 0, -1, dtype=torch.int32)
    inter_ids = torch.arange(batch, dtype=torch.int32)
    starts = torch.arange(batch + 1, dtype=torch.int32) * steps
    xstate = state.xpu()
    inter = torch.full(
        (slots, steps, hv, dim, dim), -7, dtype=state_dtype, device="xpu"
    )
    output = torch.empty(
        (1, batch * steps, hv, dim), dtype=torch.float16, device="xpu"
    )
    xb = backing.xpu()
    xba = ba.xpu()
    torch.ops.eagle_ops.gdn_target_verify_packed(
        output, xb[:, :width], xba[:, hv:], xba[:, :hv], alog.xpu(), bias.xpu(),
        xstate, inter, starts.xpu(), cache_ids.xpu(), inter_ids.xpu(),
        steps, hk, dim, hv, dim,
    )
    q, k, v = packed.float().split([hk * dim, hk * dim, hv * dim], -1)
    # Kernel normalizes using sqrt(sum(x*x) + 1e-6), not clamp(norm, eps).
    qr = packed[:, : hk * dim].float().reshape(batch, steps, hk, dim)
    kr = k.reshape(batch, steps, hk, dim)
    q = qr / (qr.square().sum(-1, keepdim=True) + 1e-6).sqrt() / dim**0.5
    k = kr / (kr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q, k = q.repeat_interleave(3, 2), k.repeat_interleave(3, 2)
    v = v.reshape(batch, steps, hv, dim)
    aa, bb = a.float().reshape(batch, steps, hv), b.float().reshape(batch, steps, hv)
    decay = torch.exp(-alog.exp() * F.softplus(aa + bias))
    beta = bb.sigmoid()
    current = state[cache_ids.long()].float()
    snapshots, expected_out = [], []
    for t in range(steps):
        current = current * decay[:, t, :, None, None]
        memory = (current * k[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - memory) * beta[:, t, :, None]
        current = current + delta.unsqueeze(-1) * k[:, t, :, None, :]
        expected_out.append((current * q[:, t, :, None, :]).sum(-1))
        snapshots.append(current.to(state_dtype))
    expected_out = torch.stack(expected_out, 1).reshape_as(output).half()
    expected_inter = torch.stack(snapshots, 1)
    torch.testing.assert_close(output.cpu(), expected_out, atol=1e-4, rtol=2e-3)
    torch.testing.assert_close(inter[:batch].cpu(), expected_inter, atol=1e-4, rtol=2e-3)
    assert torch.equal(xstate.cpu(), state), "verify must leave the main state untouched"
    assert (inter[batch:] == -7).all(), "verify wrote an unused intermediate slot"

    # Commit first, middle, or last verified state; negative index means no commit.
    accepted = torch.tensor([0, steps - 1, -1, steps // 2][:batch], dtype=torch.int32)
    expected_state = state.clone()
    for r, step in enumerate(accepted.tolist()):
        if step >= 0:
            expected_state[cache_ids[r]] = inter[r, step].cpu()
    torch.ops.eagle_ops.mamba_state_scatter(
        xstate.unsqueeze(0), inter.unsqueeze(0), cache_ids.xpu(), accepted.xpu()
    )
    assert torch.equal(xstate.cpu(), expected_state), "commit selected a rejected suffix"


@pytest.mark.parametrize("batch,steps", [(1, 2), (4, 2), (1, 4), (4, 4)])
def test_conv_verify_snapshots_and_commit(batch, steps):
    g = torch.Generator().manual_seed(20260912)
    channels, window = 5120, 4
    slots = batch + 2

    def rand(shape):
        return (torch.randn(shape, generator=g) * 0.2).half()

    backing = rand((batch, steps, channels + 3072))
    x = backing[:, :, :channels]
    weight, bias = rand((channels, window)), rand((channels,))
    state = rand((slots, channels, window - 1))
    ids = torch.arange(batch, 0, -1, dtype=torch.int32)
    out = torch.empty((batch, steps, channels), dtype=torch.float16, device="xpu")
    inter = torch.full(
        (slots, steps, channels, window - 1),
        -7, dtype=torch.float16, device="xpu",
    )
    xs = state.xpu()
    torch.ops.eagle_ops.causal_conv1d_verify_tm(
        out, backing.xpu()[:, :, :channels], weight.xpu(), bias.xpu(), xs, inter,
        ids.xpu(), torch.arange(batch, dtype=torch.int32, device="xpu"), 1,
    )
    history = state[ids.long()]
    outputs, snapshots = [], []
    for t in range(steps):
        full = torch.cat([history, x[:, t].unsqueeze(-1)], -1)
        outputs.append(
            F.silu((full.float() * weight.float()).sum(-1) + bias.float()).half()
        )
        history = full[:, :, 1:]
        snapshots.append(history)
    torch.testing.assert_close(out.cpu(), torch.stack(outputs, 1), atol=1e-4, rtol=2e-3)
    assert torch.equal(inter[:batch].cpu(), torch.stack(snapshots, 1))
    assert torch.equal(xs.cpu(), state)
    assert (inter[batch:] == -7).all()
    accepted = torch.tensor([0, steps - 1, -1, steps // 2][:batch], dtype=torch.int32)
    expected = state.clone()
    for r, step in enumerate(accepted.tolist()):
        if step >= 0:
            expected[ids[r]] = inter[r, step].cpu()
    torch.ops.eagle_ops.mamba_state_scatter(
        xs.unsqueeze(0), inter.unsqueeze(0), ids.xpu(), accepted.xpu()
    )
    assert torch.equal(xs.cpu(), expected)
