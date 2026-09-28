"""Kitchen 0.2.35 causal decode convolution on Intel XPU."""

import pytest
import torch
from torch.nn import functional

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is unavailable",
)


@pytest.mark.parametrize("steps", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_deltanet_conv_step_matches_cuda_ut_contract(steps, dtype):
    batch, channels, kernel_size = 2, 1024, 4
    generator = torch.Generator(device="xpu").manual_seed(20260926)
    proj = torch.randn(batch, steps, channels, device="xpu", dtype=dtype, generator=generator)
    state = torch.randn(batch, channels, kernel_size - 1, device="xpu", dtype=dtype, generator=generator)
    weight = torch.randn(channels, 1, kernel_size, device="xpu", dtype=dtype, generator=generator) * 0.5
    bias = torch.randn(channels, device="xpu", dtype=dtype, generator=generator) * 0.1

    combined = torch.cat([state, proj.transpose(1, 2)], dim=-1)
    expected = functional.silu(functional.conv1d(combined, weight, bias, groups=channels))
    expected_state = combined[:, :, steps:].contiguous()
    expected_snapshots = (
        torch.stack([combined[:, :, 1 + s:1 + s + kernel_size - 1] for s in range(steps - 1)])
        if steps > 1 else None
    )

    actual_state = state.clone()
    snapshots = (
        torch.empty(steps - 1, batch, channels, kernel_size - 1, device="xpu", dtype=dtype)
        if steps > 1 else None
    )
    assert kitchen.supports_deltanet_conv_step()
    actual = kitchen.deltanet_conv_step(proj, actual_state, weight, bias, snapshots)
    torch.xpu.synchronize()

    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative_error.item() < (1e-5 if dtype == torch.float32 else 1e-2)
    assert torch.equal(actual_state, expected_state)
    if steps > 1:
        assert torch.equal(snapshots, expected_snapshots)


def test_deltanet_conv_step_rejects_more_than_eight_steps():
    proj = torch.zeros(1, 9, 8, device="xpu", dtype=torch.float32)
    state = torch.zeros(1, 8, 3, device="xpu", dtype=torch.float32)
    weight = torch.zeros(8, 1, 4, device="xpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="1<=S<=8"):
        kitchen.deltanet_conv_step(proj, state, weight)


@pytest.mark.parametrize("steps", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("value_dim", [32, 96, 128, 256, 512])
def test_gated_delta_decode_matches_cuda_ut_contract(steps, dtype, value_dim):
    batch, heads, key_heads, key_head_dim, hidden = 2, 4, 2, 128, 256
    key_dim = key_heads * key_head_dim
    channels = 2 * key_dim + heads * value_dim
    scale, eps = key_head_dim ** -0.5, 1e-6
    torch.manual_seed(20260926)
    mixed_qkv = torch.randn(batch, channels, steps, device="xpu", dtype=dtype)
    x = torch.randn(batch, steps, hidden, device="xpu", dtype=dtype)
    w_a = torch.randn(heads, hidden, device="xpu", dtype=dtype) * 0.05
    w_b = torch.randn(heads, hidden, device="xpu", dtype=dtype) * 0.05
    dt_bias = torch.randn(heads, device="xpu")
    g_decay = -torch.rand(heads, device="xpu") - 0.5
    state = torch.randn(batch, heads, key_head_dim, value_dim, device="xpu") * 0.1
    z = torch.randn(batch, steps, heads * value_dim, device="xpu", dtype=dtype)
    norm_weight = torch.rand(value_dim, device="xpu", dtype=dtype) + 0.5

    a = functional.linear(x, w_a)
    b = functional.linear(x, w_b)
    beta = b.sigmoid().reshape(batch, steps, heads)
    decay = (g_decay * functional.softplus(a.float() + dt_bias)).reshape(batch, steps, heads).exp()
    query, key, value = mixed_qkv.transpose(1, 2).split(
        [key_dim, key_dim, heads * value_dim], dim=-1,
    )
    repeat = heads // key_heads
    q = functional.normalize(query.reshape(batch, steps, key_heads, key_head_dim).float(), dim=-1)
    k = functional.normalize(key.reshape(batch, steps, key_heads, key_head_dim).float(), dim=-1)
    q = q.repeat_interleave(repeat, dim=2) * scale
    k = k.repeat_interleave(repeat, dim=2)
    v = value.reshape(batch, steps, heads, value_dim).float()
    reference_state = state.clone()
    reference_rows, reference_snapshots = [], []
    for step in range(steps):
        reference_state.mul_(decay[:, step, :, None, None])
        memory = torch.einsum("bhk,bhkv->bhv", k[:, step], reference_state)
        delta = (v[:, step] - memory) * beta[:, step, :, None]
        reference_state.add_(torch.einsum("bhk,bhv->bhkv", k[:, step], delta))
        reference_rows.append(torch.einsum("bhk,bhkv->bhv", q[:, step], reference_state))
        if step + 1 < steps:
            reference_snapshots.append(reference_state.clone())
    reference = torch.stack(reference_rows, dim=1).to(dtype)
    reference = functional.rms_norm(
        reference.reshape(-1, value_dim), (value_dim,), norm_weight, eps,
    ) * functional.silu(z.reshape(-1, value_dim))
    reference = reference.reshape(batch, steps, heads, value_dim)

    actual_state = state.clone()
    snapshots = (
        torch.empty(steps - 1, batch, heads, key_head_dim, value_dim, device="xpu")
        if steps > 1 else None
    )
    assert kitchen.supports_gated_delta_decode_fused()
    actual = kitchen.gated_delta_decode_fused(
        mixed_qkv, x, w_a, w_b, dt_bias, g_decay, actual_state,
        key_dim, key_heads, scale, z, norm_weight, eps, snapshots,
    )
    torch.xpu.synchronize()
    tolerance = 1e-5 if dtype == torch.float32 else 5e-3
    output_error = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
    state_error = (actual_state - reference_state).norm() / reference_state.norm().clamp_min(1e-12)
    assert actual.shape == (batch, steps, heads, value_dim)
    assert output_error.item() < tolerance
    assert state_error.item() < tolerance
    if steps > 1:
        expected = torch.stack(reference_snapshots)
        snapshot_error = (snapshots - expected).norm() / expected.norm().clamp_min(1e-12)
        assert snapshot_error.item() < tolerance


def test_gated_delta_decode_rejects_nine_steps():
    batch, steps, heads, key_heads, dim, hidden = 1, 9, 4, 2, 128, 256
    key_dim = key_heads * dim
    mixed_qkv = torch.zeros(batch, 2 * key_dim + heads * dim, steps, device="xpu")
    x = torch.zeros(batch, steps, hidden, device="xpu")
    weight = torch.zeros(heads, hidden, device="xpu")
    bias = torch.zeros(heads, device="xpu")
    state = torch.zeros(batch, heads, dim, dim, device="xpu")
    z = torch.zeros(batch, steps, heads * dim, device="xpu")
    norm_weight = torch.ones(dim, device="xpu")
    with pytest.raises(RuntimeError, match="1<=S<=8"):
        kitchen.gated_delta_decode_fused(
            mixed_qkv, x, weight, weight, bias, bias, state,
            key_dim, key_heads, dim ** -0.5, z, norm_weight, 1e-6,
        )
