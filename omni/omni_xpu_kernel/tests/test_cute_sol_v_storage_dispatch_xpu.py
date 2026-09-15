"""Exact V storage conversion and generic-layout B70 prepared dispatch."""
import pytest
import torch


def available():
    try:
        from omni_xpu_kernel import cute
        cute._ensure_loaded()
        return torch.xpu.is_available() and torch.xpu.get_device_properties(0).device_id == 0xE223
    except (ImportError, RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(not available(), reason='B70 prepared CUTE API required')


def inputs(tokens, layout, route_mode):
    b, h, d = 2, 3, 128
    generator = torch.Generator(device='xpu').manual_seed(5014 + tokens)
    if layout == 'packed':
        packed = torch.randint(-128, 128, (b, tokens, 3, h, d), device='xpu', dtype=torch.int8, generator=generator)
        q, k, v = packed.unbind(2)
    else:
        values = [torch.randint(-128, 128, (b, h, tokens, d), device='xpu', dtype=torch.int8, generator=generator) for _ in range(3)]
        q, k, v = [value.permute(0, 2, 1, 3) for value in values]
    qs = torch.rand((b, h, tokens), device='xpu', generator=generator) * 0.02 + 0.001
    ks = torch.rand((b, h, tokens), device='xpu', generator=generator) * 0.02 + 0.001
    ks[..., ::7] = 0
    vs = torch.rand((b, h, d), device='xpu', generator=generator) * 0.1 + 0.001
    n = (tokens + 63) // 64
    routes = (torch.rand((b, h, n, n), device='xpu', generator=generator) < 0.2).to(torch.uint8)
    if route_mode != 'sparse':
        routes.fill_(route_mode == 'all')
    tail = torch.randn((b, h, n, 130), device='xpu', generator=generator)
    tail[..., 1] = tail[..., 1].abs() + 0.1
    tail[:, 0] = 0
    return q, k, v, qs, ks, vs, routes, tail, d ** -0.5


@pytest.mark.parametrize('tokens,layout', [(1, 'bhtd'), (63, 'bhtd'), (65, 'packed'),
                                          (257, 'packed'), (513, 'bhtd'), (1025, 'packed')])
@pytest.mark.parametrize('route_mode', ['all', 'sparse', 'none'])
@pytest.mark.parametrize('fp16', [False, True])
def test_half_v_matches_legacy_bytes(monkeypatch, tokens, layout, route_mode, fp16):
    values = inputs(tokens, layout, route_mode)
    originals = [value.clone() for value in values[:3]]
    op = torch.ops.omni_xpu_sol_attn.forward_cute_prepared
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    candidate = op(*values, None, None, None, fp16)
    with monkeypatch.context() as context:
        context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
        legacy = op(*values, None, None, None, fp16)
    assert torch.equal(candidate.view(torch.uint8), legacy.view(torch.uint8))
    assert torch.isfinite(candidate).all().item()
    for actual, original in zip(values[:3], originals):
        assert torch.equal(actual, original)


def test_half_v_route_is_not_tied_to_captured_sequence_lengths(monkeypatch):
    values = inputs(65, 'packed', 'sparse')
    op = torch.ops.omni_xpu_sol_attn.forward_cute_prepared
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    op(*values); torch.xpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.XPU]) as profile:
        op(*values); torch.xpu.synchronize()
    assert any('SolPreparedHalfVKernelTag' in event.name for event in profile.events())
    with monkeypatch.context() as context:
        context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.XPU]) as legacy_profile:
            op(*values); torch.xpu.synchronize()
    names = [event.name for event in legacy_profile.events()]
    assert any('SolCuteKernelTag' in name for name in names)
    assert not any('SolPreparedHalfVKernelTag' in name for name in names)


def row_state_inputs(tokens, layout, route_mode, with_bias):
    values = inputs(tokens, layout, route_mode)
    q = values[0]
    b, t, h, _ = q.shape
    generator = torch.Generator(device='xpu').manual_seed(5114 + tokens)
    state = torch.randn((b, h, t, 144), device='xpu', generator=generator)
    state[..., 0] *= 3
    state[..., 1] = state[..., 1].abs() + 0.1
    state[:, 0, :, 0] = float('-inf')
    state[:, 0, :, 1] = 0
    state[:, 0, :, 16:] = 0
    # These alignment columns are outside the selected-state value contract.
    state[..., 2:16] = float('nan')
    bias = None
    if with_bias:
        bias = torch.randn((b, t), device='xpu', generator=generator) * 2
        bias[0, ::17] = float('-inf')
    return (*values[:8], state, values[8], bias)


@pytest.mark.parametrize('tokens,layout', [(1, 'bhtd'), (63, 'bhtd'), (65, 'packed'),
                                          (257, 'packed'), (513, 'bhtd'), (1025, 'packed')])
@pytest.mark.parametrize('route_mode', ['all', 'sparse', 'none'])
@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('with_bias', [False, True])
def test_half_v_row_state_matches_legacy_bytes(monkeypatch, tokens, layout, route_mode, fp16, with_bias):
    values = row_state_inputs(tokens, layout, route_mode, with_bias)
    originals = [value.clone() for value in (*values[:3], values[8])]
    op = torch.ops.omni_xpu_sol_attn.forward_cute_prepared_split
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    candidate = op(*values, fp16)
    with monkeypatch.context() as context:
        context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
        baseline = op(*values, fp16)
    assert torch.equal(candidate.view(torch.uint8), baseline.view(torch.uint8))
    assert torch.isfinite(candidate).all().item()
    for actual, original in zip((*values[:3], values[8]), originals):
        assert torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))


def test_row_state_uses_half_v_only_on_default_b70(monkeypatch):
    values = row_state_inputs(65, 'packed', 'sparse', True)
    op = torch.ops.omni_xpu_sol_attn.forward_cute_prepared_split
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    op(*values); torch.xpu.synchronize()
    for forced in (False, True):
        with monkeypatch.context() as context:
            if forced:
                context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.XPU]) as profile:
                op(*values); torch.xpu.synchronize()
            names = [event.name for event in profile.events()]
            assert any(('SolCuteKernelTag' if forced else 'SolPreparedHalfVKernelTag') in name for name in names)
            if forced:
                assert not any('SolPreparedHalfVKernelTag' in name for name in names)
