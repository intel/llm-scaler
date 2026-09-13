"""Disabled pooled-tail reuse preserves all four returned tensors byte for byte."""
import pytest
import torch


def available():
    try:
        from omni_xpu_kernel import cute
        cute._ensure_loaded()
        return torch.xpu.is_available() and torch.xpu.get_device_properties(0).device_id == 0xE223
    except (RuntimeError, ImportError, OSError):
        return False


pytestmark = pytest.mark.skipif(not available(), reason='B70 pooled API required')


def inputs(n, unusual=False):
    gen = torch.Generator(device='xpu').manual_seed(50300 + n)
    b, h = 2, 3
    scores = torch.randint(-16, 17, (b, h, n, n), device='xpu', generator=gen).float() * 0.25
    threshold = torch.zeros((b, h, n), device='xpu')
    vsum = torch.randn((b, h, n, 128), device='xpu', generator=gen).bfloat16() * 8
    vscale = torch.rand((b, h, 128), device='xpu', generator=gen) * 0.2 + 0.01
    lengths = torch.arange(n, device='xpu', dtype=torch.int32) % 65
    if unusual:
        vsum[..., 0, 0] = float('inf')
        vsum[..., -1, 1] = float('-inf')
        vsum[..., n//2, 2] = float('nan')
        vscale[..., 3] = 0.0
        vscale[..., 4] = -0.0
        vscale[..., 5] = -1.0
        vscale[..., 6] = float('inf')
        vscale[..., 7] = float('nan')
    return scores, threshold, vsum, vscale, lengths, n * 64 - 17, 0, min(n, 1), 0, min(n, 1), min(n - 1, 28)


@pytest.mark.parametrize('n', [1, 3, 65, 294, 674])
@pytest.mark.parametrize('unusual', [False, True])
@pytest.mark.parametrize('tail,groups', [(False, False), (False, True), (True, False), (True, True)])
def test_disabled_and_legacy_controls_match_bytes(monkeypatch, n, unusual, tail, groups):
    values = inputs(n, unusual)
    op = torch.ops.omni_xpu_sol_attn.pooled_routes
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    candidate = op(*values, tail, groups)
    with monkeypatch.context() as context:
        context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
        baseline = op(*values, tail, groups)
    for actual, expected in zip(candidate, baseline):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_disabled_tail_dispatch_is_control_driven(monkeypatch):
    op = torch.ops.omni_xpu_sol_attn.pooled_routes
    values = inputs(3)
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    op(*values, False, False); torch.xpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.XPU]) as profile:
        op(*values, False, False); torch.xpu.synchronize()
    names = [event.name for event in profile.events()]
    assert any('SolDisabledTailHeadKernel' in name for name in names)
    assert any('SolDisabledTailFillKernel' in name for name in names)
    assert not any('SolPreparedTailKernel' in name for name in names)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.XPU]) as enabled:
        op(*values, True, False); torch.xpu.synchronize()
    assert any('SolPreparedTailKernel' in event.name for event in enabled.events())
