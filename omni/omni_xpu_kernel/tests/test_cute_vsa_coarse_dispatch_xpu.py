"""Generic-N B70 coarse dispatch and preserved legacy fallback."""
import importlib

import pytest
import torch


def has_b70_coarse():
    try:
        return (torch.xpu.is_available() and
                torch.xpu.get_device_properties(0).device_id == 0xE223 and
                hasattr(importlib.import_module('omni_xpu_kernel.cute.sol_attn_v2')._ops(), 'coarse_output'))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not has_b70_coarse(), reason='B70 coarse API unavailable')


def inputs(n):
    gen = torch.Generator(device='xpu').manual_seed(4900+n)
    q = torch.randn((2, 3, n, 128), device='xpu', generator=gen) * 0.125
    k = torch.randn(q.shape, device='xpu', generator=gen) * 0.125
    v = (torch.randn(q.shape, device='xpu', generator=gen) * 8).bfloat16()
    lengths = torch.randint(0, 65, (n,), device='xpu', generator=gen, dtype=torch.int32)
    return q, k, v, lengths, (n-1)*64+17, 128**-0.5


@pytest.mark.parametrize('n', [1, 3, 4, 5, 31, 33, 65, 129, 294, 674])
def test_b70_coarse_dynamic_lengths_match_legacy_bytes(monkeypatch, n):
    op = importlib.import_module('omni_xpu_kernel.cute.sol_attn_v2')._ops().coarse_output
    values = inputs(n)
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    tiled = op(*values)
    with monkeypatch.context() as context:
        context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
        legacy = op(*values)
    torch.xpu.synchronize()
    assert tiled.shape == values[0].shape and tiled.dtype == torch.float32
    assert torch.equal(tiled.view(torch.uint8), legacy.view(torch.uint8))
    assert torch.isfinite(tiled).all()


def test_b70_coarse_route_does_not_require_a_captured_length(monkeypatch):
    op = importlib.import_module('omni_xpu_kernel.cute.sol_attn_v2')._ops().coarse_output
    values = inputs(5)
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU', raising=False)
    op(*values); torch.xpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.XPU]) as profile:
        op(*values); torch.xpu.synchronize()
    assert any('SolCoarseScoreTiledScalarKernel' in event.name for event in profile.events())
    with monkeypatch.context() as context:
        context.setenv('OMNI_XPU_FORCE_SKU', 'generic')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.XPU]) as legacy_profile:
            op(*values); torch.xpu.synchronize()
    names = [event.name for event in legacy_profile.events()]
    assert any('SolCoarseScoreKernel' in name for name in names)
    assert not any('SolCoarseScoreTiledScalarKernel' in name for name in names)
