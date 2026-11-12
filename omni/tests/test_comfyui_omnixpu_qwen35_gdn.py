"""GDN precision boundaries and integration with the maintained chunk algorithm."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def adapter():
    path = (Path(__file__).parents[1] / "ComfyUI-OmniXPU" / "adapters"
            / "qwen35_gdn.py")
    spec = importlib.util.spec_from_file_location("qwen35_gdn_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_norm_against_independent_fp64(adapter, dtype):
    x = torch.randn(2, 7, 3, 142, generator=torch.Generator().manual_seed(31))
    x = x.to(dtype)[..., ::2]  # noncontiguous, width 71; no captured shape assumption
    x[0, 0] = 0
    x[0, 1] = 1e-6
    expected = x.double() / x.double().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-5)
    actual = adapter._l2norm_fp32(x, 1e-5)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-8)


def test_non_xpu_normalization_preserves_original(adapter):
    x = torch.randn(3, 71).to(torch.bfloat16)
    original = F.normalize
    actual = adapter._Functional(F).normalize(x, dim=-1)
    assert F.normalize is original and actual.dtype == x.dtype
    torch.testing.assert_close(actual, original(x, dim=-1), rtol=0, atol=0)


def test_xpu_chunk_keeps_output_and_state_contract(adapter, monkeypatch):
    if not torch.xpu.is_available():
        pytest.skip("XPU integration")
    qwen35 = pytest.importorskip("comfy.text_encoders.qwen35")
    current = qwen35.torch_chunk_gated_delta_rule
    original = getattr(current, adapter._MARKER, current)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", original)
    original_normalize = F.normalize
    assert adapter.apply() == (True, None)
    patched = qwen35.torch_chunk_gated_delta_rule
    assert adapter.apply() == (True, None)
    assert qwen35.torch_chunk_gated_delta_rule is patched

    g = torch.Generator().manual_seed(71)
    q, k, v = [torch.randn(2, 9, 3, width, generator=g).bfloat16().xpu()
               for width in (71, 71, 37)]
    decay = -torch.rand(2, 9, 3, generator=g).xpu()
    beta = torch.rand(2, 9, 3, generator=g).bfloat16().xpu()
    with torch.inference_mode():
        out, state = patched(q, k, v, decay, beta, output_final_state=True)
        ref, ref_state = original(q.float(), k.float(), v, decay, beta,
                                  output_final_state=True)
    assert out.dtype == torch.bfloat16 and state.dtype == torch.float32
    assert F.normalize is original_normalize
    torch.testing.assert_close(out, ref.bfloat16(), rtol=8e-3, atol=1e-5)
    torch.testing.assert_close(state, ref_state, rtol=2e-5, atol=1e-6)
