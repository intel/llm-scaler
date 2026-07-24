import pytest
import torch

from omni_xpu_kernel import norm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [64, 768, 3072])
def test_fused_adaln(dtype, hidden):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    batch, tokens = 2, 7
    x = torch.randn(batch * tokens, hidden, device="xpu", dtype=dtype)
    scale = torch.randn(batch, hidden, device="xpu", dtype=dtype) * 0.1
    shift = torch.randn(batch, hidden, device="xpu", dtype=dtype) * 0.1
    actual = norm.fused_adaln(x, scale, shift, tokens, 1e-6)
    expected = torch.nn.functional.layer_norm(x.float(), (hidden,), eps=1e-6)
    expected = (
        expected * (1 + scale.repeat_interleave(tokens, 0).float())
        + shift.repeat_interleave(tokens, 0).float()
    ).to(dtype)
    tolerance = 3e-2 if dtype != torch.float32 else 3e-4
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
