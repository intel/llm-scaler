import pytest
import torch

from omni_xpu_kernel import fp8


pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU unavailable")


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quantize_dequantize_matches_torch(input_dtype, fp8_dtype):
    x = torch.randn(17, 65, device="xpu", dtype=input_dtype) * 4
    scale = torch.tensor(0.125, device="xpu", dtype=torch.float32)
    actual = fp8.quantize_per_tensor(x, scale, fp8_dtype)
    limit = torch.finfo(fp8_dtype).max
    expected = torch.clamp(x / scale.to(input_dtype), -limit, limit).to(fp8_dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for out_dtype in (torch.float32, torch.float16, torch.bfloat16):
        restored = fp8.dequantize_per_tensor(actual, scale, out_dtype)
        reference = actual.to(out_dtype) * scale.to(out_dtype)
        assert torch.equal(restored, reference)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_stochastic_rounding_is_deterministic_for_rng(fp8_dtype):
    x = torch.randn(32, 64, device="xpu", dtype=torch.float32)
    rng = torch.randint(0, 256, x.shape, device="xpu", dtype=torch.uint8)
    first = fp8.stochastic_rounding(x, rng, fp8_dtype)
    second = fp8.stochastic_rounding(x, rng, fp8_dtype)
    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
