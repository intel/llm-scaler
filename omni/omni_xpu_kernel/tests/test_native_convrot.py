import pytest
import torch

from omni_xpu_kernel import int8
from omni_xpu_kernel.int8._reference import _build_hadamard, _rotate_activation


@pytest.mark.parametrize("group_size", [4, 16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_radix4_rotation_matches_matrix(group_size, dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    native = int8._get_native()
    x = torch.randn(5, group_size * 2, device="xpu", dtype=dtype)
    h = _build_hadamard(group_size, x.device, dtype)
    expected = _rotate_activation(x, h, group_size)
    actual = native.rotate_convrot(x, group_size)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_native_convrot_weight_roundtrip():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    native = int8._get_native()
    weight = torch.randn(32, 128, device="xpu", dtype=torch.bfloat16)
    q, scale = native.quantize_int8_convrot_weight(weight, 64, 0)
    restored = native.dequantize_int8_convrot_weight(q, scale, 64)
    assert q.dtype == torch.int8
    assert scale.shape == (32, 1)
    assert (restored - weight.float()).abs().mean().item() < 0.02
