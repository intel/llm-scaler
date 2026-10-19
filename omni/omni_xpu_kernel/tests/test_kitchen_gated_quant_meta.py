"""Device-neutral fake/schema and SiTU fallback checks; no XPU allocation."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from omni_xpu_kernel import kitchen
from omni_xpu_kernel.int8 import _apply_input_act


@pytest.mark.parametrize("shape", [(512,), (0, 512), (2, 3, 1024)])
def test_gated_quantize_fake_shapes_and_dtypes(shape):
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16)
        codes, scales = torch.ops.omni_xpu.kitchen_gated_quantize_int8(x, "situ", 256)
        assert codes.shape == (*shape[:-1], shape[-1] // 2)
        assert scales.shape == (*shape[:-1], 1)
        assert codes.dtype == torch.int8 and scales.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_omni_situ_fallback_preserves_input_and_half_order(dtype):
    x = torch.tensor([[1000, -1000, 3, -3]], dtype=dtype)
    original = x.clone()
    h, g = x.chunk(2, dim=-1)
    expected = (g / 4).tanh().mul(4).mul(g.sigmoid()).mul((h / 25).tanh().mul(25))
    actual = _apply_input_act(x, "situ")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
