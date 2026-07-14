import pytest
import torch

from omni_xpu_kernel import svdq


def test_unsigned_svdq_quantize_dequantize():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    x = torch.rand(7, 128, device="xpu", dtype=torch.bfloat16) * 4
    packed, scales = svdq.quantize_act_uint4(x, 64)
    unpacked = svdq.unpack_int4(packed, signed=False)
    restored = svdq.dequantize_u4(packed, scales, torch.bfloat16)
    assert unpacked.min().item() >= 0
    assert unpacked.max().item() <= 15
    assert scales.shape == (2, 7)
    assert (restored.float() - x.float()).abs().mean().item() < 0.08
