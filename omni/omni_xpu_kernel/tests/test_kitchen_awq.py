"""Kitchen AWQ W4A16 contracts derived from the CUDA/HIP backend tests."""

import pytest
import torch

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is unavailable",
)


@pytest.mark.parametrize("rows,columns,width,group_size", [
    (1, 512, 512, 64), (8, 1024, 1024, 64),
    (64, 1152, 1152, 64), (2, 128, 256, 128),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_matches_eager_cuda_hip_contract(
    rows, columns, width, group_size, dtype
):
    from comfy_kitchen.backends.eager.awq import gemv_awq_w4a16 as eager_awq

    torch.manual_seed(20260927)
    x = torch.randn(rows, width, device="xpu", dtype=dtype)
    packed = torch.randint(
        0, 256, (columns, width // 2), device="xpu", dtype=torch.uint8,
    ).view(torch.int8)
    scales = torch.randn(width // group_size, columns, device="xpu", dtype=dtype).abs() * 0.01
    zeros = torch.randn(width // group_size, columns, device="xpu", dtype=dtype) * 0.01
    bias = torch.randn(columns, device="xpu", dtype=dtype)

    assert kitchen.supports_gemv_awq_w4a16()
    actual = kitchen.gemv_awq_w4a16(
        x, packed, scales, zeros, bias, group_size,
    )
    expected = eager_awq(x, packed, scales, zeros, bias, group_size)
    torch.xpu.synchronize()
    relative = (
        (actual.float() - expected.float()).norm() /
        expected.float().norm().clamp_min(1e-9)
    ).item()
    assert actual.shape == (rows, columns)
    assert actual.dtype == dtype
    assert relative < 1e-2


def test_awq_rejects_mismatched_scale_shape():
    x = torch.ones(1, 128, device="xpu", dtype=torch.bfloat16)
    packed = torch.zeros(8, 64, device="xpu", dtype=torch.int8)
    scales = torch.ones(1, 8, device="xpu", dtype=torch.bfloat16)
    zeros = torch.zeros_like(scales)
    with pytest.raises(RuntimeError, match="scales/zeros"):
        kitchen.gemv_awq_w4a16(x, packed, scales, zeros, None, 64)
