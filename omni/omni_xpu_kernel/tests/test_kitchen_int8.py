"""Native Kitchen RMSNorm and residual pieces for INT8 projection on XPU."""

import pytest
import torch
from torch.nn import functional

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is unavailable",
)


def _relative_error(got, expected):
    return ((got.float() - expected.float()).abs().max() /
            expected.float().abs().max().clamp_min(1e-9)).item()


@pytest.mark.parametrize("rows,features", [
    (512, 256), (1024, 4096), (37, 2048), (256, 8192),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_for_int8_matches_cuda_ut_boundary(rows, features, dtype):
    torch.manual_seed(20260926)
    x = torch.randn(rows, features, device="xpu", dtype=dtype) * 2
    weight = torch.randn(features, device="xpu", dtype=dtype)
    assert kitchen.supports_rms_norm_for_int8()
    actual = kitchen.rms_norm_for_int8(x, weight, 1e-5)
    expected = functional.rms_norm(
        x, (features,), weight=weight, eps=1e-5,
    )
    torch.xpu.synchronize()
    assert actual.shape == expected.shape
    assert _relative_error(actual, expected) < (
        1e-5 if dtype == torch.float32 else 2e-2
    )


@pytest.mark.parametrize("rows,features", [(2, 128), (37, 2048), (256, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_quantize_int8_matches_separate_cuda_contract(rows, features, dtype):
    from omni_xpu_kernel import int8

    torch.manual_seed(20260927)
    x = torch.randn(rows, features, device="xpu", dtype=dtype)
    weight = torch.randn(features, device="xpu", dtype=dtype)
    assert kitchen.supports_rms_norm_quantize_int8()
    actual_q, actual_scale = kitchen.rms_norm_quantize_int8(x, weight, 1e-6)
    normalized = kitchen.rms_norm_for_int8(x, weight, 1e-6)
    expected_q, expected_scale = int8.quantize_int8_rowwise(normalized)
    torch.xpu.synchronize()

    assert actual_q.shape == x.shape and actual_q.dtype == torch.int8
    assert actual_scale.shape == (rows, 1) and actual_scale.dtype == torch.float32
    torch.testing.assert_close(
        actual_scale, expected_scale,
        rtol=1e-5 if dtype == torch.float32 else 2e-3,
        atol=1e-6,
    )
    differences = (actual_q.to(torch.int16) - expected_q.to(torch.int16)).abs()
    assert differences.max().item() <= 1
    assert (differences != 0).float().mean().item() < 0.01


@pytest.mark.parametrize("rows,features,group_size", [
    (2, 256, 256), (37, 2048, 64), (37, 2048, 256),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_convrot_quantize_matches_composed_route(
    rows, features, group_size, dtype
):
    from omni_xpu_kernel import int8

    torch.manual_seed(20260927)
    x = torch.randn(rows, features, device="xpu", dtype=dtype)
    weight = torch.randn(features, device="xpu", dtype=dtype)
    assert kitchen.supports_rms_norm_convrot_quantize_int8()
    actual_q, actual_scale = kitchen.rms_norm_convrot_quantize_int8(
        x, weight, 1e-6, group_size,
    )
    normalized = kitchen.rms_norm_for_int8(x, weight, 1e-6)
    rotated = int8.rotate_convrot(normalized, group_size)
    expected_q, expected_scale = int8.quantize_int8_rowwise(rotated)
    torch.xpu.synchronize()

    assert actual_q.shape == x.shape and actual_q.dtype == torch.int8
    assert actual_scale.shape == (rows, 1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=2e-3, atol=1e-6)
    differences = (actual_q.to(torch.int16) - expected_q.to(torch.int16)).abs()
    assert differences.max().item() <= 1
    assert (differences != 0).float().mean().item() < 0.01


@pytest.mark.parametrize("rows,outputs", [
    (1797, 2048), (1797, 6144), (37, 256),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_residual_matches_cuda_ut_epilogue(rows, outputs, dtype):
    torch.manual_seed(20260926)
    projection = torch.randn(rows, outputs, device="xpu", dtype=dtype)
    residual = torch.randn(rows, outputs, device="xpu", dtype=dtype)
    scale = torch.randn(outputs, device="xpu", dtype=dtype)
    assert kitchen.supports_scaled_residual()
    actual = kitchen.scaled_residual(projection, residual, scale)
    expected = torch.addcmul(residual, projection, scale)
    torch.xpu.synchronize()
    assert actual.shape == expected.shape
    assert _relative_error(actual, expected) < 1e-2


def test_rms_norm_for_int8_rejects_wrong_weight_length():
    x = torch.randn(4, 128, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(127, device="xpu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="weight"):
        kitchen.rms_norm_for_int8(x, weight, 1e-5)
