"""Kitchen 0.2.35 FP16 linear contracts on Intel XPU."""

import math

import pytest
import torch
from torch.nn import functional

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is unavailable",
)


def _relative_error(got, reference):
    return ((got.float() - reference.float()).abs().max() /
            reference.float().abs().max().clamp_min(1e-9)).item()


def _fp16_accum_tol(k):
    return max(0.005, 3.0 * math.sqrt(k) * 2.0 ** -11)


@pytest.mark.parametrize("rows,outputs,features", [
    (37, 264, 0),
    (37, 264, 128),
    (512, 512, 512),
    (1797, 2048, 2048),
    (1797, 2048, 8192),
])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fp16_linear_matches_cuda_ut(rows, outputs, features, with_bias):
    torch.manual_seed(20260926)
    x = torch.randn(rows, features, device="xpu", dtype=torch.float16)
    weight = torch.randn(outputs, features, device="xpu", dtype=torch.float16) * 0.02
    bias = torch.randn(outputs, device="xpu", dtype=torch.float16) if with_bias else None
    assert kitchen.supports_fp16_linear()
    actual = kitchen.fp16_linear(x, weight, bias)
    expected = functional.linear(x, weight, bias)
    torch.xpu.synchronize()
    assert actual.shape == (rows, outputs)
    assert _relative_error(actual, expected) < _fp16_accum_tol(features)


@pytest.mark.parametrize("rows,outputs,features", [
    (37, 264, 128),
    (512, 512, 512),
    (1797, 2048, 2048),
])
def test_fp16_linear_residual_matches_cuda_ut(rows, outputs, features):
    torch.manual_seed(20260926)
    x = torch.randn(rows, features, device="xpu", dtype=torch.float16)
    weight = torch.randn(outputs, features, device="xpu", dtype=torch.float16) * 0.02
    bias = torch.randn(outputs, device="xpu", dtype=torch.float16)
    residual = torch.randn(rows, outputs, device="xpu", dtype=torch.float16)
    scale = torch.randn(outputs, device="xpu", dtype=torch.float16)
    plain = kitchen.fp16_linear(x, weight, bias)
    actual = kitchen.fp16_linear(x, weight, bias, residual, scale)
    expected = torch.addcmul(residual, plain, scale)
    assert _relative_error(actual, expected) < 1e-2


def test_fp16_linear_3d_input_matches_cuda_ut():
    x = torch.randn(2, 512, 2048, device="xpu", dtype=torch.float16)
    weight = torch.randn(1024, 2048, device="xpu", dtype=torch.float16) * 0.02
    bias = torch.randn(1024, device="xpu", dtype=torch.float16)
    residual = torch.randn(2, 512, 1024, device="xpu", dtype=torch.float16)
    scale = torch.randn(1024, device="xpu", dtype=torch.float16)
    actual = kitchen.fp16_linear(x, weight, bias, residual, scale)
    expected = torch.addcmul(residual, functional.linear(x, weight, bias), scale)
    assert actual.shape == expected.shape
    assert _relative_error(actual, expected) < _fp16_accum_tol(2048)


def test_fp16_linear_residual_requires_scale():
    x = torch.randn(4, 128, device="xpu", dtype=torch.float16)
    weight = torch.randn(64, 128, device="xpu", dtype=torch.float16)
    residual = torch.randn(4, 64, device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="requires residual_scale"):
        kitchen.fp16_linear(x, weight, residual=residual)
