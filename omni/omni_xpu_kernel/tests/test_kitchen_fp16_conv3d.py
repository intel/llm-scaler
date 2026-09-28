"""Kitchen FP16 Conv3D CUDA unit contracts on Intel XPU."""

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


@pytest.mark.parametrize(
    "channels,outputs,frames,height,width,kernel_size,stride",
    [
        (128, 128, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
        (256, 256, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
        (128, 128, 9, 129, 129, (3, 3, 3), (1, 2, 2)),
        (128, 256, 5, 128, 128, (1, 1, 1), (1, 1, 1)),
    ],
)
@pytest.mark.parametrize("with_bias,with_residual", [
    (True, False), (False, False), (True, True),
])
def test_fp16_conv3d_matches_cuda_ut(
    channels, outputs, frames, height, width, kernel_size, stride,
    with_bias, with_residual,
):
    torch.manual_seed(20260926)
    x = torch.randn(1, channels, frames, height, width, device="xpu", dtype=torch.float16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = torch.randn(outputs, channels, *kernel_size, device="xpu", dtype=torch.float16) * 0.02
    weight = weight.contiguous(memory_format=torch.channels_last_3d)
    bias = torch.randn(outputs, device="xpu", dtype=torch.float16) if with_bias else None
    residual = None
    if with_residual:
        shape = (1, outputs,
                 (frames - kernel_size[0]) // stride[0] + 1,
                 (height - kernel_size[1]) // stride[1] + 1,
                 (width - kernel_size[2]) // stride[2] + 1)
        residual = torch.randn(shape, device="xpu", dtype=torch.float16)
        residual = residual.contiguous(memory_format=torch.channels_last_3d)
    assert kitchen.supports_fp16_conv3d()
    actual = kitchen.fp16_conv3d(x, weight, bias, residual, stride)
    reference = functional.conv3d(
        x.float(), weight.float(), None if bias is None else bias.float(),
        stride=stride,
    )
    if residual is not None:
        reference = reference + residual.float()
    torch.xpu.synchronize()
    assert actual.shape == reference.shape
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert _relative_error(actual, reference) < _fp16_accum_tol(
        channels * math.prod(kernel_size)
    )


def test_fp16_conv3d_small_native_case():
    x = torch.randn(1, 64, 3, 6, 6, device="xpu", dtype=torch.float16)
    weight = torch.randn(64, 64, 3, 3, 3, device="xpu", dtype=torch.float16) * 0.02
    residual = torch.randn(1, 64, 1, 4, 4, device="xpu", dtype=torch.float16)
    actual = kitchen.fp16_conv3d(x, weight, residual=residual)
    expected = functional.conv3d(x.float(), weight.float()) + residual.float()
    assert _relative_error(actual, expected) < _fp16_accum_tol(64 * 27)


def test_fp16_conv3d_pixel_channels_are_supported():
    x = torch.randn(1, 3, 5, 130, 130, device="xpu", dtype=torch.float16)
    weight = torch.randn(128, 3, 3, 3, 3, device="xpu", dtype=torch.float16) * 0.1
    actual = kitchen.fp16_conv3d(x, weight)
    expected = functional.conv3d(x.float(), weight.float())
    assert actual.shape == (1, 128, 3, 128, 128)
    assert _relative_error(actual, expected) < _fp16_accum_tol(3 * 27)


def test_fp16_conv3d_rejects_invalid_stride():
    x = torch.zeros(1, 16, 4, 10, 10, device="xpu", dtype=torch.float16)
    weight = torch.zeros(16, 16, 3, 3, 3, device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="stride"):
        kitchen.fp16_conv3d(x, weight, stride=(0, 1, 1))


def test_fp16_conv3d_rejects_residual_shape_mismatch():
    x = torch.zeros(1, 16, 4, 10, 10, device="xpu", dtype=torch.float16)
    weight = torch.zeros(16, 16, 3, 3, 3, device="xpu", dtype=torch.float16)
    bad_residual = torch.zeros(1, 8, 2, 8, 8, device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="residual"):
        kitchen.fp16_conv3d(x, weight, residual=bad_residual)
