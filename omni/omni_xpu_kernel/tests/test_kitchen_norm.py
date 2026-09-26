"""Kitchen 0.2.35 per-frame GroupNorm/SiLU/Pad3D on Intel XPU."""

import pytest
import torch
from torch.nn import functional

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is unavailable",
)


def _reference(x, weight, bias, groups, eps, pad, silu):
    batch, channels, frames, height, width = x.shape
    if weight is not None:
        frame_input = x.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width,
        )
        normalized = functional.group_norm(
            frame_input, groups, weight.to(x.dtype),
            None if bias is None else bias.to(x.dtype), eps,
        )
        x = normalized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    if silu:
        x = functional.silu(x)
    left, right, top, bottom, front = pad
    if left or right or top or bottom:
        x = functional.pad(x, (left, right, top, bottom, 0, 0), mode="reflect")
    if front:
        x = functional.pad(x, (0, 0, 0, 0, front, 0))
    return x.contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize(
    "channels,frames,height,width,pad",
    [
        (128, 3, 40, 56, (1, 1, 1, 1, 2)),
        (256, 2, 33, 17, (1, 1, 1, 1, 2)),
        (512, 2, 9, 9, (0, 1, 0, 1, 2)),
        (1024, 1, 16, 16, (1, 1, 1, 1, 0)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_group_norm_silu_pad3d_matches_cuda_ut(channels, frames, height, width, pad, dtype):
    torch.manual_seed(20260926)
    x = (torch.randn(1, channels, frames, height, width, device="xpu", dtype=dtype) * 3 + 0.5)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = torch.randn(channels, device="xpu", dtype=dtype) * 0.5 + 1
    bias = torch.randn(channels, device="xpu", dtype=dtype) * 0.2
    assert kitchen.supports_group_norm_silu_pad3d()
    actual = kitchen.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, True)
    expected = _reference(x, weight, bias, 32, 1e-6, pad, True)
    torch.xpu.synchronize()
    relative = ((actual.float() - expected.float()).abs().max() /
                expected.float().abs().max().clamp_min(1e-9)).item()
    assert actual.shape == expected.shape
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert relative < 5 * torch.finfo(dtype).eps
    if pad[4]:
        assert torch.all(actual[:, :, :pad[4]] == 0)


def test_group_norm_silu_pad3d_pad_only_is_exact():
    x = torch.randn(1, 128, 3, 31, 31, device="xpu", dtype=torch.float16)
    pad = (0, 1, 0, 1, 2)
    actual = kitchen.group_norm_silu_pad3d(x, None, None, 1, 0.0, pad, False)
    expected = _reference(x, None, None, 1, 0.0, pad, False)
    assert torch.equal(actual, expected)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)


def test_group_norm_silu_pad3d_fp32_affine_matches_cuda_ut():
    x = torch.randn(1, 128, 2, 12, 12, device="xpu", dtype=torch.float16)
    weight = torch.randn(128, device="xpu", dtype=torch.float32)
    bias = torch.randn(128, device="xpu", dtype=torch.float32)
    pad = (1, 1, 1, 1, 2)
    actual = kitchen.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, True)
    expected = _reference(x, weight, bias, 32, 1e-6, pad, True)
    relative = ((actual.float() - expected.float()).abs().max() /
                expected.float().abs().max().clamp_min(1e-9)).item()
    assert relative < 5e-3


def test_group_norm_silu_pad3d_rejects_negative_padding():
    x = torch.zeros(1, 64, 3, 8, 8, device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="padding must be non-negative"):
        kitchen.group_norm_silu_pad3d(x, None, None, 1, 0.0,
                                             (0, 0, 0, 0, -1), False)


def test_group_norm_zero_pad_strided_input_and_wide_border():
    x = torch.randn(1, 32, 2, 3, 4, device="xpu", dtype=torch.bfloat16)
    view = x[:, :, 1:, :, ::2]
    pad = (3, 3, 3, 3, 1)
    actual = kitchen.group_norm_silu_pad3d(
        view, None, None, 1, 0.0, pad, False, True,
    )
    expected = functional.pad(view, (3, 3, 3, 3, 1, 0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_group_norm_out_frame_view_preserves_other_frames():
    x = torch.randn(1, 32, 2, 4, 5, device="xpu", dtype=torch.float16)
    weight = torch.randn(32, device="xpu", dtype=torch.float16)
    bias = torch.randn(32, device="xpu", dtype=torch.float16)
    pad = (1, 1, 1, 1, 0)
    full = torch.full(
        (1, 32, 4, 6, 7), 7.0, device="xpu", dtype=torch.float16,
    ).contiguous(memory_format=torch.channels_last_3d)
    view = full[:, :, 2:]
    expected = kitchen.group_norm_silu_pad3d(
        x, weight, bias, 8, 1e-6, pad, True, True,
    )
    kitchen.group_norm_silu_pad3d_out(
        x, weight, bias, 8, 1e-6, pad, True, True, view,
    )
    torch.testing.assert_close(view, expected, rtol=0, atol=0)
    assert bool((full[:, :, :2] == 7).all())
