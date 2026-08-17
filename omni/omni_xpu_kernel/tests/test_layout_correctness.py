import pytest
import torch
import torch.nn.functional as functional

from omni_xpu_kernel import __xpu_target__, layout


pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or __xpu_target__ != "bmg",
    reason="BMG XPU is required",
)

_CASES = (
    (1, 128, 4, 512, 512),
    (1, 256, 4, 256, 256),
    (1, 256, 4, 512, 512),
    (1, 512, 4, 256, 256),
    (1, 512, 2, 128, 128),
)


def _make_input(shape):
    _batch, channels, temporal, height, width = shape
    backing = torch.randn(
        (temporal, channels, height, width),
        device="xpu",
        dtype=torch.float16,
    )
    value = backing.reshape(1, temporal, channels, height, width).transpose(1, 2)
    assert value.stride() == (
        channels * temporal * height * width,
        height * width,
        channels * height * width,
        width,
        1,
    )
    return value


def _assert_bitwise_equal(actual, expected):
    torch.xpu.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_cat_pad_bmg_capability_is_native():
    assert layout.supports_cat_pad_bmg()


@pytest.mark.parametrize("shape", _CASES)
def test_cat_pad_bmg_matches_arbitrary_materialized_prefix(shape):
    value = _make_input(shape)
    prefix = torch.randn(
        (1, shape[1], 2, shape[3], shape[4]),
        device="xpu",
        dtype=torch.float16,
    )
    expected = functional.pad(
        torch.cat((prefix, value), dim=2),
        (1, 1, 1, 1, 0, 0),
    )
    actual = layout.cat_pad_bmg(prefix, value)

    assert actual.is_contiguous()
    assert actual.shape == expected.shape
    _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    "pattern",
    ("positive_zero", "negative_zero", "alternating_extrema", "nan_inf"),
)
def test_cat_pad_bmg_preserves_adversarial_bits(pattern):
    shape = (1, 512, 2, 128, 128)
    value = _make_input(shape)
    prefix = torch.empty(
        (1, 512, 2, 128, 128), device="xpu", dtype=torch.float16
    )
    if pattern == "positive_zero":
        value.zero_()
        prefix.zero_()
    elif pattern == "negative_zero":
        value.fill_(-0.0)
        prefix.fill_(-0.0)
    elif pattern == "alternating_extrema":
        value[..., 0::2].fill_(65504.0)
        value[..., 1::2].fill_(-65504.0)
        prefix[..., 0::2].fill_(-65504.0)
        prefix[..., 1::2].fill_(65504.0)
    else:
        value.zero_()
        prefix.zero_()
        value[0, 0, 0, 0, 0] = float("nan")
        value[0, 1, 0, 0, 0] = float("inf")
        prefix[0, 2, 0, 0, 0] = float("-inf")

    expected = functional.pad(
        torch.cat((prefix, value), dim=2),
        (1, 1, 1, 1, 0, 0),
    )
    actual = layout.cat_pad_bmg(prefix, value)
    _assert_bitwise_equal(actual, expected)


def test_cat_pad_bmg_rejects_unvalidated_contiguous_input():
    value = torch.randn(
        (1, 512, 2, 128, 128), device="xpu", dtype=torch.float16
    )
    prefix = torch.randn(
        (1, 512, 2, 128, 128), device="xpu", dtype=torch.float16
    )
    with pytest.raises(RuntimeError, match="temporal-major"):
        layout.cat_pad_bmg(prefix, value)
