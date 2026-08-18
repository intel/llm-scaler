"""Focused coverage for model-independent large-M INT8 linear tiling."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def int8_native():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("large-M INT8 tiling requires XPU")

    from omni_xpu_kernel import int8

    native = int8._get_native()
    required = (
        "int8_linear",
        "int8_linear_prequantized_out",
        "quantize_int8_rowwise_fused",
    )
    if native is None or any(not hasattr(native, name) for name in required):
        pytest.skip("native extension lacks the tiled INT8 primitives")
    return int8, native


def test_chunk_rows_are_derived_from_live_byte_contract(monkeypatch):
    from omni_xpu_kernel import int8

    monkeypatch.setattr(int8, "_INT8_LINEAR_TILING_CHUNK_BYTES", 1 << 20)
    monkeypatch.setattr(int8, "_INT8_LINEAR_TILING_ROW_ALIGNMENT", 16)

    assert int8._int8_linear_chunk_rows(128, 2560, 2) == 192
    assert int8._int8_linear_chunk_rows(4096, 96, 2) == 240
    assert int8._int8_linear_chunk_rows(131072, 1, 4) == 7


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("input_kind", ["normal", "alternating_extremes"])
def test_public_large_m_tiling_is_bit_exact(
    int8_native, monkeypatch, with_bias, input_kind
):
    int8, _native = int8_native
    generator = torch.Generator(device="xpu").manual_seed(20260817)
    shape = (2, 7, 256)
    if input_kind == "normal":
        value = torch.randn(
            shape,
            device="xpu",
            dtype=torch.bfloat16,
            generator=generator,
        )
    else:
        pattern = torch.tensor(
            [-32768.0, -16.0, -1.0, -0.0, 0.0, 1.0, 16.0, 32768.0],
            device="xpu",
            dtype=torch.bfloat16,
        )
        repeats = (value_count := 2 * 7 * 256) // pattern.numel() + 1
        value = pattern.repeat(repeats)[:value_count].reshape(shape)
    weight = torch.randint(
        -127,
        128,
        (96, shape[-1]),
        device="xpu",
        dtype=torch.int8,
        generator=generator,
    )
    weight_scale = torch.rand(
        96, device="xpu", dtype=torch.float32, generator=generator
    ).add_(0.01)
    bias = None
    if with_bias:
        bias = torch.randn(
            96, device="xpu", dtype=torch.bfloat16, generator=generator
        )

    monkeypatch.setattr(int8, "_INT8_LINEAR_TILING_MIN_QUANTIZED_BYTES", 1 << 60)
    expected = int8.int8_linear(
        value, weight, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )

    monkeypatch.setattr(int8, "_INT8_LINEAR_TILING_MIN_QUANTIZED_BYTES", 1)
    monkeypatch.setattr(int8, "_INT8_LINEAR_TILING_MIN_LIVE_BYTES", 1)
    monkeypatch.setattr(
        int8, "_INT8_LINEAR_TILING_CHUNK_BYTES", 4 * (256 + 4 + 96 * 2)
    )
    monkeypatch.setattr(int8, "_INT8_LINEAR_TILING_ROW_ALIGNMENT", 4)
    actual = int8.int8_linear(
        value, weight, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )
    torch.xpu.synchronize()

    assert actual.shape == expected.shape == (2, 7, 96)
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
