"""Focused coverage for bounded-memory MiniMax H3 FFN-down execution."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def int8_native():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("H3 low-peak route requires XPU")

    from omni_xpu_kernel import int8

    native = int8._get_native()
    required = (
        "fused_silu_mul_exact_bf16",
        "int8_linear_prequantized",
        "int8_linear_prequantized_out",
        "quantize_int8_rowwise_fused",
        "rotate_convrot",
    )
    if native is None or any(not hasattr(native, name) for name in required):
        pytest.skip("native extension lacks the H3 low-peak primitives")
    return int8, native


def test_low_peak_threshold_is_allocation_structural():
    from omni_xpu_kernel import int8

    assert int8._H3_LOW_PEAK_MIN_ACTIVATION_BYTES == 2 * 1024**3
    assert int8._h3_low_peak_chunk_rows(14336) == 8192
    below = 44929 * 14336 * 2
    above = 100034 * 14336 * 2
    assert below < int8._H3_LOW_PEAK_MIN_ACTIVATION_BYTES
    assert above >= int8._H3_LOW_PEAK_MIN_ACTIVATION_BYTES
    assert 44929 * 7168 * 2 < int8._H3_LOW_PEAK_MIN_ROTATION_BYTES
    assert 100034 * 7168 * 2 >= int8._H3_LOW_PEAK_MIN_ROTATION_BYTES


def test_prequantized_out_is_bit_exact_and_reuses_storage(int8_native):
    _int8, native = int8_native
    generator = torch.Generator(device="xpu").manual_seed(20260805)
    value = torch.randn(
        17, 256, device="xpu", dtype=torch.bfloat16, generator=generator
    )
    x_int8, x_scale = native.quantize_int8_rowwise_fused(value)
    weight = torch.randint(
        -127,
        128,
        (96, 256),
        device="xpu",
        dtype=torch.int8,
        generator=generator,
    )
    weight_scale = torch.rand(
        96, device="xpu", dtype=torch.float32, generator=generator
    ).add_(0.01)

    expected = native.int8_linear_prequantized(
        x_int8, x_scale, weight, weight_scale, None, 2
    )
    storage = torch.empty_like(expected)
    actual = native.int8_linear_prequantized_out(
        x_int8, x_scale, weight, weight_scale, None, 2, storage
    )
    torch.xpu.synchronize()

    assert actual.data_ptr() == storage.data_ptr()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("input_kind", ["normal", "alternating_extremes"])
def test_public_low_peak_convrot_route_is_bit_exact(
    int8_native, monkeypatch, input_kind
):
    int8, _native = int8_native
    generator = torch.Generator(device="xpu").manual_seed(20260807)
    rows = 9
    shape = (rows, int8._H3_ATTN_OUTPUT_INPUT_FEATURES)
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
        repeats = (rows * shape[1] + pattern.numel() - 1) // pattern.numel()
        value = pattern.repeat(repeats)[: rows * shape[1]].reshape(shape)
    weight = torch.randint(
        -127,
        128,
        (int8._H3_ATTN_OUTPUT_FEATURES, shape[1]),
        device="xpu",
        dtype=torch.int8,
        generator=generator,
    )
    weight_scale = torch.rand(
        int8._H3_ATTN_OUTPUT_FEATURES,
        device="xpu",
        dtype=torch.float32,
        generator=generator,
    ).add_(0.01)

    monkeypatch.setattr(int8, "_H3_LOW_PEAK_MIN_ROTATION_BYTES", 1 << 60)
    expected = int8.int8_linear(
        value,
        weight,
        weight_scale,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    )

    monkeypatch.setattr(int8, "_H3_LOW_PEAK_MIN_ROTATION_BYTES", 1)
    monkeypatch.setattr(int8, "_H3_LOW_PEAK_CHUNK_BYTES", 4 * shape[1] * 2)
    monkeypatch.setattr(int8, "_H3_LOW_PEAK_ROW_ALIGNMENT", 4)
    actual = int8.int8_linear(
        value,
        weight,
        weight_scale,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    )
    torch.xpu.synchronize()

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("input_kind", ["normal", "alternating_extremes"])
def test_public_low_peak_route_is_bit_exact(
    int8_native, monkeypatch, input_kind
):
    int8, _native = int8_native
    generator = torch.Generator(device="xpu").manual_seed(20260806)
    rows = 9
    if input_kind == "normal":
        value = torch.randn(
            rows,
            int8._H3_SWIGLU_INPUT_FEATURES,
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
        repeats = (
            rows * int8._H3_SWIGLU_INPUT_FEATURES
            + pattern.numel()
            - 1
        ) // pattern.numel()
        value = pattern.repeat(repeats)[
            : rows * int8._H3_SWIGLU_INPUT_FEATURES
        ].reshape(rows, int8._H3_SWIGLU_INPUT_FEATURES)
    weight = torch.randint(
        -127,
        128,
        (int8._H3_FFN_DOWN_FEATURES, int8._H3_SWIGLU_OUTPUT_FEATURES),
        device="xpu",
        dtype=torch.int8,
        generator=generator,
    )
    weight_scale = torch.rand(
        int8._H3_FFN_DOWN_FEATURES,
        device="xpu",
        dtype=torch.float32,
        generator=generator,
    ).add_(0.01)

    monkeypatch.setattr(int8, "_H3_LOW_PEAK_MIN_ACTIVATION_BYTES", 1 << 60)
    expected = int8.int8_linear(
        value,
        weight,
        weight_scale,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
        input_act="swiglu",
    )

    monkeypatch.setattr(int8, "_H3_LOW_PEAK_MIN_ACTIVATION_BYTES", 1)
    monkeypatch.setattr(int8, "_H3_LOW_PEAK_CHUNK_BYTES", 4 * 14336 * 2)
    monkeypatch.setattr(int8, "_H3_LOW_PEAK_ROW_ALIGNMENT", 4)
    actual = int8.int8_linear(
        value,
        weight,
        weight_scale,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
        input_act="swiglu",
    )
    torch.xpu.synchronize()

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
