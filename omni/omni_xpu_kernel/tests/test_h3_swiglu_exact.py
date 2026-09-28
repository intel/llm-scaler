"""Focused tests for the BMG MiniMax H3 exact-order SwiGLU route."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


@pytest.fixture
def seed():
    torch.manual_seed(42)


def _native_or_skip(device):
    if device.type != "xpu":
        pytest.skip("native H3 SwiGLU kernel requires XPU")
    from omni_xpu_kernel import int8

    native = int8._get_native()
    if native is None or not hasattr(native, "fused_silu_mul_exact_bf16"):
        pytest.skip("native extension lacks the exact H3 SwiGLU kernel")
    return native


@pytest.mark.parametrize("columns", [257, 14336])
@pytest.mark.parametrize("scale", [1.0, 4.0, 32.0])
def test_chunked_strided_bf16_is_bit_exact(device, seed, columns, scale):
    native = _native_or_skip(device)
    combined = (
        torch.randn(3, columns * 2, device=device) * scale
    ).to(torch.bfloat16)
    gate, up = combined.chunk(2, dim=-1)

    expected = torch.nn.functional.silu(gate).mul_(up)
    actual = native.fused_silu_mul_exact_bf16(gate, up)

    assert gate.stride() == (columns * 2, 1)
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_full_finite_bf16_gate_domain_is_bit_exact(device):
    native = _native_or_skip(device)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16).to(device)
    up = torch.ones_like(gate)

    expected = torch.nn.functional.silu(gate).mul_(up)
    actual = native.fused_silu_mul_exact_bf16(gate, up)
    finite = torch.isfinite(expected)

    assert torch.equal(
        actual[finite].view(torch.int16), expected[finite].view(torch.int16)
    )
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))


def test_public_int8_linear_dispatches_structural_h3_route(
    device, seed, monkeypatch
):
    native = _native_or_skip(device)
    from omni_xpu_kernel import int8

    calls = {"silu": 0, "rotate": 0, "linear": 0}

    class NativeProxy:
        def fused_silu_mul_exact_bf16(self, gate, up):
            calls["silu"] += 1
            assert gate.shape == (3, 256)
            assert gate.stride() == (512, 1)
            return native.fused_silu_mul_exact_bf16(gate, up)

        def rotate_convrot(self, value, group_size):
            calls["rotate"] += 1
            assert value.shape == (3, 256)
            assert group_size == 256
            return value

        def int8_linear(self, value, weight, scale, bias, dtype_code, *_args):
            calls["linear"] += 1
            assert value.shape == (3, 256)
            assert dtype_code == 2
            return torch.empty(
                3, weight.shape[0], device=value.device, dtype=torch.bfloat16
            )

    x = torch.randn(3, 512, device=device, dtype=torch.bfloat16)
    weight = torch.empty(96, 256, device=device, dtype=torch.int8)
    scale = torch.ones(96, device=device, dtype=torch.float32)
    monkeypatch.setattr(int8, "_get_native", lambda: NativeProxy())
    monkeypatch.setattr(int8, "_is_supported_h3_swiglu_target", lambda: True)

    output = int8.int8_linear(
        x,
        weight,
        scale,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
        input_act="swiglu",
    )

    assert output.shape == (3, 96)
    assert calls == {"silu": 1, "rotate": 1, "linear": 1}


def test_public_int8_linear_fuses_3d_real_width_with_exact_bf16(
    device, seed, monkeypatch
):
    native = _native_or_skip(device)
    from omni_xpu_kernel import int8

    calls = {"silu": 0, "rotate": 0, "linear": 0}
    x = torch.randn(2, 3, 24576, device=device, dtype=torch.bfloat16)
    weight = torch.empty(4096, 12288, device=device, dtype=torch.int8)
    scale = torch.ones(4096, device=device, dtype=torch.float32)

    class NativeProxy:
        def fused_silu_mul_exact_bf16(self, gate, up):
            calls["silu"] += 1
            assert gate.shape == (6, 12288)
            assert gate.stride() == (24576, 1)
            assert gate.data_ptr() == x.data_ptr()
            expected = torch.nn.functional.silu(gate).mul_(up)
            actual = native.fused_silu_mul_exact_bf16(gate, up)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            return actual

        def rotate_convrot(self, value, group_size):
            calls["rotate"] += 1
            assert value.shape == (2, 3, 12288)
            assert value.dtype == torch.bfloat16
            assert group_size == 256
            return value

        def int8_linear(self, value, weight, _scale, bias, dtype_code, *_args):
            calls["linear"] += 1
            assert value.shape == (2, 3, 12288)
            assert bias is None
            assert dtype_code == 2
            return torch.empty(
                2, 3, weight.shape[0], device=value.device, dtype=torch.bfloat16
            )

    monkeypatch.setattr(int8, "_get_native", lambda: NativeProxy())
    monkeypatch.setattr(int8, "_is_supported_h3_swiglu_target", lambda: True)

    output = int8.int8_linear(
        x,
        weight,
        scale,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
        input_act="swiglu",
    )

    assert output.shape == (2, 3, 4096)
    assert calls == {"silu": 1, "rotate": 1, "linear": 1}


@pytest.mark.parametrize(
    ("case", "dtype", "groupsize"),
    [
        ("noncontiguous", torch.bfloat16, 256),
        ("non_bf16", torch.float16, 256),
        ("requires_grad", torch.bfloat16, 256),
        ("non_g256", torch.bfloat16, 128),
    ],
)
def test_public_int8_linear_keeps_eager_fallback_for_unsupported_3d_exact_fusion(
    device, case, dtype, groupsize, monkeypatch
):
    _native_or_skip(device)
    from omni_xpu_kernel import int8

    if case == "noncontiguous":
        x = torch.randn(2, 512, 3, device=device, dtype=dtype).transpose(1, 2)
        assert x.shape == (2, 3, 512) and not x.is_contiguous()
    else:
        x = torch.randn(2, 3, 512, device=device, dtype=dtype)
        if case == "requires_grad":
            x.requires_grad_()

    weight = torch.empty(96, 256, device=device, dtype=torch.int8)
    scale = torch.ones(96, device=device, dtype=torch.float32)
    calls = {"eager": 0, "silu": 0, "rotate": 0, "linear": 0}
    apply_input_act = int8._apply_input_act

    class NativeProxy:
        def fused_silu_mul_exact_bf16(self, _gate, _up):
            calls["silu"] += 1
            pytest.fail("unsupported 3D input reached exact SwiGLU fusion")

        def rotate_convrot(self, value, group_size):
            calls["rotate"] += 1
            assert group_size == groupsize
            return value

        def int8_linear(self, value, weight, _scale, _bias, _dtype_code, *_args):
            calls["linear"] += 1
            return torch.empty(
                *value.shape[:-1], weight.shape[0], device=value.device, dtype=dtype
            )

    def tracked_eager(value, input_act):
        calls["eager"] += 1
        return apply_input_act(value, input_act)

    monkeypatch.setattr(int8, "_get_native", lambda: NativeProxy())
    monkeypatch.setattr(int8, "_is_supported_h3_swiglu_target", lambda: True)
    monkeypatch.setattr(int8, "_apply_input_act", tracked_eager)

    output = int8.int8_linear(
        x,
        weight,
        scale,
        out_dtype=dtype,
        convrot=True,
        convrot_groupsize=groupsize,
        input_act="swiglu",
    )

    assert output.shape == (2, 3, 96)
    assert calls == {"eager": 1, "silu": 0, "rotate": 1, "linear": 1}
