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
