import pytest
import torch

from omni_xpu_kernel import fp8


pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU unavailable")


def _stochastic_rounding_reference(input, rng, out_dtype):
    if out_dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    else:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15
    x = input.to(torch.float16)
    abs_x = torch.abs(x)
    sign = torch.where(abs_x == 0, torch.zeros_like(x), torch.sign(x))
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + exponent_bias,
        0,
        (1 << exponent_bits) - 1,
    )
    normal = exponent != 0
    levels = float(1 << mantissa_bits)
    normal_base = torch.exp2(exponent - exponent_bias)
    denorm_divisor = 2.0 ** (-exponent_bias + 1 - mantissa_bits)
    mantissa_scaled = torch.where(
        normal,
        (abs_x / normal_base - 1.0) * levels,
        abs_x / denorm_divisor,
    )
    mantissa = torch.floor(
        mantissa_scaled + rng.to(mantissa_scaled.dtype) / 256.0
    ) / levels
    magnitude = torch.where(
        normal,
        normal_base * (1.0 + mantissa),
        2.0 ** (-exponent_bias + 1) * mantissa,
    )
    limit = torch.finfo(out_dtype).max
    return torch.clamp(sign * magnitude, -limit, limit).to(out_dtype)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quantize_dequantize_matches_torch(input_dtype, fp8_dtype):
    x = torch.randn(17, 65, device="xpu", dtype=input_dtype) * 4
    scale = torch.tensor(0.125, device="xpu", dtype=torch.float32)
    actual = fp8.quantize_per_tensor(x, scale, fp8_dtype)
    limit = torch.finfo(fp8_dtype).max
    expected = torch.clamp(x / scale.to(input_dtype), -limit, limit).to(fp8_dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for out_dtype in (torch.float32, torch.float16, torch.bfloat16):
        restored = fp8.dequantize_per_tensor(actual, scale, out_dtype)
        reference = actual.to(out_dtype) * scale.to(out_dtype)
        assert torch.equal(restored, reference)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_value", [0.1, 0.375, 1.3, -0.375])
def test_quantize_fused_path_matches_torch(
    input_dtype, fp8_dtype, scale_value
):
    torch.manual_seed(20260721)
    random = torch.randn(4093, device="xpu", dtype=input_dtype) * 100
    special = torch.tensor(
        [
            0.0,
            -0.0,
            448.0,
            -448.0,
            57344.0,
            -57344.0,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device="xpu",
        dtype=input_dtype,
    )
    x = torch.cat((random, special))
    scale = torch.tensor(scale_value, device="xpu", dtype=torch.float32)
    actual = fp8.quantize_per_tensor(x, scale, fp8_dtype)
    limit = torch.finfo(fp8_dtype).max
    expected = torch.clamp(
        x / scale.to(input_dtype), -limit, limit
    ).to(fp8_dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dequantize_covers_all_fp8_encodings(fp8_dtype, out_dtype):
    # Repeat past the vector width and leave a tail so both paths are covered.
    codes = torch.arange(256, dtype=torch.uint8).repeat(3)[:519].to("xpu")
    x = codes.view(fp8_dtype)
    scale = torch.tensor(1.25, device="xpu", dtype=torch.float32)
    actual = fp8.dequantize_per_tensor(x, scale, out_dtype)
    expected = x.to(out_dtype) * scale.to(out_dtype)

    finite = torch.isfinite(expected)
    assert torch.equal(
        actual[finite].contiguous().view(torch.uint8),
        expected[finite].contiguous().view(torch.uint8),
    )
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_stochastic_rounding_is_deterministic_for_rng(fp8_dtype):
    x = torch.randn(32, 64, device="xpu", dtype=torch.float32)
    rng = torch.randint(0, 256, x.shape, device="xpu", dtype=torch.uint8)
    first = fp8.stochastic_rounding(x, rng, fp8_dtype)
    second = fp8.stochastic_rounding(x, rng, fp8_dtype)
    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_stochastic_rounding_matches_composite_reference(input_dtype, fp8_dtype):
    torch.manual_seed(20260721)
    random = torch.randn(32, 65, device="xpu", dtype=input_dtype) * 100
    # Include zeros, subnormals, infinities/NaNs, and values immediately below
    # powers of two where approximate device log2 implementations can differ.
    half_bits = torch.tensor(
        [
            0x0000,
            -0x8000,
            0x0001,
            0x03FF,
            0x0400,
            0x27FE,
            0x27FF,
            0x2BFE,
            0x2BFF,
            0x7BFF,
            0x7C00,
            0x7C01,
            -0x0400,
            -0x7C00,
            -0x7C01,
        ],
        dtype=torch.int16,
    ).view(torch.float16).to(device="xpu", dtype=input_dtype)
    x = torch.cat((random.flatten(), half_bits))
    rng = (torch.arange(x.numel(), device="xpu") % 256).to(torch.uint8)

    actual = fp8.stochastic_rounding(x, rng, fp8_dtype)
    expected = _stochastic_rounding_reference(x, rng, fp8_dtype)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
