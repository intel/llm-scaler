#include <torch/extension.h>
#include <cmath>

namespace omni_xpu {
namespace fp8 {

namespace {

double fp8_max(torch::ScalarType dtype) {
    if (dtype == torch::kFloat8_e4m3fn) return 448.0;
    if (dtype == torch::kFloat8_e5m2) return 57344.0;
    TORCH_CHECK(false, "output dtype must be float8_e4m3fn or float8_e5m2");
}

void check_xpu(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_xpu(), name, " must be on XPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor quantize_per_tensor(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    torch::ScalarType out_dtype) {
    check_xpu(input, "input");
    check_xpu(scale, "scale");
    TORCH_CHECK(scale.numel() == 1, "scale must contain one element");
    const double limit = fp8_max(out_dtype);
    auto scaled = input / scale.to(input.scalar_type());
    return torch::clamp(scaled, -limit, limit).to(out_dtype);
}

torch::Tensor dequantize_per_tensor(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    torch::ScalarType out_dtype) {
    check_xpu(input, "input");
    check_xpu(scale, "scale");
    TORCH_CHECK(scale.numel() == 1, "scale must contain one element");
    TORCH_CHECK(
        out_dtype == torch::kFloat || out_dtype == torch::kHalf ||
            out_dtype == torch::kBFloat16,
        "output dtype must be float32, float16, or bfloat16");
    return input.to(out_dtype) * scale.to(out_dtype);
}

torch::Tensor stochastic_rounding(
    const torch::Tensor& input,
    const torch::Tensor& rng,
    torch::ScalarType out_dtype) {
    check_xpu(input, "input");
    check_xpu(rng, "rng");
    TORCH_CHECK(rng.scalar_type() == torch::kUInt8, "rng must be uint8");
    TORCH_CHECK(rng.sizes() == input.sizes(), "rng shape must match input");

    int exponent_bits;
    int mantissa_bits;
    int exponent_bias;
    if (out_dtype == torch::kFloat8_e4m3fn) {
        exponent_bits = 4;
        mantissa_bits = 3;
        exponent_bias = 7;
    } else if (out_dtype == torch::kFloat8_e5m2) {
        exponent_bits = 5;
        mantissa_bits = 2;
        exponent_bias = 15;
    } else {
        TORCH_CHECK(false, "output dtype must be float8_e4m3fn or float8_e5m2");
    }

    auto x = input.to(torch::kHalf);
    auto abs_x = torch::abs(x);
    auto sign = torch::where(abs_x == 0, torch::zeros_like(x), torch::sign(x));
    auto exponent = torch::clamp(
        torch::floor(torch::log2(abs_x)) + exponent_bias,
        0,
        (1 << exponent_bits) - 1);
    auto normal = exponent != 0;
    const double mantissa_levels = static_cast<double>(1 << mantissa_bits);
    auto normal_base = torch::exp2(exponent - exponent_bias);
    const double denorm_divisor = std::exp2(-exponent_bias + 1 - mantissa_bits);
    auto mantissa_scaled = torch::where(
        normal,
        (abs_x / normal_base - 1.0) * mantissa_levels,
        abs_x / denorm_divisor);
    auto mantissa = torch::floor(
        mantissa_scaled + rng.to(mantissa_scaled.scalar_type()) / 256.0) /
        mantissa_levels;
    auto magnitude = torch::where(
        normal,
        normal_base * (1.0 + mantissa),
        std::exp2(-exponent_bias + 1) * mantissa);
    auto result = sign * magnitude;
    const double limit = fp8_max(out_dtype);
    return torch::clamp(result, -limit, limit).to(out_dtype);
}

}  // namespace fp8
}  // namespace omni_xpu
