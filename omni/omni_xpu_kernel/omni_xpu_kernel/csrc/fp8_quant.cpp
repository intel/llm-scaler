#include <torch/extension.h>
#include <cmath>
#include <sycl/sycl.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace fp8 {

torch::Tensor dequantize_per_tensor_fused(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    torch::ScalarType out_dtype);

namespace {

#ifndef OMNI_FP8_QUANT_VEC
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_FP8_QUANT_VEC 16
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_FP8_QUANT_VEC 8
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

double fp8_max(torch::ScalarType dtype) {
    if (dtype == torch::kFloat8_e4m3fn) return 448.0;
    if (dtype == torch::kFloat8_e5m2) return 57344.0;
    TORCH_CHECK(false, "output dtype must be float8_e4m3fn or float8_e5m2");
}

void check_xpu(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_xpu(), name, " must be on XPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template<typename InputT>
void prepare_quantized_values(
    const InputT* __restrict__ input,
    const InputT* __restrict__ scale,
    InputT* __restrict__ output,
    int64_t numel,
    float limit,
    const at::Device& device) {
    constexpr int Vec = OMNI_FP8_QUANT_VEC;
    constexpr int WorkGroupSize = 256;
    const int64_t work_items = (numel + Vec - 1) / Vec;
    const int64_t padded =
        (work_items + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) {
                const int64_t first =
                    static_cast<int64_t>(item.get_global_linear_id()) * Vec;
                const InputT scale_value = scale[0];
                if (first + Vec <= numel) {
                    using InputVec = sycl::vec<InputT, Vec>;
                    const InputVec values =
                        *reinterpret_cast<const InputVec*>(input + first);
                    InputVec prepared;
#pragma unroll
                    for (int lane = 0; lane < Vec; ++lane) {
                        const InputT scaled = static_cast<InputT>(
                            static_cast<float>(values[lane]) /
                            static_cast<float>(scale_value));
                        const float value = static_cast<float>(scaled);
                        if (sycl::isnan(value)) {
                            if constexpr (std::is_same_v<InputT, bf16>) {
                                // XPU BF16 division canonicalizes NaN to the
                                // negative all-ones representation.
                                prepared[lane] = sycl::bit_cast<bf16>(
                                    uint16_t(0xffff));
                            } else {
                                prepared[lane] = scaled;
                            }
                        } else {
                            prepared[lane] = static_cast<InputT>(sycl::fmax(
                                -limit, sycl::fmin(limit, value)));
                        }
                    }
                    *reinterpret_cast<InputVec*>(output + first) = prepared;
                } else {
                    for (int64_t index = first; index < numel; ++index) {
                        const InputT scaled = static_cast<InputT>(
                            static_cast<float>(input[index]) /
                            static_cast<float>(scale_value));
                        const float value = static_cast<float>(scaled);
                        if (sycl::isnan(value)) {
                            if constexpr (std::is_same_v<InputT, bf16>) {
                                output[index] = sycl::bit_cast<bf16>(
                                    uint16_t(0xffff));
                            } else {
                                output[index] = scaled;
                            }
                        } else {
                            output[index] = static_cast<InputT>(sycl::fmax(
                                -limit, sycl::fmin(limit, value)));
                        }
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "fp8_quant_prepare");
}

template<typename InputT>
torch::Tensor quantize_per_tensor_fused(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    torch::ScalarType out_dtype,
    float limit) {
    auto scale_typed = scale.to(input.scalar_type()).contiguous();
    auto prepared = torch::empty_like(input);
    if (input.numel() != 0) {
        prepare_quantized_values(
            reinterpret_cast<const InputT*>(input.data_ptr()),
            reinterpret_cast<const InputT*>(scale_typed.data_ptr()),
            reinterpret_cast<InputT*>(prepared.data_ptr()),
            input.numel(), limit, input.device());
    }
    return prepared.to(out_dtype);
}

template<int ExponentBits, int MantissaBits, int ExponentBias>
torch::Tensor stochastic_rounding_fused(
    const torch::Tensor& input,
    const torch::Tensor& rng,
    torch::ScalarType out_dtype,
    double limit) {
    // Preserve the original implementation's FP16 arithmetic contract.  The
    // old path expressed every operation as an individual ATen kernel; doing
    // the same arithmetic in one SYCL kernel avoids materializing all of those
    // intermediate tensors while leaving the final FP8 conversion to PyTorch.
    auto x = input.to(torch::kHalf);
    auto rounded = torch::empty_like(x);
    const int64_t numel = x.numel();
    if (numel == 0) return rounded.to(out_dtype);

    const auto* x_ptr = reinterpret_cast<const fp16*>(x.data_ptr<at::Half>());
    const auto* rng_ptr = rng.data_ptr<uint8_t>();
    auto* out_ptr = reinterpret_cast<fp16*>(rounded.data_ptr<at::Half>());
    const fp16 mantissa_levels = static_cast<fp16>(1 << MantissaBits);
    const float denorm_divisor =
        static_cast<float>(std::exp2(-ExponentBias + 1 - MantissaBits));
    const fp16 denorm_base = static_cast<fp16>(std::exp2(-ExponentBias + 1));
    const fp16 limit_h = static_cast<fp16>(limit);

    constexpr int ElementsPerWorkItem = 8;
    const int64_t work_items =
        (numel + ElementsPerWorkItem - 1) / ElementsPerWorkItem;
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(sycl::range<1>(work_items), [=](sycl::id<1> item) {
            const int64_t first =
                static_cast<int64_t>(item[0]) * ElementsPerWorkItem;
#pragma unroll
            for (int lane = 0; lane < ElementsPerWorkItem; ++lane) {
                const int64_t index = first + lane;
                if (index >= numel) break;

                const fp16 value = x_ptr[index];
                const uint16_t value_bits = sycl::bit_cast<uint16_t>(value);
                const uint16_t abs_bits = value_bits & uint16_t(0x7fff);
                if (abs_bits == 0) {
                    // torch.sign maps both +0 and -0 to +0 in the old path.
                    out_ptr[index] = fp16(0);
                    continue;
                }
                if (abs_bits > uint16_t(0x7c00)) {
                    // torch.sign returns +0 for FP16 NaNs on XPU, and the old
                    // composite expression consequently produces +qNaN.
                    out_ptr[index] = sycl::bit_cast<fp16>(uint16_t(0x7e00));
                    continue;
                }

                const fp16 abs_value = sycl::fabs(value);
                const fp16 sign = (value_bits & uint16_t(0x8000))
                    ? fp16(-1)
                    : fp16(1);

                // For a finite FP16 value, floor(log2(abs(value))) is exactly
                // recoverable from the exponent/significand bits.  This avoids
                // device log2 approximation errors immediately below powers of
                // two and is materially cheaper than a transcendental.
                const int half_exponent = (abs_bits >> 10) & 0x1f;
                int floor_log2;
                if (half_exponent != 0) {
                    floor_log2 = half_exponent - 15;
                    const int mantissa = abs_bits & uint16_t(0x03ff);
                    // The composite path stores log2 in FP16 before floor().
                    // These are the exact mantissa thresholds where that FP16
                    // rounding reaches the next integer exponent.
                    int round_up_threshold = 1024;
                    if (half_exponent <= 6 || half_exponent >= 23) {
                        round_up_threshold = 1019;
                    } else if ((half_exponent >= 7 && half_exponent <= 10) ||
                               (half_exponent >= 19 && half_exponent <= 22)) {
                        round_up_threshold = 1022;
                    } else if ((half_exponent >= 11 && half_exponent <= 12) ||
                               (half_exponent >= 17 && half_exponent <= 18)) {
                        round_up_threshold = 1023;
                    }
                    if (mantissa >= round_up_threshold) ++floor_log2;
                } else {
                    const uint16_t significand = abs_bits & uint16_t(0x03ff);
                    int highest_bit = 0;
#pragma unroll
                    for (int bit = 1; bit < 10; ++bit) {
                        if (significand & (uint16_t(1) << bit)) highest_bit = bit;
                    }
                    floor_log2 = highest_bit - 24;
                    if (significand == 255 || significand == 511 ||
                        significand >= 1022) {
                        ++floor_log2;
                    }
                }
                int exponent_i = floor_log2 + ExponentBias;
                exponent_i = sycl::max(0, sycl::min(
                    exponent_i, (1 << ExponentBits) - 1));
                if constexpr (ExponentBits == 5) {
                    if (exponent_i == 31) {
                        // exp2(16) and the following arithmetic produce the
                        // same canonical -qNaN in the composite FP16 path.
                        out_ptr[index] = sycl::bit_cast<fp16>(uint16_t(0xfe00));
                        continue;
                    }
                }
                const bool normal = exponent_i != 0;

                // Build exp2(exponent - bias) exactly as an FP16 power of two.
                const int power = exponent_i - ExponentBias;
                uint16_t base_bits;
                if (power > 15) {
                    base_bits = uint16_t(0x7c00);
                } else if (power >= -14) {
                    base_bits = static_cast<uint16_t>((power + 15) << 10);
                } else if (power >= -24) {
                    base_bits = static_cast<uint16_t>(uint16_t(1) << (power + 24));
                } else {
                    base_bits = 0;
                }
                const fp16 normal_base = sycl::bit_cast<fp16>(base_bits);
                const fp16 mantissa_scaled = normal
                    ? static_cast<fp16>(
                          (abs_value / normal_base - fp16(1)) * mantissa_levels)
                    // Compute in FP32 before the explicit FP16 rounding.  The
                    // E5M2 divisor is an FP16 subnormal (2^-16), which device
                    // fast-math division may otherwise flush to zero.
                    : static_cast<fp16>(
                          static_cast<float>(abs_value) / denorm_divisor);
                const fp16 random_fraction =
                    static_cast<fp16>(static_cast<fp16>(rng_ptr[index]) / fp16(256));
                const fp16 mantissa = static_cast<fp16>(
                    sycl::floor(
                        static_cast<fp16>(mantissa_scaled + random_fraction)) /
                    mantissa_levels);
                const fp16 magnitude = normal
                    ? static_cast<fp16>(
                          normal_base * static_cast<fp16>(fp16(1) + mantissa))
                    : static_cast<fp16>(denorm_base * mantissa);
                fp16 result = static_cast<fp16>(sign * magnitude);
                result = sycl::clamp(result, -limit_h, limit_h);
                out_ptr[index] = result;
            }
        });
    };
    utils::submit_kernel(cgf, input.device(), "fp8_stochastic_rounding_fused");
    return rounded.to(out_dtype);
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
    if (input.scalar_type() == torch::kBFloat16) {
        return quantize_per_tensor_fused<bf16>(
            input, scale, out_dtype, static_cast<float>(limit));
    }
    if (input.scalar_type() == torch::kFloat) {
        return quantize_per_tensor_fused<float>(
            input, scale, out_dtype, static_cast<float>(limit));
    }
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
    if (input.scalar_type() == torch::kFloat8_e4m3fn ||
        input.scalar_type() == torch::kFloat8_e5m2) {
        return dequantize_per_tensor_fused(input, scale, out_dtype);
    }
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

    if (out_dtype == torch::kFloat8_e4m3fn) {
        return stochastic_rounding_fused<4, 3, 7>(
            input, rng, out_dtype, fp8_max(out_dtype));
    } else if (out_dtype == torch::kFloat8_e5m2) {
        return stochastic_rounding_fused<5, 2, 15>(
            input, rng, out_dtype, fp8_max(out_dtype));
    } else {
        TORCH_CHECK(false, "output dtype must be float8_e4m3fn or float8_e5m2");
    }
}

}  // namespace fp8
}  // namespace omni_xpu
