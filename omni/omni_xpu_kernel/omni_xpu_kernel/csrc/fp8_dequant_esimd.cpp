#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace fp8 {
namespace {

#ifndef OMNI_FP8_DEQUANT_ELEMENTS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_FP8_DEQUANT_ELEMENTS_PER_WI 256
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_FP8_DEQUANT_ELEMENTS_PER_WI 64
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<int MantissaBits, int Bias, bool FiniteOnly>
float decode_fp8(uint8_t code) {
    const int exponent =
        (code >> MantissaBits) & ((1 << (7 - MantissaBits)) - 1);
    const int mantissa = code & ((1 << MantissaBits) - 1);
    float value;
    if (exponent == 0) {
        constexpr int Power = 1 - Bias - MantissaBits;
        constexpr float Unit = sycl::bit_cast<float>(
            static_cast<uint32_t>((Power + 127) << 23));
        value = static_cast<float>(mantissa) * Unit;
    } else if constexpr (FiniteOnly) {
        if (exponent == 15 && mantissa == 7) {
            value = sycl::bit_cast<float>(uint32_t(0x7fc00000));
        } else {
            const uint32_t bits =
                static_cast<uint32_t>((exponent - Bias + 127) << 23) |
                static_cast<uint32_t>(mantissa << (23 - MantissaBits));
            value = sycl::bit_cast<float>(bits);
        }
    } else if (exponent == 31) {
        const uint32_t bits = mantissa == 0
            ? uint32_t(0x7f800000)
            : uint32_t(0x7fc00000);
        value = sycl::bit_cast<float>(bits);
    } else {
        const uint32_t bits =
            static_cast<uint32_t>((exponent - Bias + 127) << 23) |
            static_cast<uint32_t>(mantissa << (23 - MantissaBits));
        value = sycl::bit_cast<float>(bits);
    }
    return (code & 0x80) ? -value : value;
}

template<typename OutputT, int MantissaBits, int Bias, bool FiniteOnly>
void dequantize_kernel(
    const uint8_t* __restrict__ input,
    const float* __restrict__ scale,
    OutputT* __restrict__ output,
    int64_t numel,
    const at::Device& device) {
    constexpr int ElementsPerWorkItem = OMNI_FP8_DEQUANT_ELEMENTS_PER_WI;
    // The 256-lane PTL-H decode uses double GRF and admits at most 32 work
    // items in a work-group.
    constexpr int WorkGroupSize = 32;
    const int64_t work_items =
        (numel + ElementsPerWorkItem - 1) / ElementsPerWorkItem;
    const int64_t padded =
        (work_items + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t work_item = item.get_global_id(0);
                if (work_item >= work_items) return;
                const int64_t first = work_item * ElementsPerWorkItem;
                const int64_t remaining = numel - first;

                if (remaining >= ElementsPerWorkItem) {
                    simd<uint8_t, ElementsPerWorkItem> codes =
                        block_load<uint8_t, ElementsPerWorkItem>(input + first);
                    simd<uint32_t, ElementsPerWorkItem> raw = codes;
                    simd<uint32_t, ElementsPerWorkItem> sign =
                        (raw & 0x80u) << 24;
                    simd<uint32_t, ElementsPerWorkItem> exponent =
                        (raw >> MantissaBits) &
                        ((1u << (7 - MantissaBits)) - 1u);
                    simd<uint32_t, ElementsPerWorkItem> mantissa =
                        raw & ((1u << MantissaBits) - 1u);

                    simd<uint32_t, ElementsPerWorkItem> value_bits =
                        sign |
                        ((exponent + (127 - Bias)) << 23) |
                        (mantissa << (23 - MantissaBits));

                    constexpr int DenormPower = 1 - Bias - MantissaBits;
                    constexpr float DenormUnit = sycl::bit_cast<float>(
                        static_cast<uint32_t>((DenormPower + 127) << 23));
                    simd<float, ElementsPerWorkItem> denorm = mantissa;
                    denorm *= DenormUnit;
                    simd<uint32_t, ElementsPerWorkItem> denorm_bits =
                        denorm.template bit_cast_view<uint32_t>() | sign;
                    value_bits.merge(denorm_bits, exponent == 0u);

                    if constexpr (FiniteOnly) {
                        simd<uint32_t, ElementsPerWorkItem> nan_bits =
                            sign | 0x7fc00000u;
                        value_bits.merge(
                            nan_bits, (exponent == 15u) & (mantissa == 7u));
                    } else {
                        simd<uint32_t, ElementsPerWorkItem> special_bits =
                            sign | 0x7f800000u | (mantissa << 21);
                        value_bits.merge(special_bits, exponent == 31u);
                    }

                    simd<float, ElementsPerWorkItem> values =
                        value_bits.template bit_cast_view<float>();
                    const OutputT scale_cast = static_cast<OutputT>(scale[0]);
                    simd<OutputT, ElementsPerWorkItem> converted = values;
                    converted *= scale_cast;
                    block_store<OutputT, ElementsPerWorkItem>(
                        output + first, converted);
                } else {
                    const OutputT scale_cast = static_cast<OutputT>(scale[0]);
                    for (int64_t lane = 0; lane < remaining; ++lane) {
                        const float value = decode_fp8<
                            MantissaBits, Bias, FiniteOnly>(input[first + lane]);
                        const OutputT converted = static_cast<OutputT>(value);
                        output[first + lane] = static_cast<OutputT>(
                            static_cast<float>(converted) *
                            static_cast<float>(scale_cast));
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "fp8_dequantize_esimd");
}

template<typename OutputT>
void dispatch_format(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    torch::Tensor& output) {
    const auto* input_ptr =
        reinterpret_cast<const uint8_t*>(input.data_ptr());
    auto* output_ptr = reinterpret_cast<OutputT*>(output.data_ptr());
    if (input.scalar_type() == torch::kFloat8_e4m3fn) {
        dequantize_kernel<OutputT, 3, 7, true>(
            input_ptr, scale.data_ptr<float>(), output_ptr, input.numel(),
            input.device());
    } else {
        dequantize_kernel<OutputT, 2, 15, false>(
            input_ptr, scale.data_ptr<float>(), output_ptr, input.numel(),
            input.device());
    }
}

}  // namespace

torch::Tensor dequantize_per_tensor_fused(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    torch::ScalarType out_dtype) {
    auto output = torch::empty(input.sizes(), input.options().dtype(out_dtype));
    if (input.numel() == 0) return output;
    auto scale_f = scale.to(input.device(), torch::kFloat).contiguous();
    if (out_dtype == torch::kFloat) {
        dispatch_format<float>(input, scale_f, output);
    } else if (out_dtype == torch::kHalf) {
        dispatch_format<fp16>(input, scale_f, output);
    } else {
        dispatch_format<bf16>(input, scale_f, output);
    }
    return output;
}

}  // namespace fp8
}  // namespace omni_xpu
