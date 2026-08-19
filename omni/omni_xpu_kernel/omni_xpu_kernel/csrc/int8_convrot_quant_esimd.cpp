#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <tuple>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {
namespace {

#ifndef OMNI_CONVROT_QUANT_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_CONVROT_QUANT_WG_SIZE 1
#elif defined(OMNI_XPU_ARCH_BMG)
// The fused path remains disabled on BMG until it is measured there.
#define OMNI_CONVROT_QUANT_WG_SIZE 8
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<int Stride, int GroupSize>
inline void radix4_hadamard_stage(simd<float, GroupSize>& values) {
#pragma unroll
    for (int base = 0; base < GroupSize; base += 4 * Stride) {
        simd<float, Stride> a = values.template select<Stride, 1>(base);
        simd<float, Stride> b =
            values.template select<Stride, 1>(base + Stride);
        simd<float, Stride> c =
            values.template select<Stride, 1>(base + 2 * Stride);
        simd<float, Stride> d =
            values.template select<Stride, 1>(base + 3 * Stride);
        values.template select<Stride, 1>(base) = a + b + c - d;
        values.template select<Stride, 1>(base + Stride) = a + b - c + d;
        values.template select<Stride, 1>(base + 2 * Stride) = a - b + c + d;
        values.template select<Stride, 1>(base + 3 * Stride) = -a + b + c + d;
    }
    if constexpr (Stride * 4 < GroupSize) {
        radix4_hadamard_stage<Stride * 4, GroupSize>(values);
    }
}

template<typename InputT, int GroupSize>
inline simd<float, GroupSize> load_rotated(
    const InputT* __restrict__ input) {
    simd<InputT, GroupSize> source = block_load<InputT, GroupSize>(input);
    simd<float, GroupSize> values = source;
    radix4_hadamard_stage<1, GroupSize>(values);
    values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));

    // The composed path materializes the matmul result in the input dtype
    // before rowwise quantization. Preserve that rounding boundary exactly.
    simd<InputT, GroupSize> rounded = values;
    simd<float, GroupSize> result = rounded;
    return result;
}

template<typename InputT, int GroupSize>
void quantize_convrot_kernel(
    const InputT* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    constexpr int WorkGroupSize = OMNI_CONVROT_QUANT_WG_SIZE;
    const int64_t padded =
        (rows + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t row = item.get_global_id(0);
                if (row >= rows) return;
                const int64_t row_start = row * groups * GroupSize;

                float row_max = 0.0f;
                for (int64_t group = 0; group < groups; ++group) {
                    simd<float, GroupSize> values =
                        load_rotated<InputT, GroupSize>(
                            input + row_start + group * GroupSize);
                    simd<float, GroupSize> magnitudes =
                        sycl::ext::intel::esimd::abs<float, GroupSize>(values);
                    const float group_max = hmax<float>(magnitudes);
                    row_max = group_max > row_max ? group_max : row_max;
                }

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                const float inv_scale = 1.0f / scale;
                scales[row] = scale;

                simd<float, GroupSize> clamp_lo(-128.0f);
                simd<float, GroupSize> clamp_hi(127.0f);
                for (int64_t group = 0; group < groups; ++group) {
                    const int64_t offset = row_start + group * GroupSize;
                    simd<float, GroupSize> values =
                        load_rotated<InputT, GroupSize>(input + offset);
                    simd<float, GroupSize> quantized =
                        sycl::ext::intel::esimd::rnde<float, GroupSize>(
                            values * inv_scale);
                    quantized = sycl::ext::intel::esimd::max<float, GroupSize>(
                        sycl::ext::intel::esimd::min<float, GroupSize>(
                            quantized, clamp_hi),
                        clamp_lo);
                    simd<int8_t, GroupSize> packed = quantized;
                    block_store<int8_t, GroupSize>(output + offset, packed);
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_quantize_fused");
}

template<typename InputT>
void dispatch_group(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& scales,
    int64_t group_size) {
    const int64_t rows = input.size(0);
    const int64_t groups = input.size(1) / group_size;
    const auto* input_ptr = reinterpret_cast<const InputT*>(input.data_ptr());
    auto* output_ptr = reinterpret_cast<int8_t*>(output.data_ptr());
    if (group_size == 64) {
        quantize_convrot_kernel<InputT, 64>(
            input_ptr, output_ptr, scales.data_ptr<float>(), rows, groups,
            input.device());
    } else {
        quantize_convrot_kernel<InputT, 256>(
            input_ptr, output_ptr, scales.data_ptr<float>(), rows, groups,
            input.device());
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused(
    torch::Tensor input,
    int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 ||
            input.scalar_type() == torch::kFloat16,
        "input must be bf16 or fp16");
    TORCH_CHECK(
        group_size == 64 || group_size == 256,
        "fused ConvRot quantization supports group sizes 64 and 256");
    TORCH_CHECK(input.size(1) % group_size == 0, "invalid group size");
    input = input.contiguous();
    auto output = torch::empty_like(
        input, input.options().dtype(torch::kInt8));
    auto scales = torch::empty(
        {input.size(0), 1}, input.options().dtype(torch::kFloat));
    if (input.numel() == 0) return {output, scales};
    if (input.scalar_type() == torch::kBFloat16) {
        dispatch_group<bf16>(input, output, scales, group_size);
    } else {
        dispatch_group<fp16>(input, output, scales, group_size);
    }
    return {output, scales};
}

}  // namespace int8_ops
}  // namespace omni_xpu
