#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {
namespace {

#ifndef OMNI_CONVROT_DEQUANT_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_CONVROT_DEQUANT_WG_SIZE 1
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_CONVROT_DEQUANT_WG_SIZE 8
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<int Stride, int GroupSize>
inline void radix4_hadamard_stage(simd<float, GroupSize>& values) {
#pragma unroll
    for (int base = 0; base < GroupSize; base += 4 * Stride) {
        simd<float, Stride> a =
            values.template select<Stride, 1>(base);
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

template<int GroupSize>
void dequantize_convrot_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    constexpr int WorkGroupSize = OMNI_CONVROT_DEQUANT_WG_SIZE;
    const int64_t work_items = rows * groups;
    const int64_t padded =
        (work_items + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t work_item = item.get_global_id(0);
                if (work_item >= work_items) return;
                const int64_t row = work_item / groups;
                const int64_t group = work_item % groups;
                const int64_t first =
                    row * groups * GroupSize + group * GroupSize;
                simd<int8_t, GroupSize> quantized =
                    block_load<int8_t, GroupSize>(input + first);
                simd<float, GroupSize> values = quantized;
                values *= scales[row];
                radix4_hadamard_stage<1, GroupSize>(values);
                values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));
                block_store<float, GroupSize>(output + first, values);
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_dequant_fused");
}

}  // namespace

torch::Tensor dequantize_int8_convrot_fused(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be int8");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(scale.numel() == input.size(0), "scale must be rowwise");
    input = input.contiguous();
    auto scale_f = scale.to(input.device(), torch::kFloat).contiguous();
    auto output =
        torch::empty(input.sizes(), input.options().dtype(torch::kFloat));
    if (input.numel() == 0) return output;
    const int64_t groups = input.size(1) / group_size;
    if (group_size == 64) {
        dequantize_convrot_kernel<64>(
            input.data_ptr<int8_t>(), scale_f.data_ptr<float>(),
            output.data_ptr<float>(), input.size(0), groups, input.device());
    } else {
        dequantize_convrot_kernel<256>(
            input.data_ptr<int8_t>(), scale_f.data_ptr<float>(),
            output.data_ptr<float>(), input.size(0), groups, input.device());
    }
    return output;
}

}  // namespace int8_ops
}  // namespace omni_xpu
