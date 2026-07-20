#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {
namespace {

#ifndef OMNI_INT8_DEQUANT_ELEMENTS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_INT8_DEQUANT_ELEMENTS_PER_WI 64
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_INT8_DEQUANT_ELEMENTS_PER_WI 32
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<typename OutputT>
void dequantize_esimd_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputT* __restrict__ output,
    int64_t numel,
    int64_t row_width,
    bool rowwise,
    const at::Device& device) {
    constexpr int ElementsPerWorkItem = OMNI_INT8_DEQUANT_ELEMENTS_PER_WI;
    constexpr int WorkGroupSize = 64;
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
                const float scale = scales[rowwise ? first / row_width : 0];

                if (remaining >= ElementsPerWorkItem) {
                    simd<int8_t, ElementsPerWorkItem> quantized =
                        block_load<int8_t, ElementsPerWorkItem>(input + first);
                    simd<float, ElementsPerWorkItem> values = quantized;
                    values *= scale;
                    if constexpr (std::is_same_v<OutputT, float>) {
                        block_store<float, ElementsPerWorkItem>(
                            output + first, values);
                    } else {
                        simd<OutputT, ElementsPerWorkItem> converted = values;
                        block_store<OutputT, ElementsPerWorkItem>(
                            output + first, converted);
                    }
                } else {
                    for (int64_t lane = 0; lane < remaining; ++lane) {
                        output[first + lane] = static_cast<OutputT>(
                            static_cast<float>(input[first + lane]) * scale);
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_dequantize_esimd");
}

}  // namespace

torch::Tensor dequantize_int8_fused(
    torch::Tensor input,
    torch::Tensor scale,
    torch::ScalarType out_dtype) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be int8");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(
        out_dtype == torch::kFloat || out_dtype == torch::kHalf ||
            out_dtype == torch::kBFloat16,
        "output dtype must be float32, float16, or bfloat16");

    input = input.contiguous();
    const int64_t row_width = input.size(-1);
    TORCH_CHECK(row_width > 0, "input last dimension must be non-empty");
    const int64_t rows = input.numel() / row_width;
    const bool rowwise = scale.dim() > 0 && scale.size(-1) == 1 &&
        scale.numel() == rows && scale.numel() != 1;
    constexpr int ElementsPerWorkItem = OMNI_INT8_DEQUANT_ELEMENTS_PER_WI;
    const bool supported = scale.numel() == 1 ||
        (rowwise && row_width % ElementsPerWorkItem == 0);
    if (!supported) {
        auto result = input.to(torch::kFloat32) *
            scale.to(input.device()).to(torch::kFloat32);
        return out_dtype == torch::kFloat ? result : result.to(out_dtype);
    }

    auto scales = scale.to(input.device(), torch::kFloat).contiguous();
    auto output = torch::empty(input.sizes(), input.options().dtype(out_dtype));
    if (input.numel() == 0) return output;

    const auto* input_ptr = input.data_ptr<int8_t>();
    const auto* scale_ptr = scales.data_ptr<float>();
    if (out_dtype == torch::kFloat) {
        dequantize_esimd_kernel(
            input_ptr, scale_ptr, output.data_ptr<float>(), input.numel(),
            row_width, rowwise, input.device());
    } else if (out_dtype == torch::kHalf) {
        dequantize_esimd_kernel(
            input_ptr, scale_ptr, reinterpret_cast<fp16*>(output.data_ptr()),
            input.numel(), row_width, rowwise, input.device());
    } else {
        dequantize_esimd_kernel(
            input_ptr, scale_ptr, reinterpret_cast<bf16*>(output.data_ptr()),
            input.numel(), row_width, rowwise, input.device());
    }
    return output;
}

}  // namespace int8_ops
}  // namespace omni_xpu
