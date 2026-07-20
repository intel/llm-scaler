#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace int8_ops {
namespace {

#ifndef OMNI_INT8_TENSORWISE_VEC
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_INT8_TENSORWISE_VEC 16
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_INT8_TENSORWISE_VEC 8
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<typename InputT>
void tensorwise_absmax_kernel(
    const InputT* __restrict__ input,
    float* __restrict__ absmax,
    int64_t numel,
    const at::Device& device) {
    constexpr int Vec = OMNI_INT8_TENSORWISE_VEC;
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
                uint32_t local_max_bits = 0;
                if (first + Vec <= numel) {
                    using InputVec = sycl::vec<InputT, Vec>;
                    const InputVec values =
                        *reinterpret_cast<const InputVec*>(input + first);
#pragma unroll
                    for (int lane = 0; lane < Vec; ++lane) {
                        const float value =
                            sycl::fabs(static_cast<float>(values[lane]));
                        local_max_bits = sycl::max(
                            local_max_bits, sycl::bit_cast<uint32_t>(value));
                    }
                } else {
                    for (int64_t index = first; index < numel; ++index) {
                        const float value =
                            sycl::fabs(static_cast<float>(input[index]));
                        local_max_bits = sycl::max(
                            local_max_bits, sycl::bit_cast<uint32_t>(value));
                    }
                }
                const uint32_t group_max_bits = sycl::reduce_over_group(
                    item.get_group(),
                    local_max_bits,
                    sycl::maximum<uint32_t>());
                if (item.get_local_linear_id() == 0) {
                    auto& bits = *reinterpret_cast<uint32_t*>(absmax);
                    sycl::atomic_ref<
                        uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> atomic(bits);
                    atomic.fetch_max(group_max_bits);
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_tensorwise_absmax");
}

template<typename InputT>
void tensorwise_quantize_kernel(
    const InputT* __restrict__ input,
    const float* __restrict__ scale_or_absmax,
    bool value_is_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scale_output,
    int64_t numel,
    const at::Device& device) {
    constexpr int Vec = OMNI_INT8_TENSORWISE_VEC;
    constexpr int WorkGroupSize = 256;
    const int64_t work_items = (numel + Vec - 1) / Vec;
    const int64_t padded =
        (work_items + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) {
                const int64_t work_item = item.get_global_linear_id();
                const float raw_scale = value_is_scale
                    ? scale_or_absmax[0]
                    : scale_or_absmax[0] / 127.0f;
                const float scale = value_is_scale || sycl::isnan(raw_scale)
                    ? raw_scale
                    : sycl::fmax(raw_scale, 1e-30f);
                if (work_item == 0) scale_output[0] = scale;
                const InputT scale_cast = static_cast<InputT>(scale);
                const int64_t first = work_item * Vec;
                if (first + Vec <= numel) {
                    using InputVec = sycl::vec<InputT, Vec>;
                    using OutputVec = sycl::vec<int8_t, Vec>;
                    const InputVec values =
                        *reinterpret_cast<const InputVec*>(input + first);
                    OutputVec quantized;
#pragma unroll
                    for (int lane = 0; lane < Vec; ++lane) {
                        const InputT scaled = static_cast<InputT>(
                            values[lane] / scale_cast);
                        float rounded = sycl::rint(static_cast<float>(scaled));
                        rounded = sycl::fmax(
                            -128.0f, sycl::fmin(127.0f, rounded));
                        quantized[lane] = static_cast<int8_t>(
                            static_cast<int32_t>(rounded));
                    }
                    *reinterpret_cast<OutputVec*>(output + first) = quantized;
                } else {
                    for (int64_t index = first; index < numel; ++index) {
                        const InputT scaled = static_cast<InputT>(
                            input[index] / scale_cast);
                        float rounded = sycl::rint(static_cast<float>(scaled));
                        rounded = sycl::fmax(
                            -128.0f, sycl::fmin(127.0f, rounded));
                        output[index] = static_cast<int8_t>(
                            static_cast<int32_t>(rounded));
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_tensorwise_quantize");
}

template<typename InputT>
void launch_tensorwise(
    const torch::Tensor& input,
    const torch::Tensor& scale_or_absmax,
    bool value_is_scale,
    torch::Tensor& output,
    torch::Tensor& scale_output) {
    const auto* input_ptr = reinterpret_cast<const InputT*>(input.data_ptr());
    if (!value_is_scale) {
        tensorwise_absmax_kernel(
            input_ptr,
            scale_or_absmax.data_ptr<float>(),
            input.numel(),
            input.device());
    }
    tensorwise_quantize_kernel(
        input_ptr,
        scale_or_absmax.data_ptr<float>(),
        value_is_scale,
        output.data_ptr<int8_t>(),
        scale_output.data_ptr<float>(),
        input.numel(),
        input.device());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_tensorwise_fused(
    torch::Tensor input,
    std::optional<torch::Tensor> scale_opt) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 ||
            input.scalar_type() == torch::kHalf ||
            input.scalar_type() == torch::kFloat,
        "input must be bf16, fp16, or fp32");
    TORCH_CHECK(input.numel() > 0, "input must be non-empty");
    input = input.contiguous();

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
    torch::Tensor value;
    torch::Tensor scale_output;
    const bool value_is_scale = scale_opt.has_value();
    if (value_is_scale) {
        TORCH_CHECK(scale_opt->numel() == 1, "scale must contain one element");
        value = scale_opt->to(input.device(), torch::kFloat).contiguous();
        scale_output = value;
    } else {
        value = torch::zeros({}, input.options().dtype(torch::kFloat));
        scale_output = torch::empty({}, input.options().dtype(torch::kFloat));
    }

    if (input.scalar_type() == torch::kBFloat16) {
        launch_tensorwise<bf16>(
            input, value, value_is_scale, output, scale_output);
    } else if (input.scalar_type() == torch::kHalf) {
        launch_tensorwise<fp16>(
            input, value, value_is_scale, output, scale_output);
    } else {
        launch_tensorwise<float>(
            input, value, value_is_scale, output, scale_output);
    }
    return {output, scale_output};
}

}  // namespace int8_ops
}  // namespace omni_xpu
