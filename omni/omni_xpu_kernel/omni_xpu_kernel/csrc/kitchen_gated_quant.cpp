// SPDX-License-Identifier: Apache-2.0
// Kitchen gated activation, optional ConvRot, and rowwise INT8 quantization.
#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/math.hpp>

#include <string>
#include <tuple>

#include "utils.h"

namespace omni_xpu::int8_ops {
namespace {

// One FP32 row stays in SLM. Larger rows use the existing composed path.
constexpr int64_t kMaxWidth = 8192;
constexpr int kThreads = 256;

inline float stable_tanh(float x) {
    const float t = sycl::expm1(-2.0f * sycl::fabs(x));
    return sycl::copysign(-t / (t + 2.0f), x);
}

template <typename InputT, bool Situ>
void launch(const torch::Tensor& input, torch::Tensor& output,
            torch::Tensor& scales, int group_size) {
    const int64_t width = input.size(-1) / 2;
    const int64_t rows = input.numel() / (2 * width);
    const auto* source = reinterpret_cast<const InputT*>(input.data_ptr());
    auto* codes = output.data_ptr<int8_t>();
    auto* scale_out = scales.data_ptr<float>();
    auto cgf = [&](sycl::handler& handle) {
        sycl::local_accessor<float, 1> values(sycl::range<1>(width), handle);
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * kThreads),
                              sycl::range<1>(kThreads)),
            [=](sycl::nd_item<1> item) {
                const int lane = item.get_local_id(0);
                const int64_t row = item.get_group(0);
                const int64_t offset = row * 2 * width;
                for (int64_t col = lane; col < width; col += kThreads) {
                    const float first = static_cast<float>(source[offset + col]);
                    const float second = static_cast<float>(source[offset + width + col]);
                    float value;
                    if constexpr (Situ) {
                        // SiTU uses [hidden | gate]; SwiGLU uses [gate | up].
                        const float gate = 4.0f * stable_tanh(second / 4.0f);
                        const float hidden = 25.0f * stable_tanh(first / 25.0f);
                        value = (gate / (1.0f + sycl::exp(-second))) * hidden;
                    } else {
                        value = (first / (1.0f + sycl::exp(-first))) * second;
                    }
                    values[col] = value;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Each work-item owns all four outputs of its butterfly, so
                // the in-place stage has disjoint reads/writes until its fence.
                for (int stride = 1; stride < group_size; stride *= 4) {
                    for (int64_t pair = lane; pair < width / 4; pair += kThreads) {
                        const int64_t pos = (pair / stride) * 4 * stride + pair % stride;
                        const float a = values[pos];
                        const float b = values[pos + stride];
                        const float c = values[pos + 2 * stride];
                        const float d = values[pos + 3 * stride];
                        values[pos] = a + b + c - d;
                        values[pos + stride] = a + b - c + d;
                        values[pos + 2 * stride] = a - b + c + d;
                        values[pos + 3 * stride] = -a + b + c + d;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float normalizer = group_size ? sycl::rsqrt(float(group_size)) : 1.0f;
                float maximum = 0.0f;
                for (int64_t col = lane; col < width; col += kThreads) {
                    values[col] *= normalizer;
                    maximum = sycl::fmax(maximum, sycl::fabs(values[col]));
                }
                const float row_max = sycl::reduce_over_group(
                    item.get_group(), maximum, sycl::maximum<float>());
                const float scale = sycl::fmax(row_max / 127.0f, 1e-30f);
                if (lane == 0) scale_out[row] = scale;
                for (int64_t col = lane; col < width; col += kThreads) {
                    const float value = sycl::ext::intel::math::fdiv_rn(values[col], scale);
                    codes[row * width + col] = static_cast<int8_t>(
                        sycl::clamp(sycl::rint(value), -128.0f, 127.0f));
                }
            });
    };
    utils::submit_kernel(cgf, input.device(), "kitchen_gated_quantize_int8");
}

template <typename InputT>
void dispatch(const torch::Tensor& input, torch::Tensor& output,
              torch::Tensor& scales, const std::string& act, int group_size) {
    if (act == "situ") launch<InputT, true>(input, output, scales, group_size);
    else launch<InputT, false>(input, output, scales, group_size);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> gated_quantize_int8(
    torch::Tensor input, const std::string& input_act, int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu() && input.dim() >= 1 && input.is_contiguous(),
                "input must be contiguous XPU [...,2K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 ||
                    input.scalar_type() == torch::kFloat16,
                "input must be BF16 or FP16");
    TORCH_CHECK(input.size(-1) > 0 && input.size(-1) % 2 == 0 &&
                    input.size(-1) / 2 <= kMaxWidth,
                "gated quantization requires 0<K<=8192 and an even raw width");
    TORCH_CHECK(input_act == "situ" || input_act == "swiglu",
                "input_act must be situ or swiglu");
    const int64_t width = input.size(-1) / 2;
    TORCH_CHECK(group_size == 0 || group_size == 64 || group_size == 256,
                "ConvRot group size must be 0, 64 or 256");
    TORCH_CHECK(group_size == 0 || width % group_size == 0,
                "ConvRot group size must divide the activated width");
    auto shape = input.sizes().vec();
    shape.back() = width;
    auto output = torch::empty(shape, input.options().dtype(torch::kInt8));
    shape.back() = 1;
    auto scales = torch::empty(shape, input.options().dtype(torch::kFloat));
    if (input.numel() == 0) return {output, scales};
    if (input.scalar_type() == torch::kBFloat16) {
        dispatch<sycl::ext::oneapi::bfloat16>(input, output, scales, input_act, group_size);
    } else {
        dispatch<sycl::half>(input, output, scales, input_act, group_size);
    }
    return {output, scales};
}

}  // namespace omni_xpu::int8_ops
