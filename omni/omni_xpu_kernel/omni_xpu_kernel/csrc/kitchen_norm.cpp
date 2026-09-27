#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

namespace omni_xpu::kitchen {
namespace {

template <typename T>
class GroupNormFrameMomentsKernel;

template <typename T>
class GroupNormSiluPad3dKernel;

template <typename T>
torch::Tensor launch_group_norm_silu_pad3d(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& weight,
    const std::optional<torch::Tensor>& bias,
    int64_t groups,
    double eps,
    const std::vector<int64_t>& pad,
    bool silu,
    bool zero_pad,
    const std::optional<torch::Tensor>& out) {
    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t frames = input.size(2);
    const int64_t height = input.size(3);
    const int64_t width = input.size(4);
    const int64_t left = pad[0], right = pad[1];
    const int64_t top = pad[2], bottom = pad[3], front = pad[4];
    const int64_t output_frames = frames + front;
    const int64_t output_height = height + top + bottom;
    const int64_t output_width = width + left + right;
    const int64_t channels_per_group = channels / groups;
    const int64_t group_elements = channels_per_group * height * width;
    const int64_t stride_b = input.stride(0);
    const int64_t stride_c = input.stride(1);
    const int64_t stride_t = input.stride(2);
    const int64_t stride_h = input.stride(3);
    const int64_t stride_w = input.stride(4);
    auto output = out.has_value() ? *out : torch::empty(
        {batch, output_frames, output_height, output_width, channels},
        input.options()).permute({0, 4, 1, 2, 3});
    const int64_t output_stride_b = output.stride(0);
    const int64_t output_stride_c = output.stride(1);
    const int64_t output_stride_t = output.stride(2);
    const int64_t output_stride_h = output.stride(3);
    const int64_t output_stride_w = output.stride(4);
    auto moments = torch::empty(
        {batch * frames * groups, 2},
        input.options().dtype(torch::kFloat32));

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = weight.has_value() && weight->scalar_type() == input.scalar_type()
        ? static_cast<const T*>(weight->data_ptr()) : nullptr;
    const float* weight_f32 = weight.has_value() && weight->scalar_type() == at::kFloat
        ? weight->data_ptr<float>() : nullptr;
    const T* bias_ptr = bias.has_value() && bias->scalar_type() == input.scalar_type()
        ? static_cast<const T*>(bias->data_ptr()) : nullptr;
    const float* bias_f32 = bias.has_value() && bias->scalar_type() == at::kFloat
        ? bias->data_ptr<float>() : nullptr;
    float* moment_ptr = moments.data_ptr<float>();
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const bool normalize = weight.has_value();
    const float eps_f32 = static_cast<float>(eps);

    if (normalize) {
        auto moments_cgf = [&](sycl::handler& handler) {
            handler.parallel_for<GroupNormFrameMomentsKernel<T>>(
                sycl::range<1>(static_cast<size_t>(batch * frames * groups)),
                [=](sycl::id<1> item) {
                    const int64_t index = static_cast<int64_t>(item[0]);
                    const int64_t group = index % groups;
                    const int64_t frame = (index / groups) % frames;
                    const int64_t b = index / (groups * frames);
                    // Welford avoids cancellation for nearly constant FP32 groups.
                    float mean = 0.0f, squared_deviations = 0.0f;
                    int64_t samples = 0;
                    for (int64_t channel = 0; channel < channels_per_group;
                         ++channel) {
                        const int64_t c = group * channels_per_group + channel;
                        const int64_t base = b * stride_b + c * stride_c +
                                             frame * stride_t;
                        for (int64_t h = 0; h < height; ++h) {
                            for (int64_t w = 0; w < width; ++w) {
                                const float value = static_cast<float>(
                                    input_ptr[base + h * stride_h + w * stride_w]);
                                ++samples;
                                const float delta = value - mean;
                                mean += delta / static_cast<float>(samples);
                                squared_deviations += delta * (value - mean);
                            }
                        }
                    }
                    const float variance = sycl::fmax(
                        squared_deviations / static_cast<float>(group_elements),
                        0.0f);
                    moment_ptr[2 * index] = mean;
                    moment_ptr[2 * index + 1] =
                        1.0f / sycl::sqrt(variance + eps_f32);
                });
        };
        utils::submit_kernel(
            moments_cgf, input.device(), "kitchen_group_norm_moments");
    }

    const int64_t output_count = batch * output_frames * output_height *
                                 output_width * channels;
    auto output_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<GroupNormSiluPad3dKernel<T>>(
            sycl::range<1>(static_cast<size_t>(output_count)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t c = index % channels;
                const int64_t output_w = (index / channels) % output_width;
                const int64_t output_h =
                    (index / (channels * output_width)) % output_height;
                const int64_t output_t =
                    (index / (channels * output_width * output_height)) %
                    output_frames;
                const int64_t b = index / (channels * output_width *
                                           output_height * output_frames);
                const int64_t output_offset = b * output_stride_b +
                    c * output_stride_c + output_t * output_stride_t +
                    output_h * output_stride_h + output_w * output_stride_w;
                if (output_t < front ||
                    (zero_pad && (output_h < top || output_h >= top + height ||
                                  output_w < left || output_w >= left + width))) {
                    output_ptr[output_offset] = static_cast<T>(0.0f);
                    return;
                }
                int64_t input_h = output_h - top;
                int64_t input_w = output_w - left;
                if (input_h < 0) input_h = -input_h;
                if (input_h >= height) input_h = 2 * height - 2 - input_h;
                if (input_w < 0) input_w = -input_w;
                if (input_w >= width) input_w = 2 * width - 2 - input_w;
                const int64_t input_t = output_t - front;
                const int64_t input_offset = b * stride_b + c * stride_c +
                    input_t * stride_t + input_h * stride_h +
                    input_w * stride_w;
                float value = static_cast<float>(input_ptr[input_offset]);
                if (normalize) {
                    const int64_t group = c / channels_per_group;
                    const int64_t moment_index =
                        (b * frames + input_t) * groups + group;
                    value = (value - moment_ptr[2 * moment_index]) *
                            moment_ptr[2 * moment_index + 1];
                    const float gamma = weight_ptr
                        ? static_cast<float>(weight_ptr[c]) : weight_f32[c];
                    const float beta = bias_ptr
                        ? static_cast<float>(bias_ptr[c])
                        : (bias_f32 ? bias_f32[c] : 0.0f);
                    value = value * gamma + beta;
                    value = static_cast<float>(static_cast<T>(value));
                }
                if (silu) {
                    value = value / (1.0f + sycl::exp(-value));
                }
                output_ptr[output_offset] = static_cast<T>(value);
            });
    };
    utils::submit_kernel(output_cgf, input.device(),
                         "kitchen_group_norm_silu_pad3d");
    return output;
}

}  // namespace

torch::Tensor group_norm_silu_pad3d(
    torch::Tensor input,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> bias,
    int64_t groups,
    double eps,
    std::vector<int64_t> pad,
    bool silu,
    bool zero_pad,
    std::optional<torch::Tensor> out) {
    TORCH_CHECK(input.device().is_xpu() && input.dim() == 5,
                "input must be an XPU [B,C,T,H,W] tensor");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 &&
                    input.size(2) > 0 && input.size(3) > 0 && input.size(4) > 0,
                "input dimensions must be positive");
    TORCH_CHECK(groups > 0 && input.size(1) % groups == 0,
                "num_groups must divide channels");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0,
                "eps must be finite and nonnegative");
    TORCH_CHECK(pad.size() == 5, "pad must have five elements");
    for (const auto extent : pad) {
        TORCH_CHECK(extent >= 0, "padding must be non-negative");
    }
    if (!zero_pad) {
        TORCH_CHECK(pad[0] < input.size(4) && pad[1] < input.size(4) &&
                        pad[2] < input.size(3) && pad[3] < input.size(3),
                    "reflect padding must be smaller than spatial dimensions");
    }
    if (out.has_value()) {
        TORCH_CHECK(out->device() == input.device() &&
                        out->scalar_type() == input.scalar_type() &&
                        out->sizes() == at::IntArrayRef({
                            input.size(0), input.size(1), input.size(2) + pad[4],
                            input.size(3) + pad[2] + pad[3],
                            input.size(4) + pad[0] + pad[1]}),
                    "out must have the padded shape, input dtype and device");
    }
    if (weight.has_value()) {
        TORCH_CHECK(weight->device() == input.device() &&
                        weight->dim() == 1 && weight->numel() == input.size(1) &&
                        weight->is_contiguous() &&
                        (weight->scalar_type() == input.scalar_type() ||
                         weight->scalar_type() == at::kFloat),
                    "weight must be contiguous [C] in input or FP32 dtype");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == input.device() &&
                        bias->dim() == 1 && bias->numel() == input.size(1) &&
                        bias->is_contiguous() &&
                        (bias->scalar_type() == input.scalar_type() ||
                         bias->scalar_type() == at::kFloat),
                    "bias must be contiguous [C] in input or FP32 dtype");
    }
    switch (input.scalar_type()) {
        case at::kHalf:
            return launch_group_norm_silu_pad3d<sycl::half>(
                input, weight, bias, groups, eps, pad, silu, zero_pad, out);
        case at::kBFloat16:
            return launch_group_norm_silu_pad3d<sycl::ext::oneapi::bfloat16>(
                input, weight, bias, groups, eps, pad, silu, zero_pad, out);
        case at::kFloat:
            return launch_group_norm_silu_pad3d<float>(
                input, weight, bias, groups, eps, pad, silu, zero_pad, out);
        default:
            TORCH_CHECK(false,
                        "group_norm_silu_pad3d requires fp16, bf16 or fp32");
    }
}

}  // namespace omni_xpu::kitchen
