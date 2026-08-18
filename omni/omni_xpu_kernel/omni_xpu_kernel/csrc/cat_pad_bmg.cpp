// BMG fusion for an already-materialized temporal prefix, activation, and
// symmetric spatial zero padding.

#include <array>
#include <cstdint>

#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

using fp16 = sycl::half;

namespace omni_xpu {
namespace layout {
namespace {

constexpr int64_t kElementsPerWorkItem = 16;
constexpr int64_t kPrefixTemporal = 2;

template <int64_t Temporal>
class CatPadPrefixBMGKernel;

template <int64_t Temporal>
class CatPadActivationBMGKernel;

bool is_valid_shape(const torch::Tensor& input) {
    const std::array<int64_t, 5> shape = {
        input.size(0), input.size(1), input.size(2),
        input.size(3), input.size(4)};
    return shape == std::array<int64_t, 5>{1, 128, 4, 512, 512} ||
        shape == std::array<int64_t, 5>{1, 256, 4, 256, 256} ||
        shape == std::array<int64_t, 5>{1, 256, 4, 512, 512} ||
        shape == std::array<int64_t, 5>{1, 512, 4, 256, 256} ||
        shape == std::array<int64_t, 5>{1, 512, 2, 128, 128};
}

template <int64_t Temporal>
void launch_cat_pad_copy(
    const fp16* prefix_ptr,
    const fp16* input_ptr,
    fp16* output_ptr,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t spatial_pad,
    const torch::Device& device) {
    const int64_t spatial = height * width;
    const int64_t output_temporal = Temporal + kPrefixTemporal;
    const int64_t output_height = height + 2 * spatial_pad;
    const int64_t output_width = width + 2 * spatial_pad;
    const int64_t output_spatial = output_height * output_width;

    auto prefix_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<CatPadPrefixBMGKernel<Temporal>>(
            sycl::range<3>(
                static_cast<size_t>(channels * kPrefixTemporal),
                static_cast<size_t>(height),
                static_cast<size_t>(width / kElementsPerWorkItem)),
            [=](sycl::item<3> item) {
                const int64_t ct = item.get_id(0);
                const int64_t input_y = item.get_id(1);
                const int64_t input_x =
                    item.get_id(2) * kElementsPerWorkItem;
                const int64_t channel = ct / kPrefixTemporal;
                const int64_t output_t = ct - channel * kPrefixTemporal;
                const int64_t prefix_offset =
                    ct * spatial + input_y * width + input_x;
                const int64_t output_offset =
                    (channel * output_temporal + output_t) * output_spatial +
                    (input_y + spatial_pad) * output_width +
                    input_x + spatial_pad;
#pragma unroll
                for (int64_t element = 0;
                     element < kElementsPerWorkItem;
                     ++element) {
                    output_ptr[output_offset + element] =
                        prefix_ptr[prefix_offset + element];
                }
            });
    };
    omni_xpu::utils::submit_kernel(
        prefix_cgf, device, "cat_pad_prefix_bmg");

    auto input_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<CatPadActivationBMGKernel<Temporal>>(
            sycl::range<3>(
                static_cast<size_t>(channels * Temporal),
                static_cast<size_t>(height),
                static_cast<size_t>(width / kElementsPerWorkItem)),
            [=](sycl::item<3> item) {
                const int64_t ct = item.get_id(0);
                const int64_t input_y = item.get_id(1);
                const int64_t input_x =
                    item.get_id(2) * kElementsPerWorkItem;
                const int64_t channel = ct / Temporal;
                const int64_t input_t = ct - channel * Temporal;
                const int64_t input_offset =
                    channel * spatial + input_t * channels * spatial +
                    input_y * width + input_x;
                const int64_t output_offset =
                    (channel * output_temporal + input_t + kPrefixTemporal) *
                        output_spatial +
                    (input_y + spatial_pad) * output_width +
                    input_x + spatial_pad;
#pragma unroll
                for (int64_t element = 0;
                     element < kElementsPerWorkItem;
                     ++element) {
                    output_ptr[output_offset + element] =
                        input_ptr[input_offset + element];
                }
            });
    };
    omni_xpu::utils::submit_kernel(
        input_cgf, device, "cat_pad_activation_bmg");
}

}  // namespace

torch::Tensor cat_pad_bmg(
    torch::Tensor prefix,
    torch::Tensor input,
    int64_t spatial_pad) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(
        prefix.device() == input.device(),
        "prefix must be on the input device");
    TORCH_CHECK(
        !prefix.requires_grad() && !input.requires_grad(),
        "cat_pad_bmg is inference-only");
    TORCH_CHECK(input.dim() == 5, "input must be [1,C,T,H,W]");
    TORCH_CHECK(
        prefix.dim() == 5 && prefix.scalar_type() == torch::kFloat16 &&
            input.scalar_type() == torch::kFloat16,
        "prefix and input must be 5D FP16 tensors");
    TORCH_CHECK(spatial_pad == 1, "spatial_pad must be one");
    TORCH_CHECK(is_valid_shape(input), "unsupported BMG cat-pad shape");

    const int64_t channels = input.size(1);
    const int64_t temporal = input.size(2);
    const int64_t height = input.size(3);
    const int64_t width = input.size(4);
    const int64_t spatial = height * width;
    TORCH_CHECK(
        prefix.size(0) == 1 && prefix.size(1) == channels &&
            prefix.size(2) == kPrefixTemporal && prefix.size(3) == height &&
            prefix.size(4) == width && prefix.is_contiguous(),
        "prefix must be contiguous [1,C,2,H,W]");
    TORCH_CHECK(
        input.stride(4) == 1 && input.stride(3) == width &&
            input.stride(2) == channels * spatial &&
            input.stride(1) == spatial,
        "input must use the validated temporal-major backing layout");
    TORCH_CHECK(
        width % kElementsPerWorkItem == 0,
        "input width must be divisible by the copy granularity");

    const int64_t output_temporal = temporal + kPrefixTemporal;
    auto output = torch::zeros(
        {1, channels, output_temporal, height + 2 * spatial_pad,
         width + 2 * spatial_pad},
        input.options());
    const fp16* input_ptr =
        reinterpret_cast<const fp16*>(input.data_ptr());
    const fp16* prefix_ptr =
        reinterpret_cast<const fp16*>(prefix.data_ptr());
    fp16* output_ptr = reinterpret_cast<fp16*>(output.data_ptr());

    if (temporal == 4) {
        launch_cat_pad_copy<4>(
            prefix_ptr, input_ptr, output_ptr, channels, height, width,
            spatial_pad, input.device());
    } else {
        launch_cat_pad_copy<2>(
            prefix_ptr, input_ptr, output_ptr, channels, height, width,
            spatial_pad, input.device());
    }
    return output;
}

}  // namespace layout
}  // namespace omni_xpu
