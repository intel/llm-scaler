#include <cstdint>
#include <optional>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

namespace omni_xpu::kitchen {
namespace {

class KitchenFp16Conv3dScalarKernel;
class KitchenFp16Conv3dEpilogueKernel;

struct ConvShape {
    int64_t batch, channels, frames, height, width;
    int64_t output_channels, kernel_frames, kernel_height, kernel_width;
    int64_t output_frames, output_height, output_width;
    int64_t stride_frames, stride_height, stride_width;
    int64_t input_strides[5];
    int64_t output_strides[5];
    int64_t convolution_strides[5];
};

void launch_scalar(
    const sycl::half* input,
    const sycl::half* weight,
    const sycl::half* bias,
    const sycl::half* residual,
    sycl::half* output,
    ConvShape shape,
    const at::Device& device) {
    const int64_t count = shape.batch * shape.output_frames *
        shape.output_height * shape.output_width * shape.output_channels;
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenFp16Conv3dScalarKernel>(
            sycl::range<1>(static_cast<size_t>(count)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t out_c = index % shape.output_channels;
                const int64_t out_w = (index / shape.output_channels) % shape.output_width;
                const int64_t out_h = (index / (shape.output_channels * shape.output_width)) %
                    shape.output_height;
                const int64_t out_t = (index / (shape.output_channels * shape.output_width *
                    shape.output_height)) % shape.output_frames;
                const int64_t b = index / (shape.output_channels * shape.output_width *
                    shape.output_height * shape.output_frames);
                float sum = 0.0f;
                for (int64_t in_c = 0; in_c < shape.channels; ++in_c) {
                    for (int64_t kt = 0; kt < shape.kernel_frames; ++kt) {
                        const int64_t in_t = out_t * shape.stride_frames + kt;
                        for (int64_t kh = 0; kh < shape.kernel_height; ++kh) {
                            const int64_t in_h = out_h * shape.stride_height + kh;
                            for (int64_t kw = 0; kw < shape.kernel_width; ++kw) {
                                const int64_t in_w = out_w * shape.stride_width + kw;
                                const int64_t input_offset =
                                    b * shape.input_strides[0] +
                                    in_c * shape.input_strides[1] +
                                    in_t * shape.input_strides[2] +
                                    in_h * shape.input_strides[3] +
                                    in_w * shape.input_strides[4];
                                const int64_t weight_offset =
                                    ((((out_c * shape.channels + in_c) * shape.kernel_frames +
                                       kt) * shape.kernel_height + kh) * shape.kernel_width + kw);
                                sum += static_cast<float>(input[input_offset]) *
                                       static_cast<float>(weight[weight_offset]);
                            }
                        }
                    }
                }
                if (bias) sum += static_cast<float>(bias[out_c]);
                if (residual) {
                    const int64_t residual_offset =
                        ((((b * shape.output_channels + out_c) * shape.output_frames +
                           out_t) * shape.output_height + out_h) * shape.output_width + out_w);
                    sum += static_cast<float>(residual[residual_offset]);
                }
                const int64_t output_offset =
                    b * shape.output_strides[0] +
                    out_c * shape.output_strides[1] +
                    out_t * shape.output_strides[2] +
                    out_h * shape.output_strides[3] +
                    out_w * shape.output_strides[4];
                output[output_offset] = static_cast<sycl::half>(sum);
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_fp16_conv3d_scalar");
}

void launch_epilogue(
    const sycl::half* convolution,
    const sycl::half* bias,
    const sycl::half* residual,
    sycl::half* output,
    ConvShape shape,
    const at::Device& device) {
    const int64_t count = shape.batch * shape.output_frames *
        shape.output_height * shape.output_width * shape.output_channels;
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenFp16Conv3dEpilogueKernel>(
            sycl::range<1>(static_cast<size_t>(count)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t out_c = index % shape.output_channels;
                const int64_t out_w = (index / shape.output_channels) % shape.output_width;
                const int64_t out_h = (index / (shape.output_channels * shape.output_width)) %
                    shape.output_height;
                const int64_t out_t = (index / (shape.output_channels * shape.output_width *
                    shape.output_height)) % shape.output_frames;
                const int64_t b = index / (shape.output_channels * shape.output_width *
                    shape.output_height * shape.output_frames);
                const int64_t convolution_offset =
                    b * shape.convolution_strides[0] +
                    out_c * shape.convolution_strides[1] +
                    out_t * shape.convolution_strides[2] +
                    out_h * shape.convolution_strides[3] +
                    out_w * shape.convolution_strides[4];
                const int64_t output_offset =
                    b * shape.output_strides[0] +
                    out_c * shape.output_strides[1] +
                    out_t * shape.output_strides[2] +
                    out_h * shape.output_strides[3] +
                    out_w * shape.output_strides[4];
                float value = static_cast<float>(convolution[convolution_offset]);
                if (bias) value += static_cast<float>(bias[out_c]);
                const int64_t ncdhw_offset =
                    ((((b * shape.output_channels + out_c) * shape.output_frames +
                       out_t) * shape.output_height + out_h) * shape.output_width + out_w);
                if (residual) value += static_cast<float>(residual[ncdhw_offset]);
                output[output_offset] = static_cast<sycl::half>(value);
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_fp16_conv3d_epilogue");
}

}  // namespace

torch::Tensor fp16_conv3d(
    torch::Tensor input,
    torch::Tensor weight,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> residual,
    std::vector<int64_t> stride,
    std::optional<torch::Tensor> out) {
    TORCH_CHECK(input.device().is_xpu() && input.scalar_type() == at::kHalf &&
                    input.dim() == 5,
                "input must be FP16 XPU [B,C,T,H,W]");
    TORCH_CHECK(weight.device() == input.device() &&
                    weight.scalar_type() == at::kHalf && weight.dim() == 5 &&
                    weight.is_contiguous() && weight.size(1) == input.size(1),
                "weight must be contiguous FP16 XPU [K,C,T,R,S]");
    TORCH_CHECK(stride.size() == 3 && stride[0] > 0 &&
                    stride[1] > 0 && stride[2] > 0,
                "stride must contain three positive integers");
    ConvShape shape{
        input.size(0), input.size(1), input.size(2), input.size(3), input.size(4),
        weight.size(0), weight.size(2), weight.size(3), weight.size(4),
        0, 0, 0, stride[0], stride[1], stride[2],
    };
    TORCH_CHECK(shape.batch > 0 && shape.channels > 0 &&
                    shape.output_channels > 0 && shape.kernel_frames > 0 &&
                    shape.kernel_height > 0 && shape.kernel_width > 0 &&
                    shape.frames >= shape.kernel_frames &&
                    shape.height >= shape.kernel_height &&
                    shape.width >= shape.kernel_width,
                "Conv3D dimensions must be positive and kernel fit input");
    shape.output_frames = (shape.frames - shape.kernel_frames) /
                          shape.stride_frames + 1;
    shape.output_height = (shape.height - shape.kernel_height) /
                          shape.stride_height + 1;
    shape.output_width = (shape.width - shape.kernel_width) /
                         shape.stride_width + 1;
    for (int i = 0; i < 5; ++i) {
        shape.input_strides[i] = input.stride(i);
    }
    if (out.has_value()) {
        TORCH_CHECK(out->device() == input.device() &&
                        out->scalar_type() == at::kHalf &&
                        out->sizes() == at::IntArrayRef({
                            shape.batch, shape.output_channels,
                            shape.output_frames, shape.output_height,
                            shape.output_width}),
                    "out must have convolution shape, FP16 dtype and XPU device");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == input.device() &&
                        bias->scalar_type() == at::kHalf &&
                        bias->is_contiguous() &&
                        bias->numel() == shape.output_channels,
                    "bias must be contiguous FP16 [K]");
    }
    if (residual.has_value()) {
        TORCH_CHECK(residual->device() == input.device() &&
                        residual->scalar_type() == at::kHalf &&
                        residual->is_contiguous() && residual->dim() == 5 &&
                        residual->sizes() == at::IntArrayRef({
                            shape.batch, shape.output_channels,
                            shape.output_frames, shape.output_height,
                            shape.output_width}),
                    "residual must be contiguous FP16 [B,K,To,Ho,Wo]");
    }
    auto output = out.has_value() ? *out : torch::empty(
        {shape.batch, shape.output_frames, shape.output_height,
         shape.output_width, shape.output_channels},
        input.options()).permute({0, 4, 1, 2, 3});
    for (int i = 0; i < 5; ++i) {
        shape.output_strides[i] = output.stride(i);
    }
    const auto* input_ptr = static_cast<const sycl::half*>(input.data_ptr());
    const auto* weight_ptr = static_cast<const sycl::half*>(weight.data_ptr());
    const auto* bias_ptr = bias.has_value()
        ? static_cast<const sycl::half*>(bias->data_ptr()) : nullptr;
    const auto* residual_ptr = residual.has_value()
        ? static_cast<const sycl::half*>(residual->data_ptr()) : nullptr;
    auto* output_ptr = static_cast<sycl::half*>(output.data_ptr());
    const int64_t output_count = shape.batch * shape.output_frames *
        shape.output_height * shape.output_width * shape.output_channels;
    const int64_t kernel_elements = shape.channels * shape.kernel_frames *
        shape.kernel_height * shape.kernel_width;
    if (output_count <= 32 * 1024 * 1024 / kernel_elements) {
        launch_scalar(input_ptr, weight_ptr, bias_ptr, residual_ptr,
                      output_ptr, shape, input.device());
        return output;
    }

    auto convolution = out.has_value() ? output : torch::empty(
        {shape.batch, shape.output_channels, shape.output_frames,
         shape.output_height, shape.output_width}, input.options());
    for (int i = 0; i < 5; ++i) {
        shape.convolution_strides[i] = convolution.stride(i);
    }
    sycl::queue& queue = utils::get_queue(input.device());
    dnnl::engine engine = dnnl::sycl_interop::make_engine(
        queue.get_device(), queue.get_context());
    using DT = dnnl::memory::data_type;
    dnnl::memory::desc input_md(
        {shape.batch, shape.channels, shape.frames, shape.height, shape.width},
        DT::f16, std::vector<dnnl::memory::dim>(
            input.strides().begin(), input.strides().end()));
    dnnl::memory::desc weight_md(
        {shape.output_channels, shape.channels, shape.kernel_frames,
         shape.kernel_height, shape.kernel_width},
        DT::f16, dnnl::memory::format_tag::oidhw);
    dnnl::memory::desc output_md(
        {shape.batch, shape.output_channels, shape.output_frames,
         shape.output_height, shape.output_width},
        DT::f16, std::vector<dnnl::memory::dim>(
            convolution.strides().begin(), convolution.strides().end()));
    dnnl::convolution_forward::primitive_desc desc(
        engine, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        input_md, weight_md, output_md,
        {shape.stride_frames, shape.stride_height, shape.stride_width},
        {0, 0, 0}, {0, 0, 0});
    dnnl::convolution_forward primitive(desc);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);
    primitive.execute(stream, {
        {DNNL_ARG_SRC, dnnl::memory(input_md, engine, input.data_ptr())},
        {DNNL_ARG_WEIGHTS, dnnl::memory(weight_md, engine, weight.data_ptr())},
        {DNNL_ARG_DST, dnnl::memory(output_md, engine, convolution.data_ptr())},
    });
    if (!out.has_value() || bias_ptr || residual_ptr) {
        launch_epilogue(
            static_cast<const sycl::half*>(convolution.data_ptr()),
            bias_ptr, residual_ptr, output_ptr, shape, input.device());
    }
    return output;
}

}  // namespace omni_xpu::kitchen
