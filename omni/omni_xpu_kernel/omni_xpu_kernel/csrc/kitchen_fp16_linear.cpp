#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

namespace omni_xpu::kitchen {
namespace {

class KitchenFp16LinearScalarKernel;
class KitchenFp16LinearEpilogueKernel;

void launch_scalar(
    const sycl::half* input,
    const sycl::half* weight,
    const sycl::half* bias,
    const sycl::half* residual,
    const sycl::half* residual_scale,
    sycl::half* output,
    int64_t rows,
    int64_t input_features,
    int64_t output_features,
    const at::Device& device) {
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenFp16LinearScalarKernel>(
            sycl::range<1>(static_cast<size_t>(rows * output_features)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t row = index / output_features;
                const int64_t column = index % output_features;
                float sum = 0.0f;
                for (int64_t k = 0; k < input_features; ++k) {
                    sum += static_cast<float>(input[row * input_features + k]) *
                           static_cast<float>(weight[column * input_features + k]);
                }
                if (bias) sum += static_cast<float>(bias[column]);
                float value = static_cast<float>(static_cast<sycl::half>(sum));
                if (residual) {
                    value = static_cast<float>(residual[index]) +
                            static_cast<float>(residual_scale[column]) * value;
                }
                output[index] = static_cast<sycl::half>(value);
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_fp16_linear_scalar");
}

void launch_epilogue(
    const sycl::half* bias,
    const sycl::half* residual,
    const sycl::half* residual_scale,
    sycl::half* output,
    int64_t rows,
    int64_t output_features,
    const at::Device& device) {
    if (!bias && !residual) return;
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenFp16LinearEpilogueKernel>(
            sycl::range<1>(static_cast<size_t>(rows * output_features)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t column = index % output_features;
                float value = static_cast<float>(output[index]);
                if (bias) value += static_cast<float>(bias[column]);
                value = static_cast<float>(static_cast<sycl::half>(value));
                if (residual) {
                    value = static_cast<float>(residual[index]) +
                            static_cast<float>(residual_scale[column]) * value;
                }
                output[index] = static_cast<sycl::half>(value);
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_fp16_linear_epilogue");
}

}  // namespace

torch::Tensor fp16_linear(
    torch::Tensor input,
    torch::Tensor weight,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> residual,
    std::optional<torch::Tensor> residual_scale) {
    TORCH_CHECK(input.device().is_xpu() && input.scalar_type() == at::kHalf &&
                    input.dim() >= 1 && input.is_contiguous(),
                "input must be contiguous FP16 XPU [...,K]");
    TORCH_CHECK(weight.device() == input.device() &&
                    weight.scalar_type() == at::kHalf && weight.dim() == 2 &&
                    weight.is_contiguous() && weight.size(1) == input.size(-1),
                "weight must be contiguous FP16 XPU [N,K]");
    const int64_t input_features = input.size(-1);
    const int64_t output_features = weight.size(0);
    std::vector<int64_t> output_sizes = input.sizes().vec();
    output_sizes.back() = output_features;
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < output_sizes.size(); ++i) rows *= output_sizes[i];
    TORCH_CHECK(rows > 0 && output_features > 0,
                "FP16 linear requires positive rows and output features");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == input.device() &&
                        bias->scalar_type() == at::kHalf &&
                        bias->is_contiguous() && bias->numel() == output_features,
                    "bias must be contiguous FP16 [N]");
    }
    TORCH_CHECK(!residual.has_value() || residual_scale.has_value(),
                "fp16_linear residual requires residual_scale");
    if (residual.has_value()) {
        TORCH_CHECK(residual->device() == input.device() &&
                        residual->scalar_type() == at::kHalf &&
                        residual->is_contiguous() &&
                        residual->sizes() == at::IntArrayRef(output_sizes),
                    "residual must be contiguous FP16 [...,N]");
        TORCH_CHECK(residual_scale->device() == input.device() &&
                        residual_scale->scalar_type() == at::kHalf &&
                        residual_scale->is_contiguous() &&
                        residual_scale->numel() == output_features,
                    "residual_scale must be contiguous FP16 [N]");
    }
    auto output = torch::empty({rows, output_features}, input.options());
    const auto* input_ptr = static_cast<const sycl::half*>(input.data_ptr());
    const auto* weight_ptr = static_cast<const sycl::half*>(weight.data_ptr());
    const auto* bias_ptr = bias.has_value()
        ? static_cast<const sycl::half*>(bias->data_ptr()) : nullptr;
    const auto* residual_ptr = residual.has_value()
        ? static_cast<const sycl::half*>(residual->data_ptr()) : nullptr;
    const auto* scale_ptr = residual_scale.has_value()
        ? static_cast<const sycl::half*>(residual_scale->data_ptr()) : nullptr;
    auto* output_ptr = static_cast<sycl::half*>(output.data_ptr());

    if (input_features == 0 ||
        rows * output_features * input_features <= 16 * 1024 * 1024) {
        launch_scalar(input_ptr, weight_ptr, bias_ptr, residual_ptr, scale_ptr,
                      output_ptr, rows, input_features, output_features,
                      input.device());
        return output.reshape(output_sizes);
    }

    sycl::queue& queue = utils::get_queue(input.device());
    dnnl::engine engine = dnnl::sycl_interop::make_engine(
        queue.get_device(), queue.get_context());
    using DT = dnnl::memory::data_type;
    dnnl::memory::desc input_md(
        {rows, input_features}, DT::f16, dnnl::memory::format_tag::ab);
    dnnl::memory::desc weight_md(
        {input_features, output_features}, DT::f16,
        dnnl::memory::format_tag::ba);
    dnnl::memory::desc output_md(
        {rows, output_features}, DT::f16, dnnl::memory::format_tag::ab);
    dnnl::matmul::primitive_desc desc(engine, input_md, weight_md, output_md);
    dnnl::matmul primitive(desc);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);
    primitive.execute(stream, {
        {DNNL_ARG_SRC, dnnl::memory(input_md, engine, input.data_ptr())},
        {DNNL_ARG_WEIGHTS, dnnl::memory(weight_md, engine, weight.data_ptr())},
        {DNNL_ARG_DST, dnnl::memory(output_md, engine, output.data_ptr())},
    });
    launch_epilogue(bias_ptr, residual_ptr, scale_ptr, output_ptr,
                    rows, output_features, input.device());
    return output.reshape(output_sizes);
}

}  // namespace omni_xpu::kitchen
