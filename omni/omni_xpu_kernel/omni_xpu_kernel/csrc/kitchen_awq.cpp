#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "utils.h"

namespace omni_xpu::kitchen {
namespace {

template <typename T>
class KitchenAwqGemvKernel;
template <typename T>
class KitchenAwqDequantKernel;
template <typename T>
class KitchenAwqBiasKernel;

template <typename T>
void launch_gemv(
    const T* input, const int8_t* packed, const T* scales,
    const T* zeros, const T* bias, T* output, int64_t rows,
    int64_t columns, int64_t width, int64_t group_size,
    const at::Device& device) {
    constexpr int WorkGroupSize = 128;
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenAwqGemvKernel<T>>(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(
                    rows * columns * WorkGroupSize)),
                sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) {
                const int64_t index = static_cast<int64_t>(item.get_group(0));
                const int64_t lane = static_cast<int64_t>(item.get_local_id(0));
                const int64_t row = index / columns;
                const int64_t col = index % columns;
                float sum = 0.0f;
                for (int64_t k = lane; k < width; k += WorkGroupSize) {
                    const uint8_t byte = static_cast<uint8_t>(
                        packed[col * (width / 2) + k / 2]);
                    const int code = (k & 1) ? (byte >> 4) : (byte & 15);
                    const int64_t group_offset = (k / group_size) * columns + col;
                    const float weight =
                        (static_cast<float>(code) - 8.0f) *
                            static_cast<float>(scales[group_offset]) +
                        static_cast<float>(zeros[group_offset]);
                    sum += static_cast<float>(input[row * width + k]) * weight;
                }
                const float total = sycl::reduce_over_group(
                    item.get_group(), sum, sycl::plus<float>());
                if (lane == 0) {
                    output[index] = static_cast<T>(
                        total + (bias ? static_cast<float>(bias[col]) : 0.0f));
                }
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_awq_w4a16_gemv");
}

template <typename T>
void launch_dequant(
    const int8_t* packed, const T* scales, const T* zeros,
    T* weight, int64_t columns, int64_t width, int64_t group_size,
    const at::Device& device) {
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenAwqDequantKernel<T>>(
            sycl::range<1>(static_cast<size_t>(columns * width)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t col = index / width;
                const int64_t k = index % width;
                const uint8_t byte = static_cast<uint8_t>(packed[index / 2]);
                const int code = (k & 1) ? (byte >> 4) : (byte & 15);
                const int64_t group_offset = (k / group_size) * columns + col;
                const float value =
                    (static_cast<float>(code) - 8.0f) *
                        static_cast<float>(scales[group_offset]) +
                    static_cast<float>(zeros[group_offset]);
                weight[index] = static_cast<T>(value);
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_awq_w4a16_dequant");
}

template <typename T>
void launch_bias(
    const T* bias, T* output, int64_t rows, int64_t columns,
    const at::Device& device) {
    if (!bias) return;
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenAwqBiasKernel<T>>(
            sycl::range<1>(static_cast<size_t>(rows * columns)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                output[index] = static_cast<T>(
                    static_cast<float>(output[index]) +
                    static_cast<float>(bias[index % columns]));
            });
    };
    utils::submit_kernel(cgf, device, "kitchen_awq_w4a16_bias");
}

template <typename T>
torch::Tensor run_native(
    const torch::Tensor& input, const torch::Tensor& packed,
    const torch::Tensor& scales, const torch::Tensor& zeros,
    const std::optional<torch::Tensor>& bias, int64_t group_size,
    int64_t rows, int64_t columns, int64_t width) {
    std::vector<int64_t> output_sizes = input.sizes().vec();
    output_sizes.back() = columns;
    auto output = torch::empty(output_sizes, input.options());
    const auto* input_ptr = static_cast<const T*>(input.data_ptr());
    const auto* packed_ptr = packed.data_ptr<int8_t>();
    const auto* scales_ptr = static_cast<const T*>(scales.data_ptr());
    const auto* zeros_ptr = static_cast<const T*>(zeros.data_ptr());
    const auto* bias_ptr = bias.has_value()
        ? static_cast<const T*>(bias->data_ptr()) : nullptr;
    auto* output_ptr = static_cast<T*>(output.data_ptr());
    if (rows <= 8) {
        launch_gemv(
            input_ptr, packed_ptr, scales_ptr, zeros_ptr, bias_ptr,
            output_ptr, rows, columns, width, group_size, input.device());
        return output;
    }

    auto dequant = torch::empty({columns, width}, input.options());
    launch_dequant(
        packed_ptr, scales_ptr, zeros_ptr,
        static_cast<T*>(dequant.data_ptr()), columns, width, group_size,
        input.device());
    sycl::queue& queue = utils::get_queue(input.device());
    dnnl::engine engine = dnnl::sycl_interop::make_engine(
        queue.get_device(), queue.get_context());
    using DT = dnnl::memory::data_type;
    constexpr auto dtype = std::is_same_v<T, sycl::half> ? DT::f16 : DT::bf16;
    dnnl::memory::desc input_md({rows, width}, dtype, dnnl::memory::format_tag::ab);
    dnnl::memory::desc weight_md({width, columns}, dtype, dnnl::memory::format_tag::ba);
    dnnl::memory::desc output_md({rows, columns}, dtype, dnnl::memory::format_tag::ab);
    dnnl::matmul::primitive_desc desc(engine, input_md, weight_md, output_md);
    dnnl::matmul primitive(desc);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);
    primitive.execute(stream, {
        {DNNL_ARG_SRC, dnnl::memory(input_md, engine, input.data_ptr())},
        {DNNL_ARG_WEIGHTS, dnnl::memory(weight_md, engine, dequant.data_ptr())},
        {DNNL_ARG_DST, dnnl::memory(output_md, engine, output.data_ptr())},
    });
    launch_bias(bias_ptr, output_ptr, rows, columns, input.device());
    return output;
}

}  // namespace

torch::Tensor gemv_awq_w4a16(
    torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
    torch::Tensor zeros, std::optional<torch::Tensor> bias,
    int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu() && input.dim() >= 1 &&
                    input.is_contiguous() && input.size(-1) > 0,
                "AWQ input must be contiguous XPU [..., K]");
    TORCH_CHECK(input.scalar_type() == at::kHalf ||
                    input.scalar_type() == at::kBFloat16,
                "AWQ input must be FP16 or BF16");
    const int64_t width = input.size(-1);
    const int64_t rows = input.numel() / width;
    TORCH_CHECK(rows > 0 && group_size > 0 && width % group_size == 0 &&
                    width % 2 == 0,
                "AWQ requires positive rows and a group size dividing even K");
    TORCH_CHECK(packed.device() == input.device() &&
                    packed.scalar_type() == at::kChar && packed.dim() == 2 &&
                    packed.is_contiguous() && packed.size(1) == width / 2,
                "AWQ qweight must be contiguous INT8 [N, K/2]");
    const int64_t columns = packed.size(0);
    TORCH_CHECK(columns > 0, "AWQ requires positive output columns");
    for (const auto& tensor : {scales, zeros}) {
        TORCH_CHECK(tensor.device() == input.device() &&
                        tensor.scalar_type() == input.scalar_type() &&
                        tensor.dim() == 2 && tensor.is_contiguous() &&
                        tensor.sizes() == at::IntArrayRef({width / group_size, columns}),
                    "AWQ scales/zeros must be contiguous input-dtype [K/G, N]");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == input.device() &&
                        bias->scalar_type() == input.scalar_type() &&
                        bias->dim() == 1 && bias->is_contiguous() &&
                        bias->numel() == columns,
                    "AWQ bias must be contiguous input-dtype [N]");
    }
    if (input.scalar_type() == at::kHalf) {
        return run_native<sycl::half>(
            input, packed, scales, zeros, bias, group_size,
            rows, columns, width);
    }
    return run_native<sycl::ext::oneapi::bfloat16>(
        input, packed, scales, zeros, bias, group_size,
        rows, columns, width);
}

}  // namespace omni_xpu::kitchen
