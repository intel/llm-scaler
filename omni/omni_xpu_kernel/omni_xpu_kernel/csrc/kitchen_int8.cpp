#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

namespace omni_xpu::kitchen {
namespace {

template <typename T>
class KitchenRmsNormForInt8Kernel;

template <typename T>
class KitchenScaledResidualKernel;

template <typename T>
class KitchenRmsNormQuantizeInt8Kernel;

template <typename T>
std::tuple<torch::Tensor, torch::Tensor> launch_rms_norm_quantize_int8(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
    const int64_t width = input.size(-1);
    const int64_t rows = input.numel() / width;
    auto q = torch::empty(input.sizes(), input.options().dtype(at::kChar));
    std::vector<int64_t> scale_sizes = input.sizes().vec();
    scale_sizes.back() = 1;
    auto scales = torch::empty(scale_sizes, input.options().dtype(at::kFloat));
    if (rows == 0) return {q, scales};

    const auto* input_ptr = static_cast<const T*>(input.data_ptr());
    const auto* weight_ptr = weight.scalar_type() == input.scalar_type()
        ? static_cast<const T*>(weight.data_ptr()) : nullptr;
    const auto* weight_f32 = weight.scalar_type() == at::kFloat
        ? weight.data_ptr<float>() : nullptr;
    auto* q_ptr = q.data_ptr<int8_t>();
    auto* scale_ptr = scales.data_ptr<float>();
    const float epsilon = static_cast<float>(eps);
    constexpr size_t subgroup = 32;
    constexpr size_t rows_per_workgroup = 8;
    constexpr size_t workgroup = subgroup * rows_per_workgroup;
    const size_t groups =
        (static_cast<size_t>(rows) + rows_per_workgroup - 1) /
        rows_per_workgroup;

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenRmsNormQuantizeInt8Kernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(groups * workgroup),
                              sycl::range<1>(workgroup)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = item.get_sub_group();
                const int64_t lane = sg.get_local_linear_id();
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0)) *
                        rows_per_workgroup + sg.get_group_linear_id();
                if (row >= rows) return;
                const T* input_row = input_ptr + row * width;
                int8_t* q_row = q_ptr + row * width;
                float square_sum = 0.0f;
                for (int64_t column = lane; column < width; column += subgroup) {
                    const float value = static_cast<float>(input_row[column]);
                    square_sum += value * value;
                }
                square_sum = sycl::reduce_over_group(
                    sg, square_sum, sycl::plus<float>());
                const float inverse = 1.0f / sycl::sqrt(
                    square_sum / static_cast<float>(width) + epsilon);

                float local_max = 0.0f;
                for (int64_t column = lane; column < width; column += subgroup) {
                    const float gamma = weight_ptr
                        ? static_cast<float>(weight_ptr[column])
                        : weight_f32[column];
                    const float normalized = static_cast<float>(static_cast<T>(
                        static_cast<float>(input_row[column]) * inverse * gamma));
                    local_max = sycl::fmax(local_max, sycl::fabs(normalized));
                }
                const float row_max = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());
                const float scale = sycl::fmax(row_max / 127.0f, 1e-30f);
                if (lane == 0) scale_ptr[row] = scale;
                const float inverse_scale = 1.0f / scale;

                for (int64_t column = lane; column < width; column += subgroup) {
                    const float gamma = weight_ptr
                        ? static_cast<float>(weight_ptr[column])
                        : weight_f32[column];
                    const float normalized = static_cast<float>(static_cast<T>(
                        static_cast<float>(input_row[column]) * inverse * gamma));
                    const float rounded = sycl::rint(normalized * inverse_scale);
                    q_row[column] = static_cast<int8_t>(
                        sycl::fmax(-128.0f, sycl::fmin(127.0f, rounded)));
                }
            });
    };
    utils::submit_kernel(cgf, input.device(), "kitchen_rms_norm_quantize_int8");
    return {q, scales};
}

template <typename T>
torch::Tensor launch_rms_norm_for_int8(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
    const int64_t width = input.size(-1);
    const int64_t rows = input.numel() / width;
    auto output = torch::empty_like(input);
    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = weight.scalar_type() == input.scalar_type()
        ? static_cast<const T*>(weight.data_ptr()) : nullptr;
    const float* weight_f32 = weight.scalar_type() == at::kFloat
        ? weight.data_ptr<float>() : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const float epsilon = static_cast<float>(eps);
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenRmsNormForInt8Kernel<T>>(
            sycl::range<1>(static_cast<size_t>(rows)),
            [=](sycl::id<1> item) {
                const int64_t row = static_cast<int64_t>(item[0]);
                const T* input_row = input_ptr + row * width;
                T* output_row = output_ptr + row * width;
                float square_sum = 0.0f;
                for (int64_t column = 0; column < width; ++column) {
                    const float value = static_cast<float>(input_row[column]);
                    square_sum += value * value;
                }
                const float inverse = 1.0f / sycl::sqrt(
                    square_sum / static_cast<float>(width) + epsilon);
                for (int64_t column = 0; column < width; ++column) {
                    const float gamma = weight_ptr
                        ? static_cast<float>(weight_ptr[column])
                        : weight_f32[column];
                    output_row[column] = static_cast<T>(
                        static_cast<float>(input_row[column]) * inverse * gamma);
                }
            });
    };
    utils::submit_kernel(cgf, input.device(), "kitchen_rms_norm_for_int8");
    return output;
}

template <typename T>
torch::Tensor launch_scaled_residual(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& scale) {
    const int64_t count = input.numel();
    const int64_t width = input.size(-1);
    auto output = torch::empty_like(input);
    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* residual_ptr = static_cast<const T*>(residual.data_ptr());
    const T* scale_ptr = static_cast<const T*>(scale.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenScaledResidualKernel<T>>(
            sycl::range<1>(static_cast<size_t>(count)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t column = index % width;
                const float value = static_cast<float>(residual_ptr[index]) +
                    static_cast<float>(scale_ptr[column]) *
                    static_cast<float>(input_ptr[index]);
                output_ptr[index] = static_cast<T>(value);
            });
    };
    utils::submit_kernel(cgf, input.device(), "kitchen_scaled_residual");
    return output;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> rms_norm_quantize_int8(
    torch::Tensor input, torch::Tensor weight, double eps) {
    TORCH_CHECK(input.device().is_xpu() && input.dim() >= 1 &&
                    input.is_contiguous() && input.size(-1) > 0,
                "input must be contiguous XPU [...,K] with K>0");
    TORCH_CHECK(weight.device() == input.device() && weight.dim() == 1 &&
                    weight.numel() == input.size(-1) && weight.is_contiguous() &&
                    (weight.scalar_type() == input.scalar_type() ||
                     weight.scalar_type() == at::kFloat),
                "weight must be contiguous [K] in input or FP32 dtype");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0,
                "eps must be finite and nonnegative");
    switch (input.scalar_type()) {
        case at::kHalf:
            return launch_rms_norm_quantize_int8<sycl::half>(input, weight, eps);
        case at::kBFloat16:
            return launch_rms_norm_quantize_int8<sycl::ext::oneapi::bfloat16>(
                input, weight, eps);
        case at::kFloat:
            return launch_rms_norm_quantize_int8<float>(input, weight, eps);
        default:
            TORCH_CHECK(false, "RMSNorm input must be fp16, bf16 or fp32");
    }
}

torch::Tensor rms_norm_for_int8(
    torch::Tensor input, torch::Tensor weight, double eps) {
    TORCH_CHECK(input.device().is_xpu() && input.dim() >= 1 &&
                    input.is_contiguous() && input.size(-1) > 0,
                "input must be contiguous XPU [...,K] with K>0");
    TORCH_CHECK(weight.device() == input.device() && weight.dim() == 1 &&
                    weight.numel() == input.size(-1) && weight.is_contiguous() &&
                    (weight.scalar_type() == input.scalar_type() ||
                     weight.scalar_type() == at::kFloat),
                "weight must be contiguous [K] in input or FP32 dtype");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0,
                "eps must be finite and nonnegative");
    switch (input.scalar_type()) {
        case at::kHalf:
            return launch_rms_norm_for_int8<sycl::half>(input, weight, eps);
        case at::kBFloat16:
            return launch_rms_norm_for_int8<sycl::ext::oneapi::bfloat16>(
                input, weight, eps);
        case at::kFloat:
            return launch_rms_norm_for_int8<float>(input, weight, eps);
        default:
            TORCH_CHECK(false, "RMSNorm input must be fp16, bf16 or fp32");
    }
}

torch::Tensor scaled_residual(
    torch::Tensor input, torch::Tensor residual,
    torch::Tensor residual_scale) {
    TORCH_CHECK(input.device().is_xpu() && input.dim() >= 1 &&
                    input.is_contiguous() && input.numel() > 0,
                "input must be nonempty contiguous XPU [...,N]");
    TORCH_CHECK(residual.device() == input.device() &&
                    residual.scalar_type() == input.scalar_type() &&
                    residual.is_contiguous() &&
                    residual.sizes() == input.sizes(),
                "residual must match input device, dtype and shape");
    TORCH_CHECK(residual_scale.device() == input.device() &&
                    residual_scale.scalar_type() == input.scalar_type() &&
                    residual_scale.dim() == 1 &&
                    residual_scale.numel() == input.size(-1) &&
                    residual_scale.is_contiguous(),
                "residual_scale must be contiguous [N] on input device/dtype");
    switch (input.scalar_type()) {
        case at::kHalf:
            return launch_scaled_residual<sycl::half>(
                input, residual, residual_scale);
        case at::kBFloat16:
            return launch_scaled_residual<sycl::ext::oneapi::bfloat16>(
                input, residual, residual_scale);
        case at::kFloat:
            return launch_scaled_residual<float>(
                input, residual, residual_scale);
        default:
            TORCH_CHECK(false, "scaled_residual requires fp16, bf16 or fp32");
    }
}

}  // namespace omni_xpu::kitchen
