#include <torch/extension.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "bmg_kernel_policy.h"
#include "device_utils.h"
#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace norm {

template <typename T, typename Policy>
void fused_adaln_kernel(
    const T* input,
    const T* modulation_scale,
    const T* modulation_shift,
    T* output,
    int64_t rows,
    int64_t hidden,
    int64_t modulation_rows,
    int64_t row_repeat,
    float eps,
    const at::Device& device) {
    constexpr int BS = Policy::adaln_block_size;
    constexpr int WG = Policy::adaln_work_group_size;
    const int64_t padded = (rows + WG - 1) / WG * WG;
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t row = item.get_global_id(0);
                if (row >= rows) return;
                const T* src = input + row * hidden;
                T* dst = output + row * hidden;

                simd<float, BS> sums(0.0f);
                simd<float, BS> squares(0.0f);
                for (int64_t col = 0; col < hidden; col += BS) {
                    simd<T, BS> raw = block_load<T, BS>(src + col);
                    simd<float, BS> value = raw;
                    sums += value;
                    squares += value * value;
                }
                const float total = sycl::ext::intel::esimd::detail::sum<float, float, BS>(sums);
                const float total_sq = sycl::ext::intel::esimd::detail::sum<float, float, BS>(squares);
                const float mean = total / hidden;
                const float variance = total_sq / hidden - mean * mean;
                const float inv_std = rsqrt(variance + eps);

                int64_t modulation_row = row_repeat > 0 ? row / row_repeat : row;
                if (modulation_rows == 1) modulation_row = 0;
                const T* scale = modulation_scale + modulation_row * hidden;
                const T* shift = modulation_shift + modulation_row * hidden;
                for (int64_t col = 0; col < hidden; col += BS) {
                    simd<T, BS> raw = block_load<T, BS>(src + col);
                    simd<T, BS> raw_scale = block_load<T, BS>(scale + col);
                    simd<T, BS> raw_shift = block_load<T, BS>(shift + col);
                    simd<float, BS> value = raw;
                    simd<float, BS> mod_scale = raw_scale;
                    simd<float, BS> mod_shift = raw_shift;
                    simd<float, BS> result = (value - mean) * inv_std;
                    result = result * (1.0f + mod_scale) + mod_shift;
                    block_store<T, BS>(dst + col, result);
                }
            });
    };
    utils::submit_kernel(cgf, device, "fused_adaln_esimd");
}

torch::Tensor fused_adaln(
    torch::Tensor input,
    torch::Tensor modulation_scale,
    torch::Tensor modulation_shift,
    int64_t row_repeat,
    double eps) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.dim() == 2, "input must be [rows, hidden]");
    TORCH_CHECK(modulation_scale.dim() == 2 && modulation_shift.dim() == 2,
                "modulation tensors must be 2D");
    TORCH_CHECK(input.is_contiguous() && modulation_scale.is_contiguous() && modulation_shift.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.scalar_type() == modulation_scale.scalar_type() &&
                    input.scalar_type() == modulation_shift.scalar_type(),
                "input, scale, and shift dtypes must match");
    TORCH_CHECK(modulation_scale.sizes() == modulation_shift.sizes(),
                "scale and shift shapes must match");
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    const int64_t modulation_rows = modulation_scale.size(0);
    TORCH_CHECK(hidden > 0 && hidden <= 8192 && hidden % 32 == 0,
                "hidden size must be nonzero, <=8192, and divisible by 32");
    TORCH_CHECK(modulation_scale.size(1) == hidden, "modulation hidden size mismatch");
    TORCH_CHECK(modulation_rows == 1 ||
                    (row_repeat > 0 && modulation_rows * row_repeat == rows),
                "modulation rows and row_repeat do not cover input rows");
    auto output = torch::empty_like(input);
    auto& queue = utils::get_queue(input.device());
    const bool use_b60 =
        device::use_b60_kernel_profile(queue) &&
        rows == 4096 && hidden == 3072;
#define DISPATCH_ADALN(T, input_ptr, scale_ptr, shift_ptr, output_ptr)     \
    do {                                                                  \
        if (use_b60) {                                                     \
            fused_adaln_kernel<T, device::B60KernelPolicy>(                \
                input_ptr, scale_ptr, shift_ptr, output_ptr, rows, hidden, \
                modulation_rows, row_repeat, static_cast<float>(eps),      \
                input.device());                                           \
        } else {                                                           \
            fused_adaln_kernel<T, device::B70KernelPolicy>(                \
                input_ptr, scale_ptr, shift_ptr, output_ptr, rows, hidden, \
                modulation_rows, row_repeat, static_cast<float>(eps),      \
                input.device());                                           \
        }                                                                  \
    } while (false)
    if (input.scalar_type() == torch::kFloat32) {
        DISPATCH_ADALN(
            float, input.data_ptr<float>(), modulation_scale.data_ptr<float>(),
            modulation_shift.data_ptr<float>(), output.data_ptr<float>());
    } else if (input.scalar_type() == torch::kFloat16) {
        DISPATCH_ADALN(
            fp16, reinterpret_cast<const fp16*>(input.data_ptr()),
            reinterpret_cast<const fp16*>(modulation_scale.data_ptr()),
            reinterpret_cast<const fp16*>(modulation_shift.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()));
    } else if (input.scalar_type() == torch::kBFloat16) {
        DISPATCH_ADALN(
            bf16, reinterpret_cast<const bf16*>(input.data_ptr()),
            reinterpret_cast<const bf16*>(modulation_scale.data_ptr()),
            reinterpret_cast<const bf16*>(modulation_shift.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()));
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }
#undef DISPATCH_ADALN
    return output;
}

}  // namespace norm
}  // namespace omni_xpu
