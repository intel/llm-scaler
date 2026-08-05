#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

#include "utils.h"

namespace omni_xpu {
namespace int8_ops {

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kWorkGroupSize = 256;
constexpr int kVectorSize = 8;

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
    T values[Size];
};

inline bf16 silu_mul_exact_bf16(bf16 gate, bf16 up) {
    const float x = static_cast<float>(gate);
    const float denominator = 1.0f + sycl::exp(-x);
    float silu = x / denominator;

    // The core extension deliberately keeps its existing AOT backend options.
    // Refine only finite positive SiLU division so its BF16 materialization
    // matches PyTorch XPU's correctly-rounded divide contract.  The guard also
    // preserves PyTorch's negative-infinity and NaN behavior.
    if (x > 0.0f && sycl::isfinite(x)) {
        silu += sycl::fma(-silu, denominator, x) / denominator;
    }

    const bf16 rounded_silu = static_cast<bf16>(silu);
    return static_cast<bf16>(
        static_cast<float>(rounded_silu) * static_cast<float>(up));
}

template <bool Vectorized>
void launch_h3_silu_mul(
    const bf16* __restrict__ gate,
    const bf16* __restrict__ up,
    bf16* __restrict__ output,
    int32_t rows,
    int32_t columns,
    int32_t gate_row_stride,
    int32_t up_row_stride,
    const at::Device& device) {
    const int32_t logical_items = Vectorized
        ? rows * (columns / kVectorSize)
        : rows * columns;
    const int32_t groups =
        (logical_items + kWorkGroupSize - 1) / kWorkGroupSize;
    const int32_t global_items = groups * kWorkGroupSize;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(global_items),
                sycl::range<1>(kWorkGroupSize)),
            [=](sycl::nd_item<1> item) {
                const int32_t logical_index =
                    item.get_global_linear_id();
                if (logical_index >= logical_items) return;

                if constexpr (Vectorized) {
                    using Vector = AlignedVector<bf16, kVectorSize>;
                    const int32_t vectors_per_row =
                        columns / kVectorSize;
                    const int32_t row = logical_index / vectors_per_row;
                    const int32_t column_vector =
                        logical_index - row * vectors_per_row;
                    const int32_t input_column =
                        column_vector * kVectorSize;
                    const Vector gate_values =
                        *reinterpret_cast<const Vector*>(
                            gate + row * gate_row_stride + input_column);
                    const Vector up_values =
                        *reinterpret_cast<const Vector*>(
                            up + row * up_row_stride + input_column);
                    Vector result;
#pragma unroll
                    for (int index = 0; index < kVectorSize; ++index) {
                        result.values[index] = silu_mul_exact_bf16(
                            gate_values.values[index], up_values.values[index]);
                    }
                    reinterpret_cast<Vector*>(output)[logical_index] = result;
                } else {
                    const int32_t row = logical_index / columns;
                    const int32_t column =
                        logical_index - row * columns;
                    output[logical_index] = silu_mul_exact_bf16(
                        gate[row * gate_row_stride + column],
                        up[row * up_row_stride + column]);
                }
            });
    };
    utils::submit_kernel(cgf, device, "h3_silu_mul_exact_bf16");
}

}  // namespace

torch::Tensor fused_silu_mul_exact_bf16(
    torch::Tensor gate,
    torch::Tensor up) {
    TORCH_CHECK(gate.device().is_xpu() && up.device().is_xpu(),
        "gate and up must be on XPU");
    TORCH_CHECK(gate.device() == up.device(),
        "gate and up must be on the same device");
    TORCH_CHECK(gate.scalar_type() == torch::kBFloat16 &&
                    up.scalar_type() == torch::kBFloat16,
        "H3 exact SiLU-mul accepts BF16 only");
    TORCH_CHECK(gate.sizes() == up.sizes(),
        "gate and up must have identical shapes");
    TORCH_CHECK(gate.dim() == 1 || gate.dim() == 2,
        "H3 exact SiLU-mul accepts 1D or 2D tensors");
    TORCH_CHECK(gate.stride(-1) == 1 && up.stride(-1) == 1,
        "H3 exact SiLU-mul requires inner stride 1");
    TORCH_CHECK(gate.numel() <= std::numeric_limits<int32_t>::max(),
        "H3 exact SiLU-mul requires numel <= INT32_MAX");

    auto output = torch::empty(gate.sizes(), gate.options());
    if (gate.numel() == 0) return output;

    const int64_t rows64 = gate.dim() == 1 ? 1 : gate.size(0);
    const int64_t columns64 = gate.size(-1);
    const int64_t gate_row_stride64 =
        gate.dim() == 1 ? columns64 : gate.stride(0);
    const int64_t up_row_stride64 =
        up.dim() == 1 ? columns64 : up.stride(0);
    TORCH_CHECK(
        rows64 <= std::numeric_limits<int32_t>::max() &&
            columns64 <= std::numeric_limits<int32_t>::max() &&
            gate_row_stride64 > 0 &&
            gate_row_stride64 <= std::numeric_limits<int32_t>::max() &&
            up_row_stride64 > 0 &&
            up_row_stride64 <= std::numeric_limits<int32_t>::max(),
        "H3 exact SiLU-mul requires positive INT32 rows/columns/strides");

    const int32_t rows = static_cast<int32_t>(rows64);
    const int32_t columns = static_cast<int32_t>(columns64);
    const int32_t gate_row_stride =
        static_cast<int32_t>(gate_row_stride64);
    const int32_t up_row_stride =
        static_cast<int32_t>(up_row_stride64);
    using Vector = AlignedVector<bf16, kVectorSize>;
    const auto gate_address =
        reinterpret_cast<std::uintptr_t>(gate.data_ptr());
    const auto up_address =
        reinterpret_cast<std::uintptr_t>(up.data_ptr());
    const bool vector_aligned =
        columns % kVectorSize == 0 &&
        gate_row_stride % kVectorSize == 0 &&
        up_row_stride % kVectorSize == 0 &&
        gate_address % alignof(Vector) == 0 &&
        up_address % alignof(Vector) == 0;

    if (vector_aligned) {
        launch_h3_silu_mul<true>(
            reinterpret_cast<const bf16*>(gate.data_ptr()),
            reinterpret_cast<const bf16*>(up.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()),
            rows, columns, gate_row_stride, up_row_stride, gate.device());
    } else {
        launch_h3_silu_mul<false>(
            reinterpret_cast<const bf16*>(gate.data_ptr()),
            reinterpret_cast<const bf16*>(up.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()),
            rows, columns, gate_row_stride, up_row_stride, gate.device());
    }
    return output;
}

}  // namespace int8_ops
}  // namespace omni_xpu
