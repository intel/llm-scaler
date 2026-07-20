// ============================================================================
// Fused INT8 Quantization Kernels
// ============================================================================
// High-performance quantization kernels that fuse:
//   absmax reduction + scale computation + divide + round + clamp + cast
// into minimal kernel launches, eliminating Python-level multi-op overhead.
//
// Input:  [M, K] bf16/f16
// Output: [M, K] int8 + [M, 1] float32 scales
//
// Design: One plain-SYCL sub-group cooperatively processes each row. Each lane
// handles contiguous vector chunks so both passes use wide coalesced accesses.
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace int8_ops {

// ============================================================================
// Kernel: Fused per-row INT8 quantization
// ============================================================================
// Each sub-group processes one full row of K elements:
//   1. Compute absmax across the row (vectorized reduce)
//   2. Compute scale = absmax / 127
//   3. Quantize: round(x / scale).clamp(-128, 127) → int8
// ============================================================================

template <typename InputT>
void quantize_int8_rowwise_kernel(
    const InputT* __restrict__ input,  // [M, K]
    int8_t* __restrict__ output,       // [M, K]
    float* __restrict__ scales,        // [M]
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    // Subgroup-cooperative rowwise INT8 quantization (plain SYCL).
    //
    // One sub-group (SG lanes) processes one row of K elements. Large aligned
    // rows use contiguous VEC-element chunks per lane; other rows use scalar
    // interleaving. Both layouts give fully coalesced HBM access. Row absmax is
    // reduced via a sub-group collective (no SLM, no barrier). Pass 2 re-reads
    // the row (served from L2 for typical K) and writes int8.
    //
    // This replaces the previous ESIMD SLM kernel, which suffered a large IGC
    // JIT register-spill penalty inside the multi-kernel _C module (measured
    // ~38 GB/s vs ~140 GB/s for the original scalar sub-group design at
    // M=4128, K=3840).
    //
    // HBM traffic: K*2 read (pass1) + K*2 read (pass2, L2-cached) + K*1 write.
    constexpr int SG = 32;            // sub-group size (lanes per row)
    constexpr int ROWS_PER_WG = 8;    // rows (sub-groups) per work-group
    constexpr int WG = SG * ROWS_PER_WG;
    constexpr int VEC = 8;            // contiguous elements handled by each lane
    constexpr int MIN_VECTOR_K = SG * VEC * 2;

    const int64_t n_wg = (M + ROWS_PER_WG - 1) / ROWS_PER_WG;
    sycl::range<1> global_size(static_cast<size_t>(n_wg) * WG);
    sycl::range<1> local_size(WG);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0)) * ROWS_PER_WG +
                    sg.get_group_linear_id();
                if (row >= M) return;

                const InputT* __restrict__ row_ptr = input + row * K;
                int8_t* __restrict__ out_ptr = output + row * K;

                // Process aligned rows with wider per-lane transactions. One
                // sub-group iteration covers SG*VEC contiguous elements, which
                // reduces loop/address overhead without increasing GRF pressure
                // enough to trigger the spills seen in the old ESIMD kernel.
                float local_max = 0.0f;
                const bool use_vector = K >= MIN_VECTOR_K && K % VEC == 0;
                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values =
                            *reinterpret_cast<const InputVec*>(row_ptr + k);
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const float v = static_cast<float>(values[i]);
                            local_max = sycl::fmax(local_max, sycl::fabs(v));
                        }
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        const float v = static_cast<float>(row_ptr[k]);
                        local_max = sycl::fmax(local_max, sycl::fabs(v));
                    }
                }
                float row_max =
                    sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                float inv_scale = 1.0f / scale;
                if (lane == 0) scales[row] = scale;

                // Pass 2: coalesced quantize + write (round-to-nearest-even).
                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    using OutputVec = sycl::vec<int8_t, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values =
                            *reinterpret_cast<const InputVec*>(row_ptr + k);
                        OutputVec quantized;
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            float r = sycl::rint(
                                static_cast<float>(values[i]) * inv_scale);
                            r = sycl::fmax(-128.0f, sycl::fmin(127.0f, r));
                            quantized[i] =
                                static_cast<int8_t>(static_cast<int32_t>(r));
                        }
                        *reinterpret_cast<OutputVec*>(out_ptr + k) = quantized;
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        float r = sycl::rint(
                            static_cast<float>(row_ptr[k]) * inv_scale);
                        r = sycl::fmax(-128.0f, sycl::fmin(127.0f, r));
                        out_ptr[k] =
                            static_cast<int8_t>(static_cast<int32_t>(r));
                    }
                }
            }
        );
    };
    utils::submit_kernel(cgf, device, "quantize_int8_rowwise_sg2pass");
}

template <typename InputT>
inline float silu_mul_rounded(InputT x1, InputT x2) {
    const float a = static_cast<float>(x1);
    const float b = static_cast<float>(x2);

    // Stable sigmoid formulation avoids exp overflow for large negative input.
    const float exp_value = sycl::exp(a >= 0.0f ? -a : a);
    const float sigmoid = a >= 0.0f
        ? 1.0f / (1.0f + exp_value)
        : exp_value / (1.0f + exp_value);

    // Match the existing PyTorch inference boundary as closely as possible:
    // F.silu(x1) is stored in the input dtype before the multiply, and the
    // product is stored in that dtype before rowwise quantization.
    const InputT silu_value = static_cast<InputT>(a * sigmoid);
    const InputT product = static_cast<InputT>(
        static_cast<float>(silu_value) * b);
    float value = static_cast<float>(product);

    // Lumina's clamp_fp16 applies nan_to_num to the floating SwiGLU result.
    if constexpr (std::is_same_v<InputT, fp16>) {
        if (sycl::isnan(value)) {
            value = 0.0f;
        } else {
            value = sycl::fmax(-65504.0f, sycl::fmin(65504.0f, value));
        }
    }
    return value;
}

template <typename InputT>
void fused_silu_mul_kernel(
    const InputT* __restrict__ input1,
    const InputT* __restrict__ input2,
    InputT* __restrict__ output,
    int64_t numel,
    const at::Device& device
) {
    constexpr int WG = 256;
    constexpr int VEC = 8;
    const int64_t work_items = (numel + VEC - 1) / VEC;
    const int64_t global_items = ((work_items + WG - 1) / WG) * WG;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(global_items)),
                sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const int64_t base =
                    static_cast<int64_t>(item.get_global_linear_id()) * VEC;
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int64_t index = base + i;
                    if (index < numel) {
                        output[index] = static_cast<InputT>(
                            silu_mul_rounded(input1[index], input2[index]));
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "fused_silu_mul");
}

// One subgroup owns one row. The first pass recomputes the rounded SwiGLU
// value for absmax; the second pass recomputes it for quantization. This avoids
// the full floating [M,K] intermediate without introducing per-row scratch
// storage that would exceed registers/SLM at K=10240.
template <typename InputT>
void fused_silu_mul_quantize_rowwise_kernel(
    const InputT* __restrict__ input1,
    const InputT* __restrict__ input2,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    constexpr int SG = 32;
    constexpr int ROWS_PER_WG = 8;
    constexpr int WG = SG * ROWS_PER_WG;
    constexpr int VEC = 8;
    constexpr int MIN_VECTOR_K = SG * VEC * 2;

    const int64_t n_wg = (M + ROWS_PER_WG - 1) / ROWS_PER_WG;
    sycl::range<1> global_size(static_cast<size_t>(n_wg) * WG);
    sycl::range<1> local_size(WG);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0)) * ROWS_PER_WG +
                    sg.get_group_linear_id();
                if (row >= M) return;

                const InputT* __restrict__ row1 = input1 + row * K;
                const InputT* __restrict__ row2 = input2 + row * K;
                int8_t* __restrict__ out_ptr = output + row * K;

                float local_max = 0.0f;
                const bool use_vector = K >= MIN_VECTOR_K && K % VEC == 0;
                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values1 =
                            *reinterpret_cast<const InputVec*>(row1 + k);
                        const InputVec values2 =
                            *reinterpret_cast<const InputVec*>(row2 + k);
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const float value =
                                silu_mul_rounded(values1[i], values2[i]);
                            local_max = sycl::fmax(
                                local_max, sycl::fabs(value));
                        }
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        const float value =
                            silu_mul_rounded(row1[k], row2[k]);
                        local_max = sycl::fmax(
                            local_max, sycl::fabs(value));
                    }
                }

                const float row_max = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());
                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                const float inv_scale = 1.0f / scale;
                if (lane == 0) scales[row] = scale;

                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    using OutputVec = sycl::vec<int8_t, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values1 =
                            *reinterpret_cast<const InputVec*>(row1 + k);
                        const InputVec values2 =
                            *reinterpret_cast<const InputVec*>(row2 + k);
                        OutputVec quantized;
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            float rounded = sycl::rint(
                                silu_mul_rounded(values1[i], values2[i]) *
                                inv_scale);
                            rounded = sycl::fmax(
                                -128.0f, sycl::fmin(127.0f, rounded));
                            quantized[i] = static_cast<int8_t>(
                                static_cast<int32_t>(rounded));
                        }
                        *reinterpret_cast<OutputVec*>(out_ptr + k) = quantized;
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        float rounded = sycl::rint(
                            silu_mul_rounded(row1[k], row2[k]) * inv_scale);
                        rounded = sycl::fmax(
                            -128.0f, sycl::fmin(127.0f, rounded));
                        out_ptr[k] = static_cast<int8_t>(
                            static_cast<int32_t>(rounded));
                    }
                }
            });
    };
    utils::submit_kernel(
        cgf, device, "fused_silu_mul_quantize_rowwise_sg2pass");
}

// ============================================================================
// Public C++ API for fused quantization
// ============================================================================

torch::Tensor fused_silu_mul(
    torch::Tensor x1,
    torch::Tensor x2
) {
    TORCH_CHECK(x1.device().is_xpu(), "x1 must be on XPU device");
    TORCH_CHECK(x2.device().is_xpu(), "x2 must be on XPU device");
    TORCH_CHECK(x1.device() == x2.device(),
        "x1 and x2 must be on the same XPU device");
    TORCH_CHECK(x1.sizes() == x2.sizes(),
        "x1 and x2 must have identical shapes, got ",
        x1.sizes(), " and ", x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
        "x1 and x2 must have identical dtypes");
    TORCH_CHECK(
        x1.scalar_type() == torch::kBFloat16 ||
        x1.scalar_type() == torch::kHalf,
        "x1 and x2 must be bf16 or f16, got ", x1.scalar_type());

    x1 = x1.contiguous();
    x2 = x2.contiguous();
    auto output = torch::empty_like(x1);
    if (x1.numel() == 0) return output;

    if (x1.scalar_type() == torch::kBFloat16) {
        fused_silu_mul_kernel<bf16>(
            reinterpret_cast<const bf16*>(x1.data_ptr()),
            reinterpret_cast<const bf16*>(x2.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()),
            x1.numel(), x1.device());
    } else {
        fused_silu_mul_kernel<fp16>(
            reinterpret_cast<const fp16*>(x1.data_ptr()),
            reinterpret_cast<const fp16*>(x2.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()),
            x1.numel(), x1.device());
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise_fused(
    torch::Tensor x
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1D");
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 ||
            x.scalar_type() == torch::kHalf ||
            x.scalar_type() == torch::kFloat,
        "x must be bf16, f16, or f32, got ", x.scalar_type()
    );

    x = x.contiguous();
    TORCH_CHECK(x.size(-1) > 0, "x last dimension must be non-empty");
    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);

    torch::Tensor output = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    if (x.scalar_type() == torch::kBFloat16) {
        quantize_int8_rowwise_kernel<bf16>(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    } else if (x.scalar_type() == torch::kHalf) {
        quantize_int8_rowwise_kernel<fp16>(
            reinterpret_cast<const fp16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    } else {
        quantize_int8_rowwise_kernel<float>(
            x.data_ptr<float>(),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    }

    // Reshape scales to [..., 1] to match PyTorch convention
    auto scale_shape = x.sizes().vec();
    scale_shape.back() = 1;
    return {output.reshape(x.sizes()), scales.reshape(scale_shape)};
}

std::tuple<torch::Tensor, torch::Tensor> fused_silu_mul_quantize_rowwise(
    torch::Tensor x1,
    torch::Tensor x2
) {
    TORCH_CHECK(x1.device().is_xpu(), "x1 must be on XPU device");
    TORCH_CHECK(x2.device().is_xpu(), "x2 must be on XPU device");
    TORCH_CHECK(x1.device() == x2.device(),
        "x1 and x2 must be on the same XPU device");
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2,
        "x1 and x2 must be at least 2D");
    TORCH_CHECK(x1.sizes() == x2.sizes(),
        "x1 and x2 must have identical shapes, got ",
        x1.sizes(), " and ", x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
        "x1 and x2 must have identical dtypes");
    TORCH_CHECK(
        x1.scalar_type() == torch::kBFloat16 ||
        x1.scalar_type() == torch::kHalf,
        "x1 and x2 must be bf16 or f16, got ", x1.scalar_type());

    x1 = x1.contiguous();
    x2 = x2.contiguous();
    TORCH_CHECK(x1.size(-1) > 0,
        "x1 and x2 last dimension must be non-empty");
    const int64_t M = x1.numel() / x1.size(-1);
    const int64_t K = x1.size(-1);

    auto output = torch::empty_like(
        x1, torch::TensorOptions().dtype(torch::kInt8));
    auto scales = torch::empty(
        {M}, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));

    if (x1.scalar_type() == torch::kBFloat16) {
        fused_silu_mul_quantize_rowwise_kernel<bf16>(
            reinterpret_cast<const bf16*>(x1.data_ptr()),
            reinterpret_cast<const bf16*>(x2.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(), M, K, x1.device());
    } else {
        fused_silu_mul_quantize_rowwise_kernel<fp16>(
            reinterpret_cast<const fp16*>(x1.data_ptr()),
            reinterpret_cast<const fp16*>(x2.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(), M, K, x1.device());
    }

    auto scale_shape = x1.sizes().vec();
    scale_shape.back() = 1;
    return {output.reshape(x1.sizes()), scales.reshape(scale_shape)};
}

}  // namespace int8_ops
}  // namespace omni_xpu
