// ============================================================================
// ESIMD Fused INT8 Scale-back + Bias Kernel
// ============================================================================
// Fuses post-GEMM rescaling into a single memory pass:
//   output[m,n] = (int32_result[m,n] * x_scale[m] * w_scale[n]).to(bf16) + bias[n]
//
// Uses 2D (row, col_tile) work-item grid with block_load/block_store for
// maximum memory throughput — pattern from svdq_fused_postproc.cpp.
//
// Memory traffic: read M*N int32 (4B) + write M*N bf16 (2B) = 6 bytes/elem
// vs original 3-pass: ~20 bytes/elem (int32→f32 + f32*scale + f32→bf16)
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {

// ============================================================================
// Fast path: N divisible by ELEM_PER_WI (32)
// 2D grid [M, N/32] — each WI does block_load<int32,32> → scale → block_store
// ============================================================================

template <typename OutputT, int ELEM_PER_WI>
static void fused_scaleback_fast_kernel(
    const int32_t* __restrict__ gemm_result,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const OutputT* __restrict__ bias,
    OutputT* __restrict__ output,
    int64_t M,
    int64_t N,
    bool w_scale_is_scalar,
    bool has_bias,
    const at::Device& device
) {
    const int64_t col_tiles = N / ELEM_PER_WI;
    constexpr int WG_COLS = 8;
    constexpr int WG_ROWS = 4;
    const int64_t global_cols = ((col_tiles + WG_COLS - 1) / WG_COLS) * WG_COLS;
    const int64_t global_rows = ((M + WG_ROWS - 1) / WG_ROWS) * WG_ROWS;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(global_rows, global_cols),
                sycl::range<2>(WG_ROWS, WG_COLS)
            ),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                const int64_t row = item.get_global_id(0);
                const int64_t col_tile = item.get_global_id(1);
                if (row >= M || col_tile >= col_tiles) return;

                const int64_t col_start = col_tile * ELEM_PER_WI;
                const int64_t off = row * N + col_start;

                // Load int32 GEMM result (128 bytes for 32 int32s)
                simd<int32_t, ELEM_PER_WI> i32_vec =
                    block_load<int32_t, ELEM_PER_WI>(gemm_result + off);

                // Convert to float32
                simd<float, ELEM_PER_WI> f32_vec = i32_vec;

                // Multiply by row scale (scalar broadcast — read once per WI)
                float rs = x_scale[row];
                f32_vec = f32_vec * rs;

                // Multiply by column scales
                if (w_scale_is_scalar) {
                    f32_vec = f32_vec * w_scale[0];
                } else {
                    simd<float, ELEM_PER_WI> ws_vec;
                    ws_vec.copy_from(w_scale + col_start);
                    f32_vec = f32_vec * ws_vec;
                }

                // Add bias (if present)
                if (has_bias) {
                    simd<OutputT, ELEM_PER_WI> bias_vec;
                    bias_vec.copy_from(bias + col_start);
                    simd<float, ELEM_PER_WI> bias_f32 = bias_vec;
                    f32_vec = f32_vec + bias_f32;
                }

                // Convert to output dtype and store (64 bytes for 32 bf16s)
                simd<OutputT, ELEM_PER_WI> out_vec = f32_vec;
                block_store<OutputT, ELEM_PER_WI>(output + off, out_vec);
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_scaleback_int8_fast");
}

// ============================================================================
// Slow fallback for N not divisible by 32
// ============================================================================

template <typename OutputT>
static void fused_scaleback_slow_kernel(
    const int32_t* __restrict__ gemm_result,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const OutputT* __restrict__ bias,
    OutputT* __restrict__ output,
    int64_t M,
    int64_t N,
    bool w_scale_is_scalar,
    bool has_bias,
    const at::Device& device
) {
    constexpr int WG_SIZE = 32;
    const int64_t padded = ((M + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t row = item.get_global_id(0);
                if (row >= M) return;

                float rs = x_scale[row];
                const int64_t row_off = row * N;

                for (int64_t n = 0; n < N; ++n) {
                    float val = static_cast<float>(gemm_result[row_off + n]);
                    float ws = w_scale_is_scalar ? w_scale[0] : w_scale[n];
                    val = val * rs * ws;
                    if (has_bias) val += static_cast<float>(bias[n]);
                    output[row_off + n] = static_cast<OutputT>(val);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_scaleback_int8_slow");
}

// ============================================================================
// Public API
// ============================================================================

torch::Tensor fused_scaleback(
    torch::Tensor gemm_result,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    std::optional<torch::Tensor> bias,
    int64_t out_dtype_code
) {
    TORCH_CHECK(gemm_result.device().is_xpu(), "gemm_result must be on XPU");
    TORCH_CHECK(gemm_result.scalar_type() == torch::kInt32, "gemm_result must be int32");
    TORCH_CHECK(gemm_result.dim() == 2, "gemm_result must be 2D [M, N]");

    gemm_result = gemm_result.contiguous();
    x_scale = x_scale.to(torch::kFloat32).contiguous().reshape(-1);
    w_scale = w_scale.to(torch::kFloat32).contiguous().reshape(-1);

    const int64_t M = gemm_result.size(0);
    const int64_t N = gemm_result.size(1);
    bool w_scale_is_scalar = (w_scale.numel() == 1);
    bool has_bias = bias.has_value();

    TORCH_CHECK(x_scale.numel() == M, "x_scale must have M=", M, " elements");
    TORCH_CHECK(w_scale.numel() == 1 || w_scale.numel() == N, "w_scale must be scalar or [N]");

    torch::ScalarType out_dtype;
    switch (out_dtype_code) {
        case 0: out_dtype = torch::kFloat; break;
        case 1: out_dtype = torch::kHalf; break;
        case 2: out_dtype = torch::kBFloat16; break;
        default: out_dtype = torch::kBFloat16; break;
    }

    torch::Tensor output = torch::empty({M, N},
        torch::TensorOptions().dtype(out_dtype).device(gemm_result.device()));

    constexpr int FAST_ELEM = 32;
    bool use_fast = (N % FAST_ELEM == 0);

    torch::Tensor bias_c;
    if (has_bias) bias_c = bias->to(out_dtype).contiguous();

    #define DISPATCH_SCALEBACK(OT, bias_ptr) \
        if (use_fast) { \
            fused_scaleback_fast_kernel<OT, FAST_ELEM>( \
                gemm_result.data_ptr<int32_t>(), x_scale.data_ptr<float>(), \
                w_scale.data_ptr<float>(), bias_ptr, \
                reinterpret_cast<OT*>(output.data_ptr()), \
                M, N, w_scale_is_scalar, has_bias, gemm_result.device()); \
        } else { \
            fused_scaleback_slow_kernel<OT>( \
                gemm_result.data_ptr<int32_t>(), x_scale.data_ptr<float>(), \
                w_scale.data_ptr<float>(), bias_ptr, \
                reinterpret_cast<OT*>(output.data_ptr()), \
                M, N, w_scale_is_scalar, has_bias, gemm_result.device()); \
        }

    if (out_dtype == torch::kBFloat16) {
        const bf16* bp = has_bias ? reinterpret_cast<const bf16*>(bias_c.data_ptr()) : nullptr;
        DISPATCH_SCALEBACK(bf16, bp);
    } else if (out_dtype == torch::kHalf) {
        const fp16* bp = has_bias ? reinterpret_cast<const fp16*>(bias_c.data_ptr()) : nullptr;
        DISPATCH_SCALEBACK(fp16, bp);
    } else {
        const float* bp = has_bias ? bias_c.data_ptr<float>() : nullptr;
        DISPATCH_SCALEBACK(float, bp);
    }

    #undef DISPATCH_SCALEBACK
    return output;
}

}  // namespace int8_ops
}  // namespace omni_xpu
