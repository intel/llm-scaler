// ============================================================================
// SVDQuant Fused Post-Processing Kernels - Intel XPU ESIMD
// ============================================================================
// Memory-bound fusion kernels that eliminate intermediate memory round-trips
// in the SVDQuant W4A4 inference hot path.
//
// Two kernels:
//
// 1. fused_smooth_convert:
//    Fuses `x / smooth_factor` (bf16 element-wise division) + `.to(f16)` conversion
//    into a single kernel. Reads bf16 x once, divides by bf16 smooth_factor row,
//    writes f16 output directly. Eliminates one full tensor read+write cycle.
//    Saves: ~0.22s (smooth) + ~0.41s (convert) → fused ~0.35s = ~0.28s savings
//
// 2. fused_convert_add:
//    Fuses `out.copy_(f16_result)` (f16→bf16 conversion) + `out.add_(bf16_residual)`
//    into a single kernel. Reads f16 result + bf16 residual, writes bf16 output
//    in one pass. Eliminates one full tensor write+read cycle.
//    Saves: ~0.86s (copy_) + ~0.59s (add_) → fused ~0.55s = ~0.90s savings
//
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace svdq {

// ============================================================================
// Kernel 1: fused_smooth_convert (division version — legacy)
// ============================================================================
// Fuses element-wise division by smooth_factor + bf16→f16 dtype conversion.
//
// Input:  x [M, K] bf16 — activation tensor
//         smooth_factor [K] bf16 — per-channel smooth factor (broadcast over M)
// Output: [M, K] f16
//
// Each work-item processes a contiguous chunk of elements.
// We use a flat 1D decomposition over total elements, with each work-item
// processing 32 elements (one ESIMD vector width for bf16).
// smooth_factor is indexed by column: smooth[elem_idx % K].
// ============================================================================

static void fused_smooth_convert_kernel(
    const bf16* __restrict__ x,
    const bf16* __restrict__ smooth,
    fp16* __restrict__ output,
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    // Each work-item processes 32 elements (bf16 = 2 bytes, 32 elements = 64 bytes)
    constexpr int ELEM_PER_WI = 32;
    const int64_t total_elements = M * K;
    const int64_t total_wi = (total_elements + ELEM_PER_WI - 1) / ELEM_PER_WI;
    constexpr int WG_SIZE = 64;
    const int64_t padded = (total_wi + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t wi_id = item.get_global_id(0);
                if (wi_id >= total_wi) return;

                const int64_t elem_start = wi_id * ELEM_PER_WI;
                const int64_t remaining = total_elements - elem_start;

                if (remaining >= ELEM_PER_WI) {
                    // Full block: load 32 bf16 values from x
                    simd<bf16, ELEM_PER_WI> x_vec =
                        block_load<bf16, ELEM_PER_WI>(x + elem_start);

                    // Load smooth_factor values for these columns
                    // Column index = elem_start % K ... (elem_start + 31) % K
                    // If the 32 elements span a single row (common for K >= 32),
                    // we can compute the column offset and load contiguously from smooth
                    const int64_t col_start = elem_start % K;

                    simd<float, ELEM_PER_WI> x_f32 = x_vec;
                    simd<float, ELEM_PER_WI> result_f32;

                    if (col_start + ELEM_PER_WI <= K) {
                        // All 32 elements are within a single row — contiguous smooth load
                        simd<bf16, ELEM_PER_WI> smooth_vec =
                            block_load<bf16, ELEM_PER_WI>(smooth + col_start);
                        simd<float, ELEM_PER_WI> smooth_f32 = smooth_vec;
                        result_f32 = x_f32 / smooth_f32;
                    } else {
                        // Elements span a row boundary — per-element column lookup
                        #pragma unroll
                        for (int i = 0; i < ELEM_PER_WI; ++i) {
                            int64_t col = (elem_start + i) % K;
                            float sv = static_cast<float>(smooth[col]);
                            result_f32[i] = x_f32[i] / sv;
                        }
                    }

                    // Convert f32 → f16 and store
                    simd<fp16, ELEM_PER_WI> out_vec = result_f32;
                    block_store<fp16, ELEM_PER_WI>(output + elem_start, out_vec);
                } else {
                    // Partial block at end
                    for (int64_t i = 0; i < remaining; ++i) {
                        int64_t idx = elem_start + i;
                        int64_t col = idx % K;
                        float xv = static_cast<float>(x[idx]);
                        float sv = static_cast<float>(smooth[col]);
                        output[idx] = static_cast<fp16>(xv / sv);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_smooth_convert");
}

// ============================================================================
// Kernel 1b: fused_smooth_mul_convert (multiply-by-reciprocal version)
// ============================================================================
// Optimized version that uses pre-computed reciprocal of smooth_factor.
// Multiply is ~4x cheaper than divide on Intel XPU EU.
//
// Input:  x [M, K] bf16 — activation tensor
//         rcp_smooth [K] f16 — pre-computed 1/smooth_factor in f16
// Output: [M, K] f16
//
// The rcp_smooth is in f16 because the output is f16 — this avoids an
// extra bf16→f16 conversion for the smooth factor.
//
// Row-based decomposition: each work-item processes one row-tile of K elements.
// For K=3840 with ELEM_PER_WI=32, that's 120 iterations per row per WI.
// With M=4096, total_wi = M * (K / ELEM_PER_WI) = 4096 * 120 = 491520.
// ============================================================================

static void fused_smooth_mul_convert_kernel(
    const bf16* __restrict__ x,
    const fp16* __restrict__ rcp_smooth,
    fp16* __restrict__ output,
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    constexpr int ELEM_PER_WI = 32;
    const int64_t total_elements = M * K;
    const int64_t total_wi = (total_elements + ELEM_PER_WI - 1) / ELEM_PER_WI;
    constexpr int WG_SIZE = 64;
    const int64_t padded = (total_wi + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t wi_id = item.get_global_id(0);
                if (wi_id >= total_wi) return;

                const int64_t elem_start = wi_id * ELEM_PER_WI;
                const int64_t remaining = total_elements - elem_start;

                if (remaining >= ELEM_PER_WI) {
                    simd<bf16, ELEM_PER_WI> x_vec =
                        block_load<bf16, ELEM_PER_WI>(x + elem_start);

                    const int64_t col_start = elem_start % K;
                    simd<float, ELEM_PER_WI> x_f32 = x_vec;
                    simd<float, ELEM_PER_WI> result_f32;

                    if (col_start + ELEM_PER_WI <= K) {
                        simd<fp16, ELEM_PER_WI> rcp_vec =
                            block_load<fp16, ELEM_PER_WI>(rcp_smooth + col_start);
                        simd<float, ELEM_PER_WI> rcp_f32 = rcp_vec;
                        result_f32 = x_f32 * rcp_f32;
                    } else {
                        // Elements span a row boundary — per-element column lookup
                        #pragma unroll
                        for (int i = 0; i < ELEM_PER_WI; ++i) {
                            int64_t col = (elem_start + i) % K;
                            float rv = static_cast<float>(rcp_smooth[col]);
                            result_f32[i] = x_f32[i] * rv;
                        }
                    }

                    // Convert f32 → f16 and store
                    simd<fp16, ELEM_PER_WI> out_vec = result_f32;
                    block_store<fp16, ELEM_PER_WI>(output + elem_start, out_vec);
                } else {
                    // Partial block at end
                    for (int64_t i = 0; i < remaining; ++i) {
                        int64_t idx = elem_start + i;
                        int64_t col = idx % K;
                        float xv = static_cast<float>(x[idx]);
                        float rv = static_cast<float>(rcp_smooth[col]);
                        output[idx] = static_cast<fp16>(xv * rv);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_smooth_mul_convert");
}

// ============================================================================
// Kernel 2: fused_convert_add
// ============================================================================
// Fuses f16→bf16 conversion + bf16 element-wise addition.
//
// Input:  result [Mr, Nr] f16  — GEMM output (may be larger than needed)
//         residual [Mo, No] bf16 — LoRA residual (same size as output)
// Output: out [Mo, No] bf16 — output = bf16(result[:Mo, :No]) + residual
//
// When result and output have same width (Nr == No), we can use a flat 1D
// decomposition. When widths differ, we use a 2D index.
//
// Common case: M_out × N_out elements, result stride = Nr, output stride = No.
// ============================================================================

static void fused_convert_add_kernel_flat(
    const fp16* __restrict__ result,
    const bf16* __restrict__ residual,
    bf16* __restrict__ output,
    int64_t total_elements,
    const at::Device& device
) {
    // Flat path: all three tensors are contiguous over total_elements
    constexpr int ELEM_PER_WI = 32;
    const int64_t total_wi = (total_elements + ELEM_PER_WI - 1) / ELEM_PER_WI;
    constexpr int WG_SIZE = 64;
    const int64_t padded = (total_wi + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t wi_id = item.get_global_id(0);
                if (wi_id >= total_wi) return;

                const int64_t elem_start = wi_id * ELEM_PER_WI;
                const int64_t remaining = total_elements - elem_start;

                if (remaining >= ELEM_PER_WI) {
                    // Load f16 result
                    simd<fp16, ELEM_PER_WI> res_f16 =
                        block_load<fp16, ELEM_PER_WI>(result + elem_start);
                    // Load bf16 residual
                    simd<bf16, ELEM_PER_WI> resid_bf16 =
                        block_load<bf16, ELEM_PER_WI>(residual + elem_start);

                    // Convert both to f32, add, convert to bf16
                    simd<float, ELEM_PER_WI> res_f32 = res_f16;
                    simd<float, ELEM_PER_WI> resid_f32 = resid_bf16;
                    simd<float, ELEM_PER_WI> sum_f32 = res_f32 + resid_f32;

                    simd<bf16, ELEM_PER_WI> out_bf16 = sum_f32;
                    block_store<bf16, ELEM_PER_WI>(output + elem_start, out_bf16);
                } else {
                    for (int64_t i = 0; i < remaining; ++i) {
                        int64_t idx = elem_start + i;
                        float rv = static_cast<float>(result[idx]);
                        float resv = static_cast<float>(residual[idx]);
                        output[idx] = static_cast<bf16>(rv + resv);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_convert_add");
}

// Strided path: result has different row stride than output/residual
static void fused_convert_add_kernel_strided(
    const fp16* __restrict__ result,
    int64_t result_stride,      // number of f16 elements per row in result
    const bf16* __restrict__ residual,
    int64_t residual_stride,    // number of bf16 elements per row in residual
    bf16* __restrict__ output,
    int64_t output_stride,      // number of bf16 elements per row in output
    int64_t M,                  // rows to process
    int64_t N,                  // cols to process
    const at::Device& device
) {
    // Each work-item processes 32 contiguous elements within a row
    constexpr int ELEM_PER_WI = 32;
    const int64_t cols_per_row = (N + ELEM_PER_WI - 1) / ELEM_PER_WI;
    const int64_t total_wi = M * cols_per_row;
    constexpr int WG_SIZE = 64;
    const int64_t padded = (total_wi + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t wi_id = item.get_global_id(0);
                if (wi_id >= total_wi) return;

                const int64_t row = wi_id / cols_per_row;
                const int64_t col_block = wi_id % cols_per_row;
                const int64_t col_start = col_block * ELEM_PER_WI;
                const int64_t remaining = N - col_start;

                const fp16* res_row = result + row * result_stride + col_start;
                const bf16* resid_row = residual + row * residual_stride + col_start;
                bf16* out_row = output + row * output_stride + col_start;

                if (remaining >= ELEM_PER_WI) {
                    simd<fp16, ELEM_PER_WI> res_f16 =
                        block_load<fp16, ELEM_PER_WI>(res_row);
                    simd<bf16, ELEM_PER_WI> resid_bf16 =
                        block_load<bf16, ELEM_PER_WI>(resid_row);

                    simd<float, ELEM_PER_WI> res_f32 = res_f16;
                    simd<float, ELEM_PER_WI> resid_f32 = resid_bf16;
                    simd<float, ELEM_PER_WI> sum_f32 = res_f32 + resid_f32;

                    simd<bf16, ELEM_PER_WI> out_bf16 = sum_f32;
                    block_store<bf16, ELEM_PER_WI>(out_row, out_bf16);
                } else {
                    for (int64_t i = 0; i < remaining; ++i) {
                        float rv = static_cast<float>(res_row[i]);
                        float resv = static_cast<float>(resid_row[i]);
                        out_row[i] = static_cast<bf16>(rv + resv);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_convert_add_strided");
}


// ============================================================================
// Public API
// ============================================================================

torch::Tensor fused_smooth_convert(
    const torch::Tensor& x,              // [M, K] bf16
    const torch::Tensor& smooth_factor   // [K] bf16
) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(smooth_factor.is_contiguous(), "smooth_factor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
                "x must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(smooth_factor.scalar_type() == torch::kBFloat16,
                "smooth_factor must be bfloat16, got ", smooth_factor.scalar_type());
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K], got ", x.dim(), "D");

    int64_t K = x.size(1);
    // smooth_factor can be [K] or [1, K]
    int64_t sf_numel = smooth_factor.numel();
    TORCH_CHECK(sf_numel == K,
                "smooth_factor size (", sf_numel, ") must match x dim 1 (", K, ")");

    int64_t M = x.size(0);

    auto output = torch::empty({M, K},
        torch::TensorOptions().dtype(torch::kFloat16).device(x.device()));

    fused_smooth_convert_kernel(
        reinterpret_cast<const bf16*>(x.data_ptr()),
        reinterpret_cast<const bf16*>(smooth_factor.data_ptr()),
        reinterpret_cast<fp16*>(output.data_ptr()),
        M, K, x.device()
    );

    return output;
}


torch::Tensor fused_smooth_mul_convert(
    const torch::Tensor& x,            // [M, K] bf16
    const torch::Tensor& rcp_smooth    // [K] f16 — pre-computed 1/smooth_factor
) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(rcp_smooth.is_contiguous(), "rcp_smooth must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
                "x must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(rcp_smooth.scalar_type() == torch::kFloat16,
                "rcp_smooth must be float16, got ", rcp_smooth.scalar_type());
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K], got ", x.dim(), "D");

    int64_t K = x.size(1);
    int64_t sf_numel = rcp_smooth.numel();
    TORCH_CHECK(sf_numel == K,
                "rcp_smooth size (", sf_numel, ") must match x dim 1 (", K, ")");

    int64_t M = x.size(0);

    auto output = torch::empty({M, K},
        torch::TensorOptions().dtype(torch::kFloat16).device(x.device()));

    fused_smooth_mul_convert_kernel(
        reinterpret_cast<const bf16*>(x.data_ptr()),
        reinterpret_cast<const fp16*>(rcp_smooth.data_ptr()),
        reinterpret_cast<fp16*>(output.data_ptr()),
        M, K, x.device()
    );

    return output;
}


void fused_convert_add(
    torch::Tensor& out,            // [M_out, N_out] bf16 — output (written in-place)
    const torch::Tensor& result,   // [Mr, Nr] f16 — GEMM result
    const torch::Tensor& residual  // [M_out, N_out] bf16 — LoRA residual
) {
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(result.is_contiguous(), "result must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
                "out must be bfloat16, got ", out.scalar_type());
    TORCH_CHECK(result.scalar_type() == torch::kFloat16,
                "result must be float16, got ", result.scalar_type());
    TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
                "residual must be bfloat16, got ", residual.scalar_type());
    TORCH_CHECK(out.dim() == 2, "out must be 2D");
    TORCH_CHECK(result.dim() == 2, "result must be 2D");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2D");

    int64_t M_out = out.size(0);
    int64_t N_out = out.size(1);

    TORCH_CHECK(residual.size(0) >= M_out && residual.size(1) >= N_out,
                "residual must be at least [", M_out, ", ", N_out, "]");
    TORCH_CHECK(result.size(0) >= M_out && result.size(1) >= N_out,
                "result must be at least [", M_out, ", ", N_out, "]");

    int64_t result_N = result.size(1);
    int64_t residual_N = residual.size(1);

    // Check if we can use the flat (fastest) path
    if (result_N == N_out && residual_N == N_out) {
        // All tensors have same width — flat path
        fused_convert_add_kernel_flat(
            reinterpret_cast<const fp16*>(result.data_ptr()),
            reinterpret_cast<const bf16*>(residual.data_ptr()),
            reinterpret_cast<bf16*>(out.data_ptr()),
            M_out * N_out,
            out.device()
        );
    } else {
        // Different widths — strided path
        fused_convert_add_kernel_strided(
            reinterpret_cast<const fp16*>(result.data_ptr()),
            result_N,
            reinterpret_cast<const bf16*>(residual.data_ptr()),
            residual_N,
            reinterpret_cast<bf16*>(out.data_ptr()),
            N_out,
            M_out, N_out,
            out.device()
        );
    }
}


}  // namespace svdq
}  // namespace omni_xpu
