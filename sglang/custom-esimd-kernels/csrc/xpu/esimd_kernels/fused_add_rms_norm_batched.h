/* fused_add_rms_norm_batched.h — Batched Fused residual add + RMSNorm.
 *
 * Multi-row version of fused_add_rms_norm.h:
 *   residual[i] += hidden[i]   (in-place)
 *   hidden[i] = rmsnorm(residual[i]) * weight
 *
 * Grid: rows WGs, 1 thread each. K=2048 → 4 iterations with VL=512.
 * Replaces the PyTorch dispatch chain with a single, much faster kernel.
 */

#pragma once
#include "utils.h"

// See fused_add_rms_norm.h for why VL is a host-chosen template
// parameter rather than a fixed 512.
template <int VL>
struct FusedAddRmsNorm_batched_kernel {
    fp16*       hidden_ptr;    // [rows, K] — input and output
    fp16*       residual_ptr;  // [rows, K] — updated in-place
    const fp16* weight_ptr;    // [K] — Gemma norm weight (w+1.0)
    int rows;
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int row = item.get_group(0);
        if (row >= rows) return;

        int n_chunks = K / VL;
        const int base = row * K;

        // Pass 1: residual += hidden, accumulate sum_sq
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = base + c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;

            block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));

            // VL-generic pairwise tree (the previous hand-rolled version
            // started at select<256> and so was only valid for VL == 512).
            sum_sq += sycl::ext::intel::esimd::detail::sum<float, float, VL>(added * added);
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize and write output
        for (int c = 0; c < n_chunks; c++) {
            int offset = base + c * VL;
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + c * VL);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<fp16, VL>(hidden_ptr + offset, simd<fp16, VL>(normed));
        }
    }
};

inline void fused_add_rms_norm_batched_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* weight_ptr,
    int rows, int K, float eps, sycl::queue& q)
{
    #define LAUNCH_FARNB(V)                                                   \
        q.submit([&](sycl::handler& cgh) {                                    \
            cgh.parallel_for(                                                 \
                sycl::nd_range<1>({(size_t)rows}, {1}),                       \
                FusedAddRmsNorm_batched_kernel<V>{                            \
                    hidden_ptr, residual_ptr, weight_ptr, rows, K, eps});     \
        });

    if      (K % 512 == 0) { LAUNCH_FARNB(512) }
    else if (K % 256 == 0) { LAUNCH_FARNB(256) }
    else if (K % 128 == 0) { LAUNCH_FARNB(128) }
    else                   { LAUNCH_FARNB(64)  }

    #undef LAUNCH_FARNB
}
