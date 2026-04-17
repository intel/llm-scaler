/* fused_add_rms_norm.h — Fused residual add + RMSNorm (Gemma-style).
 *
 * For decode (bsz=1): residual[1,K] += hidden[1,K]; output[1,K] = rmsnorm(residual) * weight
 * Gemma convention: weight is pre-adjusted (w+1.0 already applied by caller).
 *
 * Single WG, 1 thread. K=2048 → 4 iterations with VL=512.
 * Two-pass: pass 1 = add + sum_sq; pass 2 = normalize + write output.
 * Residual updated in-place.
 */

#pragma once
#include "utils.h"

struct FusedAddRmsNorm_kernel {
    fp16*       hidden_ptr;    // [1, K] — input, also used as output
    fp16*       residual_ptr;  // [1, K] — updated in-place
    const fp16* weight_ptr;    // [K] — Gemma norm weight (w+1.0)
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 512;
        int n_chunks = K / VL;

        // Pass 1: residual += hidden, accumulate sum_sq
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;

            // Write residual in-place
            block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));

            simd<float, VL> sq = added * added;
            sq.select<256,1>(0) += sq.select<256,1>(256);
            sq.select<128,1>(0) += sq.select<128,1>(128);
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0) += sq.select<8,1>(8);
            sq.select<4,1>(0) += sq.select<4,1>(4);
            sq.select<2,1>(0) += sq.select<2,1>(2);
            sum_sq += (float)sq[0] + (float)sq[1];
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize and write output (reuse hidden_ptr as output)
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + offset);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<fp16, VL>(hidden_ptr + offset, simd<fp16, VL>(normed));
        }
    }
};

inline void fused_add_rms_norm_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* weight_ptr,
    int K, float eps, sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(1, 1),
            FusedAddRmsNorm_kernel{hidden_ptr, residual_ptr, weight_ptr, K, eps});
    });
}
