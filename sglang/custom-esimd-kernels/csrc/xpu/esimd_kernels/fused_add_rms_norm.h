/* fused_add_rms_norm.h — Fused residual add + RMSNorm (Gemma-style).
 *
 * For decode (bsz=1): residual[1,K] += hidden[1,K]; output[1,K] = rmsnorm(residual) * weight
 * Gemma convention: weight is pre-adjusted (w+1.0 already applied by caller).
 *
 * Single WG, 1 thread. K=2048 → 4 iterations with VL=512.
 * VL is chosen by the host from K rather than fixed at 512: the loop is a whole
 * number of VL-wide chunks with no tail handling, so a hidden size that is not
 * a multiple of VL would silently drop its last K % VL elements from BOTH the
 * residual add and the normalisation. That is not hypothetical — gemma-4-31B
 * has hidden_size 5376 (= 512*10 + 256), so the fixed-512 version dropped 256
 * of 5376 channels in every decoder layer.
 * Two-pass: pass 1 = add + sum_sq; pass 2 = normalize + write output.
 * Residual updated in-place.
 */

#pragma once
#include "utils.h"

template <int VL>
struct FusedAddRmsNorm_kernel {
    fp16*       hidden_ptr;    // [1, K] — input, also used as output
    fp16*       residual_ptr;  // [1, K] — updated in-place
    const fp16* weight_ptr;    // [K] — Gemma norm weight (w+1.0)
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
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

            // VL-generic pairwise tree (the previous hand-rolled version
            // started at select<256> and so was only valid for VL == 512).
            sum_sq += sycl::ext::intel::esimd::detail::sum<float, float, VL>(added * added);
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
    #define LAUNCH_FARN(V)                                                    \
        q.submit([&](sycl::handler& cgh) {                                    \
            cgh.parallel_for(                                                 \
                sycl::nd_range<1>(1, 1),                                      \
                FusedAddRmsNorm_kernel<V>{                                    \
                    hidden_ptr, residual_ptr, weight_ptr, K, eps});           \
        });

    if      (K % 512 == 0) { LAUNCH_FARN(512) }
    else if (K % 256 == 0) { LAUNCH_FARN(256) }
    else if (K % 128 == 0) { LAUNCH_FARN(128) }
    else                   { LAUNCH_FARN(64)  }

    #undef LAUNCH_FARN
}
