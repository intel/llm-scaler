/* accum_norm_add_norm.h — Fused (top_k sum) + Norm × w1 + Add to h1 + Norm × w2.
 *
 * Designed for gemma4 MoE outlet with no-accum MoE kernel:
 *   For each token (here: M=1 decode, single WG):
 *     h2_raw = sum over k=0..top_k-1 of routed_output[k, :]    (replaces moe_accumulate)
 *     h2_normed = (h2_raw / rms(h2_raw)) * w1                   (post_ff_norm_2)
 *     h1 ← h2_normed + h1                                       (in-place add)
 *     out = (h1 / rms(h1)) * w2                                 (post_ff_norm)
 *
 * Replaces 3 launches (moe_accumulate + esimd_rms_norm + esimd_fused_add_rms_norm)
 * or 2 launches (moe_accumulate + esimd_norm_add_norm) with 1.
 *
 * Layout (decode M=1):
 *   routed_output: [top_k, K] fp16   (sg_routed_output narrow)
 *   h1:            [1, K]    fp16    (in-place)
 *   w1, w2:        [K]       fp16
 *   out:           [1, K]    fp16
 *
 * Single WG, single thread. K-streamed loads with chunk cache for h2_raw.
 */

#pragma once
#include "utils.h"

template<int VL, int MAX_CHUNKS>
struct AccumNormAddNorm_kernel {
    const fp16* routed_output;   // [top_k, K]
    fp16*       h1_ptr;           // [1, K] in-place
    const fp16* w1_ptr;
    const fp16* w2_ptr;
    fp16*       out_ptr;
    int K;
    int top_k;
    float eps1, eps2;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n_chunks = K / VL;
        simd<float, VL> h2_chunks[MAX_CHUNKS];

        // Pass 1: load all top_k routed_output rows for this token; sum into
        // h2_chunks; accumulate sum_sq for RMS_1 along the way.
        float sum_sq_1 = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> s = 0.0f;
            for (int k = 0; k < top_k; k++) {
                simd<float, VL> r = block_load<fp16, VL>(
                    routed_output + (size_t)k * K + offset);
                s += r;
            }
            h2_chunks[c] = s;
            simd<float, VL> sq = s * s;
            sum_sq_1 += reduce<float>(sq, std::plus<>());
        }

        float inv_rms_1 = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq_1 / (float)K + eps1))[0];

        // Pass 2: h1 ← h2_normed_w1 + h1; accumulate sum_sq_2
        float sum_sq_2 = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> w1 = block_load<fp16, VL>(w1_ptr + offset);
            simd<float, VL> h1 = block_load<fp16, VL>(h1_ptr + offset);
            simd<float, VL> h2_normed = h2_chunks[c] * inv_rms_1 * w1;
            simd<float, VL> h1_new = h2_normed + h1;
            block_store<fp16, VL>(h1_ptr + offset, simd<fp16, VL>(h1_new));
            simd<float, VL> sq = h1_new * h1_new;
            sum_sq_2 += reduce<float>(sq, std::plus<>());
        }

        float inv_rms_2 = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq_2 / (float)K + eps2))[0];

        // Pass 3: out ← h1 * inv_rms_2 * w2
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h1 = block_load<fp16, VL>(h1_ptr + offset);
            simd<float, VL> w2 = block_load<fp16, VL>(w2_ptr + offset);
            simd<float, VL> out = h1 * inv_rms_2 * w2;
            block_store<fp16, VL>(out_ptr + offset, simd<fp16, VL>(out));
        }
    }
};

inline void accum_norm_add_norm_host(
    const fp16* routed_output,
    fp16*       h1_ptr,
    const fp16* w1_ptr,
    const fp16* w2_ptr,
    fp16*       out_ptr,
    int K, int top_k,
    float eps1, float eps2,
    sycl::queue& q)
{
    #define LAUNCH_ANAN(V, MC)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(1, 1),                 AccumNormAddNorm_kernel<V, MC>{                     routed_output, h1_ptr, w1_ptr, w2_ptr, out_ptr,                     K, top_k, eps1, eps2});         });

    if (K % 512 == 0) {
        int mc = K / 512;
        if      (mc <= 4)  { LAUNCH_ANAN(512, 4)  }
        else if (mc <= 8)  { LAUNCH_ANAN(512, 8)  }
        else               { LAUNCH_ANAN(512, 16) }
    } else if (K % 256 == 0) {
        int mc = K / 256;
        if      (mc <= 8)  { LAUNCH_ANAN(256, 8)  }
        else if (mc <= 16) { LAUNCH_ANAN(256, 16) }
        else               { LAUNCH_ANAN(256, 32) }
    } else if (K % 128 == 0) {
        int mc = K / 128;
        if      (mc <= 16) { LAUNCH_ANAN(128, 16) }
        else               { LAUNCH_ANAN(128, 32) }
    } else {
        int mc = K / 64;
        if (mc > 64) mc = 64;
        if (mc <= 32) { LAUNCH_ANAN(64, 32) }
        else          { LAUNCH_ANAN(64, 64) }
    }

    #undef LAUNCH_ANAN
}
