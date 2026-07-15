/* norm_add_norm.h — Fused (RMSNorm × w1) + Add + (RMSNorm × w2).
 *
 * Designed for gemma4 MoE-output / attn-output fuse points:
 *   h2_raw → rms_norm × w1                                      (replaces 1)
 *   h1     ← h2_normed + h1                                     (in-place add)
 *   output  = rms_norm(h1) × w2                                 (replaces 1)
 *
 * Replaces 2 launches (esimd_rms_norm + esimd_fused_add_rms_norm) with 1.
 *
 * Layout:
 *   h2_raw_ptr: [1, K] fp16 (read; data stays in L3 across the 2 reads)
 *   h1_ptr:     [1, K] fp16 (in-place: h1 ← h2_normed_w1 + h1)
 *   w1_ptr:     [K] fp16
 *   w2_ptr:     [K] fp16
 *   out_ptr:    [1, K] fp16
 *
 * Single WG, single thread, no register cache. Data is streamed through
 * the kernel three times (h2_raw twice for sum_sq + apply, h1 twice for
 * apply + final norm) — all of K=2816 fp16 (~5.6KB) easily fits in L3, so
 * this trades a small repeat-fetch cost for a much smaller GRF footprint
 * (no MAX_CHUNKS-wide register array).
 *
 * Earlier versions cached h2_raw chunks in a `simd<float, VL>[MAX_CHUNKS]`
 * register array; for K=2816 / VL=256 this picked MAX_CHUNKS=16 → 16 KB of
 * GRF per thread, which exceeds BMG's per-thread GRF (~12 KB) and triggered
 * Level Zero `UR_RESULT_ERROR_OUT_OF_RESOURCES` after running for a while
 * under server load. The streamed layout below has zero register cache and
 * is observed to run cleanly across long evals (see /llm/models/test/gemma.sh
 * + gsm8k_eval).
 */

#pragma once
#include "utils.h"

template<int VL>
struct NormAddNorm_kernel {
    const fp16* h2_raw_ptr;
    fp16*       h1_ptr;
    const fp16* w1_ptr;
    const fp16* w2_ptr;
    fp16*       out_ptr;
    int K;
    float eps1;
    float eps2;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n_chunks = K / VL;

        // Pass 1: stream-load h2_raw, compute sum_sq_1.
        float sum_sq_1 = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h2 = block_load<fp16, VL>(h2_raw_ptr + offset);
            simd<float, VL> sq = h2 * h2;
            sum_sq_1 += reduce<float>(sq, std::plus<>());
        }

        float inv_rms_1 = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq_1 / (float)K + eps1))[0];

        // Pass 2: re-load h2_raw (L3 hot), apply norm × w1, add h1, write h1
        // in place; accumulate sum_sq_2 on the new h1.
        float sum_sq_2 = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h2 = block_load<fp16, VL>(h2_raw_ptr + offset);
            simd<float, VL> w1 = block_load<fp16, VL>(w1_ptr + offset);
            simd<float, VL> h1 = block_load<fp16, VL>(h1_ptr + offset);
            simd<float, VL> h2_normed = h2 * inv_rms_1 * w1;
            simd<float, VL> h1_new = h2_normed + h1;
            block_store<fp16, VL>(h1_ptr + offset, simd<fp16, VL>(h1_new));
            simd<float, VL> sq = h1_new * h1_new;
            sum_sq_2 += reduce<float>(sq, std::plus<>());
        }

        float inv_rms_2 = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq_2 / (float)K + eps2))[0];

        // Pass 3: out ← h1 * inv_rms_2 * w2 (h1 freshly written above; L3 hot).
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h1 = block_load<fp16, VL>(h1_ptr + offset);
            simd<float, VL> w2 = block_load<fp16, VL>(w2_ptr + offset);
            simd<float, VL> out = h1 * inv_rms_2 * w2;
            block_store<fp16, VL>(out_ptr + offset, simd<fp16, VL>(out));
        }
    }
};

inline void norm_add_norm_host(
    const fp16* h2_raw_ptr,
    fp16*       h1_ptr,
    const fp16* w1_ptr,
    const fp16* w2_ptr,
    fp16*       out_ptr,
    int K,
    float eps1,
    float eps2,
    sycl::queue& q)
{
    #define LAUNCH_NAN(V)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(1, 1),                 NormAddNorm_kernel<V>{                     h2_raw_ptr, h1_ptr, w1_ptr, w2_ptr, out_ptr,                     K, eps1, eps2});         });

    if      (K % 512 == 0) { LAUNCH_NAN(512) }
    else if (K % 256 == 0) { LAUNCH_NAN(256) }
    else if (K % 128 == 0) { LAUNCH_NAN(128) }
    else                   { LAUNCH_NAN(64)  }

    #undef LAUNCH_NAN
}
