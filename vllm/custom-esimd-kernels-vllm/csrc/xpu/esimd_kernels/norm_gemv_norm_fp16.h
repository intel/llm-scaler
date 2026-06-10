/* norm_gemv_norm_fp16.h — Fused (RMS share) Norm + FP16 GEMV + Second Norm.
 *
 * Designed for Gemma4 MoE branch:
 *   residual → 1/rms(residual)
 *   ├─ router_logits = ((residual * inv_rms) * scale_with_root) @ proj_w^T  (fp16 GEMV)
 *   └─ moe_input     = (residual * inv_rms) * pre_ff_norm_2_w               (fp16)
 *
 * Replaces 3 launches (esimd_rms_norm × 2 + esimd_gemv_fp16) with 1.
 *
 * Layout:
 *   residual:        [1, K] fp16
 *   scale_with_root: [K]    fp16   (router norm has_weight=False; constant root_size already folded in)
 *   proj_w:          [N, K] fp16   (router projection — fp16, contiguous)
 *   pre_ff_w:        [K]    fp16   (pre_feedforward_layernorm_2 weight — standard RMSNorm: x*w/rms)
 *   router_logits:   [1, N] fp16
 *   moe_input:       [1, K] fp16
 *
 * Grid: launches max(N, num_kchunks_for_moe_norm) work-groups.
 * WG i in [0, N) computes router logit i (and writes the moe_input chunk
 *   for i < n_chunks).
 * Single SYCL submit, single kernel — only 1 launch.
 *
 * Pattern:
 *   Loop 1 (K/VL iters): load residual chunk, accumulate sum_sq, store chunk in registers.
 *   Reduce sum_sq → inv_rms (private to this WG; same across all WGs since they read the same residual).
 *   Loop 2 (K/VL iters):
 *       normed = res_chunk * inv_rms                          // shared
 *       if (n == 0) moe_input_chunk = normed * pre_ff_w_chunk // only 1 WG writes
 *       router_in = normed * scale_with_root_chunk
 *       acc += router_in * proj_w_chunk                       // GEMV partial
 *   Reduce acc → router_logits[n]
 *
 * Notes:
 * - Each WG re-reads `residual` and `scale_with_root` and `pre_ff_w` from L3 cache (after first WG warms it). This is the
 *   same trade-off as resadd_norm_gemv_fp8_pert.
 * - Only WG 0 writes moe_input. If N=0 there is no router work; we
 *   require N >= 1 (gemma4 num_experts=128, N is always positive).
 * - `scale_with_root` and `pre_ff_w` are independent: keeping them
 *   separate avoids forcing the caller to merge them on the host (and
 *   they would each be applied at different points in the pipeline).
 */

#pragma once
#include "utils.h"

template<int VL, int MAX_CHUNKS>
struct NormGemvNorm_fp16_kernel {
    const fp16* residual;       // [1, K]
    const fp16* scale_with_root;// [K]
    const fp16* proj_w;         // [N, K]
    const fp16* pre_ff_w;       // [K]
    fp16*       router_logits;  // [1, N]
    fp16*       moe_input;      // [1, K]
    int N, K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        int n_chunks = K / VL;
        simd<float, VL> res_chunks[MAX_CHUNKS];

        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> r = block_load<fp16, VL>(residual + offset);
            res_chunks[c] = r;
            simd<float, VL> sq = r * r;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        simd<float, VL> acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> normed = res_chunks[c] * inv_rms;

            // Only WG 0 writes moe_input (= normed * pre_ff_w)
            if (n == 0) {
                simd<float, VL> pw = block_load<fp16, VL>(pre_ff_w + offset);
                simd<float, VL> moe_chunk = normed * pw;
                block_store<fp16, VL>(moe_input + offset, simd<fp16, VL>(moe_chunk));
            }

            simd<float, VL> sw = block_load<fp16, VL>(scale_with_root + offset);
            simd<float, VL> router_in = normed * sw;

            simd<float, VL> wf = block_load<fp16, VL>(proj_w + (size_t)n * K + offset);
            acc += router_in * wf;
        }

        float dot = reduce<float>(acc, std::plus<>());
        router_logits[n] = fp16(dot);
    }
};

inline void norm_gemv_norm_fp16_host(
    const fp16* residual,
    const fp16* scale_with_root,
    const fp16* proj_w,
    const fp16* pre_ff_w,
    fp16*       router_logits,
    fp16*       moe_input,
    int N, int K,
    float eps,
    sycl::queue& q) {

    #define LAUNCH_NGN(V, MC)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(N, 1),                 NormGemvNorm_fp16_kernel<V, MC>{                     residual, scale_with_root, proj_w, pre_ff_w,                     router_logits, moe_input,                     N, K, eps});         });

    if (K % 512 == 0) {
        int mc = K / 512;
        if      (mc <= 4)  { LAUNCH_NGN(512, 4)  }
        else if (mc <= 8)  { LAUNCH_NGN(512, 8)  }
        else               { LAUNCH_NGN(512, 16) }
    } else if (K % 256 == 0) {
        int mc = K / 256;
        if      (mc <= 8)  { LAUNCH_NGN(256, 8)  }
        else if (mc <= 16) { LAUNCH_NGN(256, 16) }
        else               { LAUNCH_NGN(256, 32) }
    } else if (K % 128 == 0) {
        int mc = K / 128;
        if      (mc <= 16) { LAUNCH_NGN(128, 16) }
        else               { LAUNCH_NGN(128, 32) }
    } else {
        // Caller must ensure K % 128 == 0; this should not be reached.
        // Use 64 as a last resort.
        int mc = K / 64;
        if (mc > 64) mc = 64;
        if (mc <= 32)      { LAUNCH_NGN(64, 32) }
        else               { LAUNCH_NGN(64, 64) }
    }

    #undef LAUNCH_NGN
}
