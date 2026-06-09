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

// ============================================================================
// V2: Templated VL with tail handling for non-aligned K (e.g. K=2816)
// ============================================================================
template<int VL>
struct FusedAddRmsNorm_v2_kernel {
    fp16*       hidden_ptr;
    fp16*       residual_ptr;
    const fp16* weight_ptr;
    int K;
    float eps;



    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int k_aligned = (K / VL) * VL;

        // Pass 1: residual += hidden, accumulate sum_sq
        float sum_sq = 0.0f;
        for (int k = 0; k < k_aligned; k += VL) {
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + k);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k);
            simd<float, VL> added = h + r;
            block_store<fp16, VL>(residual_ptr + k, simd<fp16, VL>(added));
            simd<float, VL> sq = added * added;
            sum_sq += reduce<float>(sq, std::plus<>());
        }
        // Tail
        if (k_aligned < K) {
            int k_tail = K - VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + k_tail);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k_tail);
            simd<float, VL> added = h + r;
            block_store<fp16, VL>(residual_ptr + k_tail, simd<fp16, VL>(added));
            // Only accumulate the tail portion (avoid double-counting overlap)
            int overlap = k_aligned - k_tail;
            simd<float, VL> sq = added * added;
            for (int z = 0; z < overlap; z++) sq[z] = 0.0f;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize and write output
        for (int k = 0; k < k_aligned; k += VL) {
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + k);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<fp16, VL>(hidden_ptr + k, simd<fp16, VL>(normed));
        }
        if (k_aligned < K) {
            int k_tail = K - VL;
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k_tail);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + k_tail);
            simd<float, VL> normed = r * inv_rms * w;
            // Only write tail portion to avoid corrupting overlap region
            // (overlap region was already correctly written in main loop)
            // Since we read from residual (already correct) and write to
            // hidden (output), and the main loop already wrote [0..k_aligned),
            // writing the full VL from k_tail overwrites [k_tail..k_tail+VL)
            // = [K-VL..K). The overlap [k_tail..k_aligned) gets the same
            // value (same residual, same weight, same inv_rms), so it's safe.
            block_store<fp16, VL>(hidden_ptr + k_tail, simd<fp16, VL>(normed));
        }
    }
};

// ============================================================================
// V3: Scaled variant — fuses (hs + r) * scalar, then RMSNorm * weight.
//     residual ← (hs + r) * scalar  (in-place)
//     hidden   ← rmsnorm(residual) * weight
// Used by gemma4 cross-layer fuse: layer N's `final_add + scalar_mul` and
// layer N+1's `input_norm` collapse into one kernel call.
//
// Note: rmsnorm(s*x) = sign(s) * rmsnorm(x), so the math is equivalent to
// computing the scaled residual first, then norming. We still scale the
// stored residual so subsequent reads (e.g. router on `residual`) see the
// correctly scaled value.
// ============================================================================
template<int VL>
struct FusedScaledAddRmsNorm_kernel {
    fp16*       hidden_ptr;
    fp16*       residual_ptr;
    const fp16* weight_ptr;
    int K;
    float eps;
    float scalar;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int k_aligned = (K / VL) * VL;

        // Pass 1: scaled add + sum_sq
        float sum_sq = 0.0f;
        for (int k = 0; k < k_aligned; k += VL) {
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + k);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k);
            simd<float, VL> added = (h + r) * scalar;
            block_store<fp16, VL>(residual_ptr + k, simd<fp16, VL>(added));
            simd<float, VL> sq = added * added;
            sum_sq += reduce<float>(sq, std::plus<>());
        }
        if (k_aligned < K) {
            int k_tail = K - VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + k_tail);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k_tail);
            simd<float, VL> added = (h + r) * scalar;
            block_store<fp16, VL>(residual_ptr + k_tail, simd<fp16, VL>(added));
            int overlap = k_aligned - k_tail;
            simd<float, VL> sq = added * added;
            for (int z = 0; z < overlap; z++) sq[z] = 0.0f;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize and write output
        for (int k = 0; k < k_aligned; k += VL) {
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + k);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<fp16, VL>(hidden_ptr + k, simd<fp16, VL>(normed));
        }
        if (k_aligned < K) {
            int k_tail = K - VL;
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k_tail);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + k_tail);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<fp16, VL>(hidden_ptr + k_tail, simd<fp16, VL>(normed));
        }
    }
};

inline void fused_scaled_add_rms_norm_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* weight_ptr,
    int K, float eps, float scalar, sycl::queue& q)
{
    #define LAUNCH_V3(V)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(1, 1),                 FusedScaledAddRmsNorm_kernel<V>{hidden_ptr, residual_ptr, weight_ptr, K, eps, scalar});         });

    if      (K % 512 == 0) { LAUNCH_V3(512) }
    else if (K % 256 == 0) { LAUNCH_V3(256) }
    else if (K % 128 == 0) { LAUNCH_V3(128) }
    else                   { LAUNCH_V3(64)  }

    #undef LAUNCH_V3
}

// V2 host dispatcher: picks VL based on K alignment
inline void fused_add_rms_norm_v2_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* weight_ptr,
    int K, float eps, sycl::queue& q)
{
    // Use original kernel for K%512==0 (no overhead)
    if (K % 512 == 0) {
        fused_add_rms_norm_host(hidden_ptr, residual_ptr, weight_ptr, K, eps, q);
        return;
    }

    // Pick largest VL that gives at least 1 full chunk (VL <= K)
    // VL=256 for K=2816: 2816/256=11 full + 0 tail (2816%256=0!)
    #define LAUNCH_V2(V)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(1, 1),                 FusedAddRmsNorm_v2_kernel<V>{hidden_ptr, residual_ptr, weight_ptr, K, eps});         });

    if      (K % 256 == 0) { LAUNCH_V2(256) }
    else if (K % 128 == 0) { LAUNCH_V2(128) }
    else if (K % 64 == 0)  { LAUNCH_V2(64)  }
    else                   { LAUNCH_V2(64)  }  // tail handles remainder

    #undef LAUNCH_V2
}

// ============================================================================
// V4: Plain RMSNorm (no residual add) — for the standalone RMSNorm spots
// in gemma4 (post_attn_norm, post_ff_norm_1, pre_ff_norm_2, post_ff_norm_2).
// Same VL templating as v2; reads input, writes output, no in-place residual.
// ============================================================================
template<int VL>
struct RmsNorm_kernel {
    const fp16* input_ptr;
    fp16*       output_ptr;
    const fp16* weight_ptr;
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int k_aligned = (K / VL) * VL;

        // Pass 1: sum_sq
        float sum_sq = 0.0f;
        for (int k = 0; k < k_aligned; k += VL) {
            simd<float, VL> x = block_load<fp16, VL>(input_ptr + k);
            simd<float, VL> sq = x * x;
            sum_sq += reduce<float>(sq, std::plus<>());
        }
        if (k_aligned < K) {
            int k_tail = K - VL;
            simd<float, VL> x = block_load<fp16, VL>(input_ptr + k_tail);
            int overlap = k_aligned - k_tail;
            simd<float, VL> sq = x * x;
            for (int z = 0; z < overlap; z++) sq[z] = 0.0f;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize and write output
        for (int k = 0; k < k_aligned; k += VL) {
            simd<float, VL> x = block_load<fp16, VL>(input_ptr + k);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + k);
            simd<float, VL> normed = x * inv_rms * w;
            block_store<fp16, VL>(output_ptr + k, simd<fp16, VL>(normed));
        }
        if (k_aligned < K) {
            int k_tail = K - VL;
            simd<float, VL> x = block_load<fp16, VL>(input_ptr + k_tail);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + k_tail);
            simd<float, VL> normed = x * inv_rms * w;
            block_store<fp16, VL>(output_ptr + k_tail, simd<fp16, VL>(normed));
        }
    }
};

inline void rms_norm_host(
    const fp16* input_ptr, fp16* output_ptr, const fp16* weight_ptr,
    int K, float eps, sycl::queue& q)
{
    #define LAUNCH_V4(V)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(1, 1),                 RmsNorm_kernel<V>{input_ptr, output_ptr, weight_ptr, K, eps});         });

    if      (K % 512 == 0) { LAUNCH_V4(512) }
    else if (K % 256 == 0) { LAUNCH_V4(256) }
    else if (K % 128 == 0) { LAUNCH_V4(128) }
    else                   { LAUNCH_V4(64)  }

    #undef LAUNCH_V4
}
