/* scaled_resadd_norm_gemv_fp8.h — Fused (h+r)*scalar + RMSNorm + FP8 GEMV.
 *
 * Designed for gemma4 cross-layer xfuse + attention qkv_proj entry.
 *
 *   1. residual <- (hidden + residual) * scalar    (in-place)
 *   2. normed   = (residual / rms(residual)) * w_input_norm
 *   3. qkv_out  = normed @ qkv_proj_weight^T  (fp8, per-tensor scale)
 *
 * Two variants:
 *   - K_SPLIT==1: one thread per WG, computes full K serially (best for
 *                 small N where lots of WGs already saturate the GPU).
 *   - K_SPLIT>1:  K_SPLIT threads per WG, each owns K/K_SPLIT of K. SLM
 *                 reduces sum_sq → inv_rms (broadcast), then SLM reduces
 *                 GEMV partial dots. Best when N is large enough that the
 *                 WG×K_SPLIT total threads fit the EU count.
 *
 * Layout:
 *   hidden_ptr:    [1, K] fp16  (read)
 *   residual_ptr:  [1, K] fp16  (in-place: (h+r)*scalar)
 *   norm_w:        [K]    fp16
 *   qkv_weight:    [N, K] fp8   (per-tensor scale)
 *   qkv_scale:     [1]    fp32
 *   qkv_out:       [1, N] fp16
 */

#pragma once
#include "utils.h"

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_srng(
    simd<uint8_t, VL> raw, int fp8_mode) {
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    simd<uint16_t, VL> fp8_sign = (u16 >> 7) & 1;
    simd<uint16_t, VL> fp16_bits;

    if (fp8_mode == 0) {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 3) & 0xF;
        simd<uint16_t, VL> fp8_mant = u16 & 0x7;
        fp16_bits = (fp8_sign << 15) | ((fp8_exp + 8) << 10) | (fp8_mant << 7);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    } else {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 2) & 0x1F;
        simd<uint16_t, VL> fp8_mant = u16 & 0x3;
        fp16_bits = (fp8_sign << 15) | (fp8_exp << 10) | (fp8_mant << 8);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    }

    simd<fp16, VL> wh = fp16_bits.template bit_cast_view<fp16>().read();
    return simd<float, VL>(wh);
}

// K_SPLIT==1 variant (one thread per WG)
template<int VL, int MAX_CHUNKS>
struct ScaledResAddNormGEMV_fp8_pert_kernel {
    fp16*          hidden_ptr;
    fp16*          residual_ptr;
    const fp16*    norm_w_ptr;
    const uint8_t* qkv_weight;
    const float*   qkv_scale;
    fp16*          qkv_out;
    int N, K;
    float eps;
    float scalar;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        int n_chunks = K / VL;
        simd<float, VL> res_chunks[MAX_CHUNKS];

        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = (h + r) * scalar;
            res_chunks[c] = added;

            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));
            }

            simd<float, VL> sq = added * added;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        simd<float, VL> acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = res_chunks[c] * inv_rms * nw;

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                qkv_weight + (size_t)n * K + offset);
            simd<float, VL> wf = fp8_dequant_srng<VL>(w_raw, fp8_mode);

            acc += normed * wf;
        }

        float dot = reduce<float>(acc, std::plus<>()) * (*qkv_scale);
        qkv_out[n] = fp16(dot);
    }
};

// K_SPLIT>1 variant: K_SPLIT threads per WG, SLM reduces sum_sq + GEMV partials.
// SLM layout: 0..K_SPLIT-1 floats for sum_sq, K_SPLIT..2*K_SPLIT-1 for GEMV partials.
template<int VL, int K_SPLIT>
struct ScaledResAddNormGEMV_fp8_pert_ksplit_kernel {
    fp16*          hidden_ptr;
    fp16*          residual_ptr;
    const fp16*    norm_w_ptr;
    const uint8_t* qkv_weight;
    const float*   qkv_scale;
    fp16*          qkv_out;
    int N, K;
    float eps;
    float scalar;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<2 * K_SPLIT * sizeof(float)>();

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;       // chunk size in K dim per thread
        int ks = lid * kp;

        // ---- Phase 1: scaled add + sum_sq partial over [ks, ks+kp) ----
        float my_sumsq = 0.0f;
        for (int k = ks; k < ks + kp; k += VL) {
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + k);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k);
            simd<float, VL> added = (h + r) * scalar;
            // Only WG 0 writes residual; all WGs need to compute `added` but
            // each thread writes its own chunk. WG0 lid* gets distinct k-range.
            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + k, simd<fp16, VL>(added));
            }
            simd<float, VL> sq = added * added;
            my_sumsq += reduce<float>(sq, std::plus<>());
        }

        slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sumsq));
        barrier();
        simd<float, K_SPLIT> parts =
            slm_block_load<float, K_SPLIT>(0);
        float total_sumsq = reduce<float>(parts, std::plus<>());

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(total_sumsq / (float)K + eps))[0];

        // ---- Phase 2: GEMV partial dot over [ks, ks+kp) ----
        simd<float, VL> acc = 0.0f;
        for (int k = ks; k < ks + kp; k += VL) {
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + k);
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + k);
            simd<float, VL> normed = r * inv_rms * nw;

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                qkv_weight + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant_srng<VL>(w_raw, fp8_mode);

            acc += normed * wf;
        }
        float my_dot = reduce<float>(acc, std::plus<>());

        slm_block_store<float, 1>(
            (K_SPLIT + lid) * sizeof(float), simd<float, 1>(my_dot));
        barrier();
        if (lid == 0) {
            simd<float, K_SPLIT> dot_parts =
                slm_block_load<float, K_SPLIT>(K_SPLIT * sizeof(float));
            float total = reduce<float>(dot_parts, std::plus<>()) * (*qkv_scale);
            qkv_out[n] = fp16(total);
        }
    }
};

inline void scaled_resadd_norm_gemv_fp8_pert_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* qkv_weight, const float* qkv_scale, fp16* qkv_out,
    int N, int K, float eps, float scalar, int fp8_mode, sycl::queue& q)
{
    // Decide K_SPLIT: large N + large K benefits from per-WG threading.
    // We require K/ks to be divisible by 256 so VL=256 covers each thread's
    // chunk exactly; otherwise we fall back to ks=1 to avoid OOB reads.
    // For gemma4 K=2816, only ks=1 (full-K) works cleanly; K=4096 would
    // accept ks=4 (kp=1024) or ks=8 (kp=512).
    int ks = 1;
    if (N >= 2048 && (K % (256 * 8)) == 0) ks = 8;
    else if (N >= 1024 && (K % (256 * 4)) == 0) ks = 4;

    if (ks == 1) {
        #define LAUNCH_SRNGV1(V, MC)             q.submit([&](sycl::handler& cgh) {                 cgh.parallel_for(sycl::nd_range<1>(N, 1),                     ScaledResAddNormGEMV_fp8_pert_kernel<V, MC>{                         hidden_ptr, residual_ptr, norm_w_ptr,                         qkv_weight, qkv_scale, qkv_out,                         N, K, eps, scalar, fp8_mode});             });

        if (K % 512 == 0) {
            int mc = K / 512;
            if      (mc <= 4)  { LAUNCH_SRNGV1(512, 4)  }
            else if (mc <= 8)  { LAUNCH_SRNGV1(512, 8)  }
            else               { LAUNCH_SRNGV1(512, 16) }
        } else if (K % 256 == 0) {
            int mc = K / 256;
            if      (mc <= 8)  { LAUNCH_SRNGV1(256, 8)  }
            else if (mc <= 16) { LAUNCH_SRNGV1(256, 16) }
            else               { LAUNCH_SRNGV1(256, 32) }
        } else if (K % 128 == 0) {
            int mc = K / 128;
            if      (mc <= 16) { LAUNCH_SRNGV1(128, 16) }
            else               { LAUNCH_SRNGV1(128, 32) }
        } else {
            int mc = K / 64;
            if (mc > 64) mc = 64;
            if (mc <= 32) { LAUNCH_SRNGV1(64, 32) }
            else          { LAUNCH_SRNGV1(64, 64) }
        }

        #undef LAUNCH_SRNGV1
        return;
    }

    // K_SPLIT > 1: each WG runs ks threads. global = N * ks; local = ks.
    uint32_t global = (uint32_t)N * (uint32_t)ks;
    uint32_t local  = (uint32_t)ks;

    // Pick VL such that VL divides kp = K/ks (256 by default for gemma4
    // K=2816 / ks=4 = 704: 704 % 256 != 0 → fall back to 128 or 64).
    int kp = K / ks;
    int vl = 256;
    if (kp % 256 != 0) {
        if      (kp % 128 == 0) vl = 128;
        else if (kp % 64  == 0) vl = 64;
        else                    vl = 32;
    }

    #define LAUNCH_SRNGV_KS(V, KS)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(global, local),                 ScaledResAddNormGEMV_fp8_pert_ksplit_kernel<V, KS>{                     hidden_ptr, residual_ptr, norm_w_ptr,                     qkv_weight, qkv_scale, qkv_out,                     N, K, eps, scalar, fp8_mode});         });

    if (ks == 4) {
        if      (vl == 256) { LAUNCH_SRNGV_KS(256, 4) }
        else if (vl == 128) { LAUNCH_SRNGV_KS(128, 4) }
        else if (vl == 64)  { LAUNCH_SRNGV_KS(64,  4) }
        else                { LAUNCH_SRNGV_KS(32,  4) }
    } else { // ks == 8
        if      (vl == 256) { LAUNCH_SRNGV_KS(256, 8) }
        else if (vl == 128) { LAUNCH_SRNGV_KS(128, 8) }
        else if (vl == 64)  { LAUNCH_SRNGV_KS(64,  8) }
        else                { LAUNCH_SRNGV_KS(32,  8) }
    }

    #undef LAUNCH_SRNGV_KS
}
