/* resadd_norm_gemv_fused.h — Fused ResidualAdd + RMSNorm + FP8 GEMV.
 *
 * Combines three operations into a single kernel:
 *   1. Residual add: residual = hidden_states + residual  (in-place)
 *   2. RMSNorm (Gemma-style): normed = residual / rms(residual) * weight
 *      where weight is pre-adjusted (w+1.0 already applied by caller)
 *   3. GEMV: output = normed @ dequant(gemv_weight^T) * scale
 *
 * Designed for Qwen3-Next post_attention_layernorm + MoE router:
 *   hidden_states: [1, K] fp16   (K=2048)
 *   residual:      [1, K] fp16   (updated in-place)
 *   norm_weight:   [K] fp16      (Gemma _gemma_w = original_w + 1.0)
 *   gemv_weight:   [N, K] FP8    (N=512 for router)
 *   gemv_scale:    [1] float32
 *   output:        [1, N] fp16
 *
 * Grid: N work-groups, 1 thread each.
 * Each WG redundantly computes residual_add + norm (data in L3 cache).
 * Only WG 0 writes the updated residual back to global memory.
 *
 * For K=2048 with VL=512: 4 loop iterations for norm, then 4 for GEMV.
 * Interleaved approach: compute norm chunk + GEMV chunk per iteration.
 */

#pragma once
#include "utils.h"

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_rng(
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

/* ================================================================
 * Kernel: Fused ResidualAdd + RMSNorm + FP8 GEMV (per-tensor scale)
 *
 * Two-pass approach:
 *   Pass 1: Load hidden+residual, compute residual_add, accumulate
 *           sum-of-squares for RMS, store normed chunks to registers.
 *   Pass 2 (fused with pass 1 second half): GEMV dot product.
 *
 * Since we need the full RMS before normalizing, we do:
 *   Loop 1 (K/VL iters): load h+r, add, compute partial sum_sq
 *   Reduce sum_sq → inv_rms
 *   Loop 2 (K/VL iters): normalize stored residual, load weight, FMA
 * ================================================================ */
struct ResAddNormGEMV_fp8_pert_kernel {
    fp16*          hidden_ptr;   // [1, K] — input (read-only for this kernel)
    fp16*          residual_ptr; // [1, K] — updated in-place
    const fp16*    norm_w_ptr;   // [K] — Gemma norm weight (w+1.0)
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1]
    fp16*          output;       // [1, N] — router logits
    fp16*          normed_out;   // [1, K] — normed hidden_states (for MoE experts)
    int N, K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        // Pass 1: residual_add + accumulate sum_sq
        // Store the added residual in register arrays for reuse in pass 2.
        // For K=2048, VL=512: 4 chunks, 4 * 512 * 4 bytes = 8KB registers.
        // Fits in standard GRF (256 regs × 64 bytes = 16KB).

        float sum_sq = 0.0f;

        // We store residual chunks in a small register-file buffer.
        // Since VL=512 and K/VL is small (4), unroll explicitly.
        constexpr int VL = 512;
        constexpr int MAX_CHUNKS = 8;  // supports up to K=4096
        simd<float, VL> res_chunks[MAX_CHUNKS];
        int n_chunks = K / VL;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);

            // Residual add
            simd<float, VL> added = h + r;
            res_chunks[c] = added;

            // Write back residual (only WG 0 to avoid redundant writes)
            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));
            }

            // Accumulate sum of squares for RMS
            simd<float, VL> sq = added * added;
            // Hierarchical reduction within VL=512
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

        // Compute inverse RMS
        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize + GEMV
        simd<float, VL> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            // Normalize: normed = residual * inv_rms * weight
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = res_chunks[c] * inv_rms * nw;

            // Write normed hidden_states (only WG 0)
            if (n == 0) {
                block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
            }

            // GEMV: accumulate dot product
            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_rng<VL>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        // Final reduction
        acc.select<256,1>(0) += acc.select<256,1>(256);
        acc.select<128,1>(0) += acc.select<128,1>(128);
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0) += acc.select<8,1>(8);
        acc.select<4,1>(0) += acc.select<4,1>(4);
        acc.select<2,1>(0) += acc.select<2,1>(2);
        float dot = ((float)acc[0] + (float)acc[1]) * *gemv_scale;
        output[n] = fp16(dot);
    }
};

/* Host dispatcher */
inline void resadd_norm_gemv_fp8_pert_host(
    fp16* hidden_ptr,
    fp16* residual_ptr,
    const fp16* norm_w_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    fp16* normed_out,
    int N, int K,
    float eps,
    int fp8_mode,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(N, 1),
            ResAddNormGEMV_fp8_pert_kernel{
                hidden_ptr, residual_ptr, norm_w_ptr,
                gemv_weight, gemv_scale, output, normed_out,
                N, K, eps, fp8_mode});
    });
}
