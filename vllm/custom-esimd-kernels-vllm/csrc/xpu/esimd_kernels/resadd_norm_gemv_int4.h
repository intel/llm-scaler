/* resadd_norm_gemv_int4.h — Fused ResidualAdd + RMSNorm + INT4 GEMV.
 *
 * INT4 analogue of resadd_norm_gemv_fused.h (FP8 version).
 * Combines three operations into a single kernel:
 *   1. Residual add: residual = hidden_states + residual  (in-place)
 *   2. RMSNorm (Gemma-style): normed = residual / rms(residual) * weight
 *   3. GEMV: output = normed @ dequant(int4_weight^T) (per-block scale)
 *
 * Use case: post_attention_layernorm + MoE router GEMV (INT4 quantized)
 *   hidden_states: [1, K] fp16
 *   residual:      [1, K] fp16 (updated in-place)
 *   norm_weight:   [K] fp16
 *   gemv_weight:   [N, K/8] int32 packed
 *   gemv_scale:    [N, K/128] fp16 per-block
 *   output:        [1, N] fp16
 *   normed_out:    [1, K] fp16 (written for downstream MoE)
 *
 * Grid: N work-groups, 1 thread each.
 * Two-pass: pass1 = resadd + sum_sq, pass2 = normalize + dequant GEMV.
 * Re-loads residual in pass2 from L1/L2 cache to save register pressure.
 */

#pragma once
#include "utils.h"
#include <cstdint>

struct ResAddNormGEMV_int4_pert_kernel {
    fp16*          hidden_ptr;
    fp16*          residual_ptr;
    const fp16*    norm_w_ptr;
    const int32_t* gemv_weight;   // [N, K/8]
    const fp16*    gemv_scale;    // [N, K/128]
    fp16*          output;
    fp16*          normed_out;
    int N, K;
    float eps;

    static constexpr int BLOCK_SIZE = 128;
    static constexpr int PACK = 8;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        constexpr int VL = 128;
        const int n_chunks = K / VL;
        const int packed_K = K / PACK;
        const int num_blocks_per_row = K / BLOCK_SIZE;

        // Nibble shift amounts
        simd<uint32_t, 8> nib_shifts(0u, 4u);

        // ── Pass 1: residual_add + accumulate sum_sq ──
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int off = c * VL;
            simd<float, VL> hv = block_load<fp16, VL>(hidden_ptr + off);
            simd<float, VL> rv = block_load<fp16, VL>(residual_ptr + off);
            simd<float, VL> added = hv + rv;

            // Hierarchical reduction for sum of squares
            simd<float, VL> sq = added * added;
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0)  += sq.select<8,1>(8);
            sq.select<4,1>(0)  += sq.select<4,1>(4);
            sq.select<2,1>(0)  += sq.select<2,1>(2);
            sum_sq += sq[0] + sq[1];
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // ── Pass 2: normalize + INT4 GEMV ──
        // Re-load and recompute h+r (data hot in L1/L2 from pass 1)
        simd<float, VL> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int off = c * VL;

            simd<float, VL> hv = block_load<fp16, VL>(hidden_ptr + off);
            simd<float, VL> rv = block_load<fp16, VL>(residual_ptr + off);
            simd<float, VL> added = hv + rv;

            // Write residual and normed (only WG 0)
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + off);
            simd<float, VL> normed = added * inv_rms * nw;

            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + off, simd<fp16, VL>(added));
                block_store<fp16, VL>(normed_out + off, simd<fp16, VL>(normed));
            }

            // INT4 dequant + FMA (vectorized pack-level)
            simd<int32_t, 16> packed = block_load<int32_t, 16>(
                gemv_weight + (size_t)n * packed_K + off / PACK);
            simd<uint32_t, 16> u_packed =
                packed.template bit_cast_view<uint32_t>().read();

            float s = (float)gemv_scale[(size_t)n * num_blocks_per_row + c];

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                simd<uint32_t, 8> nib = (simd<uint32_t, 8>(u_packed[i]) >> nib_shifts) & 0xFu;
                simd<float, 8> w = simd<float, 8>(
                    nib.template bit_cast_view<int32_t>().read()) - 8.0f;
                w *= s;
                acc.select<8, 1>(i * 8) += normed.select<8, 1>(i * 8) * w;
            }
        }

        // Final reduction
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0)  += acc.select<8,1>(8);
        acc.select<4,1>(0)  += acc.select<4,1>(4);
        acc.select<2,1>(0)  += acc.select<2,1>(2);
        output[n] = fp16(acc[0] + acc[1]);
    }
};

inline void resadd_norm_gemv_int4_pert_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const int32_t* gemv_weight, const fp16* gemv_scale,
    fp16* output, fp16* normed_out,
    int N, int K, float eps, sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(N, 1),
            ResAddNormGEMV_int4_pert_kernel{
                hidden_ptr, residual_ptr, norm_w_ptr,
                gemv_weight, gemv_scale, output, normed_out,
                N, K, eps});
    });
}
