/* resadd_norm_gemv_int4.h — Fused ResidualAdd + RMSNorm + INT4 GEMV.
 *
 * INT4 analogue of resadd_norm_gemv_fused.h (FP8 version).
 * Combines three operations into a single kernel:
 *   1. Residual add: residual = hidden_states + residual  (in-place)
 *   2. RMSNorm (Gemma-style): normed = residual / rms(residual) * weight
 *   3. GEMV: output = normed @ dequant(int4_weight^T) (per-block scale)
 *
 * Optimizations (referenced from IPEX and FP8 kernel patterns):
 *   - VL=512 pass 1 with register array caching (eliminates pass 2 re-read)
 *   - Vectorized INT4 dequant: pack-level broadcast+shift, no scatter
 *   - Hierarchical simd reduction for sum-of-squares and dot product
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
 * Two-pass architecture (mirrors FP8 kernel):
 *   Pass 1 (VL=512): resadd + sum_sq → register array + residual write-back
 *   Pass 2 (VL=512): normalize from registers + 4×128 INT4 dequant GEMV
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

    // ── Small-K fallback (K < 512): VL=128, two-pass with memory re-read ──
    void run_small_k(int n) const SYCL_ESIMD_FUNCTION {
        constexpr int VL = 128;
        const int n_chunks = K / VL;
        const int packed_K = K / PACK;
        const int num_blocks_per_row = K / BLOCK_SIZE;
        simd<uint32_t, 8> nib_shifts(0u, 4u);

        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int off = c * VL;
            simd<float, VL> hv = block_load<fp16, VL>(hidden_ptr + off);
            simd<float, VL> rv = block_load<fp16, VL>(residual_ptr + off);
            simd<float, VL> added = hv + rv;
            simd<float, VL> sq = added * added;
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0)  += sq.select<8,1>(8);
            sq.select<4,1>(0)  += sq.select<4,1>(4);
            sq.select<2,1>(0)  += sq.select<2,1>(2);
            sum_sq += (float)sq[0] + (float)sq[1];
        }
        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        simd<float, VL> acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int off = c * VL;
            simd<float, VL> hv = block_load<fp16, VL>(hidden_ptr + off);
            simd<float, VL> rv = block_load<fp16, VL>(residual_ptr + off);
            simd<float, VL> added = hv + rv;
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + off);
            simd<float, VL> normed = added * inv_rms * nw;
            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + off, simd<fp16, VL>(added));
                block_store<fp16, VL>(normed_out + off, simd<fp16, VL>(normed));
            }
            simd<int32_t, 16> packed = block_load<int32_t, 16>(
                gemv_weight + (size_t)n * packed_K + off / PACK);
            simd<uint32_t, 16> u_packed =
                packed.template bit_cast_view<uint32_t>().read();
            float s = (float)gemv_scale[(size_t)n * num_blocks_per_row + c];
            float neg_8s = -8.0f * s;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                simd<uint32_t, 8> nib = (simd<uint32_t, 8>(u_packed[i]) >> nib_shifts) & 0xFu;
                simd<float, 8> w = convert<float>(nib) * s + neg_8s;
                acc.select<8, 1>(i * 8) += normed.select<8, 1>(i * 8) * w;
            }
        }
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0)  += acc.select<8,1>(8);
        acc.select<4,1>(0)  += acc.select<4,1>(4);
        acc.select<2,1>(0)  += acc.select<2,1>(2);
        output[n] = fp16((float)acc[0] + (float)acc[1]);
    }

    // ── Optimized path (K >= 512): VL=512, register-cached two-pass ──
    void run_large_k(int n) const SYCL_ESIMD_FUNCTION {
        constexpr int VL = 512;
        constexpr int MAX_CHUNKS = 8;  // supports up to K=4096
        constexpr int BLOCKS_PER_VL = VL / BLOCK_SIZE;  // 4
        const int packed_K = K / PACK;
        const int num_blocks_per_row = K / BLOCK_SIZE;
        const int n_chunks = K / VL;

        simd<uint32_t, 8> nib_shifts(0u, 4u);

        // Pass 1: resadd + sum_sq, store to register array
        simd<float, VL> res_chunks[MAX_CHUNKS];
        float sum_sq = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);

            simd<float, VL> added = h + r;
            res_chunks[c] = added;

            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));
            }

            simd<float, VL> sq = added * added;
            sq.select<256,1>(0) += sq.select<256,1>(256);
            sq.select<128,1>(0) += sq.select<128,1>(128);
            sq.select<64,1>(0)  += sq.select<64,1>(64);
            sq.select<32,1>(0)  += sq.select<32,1>(32);
            sq.select<16,1>(0)  += sq.select<16,1>(16);
            sq.select<8,1>(0)   += sq.select<8,1>(8);
            sq.select<4,1>(0)   += sq.select<4,1>(4);
            sq.select<2,1>(0)   += sq.select<2,1>(2);
            sum_sq += (float)sq[0] + (float)sq[1];
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize from register array + INT4 GEMV
        simd<float, 128> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = res_chunks[c] * inv_rms * nw;

            if (n == 0) {
                block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
            }

            // Coalesced weight load: 4 blocks × 16 int32 = 64 int32 at once
            constexpr int PACKED_PER_VL = VL / PACK;  // 64
            simd<int32_t, PACKED_PER_VL> all_packed = block_load<int32_t, PACKED_PER_VL>(
                gemv_weight + (size_t)n * packed_K + c * PACKED_PER_VL);
            simd<uint32_t, PACKED_PER_VL> all_u =
                all_packed.template bit_cast_view<uint32_t>().read();

            #pragma unroll
            for (int blk = 0; blk < BLOCKS_PER_VL; blk++) {
                int blk_off = blk * BLOCK_SIZE;

                float s = (float)gemv_scale[(size_t)n * num_blocks_per_row + c * BLOCKS_PER_VL + blk];
                float neg_8s = -8.0f * s;

                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    simd<uint32_t, 8> nib = (simd<uint32_t, 8>(all_u[blk * 16 + i]) >> nib_shifts) & 0xFu;
                    // FMA dequant: nib * s + (-8*s) = (nib - 8) * s
                    simd<float, 8> w = convert<float>(nib) * s + neg_8s;
                    acc.select<8, 1>(i * 8) +=
                        normed.select<8, 1>(blk_off + i * 8) * w;
                }
            }
        }

        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0)  += acc.select<8,1>(8);
        acc.select<4,1>(0)  += acc.select<4,1>(4);
        acc.select<2,1>(0)  += acc.select<2,1>(2);
        output[n] = fp16((float)acc[0] + (float)acc[1]);
    }

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        if (K >= 512) {
            run_large_k(n);
        } else {
            run_small_k(n);
        }
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
