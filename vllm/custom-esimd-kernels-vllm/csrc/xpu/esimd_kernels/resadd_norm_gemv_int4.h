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

/* ================================================================
 * K_SPLIT variant: multiple threads per WG split K-dimension.
 * Two-pass memory re-read (mirrors resadd_norm_gemv2_fused.h):
 *   Pass 1: each thread computes partial sum_sq → SLM reduce → inv_rms
 *   Pass 2: each thread re-reads h+r (L3 hit), normalizes, INT4 GEMV
 *           → SLM reduce → output dot product
 *
 * Eliminates register arrays; L3 cache absorbs the redundant reads.
 * Grid: N work-groups × K_SPLIT threads per WG.
 * Each thread handles K/K_SPLIT elements (multiple of BLOCK_SIZE=128).
 * ================================================================ */
template<int K_SPLIT>
struct ResAddNormGEMV_int4_ksplit_kernel {
    fp16*          hidden_ptr;
    fp16*          residual_ptr;
    const fp16*    norm_w_ptr;
    const int32_t* gemv_weight;
    const fp16*    gemv_scale;
    fp16*          output;
    fp16*          normed_out;
    int N, K;
    float eps;

    static constexpr int BLOCK_SIZE = 128;
    static constexpr int PACK = 8;
    static constexpr int VL = 128;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        static_assert(K_SPLIT > 1, "Use ResAddNormGEMV_int4_pert_kernel for K_SPLIT=1");
        slm_init<K_SPLIT * sizeof(float)>();

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        const int packed_K = K / PACK;
        const int num_blocks = K / BLOCK_SIZE;
        const int k_per_thread = K / K_SPLIT;
        const int k_start = lid * k_per_thread;
        const int n_chunks = k_per_thread / VL;

        simd<uint32_t, 8> nib_shifts(0u, 4u);

        // ── Pass 1: partial sum_sq ──
        float partial_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int off = k_start + c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + off);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + off);
            simd<float, VL> added = h + r;

            simd<float, VL> sq = added * added;
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0)  += sq.select<8,1>(8);
            sq.select<4,1>(0)  += sq.select<4,1>(4);
            sq.select<2,1>(0)  += sq.select<2,1>(2);
            partial_sq += (float)sq[0] + (float)sq[1];
        }

        // SLM reduce for sum_sq — all threads compute inv_rms redundantly
        slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(partial_sq));
        barrier();
        simd<float, K_SPLIT> sq_parts = slm_block_load<float, K_SPLIT>(0);
        float total_sq = reduce<float>(sq_parts, std::plus<>());
        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(total_sq / (float)K + eps))[0];

        // ── Pass 2: re-read h+r (L3 hit), normalize, write-back, INT4 GEMV ──
        simd<float, VL> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int off = k_start + c * VL;
            int blk_idx = off / BLOCK_SIZE;

            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + off);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + off);
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + off);
            simd<float, VL> added = h + r;
            simd<float, VL> normed = added * inv_rms * nw;

            // WG 0: each thread writes its own K-range
            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + off, simd<fp16, VL>(added));
                block_store<fp16, VL>(normed_out + off, simd<fp16, VL>(normed));
            }

            // INT4 dequant + FMA (one block per VL=128 chunk)
            simd<int32_t, 16> packed = block_load<int32_t, 16>(
                gemv_weight + (size_t)n * packed_K + off / PACK);
            simd<uint32_t, 16> u_packed =
                packed.template bit_cast_view<uint32_t>().read();

            float s = (float)gemv_scale[(size_t)n * num_blocks + blk_idx];
            float neg_8s = -8.0f * s;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                simd<uint32_t, 8> nib = (simd<uint32_t, 8>(u_packed[i]) >> nib_shifts) & 0xFu;
                simd<float, 8> w = convert<float>(nib) * s + neg_8s;
                acc.select<8, 1>(i * 8) += normed.select<8, 1>(i * 8) * w;
            }
        }

        // Reduce acc → scalar partial dot
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0)  += acc.select<8,1>(8);
        acc.select<4,1>(0)  += acc.select<4,1>(4);
        acc.select<2,1>(0)  += acc.select<2,1>(2);
        float my_dot = (float)acc[0] + (float)acc[1];

        // SLM reduce for dot product → thread 0 writes output
        slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_dot));
        barrier();
        if (lid == 0) {
            simd<float, K_SPLIT> dot_parts = slm_block_load<float, K_SPLIT>(0);
            output[n] = fp16(reduce<float>(dot_parts, std::plus<>()));
        }
    }
};

/* Host dispatcher — auto-selects K_SPLIT based on N and K */
inline void resadd_norm_gemv_int4_pert_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const int32_t* gemv_weight, const fp16* gemv_scale,
    fp16* output, fp16* normed_out,
    int N, int K, float eps, sycl::queue& q)
{
    // K_SPLIT: more threads per WG when N is small and K is large.
    // Mirrors fp8_GEMV_v2.h select_vl_ks() thresholds.
    int ks = 1;
    if      (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    else if (N <= 512 && K >= 512)  ks = 2;

    // Safety: k_per_thread must be a multiple of BLOCK_SIZE=128
    while (ks > 1 && (K % (ks * 128) != 0)) ks /= 2;

    if (ks <= 1) {
        // Original optimized kernel (VL=512 register-cached path)
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(N, 1),
                ResAddNormGEMV_int4_pert_kernel{
                    hidden_ptr, residual_ptr, norm_w_ptr,
                    gemv_weight, gemv_scale, output, normed_out,
                    N, K, eps});
        });
        return;
    }

    int global = N * ks;
    int local  = ks;

    #define LAUNCH_RESADD_INT4_KS(S) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                ResAddNormGEMV_int4_ksplit_kernel<S>{ \
                    hidden_ptr, residual_ptr, norm_w_ptr, \
                    gemv_weight, gemv_scale, output, normed_out, \
                    N, K, eps}); \
        });

    switch (ks) {
        case 8: LAUNCH_RESADD_INT4_KS(8); break;
        case 4: LAUNCH_RESADD_INT4_KS(4); break;
        case 2: LAUNCH_RESADD_INT4_KS(2); break;
        default: break;
    }

    #undef LAUNCH_RESADD_INT4_KS
}
