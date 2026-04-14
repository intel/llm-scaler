/* norm_gemv_int4.h — Fused RMSNormGated + INT4 GEMV for GDN out_proj.
 *
 * INT4 analogue of norm_gemv_fused.h (FP8 version).
 * Combines two operations into a single kernel submit:
 *   1. RMSNormGated: y = rmsnorm(x) * weight * silu(z), per-head (V dims each)
 *   2. GEMV: output = y_flat @ dequant(int4_weight^T) (per-block scale)
 *
 * Designed for GDN decode path where:
 *   x (core_attn_out): [HV, V] fp16   (e.g. [8, 128])
 *   z (z_out):          [HV, V] fp16
 *   norm_weight:         [V] fp16       (shared across heads)
 *   gemv_weight:         [N, K/8] int32 (K = HV*V, packed 4-bit)
 *   gemv_scale:          [N, K/128] fp16 (per-block scale)
 *   output:              [N] fp16
 *
 * Optimizations (referenced from IPEX patterns):
 *   - K_SPLIT: multiple threads per WG split HV heads, SLM cooperative reduce
 *   - Vectorized INT4 dequant: pack-level broadcast+shift, no scatter
 *   - Hierarchical simd reduction for sum-of-squares and dot product
 *   - Fused dequant+FMA: avoid materializing full weight vector
 *
 * INT4 dequant: value = (nibble - 8) * scale
 * With V=128 and BLOCK_SIZE=128, exactly one scale per head iteration.
 */

#pragma once
#include "utils.h"
#include <cstdint>

/* Hierarchical reduction for simd<float, 128> → scalar.
 * 7 additions instead of 127 for sequential accumulation. */
ESIMD_INLINE float hreduce128(simd<float, 128> v) {
    v.select<64,1>(0) += v.select<64,1>(64);
    v.select<32,1>(0) += v.select<32,1>(32);
    v.select<16,1>(0) += v.select<16,1>(16);
    v.select<8,1>(0)  += v.select<8,1>(8);
    v.select<4,1>(0)  += v.select<4,1>(4);
    v.select<2,1>(0)  += v.select<2,1>(2);
    return v[0] + v[1];
}

/* ================================================================
 * Kernel: Fused RMSNormGated + INT4 GEMV
 *
 * K_SPLIT: number of threads per WG. Each thread handles HV/K_SPLIT
 * heads, then partial sums are reduced via SLM.
 *
 * Grid: N work-groups × K_SPLIT threads per WG
 * ================================================================ */
template<int K_SPLIT>
struct NormGEMV_int4_kernel {
    const fp16*    x_ptr;        // [HV, V] core_attn_out
    const fp16*    z_ptr;        // [HV, V] z_out
    const fp16*    norm_w_ptr;   // [V] norm weight
    const int32_t* gemv_weight;  // [N, K/8] packed int4, K = HV * V
    const fp16*    gemv_scale;   // [N, K/BLOCK_SIZE] per-block scale
    fp16*          output;       // [N]
    int N;
    int HV;      // number of value heads (per TP)
    int V;       // head_v_dim (128)
    float eps;

    static constexpr int BLOCK_SIZE = 128;
    static constexpr int PACK = 8;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        const int K = HV * V;
        const int packed_K = K / PACK;
        const int num_blocks_per_row = K / BLOCK_SIZE;

        // Partition heads among threads in the WG
        const int heads_per_thread = HV / K_SPLIT;
        const int h_start = lid * heads_per_thread;
        const int h_end   = h_start + heads_per_thread;

        // Pre-load norm weight [V=128] — all threads load (L3 cached)
        simd<float, 128> norm_w = block_load<fp16, 128>(norm_w_ptr);

        // Nibble shift amounts: extract 8 nibbles from one int32
        // nibble[b] = (packed >> (b*4)) & 0xF for b in 0..7
        simd<uint32_t, 8> nib_shifts(0u, 4u);  // {0, 4, 8, 12, 16, 20, 24, 28}

        simd<float, 128> acc = 0.0f;

        for (int h = h_start; h < h_end; h++) {
            const int offset = h * V;

            // ── Load x, z for this head ──
            simd<float, 128> x_f = block_load<fp16, 128>(x_ptr + offset);
            simd<float, 128> z_f = block_load<fp16, 128>(z_ptr + offset);

            // ── RMSNorm: inv_rms = rsqrt(mean(x^2) + eps) ──
            float sum_sq = hreduce128(x_f * x_f);
            float inv_rms = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(sum_sq * (1.0f / V) + eps))[0];

            // ── Normalize + SiLU gate ──
            simd<float, 128> normed = x_f * inv_rms * norm_w;
            simd<float, 128> exp_neg_z = sycl::ext::intel::esimd::exp(-z_f);
            normed *= z_f / (1.0f + exp_neg_z);

            // ── INT4 dequant + FMA (vectorized pack-level) ──
            // Load 16 packed int32 = 128 INT4 values = one block
            simd<int32_t, 16> packed = block_load<int32_t, 16>(
                gemv_weight + (size_t)n * packed_K + offset / PACK);
            simd<uint32_t, 16> u_packed =
                packed.template bit_cast_view<uint32_t>().read();

            float s = (float)gemv_scale[(size_t)n * num_blocks_per_row + h];
            float neg_8s = -8.0f * s;

            // Process per pack: broadcast int32 → 8-wide shift → extract nibbles
            // FMA dequant: nib * s + (-8*s) = (nib - 8) * s
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                simd<uint32_t, 8> nib = (simd<uint32_t, 8>(u_packed[i]) >> nib_shifts) & 0xFu;
                simd<float, 8> w = convert<float>(nib) * s + neg_8s;
                acc.select<8, 1>(i * 8) += normed.select<8, 1>(i * 8) * w;
            }
        }

        // ── Reduction ──
        float my_sum = hreduce128(acc);

        if constexpr (K_SPLIT == 1) {
            output[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

/* ================================================================
 * Host dispatcher — auto-selects K_SPLIT based on HV
 * ================================================================ */
inline void norm_gemv_int4_host(
    const fp16* x_ptr,
    const fp16* z_ptr,
    const fp16* norm_w_ptr,
    const int32_t* gemv_weight,
    const fp16* gemv_scale,
    fp16* output,
    int N, int HV, int V,
    float eps,
    sycl::queue& q)
{
    // K_SPLIT = min(HV, 8), must divide HV evenly
    int ks = 1;
    if      (HV >= 8) ks = 8;
    else if (HV >= 4) ks = 4;
    else if (HV >= 2) ks = 2;

    int global = N * ks;
    int local  = ks;

    #define LAUNCH_NORM_GEMV_INT4(S) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                NormGEMV_int4_kernel<S>{ \
                    x_ptr, z_ptr, norm_w_ptr, \
                    gemv_weight, gemv_scale, output, \
                    N, HV, V, eps}); \
        });

    switch (ks) {
        case 8: LAUNCH_NORM_GEMV_INT4(8); break;
        case 4: LAUNCH_NORM_GEMV_INT4(4); break;
        case 2: LAUNCH_NORM_GEMV_INT4(2); break;
        default: LAUNCH_NORM_GEMV_INT4(1); break;
    }

    #undef LAUNCH_NORM_GEMV_INT4
}
