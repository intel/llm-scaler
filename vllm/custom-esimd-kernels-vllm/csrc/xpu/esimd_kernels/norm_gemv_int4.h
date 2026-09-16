/* norm_gemv_int4.h — Fused RMSNormGated + INT4 GEMV for GDN out_proj.
 *
 * INT4 analogue of norm_gemv_fused.h (FP8 version).
 * Combines two operations into a single kernel submit:
 *   1. RMSNormGated: y = rmsnorm(x) * weight * gate(z), per-head (V dims each)
 *      The legacy specialization uses SiLU(z); the sigmoid specialization uses
 *      sigmoid(z) and is exposed through a separate ABI.
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
 *     (auto-disabled for large N where WG count provides sufficient occupancy)
 *   - Byte-level nibble extraction: bit_cast int32→uint8, extract low/high
 *     nibbles as 64-wide vectors, stride-2 dot product with normed
 *   - Hierarchical simd reduction for sum-of-squares and dot product
 *   - lsc_prefetch for next head's weight
 *
 * INT4 dequant: value = (nibble - 8) * scale
 * With V=128 and BLOCK_SIZE=128, exactly one scale per head iteration.
 */

#pragma once
#include "utils.h"
#include <cstdint>
#include <cstdlib>

namespace xesimd = sycl::ext::intel::experimental::esimd;

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
template<int K_SPLIT, bool SIGMOID_GATE, int OUTPUT_TILE = 1>
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

    static_assert(K_SPLIT == 1 || OUTPUT_TILE == 1,
                  "output tiling is only supported without K splitting");

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        const int n0 = item.get_group(0) * OUTPUT_TILE;
        int lid = item.get_local_id(0);
        if (n0 >= N) return;

        const int K = HV * V;
        const int packed_K = K / PACK;
        const int num_blocks_per_row = K / BLOCK_SIZE;

        // Partition heads among threads in the WG
        const int heads_per_thread = HV / K_SPLIT;
        const int h_start = lid * heads_per_thread;
        const int h_end   = h_start + heads_per_thread;

        // Pre-load norm weight [V=128] — all threads load (L3 cached)
        simd<float, 128> norm_w = block_load<fp16, 128>(norm_w_ptr);

        simd<float, 64> acc[OUTPUT_TILE];
#pragma unroll
        for (int output_index = 0; output_index < OUTPUT_TILE; ++output_index) {
            acc[output_index] = 0.0f;
        }

        // Prefetch first head's weight (16 int32 = 64B)
        if (h_start < h_end) {
#pragma unroll
            for (int output_index = 0; output_index < OUTPUT_TILE;
                 ++output_index) {
                const int n = n0 + output_index;
                if (n < N) {
                    xesimd::lsc_prefetch<
                        int32_t, 16, xesimd::lsc_data_size::default_size,
                        xesimd::cache_hint::uncached,
                        xesimd::cache_hint::cached>(
                        gemv_weight + (size_t)n * packed_K +
                        h_start * V / PACK);
                }
            }
        }

        for (int h = h_start; h < h_end; h++) {
            const int offset = h * V;

            // Prefetch next head's weight
            if (h + 1 < h_end) {
#pragma unroll
                for (int output_index = 0; output_index < OUTPUT_TILE;
                     ++output_index) {
                    const int n = n0 + output_index;
                    if (n < N) {
                        xesimd::lsc_prefetch<
                            int32_t, 16,
                            xesimd::lsc_data_size::default_size,
                            xesimd::cache_hint::uncached,
                            xesimd::cache_hint::cached>(
                            gemv_weight + (size_t)n * packed_K +
                            (h + 1) * V / PACK);
                    }
                }
            }

            // ── Load x, z for this head ──
            simd<float, 128> x_f = block_load<fp16, 128>(x_ptr + offset);
            simd<float, 128> z_f = block_load<fp16, 128>(z_ptr + offset);

            // ── RMSNorm: inv_rms = rsqrt(mean(x^2) + eps) ──
            float sum_sq = hreduce128(x_f * x_f);
            float inv_rms = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(sum_sq * (1.0f / V) + eps))[0];

            // ── Normalize + activation-specific gate ──
            simd<float, 128> normed = x_f * inv_rms * norm_w;
            simd<float, 128> exp_neg_z = sycl::ext::intel::esimd::exp(-z_f);
            if constexpr (SIGMOID_GATE) {
                normed *= 1.0f / (1.0f + exp_neg_z);
            } else {
                normed *= z_f / (1.0f + exp_neg_z);
            }

            // Share the norm/gate work across a small contiguous output tile.
            // Each output keeps its original FP32 accumulation order.
#pragma unroll
            for (int output_index = 0; output_index < OUTPUT_TILE;
                 ++output_index) {
                const int n = n0 + output_index;
                if (n >= N) continue;

                // Load 16 packed int32 = 128 INT4 values = one block
                simd<int32_t, 16> packed = block_load<int32_t, 16>(
                    gemv_weight + (size_t)n * packed_K + offset / PACK);

                // Reinterpret 64 bytes: each byte holds 2 nibbles
                // (lo=even, hi=odd).
                simd<uint32_t, 64> u32 = convert<uint32_t>(
                    packed.template bit_cast_view<uint8_t>().read());

                float s = (float)gemv_scale[
                    (size_t)n * num_blocks_per_row + h];
                float neg_8s = -8.0f * s;
                simd<float, 64> w_lo =
                    convert<float>(u32 & 0xFu) * s + neg_8s;
                simd<float, 64> w_hi =
                    convert<float>((u32 >> 4) & 0xFu) * s + neg_8s;

                acc[output_index] +=
                    normed.select<64, 2>(0) * w_lo +
                    normed.select<64, 2>(1) * w_hi;
            }
        }

        if constexpr (K_SPLIT == 1) {
#pragma unroll
            for (int output_index = 0; output_index < OUTPUT_TILE;
                 ++output_index) {
                const int n = n0 + output_index;
                if (n >= N) continue;
                // Hierarchical reduction: 64 → scalar (once, not per-head).
                simd<float, 64> reduced = acc[output_index];
                reduced.select<32,1>(0) += reduced.select<32,1>(32);
                reduced.select<16,1>(0) += reduced.select<16,1>(16);
                reduced.select<8,1>(0)  += reduced.select<8,1>(8);
                reduced.select<4,1>(0)  += reduced.select<4,1>(4);
                reduced.select<2,1>(0)  += reduced.select<2,1>(2);
                output[n] = fp16((float)reduced[0] + (float)reduced[1]);
            }
        } else {
            simd<float, 64> reduced = acc[0];
            reduced.select<32,1>(0) += reduced.select<32,1>(32);
            reduced.select<16,1>(0) += reduced.select<16,1>(16);
            reduced.select<8,1>(0)  += reduced.select<8,1>(8);
            reduced.select<4,1>(0)  += reduced.select<4,1>(4);
            reduced.select<2,1>(0)  += reduced.select<2,1>(2);
            const float my_sum = (float)reduced[0] + (float)reduced[1];
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n0] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

// M1 TP4: one FP32 norm/gate vector per work-group, shared by independent
// output rows. No global scratch or additional queue submission is needed.
template<int WORKERS, int OUTPUT_TILE = 1>
struct NormGEMV_int4_shared_kernel {
    const fp16* x_ptr;
    const fp16* z_ptr;
    const fp16* norm_w_ptr;
    const int32_t* gemv_weight;
    const fp16* gemv_scale;
    fp16* output;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int HV = 12, V = 128, K = HV * V, N = 2560;
        static_assert(N % (WORKERS * OUTPUT_TILE) == 0);
        slm_init<K * sizeof(float)>();
        const int lid = item.get_local_id(0);
        const int n0 = (item.get_group(0) * WORKERS + lid) * OUTPUT_TILE;

        for (int h = lid; h < HV; h += WORKERS) {
            simd<float, V> x = block_load<fp16, V>(x_ptr + h * V);
            simd<float, V> z = block_load<fp16, V>(z_ptr + h * V);
            simd<float, V> nw = block_load<fp16, V>(norm_w_ptr);
            const float ss = hreduce128(x * x);
            const float inv = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(ss * (1.0f / V) + eps))[0];
            simd<float, V> normalized = x * inv * nw;
            simd<float, V> exp_neg_z = sycl::ext::intel::esimd::exp(-z);
            normalized *= 1.0f / (1.0f + exp_neg_z);
            slm_block_store<float, V>(h * V * sizeof(float), normalized);
        }
        barrier();

        simd<float, 64> acc[OUTPUT_TILE];
#pragma unroll
        for (int j = 0; j < OUTPUT_TILE; ++j) acc[j] = 0.0f;
        for (int h = 0; h < HV; ++h) {
            simd<float, V> normalized =
                slm_block_load<float, V>(h * V * sizeof(float));
#pragma unroll
            for (int j = 0; j < OUTPUT_TILE; ++j) {
                const int n = n0 + j;
                simd<int32_t, 16> packed = block_load<int32_t, 16>(
                    gemv_weight + (size_t)n * (K / 8) + h * (V / 8));
                simd<uint32_t, 64> u = convert<uint32_t>(
                    packed.template bit_cast_view<uint8_t>().read());
                const float s = (float)gemv_scale[(size_t)n * HV + h];
                const float neg_8s = -8.0f * s;
                simd<float, 64> lo = convert<float>(u & 0xFu) * s + neg_8s;
                simd<float, 64> hi = convert<float>((u >> 4) & 0xFu) * s + neg_8s;
                acc[j] += normalized.select<64, 2>(0) * lo +
                          normalized.select<64, 2>(1) * hi;
            }
        }
#pragma unroll
        for (int j = 0; j < OUTPUT_TILE; ++j) {
            simd<float, 64> v = acc[j];
            v.select<32, 1>(0) += v.select<32, 1>(32);
            v.select<16, 1>(0) += v.select<16, 1>(16);
            v.select<8, 1>(0) += v.select<8, 1>(8);
            v.select<4, 1>(0) += v.select<4, 1>(4);
            v.select<2, 1>(0) += v.select<2, 1>(2);
            output[n0 + j] = fp16((float)v[0] + (float)v[1]);
        }
    }
};

/* ================================================================
 * Host dispatcher — auto-selects K_SPLIT based on HV and N
 * ================================================================ */
template <bool SIGMOID_GATE>
inline void norm_gemv_int4_host_impl(
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
    // Share x/z RMSNorm and sigmoid across 16 independent output rows in
    // workgroup-local FP32 storage. Keep the two-row baseline for diagnostic
    // A/B, with no extra submission or global scratch. TP8 and SiLU retain
    // their original path; the public INT4 norm-GEMV ABI only accepts M=1.
    if constexpr (SIGMOID_GATE) {
        if (N == 2560 && HV == 12 && V == 128) {
            // A fresh-process value of 0 restores the previous two-row path.
            static const int shared_workers = [] {
                const char* value = std::getenv("VLLM_XPU_GDN_OUT_SHARED_WORKERS");
                return value == nullptr ? 16 : std::atoi(value);
            }();
#define LAUNCH_SHARED_GDN(W) \
            q.submit([&](sycl::handler& cgh) { \
                cgh.parallel_for(sycl::nd_range<1>(N, W), \
                    NormGEMV_int4_shared_kernel<W>{x_ptr, z_ptr, norm_w_ptr, \
                        gemv_weight, gemv_scale, output, eps}); \
            });
            if (shared_workers == 16) { LAUNCH_SHARED_GDN(16); return; }
#undef LAUNCH_SHARED_GDN
            constexpr int output_tile = 2;
            const int tiled_global = (N + output_tile - 1) / output_tile;
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<1>(tiled_global, 1),
                    NormGEMV_int4_kernel<1, SIGMOID_GATE, output_tile>{
                        x_ptr, z_ptr, norm_w_ptr,
                        gemv_weight, gemv_scale, output,
                        N, HV, V, eps});
            });
            return;
        }
    }

    // K_SPLIT: split heads across threads for small N (EU occupancy).
    // Large N (>512) has enough WGs — use K_SPLIT=1 to avoid SLM overhead.
    int ks = 1;
    if (N <= 512) {
        // The kernel assigns exactly HV / K_SPLIT heads to each work-item,
        // so only use splits that cover HV without a remainder.
        if      (HV >= 8 && HV % 8 == 0) ks = 8;
        else if (HV >= 4 && HV % 4 == 0) ks = 4;
        else if (HV >= 2 && HV % 2 == 0) ks = 2;
    }

    int global = N * ks;
    int local  = ks;

    #define LAUNCH_NORM_GEMV_INT4(S) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                NormGEMV_int4_kernel<S, SIGMOID_GATE, 1>{ \
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
    norm_gemv_int4_host_impl<false>(
        x_ptr, z_ptr, norm_w_ptr, gemv_weight, gemv_scale, output,
        N, HV, V, eps, q);
}

inline void norm_gemv_int4_sigmoid_host(
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
    norm_gemv_int4_host_impl<true>(
        x_ptr, z_ptr, norm_w_ptr, gemv_weight, gemv_scale, output,
        N, HV, V, eps, q);
}
