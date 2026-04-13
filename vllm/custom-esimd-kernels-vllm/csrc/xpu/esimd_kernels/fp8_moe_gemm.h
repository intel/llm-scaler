#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl;
using fp16 = sycl::half;

namespace xesimd = sycl::ext::intel::experimental::esimd;

// ============================================================================
// MoE Grouped GEMM — FP8 E5M2 with per-N scale
//
// V3: Hybrid grid — each WG handles (n_per_wg N-tiles, 1 M-chunk).
//     M-chunks parallelized across WGs (no M-chunk loop → no weight re-reads).
//     N-tiles merged per WG to control total WG count.
//
// Weight: [num_experts, N, K] uint8 FP8 E5M2 (row-major per expert)
// Scale:  [num_experts, N] float32 (per-N per-expert)
// Input:  [total_tokens, K] fp16
// Output: [total_tokens, N] fp16
// Index:  [num_experts + 1] uint32 (token start per expert)
// ============================================================================

SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16>
fp8_e5m2_pair_to_vnni(simd<uint32_t, 16> b_lo, simd<uint32_t, 16> b_hi_shifted) {
    return (b_lo | b_hi_shifted) << 8;
}

template<int MT_MAX>
struct MoE_FP8_E5M2_Kernel_V3 {
    const fp16*     input;
    const uint8_t*  weight;      // [num_experts * N, K] flat
    const float*    scale;       // [num_experts * N] flat
    fp16*           output;      // [total_tokens, N]
    const uint32_t* expert_idx;  // [num_experts + 1]
    int N, K, num_experts;
    int n_wg_count;    // number of N-tile-groups per expert
    int n_per_wg;      // N-tiles per WG
    int m_chunks;      // M-chunks per expert (1 per WG in M dimension)

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_SUB = 16;
        constexpr int M_PER_CHUNK = MT_MAX * 8;

        int wgs_per_expert = n_wg_count * m_chunks;
        int wg_id = item.get_group(0);
        int expert_id = wg_id / wgs_per_expert;
        int local_wg  = wg_id % wgs_per_expert;

        if (expert_id >= num_experts) return;

        // Decode (n_wg_id, m_chunk_id) from flat local_wg
        int n_wg_id    = local_wg % n_wg_count;
        int m_chunk_id = local_wg / n_wg_count;

        uint32_t tok_start = expert_idx[expert_id];
        uint32_t tok_end   = expert_idx[expert_id + 1];
        int M_e = (int)(tok_end - tok_start);
        if (M_e <= 0) return;

        int m_row_start = m_chunk_id * M_PER_CHUNK;
        if (m_row_start >= M_e) return;

        int m_rows_left = M_e - m_row_start;
        int total_m_tiles_here = (m_rows_left + 7) / 8;
        if (total_m_tiles_here > MT_MAX) total_m_tiles_here = MT_MAX;

        // N-tile range for this WG
        int n_tiles_total = (N + 15) / 16;
        int n_tile_start = n_wg_id * n_per_wg;
        int n_tile_end = n_tile_start + n_per_wg;
        if (n_tile_end > n_tiles_total) n_tile_end = n_tiles_total;

        // 2D surface descriptors
        uint32_t total_tokens = expert_idx[num_experts];
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = total_tokens - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)(num_experts * N) - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B, 0u, 0u);

        // N-tile loop (merged tiles per WG)
        for (int nc = n_tile_start; nc < n_tile_end; nc++) {
            int n_start = nc * 16;
            if (n_start >= N) break;

            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            // Load per-N scale
            simd<float, 16> scale_n;
            if (full_n)
                scale_n = block_load<float, 16>(scale + expert_id * N + n_start);
            else {
                scale_n = 0.0f;
                for (int ni = 0; ni < n_valid; ni++)
                    scale_n[ni] = *(scale + expert_id * N + n_start + ni);
            }

            payB_t.set_y((uint32_t)(expert_id * N + n_start));

            // Accumulators
            simd<float, 128> acc[MT_MAX];
            #pragma unroll
            for (int i = 0; i < MT_MAX; i++) acc[i] = 0.0f;

            // K loop — weight read once per N-tile
            for (int k_base = 0; k_base < K; k_base += 64) {
                #pragma unroll
                for (int sub = 0; sub < 4; sub++) {
                    int k_sub = k_base + sub * K_SUB;
                    if (k_sub >= K) break;

                    payB_t.set_x((uint32_t)(k_sub / 4));
                    simd<uint32_t, 64> w_t = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                        true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);

                    // E5M2 dequant + VNNI
                    simd<uint32_t, 128> b_vnni;
                    #pragma unroll
                    for (int col = 0; col < 4; col++) {
                        simd<uint32_t, 16> g = w_t.template select<16, 1>(col * 16);
                        simd<uint32_t, 16> b0 = g & 0xFF;
                        simd<uint32_t, 16> b1 = (g & 0xFF00) << 8;
                        b_vnni.template select<16, 1>(col * 32) =
                            fp8_e5m2_pair_to_vnni(b0, b1);
                        simd<uint32_t, 16> gh = g >> 16;
                        simd<uint32_t, 16> b2 = gh & 0xFF;
                        simd<uint32_t, 16> b3 = (gh & 0xFF00) << 8;
                        b_vnni.template select<16, 1>(col * 32 + 16) =
                            fp8_e5m2_pair_to_vnni(b2, b3);
                    }
                    simd<fp16, 256> b_tile =
                        b_vnni.template bit_cast_view<fp16>().read();

                    // Input + DPAS — fully unrolled
                    payA.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < MT_MAX; m++) {
                        payA.set_y((uint32_t)(tok_start + m_row_start + m * 8));
                        simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                        acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile, a);
                    }
                }
            }

            // Output
            for (int m = 0; m < MT_MAX; m++) {
                if (m >= total_m_tiles_here) break;
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m_row_start + m * 8 + mi;
                    if (row < M_e) {
                        simd<float, 16> rf = acc[m].template select<16, 1>(mi * 16);
                        rf *= scale_n;
                        simd<fp16, 16> out = convert<fp16>(rf);
                        size_t off = (size_t)(tok_start + row) * N + n_start;
                        if (full_n)
                            block_store<fp16, 16>(output + off, out);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[off + ni] = out[ni];
                    }
                }
            }
        } // end N-tile loop
    }
};

// ============================================================================
// Per-tensor scale variant: scale is [num_experts] float32 (one per expert)
// ============================================================================

template<int MT_MAX>
struct MoE_FP8_E5M2_Kernel_V3_PerT {
    const fp16*     input;
    const uint8_t*  weight;      // [num_experts * N, K] flat
    const float*    scale;       // [num_experts] flat — one scalar per expert
    fp16*           output;      // [total_tokens, N]
    const uint32_t* expert_idx;  // [num_experts + 1]
    int N, K, num_experts;
    int n_wg_count, n_per_wg, m_chunks;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_SUB = 16;
        constexpr int M_PER_CHUNK = MT_MAX * 8;

        int wgs_per_expert = n_wg_count * m_chunks;
        int wg_id = item.get_group(0);
        int expert_id = wg_id / wgs_per_expert;
        int local_wg  = wg_id % wgs_per_expert;

        if (expert_id >= num_experts) return;

        int n_wg_id    = local_wg % n_wg_count;
        int m_chunk_id = local_wg / n_wg_count;

        uint32_t tok_start = expert_idx[expert_id];
        uint32_t tok_end   = expert_idx[expert_id + 1];
        int M_e = (int)(tok_end - tok_start);
        if (M_e <= 0) return;

        int m_row_start = m_chunk_id * M_PER_CHUNK;
        if (m_row_start >= M_e) return;

        int m_rows_left = M_e - m_row_start;
        int total_m_tiles_here = (m_rows_left + 7) / 8;
        if (total_m_tiles_here > MT_MAX) total_m_tiles_here = MT_MAX;

        int n_tiles_total = (N + 15) / 16;
        int n_tile_start = n_wg_id * n_per_wg;
        int n_tile_end = n_tile_start + n_per_wg;
        if (n_tile_end > n_tiles_total) n_tile_end = n_tiles_total;

        // Per-tensor scale: single scalar for this expert
        float expert_scale = scale[expert_id];

        // 2D surface descriptors
        uint32_t total_tokens = expert_idx[num_experts];
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = total_tokens - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)(num_experts * N) - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B, 0u, 0u);

        for (int nc = n_tile_start; nc < n_tile_end; nc++) {
            int n_start = nc * 16;
            if (n_start >= N) break;

            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            payB_t.set_y((uint32_t)(expert_id * N + n_start));

            simd<float, 128> acc[MT_MAX];
            #pragma unroll
            for (int i = 0; i < MT_MAX; i++) acc[i] = 0.0f;

            for (int k_base = 0; k_base < K; k_base += 64) {
                #pragma unroll
                for (int sub = 0; sub < 4; sub++) {
                    int k_sub = k_base + sub * K_SUB;
                    if (k_sub >= K) break;

                    payB_t.set_x((uint32_t)(k_sub / 4));
                    simd<uint32_t, 64> w_t = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                        true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);

                    simd<uint32_t, 128> b_vnni;
                    #pragma unroll
                    for (int col = 0; col < 4; col++) {
                        simd<uint32_t, 16> g = w_t.template select<16, 1>(col * 16);
                        simd<uint32_t, 16> b0 = g & 0xFF;
                        simd<uint32_t, 16> b1 = (g & 0xFF00) << 8;
                        b_vnni.template select<16, 1>(col * 32) =
                            fp8_e5m2_pair_to_vnni(b0, b1);
                        simd<uint32_t, 16> gh = g >> 16;
                        simd<uint32_t, 16> b2 = gh & 0xFF;
                        simd<uint32_t, 16> b3 = (gh & 0xFF00) << 8;
                        b_vnni.template select<16, 1>(col * 32 + 16) =
                            fp8_e5m2_pair_to_vnni(b2, b3);
                    }
                    simd<fp16, 256> b_tile =
                        b_vnni.template bit_cast_view<fp16>().read();

                    payA.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < MT_MAX; m++) {
                        payA.set_y((uint32_t)(tok_start + m_row_start + m * 8));
                        simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                        acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile, a);
                    }
                }
            }

            // Output — multiply by scalar per-tensor scale
            for (int m = 0; m < MT_MAX; m++) {
                if (m >= total_m_tiles_here) break;
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m_row_start + m * 8 + mi;
                    if (row < M_e) {
                        simd<float, 16> rf = acc[m].template select<16, 1>(mi * 16);
                        rf *= expert_scale;
                        simd<fp16, 16> out = convert<fp16>(rf);
                        size_t off = (size_t)(tok_start + row) * N + n_start;
                        if (full_n)
                            block_store<fp16, 16>(output + off, out);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[off + ni] = out[ni];
                    }
                }
            }
        }
    }
};

template<int MT_MAX>
inline void moe_gemm_fp8_e5m2_host_v3_pert(
    const fp16* input, const uint8_t* weight, const float* scale,
    fp16* output, const uint32_t* expert_idx,
    int N, int K, int num_experts,
    int n_wg_count, int n_per_wg, int m_chunks,
    sycl::queue& q)
{
    int wgs_per_expert = n_wg_count * m_chunks;
    int total_wgs = num_experts * wgs_per_expert;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)total_wgs}, {1}),
            MoE_FP8_E5M2_Kernel_V3_PerT<MT_MAX>{
                input, weight, scale, output, expert_idx,
                N, K, num_experts, n_wg_count, n_per_wg, m_chunks});
    });
}

inline void moe_gemm_fp8_e5m2_dispatch_pert(
    const fp16* input, const uint8_t* weight, const float* scale,
    fp16* output, const uint32_t* expert_idx,
    int N, int K, int num_experts, int max_tokens_per_expert,
    sycl::queue& q)
{
    int n_tiles = (N + 15) / 16;
    int mt = (max_tokens_per_expert + 7) / 8;

    int mt_max;
    if (mt <= 1) mt_max = 1;
    else if (mt <= 2) mt_max = 2;
    else mt_max = 1;

    int m_chunks = (mt + mt_max - 1) / mt_max;

    // Standard GRF (8 threads/XVE): K-aware dispatch
    //   Large K (gate_up K=2048): n_per_wg=2, weight=64KB/WG → keep moderate WG count
    //   Small K (down K=192):     n_per_wg=8, weight=24KB/WG → fits L1 well
    int wg_cap = (K <= 256) ? 51200 : 38400;

    int n_per_wg = 1;
    int total_wgs = num_experts * n_tiles * m_chunks;
    while (total_wgs > wg_cap && n_per_wg < n_tiles) {
        n_per_wg++;
        while (n_per_wg < n_tiles && n_tiles % n_per_wg != 0) n_per_wg++;
        int n_wg_count = (n_tiles + n_per_wg - 1) / n_per_wg;
        total_wgs = num_experts * n_wg_count * m_chunks;
    }

    int n_wg_count = (n_tiles + n_per_wg - 1) / n_per_wg;

    #define MOE_LAUNCH_V3_PERT(MT) moe_gemm_fp8_e5m2_host_v3_pert<MT>(input, weight, scale, output, expert_idx, N, K, num_experts, n_wg_count, n_per_wg, m_chunks, q)
    if      (mt_max == 1) MOE_LAUNCH_V3_PERT(1);
    else if (mt_max == 2) MOE_LAUNCH_V3_PERT(2);
    else                  MOE_LAUNCH_V3_PERT(4);
    #undef MOE_LAUNCH_V3_PERT
}

// Host launcher (per-N scale)
template<int MT_MAX>
inline void moe_gemm_fp8_e5m2_host_v3(
    const fp16* input, const uint8_t* weight, const float* scale,
    fp16* output, const uint32_t* expert_idx,
    int N, int K, int num_experts,
    int n_wg_count, int n_per_wg, int m_chunks,
    sycl::queue& q)
{
    int wgs_per_expert = n_wg_count * m_chunks;
    int total_wgs = num_experts * wgs_per_expert;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)total_wgs}, {1}),
            MoE_FP8_E5M2_Kernel_V3<MT_MAX>{
                input, weight, scale, output, expert_idx,
                N, K, num_experts, n_wg_count, n_per_wg, m_chunks});
    });
}

// Auto-dispatch V3
inline void moe_gemm_fp8_e5m2_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale,
    fp16* output, const uint32_t* expert_idx,
    int N, int K, int num_experts, int max_tokens_per_expert,
    sycl::queue& q)
{
    int n_tiles = (N + 15) / 16;
    int mt = (max_tokens_per_expert + 7) / 8;

    // MT_MAX selection: prefer small MT_MAX to avoid wasted input loads
    // for experts with few tokens. MT_MAX=4 would force 4 loads per K_SUB
    // even when only 1 M-tile is needed. Weight re-reads across m_chunks
    // hit L3 cache (weight per expert << 18MB L3).
    int mt_max;
    if (mt <= 1) mt_max = 1;
    else if (mt <= 2) mt_max = 2;
    else mt_max = 1;  // For mt>2: use MT_MAX=1 + m_chunks parallelism

    int m_chunks = (mt + mt_max - 1) / mt_max;

    // Merge N-tiles per WG to cap total WGs at ~4096-8192
    // Each WG loops over n_per_wg N-tiles but handles exactly 1 M-chunk
    int n_per_wg = 1;
    int total_wgs = num_experts * n_tiles * m_chunks;
    // Target: keep total WGs <= 8192
    while (total_wgs > 8192 && n_per_wg < n_tiles) {
        n_per_wg++;
        // Ensure even division
        while (n_per_wg < n_tiles && n_tiles % n_per_wg != 0) n_per_wg++;
        int n_wg_count = (n_tiles + n_per_wg - 1) / n_per_wg;
        total_wgs = num_experts * n_wg_count * m_chunks;
    }

    int n_wg_count = (n_tiles + n_per_wg - 1) / n_per_wg;

    #define MOE_LAUNCH_V3(MT) moe_gemm_fp8_e5m2_host_v3<MT>(input, weight, scale, output, expert_idx, N, K, num_experts, n_wg_count, n_per_wg, m_chunks, q)
    if      (mt_max == 1) MOE_LAUNCH_V3(1);
    else if (mt_max == 2) MOE_LAUNCH_V3(2);
    else                  MOE_LAUNCH_V3(4);
    #undef MOE_LAUNCH_V3
}
