#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

// ============================================================================
// INT4 GEMM via DPAS (XMX matrix engine), per-group scale.
// Supports GROUP_SIZE = 128 (legacy GGML) and 32 (vLLM 0.21 sym_int4 / q4_0).
// GROUP_SIZE is a template param (default 128); host dispatches on the scale
// last dim, so the op signature is unchanged and group128 callers are intact.
//
// Mirrors the FP8 DPAS V9 kernel (fp8_GEMM_pert.h:FP8_GEMM_DPAS_V9) and
// adapts three things that are INT4-specific:
//
//   (1) Weight is [N, K/2] uint8 packed int4, so the transposed 2D LSC
//       load config is <uint32, 2, N_TILE=16> (8 bytes/N = one K_SUB=16).
//       IMPORTANT: set_x is a uint32-element offset, not a byte offset.
//       One uint32 = 4 bytes = 8 int4 elements, so stepping to K=k_sub
//       means set_x(k_sub / 8).  FP8 V9 uses /4 because each uint32 holds
//       4 FP8 elements; mixing the two up gives silently wrong output.
//
//   (2) Each packed byte already pairs (K_even_nibble, K_odd_nibble) —
//       exactly DPAS's VNNI K-pair layout.  b_tile is built fully vectorized
//       on simd<uint32, 16>: extract one byte per lane, dequant both
//       nibbles, fuse per-group scale via  fp_w = uint4*scale + (-8*scale),
//       interleave the two fp16 halves into uint32 slots.  No per-lane
//       scatter.
//
//   (3) Scale is per-group (one fp16 per 128 K) rather than per-tensor.
//       Folding scale into every b_tile entry is the hot path; a post-DPAS
//       fold (multiply acc_group once per K_LOAD) is a plausible next
//       optimisation but is out of scope here.
//
// Grid: nd_range<1>({(N/16) * K_THREADS}, {K_THREADS}).  Each WG covers 16
// N-cols × all M rows; K_THREADS threads split K, partials reduce via SLM.
// Auto-dispatch picks K_THREADS and M_TILES from (M, N, K) with the same
// heuristic as the FP8 V9 dispatcher.
//
// Requirements:
//   N % 16 == 0
//   K % (K_THREADS * K_LOAD) == 0  (each K-thread owns whole 128-wide loads;
//       K_LOAD=128 is a multiple of both 128 and 32, so this also keeps each
//       thread owning whole scale groups for either GROUP_SIZE)
//   input fp16, weight uint8 [N, K/2], weight_scale fp16 [N, K/GROUP_SIZE],
//   output fp16 [M, N] pre-allocated.
// ============================================================================

static constexpr int INT4_GEMM_GROUP_SIZE = 128;

// Vectorized INT4 dequant + per-group scale + VNNI pack.
//
// byte_u32     : simd<uint32, 16>, each lane = one byte (low 8 bits valid)
//                from a distinct N-row.  Low nibble = K_even, high nibble =
//                K_odd int4.
// scales_fp16  : simd<fp16, 16>, per-N-row scale for the current K-group.
// scale_m8_fp16: precomputed (-8 * scale), so dequant collapses to one FMA.
//
// Returns simd<uint32, 16>, each lane = VNNI pair (K_even fp16 | K_odd fp16)
// with scale applied.
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16>
int4_pair_to_vnni_scaled(simd<uint32_t, 16> byte_u32,
                         simd<fp16, 16>    scales_fp16,
                         simd<fp16, 16>    scale_m8_fp16) {
    simd<uint16_t, 16> lo_u = byte_u32 & 0x0Fu;
    simd<uint16_t, 16> hi_u = (byte_u32 >> 4) & 0x0Fu;
    // uint16 -> fp16 is exact for values 0..15; avoids uint32->float->fp16.
    simd<fp16, 16> lo_fp16 = convert<fp16>(lo_u);
    simd<fp16, 16> hi_fp16 = convert<fp16>(hi_u);
    // fp_w = (uint4 - 8) * scale  ==  uint4 * scale + (-8 * scale)
    lo_fp16 = lo_fp16 * scales_fp16 + scale_m8_fp16;
    hi_fp16 = hi_fp16 * scales_fp16 + scale_m8_fp16;

    simd<uint16_t, 32> interleaved;
    interleaved.template select<16, 2>(0) = lo_fp16.template bit_cast_view<uint16_t>();
    interleaved.template select<16, 2>(1) = hi_fp16.template bit_cast_view<uint16_t>();
    return interleaved.template bit_cast_view<uint32_t>();
}

template<int K_THREADS, int M_TILES, int GROUP_SIZE = INT4_GEMM_GROUP_SIZE>
struct GEMM_int4_pgrp_kernel {
    const fp16*    input;      // [M, K]
    const uint8_t* weight;     // [N, K/2] packed int4
    const fp16*    scale;      // [N, K/GROUP_SIZE]  (per-group scales)
    fp16*          output;     // [M, N]
    int M, N, K;
    int n_groups;              // K / GROUP_SIZE

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int N_TILE = 16;
        constexpr int M_TILE = 8;
        // K_LOAD stays 128 for DPAS efficiency regardless of GROUP_SIZE.
        constexpr int K_LOAD = 128;
        // How many scale groups a single K_LOAD (=128 K) spans: 1 for
        // GROUP_SIZE=128, 4 for GROUP_SIZE=32.  K_SUB(=16) divides GROUP_SIZE
        // for both, so every K_SUB falls entirely inside one scale group.
        constexpr int SUBS_PER_GROUP = GROUP_SIZE / 16;  // 8 (g128) or 2 (g32)
        constexpr int K_SUB  = 16;
        constexpr int SUBS_PER_KLOAD = K_LOAD / K_SUB;  // 8
        constexpr int SLM_PER_THREAD = M_TILES * 128 * 4;
        constexpr int SLM_TOTAL = K_THREADS * SLM_PER_THREAD;

        if constexpr (K_THREADS > 1) {
            slm_init<SLM_TOTAL>();
        }

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * N_TILE;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end   = k_start + k_per_thread;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, K_SUB,  8, 1> payA  (input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<fp16, K_SUB, 16, 1> payA16(input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<fp16, K_SUB, 32, 1> payA32(input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // Transposed 2D load: 2 uint32 × 16 N rows = one K_SUB (8 bytes/N).
        const uint32_t surfW_B = (uint32_t)(K / 2) - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 2, N_TILE, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B,
            0u, (uint32_t)n_start);

        const fp16* s_base = scale + (size_t)n_start * n_groups;

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            // Base group index of this K_LOAD.  For GROUP_SIZE=128 this is the
            // single group; for 32 it is the first of SUBS_PER_KLOAD/SUBS_PER_GROUP
            // (=4) groups covered by the 128-wide load.
            int base_group = k_base / GROUP_SIZE;

            #pragma unroll
            for (int sub = 0; sub < SUBS_PER_KLOAD; sub++) {
                int k_sub = k_base + sub * K_SUB;

                // Which scale group this K_SUB belongs to.  Each K_SUB (=16)
                // lies entirely in one group (K_SUB divides GROUP_SIZE).  For
                // GROUP_SIZE=128, SUBS_PER_GROUP=8 so all subs share base_group;
                // for 32, SUBS_PER_GROUP=2 so the group advances every 2 subs.
                int group_idx = base_group + sub / SUBS_PER_GROUP;
                simd<fp16, N_TILE> scales_f16;
                #pragma unroll
                for (int n = 0; n < N_TILE; n++) {
                    scales_f16[n] = s_base[n * n_groups + group_idx];
                }
                simd<fp16, N_TILE> scale_m8_f16 = scales_f16 * fp16(-8.0f);

                // set_x is a uint32-element offset; one uint32 covers 8 int4
                // K-elements, so stepping by k_sub K means set_x(k_sub / 8).
                payB_t.set_x((uint32_t)(k_sub / 8));
                simd<uint32_t, 32> w_t = xesimd::lsc_load_2d<
                    uint32_t, 2, N_TILE, 1,
                    true, false,
                    xesimd::cache_hint::cached,
                    xesimd::cache_hint::cached>(payB_t);

                simd<uint32_t, 16> colA = w_t.template select<16, 1>(0);
                simd<uint32_t, 16> colB = w_t.template select<16, 1>(16);

                simd<uint32_t, 128> b_vnni_u32;
                #pragma unroll
                for (int kp = 0; kp < 4; kp++) {
                    simd<uint32_t, 16> b = (colA >> (kp * 8)) & 0xFFu;
                    b_vnni_u32.template select<16, 1>(kp * 16) =
                        int4_pair_to_vnni_scaled(b, scales_f16, scale_m8_f16);
                }
                #pragma unroll
                for (int kp = 0; kp < 4; kp++) {
                    simd<uint32_t, 16> b = (colB >> (kp * 8)) & 0xFFu;
                    b_vnni_u32.template select<16, 1>((4 + kp) * 16) =
                        int4_pair_to_vnni_scaled(b, scales_f16, scale_m8_f16);
                }

                simd<fp16, 256> b_tile =
                    b_vnni_u32.template bit_cast_view<fp16>().read();

                if constexpr (M_TILES >= 4) {
                    payA32.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m += 4) {
                        payA32.set_y((uint32_t)(m * 8));
                        simd<fp16, 512> a4 = xesimd::lsc_load_2d<fp16, K_SUB, 32, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA32);
                        #pragma unroll
                        for (int mi = 0; mi < 4; mi++) {
                            simd<fp16, 128> a = a4.template select<128, 1>(mi * 128);
                            acc[m + mi] = dpas<8, 8, float, float, fp16, fp16>(acc[m + mi], b_tile, a);
                        }
                    }
                } else if constexpr (M_TILES >= 2) {
                    payA16.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m += 2) {
                        payA16.set_y((uint32_t)(m * 8));
                        simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, K_SUB, 16, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                        simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                        simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                        acc[m]     = dpas<8, 8, float, float, fp16, fp16>(acc[m],     b_tile, a0);
                        acc[m + 1] = dpas<8, 8, float, float, fp16, fp16>(acc[m + 1], b_tile, a1);
                    }
                } else {
                    payA.set_x((uint32_t)k_sub);
                    payA.set_y(0u);
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, K_SUB, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    acc[0] = dpas<8, 8, float, float, fp16, fp16>(acc[0], b_tile, a);
                }
            }
        }

        if constexpr (K_THREADS == 1) {
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                int n_valid = (n_start + N_TILE <= N) ? N_TILE : (N - n_start);
                bool full_n = (n_valid == N_TILE);
                #pragma unroll
                for (int mi = 0; mi < M_TILE; mi++) {
                    int row = m * M_TILE + mi;
                    if (row < M) {
                        simd<float, N_TILE> row_f = acc[m].template select<N_TILE, 1>(mi * N_TILE);
                        simd<fp16, N_TILE> out_row = convert<fp16>(row_f);
                        if (full_n) {
                            block_store<fp16, N_TILE>(output + (size_t)row * N + n_start, out_row);
                        } else {
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        } else {
            uint32_t slm_base = tid * SLM_PER_THREAD;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                slm_block_store<float, 128>(slm_base + m * 128 * 4, acc[m]);
            }
            barrier();
            if (tid == 0) {
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    simd<float, 128> sum = slm_block_load<float, 128>(m * 128 * 4);
                    #pragma unroll
                    for (int t = 1; t < K_THREADS; t++) {
                        simd<float, 128> partial = slm_block_load<float, 128>(
                            t * SLM_PER_THREAD + m * 128 * 4);
                        sum += partial;
                    }
                    int n_valid = (n_start + N_TILE <= N) ? N_TILE : (N - n_start);
                    bool full_n = (n_valid == N_TILE);
                    #pragma unroll
                    for (int mi = 0; mi < M_TILE; mi++) {
                        int row = m * M_TILE + mi;
                        if (row < M) {
                            simd<float, N_TILE> row_f = sum.template select<N_TILE, 1>(mi * N_TILE);
                            simd<fp16, N_TILE> out_row = convert<fp16>(row_f);
                            if (full_n) {
                                block_store<fp16, N_TILE>(output + (size_t)row * N + n_start, out_row);
                            } else {
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                            }
                        }
                    }
                }
            }
        }
    }
};

template<int K_THREADS, int M_TILES, int GROUP_SIZE = INT4_GEMM_GROUP_SIZE>
inline void gemm_int4_pgrp_host_impl(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int n_groups = (int)K / GROUP_SIZE;
    int num_wg = ((int)N + 15) / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * K_THREADS)}, {(size_t)K_THREADS}),
            GEMM_int4_pgrp_kernel<K_THREADS, M_TILES, GROUP_SIZE>{
                input, weight, scale, output,
                (int)M, (int)N, (int)K, n_groups});
    });
}

// Auto-dispatch (K_THREADS, M_TILES) from (M, N, K).  Matches FP8 V9's
// heuristic: clamp K_THREADS so N_WG * K_THREADS stays near BMG's soft
// concurrent-thread budget, and ensure K splits cleanly on GROUP_SIZE.
inline void GEMM_int4_pgrp_host(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q, int group_size = INT4_GEMM_GROUP_SIZE) {
    int m_tiles = ((int)M + 7) / 8;
    int n_wgs = ((int)N + 15) / 16;
    int k_threads = std::max(1, std::min(4, 640 / std::max(n_wgs, 1)));
    // Each K-thread must own whole 128-wide K_LOADs (K_LOAD=128 is a multiple
    // of both group sizes), so the per-thread K span must be a multiple of 128.
    // This is independent of group_size.
    while (k_threads > 1 && ((int)K % (k_threads * 128) != 0)) k_threads--;
    if (k_threads == 3) k_threads = 2;

    #define DISPATCH(KT, MT, G) gemm_int4_pgrp_host_impl<KT, MT, G>( \
        input, weight, scale, output, M, N, K, q)
    #define DISPATCH_MT(KT, G) \
        do { \
            if      (m_tiles <= 1) DISPATCH(KT, 1, G); \
            else if (m_tiles <= 2) DISPATCH(KT, 2, G); \
            else if (m_tiles <= 4) DISPATCH(KT, 4, G); \
            else                   DISPATCH(KT, 8, G); \
        } while (0)
    #define DISPATCH_KT(G) \
        do { \
            if      (k_threads >= 4) DISPATCH_MT(4, G); \
            else if (k_threads >= 2) DISPATCH_MT(2, G); \
            else                     DISPATCH_MT(1, G); \
        } while (0)

    if (group_size == 32) { DISPATCH_KT(32); }
    else                  { DISPATCH_KT(128); }

    #undef DISPATCH_KT
    #undef DISPATCH_MT
    #undef DISPATCH
}
