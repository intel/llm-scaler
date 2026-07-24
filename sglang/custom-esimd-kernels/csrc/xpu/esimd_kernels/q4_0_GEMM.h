/* q4_0_GEMM.h — GGUF q4_0 INT4 GEMM (prefill / batched M>=2), DPAS-tiled.
 *
 * PTL Xe3 DOES support DPAS (verified 2026-06-02: xmx::dpas<8,8,...> and the
 * full lsc_load_2d/lsc_gather prefill kernel below both AOT-compile for
 * `-device ptl`). The earlier "XeLPG has no DPAS" conclusion was wrong — it
 * came from a different intrinsic failing on a multi-arch build.
 *
 * Port of pathB int4_GEMM_prefill.h (dense INT4 GEMM, DPAS<8,8,fp16> tiles +
 * inline INT4 dequant inside the K-loop), specialized to the q4_0 layout:
 *   * group_size = BS = 32   (q4_0 block; pathB hardcodes 128)
 *   * nibble map = INTERLEAVED, identical to pathB: low nibble -> even k,
 *     high nibble -> odd k. (Same bytes as q4_0_GEMV.h.)
 *   * dequant: w = (nibble - 8) * scale   (symmetric, scale may be negative)
 *
 * input  [M, K]    fp16  (contiguous, row-major)
 * weight [N, K/2]  uint8  (interleaved: byte j low=K_even, high=K_odd)
 * scale  [N, K/32] fp16
 * output [M, N]    fp16
 *
 * 2D grid (M/MAX_M, N/N_TILE). DPAS amortizes the LPDDR weight read across the
 * MAX_M rows of a tile and uses the XMX matrix engine for the MACs, vs the old
 * single-thread loop-M kernel that was ~131x slower than a dense fp16 matmul at
 * M=1024. All q4_0 layers have N%16==0 and K%32==0; only M (token count) is
 * arbitrary, so the M-tail (< MAX_M rows) falls through to a scalar path.
 *
 * Included into esimd_kernel.sycl (sycl/esimd headers + fp16 already in scope).
 * Needs the experimental-esimd alias for the 2D/gather LSC ops.
 */
#pragma once

namespace xesimd = sycl::ext::intel::experimental::esimd;

static constexpr int Q4_0_GEMM_GROUP = 32;   // q4_0 block = BS

// ---- DPAS-tiled main kernel: handles the M rows that fill full MAX_M tiles --
// BS=32 (q4_0 group), MAX_M rows x N_TILE cols per work-group.
template <int BS = 32, int MAX_M = 32, int N_TILE = 16>
struct Q4_0_gemm_dpas_kernel {
    const fp16*    x;        // [M, K]
    const uint8_t* weight;   // [N, K/2]
    const fp16*    scale;    // [N, K/GROUP]
    fp16*          output;   // [M, N]
    int M, N, K;

    void operator()(sycl::id<2> id) const SYCL_ESIMD_KERNEL {
        constexpr int MS = MAX_M / 16;   // DPAS row-tiles (16 rows each)
        constexpr int NS = N_TILE / 8;   // DPAS col-tiles (8 cols each)
        constexpr int ACC_SZ = MS * NS * 128;

        const int K_groups = K / BS;
        const int m_base = (int)id[0] * MAX_M;
        const int n_base = (int)id[1] * N_TILE;

        const uint8_t* w_ptr = weight + (size_t)n_base * (K / 2);
        const fp16* s_base = scale + (size_t)n_base * K_groups;
        const simd<uint32_t, N_TILE> scl_byte_off =
            simd<uint32_t, N_TILE>(0u, 1u) *
            (uint32_t)(K_groups * sizeof(fp16));

        simd<uint32_t, MAX_M> in_off =
            (simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)m_base) *
            (uint32_t)(K * sizeof(fp16));

        simd<float, ACC_SZ> acc(0.0f);

        for (int blk = 0; blk < K_groups; blk++) {
            // One fp16 scale per 32-group, per N-row.
            const simd<fp16, N_TILE> sc =
                gather<fp16, N_TILE>(s_base + blk, scl_byte_off);

            for (int k_base = 0; k_base < BS / 2; k_base += 8) {
                const int x_byte = blk * (BS / 2) + k_base;
                const uint32_t k_off_bytes =
                    (uint32_t)(blk * BS + k_base * 2) * (uint32_t)sizeof(fp16);

                // Gather input tile [MAX_M rows x 16 fp16 K-cols] in VNNI layout.
                simd<fp16, MS * 256> b_tile;
                #pragma unroll
                for (int ms = 0; ms < MS; ms++) {
                    simd<fp16, 256> btmp;
                    btmp.template bit_cast_view<uint32_t>() =
                        xesimd::lsc_gather<uint32_t, 8,
                            xesimd::lsc_data_size::u32,
                            xesimd::cache_hint::cached,
                            xesimd::cache_hint::cached,
                            16, uint32_t>(
                            reinterpret_cast<const uint32_t*>(x),
                            in_off.template select<16, 1>(ms * 16).read()
                                + k_off_bytes);
                    b_tile.template select<256, 1>(ms * 256) = btmp;
                }

                #pragma unroll
                for (int ns = 0; ns < NS; ns++) {
                    // Weight tile: 8 N-rows x 8 bytes (= 16 K-cols), uint8.
                    auto w_raw = xesimd::lsc_load_2d<uint8_t, 8, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached,
                        xesimd::cache_hint::cached>(
                        w_ptr,
                        (unsigned)(K / 2 - 1),
                        (unsigned)(N_TILE - 1),
                        (unsigned)(K / 2 - 1),
                        x_byte,
                        ns * 8);

                    // Unpack interleaved int4: low nibble -> even k, high -> odd.
                    simd<uint8_t, 128> wu;
                    wu.template select<64, 2>(0) = w_raw & (uint8_t)0x0F;
                    wu.template select<64, 2>(1) = w_raw >> 4;
                    simd<fp16, 128> a_tile =
                        wu.template bit_cast_view<int8_t>() - (int8_t)8;
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        a_tile.template select<16, 1>(r * 16) *= sc[ns * 8 + r];
                    }

                    #pragma unroll
                    for (int ms = 0; ms < MS; ms++) {
                        const int idx = ms * NS * 128 + ns * 128;
                        simd<float, 128> a = acc.template select<128, 1>(idx);
                        simd<fp16, 256> bv =
                            b_tile.template select<256, 1>(ms * 256);
                        a = sycl::ext::intel::esimd::xmx::dpas<8, 8, float, float,
                                                              fp16, fp16>(
                                a, bv, a_tile);
                        acc.template select<128, 1>(idx) = a;
                    }
                }
            }
        }

        // Scatter: acc[ms*NS*128 + r*16 + lane] = (M-row m_base+ms*16+lane,
        // N-col n_base + r). Mirrors pathB.
        #pragma unroll
        for (int ms = 0; ms < MS; ms++) {
            simd<uint32_t, 16> lane_off =
                (simd<uint32_t, 16>(0u, 1u) + (uint32_t)(m_base + ms * 16)) *
                (uint32_t)(N * sizeof(fp16));
            #pragma unroll
            for (int r = 0; r < N_TILE; r++) {
                simd<float, 16> v_f =
                    acc.template select<16, 1>(ms * N_TILE * 16 + r * 16);
                simd<fp16, 16> v = convert<fp16>(v_f);
                uint32_t ch = (uint32_t)(n_base + r) * (uint32_t)sizeof(fp16);
                scatter<fp16, 16>(output, lane_off + ch, v);
            }
        }
    }
};

// ---- Scalar M-tail kernel: one WG per output column n, loops the few tail
//      rows. Same interleaved dequant as q4_0_GEMV. Only runs for M % MAX_M
//      rows (typically 0; prefill token counts are usually tile-aligned). -----
struct Q4_0_gemm_tail_kernel {
    const fp16*    input;    // [M, K]
    const uint8_t* weight;   // [N, K/2]
    const fp16*    scale;    // [N, K/32]
    fp16*          output;   // [M, N]
    int M, N, K, m_start;    // process rows [m_start, M)

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;
        const uint8_t* w_row = weight + (size_t)n * (K / 2);
        const fp16*    s_row = scale  + (size_t)n * (K / Q4_0_GEMM_GROUP);
        int blocks = K / Q4_0_GEMM_GROUP;
        for (int m0 = m_start; m0 < M; m0++) {
            simd<float, 16> acc_even = 0.0f, acc_odd = 0.0f;
            const fp16* in_row = input + (size_t)m0 * K;
            for (int b = 0; b < blocks; b++) {
                int k = b * Q4_0_GEMM_GROUP;
                simd<fp16, Q4_0_GEMM_GROUP> iv =
                    block_load<fp16, Q4_0_GEMM_GROUP>(in_row + k);
                simd<float, 16> in_even = iv.template select<16, 2>(0);
                simd<float, 16> in_odd  = iv.template select<16, 2>(1);
                simd<uint8_t, 16> raw = block_load<uint8_t, 16>(w_row + k / 2);
                simd<uint16_t, 16> u16 = convert<uint16_t>(raw);
                simd<float, 16> w_even = convert<float>(u16 & 0x000F) - 8.0f;
                simd<float, 16> w_odd  = convert<float>((u16 >> 4) & 0x000F) - 8.0f;
                float s = static_cast<float>(s_row[b]);
                acc_even += in_even * (w_even * s);
                acc_odd  += in_odd  * (w_odd  * s);
            }
            float dot = reduce<float>(acc_even, std::plus<>())
                      + reduce<float>(acc_odd,  std::plus<>());
            output[(size_t)m0 * N + n] = fp16(dot);
        }
    }
};

inline void q4_0_gemm_host(
    const fp16* input, const uint8_t* weight, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, sycl::queue& q) {
    constexpr int MAX_M = 32, N_TILE = 16;
    // N%16 and K%32 hold for all q4_0 layers (verified). Only M varies.
    const int m_tiles = (int)M / MAX_M;          // full DPAS tiles
    const int m_aligned = m_tiles * MAX_M;

    if (m_tiles > 0 && (N % N_TILE == 0) && (K % Q4_0_GEMM_GROUP == 0)) {
        const int n_n_tiles = (int)N / N_TILE;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<2>(m_tiles, n_n_tiles),
                Q4_0_gemm_dpas_kernel<32, MAX_M, N_TILE>{
                    input, weight, scale, output, (int)M, (int)N, (int)K});
        });
    }
    // M-tail (rows [m_aligned, M)) or full fallback when N/K unaligned.
    int tail_start = (m_tiles > 0 && (N % N_TILE == 0) &&
                      (K % Q4_0_GEMM_GROUP == 0)) ? m_aligned : 0;
    if (tail_start < (int)M) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>((size_t)N, (size_t)1),
                Q4_0_gemm_tail_kernel{input, weight, scale, output,
                                      (int)M, (int)N, (int)K, tail_start});
        });
    }
}
