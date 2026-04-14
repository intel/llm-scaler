#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl;
using fp16 = sycl::half;

// ============================================================================
// FP8 Per-Tensor GEMM: M=1~64 extension of GEMV
//
// Three regimes based on M:
//   A) M=1~4:  Batched GEMV — M-parallel WGs, K-split SLM reduction
//   B) M=5~8:  Weight-stationary — one WG per N row, TILE_M=8 M-loop
//   C) M=9~64: Weight-stationary — 2D grid, TILE_M=16 M-tiles
//
// Per-tensor scale only. Both E4M3 and E5M2.
// Input: [M, K] fp16, Weight: [N, K] fp8, Output: [M, N] fp16
// ============================================================================

// ---- FP8 -> FP32 dequant ----
// E4M3: sign(1) exp(4,bias=7) mant(3) -> fp16 bits -> fp32
// E5M2: sign(1) exp(5,bias=15) mant(2) -> fp16 bits -> fp32
// Subnormal (exp==0): flush to signed zero
template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant(
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

// ---- FP8 -> FP16 dequant (for future DPAS path) ----
template<int VL>
SYCL_ESIMD_FUNCTION inline simd<fp16, VL> fp8_dequant_fp16(
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

    return fp16_bits.template bit_cast_view<fp16>().read();
}

// ---- VL/K_SPLIT auto-selection (reused from GEMV) ----
inline void select_vl_ks(uint32_t N, uint32_t K, int& vl, int& ks) {
    vl = 512; ks = 1;

    if (K < 512) {
        vl = 128; ks = 1;
    } else if (K == 512) {
        vl = 256; ks = 1;
    }

    if (N <= 128 && K >= 2048) {
        vl = 128; ks = 8;
    } else if (N <= 512 && K >= 2048) {
        vl = 128; ks = 4;
    }

    int kpt = K / ks;
    while (vl > kpt || kpt % vl != 0) {
        if (vl > 128) {
            vl /= 2;
        } else if (ks > 1) {
            ks /= 2;
            kpt = K / ks;
        } else {
            break;
        }
    }
}

// ============================================================================
// Regime A: Batched GEMV (M=1~4)
// nd_range<2>({N*K_SPLIT, M}, {K_SPLIT, 1})
// Each WG handles one (n, m) output element with K_SPLIT threads
// ============================================================================
template<int VL, int K_SPLIT>
struct GEMV_fp8_pert_batched_kernel {
    const fp16*    input;      // [M, K]
    const uint8_t* weight;     // [N, K]
    const float*   scale_ptr;  // scalar
    fp16*          output;     // [M, N]
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int m   = item.get_group(1);
        int lid = item.get_local_id(0);
        if (n >= N || m >= M) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + (size_t)m * K + k);
            simd<float, VL> input_f = iv;

            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(weight + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant<VL>(raw, fp8_mode);

            acc += input_f * wf;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * *scale_ptr;

        if constexpr (K_SPLIT == 1) {
            output[(size_t)m * N + n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[(size_t)m * N + n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

// ============================================================================
// Regime B/C: Weight-Stationary GEMM (M=5~64)
// nd_range<2>({N, ceil(M/TILE_M)}, {1, 1})
// Each single-thread WG handles output[m_start:m_start+TILE_M, n]
// Weight row loaded once, reused across TILE_M input rows
// ============================================================================
template<int VL, int TILE_M>
struct GEMM_fp8_pert_ws_kernel {
    const fp16*    input;      // [M, K]
    const uint8_t* weight;     // [N, K]
    const float*   scale_ptr;  // scalar
    fp16*          output;     // [M, N]
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        int n       = item.get_group(0);
        int m_tile  = item.get_group(1);
        int m_start = m_tile * TILE_M;
        if (n >= N) return;

        simd<float, VL> acc[TILE_M];
        #pragma unroll
        for (int i = 0; i < TILE_M; i++) acc[i] = 0.0f;

        for (int k = 0; k < K; k += VL) {
            // Load weight row once
            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(weight + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant<VL>(raw, fp8_mode);

            // Multiply with each active input row
            #pragma unroll
            for (int i = 0; i < TILE_M; i++) {
                if (m_start + i < M) {
                    simd<fp16, VL> iv = block_load<fp16, VL>(
                        input + (size_t)(m_start + i) * K + k);
                    acc[i] += simd<float, VL>(iv) * wf;
                }
            }
        }

        // Reduce accumulators, apply scale, store
        float s = *scale_ptr;
        #pragma unroll
        for (int i = 0; i < TILE_M; i++) {
            if (m_start + i < M) {
                float sum = reduce<float>(acc[i], std::plus<>()) * s;
                output[(size_t)(m_start + i) * N + n] = fp16(sum);
            }
        }
    }
};

#ifdef GEMM_EXPERIMENTAL  // Dead code: old DPAS kernel struct
// ============================================================================
// Regime D: DPAS-based GEMM (M>=8)
// nd_range<2>({ceil(N/TILE_N), ceil(M/8)}, {1, 1})
// Each single-thread WG computes output[m_start:m_start+8, n_start:n_start+TILE_N]
// Uses XMX dpas<8,8> for fp16 multiply-accumulate
// Weight: dequant FP8->FP16 in registers, pack into VNNI for b_tile
// Input: load as fp16 directly into a_tile
// ============================================================================
template<int TILE_N>
struct GEMM_fp8_pert_dpas_kernel {
    const fp16*    input;      // [M, K]
    const uint8_t* weight;     // [N, K]
    const float*   scale_ptr;  // scalar
    fp16*          output;     // [M, N]
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        int n_tile  = item.get_group(0);
        int m_tile  = item.get_group(1);
        int n_start = n_tile * TILE_N;
        int m_start = m_tile * 8;
        if (n_start >= N) return;

        constexpr int N_DPAS = TILE_N / 16;
        constexpr int K_CHUNK = 128;
        constexpr int K_SUBS = K_CHUNK / 16;  // = 8

        simd<float, 128> acc[N_DPAS];
        #pragma unroll
        for (int j = 0; j < N_DPAS; j++) acc[j] = 0.0f;

        for (int k_base = 0; k_base < K; k_base += K_CHUNK) {
            // Pre-load and dequant weight for all N-groups
            // w_dq[j]: 16 rows × K_CHUNK fp16 = simd<fp16, 16*K_CHUNK>
            simd<fp16, 16 * K_CHUNK> w_dq[N_DPAS];

            #pragma unroll
            for (int j = 0; j < N_DPAS; j++) {
                w_dq[j] = 0;
                int n_base = n_start + j * 16;
                #pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    if (n_base + ni < N) {
                        simd<uint8_t, K_CHUNK> raw = block_load<uint8_t, K_CHUNK>(
                            weight + (size_t)(n_base + ni) * K + k_base);
                        w_dq[j].template select<K_CHUNK, 1>(ni * K_CHUNK) =
                            fp8_dequant_fp16<K_CHUNK>(raw, fp8_mode);
                    }
                }
            }

            // K sub-steps: each processes 16 K-elements via DPAS
            for (int ks = 0; ks < K_SUBS; ks++) {
                int k = k_base + ks * 16;
                if (k >= K) break;

                // Load a_tile ONCE per sub-step (shared across all N-groups)
                simd<fp16, 128> a_tile = 0;
                #pragma unroll
                for (int m = 0; m < 8; m++) {
                    if (m_start + m < M) {
                        a_tile.template select<16, 1>(m * 16) = block_load<fp16, 16>(
                            input + (size_t)(m_start + m) * K + k);
                    }
                }

                // DPAS for each N-group using pre-loaded weight
                #pragma unroll
                for (int j = 0; j < N_DPAS; j++) {
                    auto w_u16 = w_dq[j].template bit_cast_view<uint16_t>();
                    simd<uint32_t, 128> b_vnni;

                    #pragma unroll
                    for (int kp = 0; kp < 8; kp++) {
                        simd<uint16_t, 16> lo = w_u16.template select<16, K_CHUNK>(
                            ks * 16 + 2 * kp);
                        simd<uint16_t, 16> hi = w_u16.template select<16, K_CHUNK>(
                            ks * 16 + 2 * kp + 1);
                        b_vnni.template select<16, 1>(kp * 16) =
                            convert<uint32_t>(lo) | (convert<uint32_t>(hi) << 16);
                    }

                    simd<fp16, 256> b_tile =
                        b_vnni.template bit_cast_view<fp16>().read();

                    acc[j] = dpas<8, 8, float, float, fp16, fp16>(
                        acc[j], b_tile, a_tile);
                }
            }
        }

        // Apply per-tensor scale, convert to fp16, store
        float s = *scale_ptr;
        #pragma unroll
        for (int j = 0; j < N_DPAS; j++) {
            int n_base = n_start + j * 16;
            simd<float, 128> scaled = acc[j] * s;

            #pragma unroll
            for (int m = 0; m < 8; m++) {
                if (m_start + m < M) {
                    int n_cols = (n_base + 16 <= N) ? 16 : (N - n_base);
                    simd<float, 16> row_f = scaled.template select<16, 1>(m * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (n_cols == 16) {
                        block_store<fp16, 16>(
                            output + (size_t)(m_start + m) * N + n_base, out_row);
                    } else {
                        for (int ni = 0; ni < n_cols; ni++) {
                            output[(size_t)(m_start + m) * N + n_base + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};
#endif // GEMM_EXPERIMENTAL — old DPAS kernel struct

// ============================================================================
// Regime E: DPAS V2 — Multi-thread WG (adapted from FP16 GEMM pattern)
//
// Key improvement over Regime D: multiple threads per WG share the same
// input rows via L1 cache, eliminating the N×M input re-read amplification.
//
// WG_SIZE threads per WG, each handles 16 N-columns × M_TILES×8 M-rows.
// N_WG = WG_SIZE × 16 N-columns per WG.
// Grid: ceil(N / N_WG) WGs.
//
// Per K-step (16 K-elements):
//   1. Load weight [16N × 16K] via lsc_load_2d<uint8_t> — unique per thread
//   2. Dequant FP8→FP16 in SIMD registers
//   3. VNNI pack [N=16, K=16] → [K_pair=8, N=16] via stride-16 selects
//   4. Load input [8M × 16K] via lsc_load_2d<fp16> — shared via L1 cache
//   5. DPAS<8,8> for each M-tile
// ============================================================================
namespace xesimd = sycl::ext::intel::experimental::esimd;

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<fp16, VL> fp8_e4m3_dequant_branchless(
    simd<uint8_t, VL> raw);

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<fp16, VL> fp8_e5m2_dequant_branchless(
    simd<uint8_t, VL> raw);

// --- Dequant + VNNI pack helper for 16×16 FP8 tile ---
// Input: raw uint8 in [N=16, K=16] layout from lsc_load_2d
// Output: DPAS b_tile in VNNI format [K_pair=8, N=16] as fp16(uint32)
SYCL_ESIMD_FUNCTION inline simd<fp16, 256>
fp8_tile_to_vnni(simd<uint8_t, 256> raw, int fp8_mode) {
    simd<fp16, 256> w_fp16 = (fp8_mode == 0)
        ? fp8_e4m3_dequant_branchless<256>(raw)
        : fp8_e5m2_dequant_branchless<256>(raw);
    simd<uint16_t, 256> w_u16 = w_fp16.bit_cast_view<uint16_t>().read();
    simd<uint16_t, 256> b_vnni_u16;

    #pragma unroll
    for (int kp = 0; kp < 8; kp++) {
        simd<uint16_t, 16> lo = w_u16.select<16, 16>(2 * kp);
        simd<uint16_t, 16> hi = w_u16.select<16, 16>(2 * kp + 1);
        b_vnni_u16.template select<16, 2>(kp * 32) = lo;
        b_vnni_u16.template select<16, 2>(kp * 32 + 1) = hi;
    }

    return b_vnni_u16.template bit_cast_view<fp16>().read();
}

// --- oneDNN-style branchless FP8 E4M3 → FP16 dequant ---
// Derived from ISA analysis of oneDNN JIT `jit:gemm:any` kernel.
// 4 SIMD ops vs ~8+ for the generic version, completely branchless.
// Math: shl(byte,8) → asr(1) → and(0xBFFF) → mul(256.0)
// This maps FP8 E4M3 (bias=7) directly to FP16 (bias=15) via exponent shift.
// Handles normal values and zero correctly. Subnormals get small error (not flushed).
// NaN (0x7F) maps to ~240 instead of NaN — acceptable for filtered inputs.
template<int VL>
SYCL_ESIMD_FUNCTION inline simd<fp16, VL> fp8_e4m3_dequant_branchless(
    simd<uint8_t, VL> raw) {
    // Step 1: byte → high byte of 16-bit word (equivalent to SHL 8)
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    u16 = u16 << 8;

    // Step 2: arithmetic right shift 1 — sign-extends into bit 14
    simd<int16_t, VL> s16 = u16.template bit_cast_view<int16_t>().read();
    s16 = s16 >> 1;
    simd<uint16_t, VL> result_u16 = s16.template bit_cast_view<uint16_t>().read();

    // Step 3: clear duplicated sign bit (bit 14) — AND 0xBFFF
    result_u16 = result_u16 & 0xBFFF;

    // Step 4: multiply by 256.0 to correct exponent bias (FP8 bias=7 → FP16 bias=15)
    simd<fp16, VL> result = result_u16.template bit_cast_view<fp16>().read();
    result = result * fp16(256.0f);

    return result;
}

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<fp16, VL> fp8_e5m2_dequant_branchless(
    simd<uint8_t, VL> raw) {
    simd<uint16_t, VL> fp16_bits = convert<uint16_t>(raw);
    fp16_bits = fp16_bits << 8;

    simd<uint16_t, VL> sign = fp16_bits & 0x8000;
    fp16_bits.merge(sign, (fp16_bits & 0x7C00) == 0);

    return fp16_bits.template bit_cast_view<fp16>().read();
}

SYCL_ESIMD_FUNCTION inline simd<fp16, 256>
fp8_e5m2_transposed_group_to_vnni(simd<uint32_t, 64> w_t) {
    simd<uint16_t, 256> b_vnni_u16;
    #pragma unroll
    for (int col = 0; col < 4; col++) {
        simd<uint32_t, 16> group = w_t.template select<16, 1>(col * 16);

        simd<uint8_t, 16> b0 = group & 0xFF;
        simd<uint8_t, 16> b1 = (group >> 8) & 0xFF;
        simd<uint8_t, 16> b2 = (group >> 16) & 0xFF;
        simd<uint8_t, 16> b3 = (group >> 24) & 0xFF;

        simd<fp16, 16> d0 = fp8_e5m2_dequant_branchless<16>(b0);
        simd<fp16, 16> d1 = fp8_e5m2_dequant_branchless<16>(b1);
        simd<fp16, 16> d2 = fp8_e5m2_dequant_branchless<16>(b2);
        simd<fp16, 16> d3 = fp8_e5m2_dequant_branchless<16>(b3);

        b_vnni_u16.template select<16, 2>((col * 2) * 16 * 2) = d0.template bit_cast_view<uint16_t>().read();
        b_vnni_u16.template select<16, 2>((col * 2) * 16 * 2 + 1) = d1.template bit_cast_view<uint16_t>().read();
        b_vnni_u16.template select<16, 2>((col * 2 + 1) * 16 * 2) = d2.template bit_cast_view<uint16_t>().read();
        b_vnni_u16.template select<16, 2>((col * 2 + 1) * 16 * 2 + 1) = d3.template bit_cast_view<uint16_t>().read();
    }
    return b_vnni_u16.template bit_cast_view<fp16>().read();
}

// --- Fast VNNI pack using branchless dequant ---
// Uses direct strided writes to pack lo/hi fp16 into uint32 VNNI format
// instead of convert<uint32>+shift+or which generates excessive ALU
SYCL_ESIMD_FUNCTION inline simd<fp16, 256>
fp8_tile_to_vnni_fast(simd<uint8_t, 256> raw) {
    simd<fp16, 256> w_fp16 = fp8_e4m3_dequant_branchless<256>(raw);
    simd<uint16_t, 256> w_u16 = w_fp16.template bit_cast_view<uint16_t>().read();
    simd<uint16_t, 256> b_vnni_u16;

    #pragma unroll
    for (int kp = 0; kp < 8; kp++) {
        simd<uint16_t, 16> lo = w_u16.template select<16, 16>(2 * kp);
        simd<uint16_t, 16> hi = w_u16.template select<16, 16>(2 * kp + 1);
        // Direct strided write: lo→even uint16 slots, hi→odd uint16 slots
        b_vnni_u16.template select<16, 2>(kp * 32) = lo;      // even positions
        b_vnni_u16.template select<16, 2>(kp * 32 + 1) = hi;  // odd positions
    }

    return b_vnni_u16.template bit_cast_view<fp16>().read();
}

// --- Optimized FP8→FP16 dequant (E4M3 only, combined shift+add) ---
// ~10 SIMD ops per 16 elements vs ~13 for the generic version
SYCL_ESIMD_FUNCTION inline simd<uint16_t, 128>
fp8_e4m3_to_fp16_bits_opt(simd<uint16_t, 128> raw) {
    simd<uint16_t, 128> sign_bit = (raw & 0x80) << 8;     // sign at bit 15
    simd<uint16_t, 128> upart    = raw & 0x7F;             // exp|mant (7 bits)
    simd<uint16_t, 128> fp16     = sign_bit | ((upart << 7) + 0x2000);  // bias exp by +8
    fp16.merge(sign_bit, (raw & 0x78) == 0);               // flush subnormal to ±0
    return fp16;
}

// --- Gather-based weight load + inline dequant → VNNI b_tile ---
// Uses lsc_gather<u32, 4, N=16>: 4 uint32 per lane = 16 bytes = 16 fp8 values.
// SOA layout: result[e*16+n] = {fp8[n,k+4e+0], fp8[n,k+4e+1], fp8[n,k+4e+2], fp8[n,k+4e+3]}
// Then extract bytes, dequant to fp16, VNNI pack pairs.
SYCL_ESIMD_FUNCTION inline simd<fp16, 256>
fp8_gather_to_vnni(const uint8_t* weight, int n_start, int k, int K, int fp8_mode) {
    simd<uint32_t, 16> b_off;
    #pragma unroll
    for (int n = 0; n < 16; n++)
        b_off[n] = (uint32_t)(n_start + n) * (uint32_t)K + (uint32_t)k;

    const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(weight);
    simd<uint32_t, 64> raw_u32 = xesimd::lsc_gather<uint32_t, 4,
        xesimd::lsc_data_size::default_size,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached,
        16, uint32_t>(w_u32, b_off);

    // raw_u32[e*16+n] has 4 fp8 bytes for K indices k+4e+{0,1,2,3}
    // Each group e produces 2 VNNI k-pairs: kp=2e, kp=2e+1
    simd<uint32_t, 128> b_vnni;

    #pragma unroll
    for (int e = 0; e < 4; e++) {
        simd<uint32_t, 16> grp = raw_u32.template select<16, 1>(e * 16);

        // Extract 4 bytes as uint16 for dequant
        simd<uint16_t, 16> b0 = convert<uint16_t>(grp & 0xFF);
        simd<uint16_t, 16> b1 = convert<uint16_t>((grp >> 8) & 0xFF);
        simd<uint16_t, 16> b2 = convert<uint16_t>((grp >> 16) & 0xFF);
        simd<uint16_t, 16> b3 = convert<uint16_t>((grp >> 24) & 0xFF);

        simd<uint16_t, 16> fp0, fp1, fp2, fp3;
        if (fp8_mode == 0) {
            // E4M3 optimized dequant
            auto dq = [](simd<uint16_t, 16> r) -> simd<uint16_t, 16> {
                simd<uint16_t, 16> s = (r & 0x80) << 8;
                simd<uint16_t, 16> u = r & 0x7F;
                simd<uint16_t, 16> f = s | ((u << 7) + 0x2000);
                f.merge(s, (r & 0x78) == 0);
                return f;
            };
            fp0 = dq(b0); fp1 = dq(b1); fp2 = dq(b2); fp3 = dq(b3);
        } else {
            // E5M2 dequant
            auto dq = [](simd<uint16_t, 16> r) -> simd<uint16_t, 16> {
                simd<uint16_t, 16> s = (r & 0x80) << 8;
                simd<uint16_t, 16> e = (r >> 2) & 0x1F;
                simd<uint16_t, 16> m = r & 0x3;
                simd<uint16_t, 16> f = s | (e << 10) | (m << 8);
                f.merge(s, e == 0);
                return f;
            };
            fp0 = dq(b0); fp1 = dq(b1); fp2 = dq(b2); fp3 = dq(b3);
        }

        // VNNI pack: pair (fp0,fp1) → kp=2e, pair (fp2,fp3) → kp=2e+1
        b_vnni.template select<16, 1>(e * 2 * 16) =
            convert<uint32_t>(fp0) | (convert<uint32_t>(fp1) << 16);
        b_vnni.template select<16, 1>((e * 2 + 1) * 16) =
            convert<uint32_t>(fp2) | (convert<uint32_t>(fp3) << 16);
    }

    return b_vnni.template bit_cast_view<fp16>().read();
}

#ifdef GEMM_EXPERIMENTAL  // Dead code: V2, V3, V4, V5 kernel structs
template<int WG_SIZE, int M_TILES>
struct FP8_GEMM_DPAS_V2 {
    const fp16*    input;      // [M, K]
    const uint8_t* weight;     // [N, K]
    const float*   scale_ptr;  // scalar
    fp16*          output;     // [M, N]
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int N_PER_THREAD = 16;
        constexpr int N_WG = WG_SIZE * N_PER_THREAD;
        constexpr int K_SUB = 16;

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * N_WG + tid * N_PER_THREAD;
        if (n_start >= N) return;

        // Accumulators: M_TILES × 1 DPAS tile (8×16 = 128 floats each)
        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A: [M, K] fp16
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // --- MAIN K-LOOP ---
        // K_STEP=32: load 2 weight tiles per iteration, do 2 DPAS calls per tile
        // Double-buffer: load weight[k+32], compute weight[k..k+31]
        constexpr int K_STEP = 32;

        // 2D payload for weight B: [N, K] uint8
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        simd<fp16, 256> b_tile[2][2];  // [buf][sub]

        // Prologue: load+dequant k=0,16
        payB.set_x(0u);
        {
            simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            b_tile[0][0] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
        }
        payB.set_x((uint32_t)K_SUB);
        {
            simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            b_tile[0][1] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
        }

        for (int k = 0; k < K - K_STEP; k += K_STEP) {
            // Load next K_STEP (2 sub-tiles)
            payB.set_x((uint32_t)(k + K_STEP));
            {
                simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
                b_tile[1][0] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
            }
            payB.set_x((uint32_t)(k + K_STEP + K_SUB));
            {
                simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
                b_tile[1][1] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
            }

            // Compute sub-step 0 (k)
            payA.set_x((uint32_t)k);
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA.set_y((uint32_t)(m * 8));
                simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile[0][0], a);
            }

            // Compute sub-step 1 (k+16)
            payA.set_x((uint32_t)(k + K_SUB));
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA.set_y((uint32_t)(m * 8));
                simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile[0][1], a);
            }

            b_tile[0][0] = b_tile[1][0];
            b_tile[0][1] = b_tile[1][1];
        }

        // Epilogue: last K_STEP
        {
            int k = K - K_STEP;
            payA.set_x((uint32_t)k);
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA.set_y((uint32_t)(m * 8));
                simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile[0][0], a);
            }
            payA.set_x((uint32_t)(k + K_SUB));
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA.set_y((uint32_t)(m * 8));
                simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile[0][1], a);
            }
        }

        // --- Store: apply per-tensor scale, convert fp32→fp16, write ---
        float s = *scale_ptr;
        #pragma unroll
        for (int m = 0; m < M_TILES; m++) {
            simd<float, 128> scaled = acc[m] * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = m * 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n) {
                        block_store<fp16, 16>(
                            output + (size_t)row * N + n_start, out_row);
                    } else {
                        for (int ni = 0; ni < n_valid; ni++) {
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

// ============================================================================
// Regime F: DPAS V3 — 2D grid (M-tiles as separate WGs)
//
// Key improvement over V2: always M_TILES=1 per thread, with M-tiles
// distributed across separate WGs. This gives M/8 × more total threads
// for better memory latency hiding.
//
// Grid layout: M-inner so WGs with same N but different M are adjacent
// → scheduled on same XE core → L1 weight sharing.
//
// For M=32, N=2560, WG_SIZE=8:
//   V2: 20 WGs, 160 threads (1/XVE)
//   V3: 80 WGs, 640 threads (4/XVE)
// ============================================================================
template<int WG_SIZE>
struct FP8_GEMM_DPAS_V3 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;
    int fp8_mode;
    int m_wgs;  // number of M-tile groups = ceil(M/8)

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int N_PER_THREAD = 16;
        constexpr int N_WG = WG_SIZE * N_PER_THREAD;
        constexpr int K_SUB = 16;

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        // M-inner layout: wg_id = n_wg * m_wgs + m_tile
        int m_tile = wg_id % m_wgs;
        int n_wg   = wg_id / m_wgs;
        int m_start = m_tile * 8;
        int n_start = n_wg * N_WG + tid * N_PER_THREAD;
        if (n_start >= N) return;

        simd<float, 128> acc = 0.0f;

        // 2D payloads
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, (uint32_t)m_start);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        // Double-buffered weight
        simd<fp16, 256> b_buf[2];

        // Prologue
        payB.set_x(0u);
        {
            simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            b_buf[0] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
        }

        for (int k = 0; k < K - K_SUB; k += K_SUB) {
            payB.set_x((uint32_t)(k + K_SUB));
            {
                simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
                b_buf[1] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
            }

            payA.set_x((uint32_t)k);
            simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            acc = dpas<8, 8, float, float, fp16, fp16>(acc, b_buf[0], a);

            b_buf[0] = b_buf[1];
        }

        // Epilogue
        {
            payA.set_x((uint32_t)(K - K_SUB));
            simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            acc = dpas<8, 8, float, float, fp16, fp16>(acc, b_buf[0], a);
        }

        // Store
        float s = *scale_ptr;
        simd<float, 128> scaled = acc * s;
        int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
        bool full_n = (n_valid == 16);

        #pragma unroll
        for (int mi = 0; mi < 8; mi++) {
            int row = m_start + mi;
            if (row < M) {
                simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                simd<fp16, 16> out_row = convert<fp16>(row_f);
                if (full_n) {
                    block_store<fp16, 16>(
                        output + (size_t)row * N + n_start, out_row);
                } else {
                    for (int ni = 0; ni < n_valid; ni++) {
                        output[(size_t)row * N + n_start + ni] = out_row[ni];
                    }
                }
            }
        }
    }
};

// Host dispatcher for DPAS V3
template<int WG_SIZE>
inline void dpas_v3_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    constexpr int N_WG = WG_SIZE * 16;
    int n_wgs = ((int)N + N_WG - 1) / N_WG;
    int m_wgs = ((int)M + 7) / 8;
    int total_wgs = n_wgs * m_wgs;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(total_wgs * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V3<WG_SIZE>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode, m_wgs});
    });
}

// ============================================================================
// Regime G: DPAS V4 — Prefetch pipeline (latency hiding)
//
// Same as V2 but with prefetch issued PF_DIST K-steps ahead.
// With ~550 cycle DRAM latency and ~30 cycle compute per K-step,
// need PF_DIST >= 18 to fully hide latency. Use 8 as practical value
// (128 bytes = 8 K-steps ahead = ~240 cycle look-ahead).
// ============================================================================
template<int WG_SIZE, int M_TILES, int PF_DIST = 8>
struct FP8_GEMM_DPAS_V4 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int N_PER_THREAD = 16;
        constexpr int N_WG = WG_SIZE * N_PER_THREAD;
        constexpr int K_SUB = 16;

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * N_WG + tid * N_PER_THREAD;
        if (n_start >= N) return;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payloads for weight B: one for load, one for prefetch
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);
        xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> payB_pf(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        // Issue initial prefetches
        for (int p = 0; p < PF_DIST && p * K_SUB < K; p++) {
            payB_pf.set_x((uint32_t)(p * K_SUB));
            xesimd::lsc_prefetch_2d<uint8_t, 16, 16, 1,
                false, false,
                xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(payB_pf);
        }

        // Also prefetch input tiles
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA_pf(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        for (int p = 0; p < PF_DIST && p * K_SUB < K; p++) {
            payA_pf.set_x((uint32_t)(p * K_SUB));
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA_pf.set_y((uint32_t)(m * 8));
                xesimd::lsc_prefetch_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(payA_pf);
            }
        }

        // Double-buffered weight
        simd<fp16, 256> b_buf[2];

        // Prologue: load+dequant k=0
        payB.set_x(0u);
        {
            simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            b_buf[0] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
        }

        for (int k = 0; k < K - K_SUB; k += K_SUB) {
            // Issue prefetch PF_DIST steps ahead
            int pf_k = k + PF_DIST * K_SUB;
            if (pf_k < K) {
                payB_pf.set_x((uint32_t)pf_k);
                xesimd::lsc_prefetch_2d<uint8_t, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(payB_pf);
                payA_pf.set_x((uint32_t)pf_k);
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    payA_pf.set_y((uint32_t)(m * 8));
                    xesimd::lsc_prefetch_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(payA_pf);
                }
            }

            // Load next weight
            payB.set_x((uint32_t)(k + K_SUB));
            {
                simd<uint8_t, 256> raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
                b_buf[1] = ((fp8_mode == 0) ? fp8_tile_to_vnni_fast(raw) : fp8_tile_to_vnni(raw, fp8_mode));
            }

            // Load input + DPAS
            payA.set_x((uint32_t)k);
            simd<fp16, 128> a_tile[M_TILES];
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA.set_y((uint32_t)(m * 8));
                a_tile[m] = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            }
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                acc[m] = dpas<8, 8, float, float, fp16, fp16>(
                    acc[m], b_buf[0], a_tile[m]);
            }

            b_buf[0] = b_buf[1];
        }

        // Epilogue
        {
            payA.set_x((uint32_t)(K - K_SUB));
            simd<fp16, 128> a_tile[M_TILES];
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                payA.set_y((uint32_t)(m * 8));
                a_tile[m] = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            }
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                acc[m] = dpas<8, 8, float, float, fp16, fp16>(
                    acc[m], b_buf[0], a_tile[m]);
            }
        }

        // Store
        float s = *scale_ptr;
        #pragma unroll
        for (int m = 0; m < M_TILES; m++) {
            simd<float, 128> scaled = acc[m] * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = m * 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n) {
                        block_store<fp16, 16>(
                            output + (size_t)row * N + n_start, out_row);
                    } else {
                        for (int ni = 0; ni < n_valid; ni++) {
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

// Host dispatcher for DPAS V4
template<int WG_SIZE, int M_TILES, int PF_DIST = 8>
inline void dpas_v4_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    constexpr int N_WG = WG_SIZE * 16;
    int num_wg = ((int)N + N_WG - 1) / N_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V4<WG_SIZE, M_TILES, PF_DIST>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}

// ============================================================================
// Regime H: DPAS V5 — Wide weight load (K_LOAD=64, cache-line aligned)
//
// Key insight: lsc_load_2d<uint8_t, 16, 16> fetches 16 cache lines (1024 bytes)
// for only 256 bytes of data → 25% cache utilization.
// lsc_load_2d<uint8_t, 64, 16> fetches the same 16 cache lines but uses all
// 1024 bytes → 100% utilization. This requires processing 4 DPAS K-subs per load.
//
// Requires K % 64 == 0.
// ============================================================================
template<int WG_SIZE, int M_TILES>
struct FP8_GEMM_DPAS_V5 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int N_PER_THREAD = 16;
        constexpr int N_WG = WG_SIZE * N_PER_THREAD;
        constexpr int K_LOAD = 64;  // Load 64 K-elements at once
        constexpr int K_SUB = 16;   // DPAS processes 16 K-elements

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * N_WG + tid * N_PER_THREAD;
        if (n_start >= N) return;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A: [M, K] fp16
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payload for weight B: [N, K] uint8, wide load (64 cols × 16 rows)
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        for (int k_base = 0; k_base < K; k_base += K_LOAD) {
            // Load 16×64 uint8 weight tile — uses full cache lines
            payB.set_x((uint32_t)k_base);
            simd<uint8_t, 1024> w_raw = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);

            // Process 4 K-sub-steps (each 16 K-elements)
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                simd<uint8_t, 256> sub_raw;
                #pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    sub_raw.template select<16, 1>(ni * 16) =
                        w_raw.template select<16, 1>(ni * 64 + sub * 16);
                }

                simd<fp16, 256> b_tile = (fp8_mode == 0) ?
                    fp8_tile_to_vnni_fast(sub_raw) :
                    fp8_tile_to_vnni(sub_raw, fp8_mode);

                int k = k_base + sub * K_SUB;
                payA.set_x((uint32_t)k);
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    payA.set_y((uint32_t)(m * 8));
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile, a);
                }
            }
        }

        // Store
        float s = *scale_ptr;
        #pragma unroll
        for (int m = 0; m < M_TILES; m++) {
            simd<float, 128> scaled = acc[m] * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = m * 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n) {
                        block_store<fp16, 16>(
                            output + (size_t)row * N + n_start, out_row);
                    } else {
                        for (int ni = 0; ni < n_valid; ni++) {
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};
#endif // GEMM_EXPERIMENTAL — old DPAS, V2-V5

// ============================================================================
// FP8 GEMM DPAS V7: K-split multi-thread WG
// V5 has 160 threads for N=2560 = 1 thread/XVE (12.5% occupancy).
// V6 (M-parallel) was worse because L1 weight reuse < register reuse.
// V7 splits K across threads: each thread loads DIFFERENT weight/input data,
// then reduces partial sums via SLM. More threads = better latency hiding.
// ============================================================================
template<int K_THREADS, int M_TILES>
struct FP8_GEMM_DPAS_V7 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        // SLM layout: K_THREADS × M_TILES × 128 floats
        // Each thread writes its partial accumulators at offset tid * M_TILES * 128 * 4
        constexpr int SLM_PER_THREAD = M_TILES * 128 * 4;  // bytes
        constexpr int SLM_TOTAL = K_THREADS * SLM_PER_THREAD;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * 16;
        if (n_start >= N) return;

        // Each thread handles a K-range
        int k_per_thread = K / K_THREADS;  // Assumes K divisible by K_THREADS
        int k_start = tid * k_per_thread;
        int k_end   = k_start + k_per_thread;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A: [M, K] fp16
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payload for weight B: [N, K] uint8, wide load
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB_pf(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            payB.set_x((uint32_t)k_base);
            simd<uint8_t, 1024> w_raw = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);

            if (k_base + K_LOAD < k_end) {
                payB_pf.set_x((uint32_t)(k_base + K_LOAD));
                xesimd::lsc_prefetch_2d<uint8_t, 64, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_pf);
            }

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                simd<uint8_t, 256> sub_raw;
                #pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    sub_raw.template select<16, 1>(ni * 16) =
                        w_raw.template select<16, 1>(ni * 64 + sub * 16);
                }

                simd<fp16, 256> b_tile = (fp8_mode == 0) ?
                    fp8_tile_to_vnni_fast(sub_raw) :
                    fp8_tile_to_vnni(sub_raw, fp8_mode);

                int k = k_base + sub * K_SUB;
                payA.set_x((uint32_t)k);
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    payA.set_y((uint32_t)(m * 8));
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile, a);
                }
            }
        }

        if constexpr (K_THREADS == 1) {
            // No reduction needed — single thread handles all of K
            float s = *scale_ptr;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                simd<float, 128> scaled = acc[m] * s;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m * 8 + mi;
                    if (row < M) {
                        simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                        simd<fp16, 16> out_row = convert<fp16>(row_f);
                        if (full_n)
                            block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] = out_row[ni];
                    }
                }
            }
        } else {
            if constexpr (K_THREADS == 2) {
                // Special case: keep tid=0 partials in registers and only spill
                // tid=1 to SLM, cutting SLM traffic for the common KT=2 case.
                if (tid == 1) {
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m++) {
                        slm_block_store<float, 128>(m * 128 * 4, acc[m]);
                    }
                }

                barrier();

                if (tid == 0) {
                    float s = *scale_ptr;
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m++) {
                        simd<float, 128> sum = acc[m];
                        simd<float, 128> partial = slm_block_load<float, 128>(m * 128 * 4);
                        sum += partial;
                        simd<float, 128> scaled = sum * s;
                        int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                        bool full_n = (n_valid == 16);
                        #pragma unroll
                        for (int mi = 0; mi < 8; mi++) {
                            int row = m * 8 + mi;
                            if (row < M) {
                                simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                                simd<fp16, 16> out_row = convert<fp16>(row_f);
                                if (full_n)
                                    block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                                else
                                    for (int ni = 0; ni < n_valid; ni++)
                                        output[(size_t)row * N + n_start + ni] = out_row[ni];
                            }
                        }
                    }
                }
            } else {
                // Generic K-thread reduction via SLM.
                uint32_t slm_base = tid * SLM_PER_THREAD;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    slm_block_store<float, 128>(slm_base + m * 128 * 4, acc[m]);
                }

                barrier();

                if (tid == 0) {
                    float s = *scale_ptr;
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m++) {
                        simd<float, 128> sum = slm_block_load<float, 128>(m * 128 * 4);
                        #pragma unroll
                        for (int t = 1; t < K_THREADS; t++) {
                            simd<float, 128> partial = slm_block_load<float, 128>(
                                t * SLM_PER_THREAD + m * 128 * 4);
                            sum += partial;
                        }
                        simd<float, 128> scaled = sum * s;
                        int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                        bool full_n = (n_valid == 16);
                        #pragma unroll
                        for (int mi = 0; mi < 8; mi++) {
                            int row = m * 8 + mi;
                            if (row < M) {
                                simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                                simd<fp16, 16> out_row = convert<fp16>(row_f);
                                if (full_n)
                                    block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                                else
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

// Host dispatcher for DPAS V7
template<int K_THREADS, int M_TILES>
inline void dpas_v7_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    int num_wg = ((int)N + 15) / 16;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * K_THREADS)}, {(size_t)K_THREADS}),
            FP8_GEMM_DPAS_V7<K_THREADS, M_TILES>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}

// V7 auto-dispatch: choose K_THREADS and M_TILES
inline void dpas_v7_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode, sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;
    int n_wgs = ((int)N + 15) / 16;
    // Target ~320-640 total threads. K_THREADS=4 sweet spot for most shapes,
    // but K_THREADS=2 better for large N (N>2560) to avoid SLM reduction overhead.
    // K_THREADS=8 tested and was worse (too much SLM reduction cost).
    int k_threads = std::max(1, std::min(4, 640 / std::max(n_wgs, 1)));
    while (k_threads > 1 && (K % (k_threads * 64) != 0)) k_threads--;
    if (k_threads == 3) k_threads = 2;

    #define V7_DISPATCH(KT, MT) dpas_v7_gemm_fp8_pert_host<KT, MT>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q)
    if (k_threads >= 4) {
        if      (m_tiles <= 1) V7_DISPATCH(4, 1);
        else if (m_tiles <= 2) V7_DISPATCH(4, 2);
        else if (m_tiles <= 4) V7_DISPATCH(4, 4);
        else                   V7_DISPATCH(4, 8);
    } else if (k_threads >= 2) {
        if      (m_tiles <= 1) V7_DISPATCH(2, 1);
        else if (m_tiles <= 2) V7_DISPATCH(2, 2);
        else if (m_tiles <= 4) V7_DISPATCH(2, 4);
        else                   V7_DISPATCH(2, 8);
    } else {
        if      (m_tiles <= 1) V7_DISPATCH(1, 1);
        else if (m_tiles <= 2) V7_DISPATCH(1, 2);
        else if (m_tiles <= 4) V7_DISPATCH(1, 4);
        else                   V7_DISPATCH(1, 8);
    }
    #undef V7_DISPATCH
}

// ============================================================================
// FP8 GEMM DPAS V9: Transposed uint32 load + fused dequant-VNNI
//
// Key insight: oneDNN's JIT fuses the byte→fp16 dequant with VNNI interleave
// via register regioning (shl with stride-4 source, stride-2 dest). ESIMD can't
// emit these instructions, causing 64 narrow mov(4|M0) per sub-tile for VNNI pack
// (55% of the loop body).
//
// V9 solution: Use lsc_load_2d<uint32_t, 4, 16, 1, TRANSPOSE=true> so each uint32
// contains 4 adjacent FP8 bytes for one N-row. Extract byte pairs into uint32
// with lo in [0:15] and hi in [16:31], then dequant BOTH bytes simultaneously
// using uint16 arithmetic on the packed uint32. Result IS the VNNI format —
// zero separate VNNI pack step. All ops at full (16|M0) or (32|M0) width.
//
// Per k_pair (2 bytes × 16 N-lanes):
//   Extraction: 4 ops at (16|M0)
//   Dequant:    4 ops at (32|M0) on 32 uint16 = both bytes simultaneously
//   Total:      ~12 SIMD16 cycles vs ~40 SIMD16 cycles in V7 (3.3× less)
// ============================================================================

// Fused dequant+VNNI for one k_pair from a transposed uint32 column group
// col_data[n] = {B[n, base_k], B[n, base_k+1], B[n, base_k+2], B[n, base_k+3]}
// Processes byte_lo and byte_hi (adjacent bytes within each uint32) into VNNI u32
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16>
fp8_e4m3_pair_to_vnni(simd<uint32_t, 16> b0, simd<uint32_t, 16> b1_shifted) {
    // b0 = byte_lo (0..255 in low bits), b1_shifted = byte_hi << 16
    // Pack: lo in [0:15], hi in [16:31]
    simd<uint32_t, 16> packed = b0 | b1_shifted;

    // Dequant BOTH fp16 halves in-place via bit_cast_view compound assignment.
    // Using compound assignment avoids .read() copies that generate movs.
    // Each operation works on the u16/s16/fp16 view of the same u32 registers.
    packed.template bit_cast_view<uint16_t>() <<= 8;      // SHL8 each u16
    packed.template bit_cast_view<int16_t>() >>= 1;       // ASR1 each s16
    packed.template bit_cast_view<uint16_t>() &= 0xBFFF;  // clear bit 14
    packed.template bit_cast_view<fp16>() *= fp16(256.0f); // bias adjust

    // Result: packed uint32 with {dequant(lo), dequant(hi)} = VNNI format!
    return packed;
}

namespace xesimd = sycl::ext::intel::experimental::esimd;

#ifdef GEMM_EXPERIMENTAL  // Dead code: V11, V12 kernel structs
// ============================================================================
// V11: Multi-N-thread WG — K_THREADS × N_THREADS threads per WG
// Key insight from oneDNN ISA: 4 N-subgroups in same WG share XE core L1 cache
// for input loads. V9's 160 independent WGs don't benefit from L1 input sharing.
// V11 puts N_THREADS N-tiles in the same WG so they share L1 for input loads.
// Each thread is identical to V9 — same registers, same DPAS — just different
// n_start. The L1 cache sharing happens automatically via hardware.
// ============================================================================
template<int K_THREADS, int M_TILES, int N_THREADS>
struct FP8_GEMM_DPAS_V11 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int WG_SIZE = K_THREADS * N_THREADS;
        constexpr int SLM_PER_K_THREAD = M_TILES * 128 * 4;
        constexpr int SLM_PER_N_THREAD = K_THREADS * SLM_PER_K_THREAD;
        constexpr int SLM_TOTAL = N_THREADS * SLM_PER_N_THREAD;
        slm_init<SLM_TOTAL>();

        int wg_id   = item.get_group(0);
        int local_id = item.get_local_id(0);

        // Decompose local_id into N-thread and K-thread
        int n_tid = local_id / K_THREADS;  // which N-tile
        int k_tid = local_id % K_THREADS;  // which K-split

        int n_start = wg_id * (N_THREADS * 16) + n_tid * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = k_tid * k_per_thread;
        int k_end   = k_start + k_per_thread;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A: [M, K] fp16 — SHARED across N-threads via L1
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payload for weight B: [N, K] — unique per N-thread.
        // The transposed uint32 load path is E4M3-specific; E5M2 falls back to
        // the generic gather+dequant helper so we can benchmark the multi-N WG design.
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B,
            0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;

                simd<fp16, 256> b_tile;
                if (fp8_mode == 0) {
                    // Transposed uint32 load for weight — unique per N-thread.
                    payB_t.set_x((uint32_t)(k_sub / 4));
                    simd<uint32_t, 64> w_t = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                        true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);

                    simd<uint32_t, 128> b_vnni_u32;
                    #pragma unroll
                    for (int col = 0; col < 4; col++) {
                        simd<uint32_t, 16> group = w_t.template select<16, 1>(col * 16);
                        simd<uint32_t, 16> b0 = group & 0xFF;
                        simd<uint32_t, 16> b1 = (group & 0xFF00) << 8;
                        b_vnni_u32.template select<16, 1>(col * 2 * 16) =
                            fp8_e4m3_pair_to_vnni(b0, b1);
                        simd<uint32_t, 16> group_hi = group >> 16;
                        simd<uint32_t, 16> b2 = group_hi & 0xFF;
                        simd<uint32_t, 16> b3 = (group_hi & 0xFF00) << 8;
                        b_vnni_u32.template select<16, 1>((col * 2 + 1) * 16) =
                            fp8_e4m3_pair_to_vnni(b2, b3);
                    }
                    b_tile = b_vnni_u32.template bit_cast_view<fp16>().read();
                } else {
                    b_tile = fp8_gather_to_vnni(weight, n_start, k_sub, K, fp8_mode);
                }

                // Load input and DPAS — merged 16-row loads for M_TILES≥2
                if constexpr (M_TILES >= 2) {
                    payA16.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m += 2) {
                        payA16.set_y((uint32_t)(m * 8));
                        simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                        simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                        simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                        acc[m]   = dpas<8, 8, float, float, fp16, fp16>(acc[m],   b_tile, a0);
                        acc[m+1] = dpas<8, 8, float, float, fp16, fp16>(acc[m+1], b_tile, a1);
                    }
                } else {
                    payA.set_x((uint32_t)k_sub);
                    payA.set_y(0u);
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    acc[0] = dpas<8, 8, float, float, fp16, fp16>(acc[0], b_tile, a);
                }
            }
        }

        if constexpr (K_THREADS == 1) {
            float s = *scale_ptr;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                simd<float, 128> scaled = acc[m] * s;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m * 8 + mi;
                    if (row < M) {
                        simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                        simd<fp16, 16> out_row = convert<fp16>(row_f);
                        if (full_n)
                            block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] = out_row[ni];
                    }
                }
            }
        } else {
            // SLM reduction — each N-thread has its own SLM region
            uint32_t slm_base = n_tid * SLM_PER_N_THREAD + k_tid * SLM_PER_K_THREAD;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                slm_block_store<float, 128>(slm_base + m * 128 * 4, acc[m]);
            }

            barrier();

            // Each N-thread's first K-thread reduces and stores output
            if (k_tid == 0) {
                float s = *scale_ptr;
                uint32_t my_slm_base = n_tid * SLM_PER_N_THREAD;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    simd<float, 128> sum = slm_block_load<float, 128>(
                        my_slm_base + m * 128 * 4);
                    #pragma unroll
                    for (int t = 1; t < K_THREADS; t++) {
                        simd<float, 128> partial = slm_block_load<float, 128>(
                            my_slm_base + t * SLM_PER_K_THREAD + m * 128 * 4);
                        sum += partial;
                    }
                    simd<float, 128> scaled = sum * s;
                    int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                    bool full_n = (n_valid == 16);
                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = m * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

// Host dispatcher for V11
template<int K_THREADS, int M_TILES, int N_THREADS>
inline void dpas_v11_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode, sycl::queue& q) {
    constexpr int N_PER_WG = N_THREADS * 16;
    constexpr int WG_SIZE = K_THREADS * N_THREADS;
    int num_wg = ((int)N + N_PER_WG - 1) / N_PER_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V11<K_THREADS, M_TILES, N_THREADS>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}

// V11 auto-dispatch
inline void dpas_v11_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode, sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;
    int n_16 = ((int)N + 15) / 16;

    // Choose N_THREADS to get ~40-80 WGs (matching oneDNN's approach)
    // Target: at least 40 WGs for occupancy, but N_THREADS power of 2
    int n_threads = 4;  // default: 4 N-tiles per WG like oneDNN
    while (n_threads > 1 && (n_16 / n_threads) < 20) n_threads /= 2;
    // For small N, fall back to 1 (= V9 behavior)
    int n_wgs = (n_16 + n_threads - 1) / n_threads;

    int k_threads = std::max(1, std::min(4, 640 / std::max(n_wgs * n_threads, 1)));
    while (k_threads > 1 && (K % (k_threads * 64) != 0)) k_threads--;
    if (k_threads == 3) k_threads = 2;

    // Limit total WG size
    while (k_threads * n_threads > 16) {
        if (n_threads > 1) n_threads /= 2;
        else k_threads /= 2;
    }

    #define V11_DISPATCH(KT, MT, NT) dpas_v11_gemm_fp8_pert_host<KT, MT, NT>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q)

    // Simplified dispatch — main cases
    if (n_threads == 4) {
        if (k_threads >= 4) {
            if      (m_tiles <= 1) V11_DISPATCH(4, 1, 4);
            else if (m_tiles <= 2) V11_DISPATCH(4, 2, 4);
            else if (m_tiles <= 4) V11_DISPATCH(4, 4, 4);
            else                   V11_DISPATCH(4, 8, 4);
        } else if (k_threads >= 2) {
            if      (m_tiles <= 1) V11_DISPATCH(2, 1, 4);
            else if (m_tiles <= 2) V11_DISPATCH(2, 2, 4);
            else if (m_tiles <= 4) V11_DISPATCH(2, 4, 4);
            else                   V11_DISPATCH(2, 8, 4);
        } else {
            if      (m_tiles <= 1) V11_DISPATCH(1, 1, 4);
            else if (m_tiles <= 2) V11_DISPATCH(1, 2, 4);
            else if (m_tiles <= 4) V11_DISPATCH(1, 4, 4);
            else                   V11_DISPATCH(1, 8, 4);
        }
    } else if (n_threads == 2) {
        if (k_threads >= 4) {
            if      (m_tiles <= 1) V11_DISPATCH(4, 1, 2);
            else if (m_tiles <= 2) V11_DISPATCH(4, 2, 2);
            else if (m_tiles <= 4) V11_DISPATCH(4, 4, 2);
            else                   V11_DISPATCH(4, 8, 2);
        } else if (k_threads >= 2) {
            if      (m_tiles <= 1) V11_DISPATCH(2, 1, 2);
            else if (m_tiles <= 2) V11_DISPATCH(2, 2, 2);
            else if (m_tiles <= 4) V11_DISPATCH(2, 4, 2);
            else                   V11_DISPATCH(2, 8, 2);
        } else {
            if      (m_tiles <= 1) V11_DISPATCH(1, 1, 2);
            else if (m_tiles <= 2) V11_DISPATCH(1, 2, 2);
            else if (m_tiles <= 4) V11_DISPATCH(1, 4, 2);
            else                   V11_DISPATCH(1, 8, 2);
        }
    } else {
        // N_THREADS=1 fallback (same per-thread work as V9)
        if (k_threads >= 4) {
            if      (m_tiles <= 1) V11_DISPATCH(4, 1, 1);
            else if (m_tiles <= 2) V11_DISPATCH(4, 2, 1);
            else if (m_tiles <= 4) V11_DISPATCH(4, 4, 1);
            else                   V11_DISPATCH(4, 8, 1);
        } else if (k_threads >= 2) {
            if      (m_tiles <= 1) V11_DISPATCH(2, 1, 1);
            else if (m_tiles <= 2) V11_DISPATCH(2, 2, 1);
            else if (m_tiles <= 4) V11_DISPATCH(2, 4, 1);
            else                   V11_DISPATCH(2, 8, 1);
        } else {
            if      (m_tiles <= 1) V11_DISPATCH(1, 1, 1);
            else if (m_tiles <= 2) V11_DISPATCH(1, 2, 1);
            else if (m_tiles <= 4) V11_DISPATCH(1, 4, 1);
            else                   V11_DISPATCH(1, 8, 1);
        }
    }
    #undef V11_DISPATCH
}

// Forward declarations for cross-references
inline void dpas_v9_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q);

// ============================================================================
// V12: Cooperative input loading via SLM
//
// Key insight from oneDNN ISA analysis: oneDNN's JIT uses cooperative loading
// where threads in a WG distribute input loads, barrier sync, then all threads
// read from SLM. This eliminates redundant input loads across N-subgroups.
// V9's independent WGs each redundantly load the same input data.
//
// Design:
// - WG = N_THREADS × K_THREADS threads (e.g. 4×4=16)
// - All threads step through K in K_LOAD=64 chunks (lock-step)
// - Per K_LOAD: threads cooperatively load input tiles to SLM, barrier,
//   each thread reads its K_SUB's input from SLM, loads own weight, DPAS
// - Double-buffered SLM: load next K_LOAD while computing current → 1 barrier/iter
// - After K loop: K-thread partial sum reduction via SLM (same as V9)
//
// Traffic savings (N=2560, M=32, NT=4):
//   V9:  160 WGs × full input = 20.5MB input loads
//   V12:  40 WGs × full input =  5.1MB input loads (4× reduction)
// ============================================================================
template<int K_THREADS, int M_TILES, int N_THREADS>
struct FP8_GEMM_DPAS_V12 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int WG_SIZE = K_THREADS * N_THREADS;
        constexpr int SUBS_PER_KLOAD = K_LOAD / K_SUB;  // = 4
        constexpr int SUBS_PER_THREAD = SUBS_PER_KLOAD / K_THREADS;
        constexpr int TILES_PER_KLOAD = SUBS_PER_KLOAD * M_TILES;

        // SLM: double-buffered input OR reduction (non-overlapping phases)
        constexpr int INPUT_TILE_BYTES = 128 * 2;  // 8×16 fp16 = 256 bytes
        constexpr int INPUT_BUF_BYTES = TILES_PER_KLOAD * INPUT_TILE_BYTES;
        constexpr int INPUT_SLM = 2 * INPUT_BUF_BYTES;  // double buffer
        constexpr int SLM_PER_K = M_TILES * 128 * 4;    // per K-thread reduction
        constexpr int SLM_PER_N = K_THREADS * SLM_PER_K;
        constexpr int REDUCE_SLM = N_THREADS * SLM_PER_N;
        constexpr int SLM_TOTAL = (INPUT_SLM > REDUCE_SLM) ? INPUT_SLM : REDUCE_SLM;
        slm_init<SLM_TOTAL>();

        int wg_id    = item.get_group(0);
        int local_id = item.get_local_id(0);
        int n_tid    = local_id / K_THREADS;
        int k_tid    = local_id % K_THREADS;

        int n_start = wg_id * (N_THREADS * 16) + n_tid * 16;
        bool valid_n = (n_start < N);  // Must not early-return — barriers

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A (cooperative loading)
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payload for weight B — unique per N-thread
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B,
            0u, (uint32_t)(valid_n ? n_start : 0));

        int k_iters = K / K_LOAD;
        int buf = 0;

        // === Initial cooperative load: buffer 0, k_base=0 ===
        for (int tile_idx = local_id; tile_idx < TILES_PER_KLOAD;
             tile_idx += WG_SIZE) {
            int sub_idx = tile_idx / M_TILES;
            int m_idx   = tile_idx % M_TILES;
            payA.set_x((uint32_t)(sub_idx * K_SUB));
            payA.set_y((uint32_t)(m_idx * 8));
            simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            uint32_t slm_off = (uint32_t)(
                (sub_idx * M_TILES + m_idx) * INPUT_TILE_BYTES);
            slm_block_store<fp16, 128>(slm_off, a);
        }
        barrier();

        // === Main K loop (double-buffered) ===
        for (int ki = 0; ki < k_iters; ki++) {
            int k_base = ki * K_LOAD;
            int next_buf = 1 - buf;

            // --- Load next K_LOAD into other buffer ---
            if (ki < k_iters - 1) {
                int next_k = (ki + 1) * K_LOAD;
                for (int tile_idx = local_id; tile_idx < TILES_PER_KLOAD;
                     tile_idx += WG_SIZE) {
                    int sub_idx = tile_idx / M_TILES;
                    int m_idx   = tile_idx % M_TILES;
                    payA.set_x((uint32_t)(next_k + sub_idx * K_SUB));
                    payA.set_y((uint32_t)(m_idx * 8));
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached,
                        xesimd::cache_hint::cached>(payA);
                    uint32_t slm_off = (uint32_t)(
                        next_buf * INPUT_BUF_BYTES +
                        (sub_idx * M_TILES + m_idx) * INPUT_TILE_BYTES);
                    slm_block_store<fp16, 128>(slm_off, a);
                }
            }

            // --- Compute on current buffer ---
            if (valid_n) {
                #pragma unroll
                for (int sub_off = 0; sub_off < SUBS_PER_THREAD; sub_off++) {
                    int sub_idx = k_tid * SUBS_PER_THREAD + sub_off;
                    int k_sub = k_base + sub_idx * K_SUB;

                    // Transposed uint32 weight load + dequant
                    payB_t.set_x((uint32_t)(k_sub / 4));
                    simd<uint32_t, 64> w_t =
                        xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                            true, false,
                            xesimd::cache_hint::cached,
                            xesimd::cache_hint::cached>(payB_t);

                    simd<uint32_t, 128> b_vnni_u32;
                    #pragma unroll
                    for (int col = 0; col < 4; col++) {
                        simd<uint32_t, 16> group =
                            w_t.template select<16, 1>(col * 16);
                        simd<uint32_t, 16> b0 = group & 0xFF;
                        simd<uint32_t, 16> b1 = (group & 0xFF00) << 8;
                        b_vnni_u32.template select<16, 1>(col * 2 * 16) =
                            fp8_e4m3_pair_to_vnni(b0, b1);
                        simd<uint32_t, 16> group_hi = group >> 16;
                        simd<uint32_t, 16> b2 = group_hi & 0xFF;
                        simd<uint32_t, 16> b3 = (group_hi & 0xFF00) << 8;
                        b_vnni_u32.template select<16, 1>(
                            (col * 2 + 1) * 16) =
                            fp8_e4m3_pair_to_vnni(b2, b3);
                    }
                    simd<fp16, 256> b_tile =
                        b_vnni_u32.template bit_cast_view<fp16>().read();

                    // Read input from SLM and DPAS
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m++) {
                        uint32_t slm_off = (uint32_t)(
                            buf * INPUT_BUF_BYTES +
                            (sub_idx * M_TILES + m) * INPUT_TILE_BYTES);
                        simd<fp16, 128> a =
                            slm_block_load<fp16, 128>(slm_off);
                        acc[m] = dpas<8, 8, float, float, fp16, fp16>(
                            acc[m], b_tile, a);
                    }
                }
            }

            barrier();
            buf = next_buf;
        }

        // === K-reduction and output ===
        if constexpr (K_THREADS == 1) {
            if (!valid_n) return;
            float s = *scale_ptr;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                simd<float, 128> scaled = acc[m] * s;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m * 8 + mi;
                    if (row < M) {
                        simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                        simd<fp16, 16> out_row = convert<fp16>(row_f);
                        if (full_n)
                            block_store<fp16, 16>(
                                output + (size_t)row * N + n_start, out_row);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] =
                                    out_row[ni];
                    }
                }
            }
        } else {
            // SLM reduction (reuse SLM — input phase done)
            if (valid_n) {
                uint32_t slm_base =
                    n_tid * SLM_PER_N + k_tid * SLM_PER_K;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++)
                    slm_block_store<float, 128>(
                        slm_base + m * 128 * 4, acc[m]);
            }

            barrier();

            if (valid_n && k_tid == 0) {
                float s = *scale_ptr;
                uint32_t my_slm = n_tid * SLM_PER_N;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    simd<float, 128> sum =
                        slm_block_load<float, 128>(my_slm + m * 128 * 4);
                    #pragma unroll
                    for (int t = 1; t < K_THREADS; t++) {
                        simd<float, 128> partial =
                            slm_block_load<float, 128>(
                                my_slm + t * SLM_PER_K + m * 128 * 4);
                        sum += partial;
                    }
                    simd<float, 128> scaled = sum * s;
                    int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                    bool full_n = (n_valid == 16);
                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = m * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f =
                                scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(
                                    output + (size_t)row * N + n_start,
                                    out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] =
                                        out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

// Host dispatcher for V12
template<int K_THREADS, int M_TILES, int N_THREADS>
inline void dpas_v12_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    constexpr int N_PER_WG = N_THREADS * 16;
    constexpr int WG_SIZE = K_THREADS * N_THREADS;
    int num_wg = ((int)N + N_PER_WG - 1) / N_PER_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V12<K_THREADS, M_TILES, N_THREADS>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K});
    });
}

// V12 auto-dispatch
inline void dpas_v12_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;
    int n_16 = ((int)N + 15) / 16;

    // Choose N_THREADS: target ~20-80 WGs for occupancy
    int n_threads = 4;
    while (n_threads > 1 && (n_16 + n_threads - 1) / n_threads < 10)
        n_threads /= 2;

    // K_THREADS: 4 for best occupancy (SUBS_PER_KLOAD=4, so KT=1,2,4 all work)
    int k_threads = 4;

    // Check SLM budget (64KB limit)
    int reduce_slm = n_threads * k_threads * m_tiles * 128 * 4;
    if (reduce_slm > 65536 || m_tiles > 4) {
        // Too much SLM — fall back to V9
        dpas_v9_auto_dispatch(input, weight, scale_ptr, output, M, N, K, q);
        return;
    }

    #define V12_DISPATCH(KT, MT, NT) dpas_v12_gemm_fp8_pert_host<KT, MT, NT>( \
        input, weight, scale_ptr, output, M, N, K, q)

    if (n_threads == 4) {
        if      (m_tiles <= 1) V12_DISPATCH(4, 1, 4);
        else if (m_tiles <= 2) V12_DISPATCH(4, 2, 4);
        else                   V12_DISPATCH(4, 4, 4);
    } else if (n_threads == 2) {
        if      (m_tiles <= 1) V12_DISPATCH(4, 1, 2);
        else if (m_tiles <= 2) V12_DISPATCH(4, 2, 2);
        else                   V12_DISPATCH(4, 4, 2);
    } else {
        // NT=1: no cooperative benefit, use V9 instead
        dpas_v9_auto_dispatch(input, weight, scale_ptr, output, M, N, K, q);
    }
    #undef V12_DISPATCH
}

template<int K_THREADS, int M_TILES, int N_THREADS>
struct FP8_GEMM_DPAS_V12_HYB {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int WG_SIZE = K_THREADS * N_THREADS;
        constexpr int SUBS_PER_KLOAD = K_LOAD / K_SUB;
        constexpr int SUBS_PER_THREAD = SUBS_PER_KLOAD / K_THREADS;
        constexpr int TILES_PER_KLOAD = SUBS_PER_KLOAD * M_TILES;
        constexpr int INPUT_TILE_BYTES = 128 * 2;
        constexpr int INPUT_BUF_BYTES = TILES_PER_KLOAD * INPUT_TILE_BYTES;
        constexpr int INPUT_SLM = 2 * INPUT_BUF_BYTES;
        constexpr int SLM_PER_K = M_TILES * 128 * 4;
        constexpr int SLM_PER_N = K_THREADS * SLM_PER_K;
        constexpr int REDUCE_SLM = N_THREADS * SLM_PER_N;
        constexpr int SLM_TOTAL = (INPUT_SLM > REDUCE_SLM) ? INPUT_SLM : REDUCE_SLM;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int local_id = item.get_local_id(0);
        int n_tid = local_id / K_THREADS;
        int k_tid = local_id % K_THREADS;

        int n_start = wg_id * (N_THREADS * 16) + n_tid * 16;
        bool valid_n = (n_start < N);

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)(valid_n ? n_start : 0));

        int k_iters = K / K_LOAD;
        int buf = 0;

        for (int tile_idx = local_id; tile_idx < TILES_PER_KLOAD; tile_idx += WG_SIZE) {
            int sub_idx = tile_idx / M_TILES;
            int m_idx = tile_idx % M_TILES;
            payA.set_x((uint32_t)(sub_idx * K_SUB));
            payA.set_y((uint32_t)(m_idx * 8));
            simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            uint32_t slm_off = (uint32_t)((sub_idx * M_TILES + m_idx) * INPUT_TILE_BYTES);
            slm_block_store<fp16, 128>(slm_off, a);
        }
        barrier();

        for (int ki = 0; ki < k_iters; ki++) {
            int k_base = ki * K_LOAD;
            int next_buf = 1 - buf;

            if (ki < k_iters - 1) {
                int next_k = (ki + 1) * K_LOAD;
                for (int tile_idx = local_id; tile_idx < TILES_PER_KLOAD; tile_idx += WG_SIZE) {
                    int sub_idx = tile_idx / M_TILES;
                    int m_idx = tile_idx % M_TILES;
                    payA.set_x((uint32_t)(next_k + sub_idx * K_SUB));
                    payA.set_y((uint32_t)(m_idx * 8));
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    uint32_t slm_off = (uint32_t)(
                        next_buf * INPUT_BUF_BYTES +
                        (sub_idx * M_TILES + m_idx) * INPUT_TILE_BYTES);
                    slm_block_store<fp16, 128>(slm_off, a);
                }
            }

            if (valid_n) {
                payB.set_x((uint32_t)(k_base + k_tid * (K_LOAD / K_THREADS)));
                simd<uint8_t, 1024> w_raw = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);

                #pragma unroll
                for (int sub_off = 0; sub_off < SUBS_PER_THREAD; sub_off++) {
                    int sub_idx = k_tid * SUBS_PER_THREAD + sub_off;
                    simd<uint8_t, 256> sub_raw;
                    #pragma unroll
                    for (int ni = 0; ni < 16; ni++) {
                        sub_raw.template select<16, 1>(ni * 16) =
                            w_raw.template select<16, 1>(ni * 64 + sub_off * 16);
                    }

                    simd<fp16, 256> b_tile = (fp8_mode == 0)
                        ? fp8_tile_to_vnni_fast(sub_raw)
                        : fp8_tile_to_vnni(sub_raw, fp8_mode);

                    #pragma unroll
                    for (int m = 0; m < M_TILES; m++) {
                        uint32_t slm_off = (uint32_t)(
                            buf * INPUT_BUF_BYTES +
                            (sub_idx * M_TILES + m) * INPUT_TILE_BYTES);
                        simd<fp16, 128> a = slm_block_load<fp16, 128>(slm_off);
                        acc[m] = dpas<8, 8, float, float, fp16, fp16>(acc[m], b_tile, a);
                    }
                }
            }

            barrier();
            buf = next_buf;
        }

        if constexpr (K_THREADS == 1) {
            if (!valid_n) return;
            float s = *scale_ptr;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                simd<float, 128> scaled = acc[m] * s;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m * 8 + mi;
                    if (row < M) {
                        simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                        simd<fp16, 16> out_row = convert<fp16>(row_f);
                        if (full_n)
                            block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] = out_row[ni];
                    }
                }
            }
        } else {
            if (valid_n) {
                uint32_t slm_base = n_tid * SLM_PER_N + k_tid * SLM_PER_K;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    slm_block_store<float, 128>(slm_base + m * 128 * 4, acc[m]);
                }
            }

            barrier();

            if (valid_n && k_tid == 0) {
                float s = *scale_ptr;
                uint32_t my_slm = n_tid * SLM_PER_N;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    simd<float, 128> sum = slm_block_load<float, 128>(my_slm + m * 128 * 4);
                    #pragma unroll
                    for (int t = 1; t < K_THREADS; t++) {
                        simd<float, 128> partial = slm_block_load<float, 128>(
                            my_slm + t * SLM_PER_K + m * 128 * 4);
                        sum += partial;
                    }
                    simd<float, 128> scaled = sum * s;
                    int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                    bool full_n = (n_valid == 16);
                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = m * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

template<int K_THREADS, int M_TILES, int N_THREADS>
inline void dpas_v12_hyb_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode, sycl::queue& q) {
    constexpr int N_PER_WG = N_THREADS * 16;
    constexpr int WG_SIZE = K_THREADS * N_THREADS;
    int num_wg = ((int)N + N_PER_WG - 1) / N_PER_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V12_HYB<K_THREADS, M_TILES, N_THREADS>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}
#endif // GEMM_EXPERIMENTAL — V11, V12

// ============================================================================
// E5M2 M≈12 specialized kernel
//
// Fixed design for 9<=M<=16 decode GEMM:
// - E5M2 only
// - K_THREADS=2, M_TILES=2
// - Each thread handles half of K
// - Uses merged 16x16 input loads to feed two DPAS tiles from one input send
// - Uses minimal KT=2 reduction: tid=1 spills, tid=0 keeps registers
//
// This is intentionally a dedicated kernel instead of another general family.
// ============================================================================
struct FP8_GEMM_E5M2_M12_SPECIAL {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_THREADS = 2;
        constexpr int M_TILES = 2;
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int SLM_PER_TILE = 128 * 4;
        constexpr int SLM_TOTAL = M_TILES * SLM_PER_TILE;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid = item.get_local_id(0);

        int n_start = wg_id * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end = k_start + k_per_thread;

        simd<float, 128> acc0 = 0.0f;
        simd<float, 128> acc1 = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            payB.set_x((uint32_t)k_base);
            simd<uint8_t, 1024> w_raw = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                simd<uint8_t, 256> sub_raw;
                #pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    sub_raw.template select<16, 1>(ni * 16) =
                        w_raw.template select<16, 1>(ni * 64 + sub * 16);
                }

                simd<fp16, 256> b_tile = fp8_tile_to_vnni(sub_raw, 1);

                int k_sub = k_base + sub * K_SUB;
                payA16.set_x((uint32_t)k_sub);
                payA16.set_y(0u);
                simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                acc0 = dpas<8, 8, float, float, fp16, fp16>(acc0, b_tile, a0);
                acc1 = dpas<8, 8, float, float, fp16, fp16>(acc1, b_tile, a1);
            }
        }

        if (tid == 1) {
            slm_block_store<float, 128>(0, acc0);
            slm_block_store<float, 128>(SLM_PER_TILE, acc1);
        }

        barrier();

        if (tid == 0) {
            float s = *scale_ptr;
            acc0 += slm_block_load<float, 128>(0);
            acc1 += slm_block_load<float, 128>(SLM_PER_TILE);
            simd<float, 128> scaled0 = acc0 * s;
            simd<float, 128> scaled1 = acc1 * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled0.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled1.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }
        }
    }
};

inline void e5m2_m12_special_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int num_wg = ((int)N + 15) / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * 2)}, {(size_t)2}),
            FP8_GEMM_E5M2_M12_SPECIAL{input, weight, scale_ptr, output, (int)M, (int)N, (int)K});
    });
}

struct FP8_GEMM_E5M2_M12_SPECIAL_NT2 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_THREADS = 2;
        constexpr int N_TILES = 2;
        constexpr int M_TILES = 2;
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int ACC_COUNT = N_TILES * M_TILES;
        constexpr int SLM_PER_THREAD = ACC_COUNT * 128 * 4;
        constexpr int SLM_TOTAL = K_THREADS * SLM_PER_THREAD;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid = item.get_local_id(0);

        int n_base = wg_id * 32;
        if (n_base >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end = k_start + k_per_thread;

        simd<float, 128> acc[N_TILES][M_TILES];
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++)
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++)
                acc[nt][mt] = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB0(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_base);
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB1(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)(n_base + 16));

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            payB0.set_x((uint32_t)k_base);
            simd<uint8_t, 1024> w_raw0 = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB0);

            simd<uint8_t, 1024> w_raw1;
            bool valid_nt1 = (n_base + 16 < N);
            if (valid_nt1) {
                payB1.set_x((uint32_t)k_base);
                w_raw1 = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB1);
            }

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                simd<uint8_t, 256> sub_raw0;
                simd<uint8_t, 256> sub_raw1;
                #pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    sub_raw0.template select<16, 1>(ni * 16) =
                        w_raw0.template select<16, 1>(ni * 64 + sub * 16);
                    if (valid_nt1) {
                        sub_raw1.template select<16, 1>(ni * 16) =
                            w_raw1.template select<16, 1>(ni * 64 + sub * 16);
                    }
                }

                simd<fp16, 256> b_tile0 = fp8_tile_to_vnni(sub_raw0, 1);
                simd<fp16, 256> b_tile1;
                if (valid_nt1) {
                    b_tile1 = fp8_tile_to_vnni(sub_raw1, 1);
                }

                int k_sub = k_base + sub * K_SUB;
                payA16.set_x((uint32_t)k_sub);
                payA16.set_y(0u);
                simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                acc[0][0] = dpas<8, 8, float, float, fp16, fp16>(acc[0][0], b_tile0, a0);
                acc[0][1] = dpas<8, 8, float, float, fp16, fp16>(acc[0][1], b_tile0, a1);
                if (valid_nt1) {
                    acc[1][0] = dpas<8, 8, float, float, fp16, fp16>(acc[1][0], b_tile1, a0);
                    acc[1][1] = dpas<8, 8, float, float, fp16, fp16>(acc[1][1], b_tile1, a1);
                }
            }
        }

        uint32_t slm_base = tid * SLM_PER_THREAD;
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++)
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++)
                slm_block_store<float, 128>(slm_base + (nt * M_TILES + mt) * 128 * 4, acc[nt][mt]);

        barrier();

        if (tid == 0) {
            float s = *scale_ptr;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++) {
                int n_start = n_base + nt * 16;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                if (n_valid <= 0) continue;
                bool full_n = (n_valid == 16);

                #pragma unroll
                for (int mt = 0; mt < M_TILES; mt++) {
                    simd<float, 128> sum = slm_block_load<float, 128>((nt * M_TILES + mt) * 128 * 4);
                    simd<float, 128> partial = slm_block_load<float, 128>(SLM_PER_THREAD + (nt * M_TILES + mt) * 128 * 4);
                    sum += partial;
                    simd<float, 128> scaled = sum * s;

                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = mt * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

inline void e5m2_m12_special_nt2_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int num_wg = ((int)N + 31) / 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * 2)}, {(size_t)2}),
            FP8_GEMM_E5M2_M12_SPECIAL_NT2{input, weight, scale_ptr, output, (int)M, (int)N, (int)K});
    });
}

struct FP8_GEMM_E5M2_M12_SPECIAL_PIPE {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_THREADS = 2;
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int SLM_PER_TILE = 128 * 4;
        constexpr int SLM_TOTAL = 2 * SLM_PER_TILE;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid = item.get_local_id(0);
        int n_start = wg_id * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end = k_start + k_per_thread;

        simd<float, 128> acc0 = 0.0f;
        simd<float, 128> acc1 = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint8_t, 64, 16, 1> payB(
            weight, surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        simd<fp16, 256> b_tiles[2][4];
        int buf = 0;

        auto load_b_tiles = [&](int load_k_base, int dst_buf) SYCL_ESIMD_FUNCTION {
            payB.set_x((uint32_t)load_k_base);
            simd<uint8_t, 1024> w_raw = xesimd::lsc_load_2d<uint8_t, 64, 16, 1,
                false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                simd<uint8_t, 256> sub_raw;
                #pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    sub_raw.template select<16, 1>(ni * 16) =
                        w_raw.template select<16, 1>(ni * 64 + sub * 16);
                }
                b_tiles[dst_buf][sub] = fp8_tile_to_vnni(sub_raw, 1);
            }
        };

        load_b_tiles(k_start, buf);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            int next_buf = 1 - buf;
            if (k_base + K_LOAD < k_end) {
                load_b_tiles(k_base + K_LOAD, next_buf);
            }

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;
                payA16.set_x((uint32_t)k_sub);
                payA16.set_y(0u);
                simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                acc0 = dpas<8, 8, float, float, fp16, fp16>(acc0, b_tiles[buf][sub], a0);
                acc1 = dpas<8, 8, float, float, fp16, fp16>(acc1, b_tiles[buf][sub], a1);
            }

            buf = next_buf;
        }

        if (tid == 1) {
            slm_block_store<float, 128>(0, acc0);
            slm_block_store<float, 128>(SLM_PER_TILE, acc1);
        }

        barrier();

        if (tid == 0) {
            float s = *scale_ptr;
            acc0 += slm_block_load<float, 128>(0);
            acc1 += slm_block_load<float, 128>(SLM_PER_TILE);
            simd<float, 128> scaled0 = acc0 * s;
            simd<float, 128> scaled1 = acc1 * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled0.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled1.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }
        }
    }
};

inline void e5m2_m12_special_pipe_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int num_wg = ((int)N + 15) / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * 2)}, {(size_t)2}),
            FP8_GEMM_E5M2_M12_SPECIAL_PIPE{input, weight, scale_ptr, output, (int)M, (int)N, (int)K});
    });
}

struct FP8_GEMM_E5M2_M12_SPECIAL_TLOAD {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_THREADS = 2;
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int SLM_PER_TILE = 128 * 4;
        constexpr int SLM_TOTAL = 2 * SLM_PER_TILE;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid = item.get_local_id(0);
        int n_start = wg_id * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end = k_start + k_per_thread;

        simd<float, 128> acc0 = 0.0f;
        simd<float, 128> acc1 = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;
                payB_t.set_x((uint32_t)(k_sub / 4));
                simd<uint32_t, 64> w_t = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                    true, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);

                simd<fp16, 256> b_tile = fp8_e5m2_transposed_group_to_vnni(w_t);

                payA16.set_x((uint32_t)k_sub);
                payA16.set_y(0u);
                simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                acc0 = dpas<8, 8, float, float, fp16, fp16>(acc0, b_tile, a0);
                acc1 = dpas<8, 8, float, float, fp16, fp16>(acc1, b_tile, a1);
            }
        }

        if (tid == 1) {
            slm_block_store<float, 128>(0, acc0);
            slm_block_store<float, 128>(SLM_PER_TILE, acc1);
        }

        barrier();

        if (tid == 0) {
            float s = *scale_ptr;
            acc0 += slm_block_load<float, 128>(0);
            acc1 += slm_block_load<float, 128>(SLM_PER_TILE);
            simd<float, 128> scaled0 = acc0 * s;
            simd<float, 128> scaled1 = acc1 * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled0.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled1.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }
        }
    }
};

inline void e5m2_m12_special_tload_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int num_wg = ((int)N + 15) / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * 2)}, {(size_t)2}),
            FP8_GEMM_E5M2_M12_SPECIAL_TLOAD{input, weight, scale_ptr, output, (int)M, (int)N, (int)K});
    });
}

struct FP8_GEMM_E5M2_M12_SPECIAL_TLOAD_PIPE {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_THREADS = 2;
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int SLM_PER_TILE = 128 * 4;
        constexpr int SLM_TOTAL = 2 * SLM_PER_TILE;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid = item.get_local_id(0);
        int n_start = wg_id * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end = k_start + k_per_thread;

        simd<float, 128> acc0 = 0.0f;
        simd<float, 128> acc1 = 0.0f;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B, 0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            simd<uint32_t, 64> w_t[4];
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;
                payB_t.set_x((uint32_t)(k_sub / 4));
                w_t[sub] = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                    true, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);
            }

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;
                simd<fp16, 256> b_tile = fp8_e5m2_transposed_group_to_vnni(w_t[sub]);
                payA16.set_x((uint32_t)k_sub);
                payA16.set_y(0u);
                simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                    false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                acc0 = dpas<8, 8, float, float, fp16, fp16>(acc0, b_tile, a0);
                acc1 = dpas<8, 8, float, float, fp16, fp16>(acc1, b_tile, a1);
            }
        }

        if (tid == 1) {
            slm_block_store<float, 128>(0, acc0);
            slm_block_store<float, 128>(SLM_PER_TILE, acc1);
        }

        barrier();

        if (tid == 0) {
            float s = *scale_ptr;
            acc0 += slm_block_load<float, 128>(0);
            acc1 += slm_block_load<float, 128>(SLM_PER_TILE);
            simd<float, 128> scaled0 = acc0 * s;
            simd<float, 128> scaled1 = acc1 * s;
            int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
            bool full_n = (n_valid == 16);

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled0.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 8; mi++) {
                int row = 8 + mi;
                if (row < M) {
                    simd<float, 16> row_f = scaled1.select<16, 1>(mi * 16);
                    simd<fp16, 16> out_row = convert<fp16>(row_f);
                    if (full_n)
                        block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                    else
                        for (int ni = 0; ni < n_valid; ni++)
                            output[(size_t)row * N + n_start + ni] = out_row[ni];
                }
            }
        }
    }
};

inline void e5m2_m12_special_tload_pipe_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int num_wg = ((int)N + 15) / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * 2)}, {(size_t)2}),
            FP8_GEMM_E5M2_M12_SPECIAL_TLOAD_PIPE{input, weight, scale_ptr, output, (int)M, (int)N, (int)K});
    });
}

template<int K_THREADS, int M_TILES>
struct FP8_GEMM_DPAS_V9 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int SLM_PER_THREAD = M_TILES * 128 * 4;
        constexpr int SLM_TOTAL = K_THREADS * SLM_PER_THREAD;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end   = k_start + k_per_thread;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A: [M, K] fp16
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        // Wider payload for merged 16-row input loads (2 M-tiles at once)
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        // Super-merged payload for 32-row input loads (4 M-tiles at once)
        xesimd::config_2d_mem_access<fp16, 16, 32, 1> payA32(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payload for weight B: [N, K] uint8
        // Reinterpret as uint32 surface for transposed loads
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B,
            0u, (uint32_t)n_start);

        // Second weight payload for prefetching next sub-step
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_pf(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B,
            0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            // Process 4 sub-steps of 16 K-elements each
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;

                // Transposed uint32 load: 4 cols × 16 rows = 16 bytes/row × 16 rows
                // After transpose: result[col*16 + n] = uint32 at B[n, 4*(x+col)]
                payB_t.set_x((uint32_t)(k_sub / 4));
                simd<uint32_t, 64> w_t = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                    true, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);

                // Prefetch weight into L1 ahead of time
                // Only beneficial when send queue has spare capacity (not too many WGs)
                if constexpr (M_TILES <= 2) {
                    // For small M: prefetch next K_SUB (1 ahead)
                    if (sub < 3) {
                        payB_pf.set_x((uint32_t)((k_sub + K_SUB) / 4));
                        xesimd::lsc_prefetch_2d<uint32_t, 4, 16, 1, false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_pf);
                    } else if (k_base + K_LOAD < k_end) {
                        payB_pf.set_x((uint32_t)((k_base + K_LOAD) / 4));
                        xesimd::lsc_prefetch_2d<uint32_t, 4, 16, 1, false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_pf);
                    }
                }

                // Build VNNI b_tile from 4 transposed columns
                // Each column has 4 bytes = 2 k_pairs
                simd<uint32_t, 128> b_vnni_u32;

                #pragma unroll
                for (int col = 0; col < 4; col++) {
                    simd<uint32_t, 16> group = w_t.template select<16, 1>(col * 16);

                    // k_pair 0: bytes 0, 1
                    simd<uint32_t, 16> b0 = group & 0xFF;
                    simd<uint32_t, 16> b1 = (group & 0xFF00) << 8;
                    b_vnni_u32.template select<16, 1>(col * 2 * 16) =
                        fp8_e4m3_pair_to_vnni(b0, b1);

                    // k_pair 1: bytes 2, 3
                    simd<uint32_t, 16> group_hi = group >> 16;
                    simd<uint32_t, 16> b2 = group_hi & 0xFF;
                    simd<uint32_t, 16> b3 = (group_hi & 0xFF00) << 8;
                    b_vnni_u32.template select<16, 1>((col * 2 + 1) * 16) =
                        fp8_e4m3_pair_to_vnni(b2, b3);
                }

                simd<fp16, 256> b_tile =
                    b_vnni_u32.template bit_cast_view<fp16>().read();

                // Load input and DPAS — use super-merged 32-row loads for M_TILES>=4
                if constexpr (M_TILES >= 4) {
                    payA32.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m += 4) {
                        payA32.set_y((uint32_t)(m * 8));
                        simd<fp16, 512> a4 = xesimd::lsc_load_2d<fp16, 16, 32, 1,
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
                        simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                        simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                        simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                        acc[m]   = dpas<8, 8, float, float, fp16, fp16>(acc[m],   b_tile, a0);
                        acc[m+1] = dpas<8, 8, float, float, fp16, fp16>(acc[m+1], b_tile, a1);
                    }
                } else {
                    payA.set_x((uint32_t)k_sub);
                    payA.set_y(0u);
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    acc[0] = dpas<8, 8, float, float, fp16, fp16>(acc[0], b_tile, a);
                }
            }
        }

        if constexpr (K_THREADS == 1) {
            float s = *scale_ptr;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                simd<float, 128> scaled = acc[m] * s;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m * 8 + mi;
                    if (row < M) {
                        simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                        simd<fp16, 16> out_row = convert<fp16>(row_f);
                        if (full_n)
                            block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] = out_row[ni];
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
                float s = *scale_ptr;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    simd<float, 128> sum = slm_block_load<float, 128>(m * 128 * 4);
                    #pragma unroll
                    for (int t = 1; t < K_THREADS; t++) {
                        simd<float, 128> partial = slm_block_load<float, 128>(
                            t * SLM_PER_THREAD + m * 128 * 4);
                        sum += partial;
                    }
                    simd<float, 128> scaled = sum * s;
                    int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                    bool full_n = (n_valid == 16);
                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = m * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

#ifdef GEMM_EXPERIMENTAL  // Dead code: V13, V10 kernel structs + dispatchers
// ============================================================================
// V13: V9 + software-pipelined weight loading
// Pre-load all 4 weight K_SUBs before processing → overlaps sends with each other
// Also prefetches next K_LOAD's first weight sub during last processing sub
// ============================================================================
template<int K_THREADS, int M_TILES>
struct FP8_GEMM_DPAS_V13 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int SLM_PER_THREAD = M_TILES * 128 * 4;
        constexpr int SLM_TOTAL = K_THREADS * SLM_PER_THREAD;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_start = wg_id * 16;
        if (n_start >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end   = k_start + k_per_thread;

        simd<float, 128> acc[M_TILES];
        #pragma unroll
        for (int i = 0; i < M_TILES; i++) acc[i] = 0.0f;

        // 2D payload for input A
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<fp16, 16, 16, 1> payA16(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<fp16, 16, 32, 1> payA32(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payload for weight B (transposed uint32)
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;
        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
            reinterpret_cast<const uint32_t*>(weight),
            surfW_B, surfH_B, surfW_B,
            0u, (uint32_t)n_start);

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            // Phase 1: Pre-load all 4 weight K_SUBs (batch sends for pipelining)
            simd<uint32_t, 64> w_t0, w_t1, w_t2, w_t3;
            payB_t.set_x((uint32_t)(k_base / 4));
            w_t0 = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);
            payB_t.set_x((uint32_t)((k_base + K_SUB) / 4));
            w_t1 = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);
            payB_t.set_x((uint32_t)((k_base + 2 * K_SUB) / 4));
            w_t2 = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);
            payB_t.set_x((uint32_t)((k_base + 3 * K_SUB) / 4));
            w_t3 = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t);

            // Phase 2: Process each K_SUB — dequant + input load + DPAS
            simd<uint32_t, 64>* w_ptrs[4] = {&w_t0, &w_t1, &w_t2, &w_t3};
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;
                simd<uint32_t, 64>& w_t = *w_ptrs[sub];

                // Build VNNI b_tile from 4 transposed columns
                simd<uint32_t, 128> b_vnni_u32;
                #pragma unroll
                for (int col = 0; col < 4; col++) {
                    simd<uint32_t, 16> group = w_t.template select<16, 1>(col * 16);
                    simd<uint32_t, 16> b0 = group & 0xFF;
                    simd<uint32_t, 16> b1 = (group & 0xFF00) << 8;
                    b_vnni_u32.template select<16, 1>(col * 2 * 16) =
                        fp8_e4m3_pair_to_vnni(b0, b1);
                    simd<uint32_t, 16> group_hi = group >> 16;
                    simd<uint32_t, 16> b2 = group_hi & 0xFF;
                    simd<uint32_t, 16> b3 = (group_hi & 0xFF00) << 8;
                    b_vnni_u32.template select<16, 1>((col * 2 + 1) * 16) =
                        fp8_e4m3_pair_to_vnni(b2, b3);
                }
                simd<fp16, 256> b_tile =
                    b_vnni_u32.template bit_cast_view<fp16>().read();

                // Load input and DPAS
                if constexpr (M_TILES >= 4) {
                    payA32.set_x((uint32_t)k_sub);
                    #pragma unroll
                    for (int m = 0; m < M_TILES; m += 4) {
                        payA32.set_y((uint32_t)(m * 8));
                        simd<fp16, 512> a4 = xesimd::lsc_load_2d<fp16, 16, 32, 1,
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
                        simd<fp16, 256> a2 = xesimd::lsc_load_2d<fp16, 16, 16, 1,
                            false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA16);
                        simd<fp16, 128> a0 = a2.template select<128, 1>(0);
                        simd<fp16, 128> a1 = a2.template select<128, 1>(128);
                        acc[m]   = dpas<8, 8, float, float, fp16, fp16>(acc[m],   b_tile, a0);
                        acc[m+1] = dpas<8, 8, float, float, fp16, fp16>(acc[m+1], b_tile, a1);
                    }
                } else {
                    payA.set_x((uint32_t)k_sub);
                    payA.set_y(0u);
                    simd<fp16, 128> a = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                    acc[0] = dpas<8, 8, float, float, fp16, fp16>(acc[0], b_tile, a);
                }
            }
        }

        // Reduction and output — same as V9
        if constexpr (K_THREADS == 1) {
            float s = *scale_ptr;
            #pragma unroll
            for (int m = 0; m < M_TILES; m++) {
                simd<float, 128> scaled = acc[m] * s;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mi = 0; mi < 8; mi++) {
                    int row = m * 8 + mi;
                    if (row < M) {
                        simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                        simd<fp16, 16> out_row = convert<fp16>(row_f);
                        if (full_n)
                            block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                        else
                            for (int ni = 0; ni < n_valid; ni++)
                                output[(size_t)row * N + n_start + ni] = out_row[ni];
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
                float s = *scale_ptr;
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    simd<float, 128> sum = slm_block_load<float, 128>(m * 128 * 4);
                    #pragma unroll
                    for (int t = 1; t < K_THREADS; t++) {
                        simd<float, 128> partial = slm_block_load<float, 128>(
                            t * SLM_PER_THREAD + m * 128 * 4);
                        sum += partial;
                    }
                    simd<float, 128> scaled = sum * s;
                    int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                    bool full_n = (n_valid == 16);
                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = m * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        }
    }
};

// Host dispatcher for V13
template<int K_THREADS, int M_TILES>
inline void dpas_v13_gemm_fp8_pert_host(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int num_wg = ((int)N + 15) / 16;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * K_THREADS)}, {(size_t)K_THREADS}),
            FP8_GEMM_DPAS_V13<K_THREADS, M_TILES>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K});
    });
}

// V13 auto-dispatch (same logic as V9)
inline void dpas_v13_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;
    int n_wgs = ((int)N + 15) / 16;
    int k_threads = std::max(1, std::min(4, 640 / std::max(n_wgs, 1)));
    while (k_threads > 1 && (K % (k_threads * 64) != 0)) k_threads--;
    if (k_threads == 3) k_threads = 2;

    #define V13_DISPATCH(KT, MT) dpas_v13_gemm_fp8_pert_host<KT, MT>(input, weight, scale_ptr, output, M, N, K, q)
    if (k_threads >= 4) {
        if      (m_tiles <= 1) V13_DISPATCH(4, 1);
        else if (m_tiles <= 2) V13_DISPATCH(4, 2);
        else if (m_tiles <= 4) V13_DISPATCH(4, 4);
        else                   V13_DISPATCH(4, 8);
    } else if (k_threads >= 2) {
        if      (m_tiles <= 1) V13_DISPATCH(2, 1);
        else if (m_tiles <= 2) V13_DISPATCH(2, 2);
        else if (m_tiles <= 4) V13_DISPATCH(2, 4);
        else                   V13_DISPATCH(2, 8);
    } else {
        if      (m_tiles <= 1) V13_DISPATCH(1, 1);
        else if (m_tiles <= 2) V13_DISPATCH(1, 2);
        else if (m_tiles <= 4) V13_DISPATCH(1, 4);
        else                   V13_DISPATCH(1, 8);
    }
    #undef V13_DISPATCH
}

// ============================================================================
// V10: V9 + N_TILES — process multiple N-columns per thread to reduce input loads
// VTune showed V9 has 45% more Send instructions than oneDNN at M=32.
// Root cause: V9 processes only 16 N-cols/WG → 160 WGs each redundantly loading input.
// V10: N_TILES=2 → 32 N-cols/WG → 80 WGs → halves redundant input loads.
// ============================================================================
template<int K_THREADS, int M_TILES, int N_TILES>
struct FP8_GEMM_DPAS_V10 {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_LOAD = 64;
        constexpr int K_SUB = 16;
        constexpr int N_PER_WG = N_TILES * 16;
        constexpr int ACC_COUNT = M_TILES * N_TILES;
        constexpr int SLM_PER_THREAD = ACC_COUNT * 128 * 4;
        constexpr int SLM_TOTAL = K_THREADS * SLM_PER_THREAD;
        slm_init<SLM_TOTAL>();

        int wg_id = item.get_group(0);
        int tid   = item.get_local_id(0);

        int n_base = wg_id * N_PER_WG;
        if (n_base >= N) return;

        int k_per_thread = K / K_THREADS;
        int k_start = tid * k_per_thread;
        int k_end   = k_start + k_per_thread;

        // Accumulators: [n_tile][m_tile]
        simd<float, 128> acc[N_TILES][M_TILES];
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++)
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++)
                acc[nt][mt] = 0.0f;

        // 2D payload for input A: [M, K] fp16
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        xesimd::config_2d_mem_access<fp16, 16, 8, 1> payA(
            input, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // 2D payloads for weight B — one per N-tile
        const uint32_t surfW_B = (uint32_t)K - 1u;
        const uint32_t surfH_B = (uint32_t)N - 1u;

        xesimd::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t[N_TILES];
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            payB_t[nt] = xesimd::config_2d_mem_access<uint32_t, 4, 16, 1>(
                reinterpret_cast<const uint32_t*>(weight),
                surfW_B, surfH_B, surfW_B,
                0u, (uint32_t)(n_base + nt * 16));
        }

        for (int k_base = k_start; k_base < k_end; k_base += K_LOAD) {
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int k_sub = k_base + sub * K_SUB;

                // Input-stationary: load input once per M-tile, iterate N-tiles
                // First load all input tiles for this K-sub
                simd<fp16, 128> a_tiles[M_TILES];
                payA.set_x((uint32_t)k_sub);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; mt++) {
                    payA.set_y((uint32_t)(mt * 8));
                    a_tiles[mt] = xesimd::lsc_load_2d<fp16, 16, 8, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
                }

                // Process each N-tile: load weight, DPAS across all M-tiles
                #pragma unroll
                for (int nt = 0; nt < N_TILES; nt++) {
                    payB_t[nt].set_x((uint32_t)(k_sub / 4));
                    simd<uint32_t, 64> w_t = xesimd::lsc_load_2d<uint32_t, 4, 16, 1,
                        true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB_t[nt]);

                    simd<uint32_t, 128> b_vnni_u32;
                    #pragma unroll
                    for (int col = 0; col < 4; col++) {
                        simd<uint32_t, 16> group = w_t.template select<16, 1>(col * 16);
                        simd<uint32_t, 16> b0 = group & 0xFF;
                        simd<uint32_t, 16> b1 = (group & 0xFF00) << 8;
                        b_vnni_u32.template select<16, 1>(col * 2 * 16) =
                            fp8_e4m3_pair_to_vnni(b0, b1);
                        simd<uint32_t, 16> group_hi = group >> 16;
                        simd<uint32_t, 16> b2 = group_hi & 0xFF;
                        simd<uint32_t, 16> b3 = (group_hi & 0xFF00) << 8;
                        b_vnni_u32.template select<16, 1>((col * 2 + 1) * 16) =
                            fp8_e4m3_pair_to_vnni(b2, b3);
                    }
                    simd<fp16, 256> b_tile =
                        b_vnni_u32.template bit_cast_view<fp16>().read();

                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; mt++) {
                        acc[nt][mt] = dpas<8, 8, float, float, fp16, fp16>(
                            acc[nt][mt], b_tile, a_tiles[mt]);
                    }
                }
            }
        }

        if constexpr (K_THREADS == 1) {
            float s = *scale_ptr;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++) {
                int n_start = n_base + nt * 16;
                int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                if (n_valid <= 0) continue;
                bool full_n = (n_valid == 16);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; mt++) {
                    simd<float, 128> scaled = acc[nt][mt] * s;
                    #pragma unroll
                    for (int mi = 0; mi < 8; mi++) {
                        int row = mt * 8 + mi;
                        if (row < M) {
                            simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                            simd<fp16, 16> out_row = convert<fp16>(row_f);
                            if (full_n)
                                block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                            else
                                for (int ni = 0; ni < n_valid; ni++)
                                    output[(size_t)row * N + n_start + ni] = out_row[ni];
                        }
                    }
                }
            }
        } else {
            // SLM reduction
            uint32_t slm_base = tid * SLM_PER_THREAD;
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++)
                #pragma unroll
                for (int mt = 0; mt < M_TILES; mt++)
                    slm_block_store<float, 128>(
                        slm_base + (nt * M_TILES + mt) * 128 * 4, acc[nt][mt]);

            barrier();

            if (tid == 0) {
                float s = *scale_ptr;
                #pragma unroll
                for (int nt = 0; nt < N_TILES; nt++) {
                    int n_start = n_base + nt * 16;
                    int n_valid = (n_start + 16 <= N) ? 16 : (N - n_start);
                    if (n_valid <= 0) continue;
                    bool full_n = (n_valid == 16);
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; mt++) {
                        int acc_idx = nt * M_TILES + mt;
                        simd<float, 128> sum = slm_block_load<float, 128>(
                            acc_idx * 128 * 4);
                        #pragma unroll
                        for (int t = 1; t < K_THREADS; t++) {
                            simd<float, 128> partial = slm_block_load<float, 128>(
                                t * SLM_PER_THREAD + acc_idx * 128 * 4);
                            sum += partial;
                        }
                        simd<float, 128> scaled = sum * s;
                        #pragma unroll
                        for (int mi = 0; mi < 8; mi++) {
                            int row = mt * 8 + mi;
                            if (row < M) {
                                simd<float, 16> row_f = scaled.select<16, 1>(mi * 16);
                                simd<fp16, 16> out_row = convert<fp16>(row_f);
                                if (full_n)
                                    block_store<fp16, 16>(output + (size_t)row * N + n_start, out_row);
                                else
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

// Host dispatcher for DPAS V10
template<int K_THREADS, int M_TILES, int N_TILES>
inline void dpas_v10_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {

    constexpr int N_PER_WG = N_TILES * 16;
    int num_wg = ((int)N + N_PER_WG - 1) / N_PER_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * K_THREADS)}, {(size_t)K_THREADS}),
            FP8_GEMM_DPAS_V10<K_THREADS, M_TILES, N_TILES>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K});
    });
}

// V10 auto-dispatch: choose K_THREADS, M_TILES, N_TILES
inline void dpas_v10_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;

    // Choose N_TILES based on M: higher M benefits from more N-reuse
    int n_tiles = (m_tiles >= 4) ? 4 : (m_tiles >= 2) ? 2 : 1;
    // Cap N_TILES so we don't end up with too few WGs
    int n_per_wg = n_tiles * 16;
    int n_wgs = ((int)N + n_per_wg - 1) / n_per_wg;
    // Ensure enough WGs for occupancy
    while (n_tiles > 1 && n_wgs < 20) {
        n_tiles /= 2;
        n_per_wg = n_tiles * 16;
        n_wgs = ((int)N + n_per_wg - 1) / n_per_wg;
    }

    int k_threads = std::max(1, std::min(4, 640 / std::max(n_wgs, 1)));
    while (k_threads > 1 && (K % (k_threads * 64) != 0)) k_threads--;
    if (k_threads == 3) k_threads = 2;

    #define V10_DISPATCH(KT, MT, NT) dpas_v10_gemm_fp8_pert_host<KT, MT, NT>(input, weight, scale_ptr, output, M, N, K, q)

    // Dispatch based on k_threads, m_tiles, n_tiles
    if (n_tiles == 4) {
        if (k_threads >= 4) {
            if      (m_tiles <= 4) V10_DISPATCH(4, 4, 4);
            else                   V10_DISPATCH(4, 8, 4);
        } else if (k_threads >= 2) {
            if      (m_tiles <= 4) V10_DISPATCH(2, 4, 4);
            else                   V10_DISPATCH(2, 8, 4);
        } else {
            if      (m_tiles <= 4) V10_DISPATCH(1, 4, 4);
            else                   V10_DISPATCH(1, 8, 4);
        }
    } else if (n_tiles == 2) {
        if (k_threads >= 4) {
            if      (m_tiles <= 1) V10_DISPATCH(4, 1, 2);
            else if (m_tiles <= 2) V10_DISPATCH(4, 2, 2);
            else if (m_tiles <= 4) V10_DISPATCH(4, 4, 2);
            else                   V10_DISPATCH(4, 8, 2);
        } else if (k_threads >= 2) {
            if      (m_tiles <= 1) V10_DISPATCH(2, 1, 2);
            else if (m_tiles <= 2) V10_DISPATCH(2, 2, 2);
            else if (m_tiles <= 4) V10_DISPATCH(2, 4, 2);
            else                   V10_DISPATCH(2, 8, 2);
        } else {
            if      (m_tiles <= 1) V10_DISPATCH(1, 1, 2);
            else if (m_tiles <= 2) V10_DISPATCH(1, 2, 2);
            else if (m_tiles <= 4) V10_DISPATCH(1, 4, 2);
            else                   V10_DISPATCH(1, 8, 2);
        }
    } else {
        // N_TILES=1: same as V9
        if (k_threads >= 4) {
            if      (m_tiles <= 1) V10_DISPATCH(4, 1, 1);
            else if (m_tiles <= 2) V10_DISPATCH(4, 2, 1);
            else if (m_tiles <= 4) V10_DISPATCH(4, 4, 1);
            else                   V10_DISPATCH(4, 8, 1);
        } else if (k_threads >= 2) {
            if      (m_tiles <= 1) V10_DISPATCH(2, 1, 1);
            else if (m_tiles <= 2) V10_DISPATCH(2, 2, 1);
            else if (m_tiles <= 4) V10_DISPATCH(2, 4, 1);
            else                   V10_DISPATCH(2, 8, 1);
        } else {
            if      (m_tiles <= 1) V10_DISPATCH(1, 1, 1);
            else if (m_tiles <= 2) V10_DISPATCH(1, 2, 1);
            else if (m_tiles <= 4) V10_DISPATCH(1, 4, 1);
            else                   V10_DISPATCH(1, 8, 1);
        }
    }
    #undef V10_DISPATCH
}
#endif // GEMM_EXPERIMENTAL — V13, V10

// Host dispatcher for DPAS V9
template<int K_THREADS, int M_TILES>
inline void dpas_v9_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {

    int num_wg = ((int)N + 15) / 16;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * K_THREADS)}, {(size_t)K_THREADS}),
            FP8_GEMM_DPAS_V9<K_THREADS, M_TILES>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K});
    });
}

// V9 auto-dispatch: choose K_THREADS and M_TILES
inline void dpas_v9_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;
    int n_wgs = ((int)N + 15) / 16;
    int k_threads = std::max(1, std::min(4, 640 / std::max(n_wgs, 1)));
    while (k_threads > 1 && (K % (k_threads * 64) != 0)) k_threads--;
    if (k_threads == 3) k_threads = 2;

    #define V9_DISPATCH(KT, MT) dpas_v9_gemm_fp8_pert_host<KT, MT>(input, weight, scale_ptr, output, M, N, K, q)
    if (k_threads >= 4) {
        if      (m_tiles <= 1) V9_DISPATCH(4, 1);
        else if (m_tiles <= 2) V9_DISPATCH(4, 2);
        else if (m_tiles <= 4) V9_DISPATCH(4, 4);
        else                   V9_DISPATCH(4, 8);
    } else if (k_threads >= 2) {
        if      (m_tiles <= 1) V9_DISPATCH(2, 1);
        else if (m_tiles <= 2) V9_DISPATCH(2, 2);
        else if (m_tiles <= 4) V9_DISPATCH(2, 4);
        else                   V9_DISPATCH(2, 8);
    } else {
        if      (m_tiles <= 1) V9_DISPATCH(1, 1);
        else if (m_tiles <= 2) V9_DISPATCH(1, 2);
        else if (m_tiles <= 4) V9_DISPATCH(1, 4);
        else                   V9_DISPATCH(1, 8);
    }
    #undef V9_DISPATCH
}

#ifdef GEMM_EXPERIMENTAL  // Dead code: V5, V2 hosts, old dpas_gemm, old unified dispatcher, v5_auto_dispatch
// Host dispatcher for DPAS V5
template<int WG_SIZE, int M_TILES>
inline void dpas_v5_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    constexpr int N_WG = WG_SIZE * 16;
    int num_wg = ((int)N + N_WG - 1) / N_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V5<WG_SIZE, M_TILES>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}

// Host dispatcher for DPAS V2
template<int WG_SIZE, int M_TILES>
inline void dpas_v2_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    constexpr int N_WG = WG_SIZE * 16;
    int num_wg = ((int)N + N_WG - 1) / N_WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)(num_wg * WG_SIZE)}, {(size_t)WG_SIZE}),
            FP8_GEMM_DPAS_V2<WG_SIZE, M_TILES>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}
#endif // GEMM_EXPERIMENTAL — V5, V2 host dispatchers

// ============================================================================
// Host Dispatchers
// ============================================================================

// Regime A dispatcher: M=1~4, batched GEMV with K-split
inline void batched_gemv_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    int vl, ks;
    select_vl_ks(N, K, vl, ks);

    int global0 = N * ks;
    int global1 = M;
    int local0  = ks;
    int local1  = 1;

    #define LAUNCH_BATCHED(V, S) \
        q.submit([&](sycl::handler& h) { \
            h.parallel_for( \
                sycl::nd_range<2>({(size_t)global0, (size_t)global1}, \
                                  {(size_t)local0, (size_t)local1}), \
                GEMV_fp8_pert_batched_kernel<V, S>{ \
                    input, weight, scale_ptr, output, \
                    (int)M, (int)N, (int)K, fp8_mode}); \
        });

    if      (vl == 512 && ks == 1) { LAUNCH_BATCHED(512, 1) }
    else if (vl == 512 && ks == 2) { LAUNCH_BATCHED(512, 2) }
    else if (vl == 256 && ks == 1) { LAUNCH_BATCHED(256, 1) }
    else if (vl == 256 && ks == 2) { LAUNCH_BATCHED(256, 2) }
    else if (vl == 256 && ks == 4) { LAUNCH_BATCHED(256, 4) }
    else if (vl == 128 && ks == 1) { LAUNCH_BATCHED(128, 1) }
    else if (vl == 128 && ks == 2) { LAUNCH_BATCHED(128, 2) }
    else if (vl == 128 && ks == 4) { LAUNCH_BATCHED(128, 4) }
    else if (vl == 128 && ks == 8) { LAUNCH_BATCHED(128, 8) }
    else                           { LAUNCH_BATCHED(128, 1) }

    #undef LAUNCH_BATCHED
}

// Regime B/C dispatcher: weight-stationary GEMM
// VL and TILE_M are both template params. Key configs:
//   TM=8,  VL=128: 4KB accs — M=4~8, single M-tile
//   TM=16, VL=128: 8KB accs — M=9~16, single M-tile
//   TM=32, VL=64:  8KB accs — M=17~64, fewer M-tiles = fewer weight re-reads
template<int VL, int TILE_M>
inline void ws_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    int m_tiles = ((int)M + TILE_M - 1) / TILE_M;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>({(size_t)N, (size_t)m_tiles}, {1, 1}),
            GEMM_fp8_pert_ws_kernel<VL, TILE_M>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}

#ifdef GEMM_EXPERIMENTAL  // Dead code: old dpas_gemm, old unified dispatcher, v5_auto_dispatch
// Regime D dispatcher: DPAS-based GEMM
// TILE_N must be a multiple of 16. Requires K % 16 == 0.
template<int TILE_N>
inline void dpas_gemm_fp8_pert_host(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    int n_tiles = ((int)N + TILE_N - 1) / TILE_N;
    int m_tiles = ((int)M + 7) / 8;  // DPAS always does 8 M-rows

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>({(size_t)n_tiles, (size_t)m_tiles}, {1, 1}),
            GEMM_fp8_pert_dpas_kernel<TILE_N>{
                input, weight, scale_ptr, output,
                (int)M, (int)N, (int)K, fp8_mode});
    });
}

// Unified dispatcher: selects regime based on M
inline void GEMM_fp8_pert_host(
    uint8_t* input_data,   // fp16[M, K]
    uint8_t* weight_data,  // uint8[N, K]
    uint8_t* scale_data,   // float scalar (device ptr)
    uint8_t* output_data,  // fp16[M, N]
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    auto* p_in  = reinterpret_cast<const fp16*>(input_data);
    auto* p_w   = reinterpret_cast<const uint8_t*>(weight_data);
    auto* p_sc  = reinterpret_cast<const float*>(scale_data);
    auto* p_out = reinterpret_cast<fp16*>(output_data);

    if (M <= 3) {
        batched_gemv_fp8_pert_host(p_in, p_w, p_sc, p_out, M, N, K, fp8_mode, q);
    } else if (M <= 8) {
        ws_gemm_fp8_pert_host<128, 8>(p_in, p_w, p_sc, p_out, M, N, K, fp8_mode, q);
    } else {
        // WS-16/128 wins across all M>8 values in crossover analysis.
        // DPAS available via dpas_gemm_fp8_pert_host<16>() but FP8 dequant+VNNI
        // packing overhead makes it slower than WS for these shape sizes.
        ws_gemm_fp8_pert_host<128, 16>(p_in, p_w, p_sc, p_out, M, N, K, fp8_mode, q);
    }
}

// V5 M_TILES dispatcher helper
inline void dpas_v5_auto_dispatch(
    const fp16* input, const uint8_t* weight, const float* scale_ptr,
    fp16* output, uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode, sycl::queue& q) {
    int m_tiles = ((int)M + 7) / 8;
    if      (m_tiles <= 1) dpas_v5_gemm_fp8_pert_host<8, 1>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    else if (m_tiles <= 2) dpas_v5_gemm_fp8_pert_host<8, 2>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    else if (m_tiles <= 4) dpas_v5_gemm_fp8_pert_host<8, 4>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    else                   dpas_v5_gemm_fp8_pert_host<8, 8>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
}
#endif // GEMM_EXPERIMENTAL — V5, V2, old dispatchers

// Direct-typed unified dispatcher (no uint8_t casts needed)
inline void GEMM_fp8_pert_dispatch(
    const fp16*    input,
    const uint8_t* weight,
    const float*   scale_ptr,
    fp16*          output,
    uint32_t M, uint32_t N, uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    if (M == 1) {
        batched_gemv_fp8_pert_host(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    } else if (N <= 32 && K % 64 == 0 && K >= 2048 && M <= 16) {
        // Tiny-N projection (e.g. Qwen3-Next ba: N=16, K=2048).
        // V7 KT=4 MT=2 is far better than auto for this regime:
        // M=4..10: 48us vs auto 109us. See DECODE_BSZ12_KERNEL_INVESTIGATION.md.
        dpas_v7_gemm_fp8_pert_host<4, 2>(
            input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    } else if (K % 64 == 0 && fp8_mode == 0) {
        // V9: Transposed load + fused dequant-VNNI (E4M3 only, best for M>=2)
        dpas_v9_auto_dispatch(input, weight, scale_ptr, output, M, N, K, q);
    } else if (K % 64 == 0 && N >= 1024) {
        // V7: K-split multi-thread WG for E5M2 or when V9 not applicable
        dpas_v7_auto_dispatch(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    } else if (M <= 3) {
        batched_gemv_fp8_pert_host(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    } else if (M <= 8) {
        ws_gemm_fp8_pert_host<128, 8>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    } else {
        ws_gemm_fp8_pert_host<128, 16>(input, weight, scale_ptr, output, M, N, K, fp8_mode, q);
    }
}
