/* resadd_norm_gemv2_fused.h — Fused ResidualAdd + RMSNorm + 2-matrix FP8 GEMV.
 *
 * Single-pass: load hidden+residual, compute add, accumulate sum_sq,
 * normalize, GEMV — all without storing intermediate arrays.
 *
 * Key insight: we need sum_sq (from pass 1) before normalizing (pass 2).
 * But storing all chunks needs too many registers for large K.
 * Solution: TWO loops over global memory. Pass 1 reads h+r for sum_sq only.
 * Pass 2 re-reads h+r (from L3), normalizes, does GEMV.
 * Residual write-back is done by Python caller (residual.add_(hidden)).
 *
 * Grid: (N0 + N1) WGs, 1 thread each.
 */

#pragma once
#include "utils.h"

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_rng2(
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

struct ResAddNormGEMV2_fp8_pert_kernel {
    const fp16*    hidden_ptr;   // [1, K] — read-only
    const fp16*    residual_ptr; // [1, K] — read-only (caller updates separately)
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* w0_ptr;       // [N0, K] FP8
    const float*   s0_ptr;       // [1]
    fp16*          o0_ptr;       // [1, N0]
    const uint8_t* w1_ptr;       // [N1, K] FP8
    const float*   s1_ptr;       // [1]
    fp16*          o1_ptr;       // [1, N1]
    fp16*          new_residual_ptr;  // [1, K] — written by gid==0 (= hidden+residual); may be null
    int N0, N1, K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int gid = item.get_group(0);
        int total_N = N0 + N1;
        if (gid >= total_N) return;

        int mat_idx = (gid < N0) ? 0 : 1;
        int local_n = (mat_idx == 0) ? gid : gid - N0;

        constexpr int VL = 512;
        int n_chunks = K / VL;

        // gid 0 additionally writes the post-add residual (hidden+residual) so
        // the caller doesn't need a separate aten::add dispatch. fp16 add to
        // match the reference (h.half + r.half). Only one group writes → no race.
        if (gid == 0 && new_residual_ptr != nullptr) {
            for (int c = 0; c < n_chunks; c++) {
                int offset = c * VL;
                simd<fp16, VL> hh = block_load<fp16, VL>(hidden_ptr + offset);
                simd<fp16, VL> rr = block_load<fp16, VL>(residual_ptr + offset);
                block_store<fp16, VL>(new_residual_ptr + offset, hh + rr);
            }
        }

        // Pass 1: compute sum_sq for RMS
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;

            simd<float, VL> sq = added * added;
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

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: re-load h+r (L3 cache hit), normalize, GEMV
        const uint8_t* w_ptr = (mat_idx == 0) ? w0_ptr : w1_ptr;
        const float* s_ptr = (mat_idx == 0) ? s0_ptr : s1_ptr;
        fp16* o_ptr = (mat_idx == 0) ? o0_ptr : o1_ptr;

        simd<float, VL> acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = (h + r) * inv_rms * nw;

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                w_ptr + (size_t)local_n * K + offset);
            simd<float, VL> w_f = fp8_dequant_rng2<VL>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        acc.select<256,1>(0) += acc.select<256,1>(256);
        acc.select<128,1>(0) += acc.select<128,1>(128);
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0) += acc.select<8,1>(8);
        acc.select<4,1>(0) += acc.select<4,1>(4);
        acc.select<2,1>(0) += acc.select<2,1>(2);
        float dot = ((float)acc[0] + (float)acc[1]) * *s_ptr;
        o_ptr[local_n] = fp16(dot);
    }
};


/* ================================================================
 * M-tiled variant: ONE work-group per output column, MT rows at a time.
 *
 * The earlier attempt at supporting M>1 here simply ran the M=1 kernel M
 * times (grid * M), so every row re-did the fp8 dequant -- the dominant ALU
 * cost -- and the kernel scaled linearly in M. It lost to the unfused path,
 * which reaches esimd_gemm_fp8_pert's DPAS kernel (nearly flat in M).
 *
 * Here each work-group keeps MT accumulators and dequantises each weight
 * chunk EXACTLY ONCE per row-block, reusing it for all MT rows. Rows beyond
 * MT are handled by an outer row-block loop inside the same work-group, so a
 * single instantiation covers any M while the weight/dequant cost grows only
 * as ceil(M / MT) instead of M.
 *
 * MT and VL are chosen together so that the MT accumulators fit in the GRF:
 * MT*VL*4 bytes must stay well under 8 KB (128 registers). MT=4/VL=128 uses
 * 32 registers, MT=8/VL=64 uses 32 registers.
 * ================================================================ */

// Horizontal sum of a power-of-two-wide simd, any width.
template <int N>
SYCL_ESIMD_FUNCTION inline float rng2_hreduce(simd<float, N> v) {
    if constexpr (N == 1) {
        return (float)v[0];
    } else {
        simd<float, N / 2> h =
            v.template select<N / 2, 1>(0) + v.template select<N / 2, 1>(N / 2);
        return rng2_hreduce<N / 2>(h);
    }
}

template <int MT, int VL>
struct ResAddNormGEMM2_fp8_pert_Mtile_kernel {
    const fp16*    hidden_ptr;   // [M, K]
    const fp16*    residual_ptr; // [M, K]
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* w0_ptr;       // [N0, K] FP8
    const float*   s0_ptr;
    fp16*          o0_ptr;       // [M, N0]
    const uint8_t* w1_ptr;       // [N1, K] FP8
    const float*   s1_ptr;
    fp16*          o1_ptr;       // [M, N1]
    fp16*          new_residual_ptr;  // [M, K] or null
    int M, N0, N1, K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int gid = item.get_group(0);
        const int total_N = N0 + N1;
        if (gid >= total_N) return;

        const int mat_idx = (gid < N0) ? 0 : 1;
        const int local_n = (mat_idx == 0) ? gid : gid - N0;
        const int n_chunks = K / VL;

        const uint8_t* w_ptr = (mat_idx == 0) ? w0_ptr : w1_ptr;
        const float*   s_ptr = (mat_idx == 0) ? s0_ptr : s1_ptr;
        fp16*          o_ptr = (mat_idx == 0) ? o0_ptr : o1_ptr;
        const int      out_n = (mat_idx == 0) ? N0 : N1;
        const float    scale = *s_ptr;

        // Group 0 additionally writes the post-add residual, saving the caller
        // a separate aten::add dispatch. Only one group writes -> no race.
        if (gid == 0 && new_residual_ptr != nullptr) {
            for (int m = 0; m < M; ++m) {
                const size_t base = (size_t)m * K;
                for (int c = 0; c < n_chunks; c++) {
                    const int off = c * VL;
                    simd<fp16, VL> hh = block_load<fp16, VL>(hidden_ptr + base + off);
                    simd<fp16, VL> rr = block_load<fp16, VL>(residual_ptr + base + off);
                    block_store<fp16, VL>(new_residual_ptr + base + off, hh + rr);
                }
            }
        }

        for (int m0 = 0; m0 < M; m0 += MT) {
            // Pass 1: per-row inverse RMS. Rows past the end are clamped to the
            // last valid row so the loads stay in bounds; their results are
            // simply never stored.
            float inv_rms[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int mr = (m0 + m < M) ? (m0 + m) : (M - 1);
                const size_t base = (size_t)mr * K;
                float sum_sq = 0.0f;
                for (int c = 0; c < n_chunks; c++) {
                    const int off = c * VL;
                    simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + base + off);
                    simd<float, VL> r = block_load<fp16, VL>(residual_ptr + base + off);
                    simd<float, VL> added = h + r;
                    sum_sq += rng2_hreduce<VL>(added * added);
                }
                inv_rms[m] = sycl::ext::intel::esimd::rsqrt(
                    simd<float, 8>(sum_sq / (float)K + eps))[0];
            }

            // Pass 2: dequantise each weight chunk once, FMA into all MT rows.
            simd<float, VL> acc[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

            for (int c = 0; c < n_chunks; c++) {
                const int off = c * VL;
                simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                    w_ptr + (size_t)local_n * K + off);
                simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + off);
                simd<float, VL> wnw = fp8_dequant_rng2<VL>(w_raw, fp8_mode) * nw;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const int mr = (m0 + m < M) ? (m0 + m) : (M - 1);
                    const size_t base = (size_t)mr * K;
                    simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + base + off);
                    simd<float, VL> r = block_load<fp16, VL>(residual_ptr + base + off);
                    acc[m] += (h + r) * inv_rms[m] * wnw;
                }
            }

#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int mr = m0 + m;
                if (mr < M) {
                    o_ptr[(size_t)mr * out_n + local_n] =
                        fp16(rng2_hreduce<VL>(acc[m]) * scale);
                }
            }
        }
    }
};

// Returns false if the shape cannot be served by a compiled tile config.
inline bool resadd_norm_gemm2_fp8_pert_mtile_host(
    const fp16* hidden_ptr, const fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* o0,
    const uint8_t* w1, const float* s1, fp16* o1,
    fp16* new_residual_ptr,
    int M, int N0, int N1, int K, float eps, int fp8_mode,
    sycl::queue& q)
{
    if (M < 2) return false;
    const int total_N = N0 + N1;
#define RNG2_MTILE_LAUNCH(MT_, VL_)                                            \
    q.submit([&](sycl::handler& cgh) {                                         \
        cgh.parallel_for(sycl::nd_range<1>(total_N, 1),                        \
            ResAddNormGEMM2_fp8_pert_Mtile_kernel<MT_, VL_>{                   \
                hidden_ptr, residual_ptr, norm_w_ptr,                          \
                w0, s0, o0, w1, s1, o1, new_residual_ptr,                      \
                M, N0, N1, K, eps, fp8_mode});                                 \
    })
    if (M <= 2) {
        if (K % 128 != 0) return false;
        RNG2_MTILE_LAUNCH(2, 128);
    } else if (M <= 4) {
        if (K % 128 != 0) return false;
        RNG2_MTILE_LAUNCH(4, 128);
    } else {
        // Larger M (batched TARGET_VERIFY): narrower vectors keep the 8
        // accumulators at 32 registers, and the row-block loop covers any M.
        if (K % 64 != 0) return false;
        RNG2_MTILE_LAUNCH(8, 64);
    }
#undef RNG2_MTILE_LAUNCH
    return true;
}

inline void resadd_norm_gemv2_fp8_pert_host(
    const fp16* hidden_ptr, const fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* o0,
    const uint8_t* w1, const float* s1, fp16* o1,
    fp16* new_residual_ptr,
    int N0, int N1, int K, float eps, int fp8_mode,
    sycl::queue& q)
{
    int total_N = N0 + N1;
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(total_N, 1),
            ResAddNormGEMV2_fp8_pert_kernel{
                hidden_ptr, residual_ptr, norm_w_ptr,
                w0, s0, o0, w1, s1, o1, new_residual_ptr,
                N0, N1, K, eps, fp8_mode});
    });
}
