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

inline void resadd_norm_gemv2_fp8_pert_host(
    const fp16* hidden_ptr, const fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* o0,
    const uint8_t* w1, const float* s1, fp16* o1,
    int N0, int N1, int K, float eps, int fp8_mode,
    sycl::queue& q)
{
    int total_N = N0 + N1;
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(total_N, 1),
            ResAddNormGEMV2_fp8_pert_kernel{
                hidden_ptr, residual_ptr, norm_w_ptr,
                w0, s0, o0, w1, s1, o1,
                N0, N1, K, eps, fp8_mode});
    });
}
