/* norm_gemv_fused.h — Fused RMSNormGated + FP8 GEMV for GDN out_proj.
 *
 * Combines two operations into a single kernel submit:
 *   1. RMSNormGated: y = rmsnorm(x) * weight * silu(z), per-head (V dims each)
 *   2. GEMV: output = y_flat @ dequant(gemv_weight^T) * scale
 *
 * Designed for GDN decode path where:
 *   x (core_attn_out): [HV, V] fp16   (e.g. [8, 128])
 *   z (z_out):          [HV, V] fp16
 *   norm_weight:         [V] fp16       (shared across heads)
 *   gemv_weight:         [N, K] FP8     (K = HV*V, e.g. [2048, 1024])
 *   gemv_scale:          [1] float32
 *   output:              [N] fp16
 *
 * Each work-group computes one output element:
 *   1. Load x, z for current K-chunk (one head = V=128 elements)
 *   2. Compute RMSNorm + SiLU gate → normed[V]
 *   3. Load weight row chunk, FMA into accumulator
 *   4. Repeat for all HV heads
 *   5. Reduce accumulator → output
 *
 * The norm inputs (x, z, norm_w) are read by all WGs but stay in L3
 * cache after the first WG reads them (total 2*HV*V*2 + V*2 ≈ 4.25KB).
 *
 * Saves: 1 kernel launch + Python overhead of norm (torch.empty, reshape,
 * Triton dispatch) per GDN layer. 48 layers × ~50us = ~2.4ms.
 */

#pragma once
#include "utils.h"

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_norm(
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

/* Hierarchical reduction for simd<float, 128> → scalar */
ESIMD_INLINE float reduce128(simd<float, 128> v) {
    v.select<64,1>(0) += v.select<64,1>(64);
    v.select<32,1>(0) += v.select<32,1>(32);
    v.select<16,1>(0) += v.select<16,1>(16);
    v.select<8,1>(0)  += v.select<8,1>(8);
    v.select<4,1>(0)  += v.select<4,1>(4);
    v.select<2,1>(0)  += v.select<2,1>(2);
    return v[0] + v[1];
}

/* ================================================================
 * Kernel: Fused RMSNormGated + FP8 GEMV (per-tensor scale)
 * Grid: N work-groups, 1 thread each (same as GEMV_fp8_pert_kernel)
 * ================================================================ */
struct NormGEMV_fp8_pert_kernel {
    const fp16*    x_ptr;        // [HV, V] core_attn_out
    const fp16*    z_ptr;        // [HV, V] z_out
    const fp16*    norm_w_ptr;   // [V] norm weight
    const uint8_t* gemv_weight;  // [N, K] FP8, K = HV * V
    const float*   gemv_scale;   // [1] per-tensor scale
    fp16*          output;       // [N]
    int N;
    int HV;      // number of value heads (per TP)
    int V;       // head_v_dim (128)
    float eps;
    int fp8_mode; // 0=E4M3, 1=E5M2

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        const int K = HV * V;

        // Pre-load norm weight [V=128] — shared across all heads, reused HV times
        simd<float, 128> norm_w = block_load<fp16, 128>(norm_w_ptr);

        simd<float, 128> acc = 0.0f;

        for (int h = 0; h < HV; h++) {
            const int offset = h * V;

            // Load x and z for this head
            simd<float, 128> x_f = block_load<fp16, 128>(x_ptr + offset);
            simd<float, 128> z_f = block_load<fp16, 128>(z_ptr + offset);

            // RMSNorm: rms = sqrt(mean(x^2) + eps)
            simd<float, 128> x_sq = x_f * x_f;
            float mean_sq = reduce128(x_sq) * (1.0f / V);
            float inv_rms = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(mean_sq + eps))[0];

            // Normalize: normed = x * inv_rms * weight
            simd<float, 128> normed = x_f * inv_rms * norm_w;

            // Gate: silu(z) = z * sigmoid(z)
            simd<float, 128> neg_z = -z_f;
            simd<float, 128> exp_neg_z = sycl::ext::intel::esimd::exp(neg_z);
            simd<float, 128> silu_z = z_f / (1.0f + exp_neg_z);
            normed *= silu_z;

            // GEMV: accumulate dot product with weight row
            simd<uint8_t, 128> w_raw = block_load<uint8_t, 128>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, 128> w_f = fp8_dequant_norm<128>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        // Final reduction and scale
        float dot = reduce128(acc) * *gemv_scale;
        output[n] = fp16(dot);
    }
};

/* Host dispatcher */
inline void norm_gemv_fp8_pert_host(
    const fp16* x_ptr,
    const fp16* z_ptr,
    const fp16* norm_w_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    int N, int HV, int V,
    float eps,
    int fp8_mode,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(N, 1),
            NormGEMV_fp8_pert_kernel{
                x_ptr, z_ptr, norm_w_ptr,
                gemv_weight, gemv_scale, output,
                N, HV, V, eps, fp8_mode});
    });
}
