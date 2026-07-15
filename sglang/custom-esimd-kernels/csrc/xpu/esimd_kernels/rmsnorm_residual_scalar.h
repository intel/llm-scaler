/* rmsnorm_residual_scalar.h — Fused RMSNorm + residual add + scalar multiply.
 *
 * out = (rmsnorm(x) * (weight+1.0) + residual) * scalar
 *
 * Single WG, VL=256 (divides K=5376 evenly). Same style as FusedAddRmsNorm V1.
 */

#pragma once
#include "utils.h"

struct RmsNormResidualScalar_kernel {
    const fp16* x_ptr;
    const fp16* weight_ptr;
    const fp16* residual_ptr;
    fp16*       out_ptr;
    int K;
    float eps;
    float scalar;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 256;
        int n_chunks = K / VL;

        // Pass 1: accumulate sum_sq
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> x = block_load<fp16, VL>(x_ptr + offset);
            simd<float, VL> sq = x * x;
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

        // Pass 2: out = (x * inv_rms * (w+1.0) + residual) * scalar
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> x = block_load<fp16, VL>(x_ptr + offset);
            simd<float, VL> w = block_load<fp16, VL>(weight_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> normed = x * inv_rms * (w + 1.0f);
            simd<float, VL> result = (normed + r) * scalar;
            block_store<fp16, VL>(out_ptr + offset, simd<fp16, VL>(result));
        }
    }
};

inline void rmsnorm_residual_scalar_host(
    const fp16* x, const fp16* weight, const fp16* residual,
    fp16* out, int K, float eps, float scalar, sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        RmsNormResidualScalar_kernel kern{x, weight, residual, out, K, eps, scalar};
        h.parallel_for(sycl::nd_range<1>(1, 1), kern);
    });
}
