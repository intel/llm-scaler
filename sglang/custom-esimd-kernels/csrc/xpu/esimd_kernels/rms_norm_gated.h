/* rms_norm_gated.h — ESIMD RMSNormGated kernel for GDN output projection.
 *
 * Replaces 6+ PyTorch kernel dispatches (float cast, pow, mean, rsqrt,
 * sigmoid, mul) with a single ESIMD kernel. For [rows, V=128]:
 *   output[i] = rmsnorm(x[i]) * weight * silu(z[i])
 * where rmsnorm(x) = x / rms(x), silu(z) = z * sigmoid(z).
 *
 * Grid: rows WGs, 1 thread each. V=128 fits in registers.
 * Expected: ~5us vs ~87us for PyTorch dispatch chain.
 */

#pragma once
#include "utils.h"

struct RmsNormGated_kernel {
    const fp16* x_ptr;      // [rows, V]
    const fp16* z_ptr;      // [rows, V]
    const fp16* weight_ptr;  // [V]
    fp16*       output_ptr;  // [rows, V]
    int rows;
    int V;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int row = item.get_group(0);
        if (row >= rows) return;

        const int offset = row * V;

        // Load x, z, weight (V=128, fits in one block_load)
        simd<float, 128> x_f = block_load<fp16, 128>(x_ptr + offset);
        simd<float, 128> z_f = block_load<fp16, 128>(z_ptr + offset);
        simd<float, 128> w_f = block_load<fp16, 128>(weight_ptr);

        // RMSNorm: inv_rms = rsqrt(mean(x^2) + eps)
        simd<float, 128> x_sq = x_f * x_f;
        // Tree reduction for sum
        x_sq.select<64,1>(0) += x_sq.select<64,1>(64);
        x_sq.select<32,1>(0) += x_sq.select<32,1>(32);
        x_sq.select<16,1>(0) += x_sq.select<16,1>(16);
        x_sq.select<8,1>(0)  += x_sq.select<8,1>(8);
        x_sq.select<4,1>(0)  += x_sq.select<4,1>(4);
        x_sq.select<2,1>(0)  += x_sq.select<2,1>(2);
        float sum_sq = x_sq[0] + x_sq[1];
        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)V + eps))[0];

        // Normalize: normed = x * inv_rms * weight
        simd<float, 128> normed = x_f * inv_rms * w_f;

        // Gate: silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
        simd<float, 128> exp_neg_z = sycl::ext::intel::esimd::exp(-z_f);
        simd<float, 128> silu_z = z_f / (1.0f + exp_neg_z);

        // Output
        simd<float, 128> result = normed * silu_z;
        block_store<fp16, 128>(output_ptr + offset, simd<fp16, 128>(result));
    }
};

inline void rms_norm_gated_host(
    const fp16* x_ptr, const fp16* z_ptr, const fp16* weight_ptr,
    fp16* output_ptr, int rows, int V, float eps,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({(size_t)rows}, {1}),
            RmsNormGated_kernel{x_ptr, z_ptr, weight_ptr, output_ptr,
                                rows, V, eps});
    });
}
