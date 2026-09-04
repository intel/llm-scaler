/* resadd_norm_gemv_int4.h — ResidualAdd + RMSNorm + INT4 GEMV.
 *
 * ResidualAdd and RMSNorm run once in a single-work-item kernel. The existing
 * INT4 GEMV kernel then consumes the normalized output. Keeping the in-place
 * residual update out of the GEMV grid avoids a cross-work-group data race:
 * every GEMV work-group sees the same normalized vector.
 *
 * The current XPU stream queue preserves submission order, so the GEMV starts
 * after normalization without a host synchronization.
 */

#pragma once

#include "int4_GEMV.h"
#include "utils.h"

template<int VL>
struct ResAddRMSNorm_int4_kernel {
    const fp16* hidden_ptr;
    fp16* residual_ptr;
    const fp16* norm_w_ptr;
    fp16* normed_out;
    int K;
    float eps;

    void operator()(sycl::nd_item<1>) const SYCL_ESIMD_KERNEL {
        float sum_sq = 0.0f;
        for (int offset = 0; offset < K; offset += VL) {
            simd<float, VL> hidden = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> residual = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = hidden + residual;
            block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));
            sum_sq += reduce<float>(added * added, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / static_cast<float>(K) + eps))[0];
        for (int offset = 0; offset < K; offset += VL) {
            simd<float, VL> residual = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> weight = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = residual * inv_rms * weight;
            block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
        }
    }
};

inline void resadd_norm_gemv_int4_pert_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const int32_t* gemv_weight, const fp16* gemv_scale,
    fp16* output, fp16* normed_out,
    int N, int K, float eps, sycl::queue& q)
{
    #define LAUNCH_RESADD_NORM(V) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(1, 1), \
                ResAddRMSNorm_int4_kernel<V>{ \
                    hidden_ptr, residual_ptr, norm_w_ptr, normed_out, K, eps}); \
        });

    if      (K % 512 == 0) { LAUNCH_RESADD_NORM(512) }
    else if (K % 256 == 0) { LAUNCH_RESADD_NORM(256) }
    else                   { LAUNCH_RESADD_NORM(128) }

    #undef LAUNCH_RESADD_NORM

    GEMV_int4_host(
        reinterpret_cast<uint8_t*>(normed_out),
        reinterpret_cast<const uint8_t*>(gemv_weight),
        reinterpret_cast<const uint8_t*>(gemv_scale),
        reinterpret_cast<uint8_t*>(output),
        static_cast<uint32_t>(N), static_cast<uint32_t>(K), q);
}
