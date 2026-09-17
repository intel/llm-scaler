/* resadd_norm_gemv_fused.h — ResidualAdd + RMSNorm + FP8 GEMV.
 *
 * The residual update and RMSNorm run in one work-group, then the established
 * FP8 GEMV kernel consumes normed_out. Fusing the update into the router GEMV
 * grid is not safe: each router work-group reads residual while one of them
 * writes it in-place, and SYCL provides no grid-wide barrier inside a kernel.
 */

#pragma once

#include "fp8_GEMV_v2.h"
#include "utils.h"

template<int VL>
struct ResAddRMSNorm_fp8_kernel {
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

template<int VL>
struct FlatGEMV_fp8_pert_kernel {
    const fp16* input;
    const uint8_t* weight;
    const float* scale_ptr;
    fp16* output;
    int N;
    int K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_global_id(0);
        if (n >= N) return;

        simd<float, VL> acc = 0.0f;
        for (int k = 0; k < K; k += VL) {
            simd<float, VL> input_f = block_load<fp16, VL>(input + k);
            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(
                weight + static_cast<size_t>(n) * K + k);
            acc += input_f * fp8_dequant<VL>(raw, fp8_mode);
        }
        output[n] = fp16(reduce<float>(acc, std::plus<>()) * *scale_ptr);
    }
};

inline void resadd_norm_gemv_fp8_pert_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* gemv_weight, const float* gemv_scale,
    fp16* output, fp16* normed_out,
    int N, int K, float eps, int fp8_mode, sycl::queue& q)
{
    sycl::event norm_done;
    #define LAUNCH_RESADD_NORM(V)                                             \
        norm_done = q.submit([&](sycl::handler& cgh) {                        \
            cgh.parallel_for(                                                 \
                sycl::nd_range<1>(1, 1),                                     \
                ResAddRMSNorm_fp8_kernel<V>{                                  \
                    hidden_ptr, residual_ptr, norm_w_ptr, normed_out, K, eps}); \
        });

    if      (K % 512 == 0) { LAUNCH_RESADD_NORM(512) }
    else if (K % 256 == 0) { LAUNCH_RESADD_NORM(256) }
    else                    { LAUNCH_RESADD_NORM(128) }

    #undef LAUNCH_RESADD_NORM

    // The tuned K-split GEMV is correct for the 512-aligned decode shapes.
    // Route it explicitly after normalization instead of relying on queue order.
    if (K % 512 == 0) {
        GEMV_fp8_pert_host(
            reinterpret_cast<uint8_t*>(normed_out),
            const_cast<uint8_t*>(gemv_weight),
            reinterpret_cast<uint8_t*>(const_cast<float*>(gemv_scale)),
            reinterpret_cast<uint8_t*>(output),
            static_cast<uint32_t>(N), static_cast<uint32_t>(K), fp8_mode, q,
            &norm_done);
        return;
    }

    constexpr int router_wg_size = 32;
    const int router_global =
        ((N + router_wg_size - 1) / router_wg_size) * router_wg_size;
    q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(norm_done);
        cgh.parallel_for(
            sycl::nd_range<1>(router_global, router_wg_size),
            FlatGEMV_fp8_pert_kernel<128>{
                normed_out, gemv_weight, gemv_scale, output, N, K, fp8_mode});
    });
}

// Retained for callers that select the non-512-aligned route. The one-WG
// normalization dispatch above already covers the 256- and 128-aligned cases.
inline void resadd_norm_gemv_fp8_pert_v2_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* gemv_weight, const float* gemv_scale,
    fp16* output, fp16* normed_out,
    int N, int K, float eps, int fp8_mode, sycl::queue& q)
{
    resadd_norm_gemv_fp8_pert_host(
        hidden_ptr, residual_ptr, norm_w_ptr, gemv_weight, gemv_scale,
        output, normed_out, N, K, eps, fp8_mode, q);
}
