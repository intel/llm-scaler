#pragma once

#include "utils.h"

template<int VL>
struct RmsNormGemvFp8Kernel {
    const fp16* input;
    const fp16* norm_weight;
    const uint8_t* gemv_weight;
    const float* gemv_scale;
    fp16* output;
    int output_size;
    int hidden_size;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int output_idx = item.get_group(0);
        if (output_idx >= output_size) {
            return;
        }

        const int chunks = hidden_size / VL;
        float sum_sq = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> values =
                block_load<fp16, VL>(input + offset);
            sum_sq += reduce<float>(values * values, std::plus<>());
        }
        const float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / hidden_size + eps))[0];

        const uint8_t* weight =
            gemv_weight + static_cast<size_t>(output_idx) * hidden_size;
        simd<float, VL> accumulator = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> values =
                block_load<fp16, VL>(input + offset);
            simd<float, VL> scale =
                block_load<fp16, VL>(norm_weight + offset);
            simd<float, VL> normalized =
                simd<fp16, VL>(values * inv_rms * scale);
            accumulator += normalized * fp8_dequant_rng<VL>(
                block_load<uint8_t, VL>(weight + offset), fp8_mode);
        }
        output[output_idx] = fp16(
            reduce<float>(accumulator, std::plus<>()) * *gemv_scale);
    }
};

inline void rmsnorm_gemv_fp8_host(
    const fp16* input,
    const fp16* norm_weight,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    int output_size,
    int hidden_size,
    float eps,
    int fp8_mode,
    sycl::queue& queue) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(output_size, 1),
            RmsNormGemvFp8Kernel<256>{
                input,
                norm_weight,
                gemv_weight,
                gemv_scale,
                output,
                output_size,
                hidden_size,
                eps,
                fp8_mode});
    });
}
