#pragma once

#include "utils.h"

template<int VL, int MAX_CHUNKS>
struct NormGemvNormFp16Kernel {
    const fp16* residual;
    const fp16* scale_with_root;
    const fp16* proj_weight;
    const fp16* pre_ff_weight;
    fp16* router_logits;
    fp16* moe_input;
    int num_experts;
    int hidden_size;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int expert = item.get_group(0);
        if (expert >= num_experts) {
            return;
        }

        int num_chunks = hidden_size / VL;
        simd<float, VL> residual_chunks[MAX_CHUNKS];

        float sum_sq = 0.0f;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            int offset = chunk * VL;
            simd<float, VL> values = block_load<fp16, VL>(residual + offset);
            residual_chunks[chunk] = values;
            sum_sq += reduce<float>(values * values, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / static_cast<float>(hidden_size) + eps))[0];

        simd<float, VL> accumulator = 0.0f;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            int offset = chunk * VL;
            simd<float, VL> normalized = residual_chunks[chunk] * inv_rms;

            if (expert == 0) {
                simd<float, VL> pre_ff =
                    block_load<fp16, VL>(pre_ff_weight + offset);
                block_store<fp16, VL>(
                    moe_input + offset,
                    simd<fp16, VL>(normalized * pre_ff));
            }

            simd<float, VL> router_scale =
                block_load<fp16, VL>(scale_with_root + offset);
            simd<float, VL> weight = block_load<fp16, VL>(
                proj_weight + static_cast<size_t>(expert) * hidden_size + offset);
            accumulator += normalized * router_scale * weight;
        }

        router_logits[expert] = fp16(reduce<float>(accumulator, std::plus<>()));
    }
};

inline void norm_gemv_norm_fp16_host(
    const fp16* residual,
    const fp16* scale_with_root,
    const fp16* proj_weight,
    const fp16* pre_ff_weight,
    fp16* router_logits,
    fp16* moe_input,
    int num_experts,
    int hidden_size,
    float eps,
    sycl::queue& queue) {

#define LAUNCH_NORM_GEMV_NORM(VL, MAX_CHUNKS)                         \
    queue.submit([&](sycl::handler& cgh) {                            \
        cgh.parallel_for(                                             \
            sycl::nd_range<1>(num_experts, 1),                        \
            NormGemvNormFp16Kernel<VL, MAX_CHUNKS>{                   \
                residual, scale_with_root, proj_weight, pre_ff_weight, \
                router_logits, moe_input, num_experts, hidden_size, eps}); \
    });

    if (hidden_size % 512 == 0) {
        int chunks = hidden_size / 512;
        if (chunks <= 4) {
            LAUNCH_NORM_GEMV_NORM(512, 4)
        } else if (chunks <= 8) {
            LAUNCH_NORM_GEMV_NORM(512, 8)
        } else {
            LAUNCH_NORM_GEMV_NORM(512, 16)
        }
    } else if (hidden_size % 256 == 0) {
        int chunks = hidden_size / 256;
        if (chunks <= 8) {
            LAUNCH_NORM_GEMV_NORM(256, 8)
        } else if (chunks <= 16) {
            LAUNCH_NORM_GEMV_NORM(256, 16)
        } else {
            LAUNCH_NORM_GEMV_NORM(256, 32)
        }
    } else if (hidden_size % 128 == 0) {
        int chunks = hidden_size / 128;
        if (chunks <= 16) {
            LAUNCH_NORM_GEMV_NORM(128, 16)
        } else {
            LAUNCH_NORM_GEMV_NORM(128, 32)
        }
    } else {
        int chunks = hidden_size / 64;
        if (chunks <= 32) {
            LAUNCH_NORM_GEMV_NORM(64, 32)
        } else {
            LAUNCH_NORM_GEMV_NORM(64, 64)
        }
    }

#undef LAUNCH_NORM_GEMV_NORM
}
