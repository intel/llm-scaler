#pragma once

#include "utils.h"

template<int VL>
struct NormAddNormGemvGeluFp8Kernel {
    const fp16* attention_output;
    const fp16* residual_input;
    const fp16* post_attention_weight;
    const fp16* pre_feedforward_weight;
    const uint8_t* gate_up_weight;
    const float* gate_up_scale;
    fp16* residual_output;
    fp16* activation_output;
    int intermediate_size;
    int hidden_size;
    float post_attention_eps;
    float pre_feedforward_eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int output_idx = item.get_group(0);
        if (output_idx >= intermediate_size) {
            return;
        }

        const int chunks = hidden_size / VL;
        float post_attention_sum_sq = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> values =
                block_load<fp16, VL>(attention_output + offset);
            post_attention_sum_sq +=
                reduce<float>(values * values, std::plus<>());
        }
        const float post_attention_inv_rms =
            sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(
                    post_attention_sum_sq / hidden_size
                    + post_attention_eps))[0];

        float pre_feedforward_sum_sq = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> attention =
                block_load<fp16, VL>(attention_output + offset);
            simd<float, VL> residual =
                block_load<fp16, VL>(residual_input + offset);
            simd<float, VL> norm_weight =
                block_load<fp16, VL>(post_attention_weight + offset);
            simd<float, VL> updated =
                attention * post_attention_inv_rms * norm_weight + residual;
            if (output_idx == 0) {
                block_store<fp16, VL>(
                    residual_output + offset, simd<fp16, VL>(updated));
            }
            pre_feedforward_sum_sq +=
                reduce<float>(updated * updated, std::plus<>());
        }
        const float pre_feedforward_inv_rms =
            sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(
                    pre_feedforward_sum_sq / hidden_size
                    + pre_feedforward_eps))[0];

        const uint8_t* gate_weight =
            gate_up_weight
            + static_cast<size_t>(output_idx) * hidden_size;
        const uint8_t* up_weight =
            gate_up_weight
            + static_cast<size_t>(intermediate_size + output_idx)
                * hidden_size;
        simd<float, VL> gate_accumulator = 0.0f;
        simd<float, VL> up_accumulator = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> attention =
                block_load<fp16, VL>(attention_output + offset);
            simd<float, VL> residual =
                block_load<fp16, VL>(residual_input + offset);
            simd<float, VL> post_weight =
                block_load<fp16, VL>(post_attention_weight + offset);
            simd<float, VL> pre_weight =
                block_load<fp16, VL>(pre_feedforward_weight + offset);
            simd<float, VL> updated = simd<fp16, VL>(
                attention * post_attention_inv_rms * post_weight + residual);
            simd<float, VL> normalized =
                updated * pre_feedforward_inv_rms * pre_weight;
            gate_accumulator += normalized * fp8_dequant_rng<VL>(
                block_load<uint8_t, VL>(gate_weight + offset), fp8_mode);
            up_accumulator += normalized * fp8_dequant_rng<VL>(
                block_load<uint8_t, VL>(up_weight + offset), fp8_mode);
        }

        const float scale = *gate_up_scale;
        float gate =
            reduce<float>(gate_accumulator, std::plus<>()) * scale;
        const float up =
            reduce<float>(up_accumulator, std::plus<>()) * scale;
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coefficient = 0.044715f;
        const float inner = sqrt_2_over_pi
            * (gate + coefficient * gate * gate * gate);
        float two_inner = 2.0f * inner;
        two_inner = two_inner > 30.0f ? 30.0f : two_inner;
        two_inner = two_inner < -30.0f ? -30.0f : two_inner;
        const float exponent = sycl::exp(two_inner);
        const float gelu =
            0.5f * gate * (1.0f + (exponent - 1.0f) / (exponent + 1.0f));
        activation_output[output_idx] = fp16(gelu * up);
    }
};

inline void norm_add_norm_gemv_gelu_fp8_host(
    const fp16* attention_output,
    const fp16* residual_input,
    const fp16* post_attention_weight,
    const fp16* pre_feedforward_weight,
    const uint8_t* gate_up_weight,
    const float* gate_up_scale,
    fp16* residual_output,
    fp16* activation_output,
    int intermediate_size,
    int hidden_size,
    float post_attention_eps,
    float pre_feedforward_eps,
    int fp8_mode,
    sycl::queue& queue) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(intermediate_size, 1),
            NormAddNormGemvGeluFp8Kernel<256>{
                attention_output,
                residual_input,
                post_attention_weight,
                pre_feedforward_weight,
                gate_up_weight,
                gate_up_scale,
                residual_output,
                activation_output,
                intermediate_size,
                hidden_size,
                post_attention_eps,
                pre_feedforward_eps,
                fp8_mode});
    });
}
