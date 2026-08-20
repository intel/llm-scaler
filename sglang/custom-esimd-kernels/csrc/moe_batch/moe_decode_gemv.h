#pragma once

#include <sycl/ext/intel/esimd.hpp>

template<int N>
SYCL_ESIMD_FUNCTION
simd<float, N> moe_decode_e4m3_to_float(simd<uint8_t, N> raw) {
    simd<uint16_t, N> value = convert<uint16_t>(raw);
    simd<uint16_t, N> sign = (value >> 7) & 1;
    simd<uint16_t, N> exponent = (value >> 3) & 0xf;
    simd<uint16_t, N> mantissa = value & 0x7;
    simd<uint16_t, N> normal_bits =
        (sign << 15) | ((exponent + 8) << 10) | (mantissa << 7);
    simd<fp16, N> normal = normal_bits.template bit_cast_view<fp16>();
    simd<fp16, N> subnormal =
        convert<fp16>(mantissa) * fp16(1.0f / 512.0f);
    subnormal.merge(-subnormal, sign == 1);
    normal.merge(subnormal, exponent == 0);
    return simd<float, N>(normal);
}

template<int N>
SYCL_ESIMD_FUNCTION
simd<float, N> moe_decode_e5m2_to_float(simd<uint8_t, N> raw) {
    simd<uint16_t, N> value = convert<uint16_t>(raw);
    simd<uint16_t, N> sign = (value >> 7) & 1;
    simd<uint16_t, N> exponent = (value >> 2) & 0x1f;
    simd<uint16_t, N> mantissa = value & 0x3;
    simd<uint16_t, N> bits =
        (sign << 15) | (exponent << 10) | (mantissa << 8);
    bits.merge(sign << 15, exponent == 0);
    return simd<float, N>(bits.template bit_cast_view<fp16>());
}

template<int N>
SYCL_ESIMD_FUNCTION
simd<float, N> moe_decode_fp8_to_float(
    simd<uint8_t, N> raw, int fp8_mode) {
    if (fp8_mode == 0) {
        return moe_decode_e4m3_to_float<N>(raw);
    }
    return moe_decode_e5m2_to_float<N>(raw);
}

template<int VL, int VL_TAIL>
struct MoeUpDecodeGeluTanh {
    const fp16* x;
    const uint8_t* gate_up_weight;
    const float* gate_up_scale;
    const int* selected_experts;
    fp16* intermediates;
    int hidden;
    int intermediate;
    int top_k;
    int fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        const int route = static_cast<int>(item.get_global_id(0));
        const int output_idx = static_cast<int>(item.get_global_id(1));
        if (output_idx >= intermediate) {
            return;
        }

        const int two_intermediate = 2 * intermediate;
        const int expert = selected_experts[route];
        const uint8_t* expert_weight =
            gate_up_weight
            + static_cast<size_t>(expert) * two_intermediate * hidden;
        const uint8_t* gate_weight =
            expert_weight + static_cast<size_t>(output_idx) * hidden;
        const uint8_t* up_weight =
            expert_weight
            + static_cast<size_t>(intermediate + output_idx) * hidden;

        const int full_end = (hidden / VL) * VL;
        simd<float, VL> gate_accumulator = 0.0f;
        simd<float, VL> up_accumulator = 0.0f;
        for (int k = 0; k < full_end; k += VL) {
            simd<float, VL> input = block_load<fp16, VL>(x + k);
            gate_accumulator += input * moe_decode_fp8_to_float<VL>(
                block_load<uint8_t, VL>(gate_weight + k), fp8_mode);
            up_accumulator += input * moe_decode_fp8_to_float<VL>(
                block_load<uint8_t, VL>(up_weight + k), fp8_mode);
        }

        float gate = reduce<float>(gate_accumulator, std::plus<>());
        float up = reduce<float>(up_accumulator, std::plus<>());
        if constexpr (VL_TAIL > 0) {
            simd<float, VL_TAIL> input =
                block_load<fp16, VL_TAIL>(x + full_end);
            gate += reduce<float>(
                input * moe_decode_fp8_to_float<VL_TAIL>(
                    block_load<uint8_t, VL_TAIL>(
                        gate_weight + full_end),
                    fp8_mode),
                std::plus<>());
            up += reduce<float>(
                input * moe_decode_fp8_to_float<VL_TAIL>(
                    block_load<uint8_t, VL_TAIL>(
                        up_weight + full_end),
                    fp8_mode),
                std::plus<>());
        }

        const float scale = gate_up_scale[expert];
        gate *= scale;
        up *= scale;
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
        intermediates[
            static_cast<size_t>(route) * intermediate + output_idx
        ] = fp16(gelu * up);
    }
};

template<int VL, int VL_TAIL>
struct MoeDownDecode {
    const fp16* intermediates;
    const uint8_t* down_weight;
    const float* down_scale;
    const fp16* routing_weights;
    const int* selected_experts;
    fp16* output;
    int hidden;
    int intermediate;
    int top_k;
    int fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        const int route = static_cast<int>(item.get_global_id(0));
        const int output_idx = static_cast<int>(item.get_global_id(1));
        if (output_idx >= hidden) {
            return;
        }

        const int expert = selected_experts[route];
        const uint8_t* weight =
            down_weight
            + static_cast<size_t>(expert) * hidden * intermediate
            + static_cast<size_t>(output_idx) * intermediate;
        const fp16* input =
            intermediates + static_cast<size_t>(route) * intermediate;
        const int full_end = (intermediate / VL) * VL;
        simd<float, VL> accumulator = 0.0f;
        for (int k = 0; k < full_end; k += VL) {
            simd<float, VL> values = block_load<fp16, VL>(input + k);
            accumulator += values * moe_decode_fp8_to_float<VL>(
                block_load<uint8_t, VL>(weight + k), fp8_mode);
        }

        float value = reduce<float>(accumulator, std::plus<>());
        if constexpr (VL_TAIL > 0) {
            simd<float, VL_TAIL> values =
                block_load<fp16, VL_TAIL>(input + full_end);
            value += reduce<float>(
                values * moe_decode_fp8_to_float<VL_TAIL>(
                    block_load<uint8_t, VL_TAIL>(weight + full_end),
                    fp8_mode),
                std::plus<>());
        }

        output[static_cast<size_t>(route) * hidden + output_idx] = fp16(
            value * static_cast<float>(routing_weights[route])
            * down_scale[expert]);
    }
};

template<int VL, int VL_TAIL>
struct MoeDownAccumulateDecode {
    const fp16* intermediates;
    const uint8_t* down_weight;
    const float* down_scale;
    const fp16* routing_weights;
    const int* selected_experts;
    const float* per_expert_scale;
    fp16* output;
    int hidden;
    int intermediate;
    int top_k;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int output_idx = static_cast<int>(item.get_global_id(0));
        if (output_idx >= hidden) {
            return;
        }

        float output_value = 0.0f;
        const int full_end = (intermediate / VL) * VL;
        for (int route = 0; route < top_k; ++route) {
            const int expert = selected_experts[route];
            const uint8_t* weight =
                down_weight
                + static_cast<size_t>(expert) * hidden * intermediate
                + static_cast<size_t>(output_idx) * intermediate;
            const fp16* input =
                intermediates + static_cast<size_t>(route) * intermediate;
            simd<float, VL> accumulator = 0.0f;
            for (int k = 0; k < full_end; k += VL) {
                simd<float, VL> values =
                    block_load<fp16, VL>(input + k);
                accumulator += values * moe_decode_fp8_to_float<VL>(
                    block_load<uint8_t, VL>(weight + k), fp8_mode);
            }

            float value = reduce<float>(accumulator, std::plus<>());
            if constexpr (VL_TAIL > 0) {
                simd<float, VL_TAIL> values =
                    block_load<fp16, VL_TAIL>(input + full_end);
                value += reduce<float>(
                    values * moe_decode_fp8_to_float<VL_TAIL>(
                        block_load<uint8_t, VL_TAIL>(
                            weight + full_end),
                        fp8_mode),
                    std::plus<>());
            }
            const fp16 folded_weight = fp16(
                static_cast<float>(routing_weights[route])
                * per_expert_scale[expert]);
            output_value += value * static_cast<float>(folded_weight)
                * down_scale[expert];
        }
        output[output_idx] = fp16(output_value);
    }
};

struct MoeFoldExpertScale {
    fp16* topk_weight;
    const int* topk_idx;
    const float* per_expert_scale;
    int top_k;

    void operator()(sycl::id<1> item) const SYCL_ESIMD_KERNEL {
        const int route = static_cast<int>(item[0]);
        if (route < top_k) {
            topk_weight[route] = fp16(
                static_cast<float>(topk_weight[route])
                * per_expert_scale[topk_idx[route]]);
        }
    }
};
