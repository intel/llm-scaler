/* fp16_GEMV.h — FP16×FP16→FP16/FP32 GEMV for small-batch decode.
 *
 * Mirrors the fp8_GEMV_v2 dispatch pattern (templated VL + K_SPLIT,
 * SLM partial reduction) but skips the FP8 dequant — both input and
 * weight are loaded as fp16 directly, accumulated in fp32.
 *
 * Designed for the Gemma4 router (N=128, K=2816, weight is the fp16
 * GateLinear projection, input is the per-step normed hidden), but
 * generalizes to any small-N decode-batch GEMV that has no per-tensor
 * scale.
 *
 * Input:  [M, K] fp16
 * Weight: [N, K] fp16 (row-major, contiguous)
 * Output: [M, N] fp16
 */

#pragma once
#include "utils.h"

namespace esimd_math = sycl::ext::intel::esimd;

SYCL_ESIMD_FUNCTION inline float gelu_tanh_scalar_esimd(float x) {
    constexpr float c0 = 0.7978845608028654f;
    constexpr float c1 = 0.044715f;
    simd<float, 1> xv(x);
    simd<float, 1> inner = c0 * (xv + c1 * xv * xv * xv);
    inner = esimd_math::min<float, 1>(
        esimd_math::max<float, 1>(inner, simd<float, 1>(-30.0f)),
        simd<float, 1>(30.0f));
    simd<float, 1> e2 = esimd_math::exp<float, 1>(2.0f * inner);
    simd<float, 1> tanh_v = (e2 - 1.0f) / (e2 + 1.0f);
    return (0.5f * xv * (1.0f + tanh_v))[0];
}

template<int VL, int K_SPLIT>
struct GEMV_fp16_kernel {
    const fp16* input;
    const fp16* weight;
    fp16*       output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int mn  = item.get_group(0);
        int m   = mn / N;
        int n   = mn - m * N;
        int lid = item.get_local_id(0);

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(
                input + (size_t)m * K + k);
            simd<float, VL> input_f = iv;

            simd<fp16, VL> wv = block_load<fp16, VL>(weight + (size_t)n * K + k);
            simd<float, VL> w_f = wv;

            acc += input_f * w_f;
        }

        float my_sum = reduce<float>(acc, std::plus<>());

        if constexpr (K_SPLIT == 1) {
            output[(size_t)m * N + n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[(size_t)m * N + n] = fp16(
                    reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

template<int VL, int K_SPLIT>
struct GEMV_fp16_gelu_mul_kernel {
    const fp16* input;
    const fp16* weight;
    fp16*       output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<2 * K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;
        simd<float, VL> gate_acc = 0.0f;
        simd<float, VL> up_acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;
            simd<fp16, VL> gate_w = block_load<fp16, VL>(
                weight + (size_t)n * K + k);
            simd<fp16, VL> up_w = block_load<fp16, VL>(
                weight + (size_t)(N + n) * K + k);
            gate_acc += input_f * simd<float, VL>(gate_w);
            up_acc += input_f * simd<float, VL>(up_w);
        }

        float gate_sum = reduce<float>(gate_acc, std::plus<>());
        float up_sum = reduce<float>(up_acc, std::plus<>());
        if constexpr (K_SPLIT == 1) {
            output[n] = fp16(gelu_tanh_scalar_esimd(gate_sum) * up_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float),
                                       simd<float, 1>(gate_sum));
            slm_block_store<float, 1>(
                (K_SPLIT + lid) * sizeof(float), simd<float, 1>(up_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> gate_parts =
                    slm_block_load<float, K_SPLIT>(0);
                simd<float, K_SPLIT> up_parts =
                    slm_block_load<float, K_SPLIT>(K_SPLIT * sizeof(float));
                float gate_total = reduce<float>(gate_parts, std::plus<>());
                float up_total = reduce<float>(up_parts, std::plus<>());
                output[n] = fp16(
                    gelu_tanh_scalar_esimd(gate_total) * up_total);
            }
        }
    }
};

// Reuse fp8_GEMV_v2.h's select_vl_ks (declared there); declare again to be
// header-self-contained. Same heuristic: small-N + large-K benefits from
// K_SPLIT > 1 to spread work across more threads.
inline void select_vl_ks_fp16(uint32_t N, uint32_t K, int& vl, int& ks) {
    vl = 512; ks = 1;
    if (K < 512) {
        vl = 128; ks = 1;
    } else if (K == 512) {
        vl = 256; ks = 1;
    }
    if (N <= 128 && K >= 2048) {
        vl = 128; ks = 8;
    } else if (N <= 512 && K >= 2048) {
        vl = 128; ks = 4;
    }
    int kpt = K / ks;
    while (vl > kpt || kpt % vl != 0) {
        if (vl > 32) {
            vl /= 2;
        } else if (ks > 1) {
            ks /= 2;
            kpt = K / ks;
        } else {
            break;
        }
    }
}

inline void GEMV_fp16_host(
    const fp16* input,
    const fp16* weight,
    fp16*       output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    sycl::queue& q) {

    int vl, ks;
    select_vl_ks_fp16(N, K, vl, ks);

    uint32_t global = M * N * ks;
    uint32_t local  = ks;

    #define LAUNCH(V, KS)         q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(                 sycl::nd_range<1>(global, local),                 GEMV_fp16_kernel<V, KS>{input, weight, output, (int)N, (int)K});         });

    if      (vl == 512 && ks == 1) { LAUNCH(512, 1) }
    else if (vl == 256 && ks == 1) { LAUNCH(256, 1) }
    else if (vl == 128 && ks == 1) { LAUNCH(128, 1) }
    else if (vl == 64  && ks == 1) { LAUNCH(64,  1) }
    else if (vl == 32  && ks == 1) { LAUNCH(32,  1) }
    else if (vl == 128 && ks == 2) { LAUNCH(128, 2) }
    else if (vl == 64  && ks == 2) { LAUNCH(64,  2) }
    else if (vl == 32  && ks == 2) { LAUNCH(32,  2) }
    else if (vl == 128 && ks == 4) { LAUNCH(128, 4) }
    else if (vl == 64  && ks == 4) { LAUNCH(64,  4) }
    else if (vl == 32  && ks == 4) { LAUNCH(32,  4) }
    else if (vl == 128 && ks == 8) { LAUNCH(128, 8) }
    else if (vl == 64  && ks == 8) { LAUNCH(64,  8) }
    else if (vl == 32  && ks == 8) { LAUNCH(32,  8) }
    else                           { LAUNCH(64,  1) }

    #undef LAUNCH
}

inline void GEMV_fp16_gelu_mul_host(
    const fp16* input,
    const fp16* weight,
    fp16* output,
    uint32_t N,
    uint32_t K,
    sycl::queue& q) {
    int vl, ks;
    select_vl_ks_fp16(N, K, vl, ks);
    uint32_t global = N * ks;
    uint32_t local = ks;

    #define LAUNCH_GELU(V, KS) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                GEMV_fp16_gelu_mul_kernel<V, KS>{ \
                    input, weight, output, (int)N, (int)K}); \
        });

    if      (vl == 512 && ks == 1) { LAUNCH_GELU(512, 1) }
    else if (vl == 256 && ks == 1) { LAUNCH_GELU(256, 1) }
    else if (vl == 128 && ks == 1) { LAUNCH_GELU(128, 1) }
    else if (vl == 64  && ks == 1) { LAUNCH_GELU(64,  1) }
    else if (vl == 32  && ks == 1) { LAUNCH_GELU(32,  1) }
    else if (vl == 128 && ks == 2) { LAUNCH_GELU(128, 2) }
    else if (vl == 64  && ks == 2) { LAUNCH_GELU(64,  2) }
    else if (vl == 32  && ks == 2) { LAUNCH_GELU(32,  2) }
    else if (vl == 128 && ks == 4) { LAUNCH_GELU(128, 4) }
    else if (vl == 64  && ks == 4) { LAUNCH_GELU(64,  4) }
    else if (vl == 32  && ks == 4) { LAUNCH_GELU(32,  4) }
    else if (vl == 128 && ks == 8) { LAUNCH_GELU(128, 8) }
    else if (vl == 64  && ks == 8) { LAUNCH_GELU(64,  8) }
    else if (vl == 32  && ks == 8) { LAUNCH_GELU(32, 8) }
    else                           { LAUNCH_GELU(64, 1) }

    #undef LAUNCH_GELU
}
