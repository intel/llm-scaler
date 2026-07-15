/* fp16_GEMV.h — FP16×FP16→FP16/FP32 GEMV for decode (M=1).
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
 * Input:  [1, K] fp16
 * Weight: [N, K] fp16 (row-major, contiguous)
 * Output: [1, N] fp16
 */

#pragma once
#include "utils.h"

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

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;

            simd<fp16, VL> wv = block_load<fp16, VL>(weight + (size_t)n * K + k);
            simd<float, VL> w_f = wv;

            acc += input_f * w_f;
        }

        float my_sum = reduce<float>(acc, std::plus<>());

        if constexpr (K_SPLIT == 1) {
            output[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n] = fp16(reduce<float>(parts, std::plus<>()));
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
    uint32_t N,
    uint32_t K,
    sycl::queue& q) {

    int vl, ks;
    select_vl_ks_fp16(N, K, vl, ks);

    uint32_t global = N * ks;
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
