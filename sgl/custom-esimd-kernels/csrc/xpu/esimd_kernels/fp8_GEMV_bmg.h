/* fp8_GEMV_bmg.h — BMG-optimized FP8 GEMV with K_SPLIT and K-tail handling.
 *
 * Designed for shapes that fp8_GEMV_v2 picks suboptimal (vl, ks) on:
 *   K=1056 → V2 picks vl=32, ks=1 (loop 33×) → ~9.9us @ 300 GB/s.
 *   This kernel allows kp = K/ks not divisible by VL via masked tail.
 *
 *   N=2816, K=1056: ks=4 → kp=264; we pick VL=256 + 1 tail of 8 (256+8 -> wait, 264 = 256+8 = 256+8, but VL=256 reads OOB)
 *   Better: VL=128, kp=264 → 2 chunks of 128 + 8 tail with mask.
 *   Or VL=64, kp=264 → 4 chunks of 64 + 8 tail.
 *
 * Strategy:
 *   1. Pick kp such that ks × kp = K.
 *   2. Pick VL such that VL ≤ kp; do (kp / VL) full VL loads + 1 tail load with mask if kp % VL > 0.
 *
 * For decode M=1 only.
 */

#pragma once
#include "utils.h"

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_bmg(
    simd<uint8_t, VL> raw, int fp8_mode) {
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    simd<uint16_t, VL> fp8_sign = (u16 >> 7) & 1;
    simd<uint16_t, VL> fp16_bits;

    if (fp8_mode == 0) {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 3) & 0xF;
        simd<uint16_t, VL> fp8_mant = u16 & 0x7;
        fp16_bits = (fp8_sign << 15) | ((fp8_exp + 8) << 10) | (fp8_mant << 7);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    } else {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 2) & 0x1F;
        simd<uint16_t, VL> fp8_mant = u16 & 0x3;
        fp16_bits = (fp8_sign << 15) | (fp8_exp << 10) | (fp8_mant << 8);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    }

    simd<fp16, VL> wh = fp16_bits.template bit_cast_view<fp16>().read();
    return simd<float, VL>(wh);
}

/* GEMV per-tensor scale, BMG-tuned with K_SPLIT and tail handling.
 *
 * Each WG handles one output channel n; K_SPLIT threads cooperate, each
 * owning kp = K/K_SPLIT of the K dimension. VL is fixed per kernel; if
 * kp is not a multiple of VL, the last partial chunk uses gather_load
 * with a mask.
 */
template<int VL, int K_SPLIT>
struct GEMV_fp8_pert_bmg_kernel {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;       // chunk per thread (may be != multiple of VL)
        int ks = lid * kp;
        int kp_full = (kp / VL) * VL;
        int tail = kp - kp_full;     // 0 if VL divides kp

        simd<float, VL> acc = 0.0f;

        // Full VL loads
        int k = ks;
        for (; k < ks + kp_full; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;

            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(weight + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant_bmg<VL>(raw, fp8_mode);

            acc += input_f * wf;
        }

        // Tail handling: read the last `tail` elements; pad to VL with zeros.
        // We use unaligned gather to avoid OOB. For BMG ESIMD, simplest is to
        // do a smaller block_load with a smaller VL_TAIL constant. Instead, we
        // require K_SPLIT chosen such that kp % VL == 0 in the host dispatcher
        // (most cases). Tail support enabled via the loop above — if tail==0,
        // this branch is skipped. If tail > 0, host should not pick this template.

        float my_sum = reduce<float>(acc, std::plus<>()) * *scale_ptr;

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

/* Tail-aware variant: each thread does (kp / VL_BIG) full big chunks + 1 tail
 * chunk of VL_TAIL (where VL_TAIL = kp % VL_BIG, must be a power-of-2). */
template<int VL_BIG, int VL_TAIL, int K_SPLIT>
struct GEMV_fp8_pert_bmg_tail_kernel {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;
    fp16*          output;
    int N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;
        int kp_full = (kp / VL_BIG) * VL_BIG;

        simd<float, VL_BIG> acc = 0.0f;

        for (int k = ks; k < ks + kp_full; k += VL_BIG) {
            simd<fp16, VL_BIG> iv = block_load<fp16, VL_BIG>(input + k);
            simd<float, VL_BIG> input_f = iv;

            simd<uint8_t, VL_BIG> raw = block_load<uint8_t, VL_BIG>(weight + (size_t)n * K + k);
            simd<float, VL_BIG> wf = fp8_dequant_bmg<VL_BIG>(raw, fp8_mode);

            acc += input_f * wf;
        }

        // Tail of VL_TAIL elements
        if constexpr (VL_TAIL > 0) {
            int kt = ks + kp_full;
            simd<fp16, VL_TAIL> iv_t = block_load<fp16, VL_TAIL>(input + kt);
            simd<float, VL_TAIL> input_t = iv_t;

            simd<uint8_t, VL_TAIL> raw_t = block_load<uint8_t, VL_TAIL>(
                weight + (size_t)n * K + kt);
            simd<float, VL_TAIL> wf_t = fp8_dequant_bmg<VL_TAIL>(raw_t, fp8_mode);

            float tail_sum = reduce<float>(input_t * wf_t, std::plus<>());
            float big_sum = reduce<float>(acc, std::plus<>());
            float my_sum = (big_sum + tail_sum) * *scale_ptr;

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
        } else {
            float my_sum = reduce<float>(acc, std::plus<>()) * *scale_ptr;

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
    }
};

/* Pick (VL_BIG, VL_TAIL, K_SPLIT) for given (N, K).
 * Goal: maximize parallelism. BMG has ~640 hardware threads.
 * Strategy:
 *   - Pick K_SPLIT so that N × K_SPLIT >= 1280 (target full saturation).
 *   - K_SPLIT must divide K (kp = K/K_SPLIT).
 *   - VL_BIG: prefer 256 / 128, or whatever makes (kp / VL_BIG) ≥ 2 with a
 *     small tail.
 */
inline void select_bmg(uint32_t N, uint32_t K, int& vl_big, int& vl_tail, int& ks) {
    // Default: vl=256, ks=1 (matches v2 for nice K values).
    vl_big = 256; vl_tail = 0; ks = 1;

    // Target threads = N × ks; aim for >= 640 (BMG full occupancy).
    int target_ks = 1;
    if (N * 8 <= 640) target_ks = 8;
    else if (N * 4 <= 640) target_ks = 4;
    else if (N * 2 <= 640) target_ks = 2;
    else target_ks = 1;

    // K_SPLIT must divide K. Find largest divisor of K that's <= target_ks.
    int chosen_ks = 1;
    for (int s = target_ks; s >= 1; s /= 2) {
        if (K % s == 0) { chosen_ks = s; break; }
    }
    ks = chosen_ks;

    int kp = K / ks;
    // Pick VL_BIG: largest power-of-2 ≤ 256 that gives kp_full > 0.
    int candidates[] = {256, 128, 64, 32};
    for (int c : candidates) {
        if (kp >= c) {
            vl_big = c;
            int kp_full = (kp / c) * c;
            int tail = kp - kp_full;
            // Round tail up to next power of 2 ≤ vl_big? Tail must be a
            // power-of-2 supported by block_load: 8, 16, 32, 64, ...
            // If tail==0 we leave vl_tail=0.
            // Otherwise pick the smallest power-of-2 >= tail; if it's not
            // exactly equal, fall back to a smaller vl_big.
            if (tail == 0) {
                vl_tail = 0;
                return;
            }
            // Find power-of-2 that exactly equals tail.
            int t_pow = 0;
            for (int t : {8, 16, 32, 64, 128}) {
                if (t == tail) { t_pow = t; break; }
            }
            if (t_pow > 0) {
                vl_tail = t_pow;
                return;
            }
            // tail is not a clean power-of-2; try smaller VL_BIG.
        }
    }
    // Fallback: vl_big = 32, no tail.
    vl_big = 32; vl_tail = 0;
}

inline void GEMV_fp8_pert_bmg_host(
    const fp16* p_in, const uint8_t* p_w, const float* p_sc, fp16* p_out,
    uint32_t N, uint32_t K, int fp8_mode, sycl::queue& q)
{
    int vl_big, vl_tail, ks;
    select_bmg(N, K, vl_big, vl_tail, ks);

    uint32_t global = N * ks;
    uint32_t local  = ks;

    #define LAUNCH_NOTAIL(V, KS)         q.submit([&](sycl::handler& h) {             h.parallel_for(sycl::nd_range<1>(global, local),                 GEMV_fp8_pert_bmg_kernel<V, KS>{p_in, p_w, p_sc, p_out, (int)N, (int)K, fp8_mode});         });
    #define LAUNCH_TAIL(V, T, KS)         q.submit([&](sycl::handler& h) {             h.parallel_for(sycl::nd_range<1>(global, local),                 GEMV_fp8_pert_bmg_tail_kernel<V, T, KS>{p_in, p_w, p_sc, p_out, (int)N, (int)K, fp8_mode});         });

    if (vl_tail == 0) {
        // No tail — use simple kernel.
        if      (vl_big == 256 && ks == 1) { LAUNCH_NOTAIL(256, 1) }
        else if (vl_big == 256 && ks == 2) { LAUNCH_NOTAIL(256, 2) }
        else if (vl_big == 256 && ks == 4) { LAUNCH_NOTAIL(256, 4) }
        else if (vl_big == 128 && ks == 1) { LAUNCH_NOTAIL(128, 1) }
        else if (vl_big == 128 && ks == 2) { LAUNCH_NOTAIL(128, 2) }
        else if (vl_big == 128 && ks == 4) { LAUNCH_NOTAIL(128, 4) }
        else if (vl_big == 128 && ks == 8) { LAUNCH_NOTAIL(128, 8) }
        else if (vl_big == 64  && ks == 1) { LAUNCH_NOTAIL(64,  1) }
        else if (vl_big == 64  && ks == 2) { LAUNCH_NOTAIL(64,  2) }
        else if (vl_big == 64  && ks == 4) { LAUNCH_NOTAIL(64,  4) }
        else if (vl_big == 32  && ks == 1) { LAUNCH_NOTAIL(32,  1) }
        else                               { LAUNCH_NOTAIL(32,  1) }
    } else {
        // With tail — instantiate (VL_BIG, VL_TAIL, KS) combos.
        if      (vl_big == 256 && vl_tail == 8  && ks == 1) { LAUNCH_TAIL(256, 8,  1) }
        else if (vl_big == 256 && vl_tail == 16 && ks == 1) { LAUNCH_TAIL(256, 16, 1) }
        else if (vl_big == 256 && vl_tail == 32 && ks == 1) { LAUNCH_TAIL(256, 32, 1) }
        else if (vl_big == 256 && vl_tail == 8  && ks == 2) { LAUNCH_TAIL(256, 8,  2) }
        else if (vl_big == 256 && vl_tail == 16 && ks == 2) { LAUNCH_TAIL(256, 16, 2) }
        else if (vl_big == 128 && vl_tail == 8  && ks == 1) { LAUNCH_TAIL(128, 8,  1) }
        else if (vl_big == 128 && vl_tail == 16 && ks == 1) { LAUNCH_TAIL(128, 16, 1) }
        else if (vl_big == 128 && vl_tail == 32 && ks == 1) { LAUNCH_TAIL(128, 32, 1) }
        else if (vl_big == 128 && vl_tail == 64 && ks == 1) { LAUNCH_TAIL(128, 64, 1) }
        else if (vl_big == 128 && vl_tail == 8  && ks == 2) { LAUNCH_TAIL(128, 8,  2) }
        else if (vl_big == 128 && vl_tail == 16 && ks == 2) { LAUNCH_TAIL(128, 16, 2) }
        else if (vl_big == 128 && vl_tail == 32 && ks == 2) { LAUNCH_TAIL(128, 32, 2) }
        else if (vl_big == 128 && vl_tail == 8  && ks == 4) { LAUNCH_TAIL(128, 8,  4) }
        else if (vl_big == 128 && vl_tail == 16 && ks == 4) { LAUNCH_TAIL(128, 16, 4) }
        else if (vl_big == 64  && vl_tail == 8  && ks == 1) { LAUNCH_TAIL(64,  8,  1) }
        else if (vl_big == 64  && vl_tail == 16 && ks == 1) { LAUNCH_TAIL(64,  16, 1) }
        else if (vl_big == 64  && vl_tail == 32 && ks == 1) { LAUNCH_TAIL(64,  32, 1) }
        else if (vl_big == 64  && vl_tail == 8  && ks == 2) { LAUNCH_TAIL(64,  8,  2) }
        else if (vl_big == 64  && vl_tail == 16 && ks == 2) { LAUNCH_TAIL(64,  16, 2) }
        else if (vl_big == 64  && vl_tail == 8  && ks == 4) { LAUNCH_TAIL(64,  8,  4) }
        else                                                  { LAUNCH_NOTAIL(32, 1) }
    }

    #undef LAUNCH_NOTAIL
    #undef LAUNCH_TAIL
}
