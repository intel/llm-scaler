/* q4_k_GEMV.h — GGUF q4_K GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * GGML block_q4_K (ggml-common.h:87, QK_K=256): {half2 dm; u8 scales[12];
 * u8 qs[128]}, 8 sub-blocks of 32, ASYMMETRIC with 6-bit sub-scale + 6-bit
 * sub-min. The 6-bit unpack (get_scale_min_k4) and dall/dmin multiply are done
 * on the HOST (q4_k_repack_ref.py, skill Stage 1) so the GPU sees per-32-block
 * fp16 scale + fp16 min — identical granularity to q4_0's group-32.
 *
 * Consumes the interleaved repack (same nibble layout as q4_0_GEMV):
 *   input   [1, K]      fp16
 *   weight  [N, K/2]    uint8  (byte j: low nibble -> elem 2j, high -> elem 2j+1)
 *   scale   [N, K/32]   fp16   (= dall * sc6, pre-computed)
 *   min     [N, K/32]   fp16   (= dmin * mn6, pre-computed)
 *   output  [1, N]      fp16
 *
 * dequant: w[k] = scale[k/32] * nibble[k] - min[k/32]   (nibble in 0..15)
 *
 * Structure = q4_0_GEMV.h (K_SPLIT + SLM reduce) + a min term. The only
 * algorithmic difference from q4_0 is: q4_0 does (nibble-8)*scale (symmetric),
 * q4_K does nibble*scale - min (asymmetric). Same interleaved deinterleave.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

static constexpr int Q4_K_GROUP = 32;  // q4_K sub-block size
static constexpr int Q4_K_HALF = 16;   // qs bytes per 32-block (= group/2)

inline void select_ks_q4_k(uint32_t N, uint32_t K, int& ks) {
    ks = 1;
    if      (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    int kp = K / ks;
    while ((kp % Q4_K_GROUP != 0) && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}

template <int K_SPLIT>
struct Q4_K_gemv_kernel {
    const fp16*    input;   // [1, K]
    const uint8_t* weight;  // [N, K/2]
    const fp16*    scale;   // [N, K/32]
    const fp16*    minv;    // [N, K/32]
    fp16*          output;  // [1, N]
    int N, K;
    int n_groups;           // K / 32

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }
        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int kstart = lid * kp;

        // Even/odd K-position accumulators (interleaved layout, like q4_0).
        simd<float, Q4_K_HALF> acc_even = 0.0f;
        simd<float, Q4_K_HALF> acc_odd  = 0.0f;

        const uint8_t* w_row = weight + (size_t)n * (K / 2);
        const fp16*    s_row = scale  + (size_t)n * n_groups;
        const fp16*    m_row = minv   + (size_t)n * n_groups;
        int group_idx = kstart / Q4_K_GROUP;

        for (int k = kstart; k < kstart + kp; k += Q4_K_GROUP) {
            // input: 32 fp16 -> deinterleave into even[16] + odd[16]
            simd<fp16, Q4_K_GROUP> iv = block_load<fp16, Q4_K_GROUP>(input + k);
            simd<float, Q4_K_HALF> in_even = iv.template select<Q4_K_HALF, 2>(0);
            simd<float, Q4_K_HALF> in_odd  = iv.template select<Q4_K_HALF, 2>(1);

            // weight: 16 packed bytes -> low nibble = even K, high = odd K
            simd<uint8_t, Q4_K_HALF> raw =
                block_load<uint8_t, Q4_K_HALF>(w_row + k / 2);
            simd<uint16_t, Q4_K_HALF> u16 = convert<uint16_t>(raw);
            simd<float, Q4_K_HALF> nib_even = convert<float>(u16 & 0x000F);
            simd<float, Q4_K_HALF> nib_odd  = convert<float>((u16 >> 4) & 0x000F);

            // per-32-block scale + min (asymmetric): w = scale*nibble - min
            float s = static_cast<float>(s_row[group_idx]);
            float m = static_cast<float>(m_row[group_idx]);
            group_idx += 1;
            simd<float, Q4_K_HALF> w_even = nib_even * s - m;
            simd<float, Q4_K_HALF> w_odd  = nib_odd  * s - m;

            acc_even += in_even * w_even;
            acc_odd  += in_odd  * w_odd;
        }

        float my_sum = reduce<float>(acc_even, std::plus<>())
                     + reduce<float>(acc_odd,  std::plus<>());

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

inline void q4_k_gemv_host(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    const fp16* minv, fp16* output, uint32_t N, uint32_t K, sycl::queue& q) {
    int n_groups = K / Q4_K_GROUP;
    int ks;
    select_ks_q4_k(N, K, ks);
    int global = N * ks;
    int local = ks;

#define LAUNCH_Q4_K(S)                                                  \
    q.submit([&](sycl::handler& h) {                                    \
        h.parallel_for(sycl::nd_range<1>(global, local),                \
            Q4_K_gemv_kernel<S>{input, weight, scale, minv, output,     \
                                (int)N, (int)K, n_groups});             \
    });

    if      (ks == 1) { LAUNCH_Q4_K(1) }
    else if (ks == 2) { LAUNCH_Q4_K(2) }
    else if (ks == 4) { LAUNCH_Q4_K(4) }
    else if (ks == 8) { LAUNCH_Q4_K(8) }
    else              { LAUNCH_Q4_K(1) }
#undef LAUNCH_Q4_K
}
