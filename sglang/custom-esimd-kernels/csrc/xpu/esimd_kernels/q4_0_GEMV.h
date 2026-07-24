/* q4_0_GEMV.h — GGUF q4_0 INT4 GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * Adapted from int4_GEMV.h. Same INTERLEAVED nibble map as int4_GEMV.h
 * (byte j: low nibble -> elem 2j even, high -> elem 2j+1 odd), but with
 * group_size = 32 (q4_0 block) instead of 128.
 *
 *   * group_size = 32   (q4_0 block; int4_GEMV.h hardcodes 128)
 *   * nibble map = INTERLEAVED (matches the DPAS prefill GEMM's VNNI layout
 *     and the int4_GEMV deinterleave): byte j low->2j, high->2j+1.
 *   * dequant: w = (nibble - 8) * scale   (symmetric, scale may be negative)
 *
 * Consumes the GGUF->interleaved repacked layout (repack_q4_0_interleaved):
 *   input   [1, K]      fp16
 *   weight  [N, K/2]    uint8  (16 bytes per 32-block, blocks contiguous)
 *   scale   [N, K/32]   fp16   (per-block d, extracted out of the 18B block)
 *   output  [1, N]      fp16
 *
 * Decode is bandwidth-bound: per output row n we read K/2 weight bytes +
 * K/32 fp16 scales. q4_0's value is 0.5 byte/elem (1/4 of fp16).
 *
 * Included into esimd_kernel.sycl, which already pulls <sycl/sycl.hpp>,
 * <sycl/ext/intel/esimd.hpp>, `using namespace sycl::ext::intel::esimd;` and
 * the `fp16` typedef (via utils.h). Do NOT re-include / re-declare here.
 */
#pragma once

static constexpr int Q4_0_GROUP = 32;  // elements per q4_0 block
static constexpr int Q4_0_HALF = 16;   // qs bytes per block (= group/2)

// VL fixed at 32 (one q4_0 block per iter -> exactly one scale load/iter).
// K_SPLIT distributes the K reduction across threads; kp = K/K_SPLIT must stay
// a multiple of 32 so no block is split across threads.
inline void select_ks_q4_0(uint32_t N, uint32_t K, int& ks) {
    ks = 1;
    if      (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    int kp = K / ks;
    while ((kp % Q4_0_GROUP != 0) && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}

// Grid: N work-groups x K_SPLIT threads/WG. Each WG computes output[n];
// K_SPLIT threads split the K reduction, combined via SLM.
template <int K_SPLIT>
struct Q4_0_gemv_kernel {
    const fp16*    input;   // [1, K]
    const uint8_t* weight;  // [N, K/2]
    const fp16*    scale;   // [N, K/32]
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

        // Even/odd K-position accumulators (interleaved layout, like int4_GEMV).
        simd<float, Q4_0_HALF> acc_even = 0.0f;
        simd<float, Q4_0_HALF> acc_odd  = 0.0f;

        const uint8_t* w_row = weight + (size_t)n * (K / 2);
        const fp16*    s_row = scale  + (size_t)n * n_groups;
        int group_idx = kstart / Q4_0_GROUP;

        for (int k = kstart; k < kstart + kp; k += Q4_0_GROUP) {
            // input: 32 fp16 -> deinterleave into even[16] + odd[16]
            simd<fp16, Q4_0_GROUP> iv = block_load<fp16, Q4_0_GROUP>(input + k);
            simd<float, Q4_0_HALF> in_even = iv.template select<Q4_0_HALF, 2>(0);
            simd<float, Q4_0_HALF> in_odd  = iv.template select<Q4_0_HALF, 2>(1);

            // weight: 16 packed bytes -> low nibble = even K, high = odd K
            simd<uint8_t, Q4_0_HALF> raw =
                block_load<uint8_t, Q4_0_HALF>(w_row + k / 2);
            simd<uint16_t, Q4_0_HALF> u16 = convert<uint16_t>(raw);
            simd<float, Q4_0_HALF> w_even = convert<float>(u16 & 0x000F) - 8.0f;
            simd<float, Q4_0_HALF> w_odd  = convert<float>((u16 >> 4) & 0x000F) - 8.0f;

            // one fp16 scale per 32-block
            float s = static_cast<float>(s_row[group_idx]);
            group_idx += 1;

            acc_even += in_even * (w_even * s);
            acc_odd  += in_odd  * (w_odd  * s);
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

inline void q4_0_gemv_host(
    const fp16* input, const uint8_t* weight, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q) {
    int n_groups = K / Q4_0_GROUP;
    int ks;
    select_ks_q4_0(N, K, ks);
    int global = N * ks;
    int local = ks;

#define LAUNCH_Q4_0(S)                                                  \
    q.submit([&](sycl::handler& h) {                                    \
        h.parallel_for(sycl::nd_range<1>(global, local),                \
            Q4_0_gemv_kernel<S>{input, weight, scale, output,           \
                                (int)N, (int)K, n_groups});             \
    });

    if      (ks == 1) { LAUNCH_Q4_0(1) }
    else if (ks == 2) { LAUNCH_Q4_0(2) }
    else if (ks == 4) { LAUNCH_Q4_0(4) }
    else if (ks == 8) { LAUNCH_Q4_0(8) }
    else              { LAUNCH_Q4_0(1) }
#undef LAUNCH_Q4_0
}
