#pragma once
#include "utils.h"

// ============================================================================
// INT4 GEMV with FP32 accumulation and per-group scale.
// Optimized for decode (M=1) on BMG XPU using ESIMD intrinsics.
//
// This is the INT4 counterpart of the FP8 GEMV kernel (fp8_GEMV_v2.h).
// It targets the Qwen3.5-122B-A10B model running on vLLM, where decode-phase
// GEMV (batch size = 1) is the dominant latency contributor.
//
// ---- Weight format ----
//
// Symmetric INT4, packed 2 values per uint8 byte:
//   byte[i] = (val[2i] & 0xF) | ((val[2i+1] & 0xF) << 4)
//   Low nibble  (bits 0-3) = element at even K-position (2i)
//   High nibble (bits 4-7) = element at odd  K-position (2i+1)
//
// Weight tensor shape: [N, K/2] uint8
//
// ---- Scale format ----
//
// Per-group fp16 scale, one scale per GROUP_SIZE (128) elements along K:
//   scale shape: [N, K / GROUP_SIZE]
//
// Dequantization formula (per element):
//   int4_val ∈ [-8, 7]  (4-bit two's complement)
//   fp16_weight = int4_val * group_scale
//
// ---- Key differences from FP8 GEMV (fp8_GEMV_v2.h) ----
//
// 1. Bandwidth: INT4 loads 0.5 bytes/element vs FP8's 1 byte/element.
//    For the same (N, K) shape, INT4 reads half the weight data — this is
//    the primary performance advantage, since decode GEMV is memory-bound.
//
// 2. Scale granularity: FP8 uses per-tensor scale (one float scalar),
//    applied AFTER the K-loop (a single multiply at the end).
//    INT4 uses per-group scale (one fp16 per 128 elements), applied INSIDE
//    the K-loop — one extra scalar load + broadcast multiply per iteration.
//
// 3. Unpacking: FP8 dequant is 3 bit-manipulation ops (shift, OR, merge).
//    INT4 unpack is shift + mask + sign-extend, applied to two halves
//    (even/odd nibbles) separately, with input deinterleaving to match.
//
// 4. Loop structure: Because each byte packs two adjacent K-elements,
//    the kernel loads VL/2 bytes per iteration (covering VL elements),
//    then deinterleaves both input and weight into even/odd halves of
//    size VL/2, accumulating into two separate FP32 accumulators for ILP.
//
// ---- Tensor shapes ----
//
// Input:  [1, K]            fp16
// Weight: [N, K/2]          uint8   (packed int4)
// Scale:  [N, K/GROUP_SIZE] fp16    (per-group)
// Output: [1, N]            fp16
//
// ---- Phase 1 constraints ----
//
// VL is fixed at 128 (= GROUP_SIZE), so each K-loop iteration processes
// exactly one scale group.  K_SPLIT varies (1/2/4/8) for parallelism.
// kp = K / K_SPLIT must be a multiple of 128 to avoid splitting a group
// across threads.
// ============================================================================

static constexpr int INT4_GROUP_SIZE = 128;


// ============================================================================
// INT4 unpacking:  uint8[VL/2]  →  two float[VL/2] vectors
//
// Each input byte contains two 4-bit signed integers:
//   low nibble  (bits 0-3) = weight at even K-position
//   high nibble (bits 4-7) = weight at odd  K-position
//
// Signed conversion (4-bit two's complement):
//   0..7   →  0..7    (positive, unchanged)
//   8..15  → -8..-1   (subtract 16)
//
// The caller must pair wf_even with even-indexed input elements, and
// wf_odd with odd-indexed input elements, to compute the correct dot product.
// ============================================================================

template<int VL>
SYCL_ESIMD_FUNCTION inline void int4_dequant(
    simd<uint8_t, VL/2> raw,
    simd<float, VL/2>& wf_even,
    simd<float, VL/2>& wf_odd) {

    // Widen to int16 for arithmetic.
    // (uint8 → uint16 first to avoid sign issues, then reinterpret as int16.)
    simd<int16_t, VL/2> u16 = convert<int16_t>(convert<uint16_t>(raw));

    // Low nibble → even-position weights.
    simd<int16_t, VL/2> lo = u16 & 0x000F;
    lo.merge(lo - 16, lo >= 8);   // unsigned [0,15] → signed [-8,7]
    wf_even = convert<float>(lo);

    // High nibble → odd-position weights.
    simd<int16_t, VL/2> hi = (u16 >> 4) & 0x000F;
    hi.merge(hi - 16, hi >= 8);
    wf_odd = convert<float>(hi);
}


// ============================================================================
// VL / K_SPLIT auto-selection for INT4 GEMV.
//
// Phase 1: VL is fixed at 128, matching GROUP_SIZE for natural alignment.
// Only K_SPLIT varies to distribute work within a workgroup:
//   - Each WG has K_SPLIT threads; thread `lid` handles K-range
//     [lid * kp, lid * kp + kp), where kp = K / K_SPLIT.
//   - kp MUST be a multiple of 128 (GROUP_SIZE) so that no scale group
//     is split across two threads (which would require shared scale access
//     and complicate the kernel).
//
// Heuristic: use higher K_SPLIT when N is small (few WGs → underutilized
// GPU) and K is large (more K-work to parallelize).
// ============================================================================

inline void select_vl_ks_int4(uint32_t N, uint32_t K, int& vl, int& ks) {
    vl = 128;  // Phase 1: always 128, aligned with GROUP_SIZE.
    ks = 1;

    // Small N + large K: increase K_SPLIT for intra-WG parallelism.
    if      (N <= 128 && K >= 2048) { ks = 8; }
    else if (N <= 512 && K >= 2048) { ks = 4; }

    // Enforce: kp = K / ks must be a multiple of 128 (GROUP_SIZE).
    // If not, halve ks until it is.
    int kp = K / ks;
    while (kp % INT4_GROUP_SIZE != 0 && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}


// ============================================================================
// Single INT4 GEMV kernel.
//
// Computes:  output[n] = sum_k( input[k] * dequant(weight[n, k]) )
//            where dequant(weight[n, k]) = int4_value * group_scale[n, k/128]
//
// Grid: N workgroups × K_SPLIT threads per WG.
//   - Each WG computes one output element (for row n of the weight matrix).
//   - Within a WG, K_SPLIT threads split the K-dimension reduction.
//   - When K_SPLIT > 1, partial sums are reduced via SLM (Shared Local Memory).
//
// K-loop (per thread):
//   Each iteration processes VL=128 elements of K:
//   1. Load 128 fp16 input values, deinterleave into even[64] + odd[64].
//   2. Load 64 packed uint8 bytes (= 128 int4 weights), unpack into
//      even[64] + odd[64] float vectors.
//   3. Load 1 per-group fp16 scale (VL=128 = GROUP_SIZE → 1 group/iter).
//   4. FMA: acc_even += in_even * wf_even * scale
//           acc_odd  += in_odd  * wf_odd  * scale
//   5. After loop: horizontal reduce acc_even + acc_odd → scalar output.
// ============================================================================

template<int VL, int K_SPLIT>
struct GEMV_int4_kernel {
    const fp16*    input;     // [1, K] fp16 — input activation vector
    const uint8_t* weight;    // [N, K/2] uint8 — packed INT4 weights
    const fp16*    scale;     // [N, K/GROUP_SIZE] fp16 — per-group scale
    fp16*          output;    // [1, N] fp16 — output vector
    int N, K;
    int n_groups;             // K / GROUP_SIZE (precomputed to avoid division)

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // Allocate SLM for K_SPLIT > 1 partial-sum reduction.
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);    // which output row (0..N-1)
        int lid = item.get_local_id(0); // thread index within WG (0..K_SPLIT-1)
        if (n >= N) return;

        int kp = K / K_SPLIT;           // K-elements per thread
        int ks = lid * kp;              // starting K-offset for this thread

        // Two VL/2-wide accumulators: one for even-position products, one for
        // odd-position products.  Separate accumulators give better ILP than a
        // single accumulator with sequential dependency between even/odd FMAs.
        simd<float, VL/2> acc_even = 0.0f;
        simd<float, VL/2> acc_odd  = 0.0f;

        // Row pointers for this output row n.
        const uint8_t* w_row = weight + (size_t)n * (K / 2);    // packed weight
        const fp16*    s_row = scale  + (size_t)n * n_groups;    // group scales

        // Track which group we're in (advances by VL/GROUP_SIZE per iteration).
        int group_idx = ks / INT4_GROUP_SIZE;

        for (int k = ks; k < ks + kp; k += VL) {
            // --- Load input: VL fp16 values (VL*2 = 256 bytes) ---
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);

            // Deinterleave input to match INT4 packing layout:
            //   iv_even = input[k+0], input[k+2], input[k+4], ...  (64 values)
            //   iv_odd  = input[k+1], input[k+3], input[k+5], ...  (64 values)
            // This pairs correctly with low/high nibble weights.
            simd<fp16, VL/2> iv_even_h = iv.template select<VL/2, 2>(0).read();
            simd<fp16, VL/2> iv_odd_h  = iv.template select<VL/2, 2>(1).read();
            simd<float, VL/2> in_even = iv_even_h;   // fp16 → float
            simd<float, VL/2> in_odd  = iv_odd_h;

            // --- Load weight: VL/2 packed bytes = VL int4 values (64 bytes) ---
            simd<uint8_t, VL/2> raw = block_load<uint8_t, VL/2>(w_row + k / 2);

            // --- Unpack to even/odd float vectors ---
            simd<float, VL/2> wf_even, wf_odd;
            int4_dequant<VL>(raw, wf_even, wf_odd);

            // --- Load per-group scale ---
            // With VL=128 and GROUP_SIZE=128, exactly 1 group per iteration.
            // Scale is fp16, convert to float for FMA precision.
            float gs = static_cast<float>(s_row[group_idx]);
            group_idx += VL / INT4_GROUP_SIZE;   // +1 when VL=GROUP_SIZE

            // --- FMA: accumulate input * weight * scale ---
            // Apply scale to weight first (scalar broadcast), then multiply by input.
            wf_even *= gs;
            wf_odd  *= gs;
            acc_even += in_even * wf_even;
            acc_odd  += in_odd  * wf_odd;
        }

        // --- Horizontal reduction: sum all VL/2 even + VL/2 odd lanes ---
        float my_sum = reduce<float>(acc_even, std::plus<>())
                     + reduce<float>(acc_odd,  std::plus<>());

        if constexpr (K_SPLIT == 1) {
            // Single thread per WG: write directly.
            output[n] = fp16(my_sum);
        } else {
            // Multiple threads: reduce partial sums via SLM.
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};


// ============================================================================
// Host dispatcher for single INT4 GEMV.
//
// Selects VL and K_SPLIT via select_vl_ks_int4(), then launches the
// appropriate template instantiation.  Phase 1: VL always 128.
//
// Parameters are raw byte pointers (matching the torch extension pattern):
//   input_data:  [1, K]            fp16
//   weight_data: [N, K/2]          uint8
//   scale_data:  [N, K/GROUP_SIZE] fp16
//   output_data: [1, N]            fp16
// ============================================================================

inline void GEMV_int4_host(
    uint8_t* input_data,
    uint8_t* weight_data,
    uint8_t* scale_data,
    uint8_t* output_data,
    uint32_t N,
    uint32_t K,
    sycl::queue& q) {

    auto* p_in  = reinterpret_cast<const fp16*>(input_data);
    auto* p_w   = reinterpret_cast<const uint8_t*>(weight_data);
    auto* p_sc  = reinterpret_cast<const fp16*>(scale_data);
    auto* p_out = reinterpret_cast<fp16*>(output_data);

    int n_groups = K / INT4_GROUP_SIZE;

    int vl, ks;
    select_vl_ks_int4(N, K, vl, ks);

    int global = N * ks;   // total threads = N workgroups × K_SPLIT threads/WG
    int local  = ks;       // threads per workgroup

    // Phase 1: VL is always 128.  Only K_SPLIT varies.
    #define LAUNCH_INT4(S) \
        q.submit([&](sycl::handler& h) { \
            h.parallel_for(sycl::nd_range<1>(global, local), \
                GEMV_int4_kernel<128, S>{ \
                    p_in, p_w, p_sc, p_out, (int)N, (int)K, n_groups}); \
        });

    if      (ks == 1) { LAUNCH_INT4(1) }
    else if (ks == 2) { LAUNCH_INT4(2) }
    else if (ks == 4) { LAUNCH_INT4(4) }
    else if (ks == 8) { LAUNCH_INT4(8) }
    else              { LAUNCH_INT4(1) }

    #undef LAUNCH_INT4
}


// ============================================================================
// Fused INT4 GEMV kernel: compute GEMV_COUNT independent GEMVs sharing the
// same input vector in a single kernel submission.
//
// Motivation: in the Qwen3.5 GatedDeltaNet layer, the input projection
// computes two GEMVs with the same hidden_states input:
//   qkvz = input @ in_proj_qkvz.weight^T    (N0, e.g. 3072)
//   ba   = input @ in_proj_ba.weight^T       (N1, e.g. 16)
// Fusing them saves one kernel launch overhead (~20-50 us on BMG).
//
// Grid layout: (N0 + N1 + ...) workgroups, each WG determines which matrix
// it belongs to via cumulative N sums (same pattern as fp8_GEMV_v2.h).
// ============================================================================

template<int VL, int K_SPLIT, int GEMV_COUNT>
struct GEMV_int4_fused_kernel {
    const fp16*    input;                     // [1, K] fp16 — shared input
    const uint8_t* weights[GEMV_COUNT];       // packed INT4 weight per matrix
    const fp16*    scales[GEMV_COUNT];        // per-group scale per matrix
    fp16*          outputs[GEMV_COUNT];       // output buffer per matrix
    int            N[GEMV_COUNT];             // output dimension per matrix
    int            N_cumsum[GEMV_COUNT];      // cumulative sum for WG→matrix mapping
    int            K;
    int            n_groups;                  // K / GROUP_SIZE

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int gid = item.get_group(0);
        int lid = item.get_local_id(0);

        // Determine which of the GEMV_COUNT matrices this WG belongs to,
        // and the local row index within that matrix.
        int mat_idx = 0;
        int local_n = gid;
        #pragma unroll
        for (int i = 0; i < GEMV_COUNT; i++) {
            if (gid < N_cumsum[i]) {
                mat_idx = i;
                local_n = (i == 0) ? gid : gid - N_cumsum[i - 1];
                break;
            }
        }

        const uint8_t* w_row = weights[mat_idx] + (size_t)local_n * (K / 2);
        const fp16*    s_row = scales[mat_idx]   + (size_t)local_n * n_groups;
        fp16*          o_ptr = outputs[mat_idx];
        int            n     = local_n;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL/2> acc_even = 0.0f;
        simd<float, VL/2> acc_odd  = 0.0f;

        int group_idx = ks / INT4_GROUP_SIZE;

        for (int k = ks; k < ks + kp; k += VL) {
            // Load + deinterleave input.
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<fp16, VL/2> iv_even_h = iv.template select<VL/2, 2>(0).read();
            simd<fp16, VL/2> iv_odd_h  = iv.template select<VL/2, 2>(1).read();
            simd<float, VL/2> in_even = iv_even_h;
            simd<float, VL/2> in_odd  = iv_odd_h;

            // Load + unpack weight.
            simd<uint8_t, VL/2> raw = block_load<uint8_t, VL/2>(w_row + k / 2);
            simd<float, VL/2> wf_even, wf_odd;
            int4_dequant<VL>(raw, wf_even, wf_odd);

            // Per-group scale + FMA.
            float gs = static_cast<float>(s_row[group_idx]);
            group_idx += VL / INT4_GROUP_SIZE;

            wf_even *= gs;
            wf_odd  *= gs;
            acc_even += in_even * wf_even;
            acc_odd  += in_odd  * wf_odd;
        }

        float my_sum = reduce<float>(acc_even, std::plus<>())
                     + reduce<float>(acc_odd,  std::plus<>());

        if constexpr (K_SPLIT == 1) {
            o_ptr[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                o_ptr[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};


// ============================================================================
// Host dispatcher for fused INT4 GEMV.
//
// Launches a single kernel for GEMV_COUNT matrices sharing the same input.
// The total workgroup count is sum(Ns[i]), and each WG figures out which
// matrix it belongs to via cumulative N sums.
// ============================================================================

template<int GEMV_COUNT>
inline void GEMV_int4_fused_host(
    uint8_t* input_data,
    uint8_t* weight_ptrs[GEMV_COUNT],
    uint8_t* scale_ptrs[GEMV_COUNT],
    uint8_t* output_ptrs[GEMV_COUNT],
    uint32_t Ns[GEMV_COUNT],
    uint32_t K,
    sycl::queue& q) {

    auto* p_in = reinterpret_cast<const fp16*>(input_data);

    uint32_t total_N = 0;
    for (int i = 0; i < GEMV_COUNT; i++) total_N += Ns[i];

    int n_groups = K / INT4_GROUP_SIZE;

    int vl, ks;
    select_vl_ks_int4(total_N, K, vl, ks);

    int global = total_N * ks;
    int local  = ks;

    #define LAUNCH_INT4_FUSED(S) \
        q.submit([&](sycl::handler& h) { \
            GEMV_int4_fused_kernel<128, S, GEMV_COUNT> kern; \
            kern.input = p_in; \
            kern.K = (int)K; \
            kern.n_groups = n_groups; \
            uint32_t cum = 0; \
            for (int i = 0; i < GEMV_COUNT; i++) { \
                kern.weights[i] = reinterpret_cast<const uint8_t*>(weight_ptrs[i]); \
                kern.scales[i]  = reinterpret_cast<const fp16*>(scale_ptrs[i]); \
                kern.outputs[i] = reinterpret_cast<fp16*>(output_ptrs[i]); \
                kern.N[i] = (int)Ns[i]; \
                cum += Ns[i]; \
                kern.N_cumsum[i] = (int)cum; \
            } \
            h.parallel_for(sycl::nd_range<1>(global, local), kern); \
        });

    if      (ks == 1) { LAUNCH_INT4_FUSED(1) }
    else if (ks == 2) { LAUNCH_INT4_FUSED(2) }
    else if (ks == 4) { LAUNCH_INT4_FUSED(4) }
    else if (ks == 8) { LAUNCH_INT4_FUSED(8) }
    else              { LAUNCH_INT4_FUSED(1) }

    #undef LAUNCH_INT4_FUSED
}
