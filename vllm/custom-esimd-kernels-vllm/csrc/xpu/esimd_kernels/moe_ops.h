#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace sycl::ext::intel::esimd;
using namespace sycl;
using fp16 = sycl::half;

namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace esimd_math = sycl::ext::intel::esimd;

// Helper: horizontal sum of simd vector
template<typename T, int N>
SYCL_ESIMD_FUNCTION inline T h_sum(simd<T, N> v) {
    if constexpr (N == 1) return v[0];
    else {
        auto lo = v.template select<N/2, 1>(0);
        auto hi = v.template select<N/2, 1>(N/2);
        simd<T, N/2> sum = lo + hi;
        return h_sum<T, N/2>(sum);
    }
}

// Helper: horizontal max of simd vector
template<typename T, int N>
SYCL_ESIMD_FUNCTION inline T h_max(simd<T, N> v) {
    if constexpr (N == 1) {
        T r = v[0];
        return r;
    } else if constexpr (N == 2) {
        T a = v[0], b = v[1];
        return (a > b) ? a : b;
    } else {
        simd<T, N/2> lo = v.template select<N/2, 1>(0);
        simd<T, N/2> hi = v.template select<N/2, 1>(N/2);
        simd<T, N/2> mx = esimd_math::max<T, N/2>(lo, hi);
        return h_max<T, N/2>(mx);
    }
}

// Helper: horizontal min of simd vector
template<typename T, int N>
SYCL_ESIMD_FUNCTION inline T h_min(simd<T, N> v) {
    if constexpr (N == 1) {
        T r = v[0];
        return r;
    } else if constexpr (N == 2) {
        T a = v[0], b = v[1];
        return (a < b) ? a : b;
    } else {
        simd<T, N/2> lo = v.template select<N/2, 1>(0);
        simd<T, N/2> hi = v.template select<N/2, 1>(N/2);
        simd<T, N/2> mn = esimd_math::min<T, N/2>(lo, hi);
        return h_min<T, N/2>(mn);
    }
}

// ============================================================================
// MoE Auxiliary Ops: TopK, Scatter, SiLU_and_Mul, Gather
//
// These surround the FP8 MoE GEMM to form the full MoE forward pass:
//   TopK → Scatter → gate_up_GEMM → SiLU_and_Mul → down_GEMM → Gather
// ============================================================================

// ======================== TopK V2 Kernel (configurable experts/topk) ========================
// Uses h_max + >= comparison + integer-index zeroing to avoid float == bugs.
// No C arrays of simd types — uses flat simd<float, NUM_EXPERTS> to avoid spill issues.

// Argmax + zero: uses h_max for value, integer bit-cast for matching.
// The bit-cast comparison bypasses -ffast-math float comparison issues.
template<int C>
SYCL_ESIMD_FUNCTION inline void chunk_argmax_and_zero(
    simd<float, C>& vals, int base_idx,
    float& out_val, int32_t& out_idx)
{
    float mx = h_max<float, C>(vals);

    // Bit-cast float max to int32 for exact bit-pattern comparison
    // This is immune to -ffast-math reordering of float comparisons
    union { float f; int32_t i; } mx_bits;
    mx_bits.f = mx;

    simd<int32_t, C> vals_as_int = vals.template bit_cast_view<int32_t>().read();
    simd_mask<C> mask = (vals_as_int == mx_bits.i);

    // Pick lowest index among matches
    simd<int32_t, C> cand(999999);
    cand.merge(simd<int32_t, C>(base_idx, 1), mask);
    int32_t found_idx = h_min<int32_t, C>(cand);

    out_val = mx;
    out_idx = found_idx;

    // Zero exactly one element by integer index (always exact)
    int local_pos = found_idx - base_idx;
    simd_mask<C> zero_mask = (simd<int32_t, C>(0, 1) == local_pos);
    vals.merge(simd<float, C>(-1.0f), zero_mask);
}

template<int NUM_EXPERTS, int TOPK>
struct MoE_TopK_V2_Kernel {
    const fp16* router_logits;   // [T, NUM_EXPERTS]
    fp16* top_values;            // [T, TOPK]
    int32_t* top_indices;        // [T, TOPK]
    int T;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        static_assert(NUM_EXPERTS % 64 == 0, "NUM_EXPERTS must be multiple of 64");
        static_assert(NUM_EXPERTS <= 512, "NUM_EXPERTS must be <= 512");
        static_assert(TOPK <= 16, "TOPK must be <= 16");
        constexpr int C = 64;  // chunk size

        int row = item.get_group(0);
        if (row >= T) return;

        const fp16* row_ptr = router_logits + (size_t)row * NUM_EXPERTS;
        constexpr int N_CHUNKS = NUM_EXPERTS / C;

        // ── Load into C array of simd chunks ──
        simd<float, C> probs[N_CHUNKS];
        #pragma unroll
        for (int c = 0; c < N_CHUNKS; c++) {
            simd<fp16, C> raw = block_load<fp16, C>(row_ptr + c * C);
            probs[c] = simd<float, C>(raw);
        }

        // ── Softmax: global max ──
        float row_max = h_max<float, C>(probs[0]);
        #pragma unroll
        for (int c = 1; c < N_CHUNKS; c++) {
            float m = h_max<float, C>(probs[c]);
            if (m > row_max) row_max = m;
        }

        // ── Softmax: exp(x - max) + sum ──
        float total = 0.0f;
        #pragma unroll
        for (int c = 0; c < N_CHUNKS; c++) {
            probs[c] -= row_max;
            probs[c] = esimd_math::exp<float, C>(probs[c]);
            total += h_sum<float, C>(probs[c]);
        }

        // ── Softmax: normalize ──
        float inv = 1.0f / total;
        #pragma unroll
        for (int c = 0; c < N_CHUNKS; c++) probs[c] *= inv;

        // ── TopK selection: TOPK rounds ──
        float   tv[16];
        int32_t ti[16];

        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            // Find which chunk has the global max
            float bv = -1.0f;
            int   bc = 0;
            #pragma unroll
            for (int c = 0; c < N_CHUNKS; c++) {
                float m = h_max<float, C>(probs[c]);
                if (m > bv) { bv = m; bc = c; }
            }

            // Argmax + zero within winning chunk
            float fv; int32_t fi;
            chunk_argmax_and_zero<C>(probs[bc], bc * C, fv, fi);

            tv[k] = fv;
            ti[k] = fi;
        }

        // ── Normalize top-K ──
        float top_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOPK; k++) top_sum += tv[k];
        float inv_top = 1.0f / top_sum;
        #pragma unroll
        for (int k = 0; k < TOPK; k++) tv[k] *= inv_top;

        // ── Store ──
        fp16* vp = top_values + (size_t)row * TOPK;
        int32_t* ip = top_indices + (size_t)row * TOPK;

        // block_store first 8
        if constexpr (TOPK >= 8) {
            simd<fp16, 8> sv8;
            simd<int32_t, 8> si8;
            #pragma unroll
            for (int i = 0; i < 8; i++) { sv8[i] = (fp16)tv[i]; si8[i] = ti[i]; }
            block_store<fp16, 8>(vp, sv8);
            block_store<int32_t, 8>(ip, si8);
        }
        // Store remainder individually
        if constexpr (TOPK > 8) {
            #pragma unroll
            for (int i = 8; i < TOPK; i++) {
                simd<fp16, 1> sv((fp16)tv[i]);
                simd<int32_t, 1> si(ti[i]);
                block_store<fp16, 1>(vp + i, sv);
                block_store<int32_t, 1>(ip + i, si);
            }
        }
        if constexpr (TOPK < 8) {
            #pragma unroll
            for (int i = 0; i < TOPK; i++) {
                simd<fp16, 1> sv((fp16)tv[i]);
                simd<int32_t, 1> si(ti[i]);
                block_store<fp16, 1>(vp + i, sv);
                block_store<int32_t, 1>(ip + i, si);
            }
        }
    }
};

template<int NUM_EXPERTS, int TOPK>
inline void moe_topk_v2_host(
    const fp16* logits, fp16* values, int32_t* indices,
    int T, sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)T}, {1}),
            MoE_TopK_V2_Kernel<NUM_EXPERTS, TOPK>{logits, values, indices, T});
    });
}

// ======================== CPU Preprocessing ========================

// Build sorted_token_ids from router_indices.
// sorted_token_ids[dest_pos] = t * topk + k, grouped by expert.
// expert_idx[e] = start offset of expert e's tokens in sorted array.
inline void build_sorted_token_ids(
    const int32_t* router_indices,  // [T, topk]
    int32_t* sorted_token_ids,      // [T*topk] output
    uint32_t* expert_idx,           // [num_experts+1] output
    int T, int topk, int num_experts)
{
    std::vector<int> count(num_experts, 0);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < topk; k++)
            count[router_indices[t * topk + k]]++;

    expert_idx[0] = 0;
    for (int e = 0; e < num_experts; e++)
        expert_idx[e + 1] = expert_idx[e] + count[e];

    std::vector<int> pos(num_experts);
    for (int e = 0; e < num_experts; e++) pos[e] = expert_idx[e];
    for (int t = 0; t < T; t++)
        for (int k = 0; k < topk; k++) {
            int e = router_indices[t * topk + k];
            sorted_token_ids[pos[e]++] = t * topk + k;
        }
}

// Build reverse map: topk_ids[t*topk + k] = dest_pos in scattered array.
inline void build_topk_ids(
    const int32_t* sorted_token_ids,
    int32_t* topk_ids,   // [T*topk] output
    int total, int topk)
{
    for (int i = 0; i < total; i++) {
        topk_ids[sorted_token_ids[i]] = i;
    }
}

// ======================== TopK Kernel ========================
// Fused softmax + top-8 + normalize.
// Input: router_logits [T, num_experts] fp16
// Output: top_values [T, topk] fp16, top_indices [T, topk] int32

struct MoE_TopK_Kernel {
    const fp16* router_logits;  // [T, 128]
    fp16* top_values;           // [T, 8]
    int32_t* top_indices;       // [T, 8]
    int T;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int NUM_EXPERTS = 128;
        constexpr int TOPK = 8;

        int row = item.get_group(0);
        if (row >= T) return;

        // Load 128 fp16 → fp32
        simd<fp16, 64> raw_lo = block_load<fp16, 64>(router_logits + row * NUM_EXPERTS);
        simd<fp16, 64> raw_hi = block_load<fp16, 64>(router_logits + row * NUM_EXPERTS + 64);
        simd<float, 64> logits_lo(raw_lo);
        simd<float, 64> logits_hi(raw_hi);

        // Softmax step 1: find max
        float max_lo = h_max<float, 64>(logits_lo);
        float max_hi = h_max<float, 64>(logits_hi);
        float row_max = (max_lo > max_hi) ? max_lo : max_hi;

        // Softmax step 2: exp(x - max)
        logits_lo -= row_max;
        logits_hi -= row_max;
        simd<float, 64> probs_lo = esimd_math::exp<float, 64>(logits_lo);
        simd<float, 64> probs_hi = esimd_math::exp<float, 64>(logits_hi);

        // Softmax step 3: normalize
        float sum_lo = h_sum<float, 64>(probs_lo);
        float sum_hi = h_sum<float, 64>(probs_hi);
        float inv_sum = 1.0f / (sum_lo + sum_hi);
        probs_lo *= inv_sum;
        probs_hi *= inv_sum;

        // Index vectors for vectorized TopK — iota (no scalar writes)
        simd<int32_t, 64> idx_lo(0, 1);     // {0, 1, 2, ..., 63}
        simd<int32_t, 64> idx_hi(64, 1);    // {64, 65, ..., 127}

        simd<float, TOPK> top_vals;
        simd<int32_t, TOPK> top_idx;

        // TopK: 8 rounds — bit-cast comparison to bypass -ffast-math
        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            float mx_lo = h_max<float, 64>(probs_lo);
            float mx_hi = h_max<float, 64>(probs_hi);

            if (mx_lo >= mx_hi) {
                top_vals[k] = mx_lo;
                // Bit-cast comparison: immune to -ffast-math
                union { float f; int32_t i; } mx_bits;
                mx_bits.f = mx_lo;
                simd<int32_t, 64> lo_bits = probs_lo.template bit_cast_view<int32_t>().read();
                simd_mask<64> mask = (lo_bits == mx_bits.i);
                simd<int32_t, 64> cand = 999;
                cand.merge(idx_lo, mask);
                top_idx[k] = h_min<int32_t, 64>(cand);
                // Zero exactly one element by integer index
                int local_pos = top_idx[k];
                simd_mask<64> zmask = (idx_lo == local_pos);
                probs_lo.merge(simd<float, 64>(-1.0f), zmask);
            } else {
                top_vals[k] = mx_hi;
                union { float f; int32_t i; } mx_bits;
                mx_bits.f = mx_hi;
                simd<int32_t, 64> hi_bits = probs_hi.template bit_cast_view<int32_t>().read();
                simd_mask<64> mask = (hi_bits == mx_bits.i);
                simd<int32_t, 64> cand = 999;
                cand.merge(idx_hi, mask);
                top_idx[k] = h_min<int32_t, 64>(cand);
                int local_pos = top_idx[k];
                simd_mask<64> zmask = (idx_hi == local_pos);
                probs_hi.merge(simd<float, 64>(-1.0f), zmask);
            }
        }

        // Normalize top-8
        float top_sum = h_sum<float, TOPK>(top_vals);
        top_vals *= (1.0f / top_sum);

        // Store
        simd<fp16, TOPK> out_vals = convert<fp16, float, TOPK>(top_vals);
        block_store<fp16, TOPK>(top_values + row * TOPK, out_vals);
        block_store<int32_t, TOPK>(top_indices + row * TOPK, top_idx);
    }
};

inline void moe_topk_host(
    const fp16* logits, fp16* values, int32_t* indices,
    int T, sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)T}, {1}),
            MoE_TopK_Kernel{logits, values, indices, T});
    });
}

// ======================== Scatter Kernel ========================
// Reorder hidden_states [T, K] → scattered [T*topk, K] by expert grouping.
// Also scatter router weights.

struct MoE_Scatter_Kernel {
    const fp16* hidden_states;      // [T, K]
    const fp16* router_top_value;   // [T, topk]
    const int32_t* sorted_token_ids; // [T*topk]
    fp16* scattered_hidden;         // [T*topk, K]
    fp16* scattered_weights;        // [T*topk]
    int K, topk;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int dest_pos = item.get_group(0);

        simd<int32_t, 1> enc = block_load<int32_t, 1>(sorted_token_ids + dest_pos);
        int encoded = enc[0];
        int src_row = encoded / topk;
        int slot    = encoded % topk;

        // Copy K fp16 in chunks of 128
        for (int off = 0; off < K; off += 128) {
            simd<fp16, 128> v = block_load<fp16, 128>(
                hidden_states + (size_t)src_row * K + off);
            block_store<fp16, 128>(
                scattered_hidden + (size_t)dest_pos * K + off, v);
        }

        // Scatter weight
        simd<fp16, 1> wv = block_load<fp16, 1>(router_top_value + src_row * topk + slot);
        block_store<fp16, 1>(scattered_weights + dest_pos, wv);
    }
};

inline void moe_scatter_host(
    const fp16* hidden, const fp16* top_values,
    const int32_t* sorted_ids,
    fp16* scattered_hidden, fp16* scattered_weights,
    int K, int topk, int total_expanded,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)total_expanded}, {1}),
            MoE_Scatter_Kernel{hidden, top_values, sorted_ids,
                               scattered_hidden, scattered_weights,
                               K, topk});
    });
}

// ======================== Fused Scatter (GPU-only, no CPU preprocessing) ========================
// Replaces CPU build_sorted_token_ids + old Scatter kernel.
// Uses atomic counting (IPEX pattern) to build expert grouping entirely on GPU.
//
// Phase 1 (Init):  Per-token, atomic_add on expert counts → per-slot offset
// Phase 2 (Prefix): Prefix-sum counts → expert_start[0..num_experts]
// Phase 3 (Copy):  Per-token, copy hidden + weights + build topk_ids reverse map

struct MoE_Scatter_Init_Kernel {
    const int32_t* top_indices;         // [T, topk]
    int32_t* experts_token_count;       // [num_experts] — zero-initialized
    int32_t* token_to_scatter_offset;   // [T*topk] output
    int T, topk;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int t = item.get_group(0);
        if (t >= T) return;

        // Process each top-k slot individually (supports any topk value)
        for (int k = 0; k < topk; k++) {
            int expert_id = *(top_indices + (size_t)t * topk + k);

            simd<uint32_t, 1> byte_off((uint32_t)expert_id * (uint32_t)sizeof(int32_t));
            simd<int32_t, 1> val(1);
            simd<int32_t, 1> old = atomic_update<atomic_op::add>(
                experts_token_count, byte_off, val, simd_mask<1>(1));

            *(token_to_scatter_offset + (size_t)t * topk + k) = old[0];
        }
    }
};

struct MoE_Scatter_Prefix_Kernel {
    const int32_t* experts_token_count; // [num_experts]
    uint32_t* expert_start;             // [num_experts+1] output
    int32_t* max_tokens_out;            // [1] output
    int num_experts;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        uint32_t running_sum = 0;
        int32_t max_count = 0;

        for (int base = 0; base < num_experts; base += 32) {
            simd<int32_t, 32> counts = block_load<int32_t, 32>(
                experts_token_count + base);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int32_t c = counts[i];
                if (c > max_count) max_count = c;
                simd<uint32_t, 1> val = running_sum;
                block_store<uint32_t, 1>(expert_start + base + i, val);
                running_sum += (uint32_t)c;
            }
        }
        simd<uint32_t, 1> total_val = running_sum;
        block_store<uint32_t, 1>(expert_start + num_experts, total_val);
        simd<int32_t, 1> max_val = max_count;
        block_store<int32_t, 1>(max_tokens_out, max_val);
    }
};

struct MoE_Scatter_Copy_Kernel {
    const fp16* hidden_states;          // [T, K]
    const fp16* top_values;             // [T, topk] — routing weights
    const int32_t* top_indices;         // [T, topk] — expert IDs
    const int32_t* token_to_scatter_offset; // [T*topk]
    const uint32_t* expert_start;       // [num_experts+1]
    fp16* scattered_hidden;             // [T*topk, K]
    fp16* scattered_weights;            // [T*topk]
    int32_t* topk_ids;                  // [T*topk] output — reverse map for Gather
    int K, topk;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int t = item.get_group(0);

        for (int k = 0; k < topk; k++) {
            int expert_id = *(top_indices + (size_t)t * topk + k);
            int offset = *(token_to_scatter_offset + (size_t)t * topk + k);

            simd<uint32_t, 1> es = block_load<uint32_t, 1>(expert_start + expert_id);
            int dp = (int)es[0] + offset;

            // Copy hidden states
            for (int off = 0; off < K; off += 128) {
                simd<fp16, 128> v = block_load<fp16, 128>(
                    hidden_states + (size_t)t * K + off);
                block_store<fp16, 128>(
                    scattered_hidden + (size_t)dp * K + off, v);
            }

            // Scatter weight
            fp16 w = *(top_values + (size_t)t * topk + k);
            simd<fp16, 1> wv;
            wv[0] = w;
            block_store<fp16, 1>(scattered_weights + dp, wv);

            // Build reverse map
            simd<int32_t, 1> dp_val;
            dp_val[0] = dp;
            block_store<int32_t, 1>(topk_ids + (size_t)t * topk + k, dp_val);
        }
    }
};

inline void moe_scatter_fused_host(
    const fp16* hidden, const fp16* top_values, const int32_t* top_indices,
    fp16* scattered_hidden, fp16* scattered_weights,
    int32_t* topk_ids, uint32_t* expert_start, int32_t* max_tokens_out,
    int32_t* experts_token_count, int32_t* token_to_scatter_offset,
    int K, int topk, int T, int num_experts,
    sycl::queue& q)
{
    // Zero-initialize expert counts
    q.memset(experts_token_count, 0, num_experts * sizeof(int32_t));

    // Kernel 1: Atomic counting — T WGs
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)T}, {1}),
            MoE_Scatter_Init_Kernel{top_indices, experts_token_count,
                                     token_to_scatter_offset, T, topk});
    });

    // Kernel 2: Prefix-sum — 1 WG, 1 thread
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({1}, {1}),
            MoE_Scatter_Prefix_Kernel{experts_token_count, expert_start,
                                       max_tokens_out, num_experts});
    });

    // Kernel 3: Copy + build topk_ids — T WGs
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)T}, {1}),
            MoE_Scatter_Copy_Kernel{hidden, top_values, top_indices,
                                     token_to_scatter_offset, expert_start,
                                     scattered_hidden, scattered_weights,
                                     topk_ids, K, topk});
    });
}

// ======================== SiLU_and_Mul Kernel ========================
// Input: [rows, N_gate_up] fp16, first N_half is gate, second N_half is up.
// Output: [rows, N_half] fp16 = SiLU(gate) * up
// N_gate_up = 384, N_half = 192 for Qwen3-Next TP4.

struct MoE_SiLU_Mul_Kernel {
    const fp16* input;   // [rows, N_gate_up]
    fp16* output;        // [rows, N_half]
    int N_gate_up;       // 384
    int N_half;          // 192
    int total_rows;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int row = item.get_group(0);
        if (row >= total_rows) return;

        const fp16* row_in = input + (size_t)row * N_gate_up;
        fp16* row_out = output + (size_t)row * N_half;

        // Process in chunks of 64 (192 = 3 × 64)
        for (int off = 0; off < N_half; off += 64) {
            simd<fp16, 64> gate_h = block_load<fp16, 64>(row_in + off);
            simd<fp16, 64> up_h   = block_load<fp16, 64>(row_in + N_half + off);

            simd<float, 64> gate = convert<float, fp16, 64>(gate_h);
            simd<float, 64> up   = convert<float, fp16, 64>(up_h);

            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            simd<float, 64> neg_gate = -gate;
            simd<float, 64> exp_neg = esimd_math::exp<float, 64>(neg_gate);
            simd<float, 64> sigmoid = 1.0f / (1.0f + exp_neg);
            simd<float, 64> silu = gate * sigmoid;
            simd<float, 64> result = silu * up;

            simd<fp16, 64> out = convert<fp16, float, 64>(result);
            block_store<fp16, 64>(row_out + off, out);
        }
    }
};

inline void moe_silu_mul_host(
    const fp16* input, fp16* output,
    int N_gate_up, int N_half, int total_rows,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)total_rows}, {1}),
            MoE_SiLU_Mul_Kernel{input, output, N_gate_up, N_half, total_rows});
    });
}

// ======================== Gather Kernel ========================
// Weighted reduce: moe_output [T*topk, K] → final_hidden [T, K]
// For each token t: final[t] = sum_k(moe_out[topk_ids[t,k]] * weight[topk_ids[t,k]])

struct MoE_Gather_Kernel {
    const fp16* moe_output;         // [T*topk, K]
    const int32_t* topk_ids;        // [T, topk] → positions in scattered array
    const fp16* scattered_weights;  // [T*topk]
    fp16* final_hidden;             // [T, K]
    int K, topk;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int t = item.get_group(0);
        constexpr int CHUNK = 128;
        constexpr int MAX_TOPK = 32;

        // Load positions and weights (scalar loop, supports any topk)
        int ids[MAX_TOPK];
        float wts[MAX_TOPK];
        for (int k = 0; k < topk; k++) {
            ids[k] = *(topk_ids + (size_t)t * topk + k);
            simd<fp16, 1> wv = block_load<fp16, 1>(scattered_weights + ids[k]);
            fp16 w_scalar = wv[0];
            wts[k] = (float)w_scalar;
        }

        // Iterate K chunks: accumulate topk experts per chunk, store immediately
        for (int off = 0; off < K; off += CHUNK) {
            simd<float, CHUNK> acc = 0.0f;
            for (int k = 0; k < topk; k++) {
                float wk = wts[k];
                simd<fp16, CHUNK> row = block_load<fp16, CHUNK>(
                    moe_output + (size_t)ids[k] * K + off);
                acc += convert<float, fp16, CHUNK>(row) * wk;
            }
            simd<fp16, CHUNK> out = convert<fp16, float, CHUNK>(acc);
            block_store<fp16, CHUNK>(final_hidden + (size_t)t * K + off, out);
        }
    }
};

inline void moe_gather_host(
    const fp16* moe_output, const int32_t* topk_ids,
    const fp16* scattered_weights, fp16* final_hidden,
    int K, int topk, int T,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>({(size_t)T}, {1}),
            MoE_Gather_Kernel{moe_output, topk_ids, scattered_weights,
                              final_hidden, K, topk});
    });
}
