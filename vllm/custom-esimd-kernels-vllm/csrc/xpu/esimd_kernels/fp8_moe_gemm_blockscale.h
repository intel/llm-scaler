/* fp8_moe_gemm_blockscale.h — grouped (MoE) w8a16 block-scaled FP8 GEMM.
 *
 * The routed-expert analogue of fp8_GEMM_blockscale.h: computes, per expert e,
 *   output[t, :] = input[t, :] @ dequant(weight[e])^T   for tokens t of expert e
 * where weight[e] is fp8_e4m3 dequantized on the fly with a DeepSeek 128x128
 * block scale (weight_scale_inv[e, nb, kb]). The activation stays fp16 (w8a16,
 * matching the dense XPUFp8BlockScaledMMKernel) — no per-token-group act quant.
 *
 * Tokens are assumed pre-scattered/grouped by expert (as produced by
 * torch.ops._moe_C.remap_hidden_states): expert e owns the contiguous input
 * rows [expert_idx[e], expert_idx[e+1]).
 *
 * Layouts (all row-major, contiguous):
 *   input        [total_tokens, K]                fp16   (scattered, expert-grouped)
 *   weight       [E, N, K]                        uint8  (fp8_e4m3 bits)
 *   weight_scale [E, ceil(N/BN), ceil(K/BK)]      float32 (== weight_scale_inv)
 *   output       [total_tokens, N]                fp16   (pre-allocated)
 *   expert_idx   [E + 1]                          uint32 (token start offsets)
 *
 * Grid: E * N work-groups, one thread each (K_SPLIT=1). Since E*N is large the
 * occupancy comes from the grid, not from K-splitting. Each WG owns one output
 * channel n of one expert e; it streams the K row once per M-tile, dequantizes
 * + folds the per-128 block scale into the weight (as in the dense kernel), and
 * reduces a dot product for each of its expert's tokens. Experts with no tokens
 * return immediately. Bandwidth-bound, tuned for the small per-expert token
 * counts of decode; large-M prefill is functional (weight reloaded per M-tile).
 *
 * Active-experts-only grid: at decode only topk*num_tokens (<= total_tokens)
 * experts are non-empty, but there are E=256 of them, so an E*N grid launches
 * ~99% empty work-groups (pure dispatch overhead, which dominates the small
 * down-projection GEMV). A tiny prep kernel first compacts the non-empty expert
 * ids into active_experts[], and the main grid is launched over
 * min(E, total_tokens) * N work-groups: WG slot s handles expert
 * active_experts[s] (sentinel == num_experts for the unused tail slots, which
 * return early). total_tokens is known on the host (input rows), so no
 * device->host sync is needed.
 */
#pragma once

#include "fp8_GEMM_blockscale.h"  // fp8_blockscale::fp8e4m3_to_fp16
#include <algorithm>
#include <cstdint>

namespace fp8_moe_blockscale {

using fp8_blockscale::fp8e4m3_to_fp16;

// Compact the ids of experts that own >=1 token into active_experts[0..count).
// active_experts must be pre-filled with the sentinel value `num_experts` so
// the unused tail slots make their work-groups return early. Single thread
// (num_experts is small, e.g. 256); deterministic order, no atomics.
struct build_active_experts_kernel {
  const uint32_t* expert_idx;  // [E + 1]
  int32_t* active_experts;     // [bound] (pre-filled with num_experts)
  int num_experts;

  void operator()(sycl::id<1>) const {
    int idx = 0;
    for (int e = 0; e < num_experts; e++) {
      if (expert_idx[e + 1] > expert_idx[e]) {
        active_experts[idx++] = e;
      }
    }
  }
};

inline void build_active_experts(const uint32_t* expert_idx,
                                 int32_t* active_experts, int num_experts,
                                 sycl::queue& q) {
  q.submit([&](handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1),
                     build_active_experts_kernel{expert_idx, active_experts,
                                                 num_experts});
  });
}

// One WG per (active-expert slot, output-channel n). MAX_M tokens per inner tile.
template <int VL, int BK, int MAX_M>
struct moe_gemv_block_kernel {
  const fp16* input;             // [total_tokens, K]
  const uint8_t* weight;         // [E, N, K] fp8_e4m3 bits
  const float* wscale;           // [E, Nb, Kb]
  fp16* output;                  // [total_tokens, N]
  const uint32_t* expert_idx;    // [E + 1]
  const int32_t* active_experts; // [bound] compacted non-empty expert ids
  int N, K, Nb, Kb, num_experts;

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = item.get_group(0);
    const int slot = gid / N;
    const int n = gid - slot * N;
    const int e = active_experts[slot];   // sentinel num_experts for tail slots
    if (e >= num_experts) return;

    const uint32_t ts = expert_idx[e];
    const uint32_t te = expert_idx[e + 1];
    const int Me = (int)(te - ts);
    if (Me <= 0) return;

    const uint8_t* w_row = weight + ((size_t)e * N + n) * K;
    const float* s_row = wscale + ((size_t)e * Nb + (n / BK)) * Kb;

    for (int m0 = 0; m0 < Me; m0 += MAX_M) {
      simd<float, MAX_M> acc = 0.0f;

      for (int k = 0; k < K; k += VL) {
        // Load + dequant the weight slice once, fold in the per-128 block scale.
        simd<uint8_t, VL> raw = block_load<uint8_t, VL>(w_row + k);
        simd<float, VL> wf = fp8e4m3_to_fp16<VL>(raw);
#pragma unroll
        for (int sb = 0; sb < VL / BK; sb++) {
          const float sc = s_row[(k / BK) + sb];
          wf.template select<BK, 1>(sb * BK) =
              wf.template select<BK, 1>(sb * BK) * sc;
        }
        // Reuse the scaled weight slice across this tile's tokens.
#pragma unroll
        for (int mm = 0; mm < MAX_M; mm++) {
          if (m0 + mm < Me) {
            simd<fp16, VL> iv =
                block_load<fp16, VL>(input + (size_t)(ts + m0 + mm) * K + k);
            simd<float, VL> ivf = iv;
            acc[mm] += reduce<float>(ivf * wf, std::plus<>());
          }
        }
      }

#pragma unroll
      for (int mm = 0; mm < MAX_M; mm++) {
        if (m0 + mm < Me)
          output[(size_t)(ts + m0 + mm) * N + n] = fp16(acc[mm]);
      }
    }
  }
};

template <int VL, int MAX_M>
inline void launch_moe_gemv_block(const fp16* input, const uint8_t* weight,
                                  const float* wscale, fp16* output,
                                  const uint32_t* expert_idx,
                                  const int32_t* active_experts, int n_active,
                                  int N, int K, int Nb, int Kb, int num_experts,
                                  sycl::queue& q) {
  constexpr int BK = 128;
  moe_gemv_block_kernel<VL, BK, MAX_M> kern{
      input,      weight, wscale, output,      expert_idx,
      active_experts, N,  K,      Nb,          Kb,         num_experts};
  sycl::range<1> global(static_cast<size_t>(n_active) * N);
  sycl::range<1> local(1);
  q.submit([&](handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(global, local), kern);
  });
}

// Host launcher. block_n/block_k must be 128. Handles any per-expert token
// count by tiling into groups of MAX_M rows inside the kernel. `active_experts`
// (length >= n_active) holds the compacted non-empty expert ids (built via
// build_active_experts); n_active = min(num_experts, total_tokens) caps the
// grid to the experts that can be non-empty this call.
inline void moe_gemm_fp8_blockscale_host(
    const fp16* input, const uint8_t* weight, const float* weight_scale,
    fp16* output, const uint32_t* expert_idx, const int32_t* active_experts,
    int n_active, int N, int K, int num_experts, int block_n, int block_k,
    sycl::queue& q) {
  if (n_active <= 0) return;
  const int Nb = (N + block_n - 1) / block_n;
  const int Kb = (K + block_k - 1) / block_k;
  constexpr int MAX_M = 8;
  const int VL = (K % 256 == 0) ? 256 : 128;
  if (VL == 256)
    launch_moe_gemv_block<256, MAX_M>(input, weight, weight_scale, output,
                                      expert_idx, active_experts, n_active, N, K,
                                      Nb, Kb, num_experts, q);
  else
    launch_moe_gemv_block<128, MAX_M>(input, weight, weight_scale, output,
                                      expert_idx, active_experts, n_active, N, K,
                                      Nb, Kb, num_experts, q);
}

}  // namespace fp8_moe_blockscale
