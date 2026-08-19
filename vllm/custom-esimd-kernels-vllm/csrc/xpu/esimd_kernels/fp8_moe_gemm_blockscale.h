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
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

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

// Build one descriptor for every contiguous tile of up to 64 expert-grouped
// rows.  The descriptor count is bounded on the host by ceil(total_rows/64)+E,
// so the large-M kernel needs no device-to-host synchronization to discover the
// maximum number of tokens routed to one expert.
struct build_expert_m_tiles_kernel {
  const uint32_t* expert_idx;
  int32_t* tile_experts;
  int32_t* tile_rows;
  int num_experts;
  int tile_capacity;

  void operator()(sycl::id<1>) const {
    int tile = 0;
    for (int e = 0; e < num_experts; e++) {
      const int start = (int)expert_idx[e];
      const int end = (int)expert_idx[e + 1];
      for (int row = start; row < end; row += 64) {
        tile_experts[tile] = e;
        tile_rows[tile] = row;
        tile++;
      }
    }
    // Fill unused descriptors here so the caller does not need a separate
    // tensor-fill submission. The prefill kernel checks this sentinel before
    // reading tile_rows.
    for (; tile < tile_capacity; tile++) {
      tile_experts[tile] = num_experts;
    }
  }
};

inline void build_expert_m_tiles(const uint32_t* expert_idx,
                                 int32_t* tile_experts,
                                 int32_t* tile_rows, int num_experts,
                                 int tile_capacity, sycl::queue& q) {
  q.submit([&](handler& cgh) {
    cgh.parallel_for(sycl::range<1>(1),
                     build_expert_m_tiles_kernel{expert_idx, tile_experts,
                                                 tile_rows, num_experts,
                                                 tile_capacity});
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

// Large-M grouped W8A16 GEMM.  Each work-group owns one 64-row expert tile and
// one group of 16-column N tiles.  FP8 E4M3 weights are converted to FP16,
// multiplied by their 128x128 scale, then consumed by FP16 DPAS.  This keeps
// the platform restriction (no FP8 matrix multiply) while replacing the
// scalar output-channel reductions used by the decode-tuned GEMV.
struct moe_gemm_block_prefill_kernel {
  const fp16* input;
  const uint8_t* weight;
  const float* wscale;
  fp16* output;
  const uint32_t* expert_idx;
  const int32_t* tile_experts;
  const int32_t* tile_rows;
  int total_tokens;
  int tile_capacity;
  int N, K, Nb, Kb, num_experts;
  int n_wg_count, n_per_wg;

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    namespace mem = sycl::ext::intel::experimental::esimd;
    const int wg = item.get_group(0);
    const int tile_id = wg / n_wg_count;
    const int n_wg_id = wg - tile_id * n_wg_count;
    if (tile_id >= tile_capacity) return;
    const int e = tile_experts[tile_id];
    if (e < 0 || e >= num_experts) return;

    const int row_start = tile_rows[tile_id];
    const int row_end = (int)expert_idx[e + 1];
    const int valid_rows = row_end - row_start < 64 ? row_end - row_start : 64;
    if (valid_rows <= 0) return;

    const int n_tiles = (N + 15) / 16;
    const int n_tile_start = n_wg_id * n_per_wg;
    const int n_tile_end = n_tile_start + n_per_wg < n_tiles
                               ? n_tile_start + n_per_wg
                               : n_tiles;

    const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
    const uint32_t surfH_A = (uint32_t)total_tokens - 1u;
    mem::config_2d_mem_access<fp16, 16, 8, 1> payA(
        input, surfW_A, surfH_A, surfW_A, 0u, (uint32_t)row_start);

    const uint32_t surfW_B = (uint32_t)K - 1u;
    const uint32_t surfH_B = (uint32_t)(num_experts * N) - 1u;
    mem::config_2d_mem_access<uint32_t, 4, 16, 1> payB_t(
        reinterpret_cast<const uint32_t*>(weight), surfW_B, surfH_B, surfW_B,
        0u, 0u);

    for (int nc = n_tile_start; nc < n_tile_end; nc++) {
      const int n_start = nc * 16;
      if (n_start >= N) break;
      const int n_valid = N - n_start < 16 ? N - n_start : 16;
      payB_t.set_y((uint32_t)(e * N + n_start));
      simd<float, 128> acc[8];
#pragma unroll
      for (int mt = 0; mt < 8; mt++) acc[mt] = 0.0f;

      for (int k_base = 0; k_base < K; k_base += 64) {
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
          const int k_sub = k_base + sub * 16;
          payB_t.set_x((uint32_t)(k_sub / 4));
          simd<uint32_t, 64> w_t =
              mem::lsc_load_2d<uint32_t, 4, 16, 1, true, false,
                               mem::cache_hint::cached,
                               mem::cache_hint::cached>(payB_t);

          // Transposed 16x16 FP8 tile -> DPAS VNNI2 FP16 layout.
          simd<uint8_t, 256> raw_vnni;
#pragma unroll
          for (int col = 0; col < 4; col++) {
            simd<uint32_t, 16> g = w_t.template select<16, 1>(col * 16);
            simd<uint8_t, 16> b0 = convert<uint8_t>(g & 0xFF);
            simd<uint8_t, 16> b1 = convert<uint8_t>((g >> 8) & 0xFF);
            simd<uint8_t, 16> b2 = convert<uint8_t>((g >> 16) & 0xFF);
            simd<uint8_t, 16> b3 = convert<uint8_t>((g >> 24) & 0xFF);
            raw_vnni.template select<16, 2>(col * 64) = b0;
            raw_vnni.template select<16, 2>(col * 64 + 1) = b1;
            raw_vnni.template select<16, 2>(col * 64 + 32) = b2;
            raw_vnni.template select<16, 2>(col * 64 + 33) = b3;
          }
          simd<fp16, 256> b_tile = fp8e4m3_to_fp16<256>(raw_vnni);
          const float scale =
              wscale[((size_t)e * Nb + n_start / 128) * Kb + k_sub / 128];
          b_tile *= fp16(scale);

          payA.set_x((uint32_t)k_sub);
#pragma unroll
          for (int mt = 0; mt < 8; mt++) {
            payA.set_y((uint32_t)(row_start + mt * 8));
            simd<fp16, 128> a =
                mem::lsc_load_2d<fp16, 16, 8, 1, false, false,
                                 mem::cache_hint::cached,
                                 mem::cache_hint::cached>(payA);
            acc[mt] = sycl::ext::intel::esimd::xmx::dpas<
                8, 8, float, float, fp16, fp16>(acc[mt], b_tile, a);
          }
        }
      }

#pragma unroll
      for (int mt = 0; mt < 8; mt++) {
#pragma unroll
        for (int mi = 0; mi < 8; mi++) {
          const int row_id = mt * 8 + mi;
          if (row_id < valid_rows) {
            simd<float, 16> row = acc[mt].template select<16, 1>(mi * 16);
            simd<fp16, 16> out = convert<fp16>(row);
            const size_t off = (size_t)(row_start + row_id) * N + n_start;
            if (n_valid == 16) {
              block_store<fp16, 16>(output + off, out);
            } else {
              for (int ni = 0; ni < n_valid; ni++) output[off + ni] = out[ni];
            }
          }
        }
      }
    }
  }
};

inline void moe_gemm_fp8_blockscale_prefill_host(
    const fp16* input, const uint8_t* weight, const float* weight_scale,
    fp16* output, const uint32_t* expert_idx, const int32_t* tile_experts,
    const int32_t* tile_rows, int total_tokens, int tile_capacity, int N, int K,
    int num_experts, int block_n, int block_k, sycl::queue& q) {
  const int Nb = (N + block_n - 1) / block_n;
  const int Kb = K / block_k;
  const int n_tiles = (N + 15) / 16;
  // Empirical BMG grid-shaping caps. Small-K groups do less work and tolerate
  // a larger grid; larger-K groups use a lower cap to bound scheduling cost.
  // These values only coalesce adjacent N tiles into a work-group and never
  // truncate the launched computation.
  const int wg_cap = (K <= 512) ? 51200 : 38400;
  int n_per_wg = 1;
  int total_wgs = tile_capacity * n_tiles;
  while (total_wgs > wg_cap && n_per_wg < n_tiles) {
    n_per_wg++;
    while (n_per_wg < n_tiles && n_tiles % n_per_wg != 0) n_per_wg++;
    total_wgs = tile_capacity * ((n_tiles + n_per_wg - 1) / n_per_wg);
  }
  const int n_wg_count = (n_tiles + n_per_wg - 1) / n_per_wg;
  const int groups = tile_capacity * n_wg_count;
  q.submit([&](handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>((size_t)groups, 1),
        moe_gemm_block_prefill_kernel{
            input, weight, weight_scale, output, expert_idx, tile_experts,
            tile_rows, total_tokens, tile_capacity, N, K, Nb, Kb, num_experts,
            n_wg_count, n_per_wg});
  });
}

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
