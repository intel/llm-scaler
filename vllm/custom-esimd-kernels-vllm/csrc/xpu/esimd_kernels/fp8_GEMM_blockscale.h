/* fp8_GEMM_blockscale.h — w8a16 block-scaled FP8 GEMM (DeepSeek-style).
 *
 * Computes  output[M, N] = input[M, K] @ dequant(weight[N, K])^T
 * where the fp8_e4m3 weight is dequantized on the fly with a 2D block scale:
 *   weight_scale[nb, kb]  applies to  weight[nb*BN : nb*BN+BN, kb*BK : kb*BK+BK]
 * i.e. DeepSeek 128x128 weight block scale. The activation stays fp16 (w8a16):
 * no activation quantization is performed, which keeps decode-path accuracy high
 * and avoids a separate per-token-group quant launch.
 *
 * Layouts (all row-major, contiguous):
 *   input        [M, K]                         fp16
 *   weight       [N, K]                         uint8 (fp8_e4m3 bits)
 *   weight_scale [ceil(N/BN), ceil(K/BK)]       float32   (== weight_scale_inv)
 *   output       [M, N]                         fp16  (pre-allocated)
 *
 * Derived from the proven GEMV_a16_wfp8_block (sglang fp8_GEMV.h): the batch/
 * "absorb" dimension is dropped (linear layers are batch-1), the block scale is
 * read as float32 (checkpoint weight_scale_inv is fp32, not fp16), and the dot
 * product accumulates in fp32 for tighter agreement with a bf16/fp32 reference.
 *
 * The kernel is a K-split GEMV: NT threads per work-group each own an HD-wide
 * slice of K, reduce partials through SLM. It is bandwidth-bound and tuned for
 * small M (decode). The host launcher tiles M so any M is handled correctly;
 * large-M prefill is functional but not throughput-optimal (see K4).
 */
#pragma once

#include "utils.h"
#include <cstdint>

namespace fp8_blockscale {

// fp8_e4m3 field widths.
#define BS_WE 4
#define BS_WM 3

// Vectorized fp8_e4m3 (fn) -> fp16 conversion (handles zero + subnormals; NaN
// inputs are not expected in weights and are not special-cased). Mirrors the
// bit manipulation in GEMV_a16_wfp8_block.
template <uint32_t N>
inline simd<fp16, N> fp8e4m3_to_fp16(simd<uint8_t, N> x) {
  constexpr uint16_t weo = 5;   // fp16 exponent bits
  constexpr uint16_t wmo = 10;  // fp16 mantissa bits

  auto is_zero = (x == 0);

  simd<uint16_t, N> mantissa = x & ((1 << BS_WM) - 1);
  simd<uint16_t, N> exponent = (x & 0x7F) >> BS_WM;

  auto zero_exponent = (exponent == 0);
  simd<uint16_t, N> mantissa_subnormal = mantissa;
  simd<uint16_t, N> exponent_subnormal = exponent;
  {
    // Re-normalize subnormal fp8 mantissa: count leading zeros via the exponent
    // of (float)mantissa, then shift into a normalized fp16 representation.
    simd<uint16_t, N> vec = mantissa;
    simd<float, N> vec_float = vec;
    simd<uint32_t, N> vec_uint = vec_float.template bit_cast_view<uint32_t>();
    simd<uint32_t, N> exponent_tmp = (vec_uint >> 23) & 0xFF;
    simd<uint32_t, N> lz = 158 - exponent_tmp;  // 158 = 127 + 31
    simd<uint16_t, N> renorm_shift = lz;

    simd<uint16_t, N> sh = 1 + renorm_shift - (32 - BS_WM);
    mantissa_subnormal <<= sh;
    exponent_subnormal += 1 - sh;
    mantissa_subnormal &= ((1 << BS_WM) - 1);
  }

  mantissa.merge(mantissa_subnormal, zero_exponent);
  exponent.merge(exponent_subnormal, zero_exponent);

  const uint16_t exp_low_cutoff = (1 << (weo - 1)) - (1 << (BS_WE - 1));
  exponent += exp_low_cutoff;
  mantissa <<= wmo - BS_WM;

  simd<uint16_t, N> sign = x >> 7;
  simd<uint16_t, N> retval = (sign << 15) | (exponent << 10) | mantissa;
  retval.merge(0, is_zero);

  return retval.template bit_cast_view<fp16>();
}

// Block-scaled FP8 GEMV, BMG-tuned (modeled on GEMV_fp8_pert_bmg_kernel).
//
// One work-group per output channel n; K_SPLIT threads cooperate on the K
// reduction (chosen for HW-thread occupancy). Each thread streams its K-slice in
// VL-wide coalesced loads, dequantizes the fp8 weight once, folds the per-128
// block scale into the weight vector (block_k=128 divides VL), and accumulates a
// dot product for every one of the M activation rows -- so the weight load is
// amortized across the (small) decode batch. Only K_SPLIT partials per row pass
// through SLM. This mirrors the near-peak-bandwidth per-tensor decode kernel.
//   VL       K elements loaded per iteration (multiple of BK=128)
//   K_SPLIT  threads per work-group (K reduction fan-in)
//   BK       weight scale K-block (128)
//   MAX_M    compile-time upper bound on rows handled per launch
template <int VL, int K_SPLIT, int BK, int MAX_M>
struct gemv_block_bmg_kernel {
  const fp16* input;      // [M, K]
  const uint8_t* weight;  // [N, K] fp8_e4m3 bits
  const float* wscale;    // [Nb, Kb]  (Kb = K/BK)
  fp16* output;           // [M, N]
  int M, N, K, Kb;

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    if constexpr (K_SPLIT > 1) {
      slm_init<K_SPLIT * MAX_M * sizeof(float)>();
    }
    const int n = item.get_group(0);
    const int lid = item.get_local_id(0);
    if (n >= N) return;

    const int kp = K / K_SPLIT;   // per-thread K span (multiple of VL)
    const int ks = lid * kp;
    const uint8_t* w_row = weight + (size_t)n * K;
    const float* s_row = wscale + (size_t)(n / 128) * Kb;

    simd<float, MAX_M> acc = 0.0f;

    for (int k = ks; k < ks + kp; k += VL) {
      // Load + dequant the weight slice once, fold in the per-128 block scale.
      simd<uint8_t, VL> raw = block_load<uint8_t, VL>(w_row + k);
      simd<float, VL> wf = fp8e4m3_to_fp16<VL>(raw);
#pragma unroll
      for (int sb = 0; sb < VL / BK; sb++) {
        const float sc = s_row[(k / BK) + sb];
        wf.template select<BK, 1>(sb * BK) = wf.template select<BK, 1>(sb * BK) * sc;
      }
      // Reuse the scaled weight across all M activation rows.
#pragma unroll
      for (int m = 0; m < MAX_M; m++) {
        if (m < M) {
          simd<fp16, VL> iv = block_load<fp16, VL>(input + (size_t)m * K + k);
          simd<float, VL> ivf = iv;
          acc[m] += reduce<float>(ivf * wf, std::plus<>());
        }
      }
    }

    if constexpr (K_SPLIT == 1) {
#pragma unroll
      for (int m = 0; m < MAX_M; m++) {
        if (m < M) output[(size_t)m * N + n] = fp16(acc[m]);
      }
    } else {
      slm_block_store<float, MAX_M>(lid * MAX_M * sizeof(float),
                                    acc.template select<MAX_M, 1>(0));
      barrier();
      if (lid == 0) {
        simd<float, K_SPLIT * MAX_M> parts =
            slm_block_load<float, K_SPLIT * MAX_M>(0);
#pragma unroll
        for (int m = 0; m < MAX_M; m++) {
          if (m < M) {
            // parts layout: [lid][m]; sum over lid (stride MAX_M).
            simd<float, K_SPLIT> col =
                parts.template select<K_SPLIT, MAX_M>(m);
            output[(size_t)m * N + n] = fp16(reduce<float>(col, std::plus<>()));
          }
        }
      }
    }
  }
};

template <int VL, int K_SPLIT, int MAX_M>
inline void launch_gemv_block_bmg(const fp16* input, const uint8_t* weight,
                                  const float* wscale, fp16* output, int M, int N,
                                  int K, sycl::queue& q) {
  constexpr int BK = 128;
  const int Kb = K / BK;
  gemv_block_bmg_kernel<VL, K_SPLIT, BK, MAX_M> kern{
      input, weight, wscale, output, M, N, K, Kb};
  sycl::range<1> global(static_cast<size_t>(N) * K_SPLIT);
  sycl::range<1> local(K_SPLIT);
  q.submit([&](handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(global, local), kern);
  });
}

// Dispatch (VL, K_SPLIT) for one M-tile. VL prefers 256; K_SPLIT is raised to
// keep ~640 HW threads busy for small N, subject to K/K_SPLIT staying a multiple
// of VL.
template <int MAX_M>
inline void dispatch_gemv_block_bmg(const fp16* input, const uint8_t* weight,
                                    const float* wscale, fp16* output, int M,
                                    int N, int K, sycl::queue& q) {
  const int VL = (K % 256 == 0) ? 256 : 128;

  // Target K_SPLIT so that N*K_SPLIT >= 640 (BMG occupancy), K%ks==0 and
  // (K/ks)%VL==0.
  int target = 1;
  if (N * 8 <= 640) target = 8;
  else if (N * 4 <= 640) target = 4;
  else if (N * 2 <= 640) target = 2;
  int ks = 1;
  for (int s = target; s >= 1; s >>= 1) {
    if (K % s == 0 && (K / s) % VL == 0) { ks = s; break; }
  }

#define BS_DISPATCH(V, S)                                                      \
  launch_gemv_block_bmg<V, S, MAX_M>(input, weight, wscale, output, M, N, K, q)
  if (VL == 256) {
    switch (ks) {
      case 8: BS_DISPATCH(256, 8); break;
      case 4: BS_DISPATCH(256, 4); break;
      case 2: BS_DISPATCH(256, 2); break;
      default: BS_DISPATCH(256, 1); break;
    }
  } else {
    switch (ks) {
      case 8: BS_DISPATCH(128, 8); break;
      case 4: BS_DISPATCH(128, 4); break;
      case 2: BS_DISPATCH(128, 2); break;
      default: BS_DISPATCH(128, 1); break;
    }
  }
#undef BS_DISPATCH
}

// Host launcher. block_n/block_k must be 128. Handles any M by tiling into
// groups of TILE rows (weight is reloaded per tile but reused across the tile).
inline void gemm_fp8_blockscale_host(const fp16* input, const uint8_t* weight,
                                     const float* weight_scale, fp16* output,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t block_n, uint32_t block_k,
                                     sycl::queue& q) {
  if (M == 1) {
    dispatch_gemv_block_bmg<1>(input, weight, weight_scale, output, 1, (int)N,
                               (int)K, q);
    return;
  }
  constexpr uint32_t TILE = 8;
  for (uint32_t m0 = 0; m0 < M; m0 += TILE) {
    const int mt = (M - m0 < TILE) ? (int)(M - m0) : (int)TILE;
    dispatch_gemv_block_bmg<TILE>(input + (size_t)m0 * K, weight, weight_scale,
                                  output + (size_t)m0 * N, mt, (int)N, (int)K, q);
  }
}

#undef BS_WE
#undef BS_WM

}  // namespace fp8_blockscale
