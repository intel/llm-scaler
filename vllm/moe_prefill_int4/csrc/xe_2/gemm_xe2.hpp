/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {

using namespace cute;

template <typename TB>
CUTE_DEVICE TB apply_scale(TB& x, float& y) {
  static_assert(
      is_any_of_v<TB, bfloat16_t, half_t>, "Only BF16 & FP16 are supported");
  uint16_t z = sycl::bit_cast<uint16_t>(x);
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  if constexpr (is_same_v<TB, half_t>) {
    asm("{\n"
        ".decl Z_FP16 v_type=G type=HF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_FP16(0,0)<1> Z_FP16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  } else {
    asm("{\n"
        ".decl Z_BF16 v_type=G type=BF num_elts=16 alias=<%0,0>\n"
        ".decl Y_FP32 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        "mul (M1, 16) Z_BF16(0,0)<1> Z_BF16(0,0)<1;1,0> Y_FP32(0,0)<1;1,0>\n"
        "}\n"
        : "+rw"(z)
        : "rw"(y));
  }
#endif
  return sycl::bit_cast<TB>(z);
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  Tensor tCgC = thr_mma.partition_C(gC);
  SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gC);

  using TD = typename DTensor::element_type;
  TD tCrC_final_frag[tCrC.size()];
  Tensor tCrC_final_tensor =
      make_tensor(make_rmem_ptr(tCrC_final_frag), tCrC.layout());
  SubgroupTensor tCrC_final_sg_tensor =
      make_subgroup_tensor(tCrC_final_tensor, tCrC.tv_layout());

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC);

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  if constexpr (is_B_fp8_type) {
    float B_scale = Scales[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= B_scale;
    }
  }

  if (Bias != nullptr) {
    static constexpr auto ATOM_M =
        get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N =
        get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    static constexpr auto tile_m = get<0>(wg_tile);
    static constexpr auto tile_n = get<1>(wg_tile);

    // 32 * 64
    static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();
    static constexpr int sg_local_range = 16;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_final_sg_tensor);
  copy(copy_c, tCrC_final_sg_tensor, tCgC);
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    int GroupSize,
    bool KMajorScale = false,
    bool MarlinUnshuffle = false,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm_4bits(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,
    int gemm_n_total = 0) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  static constexpr int group_size = GroupSize;
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  Tensor tCgC = thr_mma.partition_C(gC);
  SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gC);

  using TD = typename DTensor::element_type;
  TD tCrC_final_frag[tCrC.size()];
  Tensor tCrC_final_tensor =
      make_tensor(make_rmem_ptr(tCrC_final_frag), tCrC.layout());
  SubgroupTensor tCrC_final_sg_tensor =
      make_subgroup_tensor(tCrC_final_tensor, tCrC.tv_layout());

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 6;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K =
      get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;
  static constexpr auto SG_K = tile_k / ATOM_K;  // BLK_K / ATOM_K;

  static constexpr auto thr_N = get<1>(tCrB.shape());
  static constexpr auto channel_num = get<0>(get<0>(tCrB.shape()));
  auto n_tile_start = wg_n * tile_n;

  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / group_size;
  int x_idx = sg_local_id / channel_num;

  using scaleStoreType = conditional_t<is_same_v<TA, half_t>, half_t, float>;
  scaleStoreType scales[thr_N * channel_num];

  clear(tCrC);

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    if (k_tile_prefetch * group_size < shape<1>(A)) {
      if constexpr (KMajorScale) {
        // K-major scale: [K/GS, N] — N-contiguous, skip 2D prefetch for now
        // (scales are small, L1/L2 cache handles them well)
      } else {
        // N-major scale: [N, K/GS], base = n_start * group_num + group_idx
        auto next_scales_tensor = make_tensor(
            make_gmem_ptr(
                reinterpret_cast<const ElementS*>(
                    Scales + (n_tile_start + n_sg_start) * group_num +
                    k_tile_prefetch)),
            make_layout(
                make_shape(Int<SG_N>{}, Int<1>{}),
                make_stride(group_num, Int<1>{})));
        auto prefetch_scales = make_block_2d_prefetch<1>(
            make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
        auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
        auto pSgS = thr_prefetch_scales.partition_S(
            make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
        prefetch(prefetch_scales, pSgS(_, 0, 0));
      }
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile * tile_k % group_size == 0) {
      int group_idx = (k_tile * tile_k) / group_size;

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < thr_N; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < channel_num; ++c) {
          int real_idx = x_idx + c * (sg_local_range / channel_num);
          int sg_local_n = n * sg_local_range + real_idx;
          scaleStoreType scale;
          if constexpr (KMajorScale && std::is_same_v<TB, int4_t>) {
            // K-major scale: [K/GS, N] per expert, stride = (N, 1)
            scale = Scales
                [group_idx * gemm_n_total +
                 (n_tile_start + n_sg_start + sg_local_n)];
          } else if constexpr (std::is_same_v<TB, int4_t>) {
            // N-major scale: [N, K/GS] per expert, stride = (K/GS, 1)
            scale = Scales
                [(n_tile_start + n_sg_start + sg_local_n) * group_num +
                 group_idx];
          } else if constexpr (std::is_same_v<TB, float_e2m1_t>) {
            uint32_t scale_u32 =
                Scales
                    [(n_tile_start + n_sg_start + sg_local_n) * group_num +
                     group_idx]
                << 23;
            scale = static_cast<scaleStoreType>(
                reinterpret_cast<float&>(scale_u32));
          }

          scales[n * channel_num + c] = scale;
        }
      }

      if ((group_idx + prefetch_dist) * group_size < shape<1>(A)) {
        if constexpr (KMajorScale) {
          // K-major scale: skip 2D prefetch, rely on cache
        } else {
          auto next_scales_tensor = make_tensor(
              make_gmem_ptr(
                  reinterpret_cast<const ElementS*>(
                      Scales + (n_tile_start + n_sg_start) * group_num +
                      group_idx + prefetch_dist)),
              make_layout(
                  make_shape(Int<SG_N>{}, Int<1>{}),
                  make_stride(group_num, Int<1>{})));
          auto prefetch_scales = make_block_2d_prefetch<1>(
              make_shape(Int<SG_N>{}, Int<1>{}), next_scales_tensor);
          auto thr_prefetch_scales = prefetch_scales.get_slice(sg_local_id);
          auto pSgS = thr_prefetch_scales.partition_S(
              make_identity_tensor(make_shape(Int<SG_N>{}, Int<1>{})));
          prefetch(prefetch_scales, pSgS(_, 0, 0));
        }
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);

    // MarlinUnshuffle: no-op here, unshuffle is done in-place on global
    // memory before the kernel launch (see grouped_gemm_xe2_interface.hpp).
    reorder(tBrB, tCrB);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tCrB.size() / thr_N / channel_num; ++i) {
          if constexpr (std::is_same_v<TA, half_t>) {
            tCrB(cute::tuple(c, _), n, _)[i] *= scales[n * channel_num + c];
          } else {
            tCrB(cute::tuple(c, _), n, _)[i] = apply_scale(
                tCrB(cute::tuple(c, _), n, _)[i], scales[n * channel_num + c]);
          }
        }
      }
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  if (Bias != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_final_sg_tensor);
  copy(copy_c, tCrC_final_sg_tensor, tCgC);
}

}  // namespace MoE
