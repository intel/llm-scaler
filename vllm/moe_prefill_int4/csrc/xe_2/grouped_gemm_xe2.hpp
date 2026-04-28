/***************************************************************************************************
 * Copyright 2025 Intel corporation. All rights reserved.
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
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "gemm_xe2.hpp"
#include <cute/util/compat.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_moe_tensor(T* ptr, int r, int c) {
  auto shape = make_shape(r, c);
  if constexpr (LayoutKind == 'C')
    return make_tensor(
        make_gmem_ptr(ptr), make_layout(shape, make_stride(_1{}, r)));
  else
    return make_tensor(
        make_gmem_ptr(ptr), make_layout(shape, make_stride(c, _1{})));
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyD,
    char LayoutKindA,
    char LayoutKindB,
    char LayoutKindD,
    class TiledMMA,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
CUTE_DEVICE void MoEGEMM(
    const ElementA* Activations,
    const ElementB* Weights,
    const ElementS* Scales,
    const ElementBI* Bias,
    ElementD* Outputs,
    TiledMMA const& mma,
    const int64_t* expert_first_token_offset,
    const int32_t num_experts,
    const int32_t group_size,
    const int32_t gemm_n,
    const int32_t gemm_k,
    int32_t* atomic_buffer,
    const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');
  static constexpr bool is_B_int4 = (std::is_same_v<ElementB, uint8_t>) &&
                                    (!std::is_same_v<ElementS, uint8_t>);
  static constexpr bool is_B_mxfp4 = (std::is_same_v<ElementB, uint8_t>) &&
                                     (std::is_same_v<ElementS, uint8_t>);
  static constexpr bool is_B_4bits = std::is_same_v<ElementB, uint8_t>;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();

  if (group_id == 0 && local_id == 0) {
    auto atm = sycl::atomic_ref<
        int,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>(atomic_buffer[0]);
    atm.store(0);
  }

  int pre_rows = 0;
  int pre_tiles = 0;

  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());

  for (int i = 0; i < num_experts; ++i) {
    int cumsum_rows_for_experts = expert_first_token_offset[i + 1];
    int gemm_m = cumsum_rows_for_experts - pre_rows;
    int cumsum_tiles_for_experts =
        (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    int expert_id = i;
    int64_t B_offset = static_cast<int64_t>(expert_id) *
                       static_cast<int64_t>(gemm_n) *
                       static_cast<int64_t>(gemm_k);
    if constexpr (is_B_4bits) {
      B_offset /= 2;
    }
    ElementA* ptr_A_curr_batch =
        const_cast<ElementA*>(Activations) + pre_rows * gemm_k;
    ElementB* ptr_B_curr_batch = const_cast<ElementB*>(Weights) + B_offset;
    ElementD* ptr_D_curr_batch = Outputs + pre_rows * gemm_n;
    ElementS* ptr_Scales_curr_batch = const_cast<ElementS*>(Scales) + expert_id;
    if constexpr (is_B_4bits) {
      ptr_Scales_curr_batch =
          const_cast<ElementS*>(Scales) + B_offset * 2 / group_size;
    }
    ElementBI* ptr_Bias_curr_batch = nullptr;
    if (Bias != static_cast<ElementBI*>(nullptr)) {
      ptr_Bias_curr_batch = const_cast<ElementBI*>(Bias) + expert_id * gemm_n;
    }

    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
        ptr_A_curr_batch, gemm_m, gemm_k);
    auto B_tensor = [&]() {
      if constexpr (is_B_int4) {
        return make_moe_tensor<int4_t, actual_layout_of_B>(
            reinterpret_cast<int4_t*>(ptr_B_curr_batch), gemm_n, gemm_k);
      } else if constexpr (is_B_mxfp4) {
        return make_moe_tensor<float_e2m1_t, actual_layout_of_B>(
            reinterpret_cast<float_e2m1_t*>(ptr_B_curr_batch), gemm_n, gemm_k);
      } else {
        return make_moe_tensor<ElementB, actual_layout_of_B>(
            ptr_B_curr_batch, gemm_n, gemm_k);
      }
    }();
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(
        ptr_D_curr_batch, gemm_m, gemm_n);

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      if constexpr (is_B_4bits) {
        // Detect K-major marlin from LayoutKindB: 'R' → K-major, 'C' → N-major
        constexpr bool is_kmajor = (LayoutKindB == 'R');

#define XE_GEMM_4BITS_CALLER(GroupSize)                                                    \
  xe_gemm_4bits<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, GroupSize,                \
                is_kmajor, is_kmajor>(                                                     \
      A_tensor,                                                                            \
      B_tensor,                                                                            \
      ptr_Scales_curr_batch,                                                               \
      ptr_Bias_curr_batch,                                                                 \
      D_tensor,                                                                            \
      tile_coord,                                                                          \
      mma,                                                                                 \
      is_kmajor ? gemm_n : 0);
        if (group_size == 32) {
          XE_GEMM_4BITS_CALLER(32)
        } else if (group_size == 64) {
          XE_GEMM_4BITS_CALLER(64)
        } else if (group_size == 128) {
          XE_GEMM_4BITS_CALLER(128)
        } else if (group_size == 256) {
          XE_GEMM_4BITS_CALLER(256)
        }
#undef XE_GEMM_4BITS_CALLER
      } else {
        xe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
            A_tensor,
            B_tensor,
            ptr_Scales_curr_batch,
            ptr_Bias_curr_batch,
            D_tensor,
            tile_coord,
            mma);
      }

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      item.barrier(sycl::access::fence_space::local_space);
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }
    pre_rows = cumsum_rows_for_experts;
    pre_tiles = cumsum_tiles_for_experts;
  }
}

}  // namespace MoE
