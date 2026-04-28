/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
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
/*! \file
    \brief CUTLASS Intel BMG MoE API example based on sycl-tla Group GEMM

*/
#include <torch/all.h>
#include "../utils.h"

#include <cute/tensor.hpp>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

// ── Marlin in-place unshuffle + re-shuffle helpers ──────────────────────────
// Marlin stores nibble slot[i] = original[SHUFFLED_IDX[i]] where
// SHUFFLED_IDX = [0,4,1,5,2,6,3,7]. To convert to natural nibble order
// we need: for each int32 word, new_word[k] = old_word[UNSHUFFLE[k]]
// where UNSHUFFLE = [0,2,4,6,1,3,5,7] (the inverse of SHUFFLED_IDX).
//
// These run on XPU via SYCL parallel_for, modifying the weight tensor
// in-place. The GEMM reads natural order; after GEMM we re-shuffle back
// so the decode kernel still sees marlin format.
// Convert unsigned nibble (0-15, zp=8) to XeTLA signed int4 encoding.
// implement_zp format: sign bit (bit 3) + magnitude (bits 0-2).
//   val=0 → 0x0=0000, val=7 → 0x7=0111, val=-1 → 0x9=1001, val=-8 → 0x8=1000
inline uint32_t u4_to_s4_word(uint32_t w) {
    uint32_t out = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t u = (w >> (i * 4)) & 0xF;
        int s = (int)u - 8;  // signed value: -8..+7
        uint32_t enc;
        if (s < 0) {
            enc = (1u << 3) | (((uint32_t)(-s)) & 0x7u);
        } else {
            enc = (uint32_t)s & 0x7u;
        }
        out |= enc << (i * 4);
    }
    return out;
}

// Reverse: XeTLA signed int4 → unsigned nibble (0-15)
inline uint32_t s4_to_u4_word(uint32_t w) {
    uint32_t out = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t enc = (w >> (i * 4)) & 0xF;
        int s;
        if (enc & 0x8) {
            s = -(int)(enc & 0x7);
        } else {
            s = (int)(enc & 0x7);
        }
        uint32_t u = (uint32_t)(s + 8);
        out |= u << (i * 4);
    }
    return out;
}

inline void marlin_unshuffle_inplace(at::Tensor& B_int32, sycl::queue& q) {
    // Unshuffle marlin nibbles AND convert unsigned (0-15) → signed int4 encoding.
    int64_t total = B_int32.numel();
    uint32_t* ptr = reinterpret_cast<uint32_t*>(B_int32.data_ptr());
    constexpr int UNSHUFFLE[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        uint32_t w = ptr[idx];
        // Step 1: unshuffle
        uint32_t unshuf = 0;
        for (int k = 0; k < 8; ++k) {
            int slot = UNSHUFFLE[k];
            uint32_t nibble = (w >> (slot * 4)) & 0xF;
            unshuf |= nibble << (k * 4);
        }
        // Step 2: unsigned → signed int4 encoding
        uint32_t out = 0;
        for (int i = 0; i < 8; ++i) {
            uint32_t u = (unshuf >> (i * 4)) & 0xF;
            int s = (int)u - 8;
            uint32_t enc;
            if (s < 0) {
                enc = (1u << 3) | (((uint32_t)(-s)) & 0x7u);
            } else {
                enc = (uint32_t)s & 0x7u;
            }
            out |= enc << (i * 4);
        }
        ptr[idx] = out;
    }).wait();
}

inline void marlin_reshuffle_inplace(at::Tensor& B_int32, sycl::queue& q) {
    // Reverse: signed int4 → unsigned, then re-shuffle to marlin order.
    int64_t total = B_int32.numel();
    uint32_t* ptr = reinterpret_cast<uint32_t*>(B_int32.data_ptr());
    constexpr int SHUFFLE[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        uint32_t w = ptr[idx];
        // Step 1: signed → unsigned
        uint32_t natural = 0;
        for (int i = 0; i < 8; ++i) {
            uint32_t enc = (w >> (i * 4)) & 0xF;
            int s;
            if (enc & 0x8) {
                s = -(int)(enc & 0x7);
            } else {
                s = (int)(enc & 0x7);
            }
            uint32_t u = (uint32_t)(s + 8);
            natural |= u << (i * 4);
        }
        // Step 2: re-shuffle
        uint32_t out = 0;
        for (int k = 0; k < 8; ++k) {
            int slot = SHUFFLE[k];
            uint32_t nibble = (natural >> (slot * 4)) & 0xF;
            out |= nibble << (k * 4);
        }
        ptr[idx] = out;
    }).wait();
}

// type tag to define a unique sycl kernel name
template <typename, typename, typename, typename, char, char, class>
class GemmCuteName;

template <
    char layoutA,
    char layoutB,
    class policy,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void MoEGEMMLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementBI* bias,
    ElementD* outputs,
    const int gemm_n,
    const int gemm_k,
    const int64_t* expert_first_token_offset,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<GemmCuteName<
        ElementA,
        ElementB,
        ElementS,
        ElementD,
        layoutA,
        layoutB,
        policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEGEMM<
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R'>(
              activations,
              weights,
              scales,
              bias,
              outputs,
              mma,
              expert_first_token_offset,
              num_experts,
              group_size,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem);
        });
  });
}

at::Tensor cutlass_grouped_gemm_xe2_impl(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4,
    bool is_B_int4_kmajor = false) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));

  TORCH_CHECK(N % 32 == 0, "N must be divisible by 32");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);

  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);
  if (is_B_int4_kmajor) {
    // K-major marlin: [E, K/8, N] int32 → K = size(1)*8, N = size(2)
    B_K = ptr_B.size(1) * 8;
    B_N = ptr_B.size(2);
  } else if (is_B_int4 || is_B_mxfp4) {
    // N-major: [E, N, K/2] uint8 → K = size(2)*2, N = size(1)
    B_K = ptr_B.size(2) * 2;
    B_N = ptr_B.size(1);
  }

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  int group_size = -1;
  int A_avg_M = A_total_M / num_experts;

  TORCH_CHECK(B_E == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match ptr_B.size(1)");
  TORCH_CHECK(B_N == D_N && D_N == N, "ptr_B.size(2) must match ptr_D.size(1)");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->size(0) == num_experts,
        "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(ptr_bias->size(1) == N, "ptr_bias.size(1) must match N");
  }

  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

#define MoEGEMMLauncherCallER(                                                 \
    LayoutA, LayoutB, Policy, ElementA, ElementB, ElementS)                    \
  MoEGEMMLauncher<LayoutA, LayoutB, Policy>(                                   \
      dpcpp_queue,                                                             \
      reinterpret_cast<ElementA*>(ptr_A.data_ptr()),                           \
      reinterpret_cast<ElementB*>(ptr_B.data_ptr()),                           \
      ptr_scales.has_value()                                                   \
          ? reinterpret_cast<ElementS*>(ptr_scales->data_ptr())                \
          : static_cast<ElementS*>(nullptr),                                   \
      ptr_bias.has_value() ? reinterpret_cast<ElementA*>(ptr_bias->data_ptr()) \
                           : static_cast<ElementA*>(nullptr),                  \
      reinterpret_cast<ElementA*>(ptr_D.data_ptr()),                           \
      N,                                                                       \
      K,                                                                       \
      reinterpret_cast<int64_t*>(expert_first_token_offset.data_ptr()),        \
      num_experts,                                                             \
      group_size,                                                              \
      static_cast<int*>(atomic_buffer.data_ptr()));

  if (is_B_int4_kmajor) {
    // ── K-major marlin: weight [E, K/8, N] int32, scale [E, K/GS, N] fp16 ──
    TORCH_CHECK(is_B_int4, "is_B_int4_kmajor requires is_B_int4=true");
    TORCH_CHECK(ptr_scales.has_value(), "int4 kmajor grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(ptr_scales->dim() == 3,
        "ptr_scales of int4_kmajor must be 3D [num_experts, K/GS, N]");
    TORCH_CHECK(ptr_scales->size(0) == num_experts);
    TORCH_CHECK(ptr_scales->size(2) == N,
        "ptr_scales.size(2) of int4_kmajor must match N");
    int group_num = ptr_scales->size(1);
    group_size = K / group_num;
    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");

    // In-place unshuffle marlin nibbles → natural order before GEMM.
    // After GEMM, re-shuffle back so decode kernel still sees marlin format.
    marlin_unshuffle_inplace(ptr_B, dpcpp_queue);

#define W4A16KMajorLauncherCallER(policy)                                  \
  if (A_dtype == at::kBFloat16) {                                          \
    using scalar_t = bfloat16_t;                                           \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, uint8_t, scalar_t);  \
  } else if (A_dtype == at::kHalf) {                                       \
    using scalar_t = half_t;                                               \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, uint8_t, scalar_t);  \
  }

    if (A_avg_M <= 4) {
      using policy = w4a16_policy_m_8;
      W4A16KMajorLauncherCallER(policy);
    } else if (A_avg_M <= 8) {
      using policy = w4a16_policy_m_16;
      W4A16KMajorLauncherCallER(policy);
    } else if (A_avg_M <= 128) {
      using policy = w4a16_policy_m_32;
      W4A16KMajorLauncherCallER(policy);
    } else {
      using policy = w4a16_policy;
      W4A16KMajorLauncherCallER(policy);
    }
#undef W4A16KMajorLauncherCallER
    // Re-shuffle back to marlin format for decode kernel
    marlin_reshuffle_inplace(ptr_B, dpcpp_queue);
  } else if (is_B_int4 || is_B_mxfp4) {
    // ── N-major: weight [E, N, K/2] uint8, scale [E, N, K/GS] fp16 ──
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 3,
        "ptr_scales of int4 must be 3D [num_experts, group_num, N]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of int4 must match num_experts");
    TORCH_CHECK(
        K % ptr_scales->size(2) == 0,
        "ptr_scales.size(2) of int4 must be divisible by K");
    TORCH_CHECK(
        ptr_scales->size(1) == N, "ptr_scales.size(1) of int4 must match N");
    int group_num = ptr_scales->size(2);
    group_size = K / group_num;

    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");

#define W4A16LauncherCallER(policy)                                         \
  if (is_B_int4) {                                                          \
    if (A_dtype == at::kBFloat16) {                                         \
      using scalar_t = bfloat16_t;                                          \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, scalar_t); \
    } else if (A_dtype == at::kHalf) {                                      \
      using scalar_t = half_t;                                              \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, scalar_t); \
    }                                                                       \
  } else if (is_B_mxfp4) {                                                  \
    if (A_dtype == at::kBFloat16) {                                         \
      using scalar_t = bfloat16_t;                                          \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, uint8_t);  \
    } else if (A_dtype == at::kHalf) {                                      \
      using scalar_t = half_t;                                              \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, uint8_t);  \
    }                                                                       \
  }

    if (A_avg_M <= 4) {
      using policy = w4a16_policy_m_8;
      W4A16LauncherCallER(policy);
    } else if (A_avg_M <= 8) {
      using policy = w4a16_policy_m_16;
      W4A16LauncherCallER(policy);
    } else if (A_avg_M <= 128) {
      using policy = w4a16_policy_m_32;
      W4A16LauncherCallER(policy);
    } else {
      using policy = w4a16_policy;
      W4A16LauncherCallER(policy);
    }
#undef W4A16LauncherCallER
  } else if (is_weight_fp8) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 1, "ptr_scales of fp8 must be 1D [num_experts]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of fp8 must match num_experts");
    TORCH_CHECK(ptr_scales->dtype() == at::kFloat, "ptr_scales must be float");

#define W8A16LauncherCallER(policy)                                         \
  if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kHalf) {              \
    using scalar_t = half_t;                                                \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e4m3_t, float); \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kHalf) {         \
    using scalar_t = half_t;                                                \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e5m2_t, float); \
  } else if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kBFloat16) {   \
    using scalar_t = bfloat16_t;                                            \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e4m3_t, float); \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kBFloat16) {     \
    using scalar_t = bfloat16_t;                                            \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e5m2_t, float); \
  }

    if (A_avg_M <= 8) {
      using policy = w8a16_policy_m_16;
      W8A16LauncherCallER(policy);
    } else if (A_avg_M <= 32) {
      using policy = w8a16_policy_m_32;
      W8A16LauncherCallER(policy);
    } else {
      using policy = w8a16_policy;
      W8A16LauncherCallER(policy);
    }
#undef W8A16LauncherCallER
  } else {
    TORCH_CHECK(
        !ptr_scales.has_value(), "w16a16 grouped gemm must not have scales");

#define W16A16LauncherCallER(policy)                                       \
  if (A_dtype == at::kBFloat16) {                                          \
    using scalar_t = bfloat16_t;                                           \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  } else if (A_dtype == at::kHalf) {                                       \
    using scalar_t = half_t;                                               \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  }

    if (A_avg_M <= 8) {
      using policy = w16a16_policy_m_16;
      W16A16LauncherCallER(policy);
    } else if (A_avg_M <= 16) {
      using policy = w16a16_policy_m_32;
      W16A16LauncherCallER(policy);
    } else {
      if (B_N <= 64) {
        using policy = w16a16_policy_n_64;
        W16A16LauncherCallER(policy);
      } else if (B_N <= 512) {
        using policy = w16a16_policy_n_128;
        W16A16LauncherCallER(policy);
      } else {
        using policy = w16a16_policy;
        W16A16LauncherCallER(policy);
      }
    }
#undef W16A16LauncherCallER
  }
#undef MoEGEMMLauncherCallER
  return ptr_D;
}
}  // namespace MoE