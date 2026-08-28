/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * Copyright (C) 2026 Sol-Attn XPU contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Sol-Attn CUTE/DPAS mainloop derived from SYCL-TLA's XeDefault FMHA
 * mainloop.  It follows the algorithm used by NVIDIA's official Metal
 * backend: each Q64 route block first consumes K-centroid/V-mean summaries,
 * then consumes only the routed exact K64/V64 blocks under the same online
 * softmax state. Multiple adjacent Q64 route blocks share one workgroup to
 * reduce scheduling passes while keeping subgroup-local routing decisions.
 **************************************************************************************************/

#pragma once

#include "sol_attn_config.h"

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"

#ifndef SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
#define SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS 0
#endif

#ifndef SOL_ATTN_BMG_CACHEABLE_SUMMARY_KV_LOADS
#define SOL_ATTN_BMG_CACHEABLE_SUMMARY_KV_LOADS 0
#endif

#if SOL_ATTN_BMG_CACHEABLE_SUMMARY_KV_LOADS && !SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
#error "cacheable summary K/V loads require the accepted cacheable exact K/V parent"
#endif

#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
#if !defined(OMNI_XPU_ARCH_BMG)
#error "cacheable exact K/V block-2D loads are a BMG-only experiment"
#endif

// The pinned SYCL-TLA v0.9.2 header uses the Xe3 three-level
// `ca.ca.uc` spelling for demand loads, while its BMG prefetch path and the
// BMG ESIMD interface use two cache levels. Keep this experiment local to the
// routed exact K/V copies: Q and summary copies retain the pinned source
// policy, and the external SYCL-TLA checkout is never patched.
namespace cute {

template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_BMG_CACHEABLE
    : XE_Copy_Op_2D_Base<Bits, Height, Width, Width / BlockWidth> {
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    constexpr auto grf_aligned_size = cute::max(64, Width * Height);
    auto& dv = *reinterpret_cast<
        storage_vector_t<T, grf_aligned_size * Bits / sg_size>*>(dst);
    asm(
        "lsc_load_block2d.ugm.ca.ca (M1, 1)  "
        "%0:d%2.%3x%4x%5nn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width / BlockWidth),
          "P"(BlockWidth), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH(
        "BMG cacheable block 2D copy requires an Intel Xe target");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D<Bits, Height, Width>;
};

template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_VNNI_BMG_CACHEABLE
    : XE_Copy_Op_2D_Base<Bits, Height, Width, Width / BlockWidth> {
  static_assert(Bits == 8 || Bits == 16, "unsupported VNNI data size");

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    auto& dv = *reinterpret_cast<
        storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
    asm(
        "lsc_load_block2d.ugm.ca.ca (M1, 1)  "
        "%0:d%2.%3x%4x%5nt flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width / BlockWidth),
          "P"(BlockWidth), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH(
        "BMG cacheable VNNI block 2D copy requires an Intel Xe target");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D<Bits, Height, Width>;
};

template <int Bits, int Height, int Width>
struct XE_LOAD_2D_TRANSPOSE_BMG_CACHEABLE
    : XE_Copy_Op_2D_Base<Bits, Height, Width, 1, true> {
  static_assert(Bits == 32 || Bits == 64, "unsupported transpose data size");
  static_assert(Width <= 8, "BMG transpose width exceeds hardware limits");
  static_assert(
      Bits != 64 || (Height == 8 && Width <= 4),
      "unsupported BMG D64 transpose block size");

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    auto& dv = *reinterpret_cast<
        storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
    asm(
        "lsc_load_block2d.ugm.ca.ca (M1, 1)  "
        "%0:d%2.%3x%4tn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height));
#else
    CUTE_INVALID_CONTROL_PATH(
        "BMG cacheable transpose block 2D copy requires an Intel Xe target");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D<Bits, Height, Width>;
};

template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<
    XE_LOAD_2D_BMG_CACHEABLE<CopyBits, Height, Width, BlockWidth>,
    XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<
          XE_LOAD_2D_BMG_CACHEABLE<CopyBits, Height, Width, BlockWidth>,
          XMode, YMode, ValType, TiledStrides> {
  using Op = XE_LOAD_2D_BMG_CACHEABLE<
      CopyBits, Height, Width, BlockWidth>;
  using Super = Xe2DLoadTraitsBase<
      Op, XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  using DstLayout = XeInterleavedLayout<
      Layout<Shape<Int<BlockWidth>, Int<Height>, Int<Width / BlockWidth>>,
             Stride<_1, Int<Width>, Int<BlockWidth>>>,
      CopyBits, sizeof_bits_v<ValType>>;
  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(
      RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width, int BlockWidth>
struct Copy_Traits<
    XE_LOAD_2D_VNNI_BMG_CACHEABLE<CopyBits, Height, Width, BlockWidth>,
    XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<
          XE_LOAD_2D_VNNI_BMG_CACHEABLE<
              CopyBits, Height, Width, BlockWidth>,
          XMode, YMode, ValType, TiledStrides> {
  using Op = XE_LOAD_2D_VNNI_BMG_CACHEABLE<
      CopyBits, Height, Width, BlockWidth>;
  using Super = Xe2DLoadTraitsBase<
      Op, XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  static constexpr int BV = 32 / CopyBits;
  using DstLayout = XeInterleavedLayout<
      Layout<
          Shape<Int<BV>, Int<BlockWidth>, Int<Height / BV>,
                Int<Width / BlockWidth>>,
          Stride<Int<Width>, _1, Int<Width * BV>, Int<BlockWidth>>>,
      CopyBits, sizeof_bits_v<ValType>>;
  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(
      RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

template <class XMode, class YMode, typename ValType, typename TiledStrides,
          int CopyBits, int Height, int Width>
struct Copy_Traits<
    XE_LOAD_2D_TRANSPOSE_BMG_CACHEABLE<CopyBits, Height, Width>,
    XMode, YMode, ValType, TiledStrides>
    : Xe2DLoadTraitsBase<
          XE_LOAD_2D_TRANSPOSE_BMG_CACHEABLE<CopyBits, Height, Width>,
          XMode, YMode, ValType, TiledStrides> {
  using Op = XE_LOAD_2D_TRANSPOSE_BMG_CACHEABLE<
      CopyBits, Height, Width>;
  using Super = Xe2DLoadTraitsBase<
      Op, XMode, YMode, ValType, TiledStrides>;
  using Super::Super;

  using DstLayout = XeInterleavedLayout<
      Layout<Shape<Int<Height>, Int<Width>>, Stride<Int<Width>, _1>>,
      CopyBits, sizeof_bits_v<ValType>>;
  using RefLayout = DstLayout;
  using SrcLayout = decltype(replace<0>(
      RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));
};

template <int Bits, int Height, int Width, int BlockWidth>
struct is_xe_block_2d_atom<
    XE_LOAD_2D_BMG_CACHEABLE<Bits, Height, Width, BlockWidth>>
    : std::true_type {};

template <int Bits, int Height, int Width, int BlockWidth>
struct is_xe_block_2d_atom<
    XE_LOAD_2D_VNNI_BMG_CACHEABLE<Bits, Height, Width, BlockWidth>>
    : std::true_type {};

template <int Bits, int Height, int Width>
struct is_xe_block_2d_atom<
    XE_LOAD_2D_TRANSPOSE_BMG_CACHEABLE<Bits, Height, Width>>
    : std::true_type {};

}  // namespace cute
#endif

namespace omni_xpu_sol_attn::cute_backend {

using namespace cute;

// Current SYCL-TLA exposes only a full subgroup reduction. Sol-Attn can defer
// the horizontal part of its row sum until all routed tiles have contributed,
// so retain the vertical (per-lane) half as a package-local helper.
template <int Mode, class BinaryOp, class Engine, class FragLayout,
          class SubgroupTVLayout>
CUTE_HOST_DEVICE
auto sol_reduce_vertical(
    SubgroupTensor<Engine, FragLayout, SubgroupTVLayout> const& src,
    BinaryOp op) {
  using SrcTensor =
      SubgroupTensor<Engine, FragLayout, SubgroupTVLayout>;
  using T = typename SrcTensor::value_type;
  using TVToV = Layout<Shape<intel::_SGSize, int>, Stride<_0, _1>>;

  constexpr auto shape = atuple_coshape(SubgroupTVLayout{});
  constexpr auto coord_to_tv =
      right_inverse(project_strides(SubgroupTVLayout{})).with_shape(shape);
  constexpr auto rcoord_to_tv = make_layout(
      select<Mode>(coord_to_tv), remove<Mode>(coord_to_tv));
  constexpr auto rcoord_to_v = filter(
      composition(TVToV{}, rcoord_to_tv), Step<_1, _1>{});

  Tensor src_r = make_tensor(src.data(), rcoord_to_v);
  auto rshape = replace<Mode>(shape, _1{});
  auto out = make_subgroup_tensor(
      make_tensor<T>(Int<size<1>(rcoord_to_v)>{}),
      remove<Mode>(make_identity_layout(rshape)));

  CUTE_UNROLL
  for (int j = 0; j < size<1>(rcoord_to_v); ++j) {
    out(j) = src_r(0, j);
  }
  CUTE_UNROLL
  for (int i = 1; i < size<0>(rcoord_to_v); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); ++j) {
      if constexpr (is_same_v<BinaryOp, sycl::maximum<void>>) {
        out(j) = sycl::max(out(j), src_r(i, j));
      } else {
        out(j) = op(out(j), src_r(i, j));
      }
    }
  }
  return out;
}

template <class BinaryOp, class Engine, class FragLayout,
          class SubgroupTVLayout>
CUTE_HOST_DEVICE
auto sol_reduce_horizontal(
    SubgroupTensor<Engine, FragLayout, SubgroupTVLayout> const& src,
    BinaryOp op) {
  using SrcTensor =
      SubgroupTensor<Engine, FragLayout, SubgroupTVLayout>;
  using T = typename SrcTensor::value_type;
  constexpr int kElements = decltype(size(SrcTensor{}.layout()))::value;
  constexpr int kOutputElements =
      (kElements + intel::sg_size - 1) / intel::sg_size;
  auto out = make_subgroup_tensor(
      make_tensor<T>(Int<kOutputElements>{}),
      make_identity_layout(Shape<Int<kElements>>{}));

  constexpr bool kFloatAdd =
      is_same_v<T, float> && is_same_v<BinaryOp, sycl::plus<void>>;
  if constexpr (kFloatAdd && kElements == 8) {
    float values[8];
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      values[i] = src(i);
    }
    out(0) = hreduce8_float_add(values);
  } else {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    CUTE_UNROLL
    for (int i = 0; i < kElements; ++i) {
      set_single_value(out, i, reduce_over_group(sg, src(i), op));
    }
  }
  return out;
}

#ifndef SOL_ATTN_INLINE_ROUTE
#define SOL_ATTN_INLINE_ROUTE 0
#endif

#ifndef SOL_ATTN_SHARED_INLINE_ROUTE
#define SOL_ATTN_SHARED_INLINE_ROUTE 0
#endif

#ifndef SOL_ATTN_PREFETCH_ROUTED_KV
#define SOL_ATTN_PREFETCH_ROUTED_KV 0
#endif

#ifndef SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE
#define SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE 1
#endif

#ifndef SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH
#define SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH 0
#endif

#ifndef SOL_ATTN_STAGGER_ROUTED_K_PREFETCH
#define SOL_ATTN_STAGGER_ROUTED_K_PREFETCH 0
#endif

#ifndef SOL_ATTN_REGISTER_PIPELINE_EXACT_K
#define SOL_ATTN_REGISTER_PIPELINE_EXACT_K 0
#endif

#ifndef SOL_ATTN_HOIST_APPROXIMATE_LOG2
#define SOL_ATTN_HOIST_APPROXIMATE_LOG2 0
#endif

#ifndef SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS
#define SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS 0
#endif

#ifndef SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
#define SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE 0
#endif

#ifndef SOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS
#define SOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS 0
#endif

#ifndef SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID
#define SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID 0
#endif

#ifndef SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID
#define SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID 0
#endif

#ifndef SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID
#define SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID 0
#endif

#ifndef SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS
#define SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS 0
#endif

#ifndef SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS
#define SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS 0
#endif

#if SOL_ATTN_SHARED_INLINE_ROUTE && !SOL_ATTN_INLINE_ROUTE
#error "SOL_ATTN_SHARED_INLINE_ROUTE requires SOL_ATTN_INLINE_ROUTE"
#endif

#if SOL_ATTN_PREFETCH_ROUTED_KV && !SOL_ATTN_SHARED_INLINE_ROUTE
#error "SOL_ATTN_PREFETCH_ROUTED_KV requires SOL_ATTN_SHARED_INLINE_ROUTE"
#endif

#if SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE != 1 && \
    SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE != 2
#error "routed K/V prefetch distance must be 1 or 2"
#endif

#if SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE != 1 && \
    !SOL_ATTN_PREFETCH_ROUTED_KV
#error "routed K/V prefetch distance 2 requires routed K/V prefetch"
#endif

#if SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE != 1 && \
    (SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH || \
     SOL_ATTN_STAGGER_ROUTED_K_PREFETCH || \
     SOL_ATTN_REGISTER_PIPELINE_EXACT_K)
#error "routed K/V prefetch distance 2 requires the clustered schedule"
#endif

#if SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH && !SOL_ATTN_PREFETCH_ROUTED_KV
#error "SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH requires SOL_ATTN_PREFETCH_ROUTED_KV"
#endif

#if SOL_ATTN_STAGGER_ROUTED_K_PREFETCH && !SOL_ATTN_PREFETCH_ROUTED_KV
#error "SOL_ATTN_STAGGER_ROUTED_K_PREFETCH requires SOL_ATTN_PREFETCH_ROUTED_KV"
#endif

#if SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH && SOL_ATTN_STAGGER_ROUTED_K_PREFETCH
#error "split-distance and staggered routed prefetch are mutually exclusive"
#endif

#if SOL_ATTN_REGISTER_PIPELINE_EXACT_K && !SOL_ATTN_PREFETCH_ROUTED_KV
#error "SOL_ATTN_REGISTER_PIPELINE_EXACT_K requires routed K/V prefetch"
#endif

#if SOL_ATTN_REGISTER_PIPELINE_EXACT_K && \
    (SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH || SOL_ATTN_STAGGER_ROUTED_K_PREFETCH)
#error "register-pipelined exact K is mutually exclusive with prefetch schedules"
#endif


#if SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS && !SOL_ATTN_PREFETCH_ROUTED_KV
#error "SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS requires routed K/V prefetch"
#endif

#if SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE && !SOL_ATTN_SHARED_INLINE_ROUTE
#error "parallel shared route requires shared inline route"
#endif

#if SOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS && \
    !SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
#error "subgroup-reduced shared masks require parallel shared route"
#endif

#if SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID && \
    !SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
#error "cached route query centroid requires parallel shared route"
#endif

#if SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID && \
    !SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS
#error "shared cached route query centroid requires cross-query columns"
#endif

#if SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID && \
    SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID
#error "register and shared route query centroid caches are mutually exclusive"
#endif

#if SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID && \
    !SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
#error "group-loaded route K centroid requires parallel shared route"
#endif


#if SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS && \
    !SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
#error "hierarchical shared route masks require parallel shared route"
#endif

#if SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS && \
    !SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
#error "cross-query route columns require parallel shared route"
#endif

#if SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS && \
    (SOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS || \
     SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS)
#error "cross-query route columns are a standalone route-owner schedule"
#endif

#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS && !SOL_ATTN_PREFETCH_ROUTED_KV
#error "cacheable exact K/V loads require routed K/V prefetch"
#endif

#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
template <class CopyOp>
struct BmgCacheableExactCopyOp;

template <int Bits, int Height, int Width, int BlockWidth>
struct BmgCacheableExactCopyOp<
    cute::XE_LOAD_2D<Bits, Height, Width, BlockWidth>> {
  using type = cute::XE_LOAD_2D_BMG_CACHEABLE<
      Bits, Height, Width, BlockWidth>;
};

template <int Bits, int Height, int Width, int BlockWidth>
struct BmgCacheableExactCopyOp<
    cute::XE_LOAD_2D_VNNI<Bits, Height, Width, BlockWidth>> {
  using type = cute::XE_LOAD_2D_VNNI_BMG_CACHEABLE<
      Bits, Height, Width, BlockWidth>;
};

template <int Bits, int Height, int Width>
struct BmgCacheableExactCopyOp<
    cute::XE_LOAD_2D_TRANSPOSE<Bits, Height, Width>> {
  using type = cute::XE_LOAD_2D_TRANSPOSE_BMG_CACHEABLE<
      Bits, Height, Width>;
};
#endif

template <
    int RouteMaskSlots,
    int QuerySubgroups,
    int BufferCount,
    bool CrossQueryRouteColumns>
struct SolRouteSharedStorage;

template <int RouteMaskSlots, int QuerySubgroups, int BufferCount>
struct SolRouteSharedStorage<
    RouteMaskSlots, QuerySubgroups, BufferCount, false> {
  uint64_t route_masks[RouteMaskSlots * BufferCount];
};

template <int RouteMaskSlots, int QuerySubgroups, int BufferCount>
struct SolRouteSharedStorage<
    RouteMaskSlots, QuerySubgroups, BufferCount, true> {
  // Each owner subgroup produces two adjacent key columns. Every byte stores
  // one bit per Q64 row, so one uint16_t covers two K64 blocks.
  uint16_t route_column_pairs[QuerySubgroups * BufferCount];
};

template <
    class DenseMainloop,
    bool CacheableExactKV = (SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS != 0),
    bool ParallelSharedRoute =
        (SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE != 0),
    bool CrossQueryRouteColumns =
        (SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS != 0)>
struct SolFwdMainloop : DenseMainloop {
  using Base = DenseMainloop;
  using TiledMMAQK = typename Base::TiledMMAQK;
  using TiledMMAPV = typename Base::TiledMMAPV;
  using TileShapeQK = typename Base::TileShapeQK;
  using TileShapePV = typename Base::TileShapePV;
  using SubgroupLayoutQK = typename Base::SubgroupLayoutQK;
  using SGPerWG = typename Base::SGPerWG;

  using TensorQ = typename Base::TensorQ;
  using TensorK = typename Base::TensorK;
  using TensorV = typename Base::TensorV;
  using TensorQ2D = typename Base::TensorQ2D;
  using TensorK2D = typename Base::TensorK2D;
  using TensorV2D = typename Base::TensorV2D;
  using TensorK_cache2D = typename Base::TensorK_cache2D;
  using TensorV_cache2D = typename Base::TensorV_cache2D;
  using ElementQ = typename TensorQ::element_type;
  using ElementK = typename TensorK::element_type;
  using ElementV = typename TensorV::element_type;
  using ElementS = typename Base::ElementS;
  using ElementA = typename Base::ElementA;
  using FragA = typename Base::FragA;
  using FragARow = typename Base::FragARow;
  using FragS = typename Base::FragS;
  using FragSPartialRow = decltype(
      sol_reduce_vertical<1>(FragS{}, sycl::plus<void>{}));

  using TiledCopyQ = typename Base::TiledCopyQ;
  using TiledCopyK = typename Base::TiledCopyK;
  using TiledCopyV = typename Base::TiledCopyV;
  using BaseSharedStorage = typename Base::SharedStorage;

  static constexpr bool BlockScale = false;
  static constexpr bool F8kvF16mma = false;
  static constexpr bool PerTensorScale = false;
  static constexpr bool CausalMask = false;
  static constexpr bool CachedKV = false;
  static constexpr bool PagedKV = false;
  static constexpr int VTiles = Base::VTiles;
  static constexpr int DTiles =
      VTiles * decltype(get<1>(TileShapePV{}))::value /
      decltype(get<2>(TileShapeQK{}))::value;
  static constexpr int BLK_Q = decltype(get<0>(TileShapeQK{}))::value;
  static constexpr int BLK_K = decltype(get<1>(TileShapeQK{}))::value;
  static constexpr int QueryBlocksPerWorkgroup = BLK_Q / 64;
  static constexpr int QuerySubgroups =
      decltype(size<0>(SubgroupLayoutQK{}))::value;
  static constexpr int RouteMaskSlots =
      ParallelSharedRoute
      ? QuerySubgroups
      : QueryBlocksPerWorkgroup;

#if SOL_ATTN_SHARED_INLINE_ROUTE
  struct SharedStorage : BaseSharedStorage,
      SolRouteSharedStorage<
          RouteMaskSlots,
          QuerySubgroups,
          (SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS ? 2 : 1),
          CrossQueryRouteColumns> {
#if SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS
    uint64_t reduced_route_masks[
        QueryBlocksPerWorkgroup *
        (SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS ? 2 : 1)];
#endif
#if SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID
    float route_query_centroids[QueryBlocksPerWorkgroup * 128];
#endif
  };
#else
  using SharedStorage = BaseSharedStorage;
#endif

  static_assert(BLK_Q % 64 == 0,
                "Sol-Attn workgroups must contain whole Q64 routing blocks");
  static_assert(BLK_K == 64, "Sol-Attn exact and summary tiles are K64");
  static_assert(!CrossQueryRouteColumns || ParallelSharedRoute,
                "cross-query columns require parallel route ownership");
  static_assert(!CrossQueryRouteColumns || QueryBlocksPerWorkgroup <= 8,
                "one route-column byte supports at most eight Q64 rows");
  static_assert(!CrossQueryRouteColumns || QuerySubgroups * 2 == BLK_K,
                "packed route-column pairs require two K64 columns per subgroup");
  static_assert(std::is_same_v<ElementQ, cutlass::bfloat16_t>,
                "The initial XPU Sol-Attn CUTE backend is BF16-only");

  struct Arguments {
    ElementS scale;
    const ElementK* k_centroids;
    const ElementV* v_means;
#if SOL_ATTN_INLINE_ROUTE
    const float* q_centroids;
    const float* thresholds;
    const uint8_t* key_sinks;
#else
    const uint8_t* routes;
#endif
    const ElementK* k_base;
    int tokens;
    int heads;
    int blocks;
    int64_t k_stride_batch;
    int64_t k_stride_head;
  };

  struct Params {
    ElementS scale;
    const ElementK* k_centroids;
    const ElementV* v_means;
#if SOL_ATTN_INLINE_ROUTE
    const float* q_centroids;
    const float* thresholds;
    const uint8_t* key_sinks;
#else
    const uint8_t* routes;
#endif
    const ElementK* k_base;
    int tokens;
    int heads;
    int blocks;
    int64_t k_stride_batch;
    int64_t k_stride_head;
  };

  Params sol_params;
#if SOL_ATTN_SHARED_INLINE_ROUTE
  SharedStorage& sol_shared;
#endif

  static Params to_underlying_arguments(Arguments const& args, void*) {
    constexpr double kLog2E = 1.4426950408889634074;
    return {
        ElementS(args.scale * static_cast<ElementS>(kLog2E)),
        args.k_centroids,
        args.v_means,
#if SOL_ATTN_INLINE_ROUTE
        args.q_centroids,
        args.thresholds,
        args.key_sinks,
#else
        args.routes,
#endif
        args.k_base,
        args.tokens,
        args.heads,
        args.blocks,
        args.k_stride_batch,
        args.k_stride_head};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) {
    return args.k_centroids != nullptr && args.v_means != nullptr &&
#if SOL_ATTN_INLINE_ROUTE
        args.q_centroids != nullptr && args.thresholds != nullptr &&
        args.key_sinks != nullptr &&
#else
        args.routes != nullptr &&
#endif
        args.k_base != nullptr && args.tokens > 0 &&
        args.heads > 0 && args.blocks == cute::ceil_div(args.tokens, 64);
  }

  CUTLASS_HOST_DEVICE
  SolFwdMainloop(Params const& params, SharedStorage& shared)
      : Base(
            typename Base::Params{params.scale, nullptr, 0, nullptr},
            static_cast<BaseSharedStorage&>(shared)),
        sol_params(params)
#if SOL_ATTN_SHARED_INLINE_ROUTE
        , sol_shared(shared)
#endif
        {}

  CUTLASS_DEVICE
  auto softmax_deferred_sum(FragS& tS, FragARow& tA_max) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum<void>{});

    FragARow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tA_max.size(); ++i) {
      ElementS new_max =
          sycl::max(tA_max(i), sol_params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tA_max(i) - new_max);
      tA_max(i) = new_max;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); ++i) {
      tS(i) = sycl::native::exp2(
          sol_params.scale * tS(i) - broadcast<0>(tA_max, tS, i));
    }

    auto tS_partial_sum =
        sol_reduce_vertical<1>(tS, sycl::plus<void>{});
    return cute::make_tuple(rescale, tS_partial_sum);
  }

  template <typename QVCoord>
  CUTLASS_DEVICE
  void operator()(
      TensorQ2D const& Q_2D,
      TensorK2D const& K_2D,
      TensorV2D const& V_2D,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int blk_k0,
      int,
      int,
      int thr_id,
      int seq_len,
      int,
      int l_coord,
      int,
      int,
      TensorK_cache2D const& = TensorK_cache2D{},
      TensorV_cache2D const& = TensorV_cache2D{}) {
    using namespace sycl::ext::oneapi::this_work_item;

    constexpr int kQueryBlocksPerWorkgroup = QueryBlocksPerWorkgroup;
    constexpr int kQuerySubgroups =
        decltype(size<0>(SubgroupLayoutQK{}))::value;
    static_assert(kQuerySubgroups % kQueryBlocksPerWorkgroup == 0);
    constexpr int kSubgroupsPerQueryBlock =
        kQuerySubgroups / kQueryBlocksPerWorkgroup;
    const int subgroup_id = thr_id / cute::intel::sg_size;
    const int query_block_unclamped =
        get<0>(blk_qv) * kQueryBlocksPerWorkgroup +
        subgroup_id / kSubgroupsPerQueryBlock;
    const int query_block = cute::min(
        query_block_unclamped, sol_params.blocks - 1);
    const auto* current_k = cute::raw_pointer_cast(K_2D.data());
    const int64_t k_offset = current_k - sol_params.k_base;
    const int64_t head_offset =
        k_offset - static_cast<int64_t>(l_coord) * sol_params.k_stride_batch;
    const int head = static_cast<int>(head_offset / sol_params.k_stride_head);
    const int batch_head = l_coord * sol_params.heads + head;
    const int summary_offset = batch_head * sol_params.blocks * 128;

    auto summary_k = make_tensor(
        make_gmem_ptr(const_cast<ElementK*>(sol_params.k_centroids + summary_offset)),
        make_layout(
            make_shape(sol_params.blocks, 128),
            make_stride(int(128), _1{})));
    auto summary_v = make_tensor(
        make_gmem_ptr(const_cast<ElementV*>(sol_params.v_means + summary_offset)),
        make_layout(
            make_shape(128, sol_params.blocks),
            make_stride(_1{}, int(128))));
#if SOL_ATTN_INLINE_ROUTE
    const float* query_centroid = sol_params.q_centroids +
        summary_offset + query_block * 128;
    const float route_threshold =
        sol_params.thresholds[batch_head * sol_params.blocks + query_block];
    const uint8_t* key_sinks =
        sol_params.key_sinks + batch_head * sol_params.blocks;
#else
    const int route_offset =
        (batch_head * sol_params.blocks + query_block) * sol_params.blocks;
    const uint8_t* route = sol_params.routes + route_offset;
#endif

    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});
    TiledCopyQ copy_q{Q_2D};
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));

    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    std::array<decltype(tSrQ), DTiles> tSrQ_arr;
    FragSPartialRow tA_partial_sum;

    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < DTiles; ++d) {
      copy(copy_q, tQgQ(_, _, _, d), tQrQ);
      reorder(tQrQ, tSrQ_arr[d]);
    }

    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
      clear(tA_partial_sum);
    }

    auto tile_shape_v = make_shape(
        get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cK_exact = make_identity_tensor(K_2D.shape());
    Tensor cV_exact = make_identity_tensor(V_2D.shape());
    Tensor gK_exact = local_tile(
        cK_exact, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV_exact = local_tile(
        cV_exact, tile_shape_v, make_coord(get<1>(blk_qv), _));
    Tensor gV_exact_split = local_tile(
        gV_exact, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});
#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
    auto copy_k_exact = [&]() {
      if constexpr (CacheableExactKV) {
        using ExactKCopyOp = typename BmgCacheableExactCopyOp<
            typename cute::CPY_Op<typename TiledCopyK::Traits>::type>::type;
        return make_block_2d_copy_B(
            ExactKCopyOp{}, TiledMMAQK{}, K_2D);
      } else {
        return TiledCopyK{K_2D};
      }
    }();
    auto copy_v_exact = [&]() {
      if constexpr (CacheableExactKV) {
        using ExactVCopyOp = typename BmgCacheableExactCopyOp<
            typename cute::CPY_Op<typename TiledCopyV::Traits>::type>::type;
        return make_block_2d_copy_B(
            ExactVCopyOp{}, TiledMMAPV{}, V_2D);
      } else {
        return TiledCopyV{V_2D};
      }
    }();
#else
    static_assert(!CacheableExactKV);
    TiledCopyK copy_k_exact{K_2D};
    TiledCopyV copy_v_exact{V_2D};
#endif
    auto thr_copy_k_exact = copy_k_exact.get_slice(thr_id);
    auto thr_copy_v_exact = copy_v_exact.get_slice(thr_id);
    auto tKgK_exact = thr_copy_k_exact.partition_S(gK_exact);
    auto tVgV_exact = thr_copy_v_exact.partition_S(gV_exact_split);

#if SOL_ATTN_PREFETCH_ROUTED_KV
    auto tile_shape_k_prefetch = make_shape(
        get<1>(TileShapeQK{}), get<2>(TileShapeQK{}) * C<DTiles>{});
    Tensor gK_exact_prefetch = local_tile(
        cK_exact, tile_shape_k_prefetch, make_coord(_, 0));
    auto prefetch_k_exact = make_block_2d_prefetch<SGPerWG{}>(
        tile_shape_k_prefetch, K_2D);
    auto prefetch_v_exact = make_block_2d_prefetch<SGPerWG{}>(
        tile_shape_v, V_2D);
    auto pKgK_exact = prefetch_k_exact.get_slice(thr_id).partition_S(
        gK_exact_prefetch);
    auto pVgV_exact = prefetch_v_exact.get_slice(thr_id).partition_S(
        gV_exact);
    auto prefetch_exact_k_tile = [&](int block) {
      prefetch(prefetch_k_exact, pKgK_exact(_, _, _, block));
    };
    auto prefetch_exact_v_tile = [&](int block) {
      prefetch(prefetch_v_exact, pVgV_exact(_, _, _, block));
    };
    auto prefetch_exact_tile = [&](int block) {
#if !SOL_ATTN_REGISTER_PIPELINE_EXACT_K
      prefetch_exact_k_tile(block);
#endif
      prefetch_exact_v_tile(block);
    };
    int deferred_k_prefetch_block = -1;
#endif

    Tensor cK_summary = make_identity_tensor(summary_k.shape());
    Tensor cV_summary = make_identity_tensor(summary_v.shape());
    Tensor gK_summary = local_tile(
        cK_summary, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV_summary = local_tile(
        cV_summary, tile_shape_v, make_coord(get<1>(blk_qv), _));
    Tensor gV_summary_split = local_tile(
        gV_summary, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});
#if SOL_ATTN_BMG_CACHEABLE_SUMMARY_KV_LOADS
    using SummaryKCopyOp = typename BmgCacheableExactCopyOp<
        typename cute::CPY_Op<typename TiledCopyK::Traits>::type>::type;
    auto copy_k_summary = make_block_2d_copy_B(
        SummaryKCopyOp{}, TiledMMAQK{}, summary_k);
    using SummaryVCopyOp = typename BmgCacheableExactCopyOp<
        typename cute::CPY_Op<typename TiledCopyV::Traits>::type>::type;
    auto copy_v_summary = make_block_2d_copy_B(
        SummaryVCopyOp{}, TiledMMAPV{}, summary_v);
#else
    TiledCopyK copy_k_summary{summary_k};
    TiledCopyV copy_v_summary{summary_v};
#endif
    auto thr_copy_k_summary = copy_k_summary.get_slice(thr_id);
    auto thr_copy_v_summary = copy_v_summary.get_slice(thr_id);
    auto tKgK_summary = thr_copy_k_summary.partition_S(gK_summary);
    auto tVgV_summary = thr_copy_v_summary.partition_S(gV_summary_split);

    auto tKrK = thr_copy_k_exact.partition_sg_fragment_D(gK_exact(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK_exact(_, _, 0, 0));
#if SOL_ATTN_REGISTER_PIPELINE_EXACT_K
    // Keep one exact K depth slice in reserve. The LSC load for d+1 is
    // independent of the current DPAS chain, so the EU can cover its latency
    // without issuing a second cache-hint load for the whole K64 tile.
    decltype(tKrK) tKrK_next{};
#endif
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrV = thr_copy_v_exact.partition_sg_fragment_D(
        gV_exact_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(
        gV_exact_split(_, _, 0, 0));
    auto sg = get_sub_group();

#if SOL_ATTN_INLINE_ROUTE
    static_assert(128 % cute::intel::sg_size == 0);
    constexpr int kRouteItemsPerLane = 128 / cute::intel::sg_size;
#if SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID
    // All route-owner subgroups consume the same four Q64 centroid rows.
    // Stage each value from UGM once per workgroup and retain it for every
    // summary chunk; this trades one workgroup-lifetime barrier for removing
    // the cross-subgroup global-load replication.
    constexpr int kRouteQueryCentroidItems =
        kQueryBlocksPerWorkgroup * 128;
    constexpr int kWorkgroupThreads =
        kQuerySubgroups * cute::intel::sg_size;
    for (int index = thr_id; index < kRouteQueryCentroidItems;
         index += kWorkgroupThreads) {
      const int query_slot = index / 128;
      const int dim = index % 128;
      const int route_query_block = cute::min(
          get<0>(blk_qv) * kQueryBlocksPerWorkgroup + query_slot,
          sol_params.blocks - 1);
      sol_shared.route_query_centroids[index] = sol_params.q_centroids[
          summary_offset + route_query_block * 128 + dim];
    }
    sycl::group_barrier(get_work_group<3>());
#elif SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID && \
    SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS
    // Cross-query routing reuses every Q centroid value for two owned key
    // columns in each summary chunk. Keep the four Q64 rows lane-local so
    // that this reuse does not turn into repeated stateless UGM loads.
    std::array<
        std::array<float, kQueryBlocksPerWorkgroup>,
        kRouteItemsPerLane>
        route_query_centroid_columns;
    const int route_lane = static_cast<int>(sg.get_local_linear_id());
    CUTLASS_PRAGMA_UNROLL
    for (int item = 0; item < kRouteItemsPerLane; ++item) {
      const int dim = route_lane + item * cute::intel::sg_size;
      CUTLASS_PRAGMA_UNROLL
      for (int query_slot = 0;
           query_slot < kQueryBlocksPerWorkgroup;
           ++query_slot) {
        const int route_query_block = cute::min(
            get<0>(blk_qv) * kQueryBlocksPerWorkgroup + query_slot,
            sol_params.blocks - 1);
        route_query_centroid_columns[item][query_slot] =
            sol_params.q_centroids[
                summary_offset + route_query_block * 128 + dim];
      }
    }
#elif SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID
    std::array<float, kRouteItemsPerLane> route_query_centroid;
    const int route_lane = static_cast<int>(sg.get_local_linear_id());
    CUTLASS_PRAGMA_UNROLL
    for (int item = 0; item < kRouteItemsPerLane; ++item) {
      route_query_centroid[item] =
          query_centroid[route_lane + item * cute::intel::sg_size];
    }
#endif
    auto route_block = [&](int key_block) {
      const int lane = static_cast<int>(sg.get_local_linear_id());
      float partial = 0.0f;
#if SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID
      // The striped subgroup load returns the same lane + item*SG mapping as
      // the scalar loop below, but emits one cooperative block-read request
      // for the contiguous K128 centroid instead of eight SIMD16 gathers.
      std::array<uint16_t, kRouteItemsPerLane> route_key_centroid;
      auto route_key_centroid_ptr = sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(
              reinterpret_cast<const uint16_t*>(sol_params.k_centroids) +
              summary_offset + key_block * 128);
      sycl::ext::oneapi::experimental::group_load(
          sg,
          route_key_centroid_ptr.get_decorated(),
          sycl::span<uint16_t, kRouteItemsPerLane>(
              route_key_centroid.data(), route_key_centroid.size()),
          sycl::ext::oneapi::experimental::properties{
              sycl::ext::oneapi::experimental::data_placement_striped,
              sycl::ext::oneapi::experimental::contiguous_memory,
              sycl::ext::oneapi::experimental::full_group,
              sycl::ext::oneapi::experimental::alignment<4>});
#endif
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kRouteItemsPerLane; ++item) {
        const int dim = lane + item * cute::intel::sg_size;
        const float query_value =
#if SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID && \
    !SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS
            route_query_centroid[item];
#else
            query_centroid[dim];
#endif
        const ElementK key_value =
#if SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID
            ElementK::bitcast(route_key_centroid[item]);
#else
            sol_params.k_centroids[
                summary_offset + key_block * 128 + dim];
#endif
        partial += query_value * static_cast<float>(key_value);
      }
      const float score = sycl::reduce_over_group(
          sg, partial, sycl::plus<float>()) * sol_params.scale;
      const int distance = query_block > key_block
          ? query_block - key_block
          : key_block - query_block;
      return key_sinks[key_block] != 0 || distance <= 1 ||
          score > route_threshold;
    };
#if SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS
    auto route_columns_for_key = [&](int key_block) {
      const int lane = static_cast<int>(sg.get_local_linear_id());
      std::array<float, kQueryBlocksPerWorkgroup> partial{};
      CUTLASS_PRAGMA_UNROLL
      for (int item = 0; item < kRouteItemsPerLane; ++item) {
        const int dim = lane + item * cute::intel::sg_size;
        const float key_value = static_cast<float>(
            sol_params.k_centroids[
                summary_offset + key_block * 128 + dim]);
        CUTLASS_PRAGMA_UNROLL
        for (int query_slot = 0;
             query_slot < kQueryBlocksPerWorkgroup;
             ++query_slot) {
          const int route_query_block = cute::min(
              get<0>(blk_qv) * kQueryBlocksPerWorkgroup + query_slot,
              sol_params.blocks - 1);
          const float query_value =
#if SOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID
              sol_shared.route_query_centroids[query_slot * 128 + dim];
#elif SOL_ATTN_CACHE_ROUTE_QUERY_CENTROID
              route_query_centroid_columns[item][query_slot];
#else
              sol_params.q_centroids[
                  summary_offset + route_query_block * 128 + dim];
#endif
          partial[query_slot] += query_value * key_value;
        }
      }
      uint8_t columns = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int query_slot = 0;
           query_slot < kQueryBlocksPerWorkgroup;
           ++query_slot) {
        const int route_query_block = cute::min(
            get<0>(blk_qv) * kQueryBlocksPerWorkgroup + query_slot,
            sol_params.blocks - 1);
        const float score = sycl::reduce_over_group(
            sg, partial[query_slot], sycl::plus<float>()) *
            sol_params.scale;
        const float threshold = sol_params.thresholds[
            batch_head * sol_params.blocks + route_query_block];
        const int distance = route_query_block > key_block
            ? route_query_block - key_block
            : key_block - route_query_block;
        if (key_sinks[key_block] != 0 || distance <= 1 ||
            score > threshold) {
          columns |= uint8_t(1) << query_slot;
        }
      }
      return columns;
    };
#endif
#endif

    auto process_tile = [&](auto& copy_k, auto& copy_v, auto& tKgK,
                            auto& tVgV, int tile_index, bool approximate,
                            uint64_t route_mask, int route_begin) {
      clear(tSrS);
      constexpr int kAtomsPerD =
          decltype(get<2>(TileShapeQK{}))::value /
          decltype(get<2>(typename TiledMMAQK::AtomShape_MNK{}))::value;
      auto consume_qk_fragment = [&](int d) {
        auto const& tSrQ_d = tSrQ_arr[d];
        if (d == 0) {
          cute::gemm(mma_qk, tSrQ_d(_, _, 0), tSrK(_, _, 0), tSrS);
          CUTLASS_PRAGMA_UNROLL
          for (int atom = 1; atom < kAtomsPerD; ++atom) {
            cute::gemm(
                mma_qk, tSrQ_d(_, _, atom), tSrK(_, _, atom), tSrS);
          }
        } else {
          cute::gemm(mma_qk, tSrQ_d, tSrK, tSrS);
        }
      };
#if SOL_ATTN_REGISTER_PIPELINE_EXACT_K
      if (!approximate) {
        copy(copy_k, tKgK(_, _, _, tile_index, 0), tKrK);
        reorder(tKrK, tSrK);
        CUTLASS_PRAGMA_UNROLL
        for (int d = 0; d < DTiles; ++d) {
          if (d + 1 < DTiles) {
            copy(copy_k, tKgK(_, _, _, tile_index, d + 1), tKrK_next);
          }
          consume_qk_fragment(d);
          if (d + 1 < DTiles) {
            reorder(tKrK_next, tSrK);
          }
        }
      } else {
#endif
      CUTLASS_PRAGMA_UNROLL
      for (int d = 0; d < DTiles; ++d) {
        copy(copy_k, tKgK(_, _, _, tile_index, d), tKrK);
        reorder(tKrK, tSrK);
        consume_qk_fragment(d);
      }
#if SOL_ATTN_REGISTER_PIPELINE_EXACT_K
      }
#endif

#if SOL_ATTN_STAGGER_ROUTED_K_PREFETCH
      if (!approximate && deferred_k_prefetch_block >= 0) {
        // Match the dense CUTE schedule: V for the next tile is already in
        // flight, while current-tile QK hides the next K prefetch. Softmax and
        // PV then provide the remaining K latency overlap.
        prefetch_exact_k_tile(deferred_k_prefetch_block);
        deferred_k_prefetch_block = -1;
      }
#endif

      const int key_extent = approximate ? sol_params.blocks : seq_len;
      Tensor cScores = make_identity_tensor(make_shape(seq_len, key_extent));
      Tensor gScores = local_tile(
          cScores, take<0, 2>(TileShapeQK{}),
          make_coord(get<0>(blk_qv), tile_index));
      auto score_coords = thr_mma_qk.partition_C(gScores);
#if SOL_ATTN_HOIST_APPROXIMATE_LOG2
      ElementS full_block_log2_bias = ElementS(0.0f);
      ElementS tail_block_log2_bias = ElementS(0.0f);
      if (approximate) {
        full_block_log2_bias = ElementS(6.0f) / sol_params.scale;
        const int tail_block_length =
            seq_len - (sol_params.blocks - 1) * 64;
        tail_block_log2_bias =
            ElementS(sycl::log2(static_cast<float>(tail_block_length))) /
            sol_params.scale;
      }
#endif
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); ++i) {
        const int key = get<1>(score_coords(i));
        if (key >= key_extent) {
          tSrS(i) = ElementS(-INFINITY);
#if SOL_ATTN_INLINE_ROUTE
        } else if (approximate &&
                   ((route_mask >> (key - route_begin)) & uint64_t(1))) {
#else
        } else if (approximate && route[key] != 0) {
#endif
          tSrS(i) = ElementS(-INFINITY);
        } else if (approximate) {
#if SOL_ATTN_HOIST_APPROXIMATE_LOG2
          tSrS(i) += key == sol_params.blocks - 1
              ? tail_block_log2_bias
              : full_block_log2_bias;
#else
          const int block_length = cute::min(64, seq_len - key * 64);
          tSrS(i) += ElementS(sycl::log2(static_cast<float>(block_length))) /
              sol_params.scale;
#endif
        }
      }

      auto [rescale, tS_partial_sum] =
          softmax_deferred_sum(tSrS, tA_max);
      constexpr int kSumSize = decltype(tA_partial_sum.size())::value;
      static_assert(kSumSize % VTiles == 0);
      constexpr int kSumPerVT = kSumSize / VTiles;
      reorder(tSrS, tArP);

      CUTLASS_PRAGMA_UNROLL
      for (int vv = 0; vv < VTiles; ++vv) {
        copy(copy_v, tVgV(_, _, _, vv, tile_index), tVrV);
        reorder(tVrV, tArV);
        CUTLASS_PRAGMA_UNROLL
        for (int i = tArA.size() / VTiles - 1; i >= 0; --i) {
          tArA(_, _, _, vv)(i) *= broadcast<0>(rescale, tArA, i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kSumPerVT; ++j) {
          const int i = vv * kSumPerVT + j;
          tA_partial_sum(i) =
              tA_partial_sum(i) * group_broadcast(sg, rescale(0), i) +
              tS_partial_sum(i);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, vv));
      }
    };

    const int summary_tiles = cute::ceil_div(sol_params.blocks, BLK_K);
    for (int tile = 0; tile < summary_tiles; ++tile) {
      bool has_approximate = false;
      const int begin = tile * BLK_K;
      const int end = cute::min(begin + BLK_K, sol_params.blocks);
#if SOL_ATTN_INLINE_ROUTE
      uint64_t route_mask = 0;
#if SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS
      const int route_buffer_offset =
          (tile & 1) * RouteMaskSlots;
#if SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS
      const int reduced_route_buffer_offset =
          (tile & 1) * kQueryBlocksPerWorkgroup;
#endif
#else
      constexpr int route_buffer_offset = 0;
#if SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS
      constexpr int reduced_route_buffer_offset = 0;
#endif
#endif
#if SOL_ATTN_SHARED_INLINE_ROUTE
#if SOL_ATTN_PREFETCH_ROUTED_KV
      uint64_t union_route_mask = 0;
#endif
      if constexpr (CrossQueryRouteColumns) {
#if SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS
#if SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS
        const int route_pair_buffer_offset =
            (tile & 1) * kQuerySubgroups;
#else
        constexpr int route_pair_buffer_offset = 0;
#endif
        uint16_t route_column_pair = 0;
        const int first_key = begin + subgroup_id * 2;
        if (first_key < end) {
          route_column_pair = static_cast<uint16_t>(
              route_columns_for_key(first_key));
        }
        if (first_key + 1 < end) {
          route_column_pair |= static_cast<uint16_t>(
              route_columns_for_key(first_key + 1)) << 8;
        }
        if (sg.get_local_linear_id() == 0) {
          sol_shared.route_column_pairs[
              route_pair_buffer_offset + subgroup_id] = route_column_pair;
        }
        sycl::group_barrier(get_work_group<3>());

        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int query_slot = subgroup_id / kSubgroupsPerQueryBlock;
        uint64_t partial_route_mask = 0;
#if SOL_ATTN_PREFETCH_ROUTED_KV
        uint64_t partial_union_route_mask = 0;
#endif
        CUTLASS_PRAGMA_UNROLL
        for (int owner = lane; owner < kQuerySubgroups;
             owner += cute::intel::sg_size) {
          const uint16_t pair = sol_shared.route_column_pairs[
              route_pair_buffer_offset + owner];
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < 2; ++column) {
            const int key_offset = owner * 2 + column;
            if (begin + key_offset < end) {
              const uint8_t columns = static_cast<uint8_t>(
                  pair >> (column * 8));
              if ((columns >> query_slot) & uint8_t(1)) {
                partial_route_mask |= uint64_t(1) << key_offset;
              }
#if SOL_ATTN_PREFETCH_ROUTED_KV
              if (columns != 0) {
                partial_union_route_mask |= uint64_t(1) << key_offset;
              }
#endif
            }
          }
        }
        route_mask = sycl::reduce_over_group(
            sg, partial_route_mask, sycl::bit_or<uint64_t>());
#if SOL_ATTN_PREFETCH_ROUTED_KV
        union_route_mask = sycl::reduce_over_group(
            sg, partial_union_route_mask, sycl::bit_or<uint64_t>());
#endif
#else
        static_assert(!CrossQueryRouteColumns);
#endif
      } else if constexpr (ParallelSharedRoute) {
        const int subgroup_in_query_block =
            subgroup_id % kSubgroupsPerQueryBlock;
        uint64_t route_fragment = 0;
        for (int block = begin + subgroup_in_query_block;
             block < end; block += kSubgroupsPerQueryBlock) {
          if (route_block(block)) {
            route_fragment |= uint64_t(1) << (block - begin);
          }
        }
        if (sg.get_local_linear_id() == 0) {
          sol_shared.route_masks[route_buffer_offset + subgroup_id] =
              route_fragment;
        }
        sycl::group_barrier(get_work_group<3>());
        const int query_route_begin =
            (subgroup_id / kSubgroupsPerQueryBlock) *
            kSubgroupsPerQueryBlock;
#if SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS
        const int query_route_slot =
            subgroup_id / kSubgroupsPerQueryBlock;
        if (subgroup_id % kSubgroupsPerQueryBlock == 0 &&
            sg.get_local_linear_id() == 0) {
          uint64_t reduced_route_mask = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int part = 0; part < kSubgroupsPerQueryBlock; ++part) {
            reduced_route_mask |= sol_shared.route_masks[
                route_buffer_offset + query_route_begin + part];
          }
          sol_shared.reduced_route_masks[
              reduced_route_buffer_offset + query_route_slot] =
                  reduced_route_mask;
        }
        sycl::group_barrier(get_work_group<3>());
        route_mask = sol_shared.reduced_route_masks[
            reduced_route_buffer_offset + query_route_slot];
#elif SOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS
        auto reduce_route_slots = [&](int slot_begin, int slot_count) {
          const int lane = static_cast<int>(sg.get_local_linear_id());
          uint64_t partial_mask = 0;
          for (int slot = lane; slot < slot_count;
               slot += cute::intel::sg_size) {
            partial_mask |= sol_shared.route_masks[
                route_buffer_offset + slot_begin + slot];
          }
          return sycl::reduce_over_group(
              sg, partial_mask, sycl::bit_or<uint64_t>());
        };
        route_mask = reduce_route_slots(
            query_route_begin, kSubgroupsPerQueryBlock);
#else
        CUTLASS_PRAGMA_UNROLL
        for (int part = 0; part < kSubgroupsPerQueryBlock; ++part) {
          route_mask |= sol_shared.route_masks[
              route_buffer_offset + query_route_begin + part];
        }
#endif
#if SOL_ATTN_PREFETCH_ROUTED_KV
#if SOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS
        CUTLASS_PRAGMA_UNROLL
        for (int qblock = 0; qblock < kQueryBlocksPerWorkgroup; ++qblock) {
          union_route_mask |= sol_shared.reduced_route_masks[
              reduced_route_buffer_offset + qblock];
        }
#elif SOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS
        union_route_mask = reduce_route_slots(0, RouteMaskSlots);
#else
        CUTLASS_PRAGMA_UNROLL
        for (int qblock = 0; qblock < RouteMaskSlots; ++qblock) {
          union_route_mask |= sol_shared.route_masks[
              route_buffer_offset + qblock];
        }
#endif
#endif
      } else {
        if (subgroup_id % kSubgroupsPerQueryBlock == 0) {
          for (int block = begin; block < end; ++block) {
            if (route_block(block)) {
              route_mask |= uint64_t(1) << (block - begin);
            }
          }
          if (sg.get_local_linear_id() == 0) {
            sol_shared.route_masks[
                route_buffer_offset +
                subgroup_id / kSubgroupsPerQueryBlock] = route_mask;
          }
        }
        sycl::group_barrier(get_work_group<3>());
        route_mask = sol_shared.route_masks[
            route_buffer_offset + subgroup_id / kSubgroupsPerQueryBlock];
#if SOL_ATTN_PREFETCH_ROUTED_KV
        CUTLASS_PRAGMA_UNROLL
        for (int qblock = 0; qblock < RouteMaskSlots; ++qblock) {
          union_route_mask |= sol_shared.route_masks[
              route_buffer_offset + qblock];
        }
#endif
      }
#if SOL_ATTN_PREFETCH_ROUTED_KV && !SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS
      // Every work-item must capture the current union before any subgroup can
      // overwrite shared route slots in the next summary iteration.
      sycl::group_barrier(get_work_group<3>());
#endif
#else
      for (int block = begin; block < end; ++block) {
        if (route_block(block)) {
          route_mask |= uint64_t(1) << (block - begin);
        }
      }
#endif
      has_approximate = static_cast<int>(sycl::popcount(route_mask)) <
          end - begin;
#else
      uint64_t route_mask = 0;
      for (int block = begin; block < end; ++block) {
        has_approximate = has_approximate || route[block] == 0;
      }
#endif
#if SOL_ATTN_PREFETCH_ROUTED_KV
      int furthest_prefetched_block = -1;
      int initial_prefetches = 0;
      for (int block = begin; block < end; ++block) {
        if ((union_route_mask >> (block - begin)) & uint64_t(1)) {
          furthest_prefetched_block = block;
#if SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH
          prefetch_exact_k_tile(block);
#else
          prefetch_exact_tile(block);
#endif
          ++initial_prefetches;
          if (initial_prefetches == SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE) {
            break;
          }
        }
      }
#endif
      if (has_approximate) {
        process_tile(
            copy_k_summary, copy_v_summary, tKgK_summary, tVgV_summary,
            tile, true, route_mask, begin);
      }
#if SOL_ATTN_INLINE_ROUTE
#if SOL_ATTN_PREFETCH_ROUTED_KV
      for (int block = begin; block < end; ++block) {
        if (!((union_route_mask >> (block - begin)) & uint64_t(1))) {
          continue;
        }
#if SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH
        // K is consumed at the start of process_tile, so keep it one exact
        // tile ahead. V is consumed only after QK and softmax; prefetching the
        // current tile here shortens its cache-retention distance while QK
        // still provides useful overlap.
        prefetch_exact_v_tile(block);
#endif
        for (int next = furthest_prefetched_block + 1;
             next < end; ++next) {
          if ((union_route_mask >> (next - begin)) & uint64_t(1)) {
#if SOL_ATTN_SPLIT_ROUTED_KV_PREFETCH
            prefetch_exact_k_tile(next);
#elif SOL_ATTN_STAGGER_ROUTED_K_PREFETCH
            prefetch_exact_v_tile(next);
            deferred_k_prefetch_block = next;
#else
            prefetch_exact_tile(next);
#endif
            furthest_prefetched_block = next;
            break;
          }
        }
        if ((route_mask >> (block - begin)) & uint64_t(1)) {
          process_tile(
              copy_k_exact, copy_v_exact, tKgK_exact, tVgV_exact,
              block, false, 0, 0);
#if SOL_ATTN_STAGGER_ROUTED_K_PREFETCH
        } else if (deferred_k_prefetch_block >= 0) {
          // This subgroup has no current exact work to hide the prefetch
          // behind, but it still owns part of the cooperative next-K request.
          prefetch_exact_k_tile(deferred_k_prefetch_block);
          deferred_k_prefetch_block = -1;
#endif
        }
      }
#else
      for (int block = begin; block < end; ++block) {
        if ((route_mask >> (block - begin)) & uint64_t(1)) {
          process_tile(
              copy_k_exact, copy_v_exact, tKgK_exact, tVgV_exact,
              block, false, 0, 0);
        }
      }
#endif
#elif SOL_ATTN_NESTED_EXACT
      for (int block = begin; block < end; ++block) {
        if (route[block] != 0) {
          process_tile(
              copy_k_exact, copy_v_exact, tKgK_exact, tVgV_exact,
              block, false, 0, 0);
        }
      }
#endif
    }

#if !SOL_ATTN_INLINE_ROUTE && !SOL_ATTN_NESTED_EXACT
    for (int block = 0; block < sol_params.blocks; ++block) {
      if (route[block] != 0) {
        process_tile(
            copy_k_exact, copy_v_exact, tKgK_exact, tVgV_exact,
            block, false, 0, 0);
      }
    }
#endif

    auto tA_sum_full =
        sol_reduce_horizontal(tA_partial_sum, sycl::plus<void>{});
    static_assert(
        decltype(tA_sum_full.size())::value ==
        decltype(tA_sum.size())::value);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tA_sum.size(); ++i) {
      tA_sum(i) = tA_sum_full(i);
    }
  }
};

}  // namespace omni_xpu_sol_attn::cute_backend
