/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * Copyright (C) 2026 Sol-Attn XPU contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Torch wrapper for the Triton-free Sol-Attn CUTE/DPAS mainloop.
 * Launch and type-assembly patterns follow SYCL-TLA example 06 and the
 * maintained omni_xpu_kernel CUTE wrapper.
 **************************************************************************************************/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/util/compat.hpp"

#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

#include "sol_attn_config.h"
#include "sol_attn_mainloop.hpp"

namespace omni_xpu_sol_attn::cute_backend {

using namespace cute;

#ifndef SOL_ATTN_Q_TILE
#define SOL_ATTN_Q_TILE 128
#endif

#ifndef SOL_ATTN_SUBGROUP_LAYOUT_Q
#define SOL_ATTN_SUBGROUP_LAYOUT_Q 16
#endif

#ifndef SOL_ATTN_GRF_SIZE
#define SOL_ATTN_GRF_SIZE 256
#endif

#ifndef SOL_ATTN_NESTED_EXACT
#define SOL_ATTN_NESTED_EXACT 0
#endif

#ifndef SOL_ATTN_INLINE_ROUTE
#define SOL_ATTN_INLINE_ROUTE 0
#endif

#ifndef SOL_ATTN_PAIRED_Q256_SCHEDULER
#define SOL_ATTN_PAIRED_Q256_SCHEDULER 0
#endif

template <typename Kernel, int GrfSize>
class SolCuteKernelTag {};

template <typename Kernel, int GrfSize>
class SolParentKernelTag {};

template <typename Kernel, int GrfSize = 256, bool ParentTag = false>
void launch_on_torch_queue(typename Kernel::Params params, int device_index) {
  static_assert(GrfSize == 128 || GrfSize == 256);
  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>,
      intelex::grf_size<GrfSize>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  syclex::launch_config config(
      policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](sycl::handler& cgh) {
    auto functor =
        compat::experimental::detail::build_kernel_functor<
            cutlass::device_kernel<Kernel>>(cgh, policy, params);
    syclex::detail::LaunchConfigAccess<
        sycl::nd_range<3>, decltype(policy.get_launch_properties())>
        config_access(config);
    if constexpr (ParentTag) {
      cgh.parallel_for<SolParentKernelTag<Kernel, GrfSize>>(
          config_access.getRange(), config_access.getProperties(), functor);
    } else {
      cgh.parallel_for<SolCuteKernelTag<Kernel, GrfSize>>(
          config_access.getRange(), config_access.getProperties(), functor);
    }
  };
  c10::xpu::getCurrentXPUStream(device_index).queue().submit(cgf);
}

int checked_int(int64_t value, const char* label) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              label, " exceeds the CUTE int32 index range: ", value);
  return static_cast<int>(value);
}

#if SOL_ATTN_PAIRED_Q256_SCHEDULER
// Opt-in locality experiment: each workgroup processes two adjacent Q256
// tiles in the same descending order as XeFHMAIndividualTileScheduler.  The
// kernel-owned scheduler loop completes the mainloop and epilogue for one tile
// before advancing, so accumulator and output lifetimes do not cross tiles.
struct SolPairedQ256TileScheduler {
  struct Params {
    cute::dim3 grid;
    int q_tiles;
    cutlass::FastDivmod divmod_num_heads;
    cutlass::FastDivmod divmod_head_group_q;
  };

  Params params;
  int tile_in_pair = 0;

  CUTLASS_DEVICE
  SolPairedQ256TileScheduler(Params const& params_) : params(params_) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      cutlass::KernelHardwareInfo,
      TileShape const& tile_shape) {
    const int q_tiles = cute::ceil_div(
        shape.seq_len_qo, cute::get<0>(tile_shape));
    cute::dim3 grid(
        cute::ceil_div(shape.head_size_vo, cute::get<1>(tile_shape)),
        cute::ceil_div(q_tiles, 2),
        shape.batch * shape.num_heads_q);
    return Params{
        grid,
        q_tiles,
        {shape.num_heads_q},
        {shape.num_heads_q / shape.num_heads_kv}};
  }

  template <int NumSGs>
  static cute::dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  int current_q_tile() const {
    return params.q_tiles - 1 -
        (2 * int(::BlockIdxY()) + tile_in_pair);
  }

  CUTLASS_DEVICE
  bool is_valid() const {
    return tile_in_pair < 2 && current_q_tile() >= 0;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    int head;
    int batch = int(::BlockIdxZ());
    params.divmod_num_heads(batch, head, batch);
    return cute::make_coord(
        current_q_tile(), int(::BlockIdxX()), head, batch);
  }

  CUTLASS_DEVICE
  int divide_head_group(int head_q) const {
    return params.divmod_head_group_q.div(head_q);
  }

  CUTLASS_DEVICE
  SolPairedQ256TileScheduler& operator++() {
    ++tile_in_pair;
    return *this;
  }
};
#endif

template <
    typename Element,
    bool CacheableExactKV = (SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS != 0),
    bool ParallelSharedRoute =
        (SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE != 0),
    bool CrossQueryRouteColumns =
        (SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS != 0)>
struct SolKernel {
  static constexpr int QTile = SOL_ATTN_Q_TILE;
  static constexpr int KvTile = 64;
  static constexpr int VTile = 32;
  static constexpr int MmaK = 16;
  static constexpr int HeadDim = 128;
  static constexpr int SubgroupLayoutQ = SOL_ATTN_SUBGROUP_LAYOUT_Q;
  static constexpr int PipelineStages = 1;
  static constexpr int GrfSize = SOL_ATTN_GRF_SIZE;

  static_assert(QTile % 64 == 0,
                "Sol-Attn Q tile must contain whole Q64 route blocks");
  static_assert(SubgroupLayoutQ % (QTile / 64) == 0,
                "subgroup layout must divide evenly across Q64 route blocks");
  static_assert(GrfSize == 128 || GrfSize == 256,
                "Sol-Attn GRF size must be 128 or 256");

  using ShapeQK = Shape<Int<QTile>, Int<KvTile>, Int<MmaK>>;
  using ShapePV = Shape<Int<QTile>, Int<VTile>, Int<KvTile>>;
  using ShapeOutput = Shape<Int<QTile>, Int<HeadDim>>;
  using SubgroupLayoutQK = Layout<Shape<Int<SubgroupLayoutQ>, _1, _1>>;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  static constexpr int SGTileQ =
      get<0>(shape_div(ShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Element>;
  using SubgroupLayoutPV = decltype(
      cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
  using TiledMMAQK = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
  static constexpr int VTiles = get<1>(ShapeOutput{}) / get<1>(ShapePV{});

  using TensorQ = decltype(make_tensor(
      make_gmem_ptr((Element*)nullptr),
      make_layout(repeat<rank_v<StrideQ>>(1), StrideQ{})));
  using TensorK = decltype(make_tensor(
      make_gmem_ptr((Element*)nullptr),
      make_layout(repeat<rank_v<StrideK>>(1), StrideK{})));
  using TensorV = decltype(make_tensor(
      make_gmem_ptr((Element*)nullptr),
      make_layout(repeat<rank_v<StrideV>>(1), StrideV{})));
  using TensorO = decltype(make_tensor(
      make_gmem_ptr((Element*)nullptr),
      make_layout(repeat<rank_v<StrideO>>(1), StrideO{})));

  using DenseMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
      cutlass::fmha::XeDefault<PipelineStages>,
      false, false, false,
      TiledMMAQK, TiledMMAPV, VTiles,
      TensorQ, TensorK, TensorV,
      TensorK, TensorV,
      void, void, void, void, void>;
  using CollectiveMainloop = SolFwdMainloop<
      DenseMainloop,
      CacheableExactKV,
      ParallelSharedRoute,
      CrossQueryRouteColumns>;
  using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      CollectiveMainloop, ShapeOutput, TensorO, void>;
  using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
#if SOL_ATTN_PAIRED_Q256_SCHEDULER
  static_assert(QTile == 256,
                "paired-Q256 scheduler requires a Q256 kernel tile");
  static_assert(
      is_same_v<typename CollectiveEpilogue::ReduceK, _1>,
      "paired-Q256 scheduler requires every subgroup to finish each epilogue");
  static_assert(
      is_empty_v<typename CollectiveEpilogue::SharedStorage>,
      "paired-Q256 scheduler requires no cross-tile epilogue SLM lifetime");
  using TileScheduler = SolPairedQ256TileScheduler;
#else
  using TileScheduler =
      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler;
#endif
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue,
      TileScheduler>;
};

template <
    typename Element,
    bool CacheableExactKV,
    bool ParallelSharedRoute,
    bool CrossQueryRouteColumns,
    bool ParentTag = false>
void run(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_centroids,
    const at::Tensor& v_means,
#if SOL_ATTN_INLINE_ROUTE
    const at::Tensor& q_centroids,
    const at::Tensor& thresholds,
    const at::Tensor& key_sinks,
#else
    const at::Tensor& routes,
#endif
    at::Tensor& output,
    float scale) {
  using KT = SolKernel<
      Element,
      CacheableExactKV,
      ParallelSharedRoute,
      CrossQueryRouteColumns>;
  using K = typename KT::Kernel;

  const int B = checked_int(q.size(0), "batch");
  const int T = checked_int(q.size(1), "sequence length");
  const int H = checked_int(q.size(2), "head count");
  const int D = checked_int(q.size(3), "head dimension");
  const int blocks = checked_int(k_centroids.size(2), "block count");

  typename KT::ProblemShape shape{};
  shape.batch = B;
  shape.num_heads_q = H;
  shape.num_heads_kv = H;
  shape.seq_len_qo = T;
  shape.seq_len_kv = T;
  shape.seq_len_kv_cache = 0;
  shape.head_size_qk = D;
  shape.head_size_vo = D;

  typename KT::StrideQ stride_q = cute::make_stride(
      checked_int(q.stride(1), "Q sequence stride"), _1{},
      checked_int(q.stride(2), "Q head stride"),
      checked_int(q.stride(0), "Q batch stride"));
  typename KT::StrideK stride_k = cute::make_stride(
      checked_int(k.stride(1), "K sequence stride"), _1{},
      checked_int(k.stride(2), "K head stride"),
      checked_int(k.stride(0), "K batch stride"));
  typename KT::StrideV stride_v = cute::make_stride(
      _1{}, checked_int(v.stride(1), "V sequence stride"),
      checked_int(v.stride(2), "V head stride"),
      checked_int(v.stride(0), "V batch stride"));
  typename KT::StrideO stride_o = cute::make_stride(
      checked_int(output.stride(1), "output sequence stride"), _1{},
      checked_int(output.stride(2), "output head stride"),
      checked_int(output.stride(0), "output batch stride"));

  cutlass::KernelHardwareInfo hw_info{};
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename K::Arguments arguments{};
  arguments.kernel.shape = shape;
  arguments.kernel.Q = static_cast<const Element*>(q.data_ptr());
  arguments.kernel.dQ = stride_q;
  arguments.kernel.K = static_cast<const Element*>(k.data_ptr());
  arguments.kernel.dK = stride_k;
  arguments.kernel.V = static_cast<const Element*>(v.data_ptr());
  arguments.kernel.dV = stride_v;
  arguments.kernel.O = static_cast<Element*>(output.data_ptr());
  arguments.kernel.dO = stride_o;
  arguments.kernel.K_cache = nullptr;
  arguments.kernel.dK_cache = stride_k;
  arguments.kernel.V_cache = nullptr;
  arguments.kernel.dV_cache = stride_v;
  arguments.mainloop = {
      scale,
      static_cast<const Element*>(k_centroids.data_ptr()),
      static_cast<const Element*>(v_means.data_ptr()),
#if SOL_ATTN_INLINE_ROUTE
      q_centroids.data_ptr<float>(),
      thresholds.data_ptr<float>(),
      key_sinks.data_ptr<uint8_t>(),
#else
      routes.data_ptr<uint8_t>(),
#endif
      static_cast<const Element*>(k.data_ptr()),
      T,
      H,
      blocks,
      k.stride(0),
      k.stride(2)};
  arguments.hw_info = hw_info;

  TORCH_CHECK(K::can_implement(arguments),
              "omni_xpu_sol_attn CUTE mainloop cannot implement this contract");
  const size_t workspace_size = K::get_workspace_size(arguments);
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)},
      q.options().dtype(at::kByte));
  K::initialize_workspace(arguments, workspace.data_ptr());
  auto params = K::to_underlying_arguments(arguments, workspace.data_ptr());
  launch_on_torch_queue<K, KT::GrfSize, ParentTag>(
      params, q.device().index());
}

template <
    bool CacheableExactKV,
    bool ParallelSharedRoute,
    bool CrossQueryRouteColumns,
    bool ParentTag = false>
at::Tensor forward_cute_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_centroids,
    const at::Tensor& v_means,
#if SOL_ATTN_INLINE_ROUTE
    const at::Tensor& q_centroids,
    const at::Tensor& thresholds,
    const at::Tensor& key_sinks,
#else
    const at::Tensor& routes,
#endif
    double scale_value) {
  TORCH_CHECK(q.device().is_xpu() && k.device() == q.device() && v.device() == q.device(),
              "Sol-Attn CUTE requires Q/K/V on one XPU device");
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "Sol-Attn CUTE requires matching BTHD Q/K/V");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "Sol-Attn CUTE is BF16-only");
  TORCH_CHECK(q.size(1) > 0 && q.size(3) == 128,
              "Sol-Attn CUTE requires T>0 and head_dim 128");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "Sol-Attn CUTE requires a contiguous head dimension");
  TORCH_CHECK(std::isfinite(scale_value), "scale must be finite");

  const int64_t blocks = (q.size(1) + 63) / 64;
  TORCH_CHECK(k_centroids.device() == q.device() && v_means.device() == q.device()
#if SOL_ATTN_INLINE_ROUTE
                  && q_centroids.device() == q.device() &&
                  thresholds.device() == q.device() && key_sinks.device() == q.device(),
#else
                  && routes.device() == q.device(),
#endif
              "Sol-Attn summaries/routes must be on the Q device");
  TORCH_CHECK(k_centroids.scalar_type() == q.scalar_type() &&
                  v_means.scalar_type() == q.scalar_type() &&
                  k_centroids.sizes() == at::IntArrayRef(
                      {q.size(0), q.size(2), blocks, q.size(3)}) &&
                  v_means.sizes() == k_centroids.sizes() &&
                  k_centroids.is_contiguous() && v_means.is_contiguous(),
              "Sol-Attn K-centroid/V-mean contract mismatch");
#if SOL_ATTN_INLINE_ROUTE
  TORCH_CHECK(q_centroids.scalar_type() == at::kFloat &&
                  q_centroids.is_contiguous() &&
                  q_centroids.sizes() == k_centroids.sizes(),
              "Sol-Attn Q-centroid contract mismatch");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat &&
                  thresholds.is_contiguous() &&
                  thresholds.sizes() == at::IntArrayRef(
                      {q.size(0), q.size(2), blocks}),
              "Sol-Attn threshold contract mismatch");
  TORCH_CHECK(key_sinks.scalar_type() == at::kByte &&
                  key_sinks.is_contiguous() &&
                  key_sinks.sizes() == thresholds.sizes(),
              "Sol-Attn key-sink contract mismatch");
#else
  TORCH_CHECK(routes.scalar_type() == at::kByte && routes.is_contiguous() &&
                  routes.sizes() == at::IntArrayRef(
                      {q.size(0), q.size(2), blocks, blocks}),
              "Sol-Attn route contract mismatch");
#endif
  auto output = at::empty(q.sizes(), q.options());
  run<
      cutlass::bfloat16_t,
      CacheableExactKV,
      ParallelSharedRoute,
      CrossQueryRouteColumns,
      ParentTag>(
      q, k, v, k_centroids, v_means,
#if SOL_ATTN_INLINE_ROUTE
      q_centroids, thresholds, key_sinks,
#else
      routes,
#endif
      output,
      static_cast<float>(scale_value));
  return output;
}

at::Tensor forward_cute(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_centroids,
    const at::Tensor& v_means,
#if SOL_ATTN_INLINE_ROUTE
    const at::Tensor& q_centroids,
    const at::Tensor& thresholds,
    const at::Tensor& key_sinks,
#else
    const at::Tensor& routes,
#endif
    double scale_value) {
  return forward_cute_impl<
      (SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS != 0),
      (SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE != 0),
      (SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS != 0),
      false>(
      q, k, v, k_centroids, v_means,
#if SOL_ATTN_INLINE_ROUTE
      q_centroids, thresholds, key_sinks,
#else
      routes,
#endif
      scale_value);
}

#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
at::Tensor forward_cute_parent(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_centroids,
    const at::Tensor& v_means,
#if SOL_ATTN_INLINE_ROUTE
    const at::Tensor& q_centroids,
    const at::Tensor& thresholds,
    const at::Tensor& key_sinks,
#else
    const at::Tensor& routes,
#endif
    double scale_value) {
  return forward_cute_impl<
      false,
      (SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE != 0),
      (SOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS != 0),
      true>(
      q, k, v, k_centroids, v_means,
#if SOL_ATTN_INLINE_ROUTE
      q_centroids, thresholds, key_sinks,
#else
      routes,
#endif
      scale_value);
}
#endif

#if SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
at::Tensor forward_cute_serial_route_parent(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_centroids,
    const at::Tensor& v_means,
#if SOL_ATTN_INLINE_ROUTE
    const at::Tensor& q_centroids,
    const at::Tensor& thresholds,
    const at::Tensor& key_sinks,
#else
    const at::Tensor& routes,
#endif
    double scale_value) {
  return forward_cute_impl<
      (SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS != 0),
      false,
      false,
      true>(
      q, k, v, k_centroids, v_means,
#if SOL_ATTN_INLINE_ROUTE
      q_centroids, thresholds, key_sinks,
#else
      routes,
#endif
      scale_value);
}
#endif

}  // namespace omni_xpu_sol_attn::cute_backend

TORCH_LIBRARY_FRAGMENT(omni_xpu_sol_attn, m) {
#if SOL_ATTN_INLINE_ROUTE
  m.def(
      "forward_cute(Tensor q, Tensor k, Tensor v, Tensor k_centroids, "
      "Tensor v_means, Tensor q_centroids, Tensor thresholds, "
      "Tensor key_sinks, float scale) -> Tensor");
#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
  m.def(
      "forward_cute_parent(Tensor q, Tensor k, Tensor v, Tensor k_centroids, "
      "Tensor v_means, Tensor q_centroids, Tensor thresholds, "
      "Tensor key_sinks, float scale) -> Tensor");
#endif
#if SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
  m.def(
      "forward_cute_serial_route_parent(Tensor q, Tensor k, Tensor v, "
      "Tensor k_centroids, Tensor v_means, Tensor q_centroids, "
      "Tensor thresholds, Tensor key_sinks, float scale) -> Tensor");
#endif
#else
  m.def(
      "forward_cute(Tensor q, Tensor k, Tensor v, Tensor k_centroids, "
      "Tensor v_means, Tensor routes, float scale) -> Tensor");
#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
  m.def(
      "forward_cute_parent(Tensor q, Tensor k, Tensor v, Tensor k_centroids, "
      "Tensor v_means, Tensor routes, float scale) -> Tensor");
#endif
#if SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
  m.def(
      "forward_cute_serial_route_parent(Tensor q, Tensor k, Tensor v, "
      "Tensor k_centroids, Tensor v_means, Tensor routes, float scale) "
      "-> Tensor");
#endif
#endif
}

TORCH_LIBRARY_IMPL(omni_xpu_sol_attn, XPU, m) {
  m.impl("forward_cute", &omni_xpu_sol_attn::cute_backend::forward_cute);
#if SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS
  m.impl(
      "forward_cute_parent",
      &omni_xpu_sol_attn::cute_backend::forward_cute_parent);
#endif
#if SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE
  m.impl(
      "forward_cute_serial_route_parent",
      &omni_xpu_sol_attn::cute_backend::forward_cute_serial_route_parent);
#endif
}
