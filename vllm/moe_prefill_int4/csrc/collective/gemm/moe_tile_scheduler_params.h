/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
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
/*! \file
    \brief Parameters structures for persistent tile schedulers
*/

#include "cutlass/coord.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/workspace.h"
#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace detail {

// Parameters for SM90 persistent group scheduler (only used for Grouped Gemms)
struct PersistentTileSchedulerMoEParams {
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrder;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  FastDivmodU64Pow2 divmod_cluster_shape_major_{};
  FastDivmodU64Pow2 divmod_cluster_shape_minor_{};
  FastDivmodU64 divmod_cta_shape_m_{};
  FastDivmodU64 divmod_cta_shape_n_{};

  uint64_t blocks_across_problem_ = 0;
  bool pre_processed_problem_shapes = false;
  int32_t log_swizzle_size_ = 0;
  RasterOrder raster_order_ = RasterOrder::AlongN;

  GemmCoord cta_shape_;
  GemmCoord cluster_shape_;

  // Version of initialize that takes in as input the number of CTAs in the M
  // and N and L dimensions. This is useful for calculating the tiled shape when
  // a mode of problem and/or CTA shape has rank > 1, for which using CuTe
  // algebra for calculating tile shapes is easiest.
  void initialize(
      dim3 problem_blocks,
      GemmCoord cta_shape,
      GemmCoord cluster_shape,
      KernelHardwareInfo const& hw_info,
      int max_swizzle_size,
      RasterOrderOptions raster_order_option) {
    CUTLASS_UNUSED(hw_info);

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(
        problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m =
        round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n =
        round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    RasterOrder raster_order = get_rasterization_order(
        problem_blocks_m, problem_blocks_n, raster_order_option);

    //
    // Set members
    //
    cta_shape_ = cta_shape;
    cluster_shape_ = cluster_shape;

    blocks_across_problem_ =
        problem_blocks.x * problem_blocks.y * problem_blocks.z;
    pre_processed_problem_shapes = false;
    log_swizzle_size_ = log_swizzle_size;
    raster_order_ = raster_order;

    if (raster_order == RasterOrder::AlongN) {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.m());
    } else {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.n());
    }

    divmod_cta_shape_m_ = FastDivmodU64(cta_shape_.m());
    divmod_cta_shape_n_ = FastDivmodU64(cta_shape_.n());
  }

  // Version of get_tiled_cta_shape_mnl that takes in as input the number of
  // CTAs in the M and N dimensions. This is useful for calculating the tiled
  // shape when a mode of problem and/or CTA shape has rank > 1, for which using
  // CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3 get_tiled_cta_shape_mnl(
      GemmCoord cluster_shape, uint32_t cta_m, uint32_t cta_n) {
    // Round up to nearest multiple of cluster dim along each mode
    auto problem_blocks_m =
        ((cta_m + cluster_shape.m() - 1) / cluster_shape.m()) *
        cluster_shape.m();
    auto problem_blocks_n =
        ((cta_n + cluster_shape.n() - 1) / cluster_shape.n()) *
        cluster_shape.n();

    return {
        static_cast<uint32_t>(cta_m),
        static_cast<uint32_t>(cta_n),
        static_cast<uint32_t>(
            1)  // Only a single batch per group is currently supported
    };
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the
  // M and N and L dimensions. This is useful for calculating the tiled shape
  // when a mode of problem and/or CTA shape has rank > 1, for which using CuTe
  // algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      dim3 problem_blocks,
      GemmCoord cluster_shape,
      KernelHardwareInfo hw_info,
      int max_swizzle_size,
      RasterOrderOptions raster_order_option,
      bool truncate_by_problem_size = true) {
    int const sm_count = hw_info.sm_count;
    int const max_active_clusters = hw_info.max_active_clusters;

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(
        problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m =
        round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n =
        round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    int problem_blocks_total =
        problem_blocks_m * problem_blocks_n * problem_blocks.z;

    RasterOrder raster_order = get_rasterization_order(
        problem_blocks_m, problem_blocks_n, raster_order_option);

    dim3 launch_grid;

    if (raster_order == RasterOrder::AlongN) {
      launch_grid = dim3(cluster_shape.m(), 1, 1);
    } else {
      launch_grid = dim3(1, cluster_shape.n(), 1);
    }

    auto possibly_truncate = [&](int x, int y) {
      if (truncate_by_problem_size) {
        return platform::min(x, y);
      } else {
        return x;
      }
    };

    // The else path is generic, however, we can avoid some divs if we know
    // cluster size is 1
    auto cluster_size = cluster_shape.m() * cluster_shape.n();
    if (cluster_size == 1) {
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(sm_count, problem_blocks_total);
      } else {
        launch_grid.x = possibly_truncate(sm_count, problem_blocks_total);
      }
    }
    // In case the maximum number of clusters that could co-exist on the target
    // device is already calculated using cudaOccupancyMaxActiveClusters
    else if (
        max_active_clusters != 0 &&
        max_active_clusters * cluster_size <= sm_count) {
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = max_active_clusters * cluster_shape.n();
      } else {
        launch_grid.x = max_active_clusters * cluster_shape.m();
      }
      CUTLASS_TRACE_HOST(
          "get_grid_shape(): Proposed GridDims by the scheduler using "
          "cudaOccupancyMaxActiveClusters = "
          "("
          << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z
          << ")\n");
    } else {
      // Optimal grid size calculation is based on
      // GH100: 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU
      // Hence, maximum SMs per GPC = 18
      constexpr int max_sm_per_gpc = 18;
      int cta_per_device =
          get_max_cta_occupancy(max_sm_per_gpc, cluster_shape, sm_count);

      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(
            cta_per_device / cluster_shape.m(),
            problem_blocks_total / cluster_shape.m());
      } else {
        launch_grid.x = possibly_truncate(
            cta_per_device / cluster_shape.n(),
            problem_blocks_total / cluster_shape.n());
      }
      CUTLASS_TRACE_HOST(
          "get_grid_shape(): Proposed GridDims by the scheduler using "
          "heuristics = "
          "("
          << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z
          << ")\n");
    }
    return launch_grid;
  }

  CUTLASS_HOST_DEVICE
  static int32_t get_log_swizzle_size(
      int problem_ctas_m, int problem_ctas_n, int max_swizzle_size) {
    int min_cta_dim = platform::min(problem_ctas_m, problem_ctas_n);
    if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
      return 3;
    } else if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
      return 2;
    } else if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
      return 1;
    } else {
      return 0;
    }
  }

  CUTLASS_HOST_DEVICE
  static RasterOrder get_rasterization_order(
      uint32_t tiles_m,
      uint32_t tiles_n,
      RasterOrderOptions raster_order_option) {
    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      } else {
        return RasterOrder::AlongN;
      }
    } else {
      switch (raster_order_option) {
        case RasterOrderOptions::AlongN:
          return RasterOrder::AlongN;
          break;
        default:
          return RasterOrder::AlongM;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace detail
}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
