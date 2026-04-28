#pragma once
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include <cfloat>

#include "moe_array_mma.hpp"
#include "moe_array_epilogue.hpp"
#include "moe_callbacks.hpp"
#include "moe_dtype_policy.hpp"
#include "moe_gemm_array_cooperative.hpp"
#include "moe_tile_scheduler.hpp"
using namespace cute;

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

class moe_policy_base {
 public:
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = float;
  using ElementB = float;
  using ElementOutput = float;
  using ElementScale = float;
  using TileShape = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

class moe_bf16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
};

class moe_bf16_decode_policy : public moe_bf16_policy {
 public:
  using TileShape = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class moe_fp16_policy : public moe_policy_base {
 public:
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementScale = cutlass::half_t;
};

class moe_fp16_decode_policy : public moe_fp16_policy {
 public:
  using TileShape = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

}  // namespace grouped_gemm
}  // namespace gpu::cutlass_kernel
