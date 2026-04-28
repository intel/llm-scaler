#include <torch/all.h>
#include "grouped_gemm_xe2.h"
#include "grouped_gemm_xe2_interface.hpp"

torch::Tensor cutlass_grouped_gemm_xe2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4,
    bool is_B_int4_kmajor) {
  return MoE::cutlass_grouped_gemm_xe2_impl(
      ptr_A,
      ptr_B,
      ptr_scales,
      ptr_bias,
      ptr_D,
      expert_first_token_offset,
      N,
      K,
      num_experts,
      is_B_int4,
      is_B_mxfp4,
      is_B_int4_kmajor);
}