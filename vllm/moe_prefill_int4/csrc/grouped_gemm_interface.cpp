#include "utils.h"
#include "grouped_gemm_interface.h"
#include "xe_2/grouped_gemm_xe2.h"
#include <stdio.h>

torch::Tensor cutlass_grouped_gemm_interface(
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
  return cutlass_grouped_gemm_xe2(
      ptr_A, ptr_B, ptr_scales, ptr_bias, ptr_D,
      expert_first_token_offset, N, K, num_experts,
      is_B_int4, is_B_mxfp4, is_B_int4_kmajor);
}

// Forward declare fused op
torch::Tensor moe_forward_fused_cutlass(
    torch::Tensor x, torch::Tensor topk_weights, torch::Tensor topk_ids,
    torch::Tensor w13, torch::Tensor w13_scales,
    torch::Tensor w2, torch::Tensor w2_scales,
    int64_t num_experts, int64_t top_k);

TORCH_LIBRARY(moe_prefill_gemm, m) {
  m.def("grouped_gemm_int4(Tensor A, Tensor B, Tensor? scales, Tensor? bias, "
        "Tensor D, Tensor expert_first_token_offset, "
        "int N, int K, int num_experts, "
        "bool is_B_int4, bool is_B_mxfp4, "
        "bool is_B_int4_kmajor=False) -> Tensor");
  m.def("moe_forward_fused_cutlass(Tensor x, Tensor topk_weights, Tensor topk_ids, "
        "Tensor w13, Tensor w13_scales, Tensor w2, Tensor w2_scales, "
        "int num_experts, int top_k) -> Tensor");
}

TORCH_LIBRARY_IMPL(moe_prefill_gemm, XPU, m) {
  m.impl("grouped_gemm_int4", &cutlass_grouped_gemm_interface);
  m.impl("moe_forward_fused_cutlass", &moe_forward_fused_cutlass);
}
