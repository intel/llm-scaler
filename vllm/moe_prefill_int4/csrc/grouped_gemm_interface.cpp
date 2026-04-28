#include "utils.h"
#include "grouped_gemm_interface.h"
#include "xe_2/grouped_gemm_xe2.h"
#include <stdio.h>

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A, torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D, torch::Tensor expert_first_token_offset,
    int64_t N, int64_t K, int64_t num_experts,
    bool is_B_int4, bool is_B_mxfp4, bool is_B_int4_kmajor) {
  return cutlass_grouped_gemm_xe2(
      ptr_A, ptr_B, ptr_scales, ptr_bias, ptr_D,
      expert_first_token_offset, N, K, num_experts,
      is_B_int4, is_B_mxfp4, is_B_int4_kmajor);
}

extern torch::Tensor moe_forward_fused_cutlass(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t);

extern torch::Tensor moe_forward_full_fused_cutlass(
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t, int64_t);

TORCH_LIBRARY(moe_prefill_gemm, m) {
  m.def("grouped_gemm_int4(Tensor A, Tensor B, Tensor? scales, Tensor? bias, "
        "Tensor D, Tensor expert_first_token_offset, "
        "int N, int K, int num_experts, "
        "bool is_B_int4, bool is_B_mxfp4, "
        "bool is_B_int4_kmajor=False) -> Tensor");
  m.def("moe_forward_fused_cutlass(Tensor x, Tensor topk_weights, Tensor topk_ids, "
        "Tensor w13, Tensor w13_scales, Tensor w2, Tensor w2_scales, "
        "int num_experts, int top_k) -> Tensor");
  m.def("moe_forward_full_fused_cutlass(Tensor x, Tensor logits, "
        "Tensor w13, Tensor w13_scales, Tensor w2, Tensor w2_scales, "
        "Tensor shared_gate_up_weight, Tensor shared_gate_up_scale, "
        "Tensor shared_down_weight, Tensor shared_down_scale, "
        "Tensor shared_expert_gate_weight, "
        "int num_experts, int top_k) -> Tensor");
}

TORCH_LIBRARY_IMPL(moe_prefill_gemm, XPU, m) {
  m.impl("grouped_gemm_int4", &cutlass_grouped_gemm_interface);
  m.impl("moe_forward_fused_cutlass", &moe_forward_fused_cutlass);
  m.impl("moe_forward_full_fused_cutlass", &moe_forward_full_fused_cutlass);
}
