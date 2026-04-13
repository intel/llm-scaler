/* torch_extension_moe.cc — Op registration for MoE ESIMD kernels (standard GRF).
 * Uses TORCH_LIBRARY_FRAGMENT to add ops to the existing library namespace.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  // MoE ops — compiled without doubleGRF for 2× occupancy
  m.def("esimd_moe_topk(Tensor router_logits, Tensor top_values, "
        "Tensor top_indices, int T) -> Tensor");
  m.impl("esimd_moe_topk", torch::kXPU, &esimd_moe_topk);

  m.def("esimd_moe_scatter(Tensor hidden_states, Tensor router_top_value, "
        "Tensor sorted_token_ids, "
        "Tensor scattered_hidden, Tensor scattered_weights, "
        "int K, int topk, int total_expanded) -> Tensor");
  m.impl("esimd_moe_scatter", torch::kXPU, &esimd_moe_scatter);

  m.def("esimd_moe_scatter_fused(Tensor hidden_states, Tensor top_values, "
        "Tensor top_indices, "
        "Tensor scattered_hidden, Tensor scattered_weights, "
        "Tensor topk_ids, Tensor expert_start, Tensor max_tokens_out, "
        "int K, int topk, int T, int num_experts) -> Tensor");
  m.impl("esimd_moe_scatter_fused", torch::kXPU, &esimd_moe_scatter_fused);

  m.def("esimd_moe_silu_mul(Tensor input, Tensor output, "
        "int N_gate_up, int N_half, int total_rows) -> Tensor");
  m.impl("esimd_moe_silu_mul", torch::kXPU, &esimd_moe_silu_mul);

  m.def("esimd_moe_gather(Tensor moe_output, Tensor topk_ids, "
        "Tensor scattered_weights, Tensor final_hidden, "
        "int K, int topk, int T) -> Tensor");
  m.impl("esimd_moe_gather", torch::kXPU, &esimd_moe_gather);

  m.def("esimd_moe_gemm_fp8(Tensor input, Tensor weight, Tensor scale, "
        "Tensor output, Tensor expert_idx, "
        "int N, int K, int num_experts, int max_tokens_per_expert) -> Tensor");
  m.impl("esimd_moe_gemm_fp8", torch::kXPU, &esimd_moe_gemm_fp8);

  m.def("esimd_moe_gemm_fp8_pert(Tensor input, Tensor weight, Tensor scale, "
        "Tensor output, Tensor expert_idx, "
        "int N, int K, int num_experts, int max_tokens_per_expert) -> Tensor");
  m.impl("esimd_moe_gemm_fp8_pert", torch::kXPU, &esimd_moe_gemm_fp8_pert);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels_moe() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "custom_esimd_kernels_moe", nullptr, 0, nullptr
    };
    return PyModule_Create(&module);
}
