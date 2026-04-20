#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_vllm, m) {
  m.def("esimd_gemv_fp8_pern(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output, int N, int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern", torch::kXPU, &esimd_gemv_fp8_pern);

  m.def("esimd_gemv_fp8_pern_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, int N0, "
        "Tensor w1, Tensor s1, Tensor o1, int N1, "
        "int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern_fused2", torch::kXPU, &esimd_gemv_fp8_pern_fused2);

  m.def("esimd_gemv_fp8_pern_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, int N0, "
        "Tensor w1, Tensor s1, Tensor o1, int N1, "
        "Tensor w2, Tensor s2, Tensor o2, int N2, "
        "int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern_fused3", torch::kXPU, &esimd_gemv_fp8_pern_fused3);

  // Per-tensor scale variants (N/K inferred from weight shape)
  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);

  m.def("esimd_gemv_fp8_pert_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_fused2", torch::kXPU, &esimd_gemv_fp8_pert_fused2);

  m.def("esimd_gemv_fp8_pert_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1, "
        "Tensor w2, Tensor s2, Tensor o2) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_fused3", torch::kXPU, &esimd_gemv_fp8_pert_fused3);

  // INT4 GEMV with per-group scale (group_size=128)
  // Weight [N, K/2] uint8 packed, scale [N, K/128] fp16. N/K auto-detected.
  m.def("esimd_gemv_int4(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_int4", torch::kXPU, &esimd_gemv_int4);

  // Fused 2-matrix INT4 GEMV (GDN in_proj_qkvz + in_proj_ba)
  m.def("esimd_gemv_int4_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_int4_fused2", torch::kXPU, &esimd_gemv_int4_fused2);

  // Fused QKV Split + RMSNorm + RoPE
  m.def("esimd_qkv_split_norm_rope(Tensor qkv_state, "
        "Tensor q_out, Tensor gate_out, Tensor k_out, Tensor v_out, "
        "Tensor norm_wq, Tensor norm_wk, Tensor positions, "
        "int q_heads, int kv_heads, bool attn_output_gate, "
        "int rotary_dim, Tensor cos_sin_cache) -> Tensor");
  m.impl("esimd_qkv_split_norm_rope", torch::kXPU, &esimd_qkv_split_norm_rope);

  // Fused ResidualAdd + RMSNorm + FP8 GEMV (post_attn_norm + router)
  m.def("esimd_resadd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv_fp8_pert);

  // Fused ResidualAdd + RMSNorm + 2-matrix FP8 GEMV (input_norm + GDN in_proj)
  m.def("esimd_resadd_norm_gemv2_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1, "
        "float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv2_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv2_fp8_pert);

  // Fused RMSNormGated + FP8 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_fp8_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, "
        "int HV, int V, float eps) -> Tensor");
  m.impl("esimd_norm_gemv_fp8_pert", torch::kXPU, &esimd_norm_gemv_fp8_pert);

  // Fused ResidualAdd + RMSNorm + INT4 GEMV (post_attn_norm + router)
  m.def("esimd_resadd_norm_gemv_int4_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_int4_pert", torch::kXPU, &esimd_resadd_norm_gemv_int4_pert);

  // Fused RMSNormGated + INT4 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_int4_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, "
        "int HV, int V, float eps) -> Tensor");
  m.impl("esimd_norm_gemv_int4_pert", torch::kXPU, &esimd_norm_gemv_int4_pert);

  m.def("esimd_fused_add_rms_norm(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm", torch::kXPU, &esimd_fused_add_rms_norm);

  m.def("esimd_rms_norm_gated(Tensor x, Tensor z, Tensor weight, "
        "Tensor output, float eps) -> Tensor");
  m.impl("esimd_rms_norm_gated", torch::kXPU, &esimd_rms_norm_gated);

  m.def("esimd_fused_add_rms_norm_batched(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
