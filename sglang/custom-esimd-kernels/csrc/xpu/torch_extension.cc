#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_sglang, m) {
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

  // GGUF q4_0 GEMV: group_size=32, split-half nibble layout (llama.cpp q4_0).
  // weight [N, K/2] uint8, scale [N, K/32] fp16. N/K auto-detected.
  m.def("esimd_gemv_q4_0(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_q4_0", torch::kXPU, &esimd_gemv_q4_0);

  // GGUF q8_0 GEMV: group_size=32, signed int8, symmetric (no min).
  // weight [N, K] int8, scale [N, K/32] fp16. dequant w = d * qs.
  m.def("esimd_gemv_q8_0(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_q8_0", torch::kXPU, &esimd_gemv_q8_0);

  // M-tiled q8_0 dense GEMV (small M, MTP verify): input [M,K], output [M,N].
  m.def("esimd_gemv_q8_0_m(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_q8_0_m", torch::kXPU, &esimd_gemv_q8_0_m);

  // GGUF q4_K GEMV: group_size=32 interleaved, asymmetric (scale + min).
  // weight [N, K/2] u8, scale + min [N, K/32] fp16. dequant w = scale*nib - min.
  m.def("esimd_gemv_q4_k(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor weight_min, Tensor output) -> Tensor");
  m.impl("esimd_gemv_q4_k", torch::kXPU, &esimd_gemv_q4_k);

  // GGUF q5_K GEMV: PACKED (ql nibble + pre-shuffled 1-bit qh), asym scale+min.
  m.def("esimd_gemv_q5_k(Tensor input, Tensor ql, Tensor qh, "
        "Tensor weight_scale, Tensor weight_min, Tensor output) -> Tensor");
  m.impl("esimd_gemv_q5_k", torch::kXPU, &esimd_gemv_q5_k);

  // GGUF q6_K GEMV: PACKED (ql nibble + pre-shuffled 2-bit qh), symmetric g16.
  m.def("esimd_gemv_q6_k(Tensor input, Tensor ql, Tensor qh, "
        "Tensor weight_scale, Tensor output) -> Tensor");
  m.impl("esimd_gemv_q6_k", torch::kXPU, &esimd_gemv_q6_k);

  // M-tiled q6_K GEMV (small M, MTP verify): input [M,K], output [M,N].
  m.def("esimd_gemv_q6_k_m(Tensor input, Tensor ql, Tensor qh, "
        "Tensor weight_scale, Tensor output) -> Tensor");
  m.impl("esimd_gemv_q6_k_m", torch::kXPU, &esimd_gemv_q6_k_m);

  // Fused GGUF k-quant MoE up/gate (Q4_K) -> silu*up.
  m.def("esimd_moe_up_q4k(Tensor x, Tensor gate_ql, Tensor gate_sc, "
        "Tensor gate_mn, Tensor up_ql, Tensor up_sc, Tensor up_mn, Tensor sel, "
        "Tensor inter, int n_tokens, int hidden, int intermediate, int top_k) "
        "-> Tensor");
  m.impl("esimd_moe_up_q4k", torch::kXPU, &esimd_moe_up_q4k);

  // Fused GGUF k-quant MoE down (PACKED), separate Q5_K / Q6_K.
  m.def("esimd_moe_down_q5k(Tensor inter, Tensor ql, Tensor qh, Tensor sc, "
        "Tensor mn, Tensor sel, Tensor topk_w, Tensor out_partial, "
        "int n_tokens, int hidden, int intermediate, int top_k) -> Tensor");
  m.impl("esimd_moe_down_q5k", torch::kXPU, &esimd_moe_down_q5k);
  m.def("esimd_moe_down_q6k(Tensor inter, Tensor ql, Tensor qh, Tensor sc, "
        "Tensor sel, Tensor topk_w, Tensor out_partial, "
        "int n_tokens, int hidden, int intermediate, int top_k) -> Tensor");
  m.impl("esimd_moe_down_q6k", torch::kXPU, &esimd_moe_down_q6k);

  // GGUF q4_0 GEMM (prefill / M>=2) via DPAS. Same interleaved weight layout.
  m.def("esimd_gemm_q4_0(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemm_q4_0", torch::kXPU, &esimd_gemm_q4_0);

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
        "int rotary_dim, Tensor cos_sin_cache, bool normalize_v) -> Tensor");
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

  m.def("esimd_gemv_fp16(Tensor input, Tensor weight, Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp16", torch::kXPU, &esimd_gemv_fp16);

  m.def("esimd_norm_add_norm(Tensor h2_raw, Tensor h1, Tensor w1, Tensor w2, "
        "Tensor out, float eps1, float eps2) -> ()");
  m.impl("esimd_norm_add_norm", torch::kXPU, &esimd_norm_add_norm);

  m.def("esimd_kv_scatter(Tensor k, Tensor v, Tensor(a!) k_cache, "
        "Tensor(b!) v_cache, Tensor indices) -> ()");
  m.impl("esimd_kv_scatter", torch::kXPU, &esimd_kv_scatter);

  m.def("esimd_rmsnorm_residual_scalar(Tensor x, Tensor weight, Tensor residual, "
        "Tensor output, float eps, float scalar) -> Tensor");
  m.impl("esimd_rmsnorm_residual_scalar", torch::kXPU, &esimd_rmsnorm_residual_scalar);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
