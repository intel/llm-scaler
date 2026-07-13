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
  // FP16 GEMV (no scale): for fp16 GateLinear-style decode router projection
  m.def("esimd_gemv_fp16(Tensor input, Tensor weight, Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp16", torch::kXPU, &esimd_gemv_fp16);

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
        "Tensor(a!) output) -> ()");
  m.impl("esimd_gemv_int4", torch::kXPU,
         [](at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
            at::Tensor output) -> void {
           esimd_gemv_int4(input, weight, weight_scale, output);
         });

  // Fused 2-matrix INT4 GEMV (GDN in_proj_qkvz + in_proj_ba)
  m.def("esimd_gemv_int4_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor(a!) o0, "
        "Tensor w1, Tensor s1, Tensor(b!) o1) -> ()");
  m.impl("esimd_gemv_int4_fused2", torch::kXPU,
         [](at::Tensor input, at::Tensor w0, at::Tensor s0, at::Tensor o0,
            at::Tensor w1, at::Tensor s1, at::Tensor o1) -> void {
           esimd_gemv_int4_fused2(input, w0, s0, o0, w1, s1, o1);
         });

  // Fused QKV Split + RMSNorm + RoPE
  // q_out/gate_out/k_out/v_out are written in place (mutated). Declared with
  // (a!)..(d!) + void return so torch.compile's auto-functionalize (v1) can
  // trace it -- a `-> Tensor(a!)` alias return is rejected by v1, hence the
  // lambda-wrapped void impl that discards the kernel's (aliased) return.
  m.def("esimd_qkv_split_norm_rope(Tensor qkv_state, "
        "Tensor(a!) q_out, Tensor(b!) gate_out, Tensor(c!) k_out, Tensor(d!) v_out, "
        "Tensor norm_wq, Tensor norm_wk, Tensor positions, "
        "int q_heads, int kv_heads, bool attn_output_gate, "
        "int rotary_dim, Tensor cos_sin_cache) -> ()");
  m.impl("esimd_qkv_split_norm_rope", torch::kXPU,
         [](at::Tensor qkv_state, at::Tensor q_out, at::Tensor gate_out,
            at::Tensor k_out, at::Tensor v_out, at::Tensor norm_wq,
            at::Tensor norm_wk, at::Tensor positions, int64_t q_heads,
            int64_t kv_heads, bool attn_output_gate, int64_t rotary_dim,
            at::Tensor cos_sin_cache) -> void {
           esimd_qkv_split_norm_rope(qkv_state, q_out, gate_out, k_out, v_out,
                                     norm_wq, norm_wk, positions, q_heads,
                                     kv_heads, attn_output_gate, rotary_dim,
                                     cos_sin_cache);
         });

  // Variant with V-Norm (gemma4): same as above but also RMSNorms V heads.
  m.def("esimd_qkv_split_norm_rope_v(Tensor qkv_state, "
        "Tensor(a!) q_out, Tensor(b!) gate_out, Tensor(c!) k_out, Tensor(d!) v_out, "
        "Tensor norm_wq, Tensor norm_wk, Tensor norm_wv, Tensor positions, "
        "int q_heads, int kv_heads, bool attn_output_gate, "
        "int rotary_dim, Tensor cos_sin_cache) -> ()");
  m.impl("esimd_qkv_split_norm_rope_v", torch::kXPU,
         [](at::Tensor qkv_state, at::Tensor q_out, at::Tensor gate_out,
            at::Tensor k_out, at::Tensor v_out, at::Tensor norm_wq,
            at::Tensor norm_wk, at::Tensor norm_wv, at::Tensor positions,
            int64_t q_heads, int64_t kv_heads, bool attn_output_gate,
            int64_t rotary_dim, at::Tensor cos_sin_cache) -> void {
           esimd_qkv_split_norm_rope_v(qkv_state, q_out, gate_out, k_out, v_out,
                                       norm_wq, norm_wk, norm_wv, positions,
                                       q_heads, kv_heads, attn_output_gate,
                                       rotary_dim, cos_sin_cache);
         });

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
  m.def("esimd_resadd_norm_gemv_int4_pert(Tensor hidden_states, Tensor(a!) residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor(b!) output, Tensor(c!) normed_out, float eps) -> ()");
  m.impl("esimd_resadd_norm_gemv_int4_pert", torch::kXPU,
         [](at::Tensor hidden_states, at::Tensor residual,
            at::Tensor norm_weight, at::Tensor gemv_weight,
            at::Tensor gemv_scale, at::Tensor output, at::Tensor normed_out,
            double eps) -> void {
           esimd_resadd_norm_gemv_int4_pert(hidden_states, residual, norm_weight,
                                            gemv_weight, gemv_scale, output,
                                            normed_out, eps);
         });

  // Fused RMSNormGated + INT4 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_int4_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor(a!) output, "
        "int HV, int V, float eps) -> ()");
  m.impl("esimd_norm_gemv_int4_pert", torch::kXPU,
         [](at::Tensor x, at::Tensor z, at::Tensor norm_weight,
            at::Tensor gemv_weight, at::Tensor gemv_scale, at::Tensor output,
            int64_t HV, int64_t V, double eps) -> void {
           esimd_norm_gemv_int4_pert(x, z, norm_weight, gemv_weight,
                                     gemv_scale, output, HV, V, eps);
         });

  // Standalone RMSNorm (no add): for the spots where input differs from
  // the accumulating residual stream (post_attn_norm, post_ff_norm_1, etc.).
  m.def("esimd_rms_norm(Tensor input, Tensor output, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_rms_norm", torch::kXPU, &esimd_rms_norm);

  m.def("esimd_fused_add_rms_norm(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm", torch::kXPU, &esimd_fused_add_rms_norm);

  // Scaled variant: residual <- (hs+r)*scalar; hidden <- norm(residual)*weight
  m.def("esimd_fused_scaled_add_rms_norm(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps, float scalar) -> Tensor");
  m.impl("esimd_fused_scaled_add_rms_norm", torch::kXPU, &esimd_fused_scaled_add_rms_norm);

  m.def("esimd_rms_norm_gated(Tensor x, Tensor z, Tensor weight, "
        "Tensor output, float eps) -> Tensor");
  m.impl("esimd_rms_norm_gated", torch::kXPU, &esimd_rms_norm_gated);

  m.def("esimd_fused_add_rms_norm_batched(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);

  m.def("esimd_norm_gemv_norm_fp16(Tensor residual, Tensor scale_with_root, "
        "Tensor proj_w, Tensor pre_ff_w, Tensor router_logits, Tensor moe_input, "
        "float eps) -> ()");
  m.impl("esimd_norm_gemv_norm_fp16", torch::kXPU, &esimd_norm_gemv_norm_fp16);

  m.def("esimd_scaled_resadd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor qkv_weight, Tensor qkv_scale, Tensor qkv_out, "
        "float eps, float scalar) -> ()");
  m.impl("esimd_scaled_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_scaled_resadd_norm_gemv_fp8_pert);

  m.def("esimd_norm_add_norm(Tensor h2_raw, Tensor h1, Tensor w1, Tensor w2, "
        "Tensor out, float eps1, float eps2) -> ()");
  m.impl("esimd_norm_add_norm", torch::kXPU, &esimd_norm_add_norm);

  m.def("esimd_accum_norm_add_norm(Tensor routed_output, Tensor h1, "
        "Tensor w1, Tensor w2, Tensor out, int top_k, "
        "float eps1, float eps2) -> ()");
  m.impl("esimd_accum_norm_add_norm", torch::kXPU, &esimd_accum_norm_add_norm);

  m.def("esimd_gemv_fp8_pert_bmg(Tensor input, Tensor weight, Tensor weight_scale, Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_bmg", torch::kXPU, &esimd_gemv_fp8_pert_bmg);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
