#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <torch/torch.h>

// FP8 weight GEMV with per-N scale: output = input @ dequant(weight_fp8) * scale
// FP32 accumulation, element-wise acc + deferred scale. Optimized for decode (M=1).
at::Tensor esimd_gemv_fp8_pern(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output,
    int64_t N, int64_t K);

// Fused FP8 GEMV: single kernel submit for 2 weight matrices (same input, same K)
at::Tensor esimd_gemv_fp8_pern_fused2(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0, int64_t N0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1, int64_t N1,
    int64_t K);

// Fused FP8 GEMV: single kernel submit for 3 weight matrices (same input, same K)
at::Tensor esimd_gemv_fp8_pern_fused3(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0, int64_t N0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1, int64_t N1,
    at::Tensor w2, at::Tensor s2, at::Tensor o2, int64_t N2,
    int64_t K);

// Per-tensor scale variants: scale is fp32 scalar, N/K auto-detected from tensor shapes
at::Tensor esimd_gemv_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

at::Tensor esimd_gemv_fp8_pert_fused2(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1);

at::Tensor esimd_gemv_fp8_pert_fused3(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1,
    at::Tensor w2, at::Tensor s2, at::Tensor o2);

// Fused QKV Split + RMSNorm(weight+1.0, eps=1e-6) + RoPE(theta=10M, headDim=256)
// qkv_state: [nTokens, hiddenDim] fp16 — packed QKV (or QGKV with gating)
// q_out:     [nTokens, qHead*256] fp16
// gate_out:  [nTokens, qHead*256] fp16 (ignored if !attn_output_gate)
// k_out:     [nTokens, kvHead*256] fp16
// v_out:     [nTokens, kvHead*256] fp16
// norm_wq/wk: [256] fp16 — RMSNorm weights
// positions: [nTokens] int32 — RoPE position indices
at::Tensor esimd_qkv_split_norm_rope(
    at::Tensor qkv_state,
    at::Tensor q_out, at::Tensor gate_out,
    at::Tensor k_out, at::Tensor v_out,
    at::Tensor norm_wq, at::Tensor norm_wk,
    at::Tensor positions,
    int64_t q_heads, int64_t kv_heads, bool attn_output_gate,
    int64_t rotary_dim, at::Tensor cos_sin_cache);

// Fused Conv1d + GDN for Qwen3-Next-80B-A3B decode — reads from projections directly
// qkvz:               [N, qkvz_dim] fp16 — projected_states_qkvz, read-only
// conv_state:         [num_cache, 3, 2048] fp16, strided dim0
// conv_weight:        [2048, 4] fp16
// conv_bias:          [2048] fp16 (zeros if model has no bias)
// conv_state_indices: [N] int32
// A_log:              [HV] fp16
// dt_bias:            [HV] fp16
// ba:                 [N, 2*HV] fp16 — projected_states_ba, interleaved [b_grp, a_grp, ...]
// ssm_state:          [num_states, HV, V, K] fp16, strided dim0
// ssm_state_indices:  [N] int32
// output:             [N, HV, V] fp16
// z_out:              [N, HV, V] fp16 — z gate extracted from qkvz
at::Tensor esimd_gdn_conv_fused(
    at::Tensor qkvz,
    at::Tensor conv_state, at::Tensor conv_weight, at::Tensor conv_bias,
    at::Tensor conv_state_indices,
    at::Tensor A_log, at::Tensor dt_bias,
    at::Tensor ba,
    at::Tensor ssm_state, at::Tensor ssm_state_indices,
    at::Tensor output, at::Tensor z_out,
    int64_t N, int64_t H, int64_t HV,
    int64_t K, int64_t V,
    double scale);

// Fused Conv1d + GDN for SEQUENTIAL qkvz layout [q_all|k_all|v_all|z_all]
// Same as esimd_gdn_conv_fused but reads qkvz/ba in sequential (non-interleaved) order.
// For models like Qwen3.5-35B-A3B where GEMV outputs sequential layout.
at::Tensor esimd_gdn_conv_fused_seq(
    at::Tensor qkvz,
    at::Tensor conv_state, at::Tensor conv_weight, at::Tensor conv_bias,
    at::Tensor conv_state_indices,
    at::Tensor A_log, at::Tensor dt_bias,
    at::Tensor ba,
    at::Tensor ssm_state, at::Tensor ssm_state_indices,
    at::Tensor output, at::Tensor z_out,
    int64_t N, int64_t H, int64_t HV,
    int64_t K, int64_t V,
    double scale);

// Fused ResidualAdd + RMSNorm + FP8 GEMV (post_attn_norm + router)
at::Tensor esimd_resadd_norm_gemv_fp8_pert(
    at::Tensor hidden_states, at::Tensor residual, at::Tensor norm_weight,
    at::Tensor gemv_weight, at::Tensor gemv_scale, at::Tensor output, at::Tensor normed_out,
    double eps);

// Fused ResidualAdd + RMSNorm + INT4 GEMV (post_attn_norm + router)
// hidden_states: [1, K] fp16
// residual:      [1, K] fp16 (updated in-place)
// norm_weight:   [K] fp16
// gemv_weight:   [N, K/8] int32 — packed INT4
// gemv_scale:    [N, K/128] fp16 — per-block scale
// output:        [1, N] fp16
// normed_out:    [1, K] fp16
at::Tensor esimd_resadd_norm_gemv_int4_pert(
    at::Tensor hidden_states, at::Tensor residual, at::Tensor norm_weight,
    at::Tensor gemv_weight, at::Tensor gemv_scale, at::Tensor output, at::Tensor normed_out,
    double eps);

// Fused ResidualAdd + RMSNorm + 2-matrix FP8 GEMV (input_norm + GDN in_proj)
at::Tensor esimd_resadd_norm_gemv2_fp8_pert(
    at::Tensor hidden_states, at::Tensor residual, at::Tensor norm_weight,
    at::Tensor w0, at::Tensor s0, at::Tensor o0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1,
    double eps);

// Fused RMSNormGated + FP8 GEMV for GDN out_proj decode path
// x:            [HV, V] fp16 — core_attn_out from GDN kernel
// z:            [HV, V] fp16 — z_out from GDN kernel
// norm_weight:  [V] fp16 — RMSNorm weight (per head_v_dim)
// gemv_weight:  [N, K] FP8, K = HV*V — out_proj weight
// gemv_scale:   [1] fp32 — per-tensor FP8 scale
// output:       [1, N] fp16
at::Tensor esimd_norm_gemv_fp8_pert(
    at::Tensor x, at::Tensor z, at::Tensor norm_weight,
    at::Tensor gemv_weight, at::Tensor gemv_scale, at::Tensor output,
    int64_t HV, int64_t V, double eps);

// Fused RMSNormGated + INT4 GEMV for GDN out_proj decode path
// x:            [HV, V] fp16 — core_attn_out from GDN kernel
// z:            [HV, V] fp16 — z_out from GDN kernel
// norm_weight:  [V] fp16 — RMSNorm weight (per head_v_dim)
// gemv_weight:  [N, K/8] int32, K = HV*V — packed INT4 out_proj weight
// gemv_scale:   [N, K/128] fp16 — per-block INT4 scale
// output:       [1, N] fp16
at::Tensor esimd_norm_gemv_int4_pert(
    at::Tensor x, at::Tensor z, at::Tensor norm_weight,
    at::Tensor gemv_weight, at::Tensor gemv_scale, at::Tensor output,
    int64_t HV, int64_t V, double eps);

// Fused Add + RMSNorm (Gemma-style)
at::Tensor esimd_fused_add_rms_norm(
    at::Tensor hidden_states, at::Tensor residual,
    at::Tensor weight, double eps);

at::Tensor esimd_rms_norm_gated(
    at::Tensor x, at::Tensor z, at::Tensor weight,
    at::Tensor output, double eps);

at::Tensor esimd_fused_add_rms_norm_batched(
    at::Tensor hidden_states, at::Tensor residual,
    at::Tensor weight, double eps);

// ======================== MoE Auxiliary Ops ========================

// Fused softmax + top-8 + normalize
at::Tensor esimd_moe_topk(
    at::Tensor router_logits, at::Tensor top_values, at::Tensor top_indices,
    int64_t T);

// Scatter hidden_states by expert grouping
at::Tensor esimd_moe_scatter(
    at::Tensor hidden_states, at::Tensor router_top_value,
    at::Tensor sorted_token_ids,
    at::Tensor scattered_hidden, at::Tensor scattered_weights,
    int64_t K, int64_t topk, int64_t total_expanded);

// Fused GPU scatter: atomic counting → prefix-sum → copy (no CPU preprocessing)
at::Tensor esimd_moe_scatter_fused(
    at::Tensor hidden_states, at::Tensor top_values, at::Tensor top_indices,
    at::Tensor scattered_hidden, at::Tensor scattered_weights,
    at::Tensor topk_ids, at::Tensor expert_start, at::Tensor max_tokens_out,
    int64_t K, int64_t topk, int64_t T, int64_t num_experts);

// SiLU(gate) * up activation
at::Tensor esimd_moe_silu_mul(
    at::Tensor input, at::Tensor output,
    int64_t N_gate_up, int64_t N_half, int64_t total_rows);

// Weighted gather/reduce
at::Tensor esimd_moe_gather(
    at::Tensor moe_output, at::Tensor topk_ids, at::Tensor scattered_weights,
    at::Tensor final_hidden,
    int64_t K, int64_t topk, int64_t T);

// MoE grouped GEMM — FP8 E5M2 with per-N scale
at::Tensor esimd_moe_gemm_fp8(
    at::Tensor input, at::Tensor weight, at::Tensor scale,
    at::Tensor output, at::Tensor expert_idx,
    int64_t N, int64_t K, int64_t num_experts, int64_t max_tokens_per_expert);

// FP8 GEMM per-tensor scale: input [M, K] fp16, weight [N, K] fp8, output [M, N] fp16
// Auto-dispatches: M<=3 → batched GEMV, M>=2 E4M3 → DPAS V9, else → WS
at::Tensor esimd_gemm_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

// ======================== INT4 GEMV ========================
// Symmetric INT4 weight GEMV with per-group scale (group_size=128).
// Optimized for decode (M=1). FP32 accumulation → fp16 output.
// Weight: [N, K/2] uint8 (packed, 2 int4 per byte, low nibble = even index).
// Scale:  [N, K/128] fp16 (per-group). N and K inferred from tensor shapes.

// Single INT4 GEMV: output[1,N] = input[1,K] @ dequant(weight[N,K/2])^T
at::Tensor esimd_gemv_int4(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

// Fused 2-matrix INT4 GEMV: two GEMVs sharing the same input, single kernel submit.
// Used for GDN input projection (in_proj_qkvz + in_proj_ba).
at::Tensor esimd_gemv_int4_fused2(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1);

// MoE grouped GEMM — FP8 E5M2 with per-tensor scale (one scalar per expert)
at::Tensor esimd_moe_gemm_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor scale,
    at::Tensor output, at::Tensor expert_idx,
    int64_t N, int64_t K, int64_t num_experts, int64_t max_tokens_per_expert);
