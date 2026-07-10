/* kernel_ops.h - Function declarations for FP8/INT4 GEMM/GEMV ESIMD kernels
 * Migrated from llm-scaler/vllm/custom-esimd-kernels-vllm
 */
#pragma once
#include <ATen/ATen.h>

// ============================================================================
// FP8 GEMV (M=1) - Per-tensor scale variant
// Auto-detects N/K from weight shape
// Input: [1, K] fp16 (or [K]), Weight: [N, K] fp8, Scale: fp32 scalar, Output: [1, N] fp16
// ============================================================================
at::Tensor esimd_gemv_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

// Fused 2-matrix FP8 GEMV (single kernel submit for Q+K, FFN gate+up etc.)
at::Tensor esimd_gemv_fp8_pert_fused2(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1);

// Fused 3-matrix FP8 GEMV (Q+K+V combined)
at::Tensor esimd_gemv_fp8_pert_fused3(
    at::Tensor input,
    at::Tensor w0, at::Tensor s0, at::Tensor o0,
    at::Tensor w1, at::Tensor s1, at::Tensor o1,
    at::Tensor w2, at::Tensor s2, at::Tensor o2);

// ============================================================================
// FP8 GEMM (M >= 2) - Per-tensor scale
// Auto-dispatches: M<=3 → batched GEMV, M>=2 E4M3 → DPAS V9, else → WS
// Input: [M, K] fp16, Weight: [N, K] fp8, Output: [M, N] fp16
// ============================================================================
at::Tensor esimd_gemm_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

// BMG-specific FP8 GEMV variant
at::Tensor esimd_gemv_fp8_pert_bmg(
    at::Tensor input, at::Tensor weight,
    at::Tensor weight_scale, at::Tensor output);

// ============================================================================
// INT4 GEMM (M >= 2) - Per-group scale (group_size=128)
// ============================================================================
at::Tensor esimd_gemm_int4_pgrp(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);
