/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Custom ESIMD Kernels for Intel BMG GPU
 *
 **************************************************************************************************/
#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>

/**
 * FP8 Fused Dequantize + MatMul Kernel
 *
 * Fuses dequantization and matrix multiplication for improved performance.
 * Dequantizes FP8 weights on-the-fly during GEMM computation, avoiding separate
 * dequantization passes and reducing memory bandwidth requirements.
 *
 * @param mat_a FP8 matrix A [M, K], dtype=torch.float8_e4m3fn
 * @param mat_b FP8 matrix B [K, N], dtype=torch.float8_e4m3fn
 * @param scale_a Scalar scale for A, dtype=torch.float32, shape=[1]
 * @param scale_b Scalar scale for B, dtype=torch.float32, shape=[1]
 * @param out_dtype Output data type (torch::kBFloat16 or torch::kFloat16)
 * @return Output matrix C with shape [M, N] and dtype=out_dtype
 */
at::Tensor fp8_fused_dequant_matmul(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const torch::Dtype& out_dtype);
