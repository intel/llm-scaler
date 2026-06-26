"""Python wrappers for custom ESIMD kernels (v2)."""
import torch

_ops = torch.ops.custom_esimd_kernels


def esimd_gemm_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 GEMM with per-tensor scale (M >= 2).

    Auto-dispatches:
        M=1-3:  Batched GEMV (M-parallel WGs, K-split SLM reduction)
        M=5-8:  Weight-stationary, TILE_M=8 M-loop
        M=9-64: Weight-stationary, 2D grid, TILE_M=16 M-tiles

    Args:
        input:        [M, K] fp16
        weight:       [N, K] fp8 (e4m3 or e5m2)
        weight_scale: fp32 scalar (per-tensor)
        output:       [M, N] fp16, pre-allocated

    Returns:
        output (same as passed in).
    """
    return _ops.esimd_gemm_fp8_pert(input, weight, weight_scale, output)


def esimd_gemm_int4_pgrp(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """INT4 GEMM with per-group scale (group_size=128). M >= 2."""
    return _ops.esimd_gemm_int4_pgrp(input, weight, weight_scale, output)
