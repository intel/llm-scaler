"""SVDQuant W4A4 Kernels - Intel XPU ESIMD + oneDNN optimized."""

import torch
from typing import Tuple


def _get_native():
    from .. import _load_extension
    return _load_extension().svdq


def dequantize_w4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Fused INT4 unpack + per-group scale application via ESIMD kernel.

    Args:
        packed: [N, K/2] uint8 — packed INT4 weights
        scales: [num_groups, N] — per-group scales (num_groups = K/64)
        out_dtype: output dtype (bfloat16, float16, or float32)

    Returns:
        [N, K] dequantized tensor in out_dtype
    """
    return _get_native().dequantize_svdq_w4(packed, scales, out_dtype)


def unpack_int4(
    packed: torch.Tensor,
    signed: bool = True,
) -> torch.Tensor:
    """
    Unpack INT4 packed tensor to int8 without scaling.

    Args:
        packed: [M, K/2] uint8
        signed: if True, values are signed [-8, 7]

    Returns:
        [M, K] int8 tensor
    """
    return _get_native().unpack_svdq_int4(packed, signed)


def quantize_act_int4(
    input: torch.Tensor,
    group_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activation tensor to INT4 with per-group absmax scaling.

    Args:
        input: [M, K] bf16/f32 activation tensor
        group_size: quantization group size (must be 64)

    Returns:
        (packed [M, K/2] uint8, scales [num_groups, M])
    """
    return _get_native().quantize_svdq_act_int4(input, group_size)


def onednn_int4_gemm(
    act: torch.Tensor,
    packed: torch.Tensor,
    wscales: torch.Tensor,
) -> torch.Tensor:
    """
    Fused INT4 dequant + GEMM using oneDNN u4 matmul primitive.

    Converts signed INT4 packed weights to u4 and bf16 scales to f16 per call.
    For production use, prefer onednn_int4_gemm_preconverted() with pre-converted
    weights for better performance.

    Args:
        act: [M, K] bf16/f16/f32 activations
        packed: [N, K/2] uint8 packed signed INT4 weights
        wscales: [num_groups, N] bf16 per-group weight scales (num_groups = K/group_size)

    Returns:
        [M, N] same dtype as act
    """
    return _get_native().onednn_int4_gemm(act, packed, wscales)


def onednn_int4_gemm_preconverted(
    act: torch.Tensor,
    packed_u4: torch.Tensor,
    scales_f16: torch.Tensor,
) -> torch.Tensor:
    """
    Fused INT4 dequant + GEMM using oneDNN u4 matmul (pre-converted weights).

    Accepts weights already converted to unsigned u4 (via packed ^ 0x88) and
    scales already in f16. Avoids per-call conversion overhead (~0.18ms/call).

    Use prepare_onednn_weights() to convert weights once at model load time.

    Args:
        act: [M, K] bf16/f16/f32 activations (f16 is ~3.5x faster than bf16)
        packed_u4: [N, K/2] uint8 — unsigned u4 weights (from packed ^ 0x88)
        scales_f16: [num_groups, N] f16 — weight scales

    Returns:
        [M, N] same dtype as act
    """
    return _get_native().onednn_int4_gemm_preconverted(act, packed_u4, scales_f16)


def onednn_int4_gemm_add_to_output(
    act: torch.Tensor,
    packed_u4: torch.Tensor,
    scales_f16: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    """
    Fused INT4 GEMM + accumulate into bf16 output using oneDNN append_sum post-op.

    dst += GEMM(f16_act, u4_wgt) — caller pre-fills dst with the residual
    (e.g. LoRA result + bias). Eliminates separate fused_convert_add kernel.

    Args:
        act: [M, K] f16 activations
        packed_u4: [N, K/2] uint8 — unsigned u4 weights (from packed ^ 0x88)
        scales_f16: [num_groups, N] f16 — weight scales
        dst: [M, N] bf16 output tensor (modified in-place: dst += GEMM result)
    """
    _get_native().onednn_int4_gemm_add_to_output(act, packed_u4, scales_f16, dst)


def fused_convert_add(
    out: torch.Tensor,
    result: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    """
    Fused f16->bf16 conversion + bf16 addition via ESIMD kernel.

    Equivalent to: out.copy_(result[:Mo,:No].to(bf16)); out.add_(residual[:Mo,:No])
    but in a single memory pass (eliminates intermediate conversion tensor).

    Args:
        out: [Mo, No] bf16 output tensor (modified in-place)
        result: [Mr, Nr] f16 GEMM result (may be larger than out)
        residual: [Mo, No] bf16 residual to add
    """
    _get_native().fused_convert_add(out, result, residual)


def fused_smooth_convert(
    x: torch.Tensor,
    smooth_factor: torch.Tensor,
) -> torch.Tensor:
    """
    Fused smooth division + bf16->f16 conversion (legacy, uses division).

    Args:
        x: [M, K] bf16 activation tensor
        smooth_factor: [K] bf16 per-channel smooth factor

    Returns:
        [M, K] f16 = (x / smooth_factor).to(f16)
    """
    return _get_native().fused_smooth_convert(x, smooth_factor)


def fused_smooth_mul_convert(
    x: torch.Tensor,
    rcp_smooth: torch.Tensor,
) -> torch.Tensor:
    """
    Fused smooth multiply + bf16->f16 conversion (optimized, multiply-by-reciprocal).

    Uses pre-computed 1/smooth_factor to replace division with multiplication.

    Args:
        x: [M, K] bf16 activation tensor
        rcp_smooth: [K] f16 pre-computed reciprocal of smooth_factor

    Returns:
        [M, K] f16 = (x * rcp_smooth)
    """
    return _get_native().fused_smooth_mul_convert(x, rcp_smooth)


def prepare_onednn_weights(
    packed: torch.Tensor,
    wscales: torch.Tensor,
) -> tuple:
    """
    Pre-convert SVDQuant weights for oneDNN INT4 GEMM.

    Converts signed INT4 packed weights to unsigned u4 and bf16 scales to f16.
    Call once at model load time, then use onednn_int4_gemm_preconverted() for inference.

    Args:
        packed: [N, K/2] uint8 — signed INT4 packed weights
        wscales: [num_groups, N] bf16 — per-group weight scales

    Returns:
        (packed_u4 [N, K/2] uint8, scales_f16 [num_groups, N] f16)
    """
    packed_u4 = (packed ^ 0x88).contiguous()
    scales_f16 = wscales.to(torch.float16).clone().contiguous()
    return packed_u4, scales_f16


__all__ = [
    "dequantize_w4",
    "unpack_int4",
    "quantize_act_int4",
    "onednn_int4_gemm",
    "onednn_int4_gemm_preconverted",
    "onednn_int4_gemm_add_to_output",
    "prepare_onednn_weights",
    "fused_convert_add",
    "fused_smooth_convert",
    "fused_smooth_mul_convert",
]
