"""Validate CUTLASS-style N-major INT4 packing for MoE weights.

This is the first step for the new high-performance N-major path:

    GGML/test quantized int32 [E, N, K/8]
        -> CUTLASS N-major uint8 [E, N, K/2]
        -> signed-int4 compact format used by vllm-xpu-kernels implement_zp

The script intentionally does not call the MoE kernel. It only verifies that
the proposed weight format is lossless and has the same dequantized values as
the existing GGML N-major int32 representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_moe_int4_kernel import GROUP_SIZE, quantize_int4, dequantize_int4


def _to_cutlass_nmajor(qweight: torch.Tensor) -> torch.Tensor:
    """Convert INT4 weights to CUTLASS-style N-major byte packing.

    Accepted inputs:
      * int32/int64 [E, N, K/8] or [N, K/8], 8 uint4 values per int32.
      * uint8      [E, N, K/2] or [N, K/2], 2 uint4 values per byte.

    Output:
      * uint8 [E, N, K/2] or [N, K/2]
        low nibble stores even K, high nibble stores odd K.
    """
    if qweight.dtype == torch.uint8:
        return qweight.contiguous()

    if qweight.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"unsupported qweight dtype: {qweight.dtype}")

    if qweight.dim() not in (2, 3):
        raise ValueError(f"expected [N,K/8] or [E,N,K/8], got {tuple(qweight.shape)}")

    q_u32 = qweight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int64) * 4
    nibbles = ((q_u32.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    nibbles = nibbles.reshape(*qweight.shape[:-1], qweight.shape[-1] * 8)

    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (low | (high << 4)).contiguous()


def _implement_zp_reference(qweight_u4: torch.Tensor) -> torch.Tensor:
    """CPU-compatible equivalent of vllm_xpu_kernels implement_zp.

    Converts unsigned uint4 values in a uint8 byte to signed int4 two's-complement
    nibbles by subtracting 8 from both packed values.
    """
    if qweight_u4.dtype != torch.uint8:
        raise TypeError(f"expected uint8 qweight, got {qweight_u4.dtype}")

    high_u4 = (qweight_u4 >> 4) & 0x0F
    low_u4 = qweight_u4 & 0x0F
    high_s8 = high_u4.to(torch.int8) - 8
    low_s8 = low_u4.to(torch.int8) - 8

    def pack_compact(values: torch.Tensor) -> torch.Tensor:
        sign = (values < 0).to(torch.uint8)
        low3 = (values.view(torch.uint8) & 0x7).to(torch.uint8)
        return (sign << 3) | low3

    return ((pack_compact(high_s8) << 4) | pack_compact(low_s8)).contiguous()


def _try_import_implement_zp():
    try:
        from vllm_xpu_kernels.fused_moe_interface import implement_zp
    except Exception:
        return _implement_zp_reference
    return implement_zp


def _decode_cutlass_u4(qweight_u4: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize CUTLASS N-major unsigned uint4 [E,N,K/2] to fp16 [E,N,K]."""
    low = qweight_u4 & 0x0F
    high = (qweight_u4 >> 4) & 0x0F
    values = torch.stack((low, high), dim=-1).reshape(*qweight_u4.shape[:-1], qweight_u4.shape[-1] * 2)
    values = values.to(torch.float32) - 8.0

    k = values.shape[-1]
    n_groups = k // group_size
    values = values.reshape(*values.shape[:-1], n_groups, group_size)
    values = values * scales.to(torch.float32).unsqueeze(-1)
    return values.reshape(*qweight_u4.shape[:-1], k).half()


def _decode_cutlass_s4(qweight_s4: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize signed compact int4 [E,N,K/2] to fp16 [E,N,K]."""
    low = qweight_s4 & 0x0F
    high = (qweight_s4 >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(*qweight_s4.shape[:-1], qweight_s4.shape[-1] * 2)
    values_i16 = nibbles.to(torch.int16)
    values_i16 = torch.where(values_i16 >= 8, values_i16 - 16, values_i16)
    values = values_i16.to(torch.float32)

    k = values.shape[-1]
    n_groups = k // group_size
    values = values.reshape(*values.shape[:-1], n_groups, group_size)
    values = values * scales.to(torch.float32).unsqueeze(-1)
    return values.reshape(*qweight_s4.shape[:-1], k).half()


def _check_shape(num_experts: int, n: int, k: int) -> None:
    torch.manual_seed(17 + num_experts + n + k)
    weights = (torch.randn(num_experts, n, k) * 0.02).half()

    qweights = []
    scales = []
    refs = []
    for expert in range(num_experts):
        qweight, scale = quantize_int4(weights[expert], GROUP_SIZE)
        qweights.append(qweight)
        scales.append(scale)
        refs.append(dequantize_int4(qweight, scale, n, k, GROUP_SIZE))

    qweight_i32 = torch.stack(qweights)
    scale = torch.stack(scales)
    ref = torch.stack(refs)

    qweight_u4 = _to_cutlass_nmajor(qweight_i32)
    implement_zp = _try_import_implement_zp()
    qweight_s4 = torch.empty_like(qweight_u4)
    for expert in range(num_experts):
        qweight_s4[expert] = implement_zp(qweight_u4[expert])

    got_u4 = _decode_cutlass_u4(qweight_u4, scale, GROUP_SIZE)
    got_s4 = _decode_cutlass_s4(qweight_s4, scale, GROUP_SIZE)

    diff_u4 = (got_u4.float() - ref.float()).abs()
    diff_s4 = (got_s4.float() - ref.float()).abs()
    print(
        f"E={num_experts} N={n} K={k} "
        f"u4_shape={tuple(qweight_u4.shape)} "
        f"u4_max={diff_u4.max().item():.6g} s4_max={diff_s4.max().item():.6g}"
    )
    assert qweight_u4.shape == (num_experts, n, k // 2)
    assert qweight_u4.dtype == torch.uint8
    assert qweight_s4.shape == qweight_u4.shape
    assert diff_u4.max().item() == 0.0
    assert diff_s4.max().item() == 0.0


def main() -> None:
    _check_shape(num_experts=2, n=32, k=128)
    _check_shape(num_experts=3, n=64, k=256)
    _check_shape(num_experts=2, n=512, k=3072)
    _check_shape(num_experts=2, n=3072, k=256)
    print("CUTLASS N-major INT4 pack checks passed.")


if __name__ == "__main__":
    main()