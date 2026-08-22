"""INT4 GEMM calling wrapper over omni_xpu_kernel svdq ops.

Exposes ``onednn_int4_gemm_preconverted`` (wa4 对称) 与可选的 per-block
zero-point 参数（TINT4/torchao 非对称）。kernel 不支持 zp 时 tint4 不处理
（返回 None，调用方回退自身 python/torchao 路径），不报错、不强行走 kernel。
"""

import inspect
import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

_svdq = None


def _get_svdq():
    global _svdq
    if _svdq is None:
        from omni_xpu_kernel import svdq
        _svdq = svdq
    return _svdq


def int4_gemm(act, packed_u4, scales_f16, zp_u8=None):
    """Fused INT4 dequant + GEMM via oneDNN u4 matmul.

    Args:
        act: [M, K] bf16/f16/f32 activations.
        packed_u4: [N, K/2] uint8 — unsigned u4 weights (packed ^ 0x88，
            preconverted 约定，wa4 与 tint4 均适用)。
        scales_f16: [G, N] f16 — weight scales.
        zp_u8: [G, N] uint8, optional — per-block zero points (TINT4/torchao,
            w = (q - zp) * scale inside oneDNN).

    Returns:
        [M, N] same dtype as act；kernel 不支持 zp 时返回 None（不处理）。
    """
    svdq = _get_svdq()
    if zp_u8 is not None:
        if not _kernel_accepts_zp(svdq):
            log.warning(
                "[OmniXPU] int4_gemm: kernel 不支持 zp_u8，tint4 不处理"
                "（调用方回退自身 python/torchao 路径）"
            )
            return None
        return svdq.onednn_int4_gemm_preconverted(act, packed_u4, scales_f16, zp_u8)
    # 无 zp（wa4）：3 参调用，兼容不支持 zp 的老 kernel
    return svdq.onednn_int4_gemm_preconverted(act, packed_u4, scales_f16)


def _kernel_accepts_zp(svdq):
    try:
        return "zp_u8" in inspect.signature(
            svdq.onednn_int4_gemm_preconverted).parameters
    except (TypeError, ValueError):
        return False


def apply():
    """报告 kernel 的 INT4 GEMM 可用性与 zp 支持。"""
    try:
        svdq = _get_svdq()
        if not hasattr(svdq, "onednn_int4_gemm_preconverted"):
            return False, "kernel missing onednn_int4_gemm_preconverted"
        log.info("[OmniXPU] int4_gemm adapter: preconverted available, "
                 "zp_u8 supported=%s", _kernel_accepts_zp(svdq))
        return True, ""
    except Exception as exc:
        return False, str(exc)
