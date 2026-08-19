"""INT4 GEMM calling wrapper over omni_xpu_kernel svdq ops.

Exposes ``onednn_int4_gemm_preconverted`` with an optional per-block
zero-point parameter (TINT4/torchao format). The caller passes ``zp_u8``
only when the kernel supports it; otherwise the op falls back to the
scalar zp=8 path.
"""

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
        packed_u4: [N, K/2] uint8 — unsigned u4 weights (packed ^ 0x88).
        scales_f16: [G, N] f16 — weight scales.
        zp_u8: [G, N] uint8, optional — per-block zero points (TINT4/torchao,
            w = (q - zp) * scale inside oneDNN).

    Returns:
        [M, N] same dtype as act.
    """
    return _get_svdq().onednn_int4_gemm_preconverted(act, packed_u4, scales_f16, zp_u8)


def apply():
    """Verify the kernel exposes the INT4 GEMM op used by the wrapper."""
    try:
        svdq = _get_svdq()
        if not hasattr(svdq, "onednn_int4_gemm_preconverted"):
            return False, "kernel missing onednn_int4_gemm_preconverted"
        log.info("[OmniXPU] int4_gemm adapter: INT4 GEMM (zp) op available")
        return True, ""
    except Exception as exc:
        return False, str(exc)
