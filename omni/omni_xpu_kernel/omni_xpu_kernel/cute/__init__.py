"""cute / CUTLASS-SYCL fused Flash Attention (torch op).

Drop-in for :func:`omni_xpu_kernel.sdp.sdp` — same signature and layout::

    from omni_xpu_kernel import cute
    out = cute.sdp(q, k, v)   # q,k,v,out are [B, L, H, D] (B==1, D==128), fp16/bf16

Unlike the ESIMD ``sdp`` kernel (fp16 accumulator + adaptive V-scaling), the cute
FMHA accumulates QK and P*V in fp32, so it does not overflow on large-magnitude
activations (e.g. Qwen-Image). It is AOT-compiled into ``cute_fmha_torch.so`` and
exposes ``torch.ops.cute_fmha.sdp``.
"""

import glob
import os

import torch

_loaded = False


def _find_so():
    """Locate the cute FMHA .so.

    setuptools names it with the Python ABI suffix (cute_fmha_torch.cpython-*.so);
    a hand build may drop a plain cute_fmha_torch.so. OMNI_CUTE_FMHA_SO overrides.
    """
    env = os.environ.get("OMNI_CUTE_FMHA_SO", "")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "cute_fmha_torch.so")]
    cands += sorted(glob.glob(os.path.join(here, "cute_fmha_torch*.so")))
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    so = _find_so()
    if not so or not os.path.exists(so):
        raise ImportError(
            "cute_fmha_torch .so not found next to omni_xpu_kernel.cute "
            "(set OMNI_CUTE_FMHA_SO to override)"
        )
    torch.ops.load_library(so)
    _loaded = True


def is_available():
    try:
        _ensure_loaded()
        return True
    except Exception:
        return False


def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fused scaled-dot-product attention. Inputs [B, L, H, D] (B==1, D==128)."""
    _ensure_loaded()
    return torch.ops.cute_fmha.sdp(q, k, v)


__all__ = ["sdp", "is_available"]
