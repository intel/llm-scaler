"""cute / CUTLASS-SYCL fused Flash Attention (torch op).

Drop-in for :func:`omni_xpu_kernel.sdp.sdp` — same signature and layout::

    from omni_xpu_kernel import cute
    out = cute.sdp(q, k, v)   # self-attn [B, L, H, D] (B==1, D==128), fp16/bf16

PTL-H and BMG wheels also expose a workflow-tuned D120 entry point that
consumes dense packed-BHLD or BLHD-backed BHLD layouts without intermediate
copies::

    out = cute.sdp_bhld_d120(q, k, v)  # [B, H, L, 120]

BMG wheels additionally expose the exact Wan 2.2 14B T2V Turbo 720p
cross-attention contract through ``sdp_wan22_cross`` and a batched,
rectangular D128 BHLD entry point through ``sdp_bhld_d128``. The structural
MiniMax H3 VideoVAE D64 tile family is exposed separately through
``sdp_minimax_h3_vae_d64``.

Unlike the ESIMD ``sdp`` kernel (fp16 accumulator + adaptive V-scaling), the cute
FMHA accumulates QK and P*V in fp32, so it does not overflow on large-magnitude
activations (e.g. Qwen-Image). It is AOT-compiled into a native
``cute_fmha_torch`` extension (``.so`` on Linux or ``.pyd`` on Windows) and
exposes ``torch.ops.cute_fmha.sdp``. The generic entry point accepts
self-attention only; validated rectangular workflow contracts use dedicated
entry points.
"""

import glob
import os
from importlib.machinery import EXTENSION_SUFFIXES

import torch

_loaded = False


def _find_extension():
    """Locate the platform-native CUTE FMHA extension.

    setuptools adds a Python ABI suffix. A hand build may use a plain ``.so``
    or ``.pyd``. ``OMNI_CUTE_FMHA_SO`` remains the compatible path override.
    """
    env = os.environ.get("OMNI_CUTE_FMHA_SO", "")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    for suffix in (*EXTENSION_SUFFIXES, ".pyd", ".so"):
        cands.append(os.path.join(here, "cute_fmha_torch" + suffix))
        cands.extend(
            sorted(glob.glob(os.path.join(here, "cute_fmha_torch*" + suffix)))
        )
    seen = set()
    for c in cands:
        if c not in seen and os.path.isfile(c):
            return c
        seen.add(c)
    return ""


def _find_so():
    """Compatibility alias used by the pinned ComfyUI-SolAttn node."""
    return _find_extension()


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    extension = _find_extension()
    if not extension or not os.path.exists(extension):
        raise ImportError(
            "cute_fmha_torch native extension not found next to "
            "omni_xpu_kernel.cute "
            "(set OMNI_CUTE_FMHA_SO to override)"
        )
    torch.ops.load_library(extension)
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


def supports_wan22_cross() -> bool:
    """Whether this BMG sidecar exports the exact Wan 2.2 cross kernel."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_wan22_cross")
    except Exception:
        return False


def sdp_wan22_cross(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Wan 2.2 14B T2V Turbo 720p FP16 cross-attention."""
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_wan22_cross"):
        raise RuntimeError(
            "CUTE Wan 2.2 cross-attention kernel is unavailable "
            "in this sidecar"
        )
    return torch.ops.cute_fmha.sdp_wan22_cross(q, k, v)


def supports_d128_bhld() -> bool:
    """Whether this BMG sidecar exports batched/rectangular D128 BHLD."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_bhld_d128")
    except Exception:
        return False


def sdp_bhld_d128(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Attention for supported dense or H3 QKV-backed ``[B,H,L,128]`` inputs."""
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_bhld_d128"):
        raise RuntimeError(
            "CUTE D128 BHLD attention kernel is unavailable "
            "in this sidecar"
        )
    return torch.ops.cute_fmha.sdp_bhld_d128(q, k, v)


def supports_minimax_h3_vae_d64() -> bool:
    """Whether this BMG sidecar exports MiniMax H3 VideoVAE D64 tiles."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_minimax_h3_vae_d64")
    except Exception:
        return False


def sdp_minimax_h3_vae_d64(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """MiniMax H3 VideoVAE FP16 ``[1,32,S,64]`` tile attention.

    ``S`` is derived by the decoder from the temporal/spatial tile extent;
    Q/K use the ``H*D`` sequence stride and V remains a view into the
    three-wide QKV projection.
    """
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_minimax_h3_vae_d64"):
        raise RuntimeError(
            "CUTE MiniMax H3 VideoVAE D64 kernel is unavailable "
            "in this sidecar"
        )
    return torch.ops.cute_fmha.sdp_minimax_h3_vae_d64(q, k, v)


def supports_d120_bhld() -> bool:
    """Whether this target sidecar exports the workflow-tuned D120 kernel."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_bhld_d120")
    except Exception:
        return False


def sdp_bhld_d120(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Fused self-attention for validated dense ``[B,H,L,120]`` inputs."""
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_bhld_d120"):
        raise RuntimeError("CUTE D120 BHLD kernel is unavailable in this sidecar")
    return torch.ops.cute_fmha.sdp_bhld_d120(q, k, v)


def _sol_attn_ops():
    return torch.ops.omni_xpu_sol_attn


def supports_sol_attn() -> bool:
    """Whether this BMG sidecar exports the validated sparse Sol-Attn path."""
    try:
        _ensure_loaded()
        ops = _sol_attn_ops()
        return hasattr(ops, "prepare") and hasattr(ops, "forward_cute")
    except Exception:
        return False


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Sparse Sol-Attn for the validated BMG BF16 BTHD D128 contract.

    Route preparation remains an internal implementation detail. Unsupported
    targets and tensor contracts raise instead of silently changing attention
    semantics; callers may use :func:`supports_sol_attn` for capability
    routing.
    """
    _ensure_loaded()
    ops = _sol_attn_ops()
    if not hasattr(ops, "prepare") or not hasattr(ops, "forward_cute"):
        raise RuntimeError("packaged Sol-Attn is unavailable in this sidecar")
    if len(sink_blocks) != 2 or len(sink_q) != 2:
        raise ValueError("sink_blocks and sink_q must each contain two indices")
    scale_value = q.shape[-1] ** -0.5 if scale is None else float(scale)
    prepared = ops.prepare(
        q,
        k,
        v,
        float(scale_value),
        float(tau),
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
    )
    return ops.forward_cute(q, k, v, *prepared, float(scale_value))


__all__ = [
    "sdp",
    "sdp_wan22_cross",
    "supports_wan22_cross",
    "sdp_bhld_d128",
    "supports_d128_bhld",
    "sdp_minimax_h3_vae_d64",
    "supports_minimax_h3_vae_d64",
    "sdp_bhld_d120",
    "supports_d120_bhld",
    "sol_attn",
    "supports_sol_attn",
    "is_available",
]
