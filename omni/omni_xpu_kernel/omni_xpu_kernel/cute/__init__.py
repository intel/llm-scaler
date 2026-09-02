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
import math
import os
from importlib.machinery import EXTENSION_SUFFIXES

import torch

_loaded = False
_prepared_bmg_policy_dispatches = set()


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


def _prepare_bmg_policy_dispatch(tensor: torch.Tensor) -> None:
    """Let the core extension own the process-wide BMG policy warning."""

    from .. import __xpu_target__, device

    if __xpu_target__ == "bmg":
        index = 0 if tensor.device.index is None else tensor.device.index
        key = (
            index,
            os.environ.get("OMNI_XPU_FORCE_SKU"),
            os.environ.get("OMNI_XPU_B580_POLICY_CANDIDATE"),
        )
        if key not in _prepared_bmg_policy_dispatches:
            device.info(index)
            _prepared_bmg_policy_dispatches.add(key)


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
    _prepare_bmg_policy_dispatch(q)
    return torch.ops.cute_fmha.sdp_bhld_d120(q, k, v)


def _sol_attn_ops():
    return torch.ops.omni_xpu_sol_attn


def supports_sol_attn() -> bool:
    """Whether this BMG sidecar exports the validated sparse Sol-Attn path."""
    try:
        _ensure_loaded()
        ops = _sol_attn_ops()
        return hasattr(ops, "prepare_with_controls") and hasattr(
            ops, "forward_cute_with_controls"
        )
    except Exception:
        return False


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: tuple[int, int] | list[int] | None = None,
    sink_q: tuple[int, int] | list[int] | None = None,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse Sol-Attn for the validated BMG BF16 BTHD D128 contract.

    Route preparation remains an internal implementation detail. Unsupported
    targets and tensor contracts raise instead of silently changing attention
    semantics; callers may use :func:`supports_sol_attn` for capability
    routing. A query row with no finite routed key score returns zero.
    """
    _ensure_loaded()
    ops = _sol_attn_ops()
    if not hasattr(ops, "prepare_with_controls") or not hasattr(
        ops, "forward_cute_with_controls"
    ):
        raise RuntimeError("packaged Sol-Attn is unavailable in this sidecar")
    sink_blocks = (0, 0) if sink_blocks is None else sink_blocks
    sink_q = (0, 0) if sink_q is None else sink_q
    if len(sink_blocks) != 2 or len(sink_q) != 2:
        raise ValueError("sink_blocks and sink_q must each contain two indices")
    blocks = (q.shape[1] + 63) // 64
    if key_bias is None:
        key_bias_log2 = torch.empty(0, dtype=torch.float32, device=q.device)
    else:
        if key_bias.device != q.device:
            raise ValueError(
                f"key_bias must be on {q.device}, got {key_bias.device}"
            )
        if key_bias.dim() == 4:
            if key_bias.shape[1:3] != (1, 1):
                raise ValueError(
                    "key_bias must not vary over heads or queries"
                )
            key_bias = key_bias[:, 0, 0, :]
        if key_bias.dim() == 1:
            key_bias = key_bias.unsqueeze(0)
        if (
            key_bias.dim() != 2
            or key_bias.shape[-1] != q.shape[1]
            or key_bias.shape[0] not in (1, q.shape[0])
        ):
            raise ValueError(
                "key_bias must be (T,), (B, T), or (B|1, 1, 1, T); "
                f"got {tuple(key_bias.shape)} for B={q.shape[0]}, "
                f"T={q.shape[1]}"
            )
        if key_bias.dtype == torch.bool:
            key_bias = torch.where(key_bias, 0.0, float("-inf"))
        elif not torch.is_floating_point(key_bias):
            raise ValueError("key_bias must have bool or floating dtype")
        key_bias_log2 = (
            key_bias.float()
            .mul(math.log2(math.e))
            .expand(q.shape[0], q.shape[1])
            .contiguous()
        )
    if block_len is None:
        block_lengths = torch.empty(0, dtype=torch.int32, device=q.device)
    else:
        if (
            block_len.device != q.device
            or block_len.dtype != torch.int32
            or block_len.dim() != 1
            or block_len.numel() != blocks
        ):
            raise ValueError(
                "block_len must be contiguous int32 (N,) on the Q device; "
                f"got {block_len.dtype} {tuple(block_len.shape)} on "
                f"{block_len.device}, expected N={blocks} on {q.device}"
            )
        block_lengths = block_len.contiguous()
    if coarse_gate is not None:
        if (
            coarse_gate.device != q.device
            or tuple(coarse_gate.shape) != tuple(q.shape)
            or not torch.is_floating_point(coarse_gate)
        ):
            raise ValueError(
                "coarse_gate must be a floating tensor with Q's shape and device"
            )
    topk_ratio = float(topk_ratio)
    if topk_ratio != 0.0 and not 0.0 < topk_ratio < 1.0:
        raise ValueError("topk_ratio must be 0 or in (0, 1)")
    sink_count = max(
        0,
        min(int(sink_blocks[1]), blocks)
        - min(int(sink_blocks[0]), blocks),
    )
    selectable_blocks = blocks - sink_count
    topk_count = -1
    if topk_ratio != 0.0:
        topk_count = max(
            0,
            min(
                selectable_blocks - 1,
                max(1, round(topk_ratio * selectable_blocks)),
            ),
        )
    scale_value = q.shape[-1] ** -0.5 if scale is None else float(scale)
    prepared = ops.prepare_with_controls(
        q,
        k,
        v,
        float(scale_value),
        float(tau),
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
        int(topk_count),
        block_lengths,
    )
    output = ops.forward_cute_with_controls(
        q,
        k,
        v,
        *prepared,
        key_bias_log2,
        block_lengths,
        float(scale_value),
        bool(tail),
        topk_count >= 0,
    )
    if coarse_gate is not None:
        batch, _, heads, dim = q.shape
        q_means = prepared[2].reshape(batch * heads, blocks, dim)
        k_means = prepared[0].float().reshape(batch * heads, blocks, dim)
        v_means = prepared[1].float().reshape(batch * heads, blocks, dim)
        coarse = torch.softmax(
            torch.bmm(q_means, k_means.transpose(1, 2)) * scale_value,
            dim=-1,
        )
        coarse = torch.bmm(coarse, v_means)
        coarse = (
            coarse.view(batch, heads, blocks, dim)
            .permute(0, 2, 1, 3)
            .repeat_interleave(64, dim=1)[:, : q.shape[1]]
        )
        output.addcmul_(coarse_gate.contiguous(), coarse)
    return output


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
