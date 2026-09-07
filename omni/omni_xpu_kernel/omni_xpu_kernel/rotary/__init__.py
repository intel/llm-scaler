import torch

from .. import _compile_meta as _meta
from .._compile_ops import compile_op


def _get_native():
    from .. import _load_extension

    return _load_extension().rotary


@compile_op("rotary_rotary_emb", _meta.unchanged)
def rotary_emb(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    seq_len: int,
    heads: int,
) -> torch.Tensor:
    """
    Fused rotary position embedding using ESIMD.

    Fuses bf16→f32 + rotary rotation + f32→bf16 into a single kernel.

    Args:
        x: [total_rows, head_dim] — flattened input (from [B, S, heads, head_dim])
        cos_cache: [S, head_dim/2] f32 — cosine components
        sin_cache: [S, head_dim/2] f32 — sine components
        seq_len: sequence length S
        heads: number of attention heads

    Returns:
        [total_rows, head_dim] — rotated tensor, same dtype as x
    """
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.rotary_rotary_emb(x, cos_cache, sin_cache, seq_len, heads)
    return _get_native().rotary_emb(x, cos_cache, sin_cache, seq_len, heads)


@compile_op("rotary_apply_kitchen_rope1", _meta.kitchen_rope1)
def apply_kitchen_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.rotary_apply_kitchen_rope1(x, freqs_cis)
    return _get_native().apply_kitchen_rope1(x, freqs_cis)


def apply_kitchen_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x.copy_(apply_kitchen_rope1(x, freqs_cis))
    return x


def supports_kitchen_rope_fast() -> bool:
    """Whether the loaded core exports the single-launch Kitchen RoPE path."""
    try:
        return hasattr(_get_native(), "kitchen_rope_fast_supported")
    except (AttributeError, ImportError, RuntimeError):
        return False


def kitchen_rope_fast_supported(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> bool:
    """Whether ``x`` and ``freqs_cis`` satisfy the native fast-path contract."""
    if not supports_kitchen_rope_fast():
        return False
    try:
        return bool(_get_native().kitchen_rope_fast_supported(x, freqs_cis))
    except (RuntimeError, TypeError):
        return False


@compile_op("rotary_apply_kitchen_rope", _meta.kitchen_rope)
def apply_kitchen_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.rotary_apply_kitchen_rope(xq, xk, freqs_cis)
    return _get_native().apply_kitchen_rope(xq, xk, freqs_cis)


def apply_kitchen_rope_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out, k_out = apply_kitchen_rope(xq, xk, freqs_cis)
    xq.copy_(q_out)
    xk.copy_(k_out)
    return xq, xk


@compile_op("rotary_apply_kitchen_rope_split_half1", _meta.kitchen_split1)
def apply_kitchen_rope_split_half1(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.rotary_apply_kitchen_rope_split_half1(x, freqs_cis)
    return _get_native().apply_kitchen_rope_split_half1(x, freqs_cis)


def apply_kitchen_rope_split_half1_(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    x.copy_(apply_kitchen_rope_split_half1(x, freqs_cis))
    return x


@compile_op("rotary_apply_kitchen_rope_split_half", _meta.kitchen_split)
def apply_kitchen_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.rotary_apply_kitchen_rope_split_half(xq, xk, freqs_cis)
    return _get_native().apply_kitchen_rope_split_half(xq, xk, freqs_cis)


def apply_kitchen_rope_split_half_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out, k_out = apply_kitchen_rope_split_half(xq, xk, freqs_cis)
    xq.copy_(q_out)
    xk.copy_(k_out)
    return xq, xk


@compile_op("rotary_rms_rope1", _meta.rms_rope1)
def _rms_rope1_allocate(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor,
    epsilon: float, split_half: bool, rot_dim: int,
) -> torch.Tensor:
    return _get_native().rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half, rot_dim, False
    )


@compile_op("rotary_rms_rope1_", _meta.void, mutates_args=("x",))
def _rms_rope1_mutate(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor,
    epsilon: float, split_half: bool, rot_dim: int,
) -> None:
    _get_native().rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half, rot_dim, True
    )


@compile_op("rotary_rms_rope", _meta.rms_rope)
def _rms_rope_allocate(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor,
    epsilon: float, split_half: bool, rot_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_native().rms_kitchen_rope(
        q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, rot_dim, False
    )


@compile_op("rotary_rms_rope_", _meta.void, mutates_args=("q", "k"))
def _rms_rope_mutate(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor,
    epsilon: float, split_half: bool, rot_dim: int,
) -> None:
    _get_native().rms_kitchen_rope(
        q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, rot_dim, True
    )


def _rms_kitchen_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    *,
    split_half: bool,
    rot_dim: int = 0,
    inplace: bool,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        if inplace:
            torch.ops.omni_xpu.rotary_rms_rope1_(
                x, freqs_cis, scale, epsilon, split_half, rot_dim
            )
            return x
        return torch.ops.omni_xpu.rotary_rms_rope1(
            x, freqs_cis, scale, epsilon, split_half, rot_dim
        )
    return _get_native().rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half, rot_dim, inplace
    )


def _rms_kitchen_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None,
    epsilon: float,
    *,
    split_half: bool,
    rot_dim: int = 0,
    inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    if torch.compiler.is_compiling():
        if inplace:
            torch.ops.omni_xpu.rotary_rms_rope_(
                q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, rot_dim
            )
            return q, k
        return torch.ops.omni_xpu.rotary_rms_rope(
            q, k, freqs_cis, q_scale, k_scale, epsilon, split_half, rot_dim
        )
    return _get_native().rms_kitchen_rope(
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale,
        epsilon,
        split_half,
        rot_dim,
        inplace,
    )


def rms_kitchen_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half=False, inplace=False
    )


def rms_kitchen_rope1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half=False, inplace=True
    )


def rms_kitchen_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_kitchen_rope(
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale,
        epsilon,
        split_half=False,
        inplace=False,
    )


def rms_kitchen_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_kitchen_rope(
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale,
        epsilon,
        split_half=False,
        inplace=True,
    )


def rms_kitchen_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half=True, inplace=False
    )


def rms_kitchen_rope_split_half1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_kitchen_rope1(
        x, freqs_cis, scale, epsilon, split_half=True, inplace=True
    )


def rms_kitchen_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_kitchen_rope(
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale,
        epsilon,
        split_half=True,
        rot_dim=rot_dim,
        inplace=False,
    )


def rms_kitchen_rope_split_half_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rms_kitchen_rope(
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale,
        epsilon,
        split_half=True,
        rot_dim=rot_dim,
        inplace=True,
    )


def supports_ltx_split_rope_direct() -> bool:
    """Whether the loaded core exports direct LTX cos/sin split-half RoPE."""
    try:
        return hasattr(_get_native(), "ltx_split_rope_direct_supported")
    except (AttributeError, ImportError, RuntimeError):
        return False


def ltx_split_rope_direct_supported(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Whether the tensors satisfy the native direct LTX RoPE contract."""
    if not supports_ltx_split_rope_direct():
        return False
    try:
        return bool(
            _get_native().ltx_split_rope_direct_supported(input, cos, sin)
        )
    except (RuntimeError, TypeError):
        return False


@compile_op("rotary_apply_ltx_split_rope_direct", _meta.unchanged)
def apply_ltx_split_rope_direct(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply split-half LTX RoPE without constructing a 2x2 matrix."""
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.rotary_apply_ltx_split_rope_direct(input, cos, sin)
    return _get_native().apply_ltx_split_rope_direct(input, cos, sin)


__all__ = [
    "apply_ltx_split_rope_direct",
    "apply_kitchen_rope",
    "apply_kitchen_rope_",
    "apply_kitchen_rope1",
    "apply_kitchen_rope1_",
    "apply_kitchen_rope_split_half",
    "apply_kitchen_rope_split_half_",
    "apply_kitchen_rope_split_half1",
    "apply_kitchen_rope_split_half1_",
    "kitchen_rope_fast_supported",
    "ltx_split_rope_direct_supported",
    "rotary_emb",
    "rms_kitchen_rope",
    "rms_kitchen_rope_",
    "rms_kitchen_rope1",
    "rms_kitchen_rope1_",
    "rms_kitchen_rope_split_half",
    "rms_kitchen_rope_split_half_",
    "rms_kitchen_rope_split_half1",
    "rms_kitchen_rope_split_half1_",
    "supports_kitchen_rope_fast",
    "supports_ltx_split_rope_direct",
]
