import torch


def _get_native():
    from .. import _load_extension

    return _load_extension().rotary


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
    return _get_native().rotary_emb(x, cos_cache, sin_cache, seq_len, heads)


def apply_kitchen_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _get_native().apply_kitchen_rope1(x, freqs_cis)


def apply_kitchen_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_native().apply_kitchen_rope(xq, xk, freqs_cis)


def apply_kitchen_rope_split_half1(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    return _get_native().apply_kitchen_rope_split_half1(x, freqs_cis)


def apply_kitchen_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_native().apply_kitchen_rope_split_half(xq, xk, freqs_cis)


__all__ = [
    "apply_kitchen_rope",
    "apply_kitchen_rope1",
    "apply_kitchen_rope_split_half",
    "apply_kitchen_rope_split_half1",
    "rotary_emb",
]
