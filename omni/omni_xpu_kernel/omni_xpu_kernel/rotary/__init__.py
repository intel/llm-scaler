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


__all__ = ["rotary_emb"]
