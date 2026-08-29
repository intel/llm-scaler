"""Correctness coverage for structural MiniMax H3 VideoVAE D64 tiles."""

import pytest
import torch
import torch.nn.functional as F


def has_bmg_h3_vae_d64():
    try:
        import omni_xpu_kernel
        from omni_xpu_kernel import cute

        return (
            torch.xpu.is_available()
            and omni_xpu_kernel.__xpu_target__ == "bmg"
            and cute is not None
            and cute.supports_minimax_h3_vae_d64()
        )
    except Exception:
        return False


@pytest.mark.parametrize(
    ("temporal_tokens", "tile_height", "tile_width"),
    [
        (1, 1, 1),
        (7, 1, 1),
        (7, 2, 2),
        (7, 2, 4),
        (7, 4, 4),
        (7, 4, 8),
        (1, 16, 16),
        (7, 8, 8),
        (7, 8, 16),
        (7, 16, 16),
    ],
)
@pytest.mark.skipif(
    not has_bmg_h3_vae_d64(), reason="MiniMax H3 VideoVAE D64 unavailable"
)
def test_minimax_h3_video_vae_d64_matches_structural_tile_family(
    temporal_tokens, tile_height, tile_width
):
    from omni_xpu_kernel import cute

    sequence = temporal_tokens * tile_height * tile_width + 5
    torch.xpu.manual_seed_all(20260803 + sequence)
    q = torch.randn(
        (1, sequence, 32, 64), device="xpu", dtype=torch.float16
    ).transpose(1, 2)
    k = torch.randn_like(q)
    qkv = torch.randn(
        (1, sequence, 32, 192), device="xpu", dtype=torch.float16
    )
    v = qkv[..., 128:].transpose(1, 2)

    assert q.stride() == k.stride() == (
        sequence * 2048,
        64,
        2048,
        1,
    )
    assert v.stride() == (sequence * 6144, 192, 6144, 1)
    actual = cute.sdp_minimax_h3_vae_d64(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    assert actual.shape == q.shape
    assert actual.stride() == q.stride()
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).abs()
    assert float(difference.max().item()) <= 0.0078125


@pytest.mark.skipif(
    not has_bmg_h3_vae_d64(), reason="MiniMax H3 VideoVAE D64 unavailable"
)
def test_minimax_h3_video_vae_d64_rejects_wrong_layout_and_dtype():
    from omni_xpu_kernel import cute

    sequence = 261
    dense = torch.randn(
        (1, 32, sequence, 64), device="xpu", dtype=torch.float16
    )
    with pytest.raises(RuntimeError, match="unsupported MiniMax H3 VAE Q/K/V layout"):
        cute.sdp_minimax_h3_vae_d64(dense, dense, dense)

    q = torch.randn(
        (1, sequence, 32, 64), device="xpu", dtype=torch.bfloat16
    ).transpose(1, 2)
    with pytest.raises(RuntimeError, match="requires FP16"):
        cute.sdp_minimax_h3_vae_d64(q, q, q)
