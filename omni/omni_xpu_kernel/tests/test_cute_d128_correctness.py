"""Correctness coverage for the public CUTE D128 entry point."""

import pytest
import torch
import torch.nn.functional as F


def has_bmg_cute():
    try:
        import omni_xpu_kernel
        from omni_xpu_kernel import cute

        return (
            torch.xpu.is_available()
            and omni_xpu_kernel.__xpu_target__ == "bmg"
            and cute is not None
            and cute.is_available()
        )
    except Exception:
        return False


@pytest.mark.skipif(
    not has_bmg_cute(), reason="BMG CUTE D128 sidecar unavailable"
)
def test_cute_d128_bmg_matches_zimage_workflow_contract():
    from omni_xpu_kernel import cute

    torch.xpu.manual_seed_all(20260726)
    shape = (1, 4128, 30, 128)
    q = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    v = torch.randn(shape, device="xpu", dtype=torch.bfloat16)

    actual = cute.sdp(q, k, v)
    expected = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
    ).transpose(1, 2).contiguous()

    assert actual.shape == shape
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= 0.001953125
