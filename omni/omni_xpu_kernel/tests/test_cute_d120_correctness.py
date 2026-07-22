"""Correctness and layout contracts for the PTL-H CUTE D120 entry point."""

import pytest
import torch
import torch.nn.functional as F


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def has_d120():
    if not has_xpu():
        return False
    try:
        from omni_xpu_kernel import cute

        return cute.supports_d120_bhld()
    except Exception:
        return False


@pytest.mark.skipif(not has_d120(), reason="PTL-H CUTE D120 sidecar unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cute_d120_matches_torch_for_workflow_layouts(dtype):
    from omni_xpu_kernel import cute

    batch, heads, sequence, dim = 1, 4, 64, 120
    # Boogu Q is a BLHD-backed BHLD view, while K/V are packed BHLD tensors.
    q = torch.randn(
        batch, sequence, heads, dim, device="xpu", dtype=dtype
    ).permute(0, 2, 1, 3)
    k = torch.randn(batch, heads, sequence, dim, device="xpu", dtype=dtype)
    v = torch.randn_like(k)

    actual = cute.sdp_bhld_d120(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    assert actual.shape == expected.shape
    assert actual.stride() == q.stride()
    assert torch.isfinite(actual).all()
    rtol, atol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not has_d120(), reason="PTL-H CUTE D120 sidecar unavailable")
def test_cute_d120_rejects_wrong_head_dim():
    from omni_xpu_kernel import cute

    q = torch.randn(1, 4, 64, 128, device="xpu", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="head_dim==120"):
        cute.sdp_bhld_d120(q, q, q)


@pytest.mark.skipif(not has_d120(), reason="PTL-H CUTE D120 sidecar unavailable")
def test_cute_d120_rejects_unsupported_stride():
    from omni_xpu_kernel import cute

    base = torch.randn(1, 4, 64, 240, device="xpu", dtype=torch.float16)
    bad = base[..., ::2]
    assert bad.shape[-1] == 120 and bad.stride(-1) == 2
    with pytest.raises(RuntimeError, match="dense packed-BHLD or BLHD-backed"):
        cute.sdp_bhld_d120(bad, bad, bad)
