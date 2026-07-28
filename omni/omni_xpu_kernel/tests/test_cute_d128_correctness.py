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


@pytest.mark.skipif(
    not has_bmg_cute(), reason="BMG CUTE D128 sidecar unavailable"
)
def test_cute_d128_bmg_matches_wan22_t2v_turbo_720p_cross_contract():
    from omni_xpu_kernel import cute

    if not cute.supports_wan22_cross():
        pytest.skip("Wan 2.2 cross-attention capability is unavailable")

    torch.xpu.manual_seed_all(20260727)
    q = torch.randn(
        (1, 75600, 40, 128),
        device="xpu",
        dtype=torch.float16,
    )
    k = torch.randn(
        (1, 512, 40, 128),
        device="xpu",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    actual = cute.sdp_wan22_cross(q, k, v)
    expected = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
    ).transpose(1, 2).contiguous()

    assert actual.shape == q.shape
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).abs()
    assert float(difference.max().item()) <= 0.00390625


@pytest.mark.parametrize("kv_len", [3520, 1024])
@pytest.mark.skipif(
    not has_bmg_cute(), reason="BMG CUTE D128 sidecar unavailable"
)
def test_cute_d128_bmg_matches_ltx23_b2_bhld_contracts(kv_len):
    from omni_xpu_kernel import cute

    if not cute.supports_d128_bhld():
        pytest.skip("D128 BHLD attention capability is unavailable")

    torch.xpu.manual_seed_all(20260727 + kv_len)
    q = torch.randn(
        (2, 3520, 32, 128),
        device="xpu",
        dtype=torch.bfloat16,
    ).permute(0, 2, 1, 3)
    k = torch.randn(
        (2, kv_len, 32, 128),
        device="xpu",
        dtype=torch.bfloat16,
    ).permute(0, 2, 1, 3)
    v = torch.randn(
        (2, kv_len, 32, 128),
        device="xpu",
        dtype=torch.bfloat16,
    ).permute(0, 2, 1, 3)

    actual = cute.sdp_bhld_d128(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    assert actual.shape == q.shape
    assert actual.stride() == q.stride()
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).abs()
    assert float(difference.max().item()) <= 0.00390625


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["packed_bhld", "blhd_backed"])
@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "kv_len"),
    [
        (1, 1, 1, 1),
        (1, 4, 31, 32),
        (1, 4, 32, 31),
        (2, 7, 33, 65),
        (3, 4, 255, 257),
    ],
)
@pytest.mark.skipif(
    not has_bmg_cute(), reason="BMG CUTE D128 sidecar unavailable"
)
def test_cute_d128_bmg_bhld_coverage_matrix(
    dtype, layout, batch, heads, q_len, kv_len
):
    from omni_xpu_kernel import cute

    if not cute.supports_d128_bhld():
        pytest.skip("D128 BHLD attention capability is unavailable")

    torch.xpu.manual_seed_all(
        20260727 + batch + heads + q_len + kv_len
    )

    def make_tensor(sequence):
        if layout == "packed_bhld":
            return torch.randn(
                (batch, heads, sequence, 128),
                device="xpu",
                dtype=dtype,
            )
        return torch.randn(
            (batch, sequence, heads, 128),
            device="xpu",
            dtype=dtype,
        ).permute(0, 2, 1, 3)

    q = make_tensor(q_len)
    k = make_tensor(kv_len)
    v = make_tensor(kv_len)
    actual = cute.sdp_bhld_d128(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    assert actual.shape == q.shape
    assert actual.stride() == q.stride()
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).abs()
    bound = 0.00390625 if dtype == torch.float16 else 0.0078125
    assert float(difference.max().item()) <= bound


@pytest.mark.skipif(
    not has_bmg_cute(), reason="BMG CUTE D128 sidecar unavailable"
)
def test_cute_d128_bmg_bhld_rejects_unsupported_contracts():
    from omni_xpu_kernel import cute

    if not cute.supports_d128_bhld():
        pytest.skip("D128 BHLD attention capability is unavailable")

    bad_layout = torch.randn(
        (1, 4, 33, 256),
        device="xpu",
        dtype=torch.bfloat16,
    )[..., ::2]
    with pytest.raises(RuntimeError, match="dense packed-BHLD"):
        cute.sdp_bhld_d128(bad_layout, bad_layout, bad_layout)

    bad_dtype = torch.randn(
        (1, 4, 33, 128),
        device="xpu",
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match="supports fp16/bf16"):
        cute.sdp_bhld_d128(bad_dtype, bad_dtype, bad_dtype)


@pytest.mark.skipif(
    not has_bmg_cute(), reason="BMG CUTE D128 sidecar unavailable"
)
def test_cute_d128_bhld_api_does_not_widen_legacy_sdp_contract():
    from omni_xpu_kernel import cute

    batched = torch.randn(
        (2, 33, 4, 128),
        device="xpu",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="only B==1"):
        cute.sdp(batched, batched, batched)

    q = torch.randn(
        (1, 33, 4, 128),
        device="xpu",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        (1, 32, 4, 128),
        device="xpu",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="only self-attention"):
        cute.sdp(q, kv, kv)
