import pytest
import torch

from omni_xpu_kernel import rotary


def _adjacent_reference(x, freqs):
    paired = x.to(freqs.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if (
        paired.shape[2] != 1
        and freqs.shape[2] != 1
        and paired.shape[2] != freqs.shape[2]
    ):
        freqs = freqs[:, :, : paired.shape[2]]
    output = freqs[..., 0] * paired[..., 0]
    output.addcmul_(freqs[..., 1], paired[..., 1])
    return output.reshape_as(x).type_as(x)


def _split_reference(x, freqs):
    split = (
        x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs.dtype)
    )
    output = freqs[..., 0] * split[..., 0] + freqs[..., 1] * split[..., 1]
    return output.movedim(-1, -2).reshape_as(x).type_as(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BHND", "BNHD"])
@pytest.mark.parametrize("split_half", [False, True])
def test_kitchen_rope_arbitrary_matrix(dtype, freqs_dtype, layout, split_half):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if layout == "BHND":
        x = torch.randn(2, 3, 17, 64, device="xpu", dtype=dtype)
        freqs = torch.randn(2, 1, 17, 32, 2, 2, device="xpu", dtype=freqs_dtype)
    else:
        x = torch.randn(2, 17, 3, 64, device="xpu", dtype=dtype)
        freqs = torch.randn(1, 17, 1, 32, 2, 2, device="xpu", dtype=freqs_dtype)

    if split_half:
        actual = rotary.apply_kitchen_rope_split_half1(x, freqs)
        expected = _split_reference(x, freqs)
    else:
        actual = rotary.apply_kitchen_rope1(x, freqs)
        expected = _adjacent_reference(x, freqs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kitchen_rope_pair_allows_different_query_key_shapes():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    q = torch.randn(1, 4, 9, 64, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 9, 64, device="xpu", dtype=torch.bfloat16)
    freqs = torch.randn(1, 1, 9, 32, 2, 2, device="xpu", dtype=torch.float32)
    q_out, k_out = rotary.apply_kitchen_rope(q, k, freqs)
    torch.testing.assert_close(q_out, _adjacent_reference(q, freqs), rtol=0, atol=0)
    torch.testing.assert_close(k_out, _adjacent_reference(k, freqs), rtol=0, atol=0)


@pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("split_half", [False, True])
def test_kitchen_rope_pair_same_shape(freqs_dtype, split_half):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    q = torch.randn(2, 3, 17, 64, device="xpu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(2, 1, 17, 32, 2, 2, device="xpu", dtype=freqs_dtype)
    if split_half:
        q_out, k_out = rotary.apply_kitchen_rope_split_half(q, k, freqs)
        q_expected, k_expected = _split_reference(q, freqs), _split_reference(k, freqs)
    else:
        q_out, k_out = rotary.apply_kitchen_rope(q, k, freqs)
        q_expected, k_expected = _adjacent_reference(q, freqs), _adjacent_reference(k, freqs)
    torch.testing.assert_close(q_out, q_expected, rtol=0, atol=0)
    torch.testing.assert_close(k_out, k_expected, rtol=0, atol=0)


@pytest.mark.parametrize("freqs_dtype", [torch.float16, torch.bfloat16])
def test_kitchen_rope_adjacent_fallback_preserves_addcmul_rounding(freqs_dtype):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    base = torch.randn(2, 17, 3, 64, device="xpu", dtype=torch.bfloat16)
    x = base.transpose(1, 2)
    freqs = torch.randn(2, 1, 17, 32, 2, 2, device="xpu", dtype=freqs_dtype)

    assert not x.is_contiguous()
    assert not rotary._get_native().kitchen_rope_fast_supported(x, freqs)
    actual = rotary.apply_kitchen_rope1(x, freqs)
    expected = _adjacent_reference(x, freqs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "layout,shape",
    [
        ("BHND", (1, 24, 4352, 128)),
        ("BHND", (2, 32, 4996, 64)),
        ("BNHD", (1, 4096, 30, 128)),
        ("BNHD", (2, 12288, 16, 128)),
    ],
    ids=["FLUX", "LTX", "ZIMAGE", "WAN"],
)
@pytest.mark.parametrize("split_half", [False, True])
def test_kitchen_rope_real_workload_shapes(layout, shape, split_half):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    x = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    if layout == "BHND":
        b, _h, n, d = shape
        freqs = torch.randn(b, 1, n, d // 2, 2, 2, device="xpu", dtype=torch.float32)
    else:
        b, n, _h, d = shape
        freqs = torch.randn(1, n, 1, d // 2, 2, 2, device="xpu", dtype=torch.float32)
    assert rotary._get_native().kitchen_rope_fast_supported(x, freqs)
    if split_half:
        actual = rotary.apply_kitchen_rope_split_half1(x, freqs)
        expected = _split_reference(x, freqs)
    else:
        actual = rotary.apply_kitchen_rope1(x, freqs)
        expected = _adjacent_reference(x, freqs)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
