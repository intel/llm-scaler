import pytest
import torch

import omni_xpu_kernel
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


def _partial_rms_split_reference(x, freqs, scale, epsilon, rot_dim):
    x_float = x.float()
    inverse_rms = torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + epsilon
    )
    normalized = (x_float * inverse_rms * scale.float()).to(x.dtype)
    rotated = _split_reference(normalized[..., :rot_dim], freqs)
    return torch.cat((rotated, normalized[..., rot_dim:]), dim=-1)


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


def test_kitchen_rope_bmg_krea2_pair_exact():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if omni_xpu_kernel.core_aot_target() != "bmg":
        pytest.skip("BMG-specific Krea2 pair route")

    q = torch.randn(
        1, 48, 4192, 128, device="xpu", dtype=torch.bfloat16
    )
    k = torch.randn(
        1, 12, 4192, 128, device="xpu", dtype=torch.bfloat16
    )
    freqs = torch.randn(
        1, 1, 4192, 64, 2, 2, device="xpu", dtype=torch.float32
    )
    q_out, k_out = rotary.apply_kitchen_rope(q, k, freqs)
    torch.testing.assert_close(
        q_out, _adjacent_reference(q, freqs), rtol=0, atol=0
    )
    torch.testing.assert_close(
        k_out, _adjacent_reference(k, freqs), rtol=0, atol=0
    )


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
    assert not rotary.kitchen_rope_fast_supported(x, freqs)
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
    assert rotary.kitchen_rope_fast_supported(x, freqs)
    if split_half:
        actual = rotary.apply_kitchen_rope_split_half1(x, freqs)
        expected = _split_reference(x, freqs)
    else:
        actual = rotary.apply_kitchen_rope1(x, freqs)
        expected = _adjacent_reference(x, freqs)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "sequence_length,heads",
    [(109, 7), (109, 28), (4096, 7), (4205, 28)],
)
@pytest.mark.parametrize("split_half", [False, True])
def test_kitchen_rope_bmg_d120_single_exact(
    sequence_length, heads, split_half
):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if omni_xpu_kernel.core_aot_target() != "bmg":
        pytest.skip("BMG-specific D120 route")

    x = torch.randn(
        1,
        sequence_length,
        heads,
        120,
        device="xpu",
        dtype=torch.float16,
    )
    freqs = torch.randn(
        1,
        sequence_length,
        1,
        60,
        2,
        2,
        device="xpu",
        dtype=torch.float32,
    )
    assert rotary.kitchen_rope_fast_supported(x, freqs)
    if split_half:
        actual = rotary.apply_kitchen_rope_split_half1(x, freqs)
        expected = _split_reference(x, freqs)
    else:
        actual = rotary.apply_kitchen_rope1(x, freqs)
        expected = _adjacent_reference(x, freqs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_h3_packed_qkv_partial_rms_rope_inplace():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")

    sequence, heads, head_dim, rot_dim = 37, 56, 128, 96
    inner = heads * head_dim
    torch.xpu.manual_seed_all(20260804)
    packed = torch.randn(
        sequence,
        3 * inner,
        device="xpu",
        dtype=torch.bfloat16,
    )
    q = packed[:, :inner].view(1, sequence, heads, head_dim)
    k = packed[:, inner : 2 * inner].view(1, sequence, heads, head_dim)
    q_before = q.clone()
    k_before = k.clone()
    q_scale = torch.randn(head_dim, device="xpu", dtype=torch.float32)
    k_scale = torch.randn(head_dim, device="xpu", dtype=torch.float32)
    freqs = torch.randn(
        1,
        sequence,
        1,
        rot_dim // 2,
        2,
        2,
        device="xpu",
        dtype=torch.bfloat16,
    )

    assert not q.is_contiguous()
    q_pointer, k_pointer = q.data_ptr(), k.data_ptr()
    q_out, k_out = rotary.rms_kitchen_rope_split_half_(
        q,
        k,
        freqs,
        q_scale,
        k_scale,
        epsilon=1e-5,
        rot_dim=rot_dim,
    )
    q_expected = _partial_rms_split_reference(
        q_before, freqs, q_scale, 1e-5, rot_dim
    )
    k_expected = _partial_rms_split_reference(
        k_before, freqs, k_scale, 1e-5, rot_dim
    )

    assert q_out.data_ptr() == q_pointer
    assert k_out.data_ptr() == k_pointer
    torch.testing.assert_close(q_out, q_expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(k_out, k_expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("sequence", [31, 388, 1025])
def test_h3_bmg_bf16_scale_cached_rms_rope_matches_generic(sequence):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if omni_xpu_kernel.core_aot_target() != "bmg":
        pytest.skip("BMG-specific MiniMax H3 RMS-RoPE route")

    heads, head_dim, rot_dim = 56, 128, 96
    inner = heads * head_dim
    torch.xpu.manual_seed_all(20260804 + sequence)
    packed = torch.randn(
        sequence,
        3 * inner,
        device="xpu",
        dtype=torch.bfloat16,
    )
    candidate_packed = packed.clone()
    generic_packed = packed.clone()

    def qk_views(storage):
        q = storage[:, :inner].view(1, sequence, heads, head_dim)
        k = storage[:, inner : 2 * inner].view(
            1, sequence, heads, head_dim
        )
        return q, k

    candidate_q, candidate_k = qk_views(candidate_packed)
    generic_q, generic_k = qk_views(generic_packed)
    freqs = torch.randn(
        1,
        sequence,
        1,
        rot_dim // 2,
        2,
        2,
        device="xpu",
        dtype=torch.bfloat16,
    )
    q_scale = torch.linspace(
        0.75, 1.25, head_dim, device="xpu", dtype=torch.bfloat16
    )
    k_scale = torch.linspace(
        1.25, 0.75, head_dim, device="xpu", dtype=torch.bfloat16
    )
    q_pointer, k_pointer = candidate_q.data_ptr(), candidate_k.data_ptr()

    candidate_q, candidate_k = rotary.rms_kitchen_rope_split_half_(
        candidate_q,
        candidate_k,
        freqs,
        q_scale,
        k_scale,
        epsilon=1e-5,
        rot_dim=rot_dim,
    )
    # FP32 scales deliberately select the generic path while preserving the
    # exact values represented by the workflow's BF16 RMSNorm weights.
    generic_q, generic_k = rotary.rms_kitchen_rope_split_half_(
        generic_q,
        generic_k,
        freqs,
        q_scale.float(),
        k_scale.float(),
        epsilon=1e-5,
        rot_dim=rot_dim,
    )

    assert candidate_q.data_ptr() == q_pointer
    assert candidate_k.data_ptr() == k_pointer
    assert torch.isfinite(candidate_q).all()
    assert torch.isfinite(candidate_k).all()
    torch.testing.assert_close(candidate_q, generic_q, rtol=0, atol=0)
    torch.testing.assert_close(candidate_k, generic_k, rtol=0, atol=0)
