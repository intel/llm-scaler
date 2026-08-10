import pytest
import torch

import omni_xpu_kernel
from omni_xpu_kernel import rotary


def _reference(input_tensor, cos, sin):
    batch, tokens, hidden = input_tensor.shape
    heads = cos.shape[1]
    head_dim = hidden // heads
    split_input = (
        input_tensor.reshape(batch, tokens, heads, head_dim)
        .swapaxes(1, 2)
        .reshape(batch, heads, tokens, 2, head_dim // 2)
    )
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]
    output = split_input * cos.unsqueeze(-2)
    output[..., :1, :].addcmul_(
        -sin.unsqueeze(-2), second_half_input
    )
    output[..., 1:, :].addcmul_(
        sin.unsqueeze(-2), first_half_input
    )
    return (
        output.reshape(batch, heads, tokens, head_dim)
        .swapaxes(1, 2)
        .reshape_as(input_tensor)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("transposed_freqs", [False, True])
def test_ltx_split_rope_direct_exact(dtype, head_dim, transposed_freqs):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if omni_xpu_kernel.core_aot_target() != "bmg":
        pytest.skip("BMG-specific direct LTX RoPE route")

    batch, tokens, heads = 2, 17, 3
    input_tensor = torch.randn(
        batch,
        tokens,
        heads * head_dim,
        device="xpu",
        dtype=dtype,
    )
    if transposed_freqs:
        cos = torch.randn(
            batch,
            tokens,
            heads,
            head_dim // 2,
            device="xpu",
            dtype=dtype,
        ).swapaxes(1, 2)
        sin = torch.randn(
            batch,
            tokens,
            heads,
            head_dim // 2,
            device="xpu",
            dtype=dtype,
        ).swapaxes(1, 2)
    else:
        cos = torch.randn(
            batch,
            heads,
            tokens,
            head_dim // 2,
            device="xpu",
            dtype=dtype,
        )
        sin = torch.randn_like(cos)

    assert rotary.ltx_split_rope_direct_supported(input_tensor, cos, sin)
    actual = rotary.apply_ltx_split_rope_direct(input_tensor, cos, sin)
    expected = _reference(input_tensor, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == input_tensor.stride()


@pytest.mark.parametrize(
    "batch,tokens,head_dim",
    [
        (1, 14_080, 128),
        (2, 3_520, 128),
        (1, 14_080, 64),
        (2, 3_520, 64),
    ],
)
def test_ltx_split_rope_direct_workflow_shapes(batch, tokens, head_dim):
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if omni_xpu_kernel.core_aot_target() != "bmg":
        pytest.skip("BMG-specific direct LTX RoPE route")

    heads = 32
    input_tensor = torch.randn(
        batch,
        tokens,
        heads * head_dim,
        device="xpu",
        dtype=torch.bfloat16,
    )
    cos = torch.randn(
        batch,
        tokens,
        heads,
        head_dim // 2,
        device="xpu",
        dtype=torch.bfloat16,
    ).swapaxes(1, 2)
    sin = torch.randn(
        batch,
        tokens,
        heads,
        head_dim // 2,
        device="xpu",
        dtype=torch.bfloat16,
    ).swapaxes(1, 2)

    assert rotary.ltx_split_rope_direct_supported(input_tensor, cos, sin)
    actual = rotary.apply_ltx_split_rope_direct(input_tensor, cos, sin)
    expected = _reference(input_tensor, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
