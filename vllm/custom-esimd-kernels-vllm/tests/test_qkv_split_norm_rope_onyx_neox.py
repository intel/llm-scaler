"""Numerical regression for Muse Glimmer's fused QK-norm and NEOX RoPE."""

import torch

from custom_esimd_kernels_vllm import (
    esimd_qkv_split_norm_rope_muse_glimmer,
    esimd_qkv_split_norm_rope_muse_glimmer_neox,
)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    neox_style: bool,
) -> torch.Tensor:
    if neox_style:
        return torch.cat(
            (x[..., :64] * cos - x[..., 64:] * sin,
             x[..., 64:] * cos + x[..., :64] * sin),
            dim=-1,
        )
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin),
        dim=-1,
    ).flatten(-2)


def check(
    m: int,
    position_dtype: torch.dtype,
    neox_style: bool,
) -> None:
    q_heads, kv_heads, head_dim = 16, 1, 128
    hidden = (q_heads + 2 * kv_heads) * head_dim
    positions = torch.tensor(
        [0, 3, 7][:m], dtype=position_dtype, device="xpu"
    )
    qkv = torch.randn(m, hidden, dtype=torch.float16, device="xpu")
    cos = torch.randn(16, 64, dtype=torch.float16, device="xpu")
    sin = torch.randn(16, 64, dtype=torch.float16, device="xpu")
    cache = torch.cat((cos, sin), dim=-1).contiguous()
    q = torch.empty(m, q_heads * head_dim, dtype=torch.float16, device="xpu")
    k = torch.empty(m, kv_heads * head_dim, dtype=torch.float16, device="xpu")
    v = torch.empty_like(k)

    q_scale = 3.87
    eps = 1e-5 if neox_style else 1e-6
    if neox_style:
        esimd_qkv_split_norm_rope_muse_glimmer_neox(
            qkv, q, k, v, positions, q_heads, kv_heads, q_scale, eps, cache
        )
    else:
        esimd_qkv_split_norm_rope_muse_glimmer(
            qkv, q, k, v, positions, q_heads, kv_heads, q_scale, cache
        )
    torch.xpu.synchronize()

    chunks = qkv.reshape(m, q_heads + 2 * kv_heads, head_dim)
    normed = chunks[:, : q_heads + kv_heads].float()
    normed = normed * torch.rsqrt(
        normed.square().mean(dim=-1, keepdim=True) + eps
    )
    normed[:, :q_heads] *= q_scale
    c = cos[positions.long()].float()
    s = sin[positions.long()].float()
    q_ref = normed[:, :q_heads].reshape(m, q_heads, 128)
    k_ref = normed[:, q_heads:].reshape(m, kv_heads, 128)
    q_ref = apply_rope(q_ref, c[:, None], s[:, None], neox_style)
    k_ref = apply_rope(k_ref, c[:, None], s[:, None], neox_style)
    q_ref = q_ref.reshape_as(q).to(torch.float16)
    k_ref = k_ref.reshape_as(k).to(torch.float16)
    v_ref = chunks[:, q_heads + kv_heads:].reshape_as(v)
    torch.xpu.synchronize()
    print(
        f"m={m} positions={position_dtype} neox={neox_style}:",
        "q max", (q.float() - q_ref.float()).abs().max().item(),
        "k max", (k.float() - k_ref.float()).abs().max().item(),
        "v max", (v.float() - v_ref.float()).abs().max().item(),
    )
    assert torch.allclose(q, q_ref, atol=3e-3, rtol=3e-3)
    assert torch.allclose(k, k_ref, atol=3e-3, rtol=3e-3)
    assert torch.equal(v, v_ref)


def test_supported_position_dtypes() -> None:
    torch.xpu.set_device(0)
    torch.manual_seed(0)
    check(3, torch.int32, False)
    check(3, torch.int64, False)
    check(1, torch.int64, True)
    check(3, torch.int32, True)
    check(3, torch.int64, True)


if __name__ == "__main__":
    test_supported_position_dtypes()
