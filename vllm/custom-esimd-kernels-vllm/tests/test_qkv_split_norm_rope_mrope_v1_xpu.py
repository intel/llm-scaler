"""Accuracy and ABI regression tests for Qwen3.8 exact MRoPE v1."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.dlpack import to_dlpack

_DSOS = tuple(Path(__file__).parents[1].glob(
    "custom_esimd_kernels_gemv_only*.so"
))
if not _DSOS:
    pytest.skip("focused MRoPE DSO is not built", allow_module_level=True)
torch.ops.load_library(str(_DSOS[0]))
_QKV_OP = torch.ops.custom_esimd_kernels_vllm.esimd_qkv_split_norm_rope_mrope_v1


def _apply_mrope(value: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    out = value.float().clone()
    position_rows = positions.to(device="xpu", dtype=torch.long)
    pair_indices = torch.arange(32, device="xpu")
    axis = pair_indices.remainder(3)
    selected_positions = position_rows[axis, :].transpose(0, 1)
    selected = cache[selected_positions, pair_indices]
    cos = selected[:, None, :]
    sin = cache[selected_positions, pair_indices + 32][:, None, :]
    first = out[..., :32].clone()
    second = out[..., 32:64].clone()
    out[..., :32] = first * cos - second * sin
    out[..., 32:64] = second * cos + first * sin
    return out.to(torch.float16)


def _reference(
    qkv: torch.Tensor,
    norm_wq: torch.Tensor,
    norm_wk: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    q_gate = qkv[:, :1536].reshape(tokens, 3, 512)
    q = q_gate[..., :256]
    gate = q_gate[..., 256:].reshape(tokens, 768)
    k = qkv[:, 1536:1792].reshape(tokens, 1, 256)
    v = qkv[:, 1792:].reshape(tokens, 256)

    q = q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)
    q = q * (norm_wq.float() + 1.0)
    k = k.float() * torch.rsqrt(k.float().square().mean(-1, keepdim=True) + 1e-6)
    k = k * (norm_wk.float() + 1.0)
    q = _apply_mrope(q, positions, cache).reshape(tokens, 768).contiguous()
    k = _apply_mrope(k, positions, cache).reshape(tokens, 256).contiguous()
    return q, torch.sigmoid(gate), k, v.contiguous()


def _positions(tokens: int, dtype: torch.dtype, different: bool) -> torch.Tensor:
    base = torch.arange(1, tokens + 1, device="xpu", dtype=dtype)
    if different:
        return torch.stack((base, base + 17, base + 33), dim=0).contiguous()
    return base.repeat(3, 1).contiguous()


@pytest.mark.parametrize("tokens", [1, 2, 32])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("different_axes", [False, True])
def test_qwen38_mrope_v1_matches_reference(
    tokens: int, position_dtype: torch.dtype, different_axes: bool
) -> None:
    torch.manual_seed(1000 + tokens + int(position_dtype == torch.int64) * 10)
    qkv = torch.randn((tokens, 2048), dtype=torch.float16, device="xpu")
    norm_wq = torch.randn(256, dtype=torch.float16, device="xpu") * 0.1
    norm_wk = torch.randn(256, dtype=torch.float16, device="xpu") * 0.1
    cache = torch.randn((256, 64), dtype=torch.float16, device="xpu")
    positions = _positions(tokens, position_dtype, different_axes)
    q = torch.empty((tokens, 768), dtype=torch.float16, device="xpu")
    gate = torch.empty_like(q)
    k = torch.empty((tokens, 256), dtype=torch.float16, device="xpu")
    v = torch.empty_like(k)

    returned = _QKV_OP(
        qkv,
        q,
        gate,
        k,
        v,
        norm_wq,
        norm_wk,
        positions,
        3,
        1,
        True,
        True,
        cache,
    )
    torch.xpu.synchronize()
    q_ref, gate_ref, k_ref, v_ref = _reference(
        qkv, norm_wq, norm_wk, positions, cache
    )

    assert returned.data_ptr() == q.data_ptr()
    assert torch._C._is_alias_of(returned, q)
    assert torch.allclose(q, q_ref, atol=4e-3, rtol=4e-3)
    assert torch.allclose(k, k_ref, atol=4e-3, rtol=4e-3)
    assert torch.equal(gate, gate_ref)
    assert torch.equal(v, v_ref)


@pytest.mark.parametrize(
    ("case", "builder"),
    [
        ("q_heads", lambda q, q_out, gate, k, v: (2, 1, True, True)),
        ("qkv_dtype", lambda q, q_out, gate, k, v: (3, 1, True, True)),
        ("output_shape", lambda q, q_out, gate, k, v: (3, 1, True, True)),
        ("positions_layout", lambda q, q_out, gate, k, v: (3, 1, True, True)),
        ("position_proof", lambda q, q_out, gate, k, v: (3, 1, True, False)),
    ],
)
def test_qwen38_mrope_v1_rejects_invalid_contract(case, builder) -> None:
    tokens = 2
    qkv = torch.randn((tokens, 2048), dtype=torch.float16, device="xpu")
    norm_wq = torch.ones(256, dtype=torch.float16, device="xpu")
    norm_wk = torch.ones(256, dtype=torch.float16, device="xpu")
    cache = torch.ones((64, 64), dtype=torch.float16, device="xpu")
    positions = torch.zeros((3, tokens), dtype=torch.int32, device="xpu")
    q = torch.empty((tokens, 768), dtype=torch.float16, device="xpu")
    gate = torch.empty_like(q)
    k = torch.empty((tokens, 256), dtype=torch.float16, device="xpu")
    v = torch.empty_like(k)
    q_arg = qkv.to(torch.bfloat16) if case == "qkv_dtype" else qkv
    if case == "output_shape":
        q = torch.empty((tokens, 767), dtype=torch.float16, device="xpu")
    if case == "positions_layout":
        positions = torch.zeros((tokens, 3), dtype=torch.int32, device="xpu").t()
    q_heads, kv_heads, gate_enabled, proof = builder(q_arg, q, gate, k, v)

    with pytest.raises(RuntimeError):
        _QKV_OP(
            q_arg,
            q,
            gate,
            k,
            v,
            norm_wq,
            norm_wk,
            positions,
            q_heads,
            kv_heads,
            gate_enabled,
            proof,
            cache,
        )


def test_qwen38_mrope_v1_rejects_physical_output_overlap() -> None:
    qkv = torch.randn((1, 2048), dtype=torch.float16, device="xpu")
    norm = torch.ones(256, dtype=torch.float16, device="xpu")
    positions = torch.zeros((3, 1), dtype=torch.int32, device="xpu")
    cache = torch.ones((64, 64), dtype=torch.float16, device="xpu")
    q = torch.empty((1, 768), dtype=torch.float16, device="xpu")
    independent_q_wrapper = torch.from_dlpack(to_dlpack(q))
    k = torch.empty((1, 256), dtype=torch.float16, device="xpu")
    v = torch.empty_like(k)

    with pytest.raises(RuntimeError, match="overlap"):
        _QKV_OP(
            qkv,
            q,
            independent_q_wrapper,
            k,
            v,
            norm,
            norm,
            positions,
            3,
            1,
            True,
            True,
            cache,
        )
