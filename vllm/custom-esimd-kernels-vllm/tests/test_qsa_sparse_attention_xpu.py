"""Correctness checks for the fixed-contract Qwen3.8 QSA XPU extension."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch


ROWS = 2
HEADS = 3
HEAD_DIM = 256
PAGE_SIZE = 512
INDEX_WIDTH = 2051
PACKED_STRIDE = (262144, 512, 256, 1)


def _xpu_available() -> bool:
    try:
        return torch.xpu.is_available() and torch.xpu.device_count() > 0
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA validation requires an XPU"
)


def _load_qsa_extension():
    try:
        return importlib.import_module("custom_esimd_kernels_vllm.qsa_ops")
    except ImportError:
        # Focused in-place builds may not contain the package's other modules.
        package_dir = (
            Path(__file__).resolve().parents[1]
            / "python"
            / "custom_esimd_kernels_vllm"
        )
        candidates = sorted(package_dir.glob("qsa_ops*.so"))
        if len(candidates) != 1:
            raise
        spec = importlib.util.spec_from_file_location("qsa_ops", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSA extension: {candidates[0]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="module")
def qsa_ops():
    return _load_qsa_extension()


def _canonicalize_singleton_dim_strides(tensor: torch.Tensor) -> torch.Tensor:
    strides = list(tensor.stride())
    previous_stride = 1
    changed = False
    for dim in range(tensor.dim() - 1, -1, -1):
        if tensor.shape[dim] == 1 and strides[dim] != previous_stride:
            strides[dim] = previous_stride
            changed = True
        previous_stride = strides[dim] * tensor.shape[dim]
    return tensor.as_strided(tensor.shape, strides) if changed else tensor


def _make_inputs(case: str):
    pages_per_request = 5
    pages = ROWS * pages_per_request
    packed_kv = torch.randn(
        pages,
        1,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=torch.float16,
        device="xpu",
    )
    k_cache, v_cache = packed_kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    k_cache = _canonicalize_singleton_dim_strides(k_cache)
    v_cache = _canonicalize_singleton_dim_strides(v_cache)
    q = 0.1 * torch.randn(
        ROWS, HEADS, HEAD_DIM, dtype=torch.float16, device="xpu"
    )
    logical_indices = torch.full(
        (ROWS, INDEX_WIDTH), -1, dtype=torch.int32, device="xpu"
    )
    block_table = torch.arange(
        pages, dtype=torch.int32, device="xpu"
    ).view(ROWS, pages_per_request)
    token_to_req = torch.arange(ROWS, dtype=torch.int32, device="xpu")

    if case == "empty":
        pass
    elif case == "holes_duplicates_pages":
        values = torch.tensor(
            [0, 511, -1, 512, 513, 512, -1, 1023, 1024, 1535],
            dtype=torch.int32,
            device="xpu",
        )
        for row in range(ROWS):
            logical_indices[row, : values.numel()] = torch.roll(values, row)
    elif case == "valid_width_32":
        logical_indices[:, :32] = torch.arange(
            32, dtype=torch.int32, device="xpu"
        )
    elif case == "full_width_2051":
        logical_indices[:] = torch.arange(
            INDEX_WIDTH, dtype=torch.int32, device="xpu"
        )
    else:
        raise ValueError(case)

    assert tuple(k_cache.stride()) == PACKED_STRIDE
    assert tuple(v_cache.stride()) == PACKED_STRIDE
    assert v_cache.storage_offset() == HEAD_DIM
    return (
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        torch.empty_like(q),
        packed_kv,
    )


def _reference(q, k_cache, v_cache, logical_indices, block_table, token_to_req):
    output = torch.zeros_like(q)
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if logical.numel() == 0:
            continue
        request = int(token_to_req[row].item())
        pages = block_table[request, logical // PAGE_SIZE].long()
        offsets = logical % PAGE_SIZE
        keys = k_cache[pages, offsets, 0]
        values = v_cache[pages, offsets, 0]
        scores = torch.einsum("hd,kd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * (HEAD_DIM**-0.5), dim=-1)
        output[row] = torch.einsum(
            "hk,kd->hd", probabilities, values.float()
        ).to(q.dtype)
    return output


def test_qsa_module_contract(qsa_ops):
    assert qsa_ops.activation_dtype == "float16"
    assert qsa_ops.exact_packed_cache == 1
    assert qsa_ops.kv_tile == 4
    assert qsa_ops.subgroup == 32
    assert qsa_ops.token_loader == 1
    assert qsa_ops.trim_valid_width == 1
    assert qsa_ops.packed_kv == 1
    assert qsa_ops.fast_exp == 1


@pytest.mark.parametrize(
    "case",
    ["empty", "holes_duplicates_pages", "valid_width_32", "full_width_2051"],
)
def test_qsa_matches_reference_and_preserves_inputs(qsa_ops, case):
    torch.manual_seed(20260828)
    args = list(_make_inputs(case))
    packed_kv = args.pop()
    q, _, _, logical_indices, block_table, token_to_req, out = args
    snapshots = [
        q.clone(),
        packed_kv.clone(),
        logical_indices.clone(),
        block_table.clone(),
        token_to_req.clone(),
    ]
    expected = _reference(*args[:6])

    returned = qsa_ops.sparse_paged_attention(*args)
    torch.xpu.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    assert torch.isfinite(returned).all()
    assert torch.allclose(
        returned.float(), expected.float(), atol=2e-2, rtol=2e-2
    )
    for actual, snapshot in zip(
        [q, packed_kv, logical_indices, block_table, token_to_req], snapshots
    ):
        assert torch.equal(actual, snapshot)


def test_qsa_rejects_bf16_query(qsa_ops):
    args = list(_make_inputs("valid_width_32")[:-1])
    args[0] = args[0].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="q must be float16"):
        qsa_ops.sparse_paged_attention(*args)


def test_qsa_rejects_nonpacked_cache(qsa_ops):
    args = list(_make_inputs("valid_width_32")[:-1])
    args[1] = args[1].contiguous()
    args[2] = args[2].contiguous()
    with pytest.raises(RuntimeError, match="must have exact packed strides"):
        qsa_ops.sparse_paged_attention(*args)
