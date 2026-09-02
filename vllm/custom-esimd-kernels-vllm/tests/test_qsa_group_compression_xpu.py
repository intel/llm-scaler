"""Single-device parity and preflight checks for QSA compression ABI1."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch


def _xpu_available() -> bool:
    try:
        return torch.xpu.is_available() and torch.xpu.device_count() > 0
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA group compression requires an XPU"
)


def _load_qsa_extension():
    # Prefer the focused build artifact.  An installed older DSO must not
    # silently make this operator parity test exercise a different ABI.
    package_dir = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "custom_esimd_kernels_vllm"
    )
    candidates = sorted(package_dir.glob("qsa_ops*.so"))
    if len(candidates) == 1:
        spec = importlib.util.spec_from_file_location("qsa_ops", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSA extension: {candidates[0]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("custom_esimd_kernels_vllm.qsa_ops")


@pytest.fixture(scope="module")
def qsa_ops():
    return _load_qsa_extension()


def _make_case(dtype: torch.dtype):
    device = torch.device("xpu")
    pages, ring_size = 2, 8
    # Match QSAKeyStateCache's packed MRoPE storage: key and position views
    # share one allocation but occupy disjoint byte ranges per ring row.
    storage = torch.empty((pages, ring_size, 1, 140), dtype=dtype, device=device)
    cache = storage[..., :128]
    positions = storage[..., 128:].view(torch.int64)
    for page in range(pages):
        for slot in range(ring_size):
            cache[page, slot, 0].fill_(page * 100 + slot)
            positions[page, slot, 0] = torch.tensor(
                [page * 1000 + slot, page * 1000 + slot + 10, page * 1000 + slot + 20],
                dtype=torch.int64,
                device=device,
            )
    raw = torch.arange(128, dtype=torch.float32, device=device).reshape(1, 1, 128)
    raw = (raw / 17.0).to(dtype)
    raw_positions = torch.tensor(
        [[[11, 111, 211]]], dtype=torch.int64, device=device
    )
    block_table = torch.tensor([[1]], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    logical_positions = torch.tensor([11], dtype=torch.int64, device=device)
    compressed_slots = torch.tensor([7], dtype=torch.int64, device=device)
    pooled = torch.full((1, 1, 128), -99, dtype=dtype, device=device)
    first_positions = torch.full((1, 3), -99, dtype=torch.int64, device=device)
    return (
        raw,
        raw_positions,
        cache,
        positions,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qsa_group_compress_v1_matches_ring_reference(qsa_ops, dtype):
    case = _make_case(dtype)
    (
        raw,
        raw_positions,
        cache,
        positions,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
    ) = case
    expected = (
        cache[1, 0, 0].float()
        + cache[1, 1, 0].float()
        + cache[1, 2, 0].float()
        + raw[0, 0].float()
    ) / 4.0
    expected_position = positions[1, 0, 0].clone()

    returned = qsa_ops.qsa_group_compress_v1(
        raw,
        raw_positions,
        cache,
        positions,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        4,
        pages := cache.shape[0] * cache.shape[1],
        True,
    )
    torch.xpu.synchronize()

    assert returned.data_ptr() == pooled.data_ptr()
    assert torch.equal(first_positions, expected_position.reshape(1, 3))
    torch.testing.assert_close(
        pooled.float(), expected.reshape(1, 1, 128), atol=2e-2, rtol=2e-2
    )
    assert pages == 16


def test_qsa_group_compress_v1_invalid_block_matches_partial_fallback(qsa_ops):
    case = list(_make_case(torch.float16))
    case[4].fill_(-1)
    returned = qsa_ops.qsa_group_compress_v1(*case, 4, 16, True)
    torch.xpu.synchronize()

    torch.testing.assert_close(
        case[9].float(), case[0].float() / 4.0, atol=2e-2, rtol=2e-2
    )
    assert torch.count_nonzero(case[10]) == 0
    assert returned.data_ptr() == case[9].data_ptr()


def test_qsa_group_compress_v1_rejects_unproven_history(qsa_ops):
    case = _make_case(torch.float16)
    with pytest.raises(RuntimeError, match="historical ring proof"):
        qsa_ops.qsa_group_compress_v1(*case[:-2], case[-2], case[-1], 4, 16, False)


def test_qsa_group_compress_v1_rejects_wrong_decode_shape(qsa_ops):
    case = _make_case(torch.float16)
    raw = case[0].expand(2, 1, 128).contiguous()
    args = (raw, *case[1:])
    with pytest.raises(RuntimeError, match=r"shape \[1,1,128\]"):
        qsa_ops.qsa_group_compress_v1(*args, 4, 16, True)


def test_qsa_group_compress_v1_rejects_output_alias(qsa_ops):
    case = list(_make_case(torch.float16))
    case[9] = case[2][0, 0].view(1, 1, 128)
    with pytest.raises(RuntimeError, match="cache must not alias"):
        qsa_ops.qsa_group_compress_v1(*case, 4, 16, True)


def test_qsa_group_compress_v1_rejects_dlpack_physical_overlap(qsa_ops):
    case = list(_make_case(torch.float16))
    pooled_alias = torch.utils.dlpack.from_dlpack(
        torch.utils.dlpack.to_dlpack(case[2][0, 0].view(1, 1, 128))
    )
    case[9] = pooled_alias
    with pytest.raises(RuntimeError, match="outputs must not overlap"):
        qsa_ops.qsa_group_compress_v1(*case, 4, 16, True)
