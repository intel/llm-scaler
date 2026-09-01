"""Accuracy and contract checks for QSA cache row-store ABI3."""

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
    not _xpu_available(), reason="QSA row-store validation requires an XPU"
)


def _load_qsa_extension():
    try:
        return importlib.import_module("custom_esimd_kernels_vllm.qsa_ops")
    except ImportError:
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


def _make_cache(dtype: torch.dtype, width: int):
    pages, page_size = 3, 4
    token_stride = width + 7
    page_stride = page_size * token_stride + 11
    storage_offset = 3
    storage_size = (
        storage_offset
        + (pages - 1) * page_stride
        + (page_size - 1) * token_stride
        + width
        + 5
    )
    fill = -23 if dtype == torch.int64 else -23.0
    backing = torch.full(
        (storage_size,), fill, dtype=dtype, device="xpu"
    )
    cache = torch.as_strided(
        backing,
        (pages, page_size, 1, width),
        (page_stride, token_stride, width, 1),
        storage_offset,
    )
    return cache, backing


def _make_rows(dtype: torch.dtype, width: int, count: int):
    row_stride = width + 5
    storage_offset = 2
    storage_size = storage_offset + max(count - 1, 0) * row_stride + width + 3
    backing = torch.full(
        (storage_size,), -71, dtype=dtype, device="xpu"
    )
    rows = torch.as_strided(
        backing, (count, width), (row_stride, 1), storage_offset
    )
    values = torch.arange(count * width, dtype=torch.int64).reshape(count, width)
    if dtype != torch.int64:
        values = (values.remainder(97).float() - 48.0) / 17.0
    rows.copy_(values.to(dtype=dtype, device="xpu"))
    return rows, backing


@pytest.mark.parametrize(
    ("dtype", "width"),
    [
        (torch.float16, 128),
        (torch.bfloat16, 128),
        (torch.int64, 3),
    ],
)
@pytest.mark.parametrize("rank3_rows", [False, True])
def test_qsa_store_cache_rows_v3_matches_strided_reference(
    qsa_ops, dtype: torch.dtype, width: int, rank3_rows: bool
):
    cache, cache_backing = _make_cache(dtype, width)
    rows, rows_backing = _make_rows(dtype, width, 5)
    rows_arg = rows.unsqueeze(1) if rank3_rows else rows
    capacity = cache.shape[0] * cache.shape[1]
    slot_values = [0, capacity - 1, -1, cache.shape[1] + 1, capacity]
    slots = torch.tensor(slot_values, dtype=torch.int64, device="xpu")

    cache_before = cache_backing.cpu().clone()
    rows_before = rows_backing.cpu().clone()
    slots_before = slots.cpu().clone()
    rows_host = rows.cpu()
    expected = cache_before.clone()
    for row, slot in enumerate(slot_values):
        if not 0 <= slot < capacity:
            continue
        page, token = divmod(slot, cache.shape[1])
        begin = (
            cache.storage_offset()
            + page * cache.stride(0)
            + token * cache.stride(1)
        )
        expected[begin : begin + width] = rows_host[row]

    result = qsa_ops.qsa_store_cache_rows_v3(cache, slots, rows_arg)
    torch.xpu.synchronize()

    assert result.data_ptr() == cache.data_ptr()
    assert torch._C._is_alias_of(result, cache)
    assert torch.equal(cache_backing.cpu(), expected)
    assert torch.equal(rows_backing.cpu(), rows_before)
    assert torch.equal(slots.cpu(), slots_before)


def test_qsa_store_cache_rows_v3_duplicate_slot_last_row_wins(qsa_ops):
    cache = torch.zeros((1, 4, 1, 128), dtype=torch.float16, device="xpu")
    slots = torch.tensor([2, 2], dtype=torch.int64, device="xpu")
    rows = torch.stack(
        (
            torch.full((128,), 3.0, dtype=torch.float16, device="xpu"),
            torch.full((128,), 7.0, dtype=torch.float16, device="xpu"),
        )
    )

    qsa_ops.qsa_store_cache_rows_v3(cache, slots, rows)
    torch.xpu.synchronize()

    assert torch.equal(cache[0, 2, 0], rows[1])


@pytest.mark.parametrize(
    ("dtype", "width"),
    [
        (torch.float16, 128),
        (torch.bfloat16, 128),
        (torch.int64, 3),
    ],
)
def test_qsa_store_cache_rows_v3_all_null_and_zero_rows(
    qsa_ops, dtype: torch.dtype, width: int
):
    cache, cache_backing = _make_cache(dtype, width)
    rows, _ = _make_rows(dtype, width, 3)
    before = cache_backing.clone()

    qsa_ops.qsa_store_cache_rows_v3(
        cache,
        torch.full((3,), -1, dtype=torch.int64, device="xpu"),
        rows,
    )
    empty_rows = torch.empty((0, width), dtype=dtype, device="xpu")
    result = qsa_ops.qsa_store_cache_rows_v3(
        cache,
        torch.empty((0,), dtype=torch.int64, device="xpu"),
        empty_rows,
    )
    torch.xpu.synchronize()

    assert result.data_ptr() == cache.data_ptr()
    assert torch.equal(cache_backing, before)


def test_qsa_store_cache_rows_v3_rejects_invalid_contracts(qsa_ops):
    cache = torch.zeros((1, 4, 1, 128), dtype=torch.float16, device="xpu")
    slots = torch.zeros((1,), dtype=torch.int64, device="xpu")
    rows = torch.zeros((1, 128), dtype=torch.float16, device="xpu")

    with pytest.raises(RuntimeError, match="same dtype"):
        qsa_ops.qsa_store_cache_rows_v3(cache, slots, rows.bfloat16())
    with pytest.raises(RuntimeError, match="supports"):
        qsa_ops.qsa_store_cache_rows_v3(
            cache[..., :127], slots, rows[:, :127]
        )
    with pytest.raises(RuntimeError, match="contiguous inner width"):
        strided_rows = torch.zeros(
            (1, 256), dtype=torch.float16, device="xpu"
        )[:, ::2]
        qsa_ops.qsa_store_cache_rows_v3(cache, slots, strided_rows)
    with pytest.raises(RuntimeError, match="contiguous int64"):
        qsa_ops.qsa_store_cache_rows_v3(cache, slots.int(), rows)
    with pytest.raises(RuntimeError, match="must be on XPU"):
        qsa_ops.qsa_store_cache_rows_v3(cache.cpu(), slots.cpu(), rows.cpu())


def test_qsa_store_cache_rows_v3_rejects_overlapping_layout_and_alias(qsa_ops):
    slots = torch.zeros((1,), dtype=torch.int64, device="xpu")
    overlapping_backing = torch.zeros(256, dtype=torch.float16, device="xpu")
    overlapping_cache = torch.as_strided(
        overlapping_backing, (1, 2, 1, 128), (128, 64, 128, 1)
    )
    rows = torch.ones((1, 128), dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="non-overlapping"):
        qsa_ops.qsa_store_cache_rows_v3(overlapping_cache, slots, rows)

    cache = torch.zeros((1, 2, 1, 128), dtype=torch.float16, device="xpu")
    aliased_rows = cache[0, 0]
    with pytest.raises(RuntimeError, match="must not alias"):
        qsa_ops.qsa_store_cache_rows_v3(cache, slots, aliased_rows)

    position_cache = torch.zeros((1, 2, 1, 3), dtype=torch.int64, device="xpu")
    aliased_slots = position_cache.flatten()[:1]
    position_rows = torch.ones((1, 3), dtype=torch.int64, device="xpu")
    with pytest.raises(RuntimeError, match="must not alias"):
        qsa_ops.qsa_store_cache_rows_v3(
            position_cache, aliased_slots, position_rows
        )
