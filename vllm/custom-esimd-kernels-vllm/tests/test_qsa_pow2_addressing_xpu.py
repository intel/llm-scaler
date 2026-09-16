"""Focused pow2/non-pow2 addressing parity for QSA compression and stores."""

from __future__ import annotations

import pytest
import torch

from test_qsa_m1_transaction_xpu import case as make_store_case
from test_qsa_sparse_attention_xpu import _load_qsa_extension, _xpu_available


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA validation requires an XPU"
)


@pytest.fixture(scope="module")
def ops():
    return _load_qsa_extension()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("ring_size", "page_size"), ((4, 128), (4, 7), (12, 128))
)
@pytest.mark.parametrize("slot", [-1, 0, 7, 8, 11, 12, 14, 256])
def test_store_transaction_matches_legacy_at_division_boundaries(
    ops, dtype, ring_size, page_size, slot
):
    stores, backing = make_store_case(
        dtype,
        slot=slot,
        count=3,
        page_size=page_size,
        ring_size=ring_size,
    )
    for cache, slots, rows in stores:
        ops.qsa_store_cache_rows_v3(cache, slots, rows)
    golden = [tensor.clone() for tensor in backing]
    for tensor in backing:
        tensor.zero_()
    assert ops.try_store_m1_transaction_v1(stores, True) is True
    torch.xpu.synchronize()
    for actual, expected in zip(backing, golden):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def make_compression_case(dtype, ring_size, rows, valid, position_offset=0):
    pages = 2
    storage = torch.empty((pages, ring_size, 1, 140), dtype=dtype, device="xpu")
    cache = storage[..., :128]
    positions = storage[..., 128:].view(torch.int64)
    for page in range(pages):
        for slot in range(ring_size):
            cache[page, slot, 0].fill_(page * 100 + slot)
            positions[page, slot, 0] = torch.tensor(
                [page * 1000 + slot, page * 1000 + slot + 10, page * 1000 + slot + 20],
                dtype=torch.int64,
                device="xpu",
            )
    if rows == 1:
        raw = (torch.arange(128, dtype=torch.float32, device="xpu") / 17).to(dtype)
        raw = raw.reshape(1, 1, 128)
        logical_positions = torch.tensor([11], dtype=torch.int64, device="xpu")
        token_to_req = torch.tensor([0], dtype=torch.int32, device="xpu")
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
        block_table = torch.tensor([[1]], dtype=torch.int32, device="xpu")
    else:
        raw = torch.empty((4, 1, 128), dtype=dtype, device="xpu")
        for row in range(4):
            raw[row, 0].fill_(10 + row)
        logical_positions = torch.tensor(
            [10, 11, 20, 21], dtype=torch.int64, device="xpu"
        )
        token_to_req = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="xpu")
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32, device="xpu")
        block_table = torch.tensor([[0], [1]], dtype=torch.int32, device="xpu")
    logical_positions += position_offset
    raw_positions = torch.stack(
        (logical_positions, logical_positions + 100, logical_positions + 200), dim=1
    ).unsqueeze(1)
    compressed_slots = torch.arange(rows, dtype=torch.int64, device="xpu")
    if not valid:
        compressed_slots.fill_(-1)
    pooled = torch.full((rows, 1, 128), -99, dtype=dtype, device="xpu")
    first_positions = torch.full((rows, 3), -99, dtype=torch.int64, device="xpu")
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
@pytest.mark.parametrize("ring_size", [4, 12])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("position_offset", [0, 32768, 128000, 2**31])
def test_compression_pow2_and_nonpow2_match_independent_reference(
    ops, dtype, ring_size, rows, valid, position_offset
):
    case = make_compression_case(dtype, ring_size, rows, valid, position_offset)
    ops.qsa_group_compress_v2(*case, 4, 256, True)
    torch.xpu.synchronize()
    if not valid:
        assert torch.count_nonzero(case[9]) == 0
        assert torch.count_nonzero(case[10]) == 0
        return

    expected_rows = []
    expected_positions = []
    for row in range(rows):
        request = int(case[5][row].item())
        query_start = int(case[6][request].item())
        local_row = row - query_start
        end = int(case[7][row].item())
        chunk_start = end - local_row
        block = int(case[4][request, 0].item())
        values = []
        for position in range(end - 3, end + 1):
            if position >= chunk_start:
                raw_row = query_start + position - chunk_start
                values.append(case[0][raw_row, 0].float())
            else:
                values.append(case[2][block, position % ring_size, 0].float())
        expected_rows.append(sum(values) / 4)
        first = end - 3
        if first >= chunk_start:
            raw_row = query_start + first - chunk_start
            expected_positions.append(case[1][raw_row, 0])
        else:
            expected_positions.append(case[3][block, first % ring_size, 0])
    expected = torch.stack(expected_rows).reshape(rows, 1, 128).to(dtype)
    torch.testing.assert_close(case[9], expected, atol=0, rtol=0)
    assert torch.equal(case[10], torch.stack(expected_positions))
