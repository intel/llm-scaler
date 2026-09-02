"""Correctness checks for fused Qwen3.8 QSA index selection on XPU."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch


INDEX_HEADS = 4
HEAD_DIM = 128
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1


def _xpu_available() -> bool:
    try:
        return torch.xpu.is_available() and torch.xpu.device_count() > 0
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA selection validation requires an XPU"
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


def _expand_reference(
    blocks: torch.Tensor,
    query_positions: torch.Tensor,
    row_lengths: torch.Tensor,
) -> torch.Tensor:
    rows = blocks.shape[0]
    offsets = torch.arange(COMPRESS_RATIO, device=blocks.device)
    expanded = blocks.long().unsqueeze(-1) * COMPRESS_RATIO + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    ).reshape(rows, TOKEN_TOPK)
    expanded = torch.where(
        (expanded >= 0) & (expanded < row_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(COMPRESS_RATIO - 1, device=blocks.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // COMPRESS_RATIO * COMPRESS_RATIO
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (
        tail_offsets.unsqueeze(0)
        < (visible_tokens - tail_start).unsqueeze(1)
    ) & (tail < row_lengths.unsqueeze(1))
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(OUTPUT_WIDTH, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + OUTPUT_WIDTH)
    return result.gather(
        1, torch.argsort(sort_key, dim=1, stable=True)
    ).to(torch.int32)


def _select_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    rows = q.shape[0]
    selected = torch.full(
        (rows, TOKEN_TOPK // COMPRESS_RATIO),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    row_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.minimum(
        (query_positions + 1) // COMPRESS_RATIO,
        row_lengths.long() // COMPRESS_RATIO,
    )
    for row in range(rows):
        visible = int(visible_blocks[row].item())
        width = min(visible, TOKEN_TOPK // COMPRESS_RATIO)
        if not width:
            continue
        logical = torch.arange(visible, device=q.device)
        request = int(token_to_req[row].item())
        pages = page_table[request, logical // page_size].long()
        keys = cache[pages, logical % page_size, 0]
        scores = torch.relu(
            torch.einsum("hd,nd->nh", q[row].float(), keys.float())
        ).sum(dim=-1) * (HEAD_DIM**-0.5)
        selected[row, :width] = torch.topk(scores, width).indices.to(
            torch.int32
        )
    return _expand_reference(selected, query_positions, row_lengths.long())


def _case_metadata(case: str):
    if case == "empty_sequence":
        return [-1], [0], [0], False
    if case == "short_tail":
        return [2], [3], [0], False
    if case == "full_page":
        return [511], [512], [0], False
    if case == "cross_page_tail":
        return [518], [520], [0], False
    if case == "p8192_saturated_2051":
        return [9214], [9216], [0], False
    if case == "exact_score_ties":
        return [2079], [2080], [0], True
    if case == "duplicate_physical_pages":
        return [1027], [1028], [0], False
    if case == "multi_request":
        return [2, 518, 8192, 9214], [520, 9216], [0, 0, 1, 1], False
    raise ValueError(case)


def _make_inputs(case: str, page_size: int):
    positions, sequence_lengths, requests, zero_query = _case_metadata(case)
    rows = len(positions)
    num_requests = len(sequence_lengths)
    page_columns = max(
        1,
        max(length // COMPRESS_RATIO for length in sequence_lengths)
        // page_size
        + 1,
    )
    physical_pages = num_requests * page_columns
    page_table = torch.arange(
        physical_pages, dtype=torch.int32, device="xpu"
    ).view(num_requests, page_columns)
    if num_requests > 1:
        page_table[1] = page_table[1].flip(0)
    if case == "duplicate_physical_pages":
        page_table[:, 1::2] = page_table[:, :1]
    cache = torch.randn(
        physical_pages,
        page_size,
        1,
        HEAD_DIM,
        dtype=torch.float16,
        device="xpu",
    )
    q = torch.randn(
        rows,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device="xpu",
    )
    if zero_query:
        q.zero_()
    token_to_req = torch.tensor(requests, dtype=torch.int32, device="xpu")
    query_positions = torch.tensor(
        positions, dtype=torch.int64, device="xpu"
    )
    seq_lens = torch.tensor(
        sequence_lengths, dtype=torch.int32, device="xpu"
    )
    out = torch.empty(
        rows, OUTPUT_WIDTH, dtype=torch.int32, device="xpu"
    )
    return (
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        seq_lens,
        TOKEN_TOPK,
        COMPRESS_RATIO,
        page_size,
        out,
    )


@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize(
    "case",
    [
        "empty_sequence",
        "short_tail",
        "full_page",
        "cross_page_tail",
        "p8192_saturated_2051",
        "exact_score_ties",
        "duplicate_physical_pages",
        "multi_request",
    ],
)
def test_qsa_select_matches_reference(qsa_ops, case, page_size):
    torch.manual_seed(20260828)
    args = _make_inputs(case, page_size)
    tensor_inputs = [value for value in args if isinstance(value, torch.Tensor)]
    snapshots = [value.clone() for value in tensor_inputs[:-1]]
    expected = _select_reference(*args[:6], page_size)

    returned = qsa_ops.qsa_select_paged_tokens_v2(*args)
    torch.xpu.synchronize()

    assert returned.data_ptr() == args[-1].data_ptr()
    assert returned.dtype == torch.int32
    assert returned.shape == (args[0].shape[0], OUTPUT_WIDTH)
    assert torch.equal(returned, expected)
    for actual, snapshot in zip(tensor_inputs[:-1], snapshots):
        assert torch.equal(actual, snapshot)
    for row in returned:
        valid = row >= 0
        valid_count = int(valid.sum().item())
        assert torch.all(valid[:valid_count])
        assert not torch.any(valid[valid_count:])


def test_qsa_select_rejects_bf16(qsa_ops):
    args = list(_make_inputs("short_tail", 64))
    args[0] = args[0].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="q must be float16"):
        qsa_ops.qsa_select_paged_tokens_v2(*args)


def test_qsa_select_rejects_wrong_output_width(qsa_ops):
    args = list(_make_inputs("short_tail", 64))
    args[-1] = torch.empty(1, 2048, dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="out must have shape"):
        qsa_ops.qsa_select_paged_tokens_v2(*args)
