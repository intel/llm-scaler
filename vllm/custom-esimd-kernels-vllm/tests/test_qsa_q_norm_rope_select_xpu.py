"""Correctness checks for the fused QSA Q norm, RoPE and selection ABI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="QSA fusion validation requires an XPU"
)


@pytest.fixture(scope="module")
def qsa_ops():
    package_dir = (
        Path(__file__).resolve().parents[1] / "python" / "custom_esimd_kernels_vllm"
    )
    candidates = (
        [Path(os.environ["QSA_TEST_DSO"])]
        if "QSA_TEST_DSO" in os.environ
        else sorted(package_dir.glob("qsa_ops*.so"))
    )
    if len(candidates) != 1:
        pytest.skip("focused QSA DSO is not built")
    spec = importlib.util.spec_from_file_location("qsa_ops", candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load QSA extension: {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_case(
    rows: int, mrope: bool, length: int = 9216, compressed_page_size: int = 64
):
    device = torch.device("xpu")
    generator = torch.Generator(device="cpu").manual_seed(7000 + rows)
    projected = torch.randn(rows, 4, 128, generator=generator, dtype=torch.float16).to(
        device
    )
    weight = (torch.randn(128, generator=generator) * 0.2).to(
        device=device, dtype=torch.float16
    )
    query_positions = torch.full((rows,), length - 2, dtype=torch.int64)
    if mrope:
        positions = torch.stack(
            (
                query_positions,
                query_positions + 1,
                query_positions + 2,
            )
        )
    else:
        positions = query_positions
    positions = positions.to(device)
    tokens_per_page = compressed_page_size * 4
    cache_cpu = torch.randn(
        (length + tokens_per_page - 1) // tokens_per_page,
        compressed_page_size,
        1,
        128,
        generator=generator,
        dtype=torch.float16,
    )
    cache = cache_cpu.to(device)
    page_table = (
        torch.arange(cache_cpu.shape[0], dtype=torch.int32).view(1, -1).to(device)
    )
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    sequence_lengths = torch.full((1,), length, dtype=torch.int32, device=device)
    cos_sin_cpu = torch.empty((length + 4, 64), dtype=torch.float16)
    positions_cpu = torch.arange(length + 4, dtype=torch.float32).view(-1, 1)
    pairs = torch.arange(32, dtype=torch.float32).view(1, -1)
    angles = (positions_cpu + 1.0) * (pairs + 1.0) * 0.0007
    cos_sin_cpu[:, :32] = torch.cos(angles).to(torch.float16)
    cos_sin_cpu[:, 32:] = torch.sin(angles).to(torch.float16)
    return (
        projected,
        weight,
        positions,
        cos_sin_cpu.to(device),
        cache,
        page_table,
        token_to_req,
        query_positions.to(device),
        sequence_lengths,
        mrope,
    )


@pytest.mark.parametrize("rows", [1, 16, 32])
@pytest.mark.parametrize("mrope", [False, True])
def test_qsa_fusion_matches_two_step_chain(qsa_ops, rows: int, mrope: bool):
    (
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        mrope,
    ) = _make_case(rows, mrope)
    chain_q = torch.empty_like(projected)
    chain_out = torch.empty((rows, 2051), dtype=torch.int32, device="xpu")
    qsa_ops.qsa_indexer_norm_rope_v1(
        projected,
        chain_q,
        weight,
        positions,
        cos_sin_cache,
        mrope,
        True,
    )
    qsa_ops.qsa_select_paged_tokens_v2(
        chain_q,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        2048,
        4,
        64,
        chain_out,
    )

    fused_q = torch.empty_like(projected)
    fused_out = torch.empty_like(chain_out)
    returned = qsa_ops.qsa_q_norm_rope_select_v1(
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        fused_q,
        fused_out,
        mrope,
        True,
    )
    torch.xpu.synchronize()

    assert returned.data_ptr() == fused_out.data_ptr()
    for fused_row, chain_row in zip(fused_out, chain_out):
        fused_tokens = torch.sort(fused_row[fused_row >= 0]).values
        chain_tokens = torch.sort(chain_row[chain_row >= 0]).values
        assert torch.equal(fused_tokens, chain_tokens)
    assert torch.isfinite(fused_q).all()
    assert torch.isfinite(chain_q).all()
    assert (fused_q.float() - chain_q.float()).abs().max() <= 2.5e-4

    selection_only_out = torch.empty_like(chain_out)
    selection_only = qsa_ops.qsa_q_norm_rope_select_v1(
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        torch.empty(0, dtype=torch.float16, device="xpu"),
        selection_only_out,
        mrope,
        True,
    )
    torch.xpu.synchronize()
    assert selection_only.data_ptr() == selection_only_out.data_ptr()
    assert torch.equal(selection_only_out, fused_out)


def test_qsa_fusion_rejects_unproven_positions(qsa_ops):
    (
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        mrope,
    ) = _make_case(1, False)
    with pytest.raises(RuntimeError, match="trusted position-bound proof"):
        qsa_ops.qsa_q_norm_rope_select_v1(
            projected,
            weight,
            positions,
            cos_sin_cache,
            compressed_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            torch.empty_like(projected),
            torch.empty((1, 2051), dtype=torch.int32, device="xpu"),
            mrope,
            False,
        )


@pytest.mark.parametrize("length", [4, 2050, 32770, 128002, 256002, 1000002])
@pytest.mark.parametrize("rows,mrope", [(1, False), (4, True)])
def test_parallel_selection_preserves_serial_order_and_query(
    qsa_ops, length, rows, mrope
):
    case = _make_case(rows, mrope, length)
    args, mrope = case[:-1], case[-1]
    old_q, new_q = torch.empty_like(args[0]), torch.empty_like(args[0])
    old = torch.empty(rows, 2051, dtype=torch.int32, device="xpu")
    new = torch.empty_like(old)
    qsa_ops.qsa_q_norm_rope_select_v1(*args, old_q, old, mrope, True)
    qsa_ops.qsa_q_norm_rope_select_parallel_v1(*args, new_q, new, mrope, True)
    torch.xpu.synchronize()
    assert torch.equal(new_q, old_q)
    assert torch.equal(new, old)


def test_parallel_selection_ties_invalid_pages_and_async_streams(qsa_ops):
    case = list(_make_case(4, True, 128004))
    case[4].zero_()  # Exact score ties spanning every partition.
    case[5][:, 2:4] = -1  # Invalid physical pages must not enter top-k.
    args, mrope = case[:-1], case[-1]
    qout = torch.empty_like(args[0])
    expected = torch.empty(4, 2051, dtype=torch.int32, device="xpu")
    qsa_ops.qsa_q_norm_rope_select_v1(*args, qout, expected, mrope, True)
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    outputs = [torch.empty_like(expected) for _ in streams]
    queries = [torch.empty_like(qout) for _ in streams]
    parent = torch.xpu.current_stream()
    for stream in streams:
        stream.wait_stream(parent)
    for _ in range(20):
        for stream, query, output in zip(streams, queries, outputs):
            with torch.xpu.stream(stream):
                qsa_ops.qsa_q_norm_rope_select_parallel_v1(
                    *args, query, output, mrope, True
                )
    for stream in streams:
        parent.wait_stream(stream)
    torch.xpu.synchronize()
    for query, output in zip(queries, outputs):
        assert torch.equal(query, qout)
        assert torch.equal(output, expected)


@pytest.mark.parametrize("compressed_page_size", [64, 128])
@pytest.mark.parametrize("length", [4, 512, 1024, 1536, 2050, 128002])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 32])
@pytest.mark.parametrize(
    "operation",
    ["qsa_select_paged_tokens_parallel_v1", "qsa_select_paged_tokens_local_v1"],
)
def test_preprocessed_parallel_selection_is_bitwise(
    qsa_ops, compressed_page_size, length, rows, operation
):
    case = _make_case(rows, False, length, compressed_page_size)
    q, _, _, _, cache, table, requests, positions, lengths, _ = case
    old = torch.empty(rows, 2051, dtype=torch.int32, device="xpu")
    new = torch.empty_like(old)
    inputs = (
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        2048,
        4,
        compressed_page_size,
    )
    qsa_ops.qsa_select_paged_tokens_v2(*inputs, old)
    getattr(qsa_ops, operation)(*inputs, new)
    torch.xpu.synchronize()
    assert torch.equal(new, old)


@pytest.mark.parametrize("compressed_page_size", [64, 128])
@pytest.mark.parametrize(
    "operation",
    ["qsa_select_paged_tokens_parallel_v1", "qsa_select_paged_tokens_local_v1"],
)
def test_preprocessed_parallel_selection_async_ties_and_alias_guard(
    qsa_ops, compressed_page_size, operation
):
    case = _make_case(4, False, 128004, compressed_page_size)
    q, _, _, _, cache, table, requests, positions, lengths, _ = case
    cache.zero_()
    table[:, 2:4] = -1
    inputs = (
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        2048,
        4,
        compressed_page_size,
    )
    expected = torch.empty(4, 2051, dtype=torch.int32, device="xpu")
    qsa_ops.qsa_select_paged_tokens_v2(*inputs, expected)
    outputs = [torch.empty_like(expected), torch.empty_like(expected)]
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    parent = torch.xpu.current_stream()
    for stream in streams:
        stream.wait_stream(parent)
    for _ in range(20):
        for stream, output in zip(streams, outputs):
            with torch.xpu.stream(stream):
                getattr(qsa_ops, operation)(*inputs, output)
    for stream in streams:
        parent.wait_stream(stream)
    torch.xpu.synchronize()
    assert all(torch.equal(output, expected) for output in outputs)
    alias = cache.view(torch.int32).flatten()[:4 * 2051].view(4, 2051)
    with pytest.raises(RuntimeError, match="must not alias"):
        getattr(qsa_ops, operation)(*inputs, alias)


@pytest.mark.parametrize("compressed_page_size", [64, 128])
def test_preprocessed_parallel_selection_mixed_requests_and_visible_lengths(
    qsa_ops, compressed_page_size
):
    q, _, _, _, cache, table, _, _, _, _ = _make_case(
        8, False, 128004, compressed_page_size
    )
    table = table.flip(1).repeat(3, 1).contiguous()
    table[:, 7] = -1
    table[:, 22] = cache.shape[0]  # An out-of-range physical page is invalid.
    requests = torch.tensor(
        [-1, 0, 1, 2, 3, 0, 1, 2], dtype=torch.int32, device="xpu"
    )
    positions = torch.tensor(
        [-1, 0, 2047, 4098, 32767, 128003, 999999, 5],
        dtype=torch.int64, device="xpu",
    )
    lengths = torch.tensor([128004, 32770, 4099], dtype=torch.int32, device="xpu")
    inputs = (
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        2048,
        4,
        compressed_page_size,
    )
    old = torch.empty(8, 2051, dtype=torch.int32, device="xpu")
    new = torch.empty_like(old)
    qsa_ops.qsa_select_paged_tokens_v2(*inputs, old)
    qsa_ops.qsa_select_paged_tokens_parallel_v1(*inputs, new)
    torch.xpu.synchronize()
    assert torch.equal(new, old)


@pytest.mark.parametrize("compressed_page_size", [64, 128])
def test_preprocessed_parallel_selection_exact_page_boundaries(
    qsa_ops, compressed_page_size
):
    length = compressed_page_size * 8 + 8
    q, _, _, _, cache, table, requests, _, lengths, _ = _make_case(
        4, False, length, compressed_page_size
    )
    positions = torch.tensor(
        [
            compressed_page_size * 4 - 1,
            compressed_page_size * 4 + 3,
            compressed_page_size * 8 - 1,
            compressed_page_size * 8 + 3,
        ],
        dtype=torch.int64,
        device="xpu",
    )
    inputs = (
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        2048,
        4,
        compressed_page_size,
    )
    old = torch.empty(4, 2051, dtype=torch.int32, device="xpu")
    new = torch.empty_like(old)
    qsa_ops.qsa_select_paged_tokens_v2(*inputs, old)
    qsa_ops.qsa_select_paged_tokens_parallel_v1(*inputs, new)
    torch.xpu.synchronize()
    assert torch.equal(new, old)
