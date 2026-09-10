"""Pinned-host row lookup: independent bitwise oracle and stream ownership."""

import gc
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


@pytest.fixture(scope="module")
def operation():
    path = os.environ.get("NGRAM_OFFLOAD_TEST_DSO")
    if not path:
        pytest.skip("set NGRAM_OFFLOAD_TEST_DSO to the build-only artifact")
    assert Path(path).is_absolute() and Path(path).is_file()
    torch.ops.load_library(path)
    return torch.ops.ngram_offload_ops.lookup_out_v1


def inputs(rows, dtype=torch.float16, start=71):
    # Nontrivial bits including signed zero, finite subnormals and infinities.
    bits = torch.arange(257 * 160, dtype=torch.int32).remainder(65536).to(torch.uint16)
    table = torch.empty((257, 160), dtype=dtype, pin_memory=True)
    table.view(torch.uint16).copy_(bits.view(257, 160))
    ids = torch.arange(rows * 16, dtype=torch.int64).reshape(rows, 16) % 260 + start - 1
    if rows:
        ids[0, :4] = torch.tensor([-1, 2**40, start, start + 256])
    return table, ids.to("xpu"), torch.empty((rows, 2560), dtype=dtype, device="xpu")


def golden(table, ids, start):
    ids = ids.cpu()
    valid = (ids >= start) & (ids < start + table.shape[0])
    local = (ids - start).clamp(0, table.shape[0] - 1)
    values = table[local].view(torch.int16)
    return torch.where(valid[..., None], values, 0).reshape(ids.shape[0], 2560)


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_host_lookup_bitwise(operation, rows, dtype):
    table, ids, out = inputs(rows, dtype)
    expected = golden(table, ids, 71)
    before = torch.xpu.memory_allocated()
    returned = operation(table, ids, out, 71, 328)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.xpu.memory_allocated() == before
    assert torch.equal(out.cpu().view(torch.int16), expected)


@pytest.mark.parametrize("tp", [4, 8])
def test_tp4_tp8_shards_sum_to_same_full_table(operation, tp):
    full = (
        torch.arange(1024 * 160, dtype=torch.float32)
        .remainder(1024)
        .half()
        .view(1024, 160)
    )
    ids = torch.tensor(
        [[0, 127, 128, 255, 256, 511, 512, 767, 768, 1023, 1, 2, 3, 4, 5, 6]],
        device="xpu",
    )
    outputs = []
    for rank in range(tp):
        start = rank * (1024 // tp)
        table = full[start : start + 1024 // tp].pin_memory()
        out = torch.empty((1, 2560), dtype=torch.float16, device="xpu")
        operation(table, ids, out, start, start + 1024 // tp)
        outputs.append(out)
    reduced = torch.stack(outputs).sum(0)
    assert torch.equal(reduced.cpu(), full[ids.cpu()].reshape(1, 2560))


def test_two_streams_and_short_lived_table(operation):
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    saved = []
    for iteration in range(12):
        stream = streams[iteration % 2]
        with torch.xpu.stream(stream):
            table, ids, out = inputs(1 + iteration % 8)
            table.add_(iteration)
            expected = golden(table, ids, 71)
            operation(table, ids, out, 71, 328)
            saved.append((out, expected))
            del table, ids
        gc.collect()
    torch.xpu.synchronize()
    for out, expected in saved:
        assert torch.equal(out.cpu().view(torch.int16), expected)


def test_pinned_allocation_not_recycled_while_lookup_pending(operation):
    # Deliberately leave a queue backlog: a tiny lookup alone may finish before
    # the next CPU allocation, masking a missing allocator record_event.
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        table, ids, out = inputs(8)
        expected = golden(table, ids, 71)
        original_pointer = table.data_ptr()
        a = torch.ones((4096, 4096), dtype=torch.float16, device="xpu")
        b = torch.empty_like(a)
        stream.synchronize()
        for _ in range(128):
            torch.mm(a, a, out=b)
        operation(table, ids, out, 71, 328)
        finished = torch.xpu.Event()
        finished.record()
        assert not finished.query(), "Need an outstanding lookup for this lifetime test"
        del table
        replacement = torch.empty((257, 160), dtype=torch.float16, pin_memory=True)
        replacement.fill_(42)
        if not finished.query():
            assert replacement.data_ptr() != original_pointer
    finished.synchronize()
    assert torch.equal(out.cpu().view(torch.int16), expected)


@pytest.mark.parametrize(
    "case", ["not_pinned", "wrong_dtype", "shape", "stride", "bounds", "lazy", "alias"]
)
def test_invalid_contract_rejected_before_submit(operation, case):
    table, ids, out = inputs(2)
    start, end = 71, 328
    if case == "not_pinned":
        table = table.clone()
    elif case == "wrong_dtype":
        ids = ids.int()
    elif case == "shape":
        out = out[:, :2559].contiguous()
    elif case == "stride":
        ids = ids.T.contiguous().T
    elif case == "bounds":
        end += 1
    elif case == "lazy":
        table = torch._neg_view(table)
    elif case == "alias":
        ids = out.view(torch.int64).flatten()[:32].view(2, 16)
    with pytest.raises(RuntimeError):
        operation(table, ids, out, start, end)


@pytest.fixture(scope="module")
def chunked_operation(operation):
    return torch.ops.ngram_offload_ops.lookup_chunked_out_v1


def chunked_inputs(rows, dtype=torch.float16, start=101, num_chunks=2):
    chunk_sizes = [23] * (num_chunks - 1) + [11]
    chunks = []
    bit_offset = 0
    for size in chunk_sizes:
        bits = (
            torch.arange(bit_offset, bit_offset + size * 160, dtype=torch.int32)
            .mul_(40503)
            .add_(37)
            .remainder_(65536)
            .to(torch.uint16)
        )
        chunk = torch.empty((size, 160), dtype=dtype, pin_memory=True)
        chunk.view(torch.uint16).copy_(bits.view(size, 160))
        chunks.append(chunk)
        bit_offset += size * 160

    total_rows = sum(chunk_sizes)
    generator = torch.Generator().manual_seed(1200 + rows + num_chunks)
    ids = torch.randint(
        start - 3,
        start + total_rows + 3,
        (rows, 16),
        dtype=torch.int64,
        generator=generator,
    )
    boundaries = [start - 1, start]
    for bank in range(1, num_chunks):
        boundaries.extend(
            [start + bank * chunk_sizes[0] - 1, start + bank * chunk_sizes[0]]
        )
    boundaries.extend([start + total_rows - 1, start + total_rows])
    if rows:
        count = min(ids.numel(), len(boundaries))
        ids.view(-1)[:count] = torch.tensor(boundaries[:count])
    output = torch.empty((rows, 2560), dtype=dtype, device="xpu")
    return tuple(chunks), ids.to("xpu"), output, start, start + total_rows


def chunked_golden(chunks, ids, start, end):
    table = torch.cat(chunks)
    host_ids = ids.cpu()
    valid = (host_ids >= start) & (host_ids < end)
    local = (host_ids - start).clamp(0, table.shape[0] - 1)
    values = table[local].view(torch.int16)
    return torch.where(valid[..., None], values, 0).reshape(host_ids.shape[0], 2560)


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_lookup_bitwise_bank_boundaries(chunked_operation, rows, dtype):
    num_chunks = max(1, min(rows, 8))
    chunks, ids, output, start, end = chunked_inputs(rows, dtype, num_chunks=num_chunks)
    expected = chunked_golden(chunks, ids, start, end)
    before = torch.xpu.memory_allocated()
    returned = chunked_operation(chunks, ids, output, start, end)
    assert returned.data_ptr() == output.data_ptr()
    assert torch.xpu.memory_allocated() == before
    assert torch.equal(output.cpu().view(torch.int16), expected)


@pytest.mark.parametrize("num_chunks", range(1, 9))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_lookup_all_bank_counts(chunked_operation, num_chunks, dtype):
    chunks, ids, output, start, end = chunked_inputs(2, dtype, num_chunks=num_chunks)
    expected = chunked_golden(chunks, ids, start, end)
    chunked_operation(chunks, ids, output, start, end)
    assert torch.equal(output.cpu().view(torch.int16), expected)


@pytest.mark.parametrize("tp", [4, 8])
def test_chunked_tp4_tp8_shards_sum_to_same_full_table(chunked_operation, tp):
    full = torch.arange(2048 * 160, dtype=torch.float32).remainder(1024).half()
    full = full.view(2048, 160)
    ids = torch.tensor(
        [[0, 255, 256, 511, 512, 1023, 1024, 1535, 1536, 2047, 1, 2, 3, 4, 5, 6]],
        device="xpu",
    )
    outputs = []
    shard_rows = 2048 // tp
    for rank in range(tp):
        start = rank * shard_rows
        split_rows = (shard_rows + 2) // 3
        sizes = [split_rows, split_rows, shard_rows - 2 * split_rows]
        offset = 0
        chunks = []
        for size in sizes:
            chunks.append(full[start + offset : start + offset + size].pin_memory())
            offset += size
        output = torch.empty((1, 2560), dtype=torch.float16, device="xpu")
        chunked_operation(chunks, ids, output, start, start + shard_rows)
        outputs.append(output)
    reduced = torch.stack(outputs).sum(0)
    assert torch.equal(reduced.cpu(), full[ids.cpu()].reshape(1, 2560))


def test_chunked_two_streams_and_short_lived_banks(chunked_operation):
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    saved = []
    for iteration in range(12):
        stream = streams[iteration % 2]
        with torch.xpu.stream(stream):
            chunks, ids, output, start, end = chunked_inputs(
                1 + iteration % 8, num_chunks=1 + iteration % 8
            )
            for chunk in chunks:
                chunk.add_(iteration)
            expected = chunked_golden(chunks, ids, start, end)
            chunked_operation(chunks, ids, output, start, end)
            saved.append((output, expected))
            del chunks, ids
        gc.collect()
    torch.xpu.synchronize()
    for output, expected in saved:
        assert torch.equal(output.cpu().view(torch.int16), expected)


def test_chunked_each_bank_lifetime_is_recorded(chunked_operation):
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        chunks, ids, output, start, end = chunked_inputs(8, num_chunks=8)
        expected = chunked_golden(chunks, ids, start, end)
        pointers = {chunk.data_ptr() for chunk in chunks}
        sizes = [chunk.shape for chunk in chunks]
        a = torch.ones((4096, 4096), dtype=torch.float16, device="xpu")
        b = torch.empty_like(a)
        stream.synchronize()
        for _ in range(128):
            torch.mm(a, a, out=b)
        chunked_operation(chunks, ids, output, start, end)
        finished = torch.xpu.Event()
        finished.record()
        assert not finished.query(), "Need an outstanding lookup for this lifetime test"
        del chunks
        replacements = [
            torch.empty(shape, dtype=torch.float16, pin_memory=True) for shape in sizes
        ]
        if not finished.query():
            assert pointers.isdisjoint({item.data_ptr() for item in replacements})
    finished.synchronize()
    assert torch.equal(output.cpu().view(torch.int16), expected)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "too_many",
        "scalar",
        "not_pinned",
        "weight_device",
        "weight_dtype",
        "weight_shape",
        "weight_stride",
        "full_rows",
        "last_rows",
        "empty_last",
        "bounds",
        "ids_dtype",
        "ids_shape",
        "output_device",
        "output_shape",
        "lazy_negative",
        "lazy_conjugate",
        "alias",
    ],
)
def test_chunked_invalid_contract_rejected_before_submit(chunked_operation, case):
    chunks, ids, output, start, end = chunked_inputs(2, num_chunks=3)
    chunks = list(chunks)
    if case == "empty":
        chunks = []
    elif case == "too_many":
        chunks = chunks * 3
    elif case == "scalar":
        chunks[0] = torch.tensor(1, dtype=torch.float16, pin_memory=True)
    elif case == "not_pinned":
        chunks[0] = chunks[0].clone()
    elif case == "weight_device":
        chunks[0] = chunks[0].to("xpu")
    elif case == "weight_dtype":
        chunks[1] = chunks[1].bfloat16().pin_memory()
    elif case == "weight_shape":
        chunks[1] = torch.empty((23, 159), dtype=torch.float16, pin_memory=True)
    elif case == "weight_stride":
        chunks[1] = torch.empty((23, 320), pin_memory=True)[:, ::2]
    elif case == "full_rows":
        chunks[1] = torch.empty((22, 160), dtype=torch.float16, pin_memory=True)
    elif case == "last_rows":
        chunks[-1] = torch.empty((24, 160), dtype=torch.float16, pin_memory=True)
    elif case == "empty_last":
        chunks[-1] = torch.empty((0, 160), dtype=torch.float16, pin_memory=True)
    elif case == "bounds":
        end += 1
    elif case == "ids_dtype":
        ids = ids.int()
    elif case == "ids_shape":
        ids = ids[:, :15].contiguous()
    elif case == "output_device":
        output = output.cpu()
    elif case == "output_shape":
        output = output[:, :2559].contiguous()
    elif case == "lazy_negative":
        chunks[1] = torch._neg_view(chunks[1])
    elif case == "lazy_conjugate":
        chunks[1] = torch._conj(
            torch.ones(chunks[1].shape, dtype=torch.complex64, pin_memory=True)
        )
    elif case == "alias":
        ids = output.view(torch.int64).flatten()[:32].view(2, 16)
    with pytest.raises(RuntimeError):
        chunked_operation(tuple(chunks), ids, output, start, end)
