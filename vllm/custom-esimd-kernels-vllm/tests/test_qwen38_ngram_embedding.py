"""Correctness and safety tests for Qwen3.8 NG-2a local gather."""

import random

import torch
import torch.nn.functional as F

from custom_esimd_kernels_vllm import (
    esimd_qwen38_ngram_embedding_gather,
    esimd_qwen38_ngram_embedding_gather_out,
)


ROW_WIDTH = 160


def require_xpu():
    if not torch.xpu.is_available():
        raise RuntimeError("PyTorch XPU is unavailable")
    return torch.device("xpu:0")


def make_inputs(device, rows, start, seed):
    row = torch.arange(rows, dtype=torch.int64).view(-1, 1)
    col = torch.arange(ROW_WIDTH, dtype=torch.int64).view(1, -1)
    weight = (((row * (17 + seed) + col * (29 + seed) + 7) % 2049 - 1024)
              .float() / 64).half().to(device).contiguous()
    ids = torch.tensor(
        [
            start - 1,
            start,
            start + min(1, rows - 1),
            start + rows - 1,
            start + rows,
            start + rows // 2,
            start + rows // 2,
            start + min(3, rows - 1),
            start - 2,
            start + min(7, rows - 1),
            start + min(7, rows - 1),
            start + min(11, rows - 1),
            start + rows + 1,
            start + min(rows - 1, 17),
            start - 3,
            start + min(rows - 1, 31),
        ],
        dtype=torch.int64,
        device=device,
    ).reshape(1, 16)
    start_tensor = torch.tensor([start], dtype=torch.int64, device=device)
    rows_tensor = torch.tensor([rows], dtype=torch.int64, device=device)
    return ids, weight, start_tensor, rows_tensor


def reference(inputs):
    ids, weight, start, rows = inputs
    valid = (ids >= start) & (ids < start + rows)
    local_ids = torch.where(valid, ids - start, torch.zeros_like(ids))
    gathered = F.embedding(local_ids, weight)
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)).reshape(1, 2560)


def assert_exact(inputs):
    actual = esimd_qwen38_ngram_embedding_gather(*inputs)
    expected = reference(inputs)
    torch.xpu.synchronize()
    assert actual.shape == (1, 2560)
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_row_count_and_shard_start_generalization():
    device = require_xpu()
    for seed, rows in enumerate((1, 2, 63, 64, 65, 257, 4096)):
        for start in (0, 240_001_152, 280_001_344, 1 << 40):
            assert_exact(make_inputs(device, rows, start, seed))


def test_random_fp16_rows_and_ids():
    device = require_xpu()
    rng = random.Random(0x3802)
    for index in range(128):
        rows = rng.choice((1, 3, 17, 64, 65, 257, 1024))
        start = rng.choice((0, 7, 240_001_152, 280_001_344, 1 << 40))
        inputs = make_inputs(device, rows, start, index + 100)
        inputs = (
            torch.tensor(
                [rng.randrange(start - 2, start + rows + 2) for _ in range(16)],
                dtype=torch.int64,
                device=device,
            ).reshape(1, 16),
            inputs[1],
            inputs[2],
            inputs[3],
        )
        assert_exact(inputs)


def test_preallocated_output_and_raw_void_schema():
    device = require_xpu()
    inputs = make_inputs(device, 65, 987_654_321, 41)
    expected = reference(inputs)
    output = torch.empty((1, 2560), dtype=torch.float16, device=device)
    raw_return = torch.ops.custom_esimd_kernels_vllm.esimd_qwen38_ngram_embedding_gather_out(
        *inputs, output)
    assert raw_return is None
    returned = esimd_qwen38_ngram_embedding_gather_out(*inputs, output)
    assert returned.data_ptr() == output.data_ptr()
    torch.xpu.synchronize()
    assert torch.equal(output, expected)


def test_non_default_stream_and_reused_output():
    device = require_xpu()
    stream = torch.xpu.Stream(device=device)
    output = torch.empty((1, 2560), dtype=torch.float16, device=device)
    expected = None
    with torch.xpu.stream(stream):
        for index in range(64):
            inputs = make_inputs(device, 65, 10_000 + index * 17, index)
            expected = reference(inputs)
            esimd_qwen38_ngram_embedding_gather_out(*inputs, output)
            del inputs
            torch.empty((4096,), dtype=torch.float16, device=device).fill_(index)
    stream.synchronize()
    assert torch.equal(output, expected)


def test_invalid_inputs_rejected():
    device = require_xpu()
    inputs = list(make_inputs(device, 64, 280_001_344, 0))

    bad = inputs.copy()
    bad[0] = bad[0].to(torch.int32)
    try:
        esimd_qwen38_ngram_embedding_gather(*bad)
    except RuntimeError as exc:
        assert "int64" in str(exc)
    else:
        raise AssertionError("wrong ID dtype accepted")

    bad = inputs.copy()
    bad[1] = bad[1].to(torch.float32)
    try:
        esimd_qwen38_ngram_embedding_gather(*bad)
    except RuntimeError as exc:
        assert "float16" in str(exc)
    else:
        raise AssertionError("wrong weight dtype accepted")

    bad = inputs.copy()
    bad[1] = bad[1].t()
    try:
        esimd_qwen38_ngram_embedding_gather(*bad)
    except RuntimeError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("noncontiguous weight accepted")

    bad = inputs.copy()
    bad[0] = bad[0].reshape(16)
    try:
        esimd_qwen38_ngram_embedding_gather(*bad)
    except RuntimeError as exc:
        assert "shape [1, 16]" in str(exc)
    else:
        raise AssertionError("wrong ID shape accepted")

    bad = inputs.copy()
    bad[2] = bad[2].reshape(1, 1)
    try:
        esimd_qwen38_ngram_embedding_gather(*bad)
    except RuntimeError as exc:
        assert "shape [1]" in str(exc)
    else:
        raise AssertionError("wrong metadata shape accepted")

    bad_output = torch.empty((1, 160), dtype=torch.float16, device=device)
    try:
        esimd_qwen38_ngram_embedding_gather_out(*inputs, bad_output)
    except RuntimeError as exc:
        assert "local_partial" in str(exc)
    else:
        raise AssertionError("wrong output shape accepted")


def test_cross_device_rejected_when_available():
    if not torch.xpu.is_available() or torch.xpu.device_count() < 2:
        return
    inputs = list(make_inputs(torch.device("xpu:0"), 64, 280_001_344, 0))
    inputs[1] = inputs[1].to("xpu:1")
    try:
        esimd_qwen38_ngram_embedding_gather(*inputs)
    except RuntimeError as exc:
        assert "same XPU device" in str(exc)
    else:
        raise AssertionError("cross-device table accepted")
