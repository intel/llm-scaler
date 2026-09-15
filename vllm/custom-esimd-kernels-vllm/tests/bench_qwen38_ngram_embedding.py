#!/usr/bin/env python3
"""Benchmark NG-2a gather with one true-size TP-rank FP16 table."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from custom_esimd_kernels_vllm import (
    esimd_qwen38_ngram_embedding_gather,
    esimd_qwen38_ngram_embedding_gather_out,
)


ROW_WIDTH = 160
LOCAL_START = 280_001_344
LOCAL_ROWS = 40_000_192
TABLE_BYTES = LOCAL_ROWS * ROW_WIDTH * 2
VRAM_MARGIN = 8 * 1024**3
POOL_SIZE = 65_536
SEED = 3802
VOCAB_OFFSETS = (
    0, 20_000_003, 40_000_026, 60_000_059,
    80_000_106, 100_000_165, 120_000_228, 140_000_297,
    160_000_374, 180_000_455, 200_000_548, 220_000_655,
    240_000_802, 260_000_955, 280_001_114, 300_001_275,
)
VOCAB_SIZES = (
    20_000_003, 20_000_023, 20_000_033, 20_000_047,
    20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147,
    20_000_153, 20_000_159, 20_000_161, 20_000_171,
)


def make_id_pool() -> torch.Tensor:
    i = torch.arange(POOL_SIZE, dtype=torch.int64)
    pool = torch.empty((POOL_SIZE, 16), dtype=torch.int64)
    for head, (offset, size) in enumerate(zip(VOCAB_OFFSETS, VOCAB_SIZES)):
        if head == 14:
            base = LOCAL_START
            span = offset + size - base
        elif head == 15:
            base = offset
            span = size
        else:
            base = offset
            span = size
        pool[:, head] = base + (i * (104_729 + head * 2_003) + SEED + head) % span
    return pool.contiguous()


def reference(ids, table, start, rows):
    valid = (ids >= start) & (ids < start + rows)
    local_ids = torch.where(valid, ids - start, torch.zeros_like(ids))
    gathered = F.embedding(local_ids, table)
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)).reshape(1, 2560)


def run_calls(call, pool, order):
    for index in order:
        call(pool[index:index + 1])


def timed_calls(call, pool, order):
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    run_calls(call, pool, order)
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    device_ms = start_event.elapsed_time(end_event)
    return device_ms / len(order), wall_ms / len(order)


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main():
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
        raise RuntimeError("run with exactly one visible XPU, e.g. ZE_AFFINITY_MASK=7")
    device = torch.device("xpu:0")
    free_before, total = torch.xpu.mem_get_info(device)
    if free_before < TABLE_BYTES + VRAM_MARGIN:
        raise RuntimeError(
            f"insufficient free VRAM: free={free_before} total={total} "
            f"need_at_least={TABLE_BYTES + VRAM_MARGIN}"
        )

    pool_cpu = make_id_pool()
    start = torch.tensor([LOCAL_START], dtype=torch.int64, device=device)
    rows = torch.tensor([LOCAL_ROWS], dtype=torch.int64, device=device)
    pool = pool_cpu.to(device=device)
    table = torch.empty((LOCAL_ROWS, ROW_WIDTH), dtype=torch.float16, device=device)

    local_indices = torch.unique(
        torch.cat((pool_cpu[:, 14] - LOCAL_START, pool_cpu[:, 15] - LOCAL_START))
    )
    row = local_indices.view(-1, 1)
    col = torch.arange(ROW_WIDTH, dtype=torch.int64).view(1, -1)
    values = (((row * 37 + col * 13 + SEED) % 2049 - 1024).float() / 64).half()
    table.index_copy_(0, local_indices.to(device), values.to(device))
    torch.xpu.synchronize()
    free_after, _ = torch.xpu.mem_get_info(device)

    for index in range(128):
        ids = pool[index:index + 1]
        expected = reference(ids, table, start, rows)
        actual = esimd_qwen38_ngram_embedding_gather(ids, table, start, rows)
        output = torch.empty((1, 2560), dtype=torch.float16, device=device)
        actual_out = esimd_qwen38_ngram_embedding_gather_out(
            ids, table, start, rows, output
        )
        torch.xpu.synchronize()
        if not torch.equal(actual, expected) or not torch.equal(actual_out, expected):
            raise AssertionError(f"large companion mismatch at {index}")

    eager = lambda ids: reference(ids, table, start, rows)
    allocating = lambda ids: esimd_qwen38_ngram_embedding_gather(
        ids, table, start, rows
    )
    output = torch.empty((1, 2560), dtype=torch.float16, device=device)
    preallocated = lambda ids: esimd_qwen38_ngram_embedding_gather_out(
        ids, table, start, rows, output
    )
    order_generator = torch.Generator(device="cpu").manual_seed(SEED)
    samples = {key: [] for key in (
        "eager_device", "allocating_device", "preallocated_device",
        "eager_wall", "allocating_wall", "preallocated_wall")}
    for _ in range(repetitions):
        order = torch.randperm(POOL_SIZE, generator=order_generator).tolist()
        for key, call in (
            ("eager", eager),
            ("allocating", allocating),
            ("preallocated", preallocated),
        ):
            device_ms, wall_ms = timed_calls(call, pool, order)
            samples[f"{key}_device"].append(device_ms)
            samples[f"{key}_wall"].append(wall_ms)

    result = {
        "status": "PASS",
        "classification": "synthetic_static_performance_companion",
        "device": torch.xpu.get_device_name(0),
        "physical_card_requested": 7,
        "rows": LOCAL_ROWS,
        "row_width": ROW_WIDTH,
        "table_bytes": TABLE_BYTES,
        "table_gib": TABLE_BYTES / 1024**3,
        "local_vocab_start": LOCAL_START,
        "pool_size": POOL_SIZE,
        "initialized_rows": int(local_indices.numel()),
        "initialized_working_set_bytes": int(local_indices.numel() * ROW_WIDTH * 2),
        "free_vram_before": int(free_before),
        "free_vram_after": int(free_after),
        "total_vram": int(total),
        "seed": SEED,
        "repetitions": repetitions,
        "calls_per_rep": POOL_SIZE,
        "correctness_samples": 128,
        "samples": samples,
        "medians_us": {key: median(value) * 1000 for key, value in samples.items()},
        "speedups": {
            "allocating_device": median(samples["eager_device"]) / median(samples["allocating_device"]),
            "preallocated_device": median(samples["eager_device"]) / median(samples["preallocated_device"]),
            "allocating_wall": median(samples["eager_wall"]) / median(samples["allocating_wall"]),
            "preallocated_wall": median(samples["eager_wall"]) / median(samples["preallocated_wall"]),
        },
        "max_abs_err": 0,
        "timed_region_excludes": ["allocation", "row initialization", "H2D", "import", "JIT", "correctness check"],
    }
    out = Path("/llm/models/test/qwen38_ngram_ko/formal/ng2a_local_gather_static_001/performance_companion/llm_scaler_large_card7_results.json")
    # Keep benchmark output beside the existing static companion, while still
    # allowing this script to run directly from the llm-scaler checkout.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
