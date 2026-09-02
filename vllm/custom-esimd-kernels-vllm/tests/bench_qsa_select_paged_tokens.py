#!/usr/bin/env python3
"""B/C/B latency for the fused FP16 QSA token-selection kernel."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch


INDEX_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 128
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1
CASES = {
    "cross_page_tail": (518, 520),
    "p8192_saturated_2051": (9214, 9216),
}


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
        pages = page_table[request, logical // PAGE_SIZE].long()
        keys = cache[pages, logical % PAGE_SIZE, 0]
        scores = torch.relu(
            torch.einsum("hd,nd->nh", q[row].float(), keys.float())
        ).sum(dim=-1) * (HEAD_DIM**-0.5)
        selected[row, :width] = torch.topk(scores, width).indices.to(
            torch.int32
        )
    return _expand_reference(selected, query_positions, row_lengths.long())


def _make_inputs(query_position: int, sequence_length: int):
    visible_blocks = min(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    page_columns = max(1, math.ceil(visible_blocks / PAGE_SIZE))
    cache = torch.randn(
        page_columns,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.float16,
        device="xpu",
    )
    q = torch.randn(
        1, INDEX_HEADS, HEAD_DIM, dtype=torch.float16, device="xpu"
    )
    page_table = torch.arange(
        page_columns, dtype=torch.int32, device="xpu"
    ).view(1, page_columns)
    token_to_req = torch.zeros(1, dtype=torch.int32, device="xpu")
    query_positions = torch.tensor(
        [query_position], dtype=torch.int64, device="xpu"
    )
    sequence_lengths = torch.tensor(
        [sequence_length], dtype=torch.int32, device="xpu"
    )
    out = torch.empty(1, OUTPUT_WIDTH, dtype=torch.int32, device="xpu")
    return (
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
    )


def _measure(operation, inner: int) -> float:
    torch.xpu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(inner):
        operation()
    torch.xpu.synchronize()
    return (time.perf_counter_ns() - start) / 1_000_000 / inner


def _benchmark_case(module, values, warmup_units, samples, inner):
    q, cache, page_table, token_to_req, positions, lengths, out = values

    def baseline():
        return _select_reference(
            q, cache, page_table, token_to_req, positions, lengths
        )

    def candidate():
        return module.qsa_select_paged_tokens(
            q,
            cache,
            page_table,
            token_to_req,
            positions,
            lengths,
            TOKEN_TOPK,
            COMPRESS_RATIO,
            out,
        )

    with torch.inference_mode():
        for _ in range(warmup_units):
            _measure(baseline, inner)
            _measure(candidate, inner)
        before = []
        fused = []
        after = []
        for _ in range(samples):
            before.append(_measure(baseline, inner))
            fused.append(_measure(candidate, inner))
            after.append(_measure(baseline, inner))
    paired_baseline = [(a + b) / 2 for a, b in zip(before, after)]
    deltas = [
        (candidate_ms / baseline_ms - 1) * 100
        for candidate_ms, baseline_ms in zip(fused, paired_baseline)
    ]
    return {
        "baseline_before_ms": before,
        "candidate_ms": fused,
        "baseline_after_ms": after,
        "paired_baseline_mean_ms": statistics.mean(paired_baseline),
        "candidate_mean_ms": statistics.mean(fused),
        "paired_mean_change_percent": statistics.mean(deltas),
        "paired_deltas_percent": deltas,
        "candidate_wins": sum(value < 0 for value in deltas),
        "baseline_drift_percent": (
            statistics.mean(after) / statistics.mean(before) - 1
        )
        * 100,
        "speedup": statistics.mean(paired_baseline) / statistics.mean(fused),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--warmup-units", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--inner-iterations", type=int, default=64)
    args = parser.parse_args()
    if min(args.warmup_units, args.samples, args.inner_iterations) <= 0:
        parser.error("warm-up, samples, and inner iterations must be positive")
    selected_cases = tuple(args.cases.split(","))
    if not selected_cases or any(case not in CASES for case in selected_cases):
        parser.error(f"cases must be selected from {tuple(CASES)}")
    if torch.xpu.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible XPU")
    torch.manual_seed(20260828)
    module = _load_qsa_extension()
    results = {}
    for case in selected_cases:
        position, length = CASES[case]
        inputs = _make_inputs(position, length)
        expected = _select_reference(*inputs[:6])
        actual = module.qsa_select_paged_tokens(
            *inputs[:6], TOKEN_TOPK, COMPRESS_RATIO, inputs[-1]
        )
        torch.xpu.synchronize()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"correctness precheck failed for {case}")
        result = _benchmark_case(
            module,
            inputs,
            args.warmup_units,
            args.samples,
            args.inner_iterations,
        )
        result.update(
            {
                "query_position": position,
                "sequence_length": length,
                "visible_blocks": min((position + 1) // 4, length // 4),
                "valid_output_width": int((actual >= 0).sum().item()),
            }
        )
        results[case] = result
    report = {
        "schema_version": 1,
        "kind": "qwen38_qsa_fp16_select_bcb_microbenchmark",
        "rows": 1,
        "warmup_units": args.warmup_units,
        "samples": args.samples,
        "inner_iterations": args.inner_iterations,
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "results": results,
        "claim_boundary": (
            "rank-local Torch reference versus fused kernel; not TP8 E2E"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
