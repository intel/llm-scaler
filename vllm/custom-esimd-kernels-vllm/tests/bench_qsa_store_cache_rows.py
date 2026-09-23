"""One-warmup/one-formal benchmark for QSA cache row-store ABI3."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
from pathlib import Path

import torch

WIDTH = 128
STORES_PER_CYCLE = 8


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


def _make_cache() -> torch.Tensor:
    backing = torch.zeros(
        (4, 128, 1, WIDTH + 12), dtype=torch.float16, device="xpu"
    )
    return backing[..., :WIDTH]


def _torch_store(
    cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    flat = cache.reshape(-1, cache.shape[3])
    valid = slots >= 0
    if not bool(valid.any()):
        return cache
    indices = slots[valid].long()
    flat.index_copy_(0, indices, rows[valid].to(cache.dtype))
    return cache


def _run_cycle(function, cache, valid_slot, null_slot, rows, repetitions):
    for _ in range(repetitions):
        function(cache, valid_slot, rows)
        function(cache, valid_slot, rows)
        function(cache, valid_slot, rows)
        function(cache, valid_slot, rows)
        function(cache, valid_slot, rows)
        function(cache, null_slot, rows)
        function(cache, null_slot, rows)
        function(cache, null_slot, rows)


def _measure(function, cache, valid_slot, null_slot, rows, repetitions):
    torch.xpu.synchronize()
    start = time.perf_counter_ns()
    _run_cycle(function, cache, valid_slot, null_slot, rows, repetitions)
    torch.xpu.synchronize()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if not torch.xpu.is_available() or torch.xpu.device_count() < 1:
        raise RuntimeError("QSA row-store benchmark requires one XPU")

    qsa_ops = _load_qsa_extension()
    native_store = qsa_ops.qsa_store_cache_rows_v3
    torch_cache = _make_cache()
    native_cache = _make_cache()
    valid_slot = torch.tensor([17], dtype=torch.int64, device="xpu")
    null_slot = torch.tensor([-1], dtype=torch.int64, device="xpu")
    rows = torch.linspace(
        -2.0, 2.0, WIDTH, dtype=torch.float16, device="xpu"
    ).view(1, WIDTH)

    _run_cycle(_torch_store, torch_cache, valid_slot, null_slot, rows, 1)
    _run_cycle(native_store, native_cache, valid_slot, null_slot, rows, 1)
    torch.xpu.synchronize()
    if not torch.equal(torch_cache, native_cache):
        raise AssertionError("native row-store failed correctness precheck")

    # One warmup phase for each implementation, followed by one formal sample.
    _run_cycle(_torch_store, torch_cache, valid_slot, null_slot, rows, 1)
    _run_cycle(native_store, native_cache, valid_slot, null_slot, rows, 1)
    torch.xpu.synchronize()

    torch_ms = _measure(
        _torch_store,
        torch_cache,
        valid_slot,
        null_slot,
        rows,
        args.repetitions,
    )
    native_ms = _measure(
        native_store,
        native_cache,
        valid_slot,
        null_slot,
        rows,
        args.repetitions,
    )
    calls = args.repetitions * STORES_PER_CYCLE
    result = {
        "kind": "qwen38_qsa_row_store_fp16_r1_microbenchmark",
        "protocol": "one warmup phase plus one formal sample",
        "comparison_scope": "production Torch helper replacement",
        "claim_boundary": (
            "includes the production helper's per-call host sync and boolean "
            "indexing; not a pure memory-copy kernel speedup"
        ),
        "shape": {"rows": 1, "width": WIDTH, "dtype": "float16"},
        "cycle": {
            "raw_valid": 4,
            "compressed_valid": 1,
            "compressed_null": 3,
        },
        "repetitions": args.repetitions,
        "calls": calls,
        "torch_total_ms": torch_ms,
        "native_total_ms": native_ms,
        "torch_us_per_store": torch_ms * 1000.0 / calls,
        "native_us_per_store": native_ms * 1000.0 / calls,
        "production_helper_speedup": torch_ms / native_ms,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
