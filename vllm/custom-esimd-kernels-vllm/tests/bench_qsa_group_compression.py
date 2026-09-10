"""One-warmup/one-formal benchmark for QSA group compression ABI1."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

WIDTH = 128
RATIO = 4
RING_SIZE = 8


def _load_qsa_extension():
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


def _load_torch_reference():
    vllm_root = Path(__file__).resolve().parents[4] / (
        "applications.ai.gpu.llm-scaler-vllm"
    )
    if vllm_root.is_dir():
        sys.path.insert(0, str(vllm_root))
    from vllm.models.qwen3_8_flash_next.xpu.ops.qsa import (
        qsa_compress_groups_with_ratio,
    )

    return qsa_compress_groups_with_ratio


def _make_case(dtype: torch.dtype):
    device = torch.device("xpu")
    pages = 2
    storage = torch.empty(
        (pages, RING_SIZE, 1, WIDTH + 12), dtype=dtype, device=device
    )
    cache = storage[..., :WIDTH]
    positions = storage[..., WIDTH:].view(torch.int64)
    for page in range(pages):
        for slot in range(RING_SIZE):
            cache[page, slot, 0].fill_(page * 100 + slot)
            positions[page, slot, 0] = torch.tensor(
                [
                    page * 1000 + slot,
                    page * 1000 + slot + 10,
                    page * 1000 + slot + 20,
                ],
                dtype=torch.int64,
                device=device,
            )
    raw = torch.arange(128, dtype=torch.float32, device=device).reshape(
        1, 1, WIDTH
    )
    raw = (raw / 17.0).to(dtype)
    raw_positions = torch.tensor(
        [[[11, 111, 211]]], dtype=torch.int64, device=device
    )
    block_table = torch.tensor([[1]], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    logical_positions = torch.tensor([11], dtype=torch.int64, device=device)
    compressed_slots = torch.tensor([7], dtype=torch.int64, device=device)
    pooled = torch.full((1, 1, WIDTH), -99, dtype=dtype, device=device)
    first_positions = torch.full(
        (1, 3), -99, dtype=torch.int64, device=device
    )
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


def _native_call(qsa_ops, case):
    return qsa_ops.qsa_group_compress_v1(*case, RATIO, 16, True)


def _torch_call(torch_reference, case):
    (
        raw,
        raw_positions,
        cache,
        _positions,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        _pooled,
        _first_positions,
    ) = case
    return torch_reference(
        raw,
        raw_positions,
        cache,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        RATIO,
        case[3],
        historical_ring_proven=True,
    )


def _measure(function, repetitions: int) -> float:
    torch.xpu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(repetitions):
        function()
    torch.xpu.synchronize()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def _run_dtype(qsa_ops, torch_reference, dtype, repetitions):
    case = _make_case(dtype)
    reference_pooled, reference_positions = _torch_call(torch_reference, case)
    _native_call(qsa_ops, case)
    torch.xpu.synchronize()
    torch.testing.assert_close(
        case[9].float(),
        reference_pooled.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.equal(case[10], reference_positions)

    # One warmup phase for each implementation, followed by one formal sample.
    _torch_call(torch_reference, case)
    _native_call(qsa_ops, case)
    torch.xpu.synchronize()
    torch_ms = _measure(
        lambda: _torch_call(torch_reference, case), repetitions
    )
    native_ms = _measure(
        lambda: _native_call(qsa_ops, case), repetitions
    )
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "repetitions": repetitions,
        "torch_total_ms": torch_ms,
        "native_total_ms": native_ms,
        "torch_us_per_call": torch_ms * 1000.0 / repetitions,
        "native_us_per_call": native_ms * 1000.0 / repetitions,
        "speedup": torch_ms / native_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "both"),
        default="both",
    )
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if not torch.xpu.is_available() or torch.xpu.device_count() < 1:
        raise RuntimeError("QSA group compression benchmark requires one XPU")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtypes = tuple(dtype_map) if args.dtype == "both" else (args.dtype,)
    qsa_ops = _load_qsa_extension()
    torch_reference = _load_torch_reference()
    result = {
        "kind": "qwen38_qsa_group_compression_abi1_microbenchmark",
        "protocol": "one warmup phase plus one formal sample",
        "comparison_scope": (
            "production Torch qsa_compress_groups_with_ratio versus "
            "caller-owned native ABI1"
        ),
        "claim_boundary": (
            "single-XPU operator comparison; not TP8, model-load, "
            "production-server, or full-model E2E evidence"
        ),
        "shape": {
            "rows": 1,
            "head_dim": WIDTH,
            "compression_ratio": RATIO,
            "ring_size": RING_SIZE,
        },
        "results": [
            _run_dtype(qsa_ops, torch_reference, dtype_map[dtype], args.repetitions)
            for dtype in dtypes
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
