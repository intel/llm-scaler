#!/usr/bin/env python3
"""Steady single-row latency for the FP16 exact-packed QSA kernel."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


HEADS = 3
HEAD_DIM = 256
PAGE_SIZE = 512
INDEX_WIDTH = 2051


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


def _canonicalize_singleton_dim_strides(tensor: torch.Tensor) -> torch.Tensor:
    strides = list(tensor.stride())
    previous_stride = 1
    changed = False
    for dim in range(tensor.dim() - 1, -1, -1):
        if tensor.shape[dim] == 1 and strides[dim] != previous_stride:
            strides[dim] = previous_stride
            changed = True
        previous_stride = strides[dim] * tensor.shape[dim]
    return tensor.as_strided(tensor.shape, strides) if changed else tensor


def _make_inputs(valid_width: int):
    if not 0 <= valid_width <= INDEX_WIDTH:
        raise ValueError(f"valid width must be in [0,{INDEX_WIDTH}]")
    pages = 5
    packed_kv = torch.randn(
        pages,
        1,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=torch.float16,
        device="xpu",
    )
    k_cache, v_cache = packed_kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    k_cache = _canonicalize_singleton_dim_strides(k_cache)
    v_cache = _canonicalize_singleton_dim_strides(v_cache)
    q = 0.1 * torch.randn(
        1, HEADS, HEAD_DIM, dtype=torch.float16, device="xpu"
    )
    logical = torch.full(
        (1, INDEX_WIDTH), -1, dtype=torch.int32, device="xpu"
    )
    logical[:, :valid_width] = torch.arange(
        valid_width, dtype=torch.int32, device="xpu"
    )
    block_table = torch.arange(
        pages, dtype=torch.int32, device="xpu"
    ).view(1, pages)
    token_to_req = torch.zeros(1, dtype=torch.int32, device="xpu")
    out = torch.empty_like(q)
    return q, k_cache, v_cache, logical, block_table, token_to_req, out


def _measure(module, args, warmup_units: int, samples: int, inner: int):
    def operation():
        return module.sparse_paged_attention(*args)

    with torch.inference_mode():
        for _ in range(warmup_units):
            for _ in range(inner):
                operation()
            torch.xpu.synchronize()
        samples_ms = []
        for _ in range(samples):
            torch.xpu.synchronize()
            start = time.perf_counter_ns()
            for _ in range(inner):
                operation()
            torch.xpu.synchronize()
            samples_ms.append(
                (time.perf_counter_ns() - start) / 1_000_000 / inner
            )
    return samples_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-widths", default="32,1024,2051")
    parser.add_argument("--warmup-units", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--inner-iterations", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup_units, args.samples, args.inner_iterations) <= 0:
        parser.error("warm-up, samples, and inner iterations must be positive")
    widths = tuple(int(value) for value in args.valid_widths.split(","))
    if torch.xpu.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible XPU")

    torch.manual_seed(20260828)
    module = _load_qsa_extension()
    results = []
    for width in widths:
        inputs = _make_inputs(width)
        samples_ms = _measure(
            module,
            inputs,
            args.warmup_units,
            args.samples,
            args.inner_iterations,
        )
        results.append(
            {
                "valid_width": width,
                "mean_ms": statistics.mean(samples_ms),
                "samples_ms": samples_ms,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "qwen38_qsa_fp16_exact_packed_r1_microbenchmark",
        "rows": 1,
        "warmup_units": args.warmup_units,
        "samples": args.samples,
        "inner_iterations": args.inner_iterations,
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "results": results,
        "claim_boundary": "single-kernel rank-local latency; not TP8 E2E",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
