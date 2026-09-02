"""Rank-local microbenchmark for canonical Qwen3.8 PLE INT4 projections.

This measures only the existing projection primitives through the standalone
DSOs.  It is not a TP8, vLLM, or end-to-end PLE benchmark.

Example::

  ZE_AFFINITY_MASK=6 \
  PLE_DSO=/abs/path/custom_esimd_kernels.so \
  PLE_GEMM_DSO=/abs/path/custom_esimd_kernels_gemm.so \
  python tests/bench_ple_projection_int4.py --warmup 20 --repeat 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


K = 2560
PROJECTIONS = (("key", 10240), ("value", 2560))


def _load_dso(path: str) -> None:
    if not path:
        raise ValueError("DSO path must not be empty")
    dso = Path(path)
    if not dso.is_file():
        raise FileNotFoundError(dso)
    torch.ops.load_library(str(dso))


def _make_case(m: int, n: int) -> tuple[torch.Tensor, ...]:
    input_tensor = (torch.randn(m, K, dtype=torch.float16) * 0.05).to("xpu")
    weight = torch.randint(0, 256, (n, K // 2), dtype=torch.uint8).to("xpu")
    scale = (
        torch.rand(n, K // 128, dtype=torch.float32) * 0.25 + 0.01
    ).to(torch.float16).to("xpu")
    output = torch.empty((m, n), dtype=torch.float16, device="xpu")
    return input_tensor, weight, scale, output


def _measure(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    *,
    m: int,
    warmup: int,
    repeat: int,
) -> list[float]:
    if m == 1:
        op = torch.ops.custom_esimd_kernels_vllm.esimd_gemv_int4
    else:
        op = torch.ops.custom_esimd_kernels_vllm.esimd_gemm_int4_pgrp

    for _ in range(warmup):
        op(input_tensor, weight, scale, output)
    torch.xpu.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        op(input_tensor, weight, scale, output)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")

    ple_dso = os.environ.get("PLE_DSO", "")
    gemm_dso = os.environ.get("PLE_GEMM_DSO", "")
    _load_dso(ple_dso)
    if gemm_dso:
        _load_dso(gemm_dso)
    for op_name in ("esimd_gemv_int4", "esimd_gemm_int4_pgrp"):
        if not torch._C._jit_get_schemas_for_operator(
            f"custom_esimd_kernels_vllm::{op_name}"
        ):
            raise RuntimeError(f"required operator is not registered: {op_name}")

    records: list[dict[str, object]] = []
    for m in (1, 4):
        for name, n in PROJECTIONS:
            inputs = _make_case(m, n)
            samples = _measure(*inputs, m=m, warmup=args.warmup, repeat=args.repeat)
            dso = ple_dso if m == 1 else gemm_dso
            records.append(
                {
                    "projection": name,
                    "m": m,
                    "n": n,
                    "k": K,
                    "dispatch": "gemv" if m == 1 else "gemm",
                    "dso": str(Path(dso).resolve()),
                    "dso_sha256": hashlib.sha256(Path(dso).read_bytes()).hexdigest(),
                    "samples_ms": samples,
                    "p50_ms": _percentile(samples, 0.50),
                    "p90_ms": _percentile(samples, 0.90),
                }
            )

    report = {
        "schema": "qwen38.ple.projection-benchmark.v1",
        "claim_boundary": "rank-local projection primitive; not TP8 or end-to-end",
        "affinity": os.environ.get("ZE_AFFINITY_MASK", "unset"),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "records": records,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
