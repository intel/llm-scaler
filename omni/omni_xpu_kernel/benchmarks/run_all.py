#!/usr/bin/env python
"""
Run all benchmarks for omni_xpu_kernel.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --fp8
    python -m benchmarks.run_all --gguf
    python -m benchmarks.run_all --norm
    python -m benchmarks.run_all --sdp
"""

import argparse
import importlib.util
from pathlib import Path
import torch


def _load_benchmark_module(module_filename: str):
    benchmark_path = Path(__file__).with_name(module_filename)
    spec = importlib.util.spec_from_file_location(f"omni_bench_{benchmark_path.stem}", benchmark_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmark module from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


_BENCHMARKS = {
    "fp8": ("bench_fp8.py", "main", "FP8 GEMM"),
    "gguf": ("bench_gguf.py", "run_benchmarks", "GGUF Dequantization"),
    "norm": ("bench_norm.py", "run_benchmarks", "Normalization"),
    "sdp": ("bench_sdp.py", "run_benchmarks", "SDP"),
}


def main():
    parser = argparse.ArgumentParser(description="omni_xpu_kernel benchmarks")
    for name in _BENCHMARKS:
        parser.add_argument(f"--{name}", action="store_true",
                            help=f"Run {name} benchmarks only")
    args = parser.parse_args()

    if not has_xpu():
        print("XPU not available, skipping benchmarks")
        return

    selected = {n for n in _BENCHMARKS if getattr(args, n)}
    run_all = len(selected) == 0

    names = _BENCHMARKS if run_all else selected
    for name in names:
        filename, entrypoint, title = _BENCHMARKS[name]
        benchmark = getattr(_load_benchmark_module(filename), entrypoint)
        print("=" * 60, f" {title} ", "=" * 60)
        benchmark()
        print()


if __name__ == "__main__":
    main()
