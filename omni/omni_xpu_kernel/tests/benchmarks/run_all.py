#!/usr/bin/env python
"""
Run all benchmarks for omni_xpu_kernel.

Usage:
    python -m tests.benchmarks.run_all             # Run all benchmarks
    python -m tests.benchmarks.run_all --gguf      # GGUF dequantization only
    python -m tests.benchmarks.run_all --norm      # Normalization only
    python -m tests.benchmarks.run_all --svdq      # SVDQuant dequant only
    python -m tests.benchmarks.run_all --rmsnorm   # RMSNorm ESIMD vs PyTorch
    python -m tests.benchmarks.run_all --rotary    # Rotary ESIMD vs PyTorch
    python -m tests.benchmarks.run_all --onednn    # oneDNN INT4 GEMM
    python -m tests.benchmarks.run_all --postproc  # fused_convert_add
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


_BENCHMARKS = [
    "gguf", "norm", "svdq", "rmsnorm", "rotary", "onednn", "postproc", "sdp",
]


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

    if run_all or "gguf" in selected:
        run_gguf = _load_benchmark_module("bench_gguf.py").run_benchmarks
        print("=" * 60, " GGUF Dequantization ", "=" * 60)
        run_gguf()
        print()

    if run_all or "norm" in selected:
        run_norm = _load_benchmark_module("bench_norm.py").run_benchmarks
        print("=" * 60, " Normalization ", "=" * 60)
        run_norm()
        print()

    if run_all or "svdq" in selected:
        run_svdq = _load_benchmark_module("bench_svdq.py").run_benchmarks
        print("=" * 60, " SVDQuant Dequant ", "=" * 60)
        run_svdq()
        print()

    if run_all or "rmsnorm" in selected:
        run_rmsnorm = _load_benchmark_module("bench_rmsnorm.py").main
        print("=" * 60, " RMSNorm ESIMD ", "=" * 60)
        run_rmsnorm()
        print()

    if run_all or "rotary" in selected:
        run_rotary = _load_benchmark_module("bench_rotary.py").main
        print("=" * 60, " Rotary Embedding ESIMD ", "=" * 60)
        run_rotary()
        print()

    if run_all or "onednn" in selected:
        run_onednn = _load_benchmark_module("bench_onednn_int4.py").main
        print("=" * 60, " oneDNN INT4 GEMM ", "=" * 60)
        run_onednn()
        print()

    if run_all or "postproc" in selected:
        run_postproc = _load_benchmark_module("bench_fused_postproc.py").main
        print("=" * 60, " Fused Convert+Add ", "=" * 60)
        run_postproc()
        print()

    if run_all or "sdp" in selected:
        run_sdp = _load_benchmark_module("bench_sdp.py").run_benchmarks
        print("=" * 60, " SDP ", "=" * 60)
        run_sdp()
        print()


if __name__ == "__main__":
    main()
