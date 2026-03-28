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
import torch


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


_BENCHMARKS = [
    "gguf", "norm", "svdq", "rmsnorm", "rotary", "onednn", "postproc",
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
        from .bench_gguf import run_benchmarks as run_gguf
        print("=" * 60, " GGUF Dequantization ", "=" * 60)
        run_gguf()
        print()

    if run_all or "norm" in selected:
        from .bench_norm import run_benchmarks as run_norm
        print("=" * 60, " Normalization ", "=" * 60)
        run_norm()
        print()

    if run_all or "svdq" in selected:
        from .bench_svdq import run_benchmarks as run_svdq
        print("=" * 60, " SVDQuant Dequant ", "=" * 60)
        run_svdq()
        print()

    if run_all or "rmsnorm" in selected:
        from .bench_rmsnorm import main as run_rmsnorm
        print("=" * 60, " RMSNorm ESIMD ", "=" * 60)
        run_rmsnorm()
        print()

    if run_all or "rotary" in selected:
        from .bench_rotary import main as run_rotary
        print("=" * 60, " Rotary Embedding ESIMD ", "=" * 60)
        run_rotary()
        print()

    if run_all or "onednn" in selected:
        from .bench_onednn_int4 import main as run_onednn
        print("=" * 60, " oneDNN INT4 GEMM ", "=" * 60)
        run_onednn()
        print()

    if run_all or "postproc" in selected:
        from .bench_fused_postproc import main as run_postproc
        print("=" * 60, " Fused Convert+Add ", "=" * 60)
        run_postproc()
        print()


if __name__ == "__main__":
    main()
