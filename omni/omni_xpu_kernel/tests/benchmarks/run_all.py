#!/usr/bin/env python
"""
Run all benchmarks for omni_xpu_kernel

Usage:
    python -m benchmarks.run_all           # Run all benchmarks
    python -m benchmarks.run_all --gguf    # Run GGUF benchmarks only
    python -m benchmarks.run_all --norm    # Run normalization benchmarks only
"""

import argparse
import torch


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def main():
    parser = argparse.ArgumentParser(description="omni_xpu_kernel benchmarks")
    parser.add_argument("--gguf", action="store_true", help="Run GGUF benchmarks only")
    parser.add_argument("--norm", action="store_true", help="Run normalization benchmarks only")
    args = parser.parse_args()
    
    if not has_xpu():
        print("XPU not available, skipping benchmarks")
        return
    
    run_all = not args.gguf and not args.norm
    
    if run_all or args.gguf:
        from .bench_gguf import run_benchmarks as run_gguf
        run_gguf()
        print()
    
    if run_all or args.norm:
        from .bench_norm import run_benchmarks as run_norm
        run_norm()


if __name__ == "__main__":
    main()
