import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest


def _load_bench_sdp_module():
    module_path = Path(__file__).resolve().parent / "benchmarks" / "bench_sdp.py"
    spec = importlib.util.spec_from_file_location("bench_sdp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_benchmarks_lists_sglang_inspired_shapes(monkeypatch):
    bench_sdp = _load_bench_sdp_module()

    monkeypatch.setattr(bench_sdp, "has_xpu", lambda: True)
    monkeypatch.setattr(
        bench_sdp,
        "benchmark_sdp",
        lambda q_len, kv_len, heads, dim, dtype, warmup=10, iters=50, repeats=5: {
            "cold_kernel_ms": 1.0,
            "steady_kernel_ms": 1.0,
            "steady_kernel_std_ms": 0.0,
            "cold_torch_ms": 1.0,
            "steady_torch_ms": 1.0,
            "steady_torch_std_ms": 0.0,
            "steady_speedup": 1.0,
            "steady_kernel_tflops": 1.0,
            "steady_torch_tflops": 1.0,
        },
    )

    output = io.StringIO()
    with redirect_stdout(output):
        bench_sdp.run_benchmarks()

    rendered = output.getvalue()
    for label in (
        "wan-cross-1560x512",
        "wan-cross-3600x512",
        "wan-self-1560",
        "wan-self-3600",
        "flux-cross-4096x512",
        "flux-self-4096",
    ):
        assert label in rendered


def test_run_benchmarks_reports_cold_and_steady_state(monkeypatch):
    bench_sdp = _load_bench_sdp_module()

    monkeypatch.setattr(bench_sdp, "has_xpu", lambda: True)
    monkeypatch.setattr(
        bench_sdp,
        "benchmark_sdp",
        lambda q_len, kv_len, heads, dim, dtype, warmup=10, iters=50, repeats=5: {
            "cold_kernel_ms": 5.0,
            "steady_kernel_ms": 2.0,
            "steady_kernel_std_ms": 0.1,
            "cold_torch_ms": 6.0,
            "steady_torch_ms": 3.0,
            "steady_torch_std_ms": 0.2,
            "steady_speedup": 1.5,
            "steady_kernel_tflops": 10.0,
            "steady_torch_tflops": 5.0,
        },
    )

    output = io.StringIO()
    with redirect_stdout(output):
        bench_sdp.run_benchmarks()

    rendered = output.getvalue()
    assert "cold=" in rendered
    assert "steady=" in rendered
    assert "std=" in rendered
