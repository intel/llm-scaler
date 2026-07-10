"""
Performance benchmarks for omni_xpu_kernel INT8 operations.

Compares:
  (a) Native C++ INT8 (oneDNN + ESIMD) — when available
  (b) Python reference implementation
  (c) BF16 torch.nn.functional.linear baseline
  (d) FP8 onednn_w8a16_fp8 — when available

Tests at ComfyUI-relevant shapes.

Usage:
    python tests/bench_int8.py
    python tests/bench_int8.py --device xpu
    python tests/bench_int8.py --shapes comfyui
"""

import argparse
import time
from typing import Optional

import torch


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def warmup_and_bench(fn, warmup=5, iters=20, sync_fn=None):
    """Run warmup iterations, then measure."""
    for _ in range(warmup):
        fn()
    if sync_fn:
        sync_fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync_fn:
        sync_fn()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms


def get_sync_fn(device):
    """Get device synchronization function."""
    if device.type == 'xpu':
        return torch.xpu.synchronize
    elif device.type == 'cuda':
        return torch.cuda.synchronize
    return None


COMFYUI_SHAPES = [
    # (M, N, K) — typical ComfyUI diffusion model shapes
    (1, 4096, 4096),       # Single token generation
    (4, 4096, 4096),       # Small batch
    (32, 4096, 4096),      # Medium batch
    (128, 4096, 4096),     # Large batch
    (4096, 4096, 4096),    # Full sequence
    (1, 12288, 4096),      # MLP up-projection
    (32, 12288, 4096),     # MLP up-projection batch
    (1, 4096, 12288),      # MLP down-projection
    (32, 4096, 12288),     # MLP down-projection batch
]

SMALL_SHAPES = [
    (1, 64, 128),
    (4, 128, 256),
    (16, 256, 512),
    (64, 512, 1024),
    (128, 1024, 2048),
]


def bench_int8_linear(device, shapes, iters=20):
    """Benchmark int8_linear vs baselines."""
    from omni_xpu_kernel import int8

    sync_fn = get_sync_fn(device)
    results = []

    print(f"\n{'='*80}")
    print(f"INT8 Linear Benchmark — device={device}, iters={iters}")
    print(f"{'='*80}")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'bf16 F.linear':>14} | {'int8_linear':>14} | {'speedup':>8}")
    print(f"{'-'*6} {'-'*6} {'-'*6} | {'-'*14} | {'-'*14} | {'-'*8}")

    for m, n, k in shapes:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        bias = torch.randn(n, device=device, dtype=torch.bfloat16)

        # Quantize weight
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        # Baseline: BF16 linear
        def bf16_linear():
            return torch.nn.functional.linear(x, w, bias)

        # INT8 linear
        def int8_lin():
            return int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

        ms_bf16 = warmup_and_bench(bf16_linear, iters=iters, sync_fn=sync_fn)
        ms_int8 = warmup_and_bench(int8_lin, iters=iters, sync_fn=sync_fn)
        speedup = ms_bf16 / ms_int8 if ms_int8 > 0 else float('inf')

        print(f"{m:>6} {n:>6} {k:>6} | {ms_bf16:>11.3f} ms | {ms_int8:>11.3f} ms | {speedup:>6.2f}x")
        results.append({
            'shape': (m, n, k),
            'bf16_ms': ms_bf16,
            'int8_ms': ms_int8,
            'speedup': speedup,
        })

    return results


def bench_quantize(device, shapes, iters=50):
    """Benchmark quantization operations."""
    from omni_xpu_kernel import int8

    sync_fn = get_sync_fn(device)

    print(f"\n{'='*80}")
    print(f"INT8 Quantization Benchmark — device={device}")
    print(f"{'='*80}")
    print(f"{'M':>6} {'K':>6} | {'tensorwise':>12} | {'rowwise':>12}")
    print(f"{'-'*6} {'-'*6} | {'-'*12} | {'-'*12}")

    for m, _, k in shapes:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)

        def quant_tw():
            return int8.quantize_int8_tensorwise(x)

        def quant_rw():
            return int8.quantize_int8_rowwise(x)

        ms_tw = warmup_and_bench(quant_tw, iters=iters, sync_fn=sync_fn)
        ms_rw = warmup_and_bench(quant_rw, iters=iters, sync_fn=sync_fn)

        print(f"{m:>6} {k:>6} | {ms_tw:>9.3f} ms | {ms_rw:>9.3f} ms")


def bench_mm_int8(device, shapes, iters=20):
    """Benchmark raw INT8 matmul."""
    from omni_xpu_kernel import int8

    sync_fn = get_sync_fn(device)

    print(f"\n{'='*80}")
    print(f"INT8 MatMul (mm_int8) Benchmark — device={device}")
    print(f"{'='*80}")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'mm_int8':>12} | {'bf16 mm':>12} | {'speedup':>8}")
    print(f"{'-'*6} {'-'*6} {'-'*6} | {'-'*12} | {'-'*12} | {'-'*8}")

    for m, n, k in shapes:
        a_int8 = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
        b_int8 = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)
        a_bf16 = a_int8.to(torch.bfloat16)
        b_bf16 = b_int8.to(torch.bfloat16)

        def mm_i8():
            return int8.mm_int8(a_int8, b_int8)

        def mm_bf16():
            return torch.mm(a_bf16, b_bf16)

        ms_int8 = warmup_and_bench(mm_i8, iters=iters, sync_fn=sync_fn)
        ms_bf16 = warmup_and_bench(mm_bf16, iters=iters, sync_fn=sync_fn)
        speedup = ms_bf16 / ms_int8 if ms_int8 > 0 else float('inf')

        print(f"{m:>6} {n:>6} {k:>6} | {ms_int8:>9.3f} ms | {ms_bf16:>9.3f} ms | {speedup:>6.2f}x")


def bench_fp8_comparison(device, shapes, iters=20):
    """Compare INT8 with FP8 (when available)."""
    from omni_xpu_kernel import int8

    try:
        from omni_xpu_kernel import linear as fp8_linear_mod
        has_fp8 = True
    except Exception:
        has_fp8 = False

    if not has_fp8:
        print("\n[SKIP] FP8 comparison — native extension not available")
        return

    sync_fn = get_sync_fn(device)

    print(f"\n{'='*80}")
    print(f"INT8 vs FP8 Comparison — device={device}")
    print(f"{'='*80}")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'int8_linear':>14} | {'fp8_linear':>14} | {'ratio':>8}")
    print(f"{'-'*6} {'-'*6} {'-'*6} | {'-'*14} | {'-'*14} | {'-'*8}")

    for m, n, k in shapes:
        if k < 2048:
            continue  # FP8 shape guard requires K >= 2048

        x = torch.randn(m, k, device=device, dtype=torch.float16)
        w_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)

        # INT8 setup
        w_int8, w_scale_int8 = int8.quantize_int8_tensorwise(w_fp32.to(torch.float16))

        # FP8 setup
        scales_fp8 = (w_fp32.abs().max(dim=1).values / 448.0).clamp(min=1e-12)
        w_fp8 = (w_fp32 / scales_fp8.unsqueeze(1)).to(torch.float8_e4m3fn)

        def int8_fn():
            return int8.int8_linear(x, w_int8, w_scale_int8, out_dtype=torch.float16)

        def fp8_fn():
            return fp8_linear_mod.onednn_w8a16_fp8(x, w_fp8, scales_fp8)

        ms_int8 = warmup_and_bench(int8_fn, iters=iters, sync_fn=sync_fn)

        try:
            ms_fp8 = warmup_and_bench(fp8_fn, iters=iters, sync_fn=sync_fn)
            ratio = ms_int8 / ms_fp8 if ms_fp8 > 0 else float('inf')
            print(f"{m:>6} {n:>6} {k:>6} | {ms_int8:>11.3f} ms | {ms_fp8:>11.3f} ms | {ratio:>6.2f}x")
        except Exception as e:
            print(f"{m:>6} {n:>6} {k:>6} | {ms_int8:>11.3f} ms | {'ERROR':>14} | {str(e)[:30]}")


def main():
    parser = argparse.ArgumentParser(description="INT8 Performance Benchmarks")
    parser.add_argument("--device", default="auto",
                        help="Device to benchmark on (auto, cpu, xpu, cuda)")
    parser.add_argument("--shapes", default="comfyui",
                        choices=["comfyui", "small", "all"],
                        help="Shape set to benchmark")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    if args.device == "auto":
        if has_xpu():
            device = torch.device("xpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Benchmarking on: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if args.shapes == "comfyui":
        shapes = COMFYUI_SHAPES
    elif args.shapes == "small":
        shapes = SMALL_SHAPES
    else:
        shapes = SMALL_SHAPES + COMFYUI_SHAPES

    bench_quantize(device, shapes, iters=args.iters * 2)
    bench_mm_int8(device, shapes, iters=args.iters)
    bench_int8_linear(device, shapes, iters=args.iters)
    bench_fp8_comparison(device, shapes, iters=args.iters)

    print(f"\n{'='*80}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
