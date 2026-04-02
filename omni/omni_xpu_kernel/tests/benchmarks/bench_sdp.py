"""Benchmark standalone SDP kernel against torch SDPA."""

import statistics
import time

import torch


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def make_qkv(device, *, q_len, kv_len, heads, dim, dtype):
    q = torch.randn(1, q_len, heads, dim, device=device, dtype=dtype)
    k = torch.randn(1, kv_len, heads, dim, device=device, dtype=dtype)
    v = torch.randn(1, kv_len, heads, dim, device=device, dtype=dtype)
    return q, k, v


def _measure_once(q_len, kv_len, heads, dim, dtype, warmup, iters):
    from omni_xpu_kernel import sdp

    device = torch.device("xpu")
    q, k, v = make_qkv(device, q_len=q_len, kv_len=kv_len, heads=heads, dim=dim, dtype=dtype)

    q_bhld = q.permute(0, 2, 1, 3).contiguous()
    k_bhld = k.permute(0, 2, 1, 3).contiguous()
    v_bhld = v.permute(0, 2, 1, 3).contiguous()

    for _ in range(warmup):
        sdp.sdp(q, k, v)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        sdp.sdp(q, k, v)
    torch.xpu.synchronize()
    kernel_ms = (time.perf_counter() - t0) / iters * 1000.0

    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, dropout_p=0.0, is_causal=False)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, dropout_p=0.0, is_causal=False)
    torch.xpu.synchronize()
    torch_ms = (time.perf_counter() - t0) / iters * 1000.0

    flops = 4.0 * q_len * kv_len * heads * dim
    kernel_tflops = flops / (kernel_ms / 1000.0) / 1e12
    torch_tflops = flops / (torch_ms / 1000.0) / 1e12

    return kernel_ms, torch_ms, kernel_tflops, torch_tflops


def benchmark_sdp(q_len, kv_len, heads, dim, dtype, warmup=10, iters=50, repeats=5):
    measurements = [
        _measure_once(q_len, kv_len, heads, dim, dtype, warmup, iters)
        for _ in range(repeats)
    ]

    kernel_runs = [row[0] for row in measurements]
    torch_runs = [row[1] for row in measurements]
    kernel_tflops_runs = [row[2] for row in measurements]
    torch_tflops_runs = [row[3] for row in measurements]

    steady_kernel_runs = kernel_runs[1:] if len(kernel_runs) > 1 else kernel_runs
    steady_torch_runs = torch_runs[1:] if len(torch_runs) > 1 else torch_runs
    steady_kernel_tflops_runs = kernel_tflops_runs[1:] if len(kernel_tflops_runs) > 1 else kernel_tflops_runs
    steady_torch_tflops_runs = torch_tflops_runs[1:] if len(torch_tflops_runs) > 1 else torch_tflops_runs

    return {
        "cold_kernel_ms": kernel_runs[0],
        "steady_kernel_ms": statistics.mean(steady_kernel_runs),
        "steady_kernel_std_ms": statistics.pstdev(steady_kernel_runs) if len(steady_kernel_runs) > 1 else 0.0,
        "cold_torch_ms": torch_runs[0],
        "steady_torch_ms": statistics.mean(steady_torch_runs),
        "steady_torch_std_ms": statistics.pstdev(steady_torch_runs) if len(steady_torch_runs) > 1 else 0.0,
        "steady_speedup": statistics.mean(steady_torch_runs) / statistics.mean(steady_kernel_runs),
        "steady_kernel_tflops": statistics.mean(steady_kernel_tflops_runs),
        "steady_torch_tflops": statistics.mean(steady_torch_tflops_runs),
        "repeats": repeats,
    }


def run_benchmarks():
    if not has_xpu():
        print("XPU not available, skipping SDP benchmark")
        return

    cases = [
        ("self-attn", 14040, 14040, 12, 128),
        ("cross-attn", 14040, 512, 12, 128),
        ("wan-cross-1560x512", 1560, 512, 40, 128),
        ("wan-cross-3600x512", 3600, 512, 40, 128),
        ("wan-self-1560", 1560, 1560, 40, 128),
        ("wan-self-3600", 3600, 3600, 40, 128),
        ("flux-cross-4096x512", 4096, 512, 24, 128),
        ("flux-self-4096", 4096, 4096, 24, 128),
    ]

    for label, q_len, kv_len, heads, dim in cases:
        print(f"\n{label}: q=[1,{q_len},{heads},{dim}] kv=[1,{kv_len},{heads},{dim}]")
        for dtype in (torch.float16, torch.bfloat16):
            result = benchmark_sdp(q_len, kv_len, heads, dim, dtype)
            print(
                f"  {str(dtype).split('.')[-1]} "
                f"cold={result['cold_kernel_ms']:.2f}/{result['cold_torch_ms']:.2f}ms "
                f"steady={result['steady_kernel_ms']:.2f}/{result['steady_torch_ms']:.2f}ms "
                f"std={result['steady_kernel_std_ms']:.2f}/{result['steady_torch_std_ms']:.2f}ms "
                f"speedup={result['steady_speedup']:.2f}x "
                f"kernel={result['steady_kernel_tflops']:.1f}TF torch={result['steady_torch_tflops']:.1f}TF"
            )


if __name__ == "__main__":
    run_benchmarks()
