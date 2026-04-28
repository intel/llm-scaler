"""Prototype the new CUTLASS-style N-major INT4 grouped GEMM path.

This script validates the calling convention for
``vllm_xpu_kernels.fused_moe_interface.cutlass_grouped_gemm_xe2`` and measures
its pure GEMM latency for Qwen MoE decode-like shapes.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_cutlass_nmajor_int4_pack import _decode_cutlass_s4, _to_cutlass_nmajor
from test_moe_int4_kernel import GROUP_SIZE, quantize_int4


DEVICE = "xpu"
WARMUP = 20
RUNS = 50


def _sync() -> None:
    torch.xpu.synchronize()


def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _sync()

    samples = []
    for _ in range(RUNS):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def _random_rows_per_expert(batch_size: int, top_k: int, num_experts: int) -> torch.Tensor:
    topk_ids = torch.randint(0, num_experts, (batch_size, top_k), dtype=torch.int64)
    return torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32)


def _make_random_s4_weight(num_experts: int, n: int, k: int):
    qweight_s4 = torch.randint(
        0, 256, (num_experts, n, k // 2), dtype=torch.uint8, device=DEVICE)
    scales = torch.rand(
        (num_experts, n, k // GROUP_SIZE), dtype=torch.float16, device=DEVICE) * 0.01
    return qweight_s4.contiguous(), scales.contiguous()


def check_grouped_gemm_correctness() -> None:
    from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2, implement_zp

    torch.manual_seed(2026)
    num_experts = 4
    n = 64
    k = 128
    rows = [2, 1, 3, 0]
    total_rows = sum(rows)

    input_a = torch.randn(total_rows, k, dtype=torch.float16, device=DEVICE)
    qweights = []
    scales = []
    for _ in range(num_experts):
        weight = (torch.randn(n, k) * 0.02).half()
        qweight, scale = quantize_int4(weight, GROUP_SIZE)
        qweights.append(qweight)
        scales.append(scale)

    qweight_i32 = torch.stack(qweights)
    scale = torch.stack(scales)
    qweight_u4 = _to_cutlass_nmajor(qweight_i32)
    qweight_s4 = torch.empty_like(qweight_u4, device=DEVICE)
    for expert in range(num_experts):
        qweight_s4[expert] = implement_zp(qweight_u4[expert].to(DEVICE))

    scale_xpu = scale.to(DEVICE)
    output = torch.empty(total_rows, n, dtype=torch.float16, device=DEVICE)
    rows_per_expert = torch.tensor(rows, dtype=torch.int32, device=DEVICE)
    cutlass_grouped_gemm_xe2(
        input_a, qweight_s4, scale_xpu, None, output,
        rows_per_expert, n, k, num_experts, True, False)
    _sync()

    dequant_weight = _decode_cutlass_s4(qweight_s4.cpu(), scale, GROUP_SIZE).to(DEVICE)
    ref = torch.empty_like(output)
    offset = 0
    for expert, num_rows in enumerate(rows):
        if num_rows:
            ref[offset:offset + num_rows] = (
                input_a[offset:offset + num_rows].float()
                @ dequant_weight[expert].float().t()).half()
        offset += num_rows
    _sync()

    diff = (output - ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"correctness: max={max_diff:.6g} mean={mean_diff:.6g}")
    assert max_diff < 5e-4


def bench_qwen_decode_gemm() -> None:
    from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

    hidden_size = 3072
    intermediate_size = 256
    num_experts = 256
    top_k = 8

    print("shape: Qwen3.5-122B-A10B TP4 pure grouped GEMM")
    print("bs,total_rows,w13_us,w2_us,total_us")

    w13, w13_scales = _make_random_s4_weight(
        num_experts, 2 * intermediate_size, hidden_size)
    w2, w2_scales = _make_random_s4_weight(
        num_experts, hidden_size, intermediate_size)

    for batch_size in [1, 4, 8, 16]:
        rows_cpu = _random_rows_per_expert(batch_size, top_k, num_experts)
        rows = rows_cpu.to(DEVICE)
        total_rows = int(rows_cpu.sum().item())
        a1 = torch.randn(total_rows, hidden_size, dtype=torch.float16, device=DEVICE)
        o1 = torch.empty(total_rows, 2 * intermediate_size, dtype=torch.float16, device=DEVICE)
        a2 = torch.randn(total_rows, intermediate_size, dtype=torch.float16, device=DEVICE)
        o2 = torch.empty(total_rows, hidden_size, dtype=torch.float16, device=DEVICE)

        def run_w13():
            cutlass_grouped_gemm_xe2(
                a1, w13, w13_scales, None, o1,
                rows, 2 * intermediate_size, hidden_size, num_experts, True, False)

        def run_w2():
            cutlass_grouped_gemm_xe2(
                a2, w2, w2_scales, None, o2,
                rows, hidden_size, intermediate_size, num_experts, True, False)

        w13_us = _bench(run_w13)
        w2_us = _bench(run_w2)
        print(f"{batch_size},{total_rows},{w13_us:.1f},{w2_us:.1f},{w13_us + w2_us:.1f}")


def main() -> None:
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required for cutlass_grouped_gemm_xe2")
    check_grouped_gemm_correctness()
    bench_qwen_decode_gemm()


if __name__ == "__main__":
    main()