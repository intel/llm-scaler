"""End-to-end routed MoE prototype for CUTLASS-style N-major INT4 weights.

The path exercised here is the plan we want for high-performance N-major:

    GGML/test int32 N-major -> CUTLASS uint8 N-major -> implement_zp -> grouped GEMM

This script performs expert sorting and weighted gather in Python/Torch, while
the two INT4 grouped GEMMs use vllm-xpu-kernels. Shared experts are intentionally
excluded here so the script isolates the routed MoE path.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from custom_esimd_kernels_vllm import (
    moe_forward_routed_cutlass_nmajor_int4,
    moe_forward_tiny_cutlass_nmajor_int4,
    moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared,
    moe_route_gather_int4,
    moe_silu_mul_int4,
    moe_tiny_cutlass_nmajor_int4_down,
    moe_tiny_cutlass_nmajor_int4_up,
    moe_tiny_fp16_shared_finalize,
    moe_tiny_fp16_shared_up,
    precompute_moe_route,
    prepare_cutlass_nmajor_int4_weight,
)

from test_cutlass_nmajor_int4_pack import _decode_cutlass_u4, _to_cutlass_nmajor
from test_moe_int4_kernel import GROUP_SIZE, quantize_int4


DEVICE = "xpu"
WARMUP = 10
RUNS = 30


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


def _topk_from_logits(logits: torch.Tensor, top_k: int):
    probs = F.softmax(logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.half(), topk_ids.to(torch.int32)


def _quantize_experts(weight: torch.Tensor):
    qweights = []
    scales = []
    for expert in range(weight.shape[0]):
        qweight, scale = quantize_int4(weight[expert], GROUP_SIZE)
        qweights.append(qweight)
        scales.append(scale)
    qweight_u4 = _to_cutlass_nmajor(torch.stack(qweights))
    return qweight_u4.contiguous(), torch.stack(scales).contiguous()


def _implement_zp_experts(qweight_u4: torch.Tensor, device: str = DEVICE) -> torch.Tensor:
    return prepare_cutlass_nmajor_int4_weight(qweight_u4.to(device))


def _manual_cutlass_routed_moe(
    hidden_states: torch.Tensor,
    w13_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

    num_rows, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter_size = w2_s4.shape[2] * 2

    del top_k
    sorted_rows, sorted_weights, rows_per_expert = precompute_moe_route(
        topk_weights, topk_ids, num_experts)

    gemm1_input = hidden_states.index_select(0, sorted_rows).contiguous()
    gemm1_output = torch.empty(
        gemm1_input.shape[0], w13_s4.shape[1], dtype=hidden_states.dtype,
        device=hidden_states.device)
    cutlass_grouped_gemm_xe2(
        gemm1_input, w13_s4, w13_scales, None, gemm1_output,
        rows_per_expert, w13_s4.shape[1], hidden_size, num_experts, True, False)

    act_output = moe_silu_mul_int4(gemm1_output)
    gemm2_output = torch.empty(
        gemm1_input.shape[0], hidden_size, dtype=hidden_states.dtype,
        device=hidden_states.device)
    cutlass_grouped_gemm_xe2(
        act_output, w2_s4, w2_scales, None, gemm2_output,
        rows_per_expert, hidden_size, inter_size, num_experts, True, False)

    return moe_route_gather_int4(gemm2_output, sorted_rows, sorted_weights, num_rows)


def _precompute_route(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_rows: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del num_rows
    return precompute_moe_route(topk_weights, topk_ids, num_experts)


def _manual_cutlass_routed_moe_precomputed(
    hidden_states: torch.Tensor,
    w13_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    sorted_rows: torch.Tensor,
    sorted_weights: torch.Tensor,
    rows_per_expert: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    return moe_forward_routed_cutlass_nmajor_int4(
        hidden_states, w13_s4, w13_scales, w2_s4, w2_scales,
        topk_weights=torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device),
        topk_ids=torch.empty(0, dtype=torch.int32, device=hidden_states.device),
        num_experts=num_experts,
        route=(sorted_rows, sorted_weights, rows_per_expert))


def _manual_cutlass_routed_moe_timed(
    hidden_states: torch.Tensor,
    w13_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> dict[str, float]:
    from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

    timings: dict[str, float] = {}

    def stamp(name: str, start: float) -> float:
        _sync()
        now = time.perf_counter()
        timings[name] = (now - start) * 1e6
        return time.perf_counter()

    _sync()
    start = time.perf_counter()
    num_rows, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter_size = w2_s4.shape[2] * 2
    del top_k
    sorted_rows, sorted_weights, rows_per_expert = precompute_moe_route(
        topk_weights, topk_ids, num_experts)
    gemm1_input = hidden_states.index_select(0, sorted_rows).contiguous()
    start = stamp("prologue", start)

    gemm1_output = torch.empty(
        gemm1_input.shape[0], w13_s4.shape[1], dtype=hidden_states.dtype,
        device=hidden_states.device)
    cutlass_grouped_gemm_xe2(
        gemm1_input, w13_s4, w13_scales, None, gemm1_output,
        rows_per_expert, w13_s4.shape[1], hidden_size, num_experts, True, False)
    start = stamp("gemm1", start)

    act_output = moe_silu_mul_int4(gemm1_output)
    start = stamp("activation", start)

    gemm2_output = torch.empty(
        gemm1_input.shape[0], hidden_size, dtype=hidden_states.dtype,
        device=hidden_states.device)
    cutlass_grouped_gemm_xe2(
        act_output, w2_s4, w2_scales, None, gemm2_output,
        rows_per_expert, hidden_size, inter_size, num_experts, True, False)
    start = stamp("gemm2", start)

    moe_route_gather_int4(gemm2_output, sorted_rows, sorted_weights, num_rows)
    stamp("gather", start)
    timings["total"] = sum(timings.values())
    return timings


def _ref_routed_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    num_rows = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    hidden_size = hidden_states.shape[1]
    inter_size = w2.shape[2]
    output = torch.zeros(num_rows, hidden_size, dtype=torch.float32)
    hidden_f = hidden_states.float().cpu()

    for row in range(num_rows):
        acc = torch.zeros(hidden_size, dtype=torch.float32)
        for topk_index in range(top_k):
            expert = int(topk_ids[row, topk_index].cpu().item())
            route_weight = float(topk_weights[row, topk_index].cpu().item())
            gate = hidden_f[row] @ w13[expert, :inter_size].float().t()
            up = hidden_f[row] @ w13[expert, inter_size:].float().t()
            act = F.silu(gate) * up
            down = act @ w2[expert].float().t()
            acc += route_weight * down
        output[row] = acc
    return output.half()


def _ref_full_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    shared_gate_up: torch.Tensor,
    shared_down: torch.Tensor,
    shared_gate_weight: torch.Tensor,
) -> torch.Tensor:
    routed = _ref_routed_moe(hidden_states, topk_weights, topk_ids, w13, w2).float()
    hidden_f = hidden_states.float().cpu()
    shared_inter = shared_down.shape[-1]
    gate = hidden_f @ shared_gate_up[:shared_inter].float().t()
    up = hidden_f @ shared_gate_up[shared_inter:].float().t()
    shared_act = F.silu(gate) * up
    shared_out = shared_act @ shared_down.float().t()
    shared_gate = torch.sigmoid(hidden_f @ shared_gate_weight.float().t())
    return (routed + shared_out * shared_gate).half()


def check_correctness() -> None:
    torch.manual_seed(2027)
    num_rows = 3
    hidden_size = 128
    inter_size = 128
    num_experts = 8
    top_k = 3

    hidden_states = (torch.randn(num_rows, hidden_size) * 0.1).half()
    logits = (torch.randn(num_rows, num_experts) * 0.1).half()
    topk_weights, topk_ids = _topk_from_logits(logits, top_k)

    w13_fp = (torch.randn(num_experts, 2 * inter_size, hidden_size) * 0.02).half()
    w2_fp = (torch.randn(num_experts, hidden_size, inter_size) * 0.02).half()
    w13_u4, w13_scales = _quantize_experts(w13_fp)
    w2_u4, w2_scales = _quantize_experts(w2_fp)
    w13_dq = _decode_cutlass_u4(w13_u4, w13_scales, GROUP_SIZE)
    w2_dq = _decode_cutlass_u4(w2_u4, w2_scales, GROUP_SIZE)
    w13_s4 = _implement_zp_experts(w13_u4)
    w2_s4 = _implement_zp_experts(w2_u4)

    got = _manual_cutlass_routed_moe(
        hidden_states.to(DEVICE),
        w13_s4, w13_scales.to(DEVICE),
        w2_s4, w2_scales.to(DEVICE),
        topk_weights.to(DEVICE), topk_ids.to(DEVICE), num_experts)
    _sync()

    tiny = moe_forward_tiny_cutlass_nmajor_int4(
        hidden_states[:1].to(DEVICE),
        w13_s4, w13_scales.to(DEVICE),
        w2_s4, w2_scales.to(DEVICE),
        topk_weights[:1].to(DEVICE), topk_ids[:1].to(DEVICE))
    _sync()

    shared_gate_up = (torch.randn(2 * inter_size, hidden_size) * 0.02).half()
    shared_down = (torch.randn(hidden_size, inter_size) * 0.02).half()
    shared_gate_weight = (torch.randn(1, hidden_size) * 0.02).half()
    tiny_full = moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
        hidden_states[:1].to(DEVICE),
        w13_s4, w13_scales.to(DEVICE),
        w2_s4, w2_scales.to(DEVICE),
        topk_weights[:1].to(DEVICE), topk_ids[:1].to(DEVICE),
        shared_gate_up.to(DEVICE), shared_down.to(DEVICE),
        shared_gate_weight.to(DEVICE), 1)
    _sync()

    ref = _ref_routed_moe(hidden_states, topk_weights, topk_ids, w13_dq, w2_dq)
    ref_full = _ref_full_moe(
        hidden_states[:1], topk_weights[:1], topk_ids[:1], w13_dq, w2_dq,
        shared_gate_up, shared_down, shared_gate_weight)
    diff = (got.cpu() - ref).abs().float()
    tiny_diff = (tiny.cpu() - ref[:1]).abs().float()
    tiny_full_diff = (tiny_full.cpu() - ref_full).abs().float()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(
        f"correctness: max={max_diff:.6g} mean={mean_diff:.6g} "
        f"tiny_max={tiny_diff.max().item():.6g} tiny_mean={tiny_diff.mean().item():.6g} "
        f"tiny_full_max={tiny_full_diff.max().item():.6g} "
        f"tiny_full_mean={tiny_full_diff.mean().item():.6g}")
    assert max_diff < 5e-3
    assert tiny_diff.max().item() < 5e-3
    assert tiny_full_diff.max().item() < 5e-3


def _make_random_u4_weight(num_experts: int, n: int, k: int):
    qweight = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8, device=DEVICE)
    scales = torch.rand(num_experts, n, k // GROUP_SIZE, dtype=torch.float16, device=DEVICE) * 0.01
    return qweight.contiguous(), scales.contiguous()


def bench_qwen_routed_moe() -> None:
    hidden_size = 3072
    inter_size = 256
    num_experts = 256
    top_k = 8
    num_shared_experts = 1

    w13_u4, w13_scales = _make_random_u4_weight(num_experts, 2 * inter_size, hidden_size)
    w2_u4, w2_scales = _make_random_u4_weight(num_experts, hidden_size, inter_size)
    shared_gate_up = (torch.randn(2 * inter_size, hidden_size, device=DEVICE) * 0.02).half()
    shared_down = (torch.randn(hidden_size, inter_size, device=DEVICE) * 0.02).half()
    shared_gate_weight = (torch.randn(num_shared_experts, hidden_size, device=DEVICE) * 0.02).half()

    print("shape: Qwen3.5-122B-A10B TP4 routed-only manual sort + cutlass grouped GEMM")
    print("bs,routed_moe_us,public_api_us,precomputed_route_us,tiny_m_us,tiny_full_us,tiny_up_us,tiny_down_us,shared_up_us,shared_finalize_us")

    for batch_size in [1, 4, 8, 16]:
        hidden_states = (torch.randn(batch_size, hidden_size, device=DEVICE) * 0.1).half()
        logits = (torch.randn(batch_size, num_experts, device=DEVICE) * 0.1).half()
        topk_weights, topk_ids = _topk_from_logits(logits, top_k)
        topk_weights = topk_weights.to(DEVICE)
        topk_ids = topk_ids.to(DEVICE)
        _manual_cutlass_routed_moe(
            hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
            topk_weights, topk_ids, num_experts)
        _sync()

        def run():
            _manual_cutlass_routed_moe(
                hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                topk_weights, topk_ids, num_experts)

        routed_us = _bench(run)

        def run_public_api():
            moe_forward_routed_cutlass_nmajor_int4(
                hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                topk_weights, topk_ids, num_experts)

        public_api_us = _bench(run_public_api)
        sorted_rows, sorted_weights, rows_per_expert = _precompute_route(
            topk_weights, topk_ids, batch_size, num_experts)

        def run_precomputed():
            _manual_cutlass_routed_moe_precomputed(
                hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                sorted_rows, sorted_weights, rows_per_expert, num_experts)

        precomputed_us = _bench(run_precomputed)
        tiny_us = 0.0
        tiny_full_us = 0.0
        tiny_up_us = 0.0
        tiny_down_us = 0.0
        shared_up_us = 0.0
        shared_finalize_us = 0.0
        if batch_size == 1:
            intermediates = moe_tiny_cutlass_nmajor_int4_up(
                hidden_states, w13_u4, w13_scales, topk_ids)
            routed_output = moe_forward_tiny_cutlass_nmajor_int4(
                hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                topk_weights, topk_ids)
            shared_intermediates = moe_tiny_fp16_shared_up(
                hidden_states, shared_gate_up, num_shared_experts)
            _sync()

            def run_tiny():
                moe_forward_tiny_cutlass_nmajor_int4(
                    hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                    topk_weights, topk_ids)

            def run_tiny_up():
                moe_tiny_cutlass_nmajor_int4_up(
                    hidden_states, w13_u4, w13_scales, topk_ids)

            def run_tiny_down():
                moe_tiny_cutlass_nmajor_int4_down(
                    intermediates, w2_u4, w2_scales, topk_weights, topk_ids)

            def run_tiny_full():
                moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
                    hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                    topk_weights, topk_ids, shared_gate_up, shared_down,
                    shared_gate_weight, num_shared_experts)

            def run_shared_up():
                moe_tiny_fp16_shared_up(
                    hidden_states, shared_gate_up, num_shared_experts)

            def run_shared_finalize():
                moe_tiny_fp16_shared_finalize(
                    hidden_states, shared_intermediates, routed_output,
                    shared_down, shared_gate_weight, num_shared_experts)

            tiny_us = _bench(run_tiny)
            tiny_full_us = _bench(run_tiny_full)
            tiny_up_us = _bench(run_tiny_up)
            tiny_down_us = _bench(run_tiny_down)
            shared_up_us = _bench(run_shared_up)
            shared_finalize_us = _bench(run_shared_finalize)
        print(
            f"{batch_size},{routed_us:.1f},{public_api_us:.1f},{precomputed_us:.1f},"
            f"{tiny_us:.1f},{tiny_full_us:.1f},{tiny_up_us:.1f},{tiny_down_us:.1f},"
            f"{shared_up_us:.1f},{shared_finalize_us:.1f}")

        if batch_size in (1, 8):
            breakdown = _manual_cutlass_routed_moe_timed(
                hidden_states, w13_u4, w13_scales, w2_u4, w2_scales,
                topk_weights, topk_ids, num_experts)
            print(
                "  breakdown_us: " + ", ".join(
                    f"{name}={value:.1f}" for name, value in breakdown.items()))


def main() -> None:
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required for xpu_fused_moe")
    check_correctness()
    bench_qwen_routed_moe()


if __name__ == "__main__":
    main()