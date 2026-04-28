import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from test_moe_int4_kernel import quantize_int4, ipex_transform_expert_weights
from custom_esimd_kernels_vllm import (
    moe_forward_full_cutlass_nmajor_int4,
    moe_route_gather_int4,
    moe_silu_mul_int4,
    moe_topk_int4,
    prepare_cutlass_nmajor_int4_weight,
    precompute_moe_route,
)
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2


DEVICE = "xpu"
GROUP_SIZE = 128
WARMUP = 20
RUNS = 50


def topk_from_logits(logits: torch.Tensor, top_k: int):
    topk_weights, topk_ids = moe_topk_int4(logits.contiguous(), top_k, logits.shape[-1], True)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def build_layout_inputs(hidden_size: int, intermediate_size: int, num_experts: int):
    w13 = (torch.randn(num_experts, 2 * intermediate_size, hidden_size) * 0.02).half()
    w2 = (torch.randn(num_experts, hidden_size, intermediate_size) * 0.02).half()

    w13_q_nm = []
    w13_s_nm = []
    for expert in range(num_experts):
        qweight, scale = quantize_int4(w13[expert], GROUP_SIZE)
        w13_q_nm.append(qweight)
        w13_s_nm.append(scale)
    w13_q_nm = torch.stack(w13_q_nm)
    w13_s_nm = torch.stack(w13_s_nm)

    w2_q_nm = []
    w2_s_nm = []
    for expert in range(num_experts):
        qweight, scale = quantize_int4(w2[expert], GROUP_SIZE)
        w2_q_nm.append(qweight)
        w2_s_nm.append(scale)
    w2_q_nm = torch.stack(w2_q_nm)
    w2_s_nm = torch.stack(w2_s_nm)

    w13_q_km, w13_s_km = ipex_transform_expert_weights(
        w13_q_nm, w13_s_nm, num_experts, 2 * intermediate_size, hidden_size // 8, hidden_size // GROUP_SIZE)
    w2_q_km, w2_s_km = ipex_transform_expert_weights(
        w2_q_nm, w2_s_nm, num_experts, hidden_size, intermediate_size // 8, intermediate_size // GROUP_SIZE)

    w13_q_cutlass = prepare_cutlass_nmajor_int4_weight(w13_q_nm.to(DEVICE))
    w2_q_cutlass = prepare_cutlass_nmajor_int4_weight(w2_q_nm.to(DEVICE))

    return {
        "w13_q_nm": w13_q_nm.to(DEVICE),
        "w13_s_nm": w13_s_nm.to(DEVICE),
        "w2_q_nm": w2_q_nm.to(DEVICE),
        "w2_s_nm": w2_s_nm.to(DEVICE),
        "w13_q_km": w13_q_km.to(DEVICE),
        "w13_s_km": w13_s_km.to(DEVICE),
        "w2_q_km": w2_q_km.to(DEVICE),
        "w2_s_km": w2_s_km.to(DEVICE),
        "w13_q_cutlass": w13_q_cutlass,
        "w2_q_cutlass": w2_q_cutlass,
    }


def run_cutlass_nmajor_fullish(
    moe_int4_ops,
    x: torch.Tensor,
    logits: torch.Tensor,
    layout_inputs: dict[str, torch.Tensor],
    shared_gate_up: torch.Tensor,
    shared_down: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    top_k: int,
    num_experts: int,
) -> torch.Tensor:
    del moe_int4_ops
    return moe_forward_full_cutlass_nmajor_int4(
        x, logits,
        layout_inputs["w13_q_cutlass"], layout_inputs["w13_s_nm"],
        layout_inputs["w2_q_cutlass"], layout_inputs["w2_s_nm"],
        shared_gate_up, shared_down, shared_gate_weight,
        top_k, 1, num_experts)


def stamp_timing(timings: dict[str, float], name: str, start: float) -> float:
    torch.xpu.synchronize()
    now = time.perf_counter()
    timings[name] = (now - start) * 1e6
    return now


def profile_cutlass_nmajor_fullish(
    x: torch.Tensor,
    logits: torch.Tensor,
    layout_inputs: dict[str, torch.Tensor],
    shared_gate_up: torch.Tensor,
    shared_down: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    top_k: int,
    num_experts: int,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    torch.xpu.synchronize()
    start = time.perf_counter()

    topk_weights, topk_ids = topk_from_logits(logits, top_k)
    topk_weights = topk_weights.to(x.device)
    topk_ids = topk_ids.to(x.device)
    start = stamp_timing(timings, "topk", start)

    sorted_rows, sorted_weights, rows_per_expert = precompute_moe_route(
        topk_weights, topk_ids, num_experts)
    gemm1_input = x.index_select(0, sorted_rows).contiguous()
    start = stamp_timing(timings, "route", start)

    hidden_size = x.shape[1]
    inter_size = shared_down.shape[1]
    gemm1_output = torch.empty(
        gemm1_input.shape[0], 2 * inter_size, dtype=x.dtype, device=x.device)
    cutlass_grouped_gemm_xe2(
        gemm1_input, layout_inputs["w13_q_cutlass"], layout_inputs["w13_s_nm"], None,
        gemm1_output, rows_per_expert, 2 * inter_size, hidden_size, num_experts,
        True, False)
    start = stamp_timing(timings, "gemm1", start)

    act_output = moe_silu_mul_int4(gemm1_output)
    start = stamp_timing(timings, "activation", start)

    gemm2_output = torch.empty(
        gemm1_input.shape[0], hidden_size, dtype=x.dtype, device=x.device)
    cutlass_grouped_gemm_xe2(
        act_output, layout_inputs["w2_q_cutlass"], layout_inputs["w2_s_nm"], None,
        gemm2_output, rows_per_expert, hidden_size, inter_size, num_experts,
        True, False)
    start = stamp_timing(timings, "gemm2", start)

    routed = moe_route_gather_int4(gemm2_output, sorted_rows, sorted_weights, x.shape[0])
    start = stamp_timing(timings, "gather", start)

    shared_gate = x @ shared_gate_up[:inter_size].t()
    shared_up = x @ shared_gate_up[inter_size:].t()
    shared_act = torch.nn.functional.silu(shared_gate.float()) * shared_up.float()
    shared_out = shared_act.to(x.dtype) @ shared_down.t()
    gate = torch.sigmoid((x @ shared_gate_weight.t()).float()).to(x.dtype)
    routed + shared_out * gate
    stamp_timing(timings, "shared", start)
    timings["total"] = sum(timings.values())
    return timings


def bench_once(fn):
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()

    samples = []
    for _ in range(RUNS):
        torch.xpu.synchronize()
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    from custom_esimd_kernels_vllm import moe_int4_ops

    cfg = {
        "hidden_size": 3072,
        "intermediate_size": 256,
        "shared_intermediate_size": 256,
        "num_experts": 256,
        "top_k": 8,
        "num_shared_experts": 1,
    }

    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    shared_intermediate_size = cfg["shared_intermediate_size"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]
    num_shared_experts = cfg["num_shared_experts"]

    layout_inputs = build_layout_inputs(hidden_size, intermediate_size, num_experts)
    shared_gate_up = (torch.randn(2 * shared_intermediate_size, hidden_size) * 0.02).half().to(DEVICE)
    shared_down = (torch.randn(hidden_size, shared_intermediate_size) * 0.02).half().to(DEVICE)
    shared_gate_weight = (torch.randn(1, hidden_size) * 0.02).half().to(DEVICE)
    dummy_scale = torch.empty(0, dtype=torch.float16, device=DEVICE)

    print("config: Qwen3.5-122B-A10B TP4 moe_forward_full_int4 vs CUTLASS N-major")
    print("bs, path, total_us, (vs_ipex)")

    for batch_size in [1, 4, 8, 16, 24, 32, 48, 64]:
        x = (torch.randn(batch_size, hidden_size) * 0.1).half().to(DEVICE)
        logits = (torch.randn(batch_size, num_experts) * 0.1).half().to(DEVICE)

        def run_k_major():
            return moe_int4_ops.moe_forward_full_int4(
                x, logits,
                layout_inputs["w13_q_km"], layout_inputs["w13_s_km"],
                shared_gate_up, dummy_scale,
                layout_inputs["w2_q_km"], layout_inputs["w2_s_km"],
                shared_down, dummy_scale,
                shared_gate_weight,
                top_k, num_shared_experts, num_experts, False)

        def run_cutlass_auto():
            return run_cutlass_nmajor_fullish(
                moe_int4_ops, x, logits, layout_inputs,
                shared_gate_up, shared_down, shared_gate_weight,
                top_k, num_experts)

        def run_cutlass_tiny_full():
            return moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
                x, logits,
                layout_inputs["w13_q_cutlass"], layout_inputs["w13_s_nm"],
                layout_inputs["w2_q_cutlass"], layout_inputs["w2_s_nm"],
                shared_gate_up, shared_down, shared_gate_weight,
                top_k, num_shared_experts, num_experts)

        def run_cutlass_grouped_gemm():
            # Force grouped GEMM path by going through the profiler stages.
            from custom_esimd_kernels_vllm import (
                moe_forward_routed_cutlass_nmajor_int4,
            )
            topk_weights, topk_ids = topk_from_logits(logits, top_k)
            routed = moe_forward_routed_cutlass_nmajor_int4(
                x, layout_inputs["w13_q_cutlass"], layout_inputs["w13_s_nm"],
                layout_inputs["w2_q_cutlass"], layout_inputs["w2_s_nm"],
                topk_weights, topk_ids, num_experts)
            shared_gate = x @ shared_gate_up[:shared_intermediate_size].t()
            shared_up = x @ shared_gate_up[shared_intermediate_size:].t()
            shared_act = torch.nn.functional.silu(shared_gate.float()) * shared_up.float()
            shared_out = shared_act.to(x.dtype) @ shared_down.t()
            gate = torch.sigmoid((x @ shared_gate_weight.t()).float()).to(x.dtype)
            return routed + shared_out * gate

        ref = run_k_major()
        cut_auto = run_cutlass_auto()
        torch.xpu.synchronize()
        diff_auto = (ref - cut_auto).abs().float().max().item()

        if batch_size >= 4:
            cut_tiny = run_cutlass_tiny_full()
            cut_gg = run_cutlass_grouped_gemm()
            torch.xpu.synchronize()
            tiny_max = (ref - cut_tiny).abs().float().max().item()
            gg_max = (ref - cut_gg).abs().float().max().item()
        else:
            tiny_max = diff_auto
            gg_max = float('nan')

        k_us = bench_once(run_k_major)
        auto_us = bench_once(run_cutlass_auto)
        tiny_us = bench_once(run_cutlass_tiny_full) if batch_size >= 4 else auto_us
        gg_us = bench_once(run_cutlass_grouped_gemm) if batch_size >= 4 else float('nan')

        print(
            f"{batch_size}, ipex_k_major, {k_us:.1f}, 1.00, diff_auto={diff_auto:.2e}")
        print(
            f"{batch_size}, cutlass_auto, {auto_us:.1f}, {auto_us/k_us:.2f}")
        if batch_size >= 4:
            print(
                f"{batch_size}, cutlass_tiny_full, {tiny_us:.1f}, {tiny_us/k_us:.2f}, diff={tiny_max:.2e}")
            print(
                f"{batch_size}, cutlass_grouped_gemm, {gg_us:.1f}, {gg_us/k_us:.2f}, diff={gg_max:.2e}")

        breakdown = profile_cutlass_nmajor_fullish(
            x, logits, layout_inputs, shared_gate_up, shared_down,
            shared_gate_weight, top_k, num_experts)
        print(
            f"bs{batch_size}_breakdown: "
            + ", ".join(f"{name}={value:.1f}" for name, value in breakdown.items()))


if __name__ == "__main__":
    main()
