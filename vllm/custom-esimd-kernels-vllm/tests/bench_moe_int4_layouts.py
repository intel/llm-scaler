import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_moe_int4_kernel import quantize_int4, ipex_transform_expert_weights


DEVICE = "xpu"
GROUP_SIZE = 128
WARMUP = 20
RUNS = 50


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

    return {
        "w13_q_nm": w13_q_nm.to(DEVICE),
        "w13_s_nm": w13_s_nm.to(DEVICE),
        "w2_q_nm": w2_q_nm.to(DEVICE),
        "w2_s_nm": w2_s_nm.to(DEVICE),
        "w13_q_km": w13_q_km.to(DEVICE),
        "w13_s_km": w13_s_km.to(DEVICE),
        "w2_q_km": w2_q_km.to(DEVICE),
        "w2_s_km": w2_s_km.to(DEVICE),
    }


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

    print("config: Qwen3.5-122B-A10B TP4")
    print("bs, k_major_us, n_major_us, n_vs_k")

    for batch_size in [1, 4, 8, 16]:
        x = (torch.randn(batch_size, hidden_size) * 0.1).half().to(DEVICE)
        logits = (torch.randn(batch_size, num_experts) * 0.1).half().to(DEVICE)

        def run_k_major():
            moe_int4_ops.moe_forward_full_int4(
                x, logits,
                layout_inputs["w13_q_km"], layout_inputs["w13_s_km"],
                shared_gate_up, dummy_scale,
                layout_inputs["w2_q_km"], layout_inputs["w2_s_km"],
                shared_down, dummy_scale,
                shared_gate_weight,
                top_k, num_shared_experts, num_experts, False)

        def run_n_major():
            moe_int4_ops.moe_forward_full_int4(
                x, logits,
                layout_inputs["w13_q_nm"], layout_inputs["w13_s_nm"],
                shared_gate_up, dummy_scale,
                layout_inputs["w2_q_nm"], layout_inputs["w2_s_nm"],
                shared_down, dummy_scale,
                shared_gate_weight,
                top_k, num_shared_experts, num_experts, True)

        k_major_us = bench_once(run_k_major)
        n_major_us = bench_once(run_n_major)
        ratio = n_major_us / k_major_us if k_major_us else 0.0
        print(f"{batch_size}, {k_major_us:.1f}, {n_major_us:.1f}, {ratio:.3f}")


if __name__ == "__main__":
    main()