"""Decode microbenchmark for exact 128x128 block FP8 GEMV.

Run on one XPU:
    ZE_AFFINITY_MASK=0 python tests/bench_fp8_blockscale_decode.py
"""

import json
import statistics
import time

import torch

import custom_esimd_kernels_vllm as esimd


SHAPES = (
    ("attn_qkv", 7168, 5120),
    ("attn_o", 5120, 3072),
    ("mlp_gate_up", 17408, 5120),
    ("mlp_down", 5120, 8704),
)


def timed_ms(fn, iterations=80):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p10_ms": sorted(samples)[iterations // 10],
        "p90_ms": sorted(samples)[iterations * 9 // 10],
    }


def main():
    torch.manual_seed(20260731)
    results = []
    for name, n, k in SHAPES:
        x = (torch.randn((1, k), device="xpu") * 0.25).half()
        weight = (torch.randn((n, k), device="xpu") * 0.2).to(
            torch.float8_e4m3fn
        )
        block_scale = torch.ones(
            (n // 128, k // 128), device="xpu", dtype=torch.float32
        )
        tensor_scale = torch.ones(1, device="xpu", dtype=torch.float32)
        block_out = torch.empty((1, n), device="xpu", dtype=torch.float16)
        tensor_out = torch.empty_like(block_out)

        def block():
            esimd.esimd_gemm_fp8_blockscale(
                x, weight, block_scale, block_out, 128, 128
            )

        def tensor():
            esimd.esimd_gemv_fp8_pert(x, weight, tensor_scale, tensor_out)

        for _ in range(20):
            block()
            tensor()
        torch.xpu.synchronize()

        # Alternate measurement order to reduce clock/thermal bias.
        block_samples = []
        tensor_samples = []
        for iteration in range(80):
            order = (
                (("block", block), ("tensor", tensor))
                if iteration % 2 == 0
                else (("tensor", tensor), ("block", block))
            )
            for kind, fn in order:
                start = time.perf_counter()
                fn()
                torch.xpu.synchronize()
                elapsed = (time.perf_counter() - start) * 1e3
                (block_samples if kind == "block" else tensor_samples).append(
                    elapsed
                )

        result = {
            "name": name,
            "n": n,
            "k": k,
            "block_median_ms": statistics.median(block_samples),
            "tensor_median_ms": statistics.median(tensor_samples),
            "block_over_tensor": statistics.median(block_samples)
            / statistics.median(tensor_samples),
            "mean_abs_output_diff": (
                block_out.float() - tensor_out.float()
            ).abs().mean().item(),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del x, weight, block_scale, tensor_scale, block_out, tensor_out
        torch.xpu.empty_cache()

    print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
