"""Benchmark for esimd_resadd_norm_gemv_int4_pert vs FP8 variant.

Usage:
  python tests/bench_resadd_norm_gemv_int4.py
  python tests/bench_resadd_norm_gemv_int4.py --warmup 50 --repeat 200
"""
import argparse
import ctypes
import os
import torch
from custom_esimd_kernels_vllm import (
    esimd_resadd_norm_gemv_int4_pert,
    esimd_resadd_norm_gemv_fp8_pert,
)

DEVICE = "xpu"
CLIB_PATH = os.environ.get(
    "VLLM_QUANTIZE_Q40_LIB",
    "/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so",
)


def cpu_quantize(weight_fp16, block_size=128):
    N, K = weight_fp16.shape
    weight_f32 = weight_fp16.float().contiguous()
    qweight = torch.zeros(N, K // 8, dtype=torch.int32)
    scale = torch.zeros(N, K // block_size, dtype=torch.float16)
    clib = ctypes.CDLL(CLIB_PATH)
    clib.quantize_q4_0_to_qweight_and_scale.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    clib.quantize_q4_0_to_qweight_and_scale.restype = ctypes.c_size_t
    src = ctypes.cast(weight_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
    qw = ctypes.cast(qweight.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    sc = ctypes.cast(scale.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    clib.quantize_q4_0_to_qweight_and_scale(src, qw, sc, N, K, block_size)
    return qweight, scale


def bench_int4(hidden, residual, norm_weight, qw, sc, output, normed_out, eps, warmup, repeat):
    for _ in range(warmup):
        res = residual.clone()
        esimd_resadd_norm_gemv_int4_pert(hidden, res, norm_weight, qw, sc, output, normed_out, eps)
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        res = residual.clone()
        esimd_resadd_norm_gemv_int4_pert(hidden, res, norm_weight, qw, sc, output, normed_out, eps)
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / repeat * 1000  # us


def bench_fp8(hidden, residual, norm_weight, w_fp8, s_fp8, output, normed_out, eps, warmup, repeat):
    for _ in range(warmup):
        res = residual.clone()
        esimd_resadd_norm_gemv_fp8_pert(hidden, res, norm_weight, w_fp8, s_fp8, output, normed_out, eps)
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        res = residual.clone()
        esimd_resadd_norm_gemv_fp8_pert(hidden, res, norm_weight, w_fp8, s_fp8, output, normed_out, eps)
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / repeat * 1000  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    configs = [
        # (N, K, label)                              K_SPLIT
        (16, 128,  "minimal:  N=16   K=128"),       # ks=1
        (32, 256,  "small:    N=32   K=256"),        # ks=1
        (256, 512, "ks=2:     N=256  K=512"),        # ks=2
        (512, 2048, "ks=4:    N=512  K=2048"),       # ks=4
        (128, 2048, "Qwen3-Next: N=128  K=2048"),    # ks=8
        (128, 4096, "Qwen3.5:    N=128  K=4096"),    # ks=8
        (256, 3072, "Qwen3.5-122B-A10B: N=256  K=3072"),  # ks=4
    ]

    print(f"{'Config':<40} {'INT4 (us)':>10} {'FP8 (us)':>10} {'Ratio':>8}")
    print("-" * 72)

    for N, K, label in configs:
        torch.manual_seed(42)
        eps = 1e-6

        hidden = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
        residual = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
        norm_weight = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

        # INT4 weight
        weight_fp16 = torch.randn(N, K, dtype=torch.float16)
        qw, sc = cpu_quantize(weight_fp16, block_size=128)
        qw = qw.to(DEVICE)
        sc = sc.to(DEVICE)

        # FP8 weight
        weight_xpu = weight_fp16.to(DEVICE)
        fp8_scale_val = weight_xpu.float().abs().max().item() / 448.0
        w_fp8 = (weight_xpu.float() / fp8_scale_val).to(torch.float8_e4m3fn)
        s_fp8 = torch.tensor([fp8_scale_val], dtype=torch.float32, device=DEVICE)

        output_i = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
        normed_i = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
        output_f = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
        normed_f = torch.empty(1, K, dtype=torch.float16, device=DEVICE)

        t_int4 = bench_int4(hidden, residual, norm_weight, qw, sc,
                            output_i, normed_i, eps, args.warmup, args.repeat)
        t_fp8 = bench_fp8(hidden, residual, norm_weight, w_fp8, s_fp8,
                          output_f, normed_f, eps, args.warmup, args.repeat)
        ratio = t_int4 / t_fp8

        print(f"{label:<40} {t_int4:>10.1f} {t_fp8:>10.1f} {ratio:>7.2f}x")

    print()


if __name__ == "__main__":
    main()
