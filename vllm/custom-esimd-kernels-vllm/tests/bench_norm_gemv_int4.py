"""Benchmark for esimd_norm_gemv_int4_pert — fused RMSNormGated + INT4 GEMV.

Compares:
  1. Fused kernel: esimd_norm_gemv_int4_pert (single submit)
  2. Separate path: PyTorch norm+gate + IPEX INT4 WOQ linear

Usage:
  python tests/bench_norm_gemv_int4.py
  python tests/bench_norm_gemv_int4.py --warmup 50 --repeat 200
"""
import argparse
import ctypes
import os
import torch
import intel_extension_for_pytorch as ipex
from custom_esimd_kernels_vllm import esimd_norm_gemv_int4_pert, esimd_norm_gemv_fp8_pert

DEVICE = "xpu"
QK4_GROUP_SIZE = 128
QK4_PACK_FACTOR = 8
CLIB_PATH = os.environ.get(
    "VLLM_QUANTIZE_Q40_LIB",
    "/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so",
)


def cpu_quantize(weight_fp16, block_size=128):
    """Quantize fp16 weight to INT4 using CPU C library."""
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


def bench_fused(x, z, norm_weight, qw, sc, output, HV, V, eps, warmup, repeat):
    for _ in range(warmup):
        esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, output, HV, V, eps)
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, output, HV, V, eps)
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / repeat * 1000  # us


def bench_fp8(x, z, norm_weight, w_fp8, s_fp8, output, HV, V, eps, warmup, repeat):
    for _ in range(warmup):
        esimd_norm_gemv_fp8_pert(x, z, norm_weight, w_fp8, s_fp8, output, HV, V, eps)
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        esimd_norm_gemv_fp8_pert(x, z, norm_weight, w_fp8, s_fp8, output, HV, V, eps)
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / repeat * 1000  # us


def make_ipex_woq_linear(qw, sc, N):
    """Create IPEX INT4 WOQ linear from quantized weights (mirrors sym_int4.py)."""
    # IPEX expects transposed layout: [K//8, N] and [K//128, N]
    qw_t = qw.t().contiguous()
    sc_t = sc.t().contiguous()
    lowp_mode = ipex.quantization.WoqLowpMode.INT8
    weight_dtype = ipex.quantization.WoqWeightDtype.INT4
    act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK
    qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
        weight_dtype=weight_dtype,
        lowp_mode=lowp_mode,
        act_quant_mode=act_quant_mode,
        group_size=QK4_GROUP_SIZE,
    )
    return ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
        qw_t, sc_t,
        torch.tensor([8], device=qw_t.device, dtype=torch.int8),
        qw_t.size(0),  # K // 8
        N,
        qconfig=qconfig,
        g_idx=None,
        bias=None,
        group_size=QK4_GROUP_SIZE,
        quant_method=0,
    )


def bench_separate(x, z, norm_weight, ipex_linear, output, HV, V, K, N, eps, warmup, repeat):
    """Separate path: PyTorch norm+gate + IPEX INT4 WOQ linear."""
    nw = norm_weight.float()

    def run():
        x_f = x.float()
        z_f = z.float()
        parts = []
        for h in range(HV):
            xh = x_f[h]
            zh = z_f[h]
            inv_rms = torch.rsqrt(xh.pow(2).mean() + eps)
            normed = xh * inv_rms * nw
            silu_z = zh * torch.sigmoid(zh)
            parts.append((normed * silu_z).half())
        normed_flat = torch.cat(parts).reshape(1, K)
        output[:] = ipex_linear(normed_flat)

    for _ in range(warmup):
        run()
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / repeat * 1000  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    configs = [
        # (HV, V, N, label)
        (1,  128, 16,   "minimal: HV=1  N=16"),
        (4,  128, 32,   "small:   HV=4  N=32"),
        (8,  128, 256,  "mid:     HV=8  N=256"),
        (8,  128, 2048, "Qwen3-Next-80B: HV=8  N=2048"),
        (8,  128, 3072, "Qwen3.5-122B-A10B: HV=8  N=3072"),
    ]

    print(f"{'Config':<40} {'INT4 (us)':>10} {'FP8 (us)':>10} {'Ratio':>8} {'Separate (us)':>14} {'Speedup':>8}")
    print("-" * 94)

    for HV, V, N, label in configs:
        torch.manual_seed(42)
        K = HV * V
        eps = 1e-6

        x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
        z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
        norm_weight = torch.randn(V, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

        weight_fp16 = torch.randn(N, K, dtype=torch.float16)
        qw, sc = cpu_quantize(weight_fp16, block_size=128)
        qw = qw.to(DEVICE)
        sc = sc.to(DEVICE)
        ipex_linear = make_ipex_woq_linear(qw, sc, N)

        # FP8 weight
        weight_xpu = weight_fp16.to(DEVICE)
        fp8_scale_val = weight_xpu.float().abs().max().item() / 448.0
        w_fp8 = (weight_xpu.float() / fp8_scale_val).to(torch.float8_e4m3fn)
        s_fp8 = torch.tensor([fp8_scale_val], dtype=torch.float32, device=DEVICE)

        output_i = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
        output_f = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
        output_s = torch.empty(1, N, dtype=torch.float16, device=DEVICE)

        t_int4 = bench_fused(x, z, norm_weight, qw, sc, output_i, HV, V, eps,
                             args.warmup, args.repeat)
        t_fp8 = bench_fp8(x, z, norm_weight, w_fp8, s_fp8, output_f, HV, V, eps,
                          args.warmup, args.repeat)
        t_sep = bench_separate(x, z, norm_weight, ipex_linear, output_s, HV, V, K, N, eps,
                               args.warmup, args.repeat)
        ratio = t_int4 / t_fp8
        speedup = t_sep / t_int4

        print(f"{label:<40} {t_int4:>10.1f} {t_fp8:>10.1f} {ratio:>7.2f}x {t_sep:>14.1f} {speedup:>7.2f}x")

    print()


if __name__ == "__main__":
    main()
