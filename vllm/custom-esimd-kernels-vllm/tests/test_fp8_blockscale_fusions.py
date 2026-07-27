"""Correctness and microbenchmark for block-scaled GDN decode fusions."""

import statistics
import sys
import time
from pathlib import Path

import torch

import custom_esimd_kernels_vllm as esimd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fp8_blockscale_gemm import dequant_weight, quantize_weight_block


def error_stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    out_f = out.float().flatten()
    ref_f = ref.float().flatten()
    rel = ((out_f - ref_f).abs().mean() / ref_f.abs().mean().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(out_f, ref_f, dim=0).item()
    return rel, cos


def timed_ms(fn, warmup: int = 20, iterations: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def test_fp8_blockscale_fusions() -> None:
    torch.manual_seed(20260713)
    dev = "xpu"

    # Qwen3.6 TP2-like GDN input projections.
    k, n0, n1 = 2048, 3072, 64
    x = (torch.randn(1, k, device=dev) * 0.25).half()
    w0, s0 = quantize_weight_block(torch.randn(n0, k, device=dev) * 0.2)
    w1, s1 = quantize_weight_block(torch.randn(n1, k, device=dev) * 0.2)
    o0 = torch.empty(1, n0, dtype=torch.float16, device=dev)
    o1 = torch.empty(1, n1, dtype=torch.float16, device=dev)
    esimd.esimd_gemv_fp8_blockscale_fused2(x, w0, s0, o0, w1, s1, o1)
    torch.xpu.synchronize()
    ref0 = x.float() @ dequant_weight(w0, s0).t()
    ref1 = x.float() @ dequant_weight(w1, s1).t()
    rel0, cos0 = error_stats(o0, ref0)
    rel1, cos1 = error_stats(o1, ref1)
    assert rel0 < 5e-3 and cos0 > 0.99999, (rel0, cos0)
    assert rel1 < 5e-3 and cos1 > 0.99999, (rel1, cos1)

    def separate_input_proj():
        esimd.esimd_gemm_fp8_blockscale(x, w0, s0, o0)
        esimd.esimd_gemm_fp8_blockscale(x, w1, s1, o1)

    def fused_input_proj():
        esimd.esimd_gemv_fp8_blockscale_fused2(x, w0, s0, o0, w1, s1, o1)

    separate_ms = timed_ms(separate_input_proj)
    fused_ms = timed_ms(fused_input_proj)

    w1_fp16 = (torch.randn(n1, k, device=dev) * 0.2).half()
    esimd.esimd_gemv_fp8_blockscale_fp16_fused2(
        x, w0, s0, o0, w1_fp16, o1
    )
    torch.xpu.synchronize()
    mixed_ref0 = x.float() @ dequant_weight(w0, s0).t()
    mixed_ref1 = x.float() @ w1_fp16.float().t()
    mixed_rel0, mixed_cos0 = error_stats(o0, mixed_ref0)
    mixed_rel1, mixed_cos1 = error_stats(o1, mixed_ref1)
    assert mixed_rel0 < 5e-3 and mixed_cos0 > 0.99999, (mixed_rel0, mixed_cos0)
    assert mixed_rel1 < 5e-3 and mixed_cos1 > 0.99999, (mixed_rel1, mixed_cos1)

    # Qwen3.6 TP2-like GDN gated norm + row-parallel output projection.
    hv, v, n = 8, 128, 2048
    core = (torch.randn(hv, v, device=dev) * 0.3).half()
    gate = (torch.randn(hv, v, device=dev) * 0.3).half()
    norm_w = (torch.randn(v, device=dev) * 0.1 + 1.0).half()
    w_out, s_out = quantize_weight_block(torch.randn(n, hv * v, device=dev) * 0.2)
    out = torch.empty(1, n, dtype=torch.float16, device=dev)
    eps = 1e-6
    esimd.esimd_norm_gemv_fp8_blockscale(
        core, gate, norm_w, w_out, s_out, out, hv, v, eps
    )
    torch.xpu.synchronize()
    core_f = core.float()
    normed = core_f * torch.rsqrt(core_f.square().mean(-1, keepdim=True) + eps)
    normed *= norm_w.float()
    normed *= torch.nn.functional.silu(gate.float())
    ref_out = normed.flatten().unsqueeze(0) @ dequant_weight(w_out, s_out).t()
    rel_out, cos_out = error_stats(out, ref_out)
    assert rel_out < 5e-3 and cos_out > 0.99999, (rel_out, cos_out)

    print(f"dual_gemv: rel=({rel0:.3e},{rel1:.3e}) cos=({cos0:.8f},{cos1:.8f})")
    print(f"dual_gemv_ms: separate={separate_ms:.4f} fused={fused_ms:.4f}")
    print(
        f"mixed_gemv: rel=({mixed_rel0:.3e},{mixed_rel1:.3e}) "
        f"cos=({mixed_cos0:.8f},{mixed_cos1:.8f})"
    )
    print(f"norm_gemv: rel={rel_out:.3e} cos={cos_out:.8f}")
    print("RESULT: ALL OK")


if __name__ == "__main__":
    test_fp8_blockscale_fusions()
