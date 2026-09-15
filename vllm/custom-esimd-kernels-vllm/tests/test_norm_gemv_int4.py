"""Unit tests for esimd_norm_gemv_int4_pert — fused RMSNormGated + INT4 GEMV."""
import ctypes
import os
import sysconfig
from pathlib import Path

import pytest
import torch
from custom_esimd_kernels_vllm import (
    esimd_norm_gemv_fp8_pert,
    esimd_norm_gemv_int4_pert,
    esimd_norm_gemv_int4_sigmoid,
)

DEVICE = "xpu"
CLIB_PATH = os.environ.get(
    "VLLM_QUANTIZE_Q40_LIB",
    str(Path(sysconfig.get_path("platlib")) / "vllm_int4_for_multi_arc.so"),
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


def pack_int4_reference(qvalues):
    """Pack unsigned INT4 values in the exact low-to-high nibble order."""
    assert qvalues.dim() == 2 and qvalues.shape[1] % 8 == 0
    qvalues_i64 = qvalues.to(torch.int64)
    assert torch.all((0 <= qvalues_i64) & (qvalues_i64 <= 15)).item()
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = torch.sum(qvalues_i64.reshape(qvalues.shape[0], -1, 8) << shifts,
                      dim=-1)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32)


def dequantize_int4_reference_fp32(qweight, scale, block_size=128):
    """Independent FP32 reference for (nibble - 8) * per-block scale."""
    assert qweight.dim() == 2 and scale.dim() == 2
    assert block_size % 8 == 0
    words = qweight.cpu().to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64) * 4
    qvalues = ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(
        qweight.shape[0], -1)
    scales = scale.cpu().float().repeat_interleave(block_size, dim=1)
    assert qvalues.shape == scales.shape
    return (qvalues.float() - 8.0) * scales


def ref_norm_gemv_fp16(
    x, z, norm_weight, weight_fp16, HV, V, eps, gate="silu"
):
    """Reference: RMSNormGated + fp16 matmul on CPU (no quantization).

    1. Per-head RMSNorm: normed = x * rsqrt(mean(x^2) + eps) * weight
    2. Gate: normed *= SiLU(z) or sigmoid(z)
    3. FP16 matmul: output = normed_flat @ weight^T
    """
    K = HV * V

    x_f = x.cpu().float().reshape(HV, V)
    z_f = z.cpu().float().reshape(HV, V)
    nw = norm_weight.cpu().float()

    # Step 1+2: per-head norm + gate
    normed_flat = torch.zeros(1, K)
    for h in range(HV):
        xh = x_f[h]
        zh = z_f[h]

        # RMSNorm
        mean_sq = (xh * xh).mean()
        inv_rms = torch.rsqrt(mean_sq + eps)
        normed = xh * inv_rms * nw

        if gate == "sigmoid":
            gate_z = torch.sigmoid(zh)
        else:
            gate_z = zh * torch.sigmoid(zh)
        normed = normed * gate_z

        normed_flat[0, h * V:(h + 1) * V] = normed

    # Step 3: fp16 matmul
    result = normed_flat @ weight_fp16.cpu().float().T

    return result, normed_flat


@pytest.mark.parametrize("HV", [3, 5, 6, 7, 9])
@pytest.mark.parametrize("gate", ["silu", "sigmoid"])
def test_norm_gemv_non_divisible_hv_covers_tail_heads(HV, gate):
    """Small-N split selection must not drop a non-divisible tail head."""
    V, N = 128, 12
    K = HV * V
    eps = 1e-6

    # Only the final head contributes. The old threshold-only dispatcher chose
    # K_SPLIT=2/4/8 and silently omitted this head for every HV in the matrix.
    x_cpu = torch.zeros(HV, V, dtype=torch.float16)
    z_cpu = torch.zeros(HV, V, dtype=torch.float16)
    x_cpu[-1].fill_(1.0)
    z_cpu[-1].fill_(2.0)
    norm_weight_cpu = torch.ones(V, dtype=torch.float16)

    qvalues = torch.full((N, K), 8, dtype=torch.int32)
    qvalues[:, -V:] = 9
    qweight_cpu = pack_int4_reference(qvalues)
    scale_cpu = torch.full((N, K // 128), 0.125, dtype=torch.float16)
    dequant_weight = dequantize_int4_reference_fp32(
        qweight_cpu, scale_cpu)

    x_fp32 = x_cpu.float()
    z_fp32 = z_cpu.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    normalized = x_fp32 * inv_rms * norm_weight_cpu.float()
    gate_fp32 = (torch.sigmoid(z_fp32) if gate == "sigmoid"
                 else torch.nn.functional.silu(z_fp32))
    reference = (normalized * gate_fp32).reshape(1, K) @ dequant_weight.T

    x = x_cpu.to(DEVICE)
    z = z_cpu.to(DEVICE)
    norm_weight = norm_weight_cpu.to(DEVICE)
    qweight = qweight_cpu.to(DEVICE)
    scale = scale_cpu.to(DEVICE)
    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    op = (esimd_norm_gemv_int4_sigmoid
          if gate == "sigmoid" else esimd_norm_gemv_int4_pert)
    op(x, z, norm_weight, qweight, scale, output, HV, V, eps)
    torch.xpu.synchronize()

    assert reference.abs().min().item() > 1.0
    torch.testing.assert_close(
        output.cpu().float(), reference, rtol=0.01, atol=0.5)


@pytest.mark.parametrize("HV,V,N", [
    (1, 128, 16),       # minimal: single head
    (4, 128, 32),       # small multi-head
    (8, 128, 256),      # typical GDN: 8 value heads, out_proj to 256
    (8, 128, 2048),     # Qwen3-Next-80B: HV=8, V=128, out_proj N=2048
    (16, 128, 4096),    # larger model
])
def test_norm_gemv_correctness(HV, V, N):
    """Fused kernel should match step-by-step reference."""
    torch.manual_seed(42)
    eps = 1e-6
    K = HV * V

    x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(V, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    # Quantize out_proj weight on CPU, then move to XPU
    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)

    # Fused kernel
    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, output, HV, V, eps)
    torch.xpu.synchronize()

    # Reference (fp16 matmul, no quantization)
    ref_output, _ = ref_norm_gemv_fp16(x, z, norm_weight, weight_fp16.to(DEVICE), HV, V, eps)

    out_diff = (output.cpu().float() - ref_output).abs()
    rel_err = out_diff.mean().item() / (ref_output.abs().mean().item() + 1e-6)
    max_diff = out_diff.max().item()
    # INT4 quantization introduces ~10% relative error vs fp16 on random weights
    rel_max = max_diff / (ref_output.abs().max().item() + 1e-6)
    assert rel_err < 0.15, \
        f"Output relative error: {rel_err:.6f} (HV={HV}, V={V}, N={N})"
    assert rel_max < 0.5, \
        f"Output relative max diff: {rel_max:.6f} (max_diff={max_diff:.4f}, HV={HV}, V={V}, N={N})"


@pytest.mark.parametrize("HV,V,N", [
    (4, 128, 64),
    (8, 128, 2048),
])
def test_norm_gemv_deterministic(HV, V, N):
    """Two identical calls should produce the same output."""
    torch.manual_seed(77)
    eps = 1e-6
    K = HV * V

    x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(V, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)

    out1 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    out2 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, out1, HV, V, eps)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, out2, HV, V, eps)
    torch.xpu.synchronize()

    diff = (out1.cpu().float() - out2.cpu().float()).abs().max().item()
    assert diff == 0.0, \
        f"Non-deterministic output: max diff={diff} (HV={HV}, V={V}, N={N})"


@pytest.mark.parametrize("HV,V,N", [(4, 128, 32), (8, 128, 256)])
def test_norm_gemv_zero_x(HV, V, N):
    """Zero x input should produce zero output (norm of zero is zero)."""
    torch.manual_seed(99)
    K = HV * V

    x = torch.zeros(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.ones(V, dtype=torch.float16, device=DEVICE)

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)

    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, output, HV, V, 1e-6)
    torch.xpu.synchronize()

    assert output.cpu().abs().max().item() == 0.0, \
        f"Expected zero output for zero x, got max={output.cpu().abs().max().item()}"


@pytest.mark.parametrize("HV,V,N", [(4, 128, 32), (8, 128, 256)])
def test_norm_gemv_zero_z(HV, V, N):
    """Zero z gate should produce zero output (silu(0) = 0)."""
    torch.manual_seed(99)
    K = HV * V

    x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.zeros(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.ones(V, dtype=torch.float16, device=DEVICE)

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)

    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, output, HV, V, 1e-6)
    torch.xpu.synchronize()

    # silu(0) = 0 * sigmoid(0) = 0, so gated output = normed * 0 = 0
    assert output.cpu().abs().max().item() == 0.0, \
        f"Expected zero output for zero z gate, got max={output.cpu().abs().max().item()}"


@pytest.mark.parametrize("HV,N", [(1, 16), (4, 64), (8, 256)])
def test_norm_gemv_sigmoid_correctness(HV, N):
    """The sigmoid ABI matches a reference using the dequantized INT4 weight."""
    torch.manual_seed(123)
    V = 128
    eps = 1e-6
    K = HV * V
    x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(V, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16)
    weight_deq = cpu_dequantize(qw, sc, N, K)
    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_sigmoid(
        x, z, norm_weight, qw.to(DEVICE), sc.to(DEVICE), output,
        HV, V, eps)
    torch.xpu.synchronize()

    reference, _ = ref_norm_gemv_fp16(
        x, z, norm_weight, weight_deq.to(DEVICE), HV, V, eps,
        gate="sigmoid")
    error = (output.cpu().float() - reference).abs()
    relative = error.mean().item() / (reference.abs().mean().item() + 1e-6)
    assert relative < 0.05, f"sigmoid INT4 relative error is {relative:.6f}"
    assert error.max().item() < 0.2, f"sigmoid INT4 max error is {error.max().item():.6f}"


def test_norm_gemv_sigmoid_zero_gate_is_distinct_from_legacy_silu():
    """At z=0, sigmoid produces half the normalized value while SiLU is zero."""
    HV, V, N = 1, 128, 16
    K = HV * V
    x = torch.ones(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.zeros(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.ones(V, dtype=torch.float16, device=DEVICE)
    weight_fp16 = torch.full((N, K), 0.5, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)
    sigmoid_output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    silu_output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)

    esimd_norm_gemv_int4_sigmoid(
        x, z, norm_weight, qw, sc, sigmoid_output, HV, V, 1e-6)
    esimd_norm_gemv_int4_pert(
        x, z, norm_weight, qw, sc, silu_output, HV, V, 1e-6)
    torch.xpu.synchronize()

    assert silu_output.cpu().abs().max().item() == 0.0
    assert sigmoid_output.cpu().abs().max().item() > 1.0
    reference, _ = ref_norm_gemv_fp16(
        x, z, norm_weight, cpu_dequantize(qw, sc, N, K).to(DEVICE),
        HV, V, 1e-6, gate="sigmoid")
    torch.testing.assert_close(
        sigmoid_output.cpu().float(), reference, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("z_value", [-20.0, 20.0])
def test_norm_gemv_sigmoid_extreme_gate(z_value):
    """Large positive and negative gates remain numerically finite and accurate."""
    HV, V, N = 4, 128, 32
    K = HV * V
    x = torch.ones(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.full((HV, V), z_value, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.ones(V, dtype=torch.float16, device=DEVICE)
    weight_fp16 = torch.full((N, K), 0.25, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16)
    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_sigmoid(
        x, z, norm_weight, qw.to(DEVICE), sc.to(DEVICE), output,
        HV, V, 1e-6)
    torch.xpu.synchronize()

    reference, _ = ref_norm_gemv_fp16(
        x, z, norm_weight, cpu_dequantize(qw, sc, N, K).to(DEVICE),
        HV, V, 1e-6, gate="sigmoid")
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(
        output.cpu().float(), reference, rtol=0.05, atol=0.05)


def test_norm_gemv_sigmoid_rejects_non_128_head_dim():
    """The INT4 scale layout is invalid when V is not the fixed 128."""
    HV, V, N = 1, 64, 16
    x = torch.ones(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.ones_like(x)
    norm_weight = torch.ones(V, dtype=torch.float16, device=DEVICE)
    weight = torch.zeros(N, HV * V // 8, dtype=torch.int32, device=DEVICE)
    scale = torch.ones(N, HV * V // 128, dtype=torch.float16, device=DEVICE)
    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError, match="V == 128"):
        esimd_norm_gemv_int4_sigmoid(
            x, z, norm_weight, weight, scale, output, HV, V, 1e-6)


@pytest.mark.parametrize("HV,V,N", [
    (8, 128, 256),
    (8, 128, 2048),     # Qwen3-Next-80B: HV=8, V=128, out_proj N=2048
])
def test_norm_gemv_int4_vs_fp8(HV, V, N):
    """INT4 and FP8 fused kernels should produce similar results from same fp16 weight."""
    torch.manual_seed(42)
    eps = 1e-6
    K = HV * V

    x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(V, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)

    # INT4 path: CPU quantize → kernel
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)
    out_int4 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw, sc, out_int4, HV, V, eps)

    # FP8 path: cast weight to fp8 → kernel
    weight_xpu = weight_fp16.to(DEVICE)
    # Re-scale weight so dequant(fp8) * scale ≈ original
    fp8_scale_val = weight_xpu.float().abs().max().item() / 448.0  # E4M3 max ≈ 448
    weight_scaled = (weight_xpu.float() / fp8_scale_val).to(torch.float8_e4m3fn)
    scale_tensor = torch.tensor([fp8_scale_val], dtype=torch.float32, device=DEVICE)
    out_fp8 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_fp8_pert(x, z, norm_weight, weight_scaled, scale_tensor, out_fp8, HV, V, eps)

    torch.xpu.synchronize()

    # Both are quantized approximations of the same fp16 weight — compare against fp16 ref
    ref_output, _ = ref_norm_gemv_fp16(x, z, norm_weight, weight_xpu, HV, V, eps)

    int4_err = (out_int4.cpu().float() - ref_output).abs().mean().item()
    fp8_err = (out_fp8.cpu().float() - ref_output).abs().mean().item()
    ref_mag = ref_output.abs().mean().item() + 1e-6

    int4_rel = int4_err / ref_mag
    fp8_rel = fp8_err / ref_mag

    int4_max = (out_int4.cpu().float() - ref_output).abs().max().item()
    fp8_max = (out_fp8.cpu().float() - ref_output).abs().max().item()

    print(f"\n  HV={HV}, N={N}:"
          f"\n    REF:  first 8 = {ref_output[0, :8].tolist()}"
          f"\n    INT4: first 8 = {out_int4.cpu().float()[0, :8].tolist()}"
          f"\n    FP8:  first 8 = {out_fp8.cpu().float()[0, :8].tolist()}"
          f"\n    INT4: mean_abs_err={int4_err:.4f}, max_abs_err={int4_max:.4f}, rel_err={int4_rel:.4f}"
          f"\n    FP8:  mean_abs_err={fp8_err:.4f}, max_abs_err={fp8_max:.4f}, rel_err={fp8_rel:.4f}")

    # Both should be reasonable approximations (< 20% relative error)
    assert int4_rel < 0.2, f"INT4 rel_err too large: {int4_rel:.4f}"
    assert fp8_rel < 0.2, f"FP8 rel_err too large: {fp8_rel:.4f}"

    # INT4 and FP8 outputs should be in the same ballpark (cosine similarity > 0.9)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_int4.cpu().float(), out_fp8.cpu().float(), dim=-1).item()
    assert cos_sim > 0.9, \
        f"INT4 vs FP8 cosine similarity too low: {cos_sim:.4f} (HV={HV}, V={V}, N={N})"


def cpu_dequantize(qweight, scale, N, K, block_size=128):
    """Dequantize INT4 packed weight back to fp16 on CPU."""
    weight = torch.zeros(N, K, dtype=torch.float32)
    qw = qweight.cpu()
    sc = scale.cpu().float()
    for n in range(N):
        for blk in range(K // block_size):
            s = sc[n, blk].item()
            for i in range(block_size // 8):
                packed = qw[n, blk * (block_size // 8) + i].item() & 0xFFFFFFFF
                for b in range(8):
                    k_idx = blk * block_size + i * 8 + b
                    q = (packed >> (b * 4)) & 0xF
                    weight[n, k_idx] = (q - 8) * s
    return weight.half()


@pytest.mark.parametrize("HV,V,N", [
    (8, 128, 256),
    (8, 128, 2048),
])
def test_norm_gemv_int4_vs_fp8_dequant(HV, V, N):
    """Compare INT4 kernel vs FP8 kernel, using INT4-dequantized weight as the fp16 source.

    This isolates the kernel computation error from INT4 quantization error:
    the fp16 reference and FP8 path both use the INT4-dequantized weight,
    so the only difference is FP8 re-quantization loss vs INT4 kernel precision.
    """
    torch.manual_seed(42)
    eps = 1e-6
    K = HV * V

    x = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    z = torch.randn(HV, V, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(V, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    # Quantize to INT4, then dequantize back to fp16 as the "ground truth"
    weight_fp16_orig = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16_orig, block_size=128)
    weight_deq = cpu_dequantize(qw, sc, N, K, block_size=128)

    # INT4 path: use the same qw/sc
    qw_xpu = qw.to(DEVICE)
    sc_xpu = sc.to(DEVICE)
    out_int4 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_int4_pert(x, z, norm_weight, qw_xpu, sc_xpu, out_int4, HV, V, eps)

    # FP8 path: use dequantized weight as fp16 source → cast to fp8
    weight_deq_xpu = weight_deq.to(DEVICE)
    fp8_scale_val = weight_deq_xpu.float().abs().max().item() / 448.0
    weight_scaled = (weight_deq_xpu.float() / fp8_scale_val).to(torch.float8_e4m3fn)
    scale_tensor = torch.tensor([fp8_scale_val], dtype=torch.float32, device=DEVICE)
    out_fp8 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    esimd_norm_gemv_fp8_pert(x, z, norm_weight, weight_scaled, scale_tensor, out_fp8, HV, V, eps)

    torch.xpu.synchronize()

    # Reference: norm+gate with dequantized weight (exact INT4 values, fp32 matmul)
    ref_output, _ = ref_norm_gemv_fp16(x, z, norm_weight, weight_deq_xpu, HV, V, eps)

    int4_err = (out_int4.cpu().float() - ref_output).abs().mean().item()
    fp8_err = (out_fp8.cpu().float() - ref_output).abs().mean().item()
    ref_mag = ref_output.abs().mean().item() + 1e-6

    int4_rel = int4_err / ref_mag
    fp8_rel = fp8_err / ref_mag

    int4_max = (out_int4.cpu().float() - ref_output).abs().max().item()
    fp8_max = (out_fp8.cpu().float() - ref_output).abs().max().item()

    print(f"\n  HV={HV}, N={N} (dequant weight as source):"
          f"\n    REF:  first 8 = {ref_output[0, :8].tolist()}"
          f"\n    INT4: first 8 = {out_int4.cpu().float()[0, :8].tolist()}"
          f"\n    FP8:  first 8 = {out_fp8.cpu().float()[0, :8].tolist()}"
          f"\n    INT4: mean_abs_err={int4_err:.4f}, max_abs_err={int4_max:.4f}, rel_err={int4_rel:.4f}"
          f"\n    FP8:  mean_abs_err={fp8_err:.4f}, max_abs_err={fp8_max:.4f}, rel_err={fp8_rel:.4f}")

    # INT4 kernel should be very close to ref (same quantized weights, only fp32 accumulation diff)
    assert int4_rel < 0.05, f"INT4 rel_err too large vs dequant ref: {int4_rel:.6f}"
    # FP8 has additional re-quantization loss on top
    assert fp8_rel < 0.2, f"FP8 rel_err too large vs dequant ref: {fp8_rel:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
