"""Unit tests for esimd_norm_gemv_int4_pert — fused RMSNormGated + INT4 GEMV."""
import pytest
import torch
import ctypes
import os

from custom_esimd_kernels_vllm import esimd_norm_gemv_int4_pert

DEVICE = "xpu"
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


def ref_norm_gemv_fp16(x, z, norm_weight, weight_fp16, HV, V, eps):
    """Reference: RMSNormGated + fp16 matmul on CPU (no quantization).

    1. Per-head RMSNorm: normed = x * rsqrt(mean(x^2) + eps) * weight
    2. SiLU gate: normed *= z * sigmoid(z)
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

        # SiLU gate: silu(z) = z * sigmoid(z)
        silu_z = zh * torch.sigmoid(zh)
        normed = normed * silu_z

        normed_flat[0, h * V:(h + 1) * V] = normed

    # Step 3: fp16 matmul
    result = normed_flat @ weight_fp16.cpu().float().T

    return result, normed_flat


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
