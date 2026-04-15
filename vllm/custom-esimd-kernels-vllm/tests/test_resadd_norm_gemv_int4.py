"""Unit tests for esimd_resadd_norm_gemv_int4_pert — fused ResAdd + RMSNorm + INT4 GEMV."""
import pytest
import torch
import ctypes
import os

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


def ref_resadd_norm_gemv_fp16(hidden, residual, norm_weight, weight_fp16, eps):
    """Reference: step-by-step ResAdd + RMSNorm + fp16 matmul on CPU."""
    h = hidden.cpu().float()
    r = residual.cpu().float()

    # Step 1: ResAdd
    updated_residual = h + r

    # Step 2: RMSNorm
    variance = updated_residual.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed = updated_residual * inv_rms * norm_weight.cpu().float()

    # Step 3: fp16 matmul
    result = normed @ weight_fp16.cpu().float().T

    return result, updated_residual.half(), normed.half()


@pytest.mark.parametrize("N,K", [
    (16, 128),
    (32, 256),
    (256, 512),      # K_SPLIT=2 path
    (512, 2048),     # K_SPLIT=4 path
    (128, 2048),     # Qwen3-Next: router N=128, K=2048 (K_SPLIT=8)
    (128, 4096),     # Qwen3.5: router N=128, K=4096 (K_SPLIT=8)
])
def test_resadd_norm_gemv_correctness(N, K):
    """Fused kernel should match step-by-step reference."""
    torch.manual_seed(42)
    eps = 1e-6

    hidden = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
    residual = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)

    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    normed_out = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
    residual_copy = residual.clone()

    esimd_resadd_norm_gemv_int4_pert(
        hidden, residual_copy, norm_weight, qw, sc, output, normed_out, eps)
    torch.xpu.synchronize()

    ref_output, ref_residual, ref_normed = ref_resadd_norm_gemv_fp16(
        hidden, residual, norm_weight, weight_fp16.to(DEVICE), eps)

    # Check residual update
    res_diff = (residual_copy.cpu().float() - ref_residual.float()).abs()
    assert res_diff.max().item() < 0.01, \
        f"Residual diff: {res_diff.max().item():.4f}"

    # Check normed output
    norm_diff = (normed_out.cpu().float() - ref_normed.float()).abs()
    assert norm_diff.max().item() < 0.1, \
        f"Normed diff: {norm_diff.max().item():.4f}"

    # Check GEMV output (INT4 quantization error grows with K due to accumulation)
    out_diff = (output.cpu().float() - ref_output).abs()
    rel_err = out_diff.mean().item() / (ref_output.abs().mean().item() + 1e-6)
    assert rel_err < 0.25, \
        f"Output relative error: {rel_err:.4f} (N={N}, K={K})"


@pytest.mark.parametrize("N,K", [(32, 256), (128, 2048)])
def test_resadd_norm_gemv_zero_hidden(N, K):
    """Zero hidden_states: residual unchanged, output from normed residual."""
    torch.manual_seed(99)
    eps = 1e-6

    hidden = torch.zeros(1, K, dtype=torch.float16, device=DEVICE)
    residual = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.ones(K, dtype=torch.float16, device=DEVICE)

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)

    output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    normed_out = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
    residual_orig = residual.clone()

    esimd_resadd_norm_gemv_int4_pert(
        hidden, residual, norm_weight, qw, sc, output, normed_out, eps)
    torch.xpu.synchronize()

    # residual should be 0 + residual = unchanged
    diff = (residual.cpu().float() - residual_orig.cpu().float()).abs().max().item()
    assert diff < 0.01, f"Residual changed with zero hidden: max diff={diff}"


@pytest.mark.parametrize("N,K", [
    (128, 2048),
    (128, 4096),
])
def test_resadd_norm_gemv_int4_vs_fp8(N, K):
    """Compare INT4 and FP8 fused ResAdd+Norm+GEMV kernels."""
    torch.manual_seed(42)
    eps = 1e-6

    hidden = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
    residual = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
    norm_weight = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    weight_fp16 = torch.randn(N, K, dtype=torch.float16)

    # INT4 path
    qw, sc = cpu_quantize(weight_fp16, block_size=128)
    qw = qw.to(DEVICE)
    sc = sc.to(DEVICE)
    out_int4 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    normed_int4 = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
    res_int4 = residual.clone()
    esimd_resadd_norm_gemv_int4_pert(
        hidden, res_int4, norm_weight, qw, sc, out_int4, normed_int4, eps)

    # FP8 path
    weight_xpu = weight_fp16.to(DEVICE)
    fp8_scale_val = weight_xpu.float().abs().max().item() / 448.0
    weight_scaled = (weight_xpu.float() / fp8_scale_val).to(torch.float8_e4m3fn)
    scale_tensor = torch.tensor([fp8_scale_val], dtype=torch.float32, device=DEVICE)
    out_fp8 = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
    normed_fp8 = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
    res_fp8 = residual.clone()
    esimd_resadd_norm_gemv_fp8_pert(
        hidden, res_fp8, norm_weight, weight_scaled, scale_tensor, out_fp8, normed_fp8, eps)

    torch.xpu.synchronize()

    # Reference
    ref_output, _, _ = ref_resadd_norm_gemv_fp16(
        hidden, residual, norm_weight, weight_xpu, eps)

    int4_err = (out_int4.cpu().float() - ref_output).abs().mean().item()
    fp8_err = (out_fp8.cpu().float() - ref_output).abs().mean().item()
    ref_mag = ref_output.abs().mean().item() + 1e-6

    int4_rel = int4_err / ref_mag
    fp8_rel = fp8_err / ref_mag
    int4_max = (out_int4.cpu().float() - ref_output).abs().max().item()
    fp8_max = (out_fp8.cpu().float() - ref_output).abs().max().item()

    print(f"\n  N={N}, K={K}:"
          f"\n    REF:  first 8 = {ref_output[0, :8].tolist()}"
          f"\n    INT4: first 8 = {out_int4.cpu().float()[0, :8].tolist()}"
          f"\n    FP8:  first 8 = {out_fp8.cpu().float()[0, :8].tolist()}"
          f"\n    INT4: mean_abs_err={int4_err:.4f}, max_abs_err={int4_max:.4f}, rel_err={int4_rel:.4f}"
          f"\n    FP8:  mean_abs_err={fp8_err:.4f}, max_abs_err={fp8_max:.4f}, rel_err={fp8_rel:.4f}")

    assert int4_rel < 0.2, f"INT4 rel_err too large: {int4_rel:.4f}"
    assert fp8_rel < 0.2, f"FP8 rel_err too large: {fp8_rel:.4f}"

    # Normed outputs should match (both do the same resadd+norm)
    norm_diff = (normed_int4.cpu().float() - normed_fp8.cpu().float()).abs().max().item()
    assert norm_diff < 0.01, f"Normed outputs differ: {norm_diff:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
