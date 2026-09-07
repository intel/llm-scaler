"""
Correctness tests for the torchao/asymmetric INT4 path of
onednn_int4_gemm_preconverted (per-block zero points).

Regression-protects the format contract:
    w = (q - zp) * scale  (q = raw u4 qdata [0,15], per-block zp + f16 scale)
applied entirely inside oneDNN — no conversion, no Python-side correction.

Covers the reviewer's verified shapes (gs 32/64/128/256, fp16/bf16/f32),
plus a Qwen-scale K=12288 no-NaN case (bf16 accumulation path).
"""

import pytest
import torch


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


@pytest.fixture
def xpu_device():
    if not has_xpu():
        pytest.skip("XPU not available")
    return torch.device("xpu")


def _native_svdq():
    try:
        from omni_xpu_kernel import svdq
        return svdq
    except (ImportError, AttributeError):
        pytest.skip("omni_xpu_kernel svdq extension not available")


def ref_pack_u4(q_nk: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] u4 values (0..15) into [N, K/2] uint8 (lo | hi << 4)."""
    v = q_nk.to(torch.int16)
    lo = v[..., 0::2] & 0x0F
    hi = (v[..., 1::2] & 0x0F) << 4
    return (lo | hi).to(torch.uint8)


def ref_forward(act, q_nk, zp, scale, group_size):
    """Manual w = (q - zp) * scale reference: out = act @ W."""
    N, K = q_nk.shape
    G = K // group_size
    q_t = q_nk.float().t()                       # [K, N]
    zp_g = zp.float().repeat_interleave(group_size, dim=0)   # [K, N]
    sc_g = scale.float().repeat_interleave(group_size, dim=0)  # [K, N]
    w = (q_t - zp_g) * sc_g                      # [K, N]
    return act.float() @ w


@pytest.mark.parametrize("M,K,N,gs,dtype", [
    (128, 256, 512, 64, torch.bfloat16),   # reviewer case
    (64, 4096, 1024, 128, torch.float16),  # reviewer case
    (256, 1024, 256, 32, torch.bfloat16),  # reviewer case
    (8, 512, 512, 256, torch.float32),     # reviewer case (tiny shape)
])
def test_preconverted_zp_matches_reference(xpu_device, M, K, N, gs, dtype):
    svdq = _native_svdq()
    torch.manual_seed(7)

    act = torch.randn(M, K, dtype=dtype, device=xpu_device)
    q_nk = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=xpu_device)
    G = K // gs
    zp = torch.randint(0, 16, (G, N), dtype=torch.uint8, device=xpu_device)
    # Realistic weight scale magnitudes keep output ~O(1-100): with larger
    # scales the bf16 accumulation noise (the kernel's fpmath-any path) grows
    # into whole units and masks the format contract we are testing.
    scale = (torch.rand(G, N, dtype=torch.float16, device=xpu_device) * 0.6 + 0.05)

    packed = ref_pack_u4(q_nk)
    out = svdq.onednn_int4_gemm_preconverted(act, packed, scale, zp)
    ref = ref_forward(act, q_nk, zp, scale, gs)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # bf16 accumulation (fpmath-any) leaves ~O(0.5) absolute error even on
    # near-zero outputs from large-sum cancellation; check absolute + 2% rel.
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float32 else (1.0, 2e-2)
    assert torch.allclose(out.to(torch.float32), ref, atol=atol, rtol=rtol), (
        f"preconverted+zp GEMM mismatch: max abs {torch.abs(out - ref).max().item():.3e}")


def test_preconverted_zp_qwen_scale_k_no_nan(xpu_device):
    """K=12288 (Qwen scale) must not overflow to NaN via bf16 accumulation."""
    svdq = _native_svdq()
    torch.manual_seed(11)

    M, K, N, gs = 32, 12288, 512, 128
    act = torch.randn(M, K, dtype=torch.bfloat16, device=xpu_device)
    q_nk = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=xpu_device)
    G = K // gs
    zp = torch.randint(0, 16, (G, N), dtype=torch.uint8, device=xpu_device)
    scale = torch.rand(G, N, dtype=torch.float16, device=xpu_device) * 0.6 + 0.1

    packed = ref_pack_u4(q_nk)
    out = svdq.onednn_int4_gemm_preconverted(act, packed, scale, zp)
    ref = ref_forward(act, q_nk, zp, scale, gs)

    assert torch.isfinite(out).all()
    # K=12288 bf16 accumulation noise scales with output magnitude (~0.4%).
    assert torch.allclose(out.to(torch.float32), ref, atol=8.0, rtol=2e-2)


def test_preconverted_zp_rejects_cpu_tensors(xpu_device):
    """Format contract: every argument must be XPU-resident."""
    svdq = _native_svdq()
    M, K, N, gs = 8, 256, 64, 64
    G = K // gs
    act = torch.randn(M, K, dtype=torch.float16, device=xpu_device)
    packed = ref_pack_u4(torch.randint(0, 16, (N, K), dtype=torch.uint8))
    zp = torch.randint(0, 16, (G, N), dtype=torch.uint8)
    scale = torch.rand(G, N, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="XPU"):
        svdq.onednn_int4_gemm_preconverted(act, packed, scale, zp)
