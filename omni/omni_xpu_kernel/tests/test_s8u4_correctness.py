"""
Correctness tests for the W4A8 s8u4 path (onednn_s8u4_gemm + quantize_act_s8).

Regression-protects the A770 verification from Blackwood416's review:
- quantize_act_s8 vs manual q = round(clamp(v*127/absmax, -127, 127)),
  scale = absmax/127 (f32), incl. QW-class large-absmax groups
  (fp16 ~3.4e4, bf16 ~2.3e6) that overflowed f16 scales -> NaN -> black.
- onednn_s8u4_gemm vs manual w = (q - zp) * scale reference for both the
  wa4 scalar-zp=8 path and the tint4 per-block-zp path, up to K=12288.
- Device checks: xscales / packed_u4 / scales_f16 / zp_u8 must be on XPU.
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


def _quantize_ref(x, group_size):
    """Manual per-group symmetric s8 quantization (reference)."""
    M, K = x.shape
    G = K // group_size
    xg = x.float().view(M, G, group_size)
    absmax = xg.abs().amax(dim=-1).clamp(min=1e-10)
    scale = absmax / 127.0
    q = torch.round(xg / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.view(M, K), scale  # [M, G] f32


def _signed_to_unsigned_packed(q_signed):
    """Signed int4 [-8,7] packed bytes -> unsigned u4 (^0x88, zp=8)."""
    return (q_signed.to(torch.uint8) ^ 0x88)


def _pack_signed_nibbles(q_nk):
    """[N, K] signed int4 values -> [N, K/2] uint8 (lo | hi << 4)."""
    v = q_nk.to(torch.int16)
    lo = v[..., 0::2] & 0x0F
    hi = (v[..., 1::2] & 0x0F) << 4
    return (lo | hi).to(torch.uint8)


def _s8u4_reference(act_s8, xscales, q_u4, zp, scale, act_gs, wei_gs):
    """Manual out = (act_s8 * xscale) @ ((q_u4 - zp) * scale)."""
    M, K = act_s8.shape
    N = q_u4.shape[0]
    G_wei = K // wei_gs
    act_deq = act_s8.float() * xscales.float().repeat_interleave(act_gs, dim=1)
    q_t = q_u4.float().t()                                    # [K, N]
    w = (q_t - zp.float().repeat_interleave(wei_gs, dim=0)) * scale.float().repeat_interleave(wei_gs, dim=0)
    return act_deq @ w


def _build_weights(K, N, wei_gs, device, seed, signed=True, scale_max=2.0):
    torch.manual_seed(seed)
    G = K // wei_gs
    if signed:
        q = torch.randint(-8, 8, (N, K), device=device)
        packed = _pack_signed_nibbles(q)
        q_u4 = (q.to(torch.uint8) & 0x0F).to(torch.float32)
    else:
        q_u4 = torch.randint(0, 16, (N, K), device=device).to(torch.float32)
        packed = _pack_signed_nibbles(q_u4.to(torch.int16))
    scale = (torch.rand(G, N, device=device) * (scale_max - 0.05) + 0.05).to(torch.float16)
    return packed, q_u4, scale


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [32, 64])
def test_quantize_act_s8_matches_reference(xpu_device, dtype, group_size):
    svdq = _native_svdq()
    torch.manual_seed(3)
    M, K = 64, 4096
    x = torch.randn(M, K, dtype=dtype, device=xpu_device)

    # QW-class large-absmax groups: fp16 ~3.4e4, bf16 ~2.3e6 (f16 scale overflow).
    big = torch.finfo(torch.float16).max * 0.55 if dtype == torch.float16 else 2.3e6
    for g in range(1, 4):
        xg = torch.randn(1, group_size, dtype=dtype, device=xpu_device)
        x[:, g * group_size:(g + 1) * group_size] = xg * (big / xg.abs().amax())

    q, scale = svdq.quantize_act_s8(x, group_size)
    q_ref, scale_ref = _quantize_ref(x, group_size)

    assert q.shape == x.shape and q.dtype == torch.int8
    assert scale.shape == (M, K // group_size) and scale.dtype == torch.float32
    assert torch.isfinite(q).all() and torch.isfinite(scale).all()
    # Kernel rounding may differ from torch.round by at most 1 on ties.
    assert (q.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item() <= 1
    assert torch.allclose(scale, scale_ref, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("act_gs", [32, 64])
@pytest.mark.parametrize("wei_gs", [64, 128])
@pytest.mark.parametrize("K", [4096, 8192, 12288])
@pytest.mark.parametrize("per_block_zp", [False, True])
def test_s8u4_gemm_matches_reference(xpu_device, dtype, act_gs, wei_gs, K, per_block_zp):
    svdq = _native_svdq()
    M, N = 32, 256
    torch.manual_seed(5)

    act_s8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=xpu_device)
    G_src = K // act_gs
    G_wei = K // wei_gs
    # Keep act_dequant magnitude ~O(1) so fp16/bf16 outputs stay finite at K=12288:
    # int8 act in [-127,127] with per-group scales ~1/127.
    xscales = ((torch.rand(M, G_src, device=xpu_device) * 0.5 + 0.001) / 127.0).to(torch.float32)

    packed, q_u4, scale = _build_weights(K, N, wei_gs, xpu_device, 9, signed=not per_block_zp)
    if per_block_zp:
        zp = torch.randint(0, 16, (G_wei, N), dtype=torch.uint8, device=xpu_device)
        out = svdq.onednn_s8u4_gemm(
            act_s8, xscales, packed, scale, out_dtype=dtype, zp_u8=zp)
        ref = _s8u4_reference(act_s8, xscales, q_u4, zp.float(), scale, act_gs, wei_gs)
    else:
        out = svdq.onednn_s8u4_gemm(
            act_s8, xscales, packed, scale, out_dtype=dtype, zp_u8=None)
        # wa4: scalar zp=8, q_u4 already mapped as signed values (q_signed = q_u4 - 8).
        ref = _s8u4_reference(act_s8, xscales, q_u4 - 8.0,
                              torch.zeros(G_wei, N, device=xpu_device),
                              scale, act_gs, wei_gs)

    assert out.shape == (M, N) and out.dtype == dtype
    assert torch.isfinite(out).all(), "s8u4 GEMM produced NaN/inf"
    assert torch.allclose(out.to(torch.float32), ref, atol=1.0, rtol=5e-2), (
        f"s8u4 mismatch: max abs {torch.abs(out - ref).max().item():.3e}")


@pytest.mark.parametrize("cpu_arg", ["xscales", "packed", "scales", "zp"])
def test_s8u4_gemm_rejects_cpu_side_args(xpu_device, cpu_arg):
    """Device checks: every GEMM argument must be XPU-resident."""
    svdq = _native_svdq()
    M, K, N, act_gs, wei_gs = 16, 512, 64, 64, 64
    G_src = K // act_gs
    G_wei = K // wei_gs

    act_s8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=xpu_device)
    xscales = torch.rand(M, G_src, dtype=torch.float32, device=xpu_device)
    packed, _, scale = _build_weights(K, N, wei_gs, xpu_device, 13)
    zp = torch.randint(0, 16, (G_wei, N), dtype=torch.uint8, device=xpu_device)

    args = {
        "xscales": (act_s8, xscales.cpu(), packed, scale, torch.float32, None),
        "packed": (act_s8, xscales, packed.cpu(), scale, torch.float32, None),
        "scales": (act_s8, xscales, packed, scale.cpu(), torch.float32, None),
        "zp": (act_s8, xscales, packed, scale, torch.float32, zp.cpu()),
    }
    with pytest.raises(RuntimeError, match="XPU"):
        svdq.onednn_s8u4_gemm(*args[cpu_arg])
