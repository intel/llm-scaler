"""
Correctness tests for omni_xpu_kernel SVDQuant W4A4 kernels.

Tests against pure-Python reference implementations matching nunchaku's cpu_ops.py.
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


# ============================================================================
# Python reference implementations (matching nunchaku cpu_ops.py exactly)
# ============================================================================

def ref_unpack_int4(packed: torch.Tensor, signed: bool = True) -> torch.Tensor:
    p = packed.view(torch.uint8).to(torch.int16)
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    if signed:
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.stack([low, high], dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    return unpacked


def ref_pack_int4(values: torch.Tensor) -> torch.Tensor:
    v = values.to(torch.int16)
    even = v[..., 0::2] & 0x0F
    odd = (v[..., 1::2] & 0x0F) << 4
    return (even | odd).to(torch.uint8)


def ref_dequantize_w4(packed, scales, group_size=64):
    unpacked = ref_unpack_int4(packed, signed=True).float()
    N, K = unpacked.shape
    num_groups = K // group_size
    unpacked = unpacked.view(N, num_groups, group_size)
    sc = scales.float().T.unsqueeze(-1)  # [N, G, 1]
    return (unpacked * sc).view(N, K)


def ref_quantize_act_int4(x, group_size=64):
    M, K = x.shape
    num_groups = K // group_size
    x_f = x.float()
    x_grouped = x_f.view(M, num_groups, group_size)
    group_max = x_grouped.abs().amax(dim=-1)
    scales = (group_max / 7.0).clamp(min=1e-10)
    rscale = 7.0 / group_max.clamp(min=1e-10)
    x_q = torch.round(x_grouped * rscale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8).view(M, K)
    packed = ref_pack_int4(x_q)
    return packed, scales.T.contiguous()  # scales: [G, M]


# ============================================================================
# Tests
# ============================================================================

class TestUnpackInt4:

    @pytest.mark.parametrize("shape", [
        (1, 32),     # minimal
        (1, 1920),   # nunchaku activation shape
        (3840, 1920), # nunchaku weight shape
        (16, 64),    # small
    ])
    def test_unpack_matches_reference(self, xpu_device, shape):
        M, K_half = shape
        packed = torch.randint(0, 256, (M, K_half), dtype=torch.uint8, device=xpu_device)

        from omni_xpu_kernel import svdq
        result = svdq.unpack_int4(packed, signed=True)

        ref = ref_unpack_int4(packed.cpu(), signed=True)
        assert result.shape == ref.shape, f"Shape mismatch: {result.shape} vs {ref.shape}"
        assert torch.equal(result.cpu().to(torch.int16), ref.to(torch.int16)), \
            f"Max diff: {(result.cpu().to(torch.int16) - ref.to(torch.int16)).abs().max()}"

    def test_unpack_value_range(self, xpu_device):
        packed = torch.randint(0, 256, (100, 32), dtype=torch.uint8, device=xpu_device)
        from omni_xpu_kernel import svdq
        result = svdq.unpack_int4(packed, signed=True)
        assert result.min() >= -8, f"Min value {result.min()} < -8"
        assert result.max() <= 7, f"Max value {result.max()} > 7"


class TestDequantizeW4:

    @pytest.mark.parametrize("N,K,out_dtype", [
        (16, 64, torch.float32),
        (16, 128, torch.bfloat16),
        (3840, 3840, torch.bfloat16),   # nunchaku main layer shape
        (11520, 3840, torch.float32),   # nunchaku QKV fused shape
        (1, 64, torch.float32),         # minimal
    ])
    def test_dequantize_matches_reference(self, xpu_device, N, K, out_dtype):
        K_half = K // 2
        num_groups = K // 64
        packed = torch.randint(0, 256, (N, K_half), dtype=torch.uint8, device=xpu_device)
        scales = torch.randn(num_groups, N, dtype=torch.bfloat16, device=xpu_device) * 0.1

        from omni_xpu_kernel import svdq
        result = svdq.dequantize_w4(packed, scales, out_dtype=out_dtype)

        ref = ref_dequantize_w4(packed.cpu(), scales.cpu(), group_size=64)
        result_f32 = result.cpu().float()
        ref_f32 = ref.float()

        assert result.shape == (N, K), f"Shape mismatch: {result.shape} vs ({N}, {K})"
        max_diff = (result_f32 - ref_f32).abs().max().item()
        # Allow tolerance for bf16 intermediate precision
        atol = 1e-2 if out_dtype == torch.bfloat16 else 1e-4
        assert max_diff < atol, f"Max diff {max_diff} exceeds tolerance {atol}"

    def test_dequantize_zero_scales(self, xpu_device):
        N, K = 16, 128
        packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=xpu_device)
        scales = torch.zeros(K // 64, N, dtype=torch.bfloat16, device=xpu_device)

        from omni_xpu_kernel import svdq
        result = svdq.dequantize_w4(packed, scales, out_dtype=torch.float32)
        assert result.abs().max().item() == 0.0, "Zero scales should produce zero output"


class TestQuantizeActInt4:

    @pytest.mark.parametrize("M,K", [
        (1, 64),
        (1, 3840),     # nunchaku activation shape
        (4, 3840),
        (16, 128),
    ])
    def test_quantize_matches_reference(self, xpu_device, M, K):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=xpu_device)

        from omni_xpu_kernel import svdq
        packed_esimd, scales_esimd = svdq.quantize_act_int4(x, group_size=64)

        packed_ref, scales_ref = ref_quantize_act_int4(x.cpu(), group_size=64)

        assert packed_esimd.shape == packed_ref.shape, \
            f"Packed shape mismatch: {packed_esimd.shape} vs {packed_ref.shape}"
        assert scales_esimd.shape == scales_ref.shape, \
            f"Scales shape mismatch: {scales_esimd.shape} vs {scales_ref.shape}"

        # Scales should be close
        scales_diff = (scales_esimd.cpu().float() - scales_ref.float()).abs().max().item()
        assert scales_diff < 1e-2, f"Scales max diff {scales_diff}"

        # Packed values should match exactly (same quantization)
        packed_match = torch.equal(packed_esimd.cpu(), packed_ref)
        if not packed_match:
            # Allow 1-bit rounding differences
            unp_esimd = ref_unpack_int4(packed_esimd.cpu())
            unp_ref = ref_unpack_int4(packed_ref)
            max_q_diff = (unp_esimd - unp_ref).abs().max().item()
            assert max_q_diff <= 1, f"Quantized values differ by more than 1: max_diff={max_q_diff}"


class TestRoundtrip:
    """Test quantize → dequantize round-trip matches reference."""

    def test_roundtrip_accuracy(self, xpu_device):
        M, K = 1, 3840
        N = 3840
        x_act = torch.randn(M, K, dtype=torch.bfloat16, device=xpu_device) * 0.5
        x_wgt = torch.randn(N, K, dtype=torch.bfloat16, device=xpu_device) * 0.1

        from omni_xpu_kernel import svdq

        # Quantize both
        act_packed, act_scales = svdq.quantize_act_int4(x_act, group_size=64)
        wgt_packed, wgt_scales = ref_quantize_act_int4(x_wgt.cpu(), group_size=64)
        wgt_packed = wgt_packed.to(xpu_device)
        wgt_scales = wgt_scales.to(xpu_device)

        # Dequantize weights using ESIMD kernel
        wgt_deq = svdq.dequantize_w4(wgt_packed, wgt_scales, out_dtype=torch.float32)

        # Reference dequantize
        wgt_deq_ref = ref_dequantize_w4(wgt_packed.cpu(), wgt_scales.cpu())

        max_diff = (wgt_deq.cpu() - wgt_deq_ref).abs().max().item()
        assert max_diff < 1e-3, f"Dequantize roundtrip max diff {max_diff}"

        # Full GEMM comparison
        act_deq = svdq.dequantize_w4(act_packed, act_scales.to(torch.bfloat16), out_dtype=torch.float32)
        gemm_esimd = act_deq @ wgt_deq.T

        act_deq_ref = ref_dequantize_w4(act_packed.cpu(), act_scales.cpu().to(torch.bfloat16))
        gemm_ref = act_deq_ref @ wgt_deq_ref.T

        gemm_diff = (gemm_esimd.cpu() - gemm_ref).abs().max().item()
        rel_err = gemm_diff / (gemm_ref.abs().max().item() + 1e-10)
        assert rel_err < 0.01, f"GEMM relative error {rel_err:.6f} (abs diff {gemm_diff:.6f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
