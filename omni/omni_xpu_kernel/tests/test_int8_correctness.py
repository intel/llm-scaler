"""
Correctness tests for omni_xpu_kernel INT8 operations.

Test structure mirrors comfy-kitchen's test_int8.py + test_qdq.py:
- Quantization shape/dtype validation
- Dequantization correctness
- INT8 matmul vs float reference
- int8_linear vs PyTorch F.linear reference
- ConvRot Hadamard properties and roundtrip
- Edge cases (single-row, non-aligned K/N, large tensors)
- Cache reuse verification (when native available)
"""

import pytest
import torch


def has_xpu():
    """Check if XPU is available."""
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


@pytest.fixture
def device():
    """Get test device (XPU if available, else CPU)."""
    if has_xpu():
        return torch.device("xpu")
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Set deterministic seed."""
    torch.manual_seed(42)


# =============================================================================
# Quantization Tests
# =============================================================================


class TestQuantizeInt8Tensorwise:
    """Tests for quantize_int8_tensorwise."""

    def test_shape_dtype(self, device, seed):
        """Output is INT8, same shape, scalar float32 scale."""
        from omni_xpu_kernel import int8

        x = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_tensorwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.numel() == 1
        assert scale.dtype == torch.float32

    def test_values_in_range(self, device, seed):
        """Quantized values are within [-128, 127]."""
        from omni_xpu_kernel import int8

        x = torch.randn(256, 512, device=device, dtype=torch.float16)
        q, scale = int8.quantize_int8_tensorwise(x)

        assert q.min().item() >= -128
        assert q.max().item() <= 127

    def test_scale_matches_absmax(self, device, seed):
        """Scale should equal absmax / 127."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_tensorwise(x)

        expected_scale = x.abs().max().float() / 127.0
        torch.testing.assert_close(scale, expected_scale, rtol=1e-5, atol=1e-7)

    def test_roundtrip_error(self, device, seed):
        """Roundtrip (quant→dequant) error stays within INT8 tolerance."""
        from omni_xpu_kernel import int8

        x = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_tensorwise(x)
        dq = int8.dequantize_int8_simple(q, scale)

        rel_err = (x.float() - dq.float()).abs() / (x.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02, f"Mean relative error too high: {rel_err.mean():.4f}"

    def test_with_explicit_scale(self, device, seed):
        """Quantization with explicit scale works."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        explicit_scale = torch.tensor(0.5, dtype=torch.float32)
        q, scale = int8.quantize_int8_tensorwise(x, scale=explicit_scale)

        assert scale.item() == pytest.approx(0.5, abs=1e-6)

    def test_stochastic_rounding_seeded(self, device, seed):
        """Stochastic rounding is deterministic given same seed (reference path)."""
        from omni_xpu_kernel.int8 import _reference

        # Test with reference implementation directly (native may not support seeded RNG)
        x = torch.full((4096,), 0.5, device=device, dtype=torch.float32)
        explicit_scale = torch.tensor(1.0, dtype=torch.float32)
        q1, _ = _reference.quantize_int8_tensorwise(x, scale=explicit_scale, stochastic_rounding=123)
        q2, _ = _reference.quantize_int8_tensorwise(x, scale=explicit_scale, stochastic_rounding=123)
        q3, _ = _reference.quantize_int8_tensorwise(x, scale=explicit_scale, stochastic_rounding=124)

        assert torch.equal(q1, q2), "Same seed should produce same result"
        assert not torch.equal(q1, q3), "Different seed should produce different result"
        # Mean should be close to 0.5 (unbiased)
        assert 0.4 < q1.float().mean().item() < 0.6


class TestQuantizeInt8Rowwise:
    """Tests for quantize_int8_rowwise."""

    def test_shape_dtype(self, device, seed):
        """Output is INT8, same shape, per-row scales [..., 1]."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)
        assert scale.dtype == torch.float32

    def test_per_row_scale_matches_row_absmax(self, device, seed):
        """Each row scale should equal row absmax / 127."""
        from omni_xpu_kernel import int8

        x = torch.randn(16, 64, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_rowwise(x)

        expected_scale = (x.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
        torch.testing.assert_close(scale, expected_scale, rtol=1e-5, atol=1e-7)

    def test_3d_input(self, device, seed):
        """Handles 3D input (batch, seq, hidden)."""
        from omni_xpu_kernel import int8

        x = torch.randn(2, 16, 128, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_rowwise(x)

        assert q.shape == (2, 16, 128)
        assert scale.shape == (2, 16, 1)


# =============================================================================
# Dequantization Tests
# =============================================================================


class TestDequantizeInt8:
    """Tests for dequantize_int8_simple and dtype variant."""

    def test_simple_dtype_output(self, device, seed):
        """dequantize_int8_simple returns float32."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_tensorwise(x)
        dq = int8.dequantize_int8_simple(q, scale)

        assert dq.dtype == torch.float32
        assert dq.shape == x.shape

    def test_simple_dtype_variant(self, device, seed):
        """dequantize_int8_simple_dtype respects output dtype."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_tensorwise(x)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            dq = int8.dequantize_int8_simple_dtype(q, scale, out_dtype=dtype)
            assert dq.dtype == dtype
            assert dq.shape == x.shape

    def test_simple_dtype_matches_cast(self, device, seed):
        """Direct dtype output matches float32-then-cast path."""
        from omni_xpu_kernel import int8

        x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_rowwise(x)

        ref = int8.dequantize_int8_simple(q, scale)

        for dtype in [torch.float16, torch.bfloat16]:
            direct = int8.dequantize_int8_simple_dtype(q, scale, out_dtype=dtype)
            assert direct.dtype == dtype
            assert torch.equal(direct, ref.to(dtype))

    def test_rowwise_scale_broadcast(self, device, seed):
        """Dequantize with per-row scales broadcasts correctly."""
        from omni_xpu_kernel import int8

        x = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_rowwise(x)
        dq = int8.dequantize_int8_simple(q, scale)

        # Each row should be close to original
        for i in range(16):
            row_err = (x[i].float() - dq[i]).abs().max()
            row_scale = scale[i].item()
            # Max error is at most 1.0 * scale (round-to-nearest quantization error)
            assert row_err.item() <= row_scale * 1.01


# =============================================================================
# INT8 Matrix Multiplication Tests
# =============================================================================


class TestMmInt8:
    """Tests for mm_int8."""

    def test_basic_correctness(self, device, seed):
        """mm_int8 matches float reference."""
        from omni_xpu_kernel import int8

        a = torch.randint(-128, 127, (16, 64), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (64, 32), dtype=torch.int8, device=device)
        c = int8.mm_int8(a, b)

        assert c.dtype == torch.int32
        assert c.shape == (16, 32)

        c_ref = a.float() @ b.float()
        torch.testing.assert_close(c.float(), c_ref, rtol=0, atol=0)

    def test_large_shape(self, device, seed):
        """mm_int8 works for larger shapes."""
        from omni_xpu_kernel import int8

        a = torch.randint(-128, 127, (128, 512), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (512, 256), dtype=torch.int8, device=device)
        c = int8.mm_int8(a, b)

        assert c.shape == (128, 256)
        c_ref = a.float() @ b.float()
        torch.testing.assert_close(c.float(), c_ref, rtol=0, atol=0)

    def test_non_aligned_k(self, device, seed):
        """mm_int8 handles K not divisible by 8."""
        from omni_xpu_kernel import int8

        a = torch.randint(-128, 127, (8, 13), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (13, 16), dtype=torch.int8, device=device)
        c = int8.mm_int8(a, b)

        c_ref = a.float() @ b.float()
        torch.testing.assert_close(c.float(), c_ref, rtol=0, atol=0)

    def test_non_aligned_n(self, device, seed):
        """mm_int8 handles N not divisible by 8."""
        from omni_xpu_kernel import int8

        a = torch.randint(-128, 127, (16, 32), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (32, 7), dtype=torch.int8, device=device)
        c = int8.mm_int8(a, b)

        c_ref = a.float() @ b.float()
        torch.testing.assert_close(c.float(), c_ref, rtol=0, atol=0)

    def test_single_row(self, device, seed):
        """mm_int8 handles M=1 (GEMV-like)."""
        from omni_xpu_kernel import int8

        a = torch.randint(-128, 127, (1, 128), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (128, 64), dtype=torch.int8, device=device)
        c = int8.mm_int8(a, b)

        assert c.shape == (1, 64)
        c_ref = a.float() @ b.float()
        torch.testing.assert_close(c.float(), c_ref, rtol=0, atol=0)


# =============================================================================
# INT8 Linear Tests
# =============================================================================


class TestInt8Linear:
    """Tests for int8_linear."""

    def test_basic_shape_dtype(self, device, seed):
        """int8_linear produces correct output shape and dtype."""
        from omni_xpu_kernel import int8

        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_correctness_vs_reference(self, device, seed):
        """int8_linear result is close to full-precision linear."""
        from omni_xpu_kernel import int8

        x = torch.randn(128, 256, device=device, dtype=torch.float16)
        w = torch.randn(64, 256, device=device, dtype=torch.float16)
        bias = torch.randn(64, device=device, dtype=torch.float16)

        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)
        ref = torch.nn.functional.linear(x, w, bias)

        # INT8 introduces quantization noise; verify it's bounded
        rel_err = (out.float() - ref.float()).abs() / (ref.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.05, f"Mean rel error: {rel_err.mean():.4f}"

    def test_single_row_gemv(self, device, seed):
        """int8_linear handles single-row input (M=1)."""
        from omni_xpu_kernel import int8

        x = torch.randn(1, 512, device=device, dtype=torch.bfloat16)
        w = torch.randn(384, 512, device=device, dtype=torch.bfloat16)
        bias = torch.randn(384, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

        assert out.shape == (1, 384)
        assert out.dtype == torch.bfloat16

    def test_pads_k(self, device, seed):
        """int8_linear handles K not aligned to matmul tile size."""
        from omni_xpu_kernel import int8

        x = torch.randn(17, 12, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 12, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale)
        assert out.shape == (17, 64)

    def test_pads_n(self, device, seed):
        """int8_linear handles N not aligned to matmul tile size."""
        from omni_xpu_kernel import int8

        x = torch.randn(17, 16, device=device, dtype=torch.bfloat16)
        w = torch.randn(1, 16, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale)
        assert out.shape == (17, 1)

    def test_per_channel_weight_scale(self, device, seed):
        """int8_linear supports per-channel weight scale."""
        from omni_xpu_kernel import int8

        x = torch.randn(8, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        # Per-channel: one scale per output channel
        w_int8, w_scale = int8.quantize_int8_rowwise(w)

        out = int8.int8_linear(x, w_int8, w_scale.reshape(-1), out_dtype=torch.bfloat16)
        assert out.shape == (8, 64)

    def test_3d_input(self, device, seed):
        """int8_linear handles 3D input [B, S, K]."""
        from omni_xpu_kernel import int8

        x = torch.randn(2, 16, 256, device=device, dtype=torch.bfloat16)
        w = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)
        assert out.shape == (2, 16, 128)

    def test_no_bias(self, device, seed):
        """int8_linear works without bias."""
        from omni_xpu_kernel import int8

        x = torch.randn(8, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, bias=None, out_dtype=torch.bfloat16)
        assert out.shape == (8, 64)

    def test_dimension_mismatch_raises(self, device, seed):
        """int8_linear raises on K mismatch."""
        from omni_xpu_kernel import int8

        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        w = torch.randint(-128, 127, (64, 256), device=device, dtype=torch.int8)
        w_scale = torch.ones(1)

        with pytest.raises((ValueError, RuntimeError)):
            int8.int8_linear(x, w, w_scale)


# =============================================================================
# ConvRot Tests
# =============================================================================


class TestConvRot:
    """Tests for ConvRot (Hadamard rotation) operations."""

    def test_hadamard_properties(self, device, seed):
        """Hadamard matrix is orthogonal and symmetric."""
        from omni_xpu_kernel.int8._reference import _build_hadamard

        for size in [4, 16, 64, 256]:
            h = _build_hadamard(size, device=device, dtype=torch.float32)
            assert h.shape == (size, size)
            # Symmetric: H^T = H
            assert torch.allclose(h, h.T, atol=1e-5)
            # Orthogonal: H^T @ H = I
            identity = torch.eye(size, device=device, dtype=torch.float32)
            assert torch.allclose(torch.matmul(h.T, h), identity, atol=1e-4)

    def test_hadamard_invalid_sizes(self, device):
        """Invalid sizes raise ValueError."""
        from omni_xpu_kernel.int8._reference import _build_hadamard

        for size in [2, 8, 32, 128, 500]:
            with pytest.raises(ValueError, match="Regular Hadamard size must be a power of 4"):
                _build_hadamard(size, device=device)

    def test_convrot_weight_roundtrip(self, device, seed):
        """Quantize→dequantize with convrot preserves weight within INT8 tolerance."""
        from omni_xpu_kernel import int8

        w = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_convrot_weight(w, group_size=256)
        dq = int8.dequantize_int8_convrot_weight(q, scale, group_size=256)

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02, f"Mean rel error: {rel_err.mean():.4f}"

    def test_convrot_weight_shape_dtype(self, device, seed):
        """ConvRot quantization produces correct shapes."""
        from omni_xpu_kernel import int8

        w = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_convrot_weight(w, group_size=256)

        assert q.dtype == torch.int8
        assert q.shape == (64, 256)
        assert scale.shape == (64, 1)
        assert scale.dtype == torch.float32

    def test_convrot_divisibility_error(self, device, seed):
        """Error when channels not divisible by group_size."""
        from omni_xpu_kernel import int8

        w = torch.randn(64, 250, device=device, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="not divisible by group_size"):
            int8.quantize_int8_convrot_weight(w, group_size=256)

    def test_convrot_linear_correctness(self, device, seed):
        """int8_linear with convrot=True produces results close to fp reference."""
        from omni_xpu_kernel import int8

        group_size = 64
        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        bias = torch.randn(64, device=device, dtype=torch.bfloat16)

        q_w, s_w = int8.quantize_int8_convrot_weight(w, group_size=group_size)

        out = int8.int8_linear(
            x, q_w, s_w, bias=bias, out_dtype=torch.bfloat16,
            convrot=True, convrot_groupsize=group_size,
        )

        # Reference: plain fp linear
        ref = torch.nn.functional.linear(x, w, bias)

        assert out.shape == ref.shape
        assert out.dtype == ref.dtype

        # ConvRot + INT8 should be reasonably close to fp
        rel_err = (out.float() - ref.float()).abs() / (ref.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.05

    def test_convrot_linear_group_size_mismatch(self, device, seed):
        """int8_linear raises on group_size not dividing K."""
        from omni_xpu_kernel import int8

        x = torch.randn(4, 100, device=device, dtype=torch.bfloat16)
        w = torch.randint(-128, 127, (64, 100), device=device, dtype=torch.int8)
        s = torch.ones(64)

        with pytest.raises(ValueError, match="does not divide"):
            int8.int8_linear(x, w, s, convrot=True, convrot_groupsize=256)

    def test_convrot_vs_no_convrot_accuracy(self, device, seed):
        """ConvRot and non-ConvRot produce similar results (both approximate fp)."""
        from omni_xpu_kernel import int8

        group_size = 64
        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)

        # Normal INT8
        w_int8_normal, s_normal = int8.quantize_int8_tensorwise(w)
        out_normal = int8.int8_linear(x, w_int8_normal, s_normal, out_dtype=torch.bfloat16)

        # ConvRot INT8
        w_int8_cr, s_cr = int8.quantize_int8_convrot_weight(w, group_size=group_size)
        out_cr = int8.int8_linear(
            x, w_int8_cr, s_cr, out_dtype=torch.bfloat16,
            convrot=True, convrot_groupsize=group_size,
        )

        # Both should be in the same ballpark
        assert out_normal.shape == out_cr.shape
        # They won't be identical (different quantization), but close
        rel_diff = (out_normal.float() - out_cr.float()).abs() / (out_normal.float().abs().max() + 1e-8)
        assert rel_diff.mean().item() < 0.1


# =============================================================================
# Cache Tests (only meaningful when native is available)
# =============================================================================


class TestInt8Cache:
    """Tests for INT8 primitive cache."""

    def test_cache_stats_initial(self):
        """Cache starts empty."""
        from omni_xpu_kernel import int8

        int8.int8_cache_clear()
        stats = int8.int8_cache_stats()
        assert stats == {"hits": 0, "misses": 0, "size": 0}


# =============================================================================
# ComfyUI Integration Shape Tests
# =============================================================================


class TestComfyUIShapes:
    """Tests with shapes typical in ComfyUI diffusion models."""

    @pytest.mark.parametrize("m,n,k", [
        (1, 4096, 4096),     # Single token, large model
        (4, 4096, 4096),     # Small batch
        (32, 4096, 4096),    # Medium batch
        (128, 1024, 4096),   # Typical attention projection
        (4096, 4096, 4096),  # Large seq_len
    ])
    def test_workflow_shapes(self, device, seed, m, n, k):
        """int8_linear works for ComfyUI-relevant shapes."""
        from omni_xpu_kernel import int8

        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)

        assert out.shape == (m, n)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Tests mirroring comfy-kitchen skipped CUDA tests (XPU equivalents)
# =============================================================================


class TestXPUNativeInt8:
    """XPU-native tests covering comfy-kitchen CUDA-only test scenarios."""

    def test_weight_quantize_shape_dtype(self, device, seed):
        """Weight path: output INT8, scalar scale, shape preserved."""
        from omni_xpu_kernel import int8

        w = torch.randn(256, 512, device=device, dtype=torch.bfloat16)
        q, scale = int8.quantize_int8_tensorwise(w)

        assert q.dtype == torch.int8
        assert q.shape == w.shape
        assert scale.numel() == 1
        assert scale.dtype == torch.float32

    def test_activation_quantize_shape_dtype(self, device, seed):
        """Activation path (rowwise): per-row scales [..., 1]."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 128, device=device, dtype=torch.float16)
        q, scale = int8.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)

    def test_weight_dequantize_restores_dtype(self, device, seed):
        """Dequantize followed by cast restores original dtype."""
        from omni_xpu_kernel import int8

        for dtype in (torch.float16, torch.bfloat16):
            w = torch.randn(64, 128, device=device, dtype=dtype)
            q, scale = int8.quantize_int8_tensorwise(w)
            dq = int8.dequantize_int8_simple_dtype(q, scale, out_dtype=dtype)
            assert dq.dtype == dtype
            assert dq.shape == w.shape

    def test_linear_output_dtype_follows_activation(self, device, seed):
        """int8_linear output dtype follows activation dtype when out_dtype=None."""
        from omni_xpu_kernel import int8

        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(4, 128, device=device, dtype=dtype)
            w = torch.randn(64, 128, device=device, dtype=dtype)
            w_int8, w_scale = int8.quantize_int8_tensorwise(w)

            out = int8.int8_linear(x, w_int8, w_scale)  # out_dtype defaults to x.dtype
            assert out.dtype == dtype
            assert out.shape == (4, 64)

    def test_linear_with_bias_output_dtype(self, device, seed):
        """int8_linear with bias respects out_dtype."""
        from omni_xpu_kernel import int8

        x = torch.randn(4, 128, device=device, dtype=torch.float16)
        w = torch.randn(64, 128, device=device, dtype=torch.float32)
        bias = torch.randn(64, device=device, dtype=torch.float32)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)
        assert out.dtype == torch.float16
        assert out.shape == (4, 64)

    @pytest.mark.parametrize("backend", ["native", "reference"])
    def test_int8_linear_correctness_cross_backend(self, device, seed, backend):
        """int8_linear produces consistent results between native and reference."""
        from omni_xpu_kernel import int8
        from omni_xpu_kernel.int8 import _reference

        x = torch.randn(128, 256, device=device, dtype=torch.float16)
        w = torch.randn(64, 256, device=device, dtype=torch.float16)
        bias = torch.randn(64, device=device, dtype=torch.float16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        # Reference result (always using Python path)
        ref_out = _reference.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)

        # Native result (via dispatch)
        native_out = int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)

        # Should be very close (both use same algorithm, just different execution)
        max_diff = (native_out.float() - ref_out.float()).abs().max()
        # Allow small differences from rounding mode differences in ESIMD quant
        assert max_diff.item() < 0.5, f"Max diff between native and reference: {max_diff}"

    def test_dequantize_direct_output_dtype_matches_final_cast(self, device, seed):
        """Direct typed dequant output matches float32-then-cast path."""
        from omni_xpu_kernel import int8

        x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
        q_row, scale_row = int8.quantize_int8_rowwise(x)

        ref = int8.dequantize_int8_simple(q_row, scale_row)

        for dtype in [torch.float16, torch.bfloat16]:
            direct = int8.dequantize_int8_simple_dtype(q_row, scale_row, out_dtype=dtype)
            casted = ref.to(dtype)

            assert direct.dtype == dtype
            assert torch.equal(direct, casted), \
                f"Direct {dtype} output doesn't match float32-then-cast path"

    def test_convrot_weight_quantize_roundtrip_via_rotation(self, device, seed):
        """ConvRot weight quantize → dequant → inverse rotation recovers original."""
        from omni_xpu_kernel import int8
        from omni_xpu_kernel.int8._reference import _build_hadamard, _rotate_weight

        w = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
        group_size = 256

        # Quantize with convrot
        q, scale = int8.quantize_int8_convrot_weight(w, group_size=group_size)

        # Manual dequant + inverse rotation
        h = _build_hadamard(group_size, device=device, dtype=torch.float32)
        dq_rotated = q.float() * scale
        dq = _rotate_weight(dq_rotated, h, group_size).to(w.dtype)

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02, f"ConvRot roundtrip error: {rel_err.mean():.4f}"

    def test_convrot_dequantize_matches_manual(self, device, seed):
        """dequantize_int8_convrot_weight matches manual dequant+rotation."""
        from omni_xpu_kernel import int8

        w = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
        group_size = 256

        q, scale = int8.quantize_int8_convrot_weight(w, group_size=group_size)
        dq = int8.dequantize_int8_convrot_weight(q, scale, group_size=group_size)

        # Should recover the original weight within INT8 tolerance
        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_int8_linear_single_row_gemv(self, device, seed):
        """INT8 linear handles M=1 (GEMV path) correctly on XPU native."""
        from omni_xpu_kernel import int8

        x = torch.randn(1, 512, device=device, dtype=torch.bfloat16)
        w = torch.randn(384, 512, device=device, dtype=torch.bfloat16)
        bias = torch.randn(384, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        out = int8.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w, bias)

        assert out.shape == (1, 384)
        rel_err = (out.float() - ref.float()).abs() / (ref.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.05

    def test_int8_linear_per_channel_scale_correctness(self, device, seed):
        """Per-channel weight scale produces correct results."""
        from omni_xpu_kernel import int8

        x = torch.randn(32, 256, device=device, dtype=torch.bfloat16)
        w = torch.randn(128, 256, device=device, dtype=torch.bfloat16)

        # Per-channel quantize
        w_int8, w_scale = int8.quantize_int8_rowwise(w)  # [128, 1] scales

        out = int8.int8_linear(x, w_int8, w_scale.reshape(-1), out_dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)

        rel_err = (out.float() - ref.float()).abs() / (ref.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.05

    def test_cache_hit_reuses_primitive(self, device, seed):
        """Repeated identical shapes hit the oneDNN primitive cache."""
        from omni_xpu_kernel import int8

        int8.int8_cache_clear()

        x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = int8.quantize_int8_tensorwise(w)

        # First call — cache miss
        out1 = int8.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)
        stats1 = int8.int8_cache_stats()

        # Second call — should hit cache
        out2 = int8.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)
        stats2 = int8.int8_cache_stats()

        assert stats2["hits"] > stats1["hits"], "Second call should hit cache"
        assert stats2["size"] == stats1["size"], "Cache size should not grow"

    def test_cache_miss_on_shape_change(self, device, seed):
        """Different GEMM shapes create separate cache entries."""
        from omni_xpu_kernel import int8

        int8.int8_cache_clear()

        x1 = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w1_int8, w1_scale = int8.quantize_int8_tensorwise(w1)

        x2 = torch.randn(32, 256, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
        w2_int8, w2_scale = int8.quantize_int8_tensorwise(w2)

        int8.int8_linear(x1, w1_int8, w1_scale, out_dtype=torch.bfloat16)
        int8.int8_linear(x2, w2_int8, w2_scale, out_dtype=torch.bfloat16)

        stats = int8.int8_cache_stats()
        assert stats["misses"] >= 2, "Different shapes should cause cache misses"
        assert stats["size"] >= 2, "Cache should have separate entries"
