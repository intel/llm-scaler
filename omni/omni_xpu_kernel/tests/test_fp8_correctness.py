"""
Correctness tests for omni_xpu_kernel FP8 GEMM kernels
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
def xpu_device():
    """Get XPU device or skip test."""
    if not has_xpu():
        pytest.skip("XPU not available")
    return torch.device("xpu")


def make_fp8_case(device, m, n, k, dtype, has_bias):
    """Create a reusable FP8 GEMM test case."""
    x = torch.randn(m, k, device=device, dtype=dtype)
    weight_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)
    scales = (weight_fp32.abs().max(dim=1).values / 448.0).to(torch.float32)
    scales = torch.clamp(scales, min=1e-12)
    qweight = (weight_fp32 / scales.unsqueeze(1)).to(torch.float8_e4m3fn)
    bias = torch.randn(n, device=device, dtype=dtype) if has_bias else None
    return x, qweight, scales, bias


KNOWN_BAD_SHAPES = [
    (4096, 12288, 4096),
    (4608, 16384, 4096),
]


WORKFLOW_BF16_FALLBACK_SHAPES = [
    (4096, 24576, 4096),
    (4096, 4096, 12288),
    (4608, 36864, 4096),
    (4608, 4096, 16384),
]


WORKFLOW_BF16_CHUNK_CACHE_EXPECTATIONS = [
    (4096, 24576, 4096, 5),
    (4096, 4096, 12288, 1),
    (4608, 36864, 4096, 8),
    (4608, 4096, 16384, 3),
]


class TestFP8GEMMCorrectness:
    """Correctness tests for FP8 GEMM kernel."""

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_onednn_w8a16_fp8_cache_reuses_same_shape(self, xpu_device):
        """Repeated identical shapes should hit the FP8 primitive cache."""
        from omni_xpu_kernel import linear

        linear.fp8_cache_clear()
        assert linear.fp8_cache_stats() == {"hits": 0, "misses": 0, "size": 0}

        x, qweight, scales, bias = make_fp8_case(
            xpu_device,
            m=1,
            n=1024,
            k=1024,
            dtype=torch.float16,
            has_bias=False,
        )

        output_first = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
        output_second = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)

        stats = linear.fp8_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] >= 1
        assert stats["size"] == 1
        torch.testing.assert_close(output_first, output_second, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_onednn_w8a16_fp8_cache_misses_on_shape_change(self, xpu_device):
        """Changing the GEMM shape should create a second cached entry."""
        from omni_xpu_kernel import linear

        linear.fp8_cache_clear()

        case_a = make_fp8_case(
            xpu_device,
            m=1,
            n=1024,
            k=1024,
            dtype=torch.float16,
            has_bias=False,
        )
        case_b = make_fp8_case(
            xpu_device,
            m=16,
            n=1024,
            k=1024,
            dtype=torch.float16,
            has_bias=False,
        )

        linear.onednn_w8a16_fp8(*case_a)
        linear.onednn_w8a16_fp8(*case_b)

        stats = linear.fp8_cache_stats()
        assert stats["misses"] == 2
        assert stats["size"] == 2
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("m", [1, 16, 67]) # Including non-power-of-two shape
    @pytest.mark.parametrize("n", [1024, 4096])
    @pytest.mark.parametrize("k", [1024, 4097]) # Including non-power-of-two shape
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_onednn_w8a16_fp8_correctness(self, xpu_device, m, n, k, dtype, has_bias):
        """Test onednn_w8a16_fp8 correctness against PyTorch reference."""
        from omni_xpu_kernel import linear
        
        # input: [m, k] in dtype (fp16/bf16)
        # qweight: [n, k] in float8_e4m3fn
        # scales: [n] in float32 (per-output-channel scales)
        
        x = torch.randn(m, k, device=xpu_device, dtype=dtype)
        
        # Create FP8 weight for reference
        weight_fp32 = torch.randn(n, k, device=xpu_device, dtype=torch.float32)
        
        # Per-output-channel scaling (per row of [N, K])
        # scale = absmax(row) / 448.0
        scales = (weight_fp32.abs().max(dim=1).values / 448.0).to(torch.float32)
        # Ensure no zero scales to avoid division by zero
        scales = torch.clamp(scales, min=1e-12)
        
        # Quantize: [n, k] / [n, 1]
        qweight = (weight_fp32 / scales.unsqueeze(1)).to(torch.float8_e4m3fn)
        
        bias = None
        if has_bias:
            bias = torch.randn(n, device=xpu_device, dtype=dtype)
        
        # Kernel result
        # Expected API: linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
        # scales is [n] float32 tensor
        output_kernel = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
        
        # PyTorch reference
        # Dequantize weight: [n, k] * [n, 1]
        weight_dequant = qweight.to(torch.float32) * scales.unsqueeze(1)
        bias_fp32 = bias.to(torch.float32) if bias is not None else None
        output_ref = torch.nn.functional.linear(x.to(torch.float32), weight_dequant.to(torch.float32), 
                                               bias_fp32).to(dtype)
        
        # Compare
        # Tightened BF16 tolerance based on typical float8/bf16 accumulation expectations
        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            # BF16 has larger epsilon but better range, 5e-2 is a more standard tightened value for mixed-precision GEMM
            rtol, atol = 5e-2, 5e-2
            
        torch.testing.assert_close(output_kernel, output_ref, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("m,n,k", KNOWN_BAD_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_onednn_w8a16_fp8_known_bad_shapes_match_reference(self, xpu_device, m, n, k, dtype, has_bias):
        from omni_xpu_kernel import linear

        x, qweight, scales, bias = make_fp8_case(xpu_device, m=m, n=n, k=k, dtype=dtype, has_bias=has_bias)

        output_kernel = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)

        weight_dequant = qweight.to(torch.float32) * scales.unsqueeze(1)
        bias_fp32 = bias.to(torch.float32) if bias is not None else None
        output_ref = torch.nn.functional.linear(
            x.to(torch.float32),
            weight_dequant.to(torch.float32),
            bias_fp32,
        ).to(dtype)

        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 5e-2, 5e-2

        torch.testing.assert_close(output_kernel, output_ref, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_onednn_w8a16_fp8_known_bad_shape_chunked_path_reuses_per_chunk_cache(self, xpu_device):
        from omni_xpu_kernel import linear

        linear.fp8_cache_clear()
        assert linear.fp8_cache_stats() == {"hits": 0, "misses": 0, "size": 0}

        x, qweight, scales, bias = make_fp8_case(
            xpu_device,
            m=4096,
            n=12288,
            k=4096,
            dtype=torch.bfloat16,
            has_bias=False,
        )

        output_first = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
        output_second = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)

        stats = linear.fp8_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] >= 5
        assert stats["size"] == 1
        torch.testing.assert_close(output_first, output_second, rtol=5e-2, atol=5e-2)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("m,n,k", WORKFLOW_BF16_FALLBACK_SHAPES)
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_onednn_w8a16_fp8_workflow_bf16_fallback_shapes_match_reference(self, xpu_device, m, n, k, has_bias):
        from omni_xpu_kernel import linear

        dtype = torch.bfloat16
        x, qweight, scales, bias = make_fp8_case(xpu_device, m=m, n=n, k=k, dtype=dtype, has_bias=has_bias)

        output_kernel = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)

        weight_dequant = qweight.to(torch.float32) * scales.unsqueeze(1)
        bias_fp32 = bias.to(torch.float32) if bias is not None else None
        output_ref = torch.nn.functional.linear(
            x.to(torch.float32),
            weight_dequant.to(torch.float32),
            bias_fp32,
        ).to(dtype)

        torch.testing.assert_close(output_kernel, output_ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("m,n,k,min_hits", WORKFLOW_BF16_CHUNK_CACHE_EXPECTATIONS)
    def test_onednn_w8a16_fp8_workflow_bf16_fallback_shapes_reuse_chunk_cache(self, xpu_device, m, n, k, min_hits):
        from omni_xpu_kernel import linear

        linear.fp8_cache_clear()
        assert linear.fp8_cache_stats() == {"hits": 0, "misses": 0, "size": 0}

        x, qweight, scales, bias = make_fp8_case(
            xpu_device,
            m=m,
            n=n,
            k=k,
            dtype=torch.bfloat16,
            has_bias=False,
        )

        output_first = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)
        output_second = linear.onednn_w8a16_fp8(x, qweight, scales, bias=bias)

        stats = linear.fp8_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] >= min_hits
        assert stats["size"] == 1
        torch.testing.assert_close(output_first, output_second, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
