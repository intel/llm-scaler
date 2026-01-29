"""
Correctness tests for omni_xpu_kernel normalization kernels
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


class TestRMSNormCorrectness:
    """Correctness tests for RMSNorm kernel."""
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    @pytest.mark.parametrize("hidden_size", [2048, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_rms_norm_correctness(self, batch_size, hidden_size, dtype):
        """Test RMSNorm correctness against PyTorch reference."""
        from omni_xpu_kernel import norm
        
        eps = 1e-6
        input = torch.randn(batch_size, hidden_size, device="xpu", dtype=dtype)
        weight = torch.randn(hidden_size, device="xpu", dtype=dtype)
        
        # ESIMD kernel result
        output_esimd = norm.rms_norm(weight, input, eps=eps)
        
        # PyTorch reference (compute in fp32 for accuracy)
        input_fp32 = input.float()
        weight_fp32 = weight.float()
        rms = torch.sqrt(torch.mean(input_fp32 ** 2, dim=-1, keepdim=True) + eps)
        output_ref = ((input_fp32 / rms) * weight_fp32).to(dtype)
        
        # Compare with tolerance based on dtype
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-2, 1e-2
        
        torch.testing.assert_close(output_esimd, output_ref, rtol=rtol, atol=atol)


class TestLayerNormCorrectness:
    """Correctness tests for LayerNorm kernel."""
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    @pytest.mark.parametrize("hidden_size", [2048, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_layer_norm_with_affine(self, batch_size, hidden_size, dtype):
        """Test LayerNorm with weight and bias."""
        from omni_xpu_kernel import norm
        
        eps = 1e-5
        input = torch.randn(batch_size, hidden_size, device="xpu", dtype=dtype)
        weight = torch.randn(hidden_size, device="xpu", dtype=dtype)
        bias = torch.randn(hidden_size, device="xpu", dtype=dtype)
        
        # ESIMD kernel result
        output_esimd = norm.layer_norm(input, weight, bias, eps=eps)
        
        # PyTorch reference
        output_ref = torch.nn.functional.layer_norm(input, (hidden_size,), weight, bias, eps=eps)
        
        # Compare with tolerance based on dtype
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-2, 1e-2
        
        torch.testing.assert_close(output_esimd, output_ref, rtol=rtol, atol=atol)
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("hidden_size", [2048, 4096])
    def test_layer_norm_without_affine(self, batch_size, hidden_size):
        """Test LayerNorm without weight and bias."""
        from omni_xpu_kernel import norm
        
        eps = 1e-5
        dtype = torch.float32
        input = torch.randn(batch_size, hidden_size, device="xpu", dtype=dtype)
        
        # ESIMD kernel result
        output_esimd = norm.layer_norm(input, eps=eps)
        
        # PyTorch reference
        output_ref = torch.nn.functional.layer_norm(input, (hidden_size,), None, None, eps=eps)
        
        torch.testing.assert_close(output_esimd, output_ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
