"""
Correctness tests for omni_xpu_kernel GGUF kernels
"""

import pytest
import torch
import numpy as np


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


@pytest.fixture
def q4_0_data(xpu_device):
    """Create sample Q4_0 quantized data with valid FP16 scales."""
    n_blocks = 1000
    block_size = 18  # Q4_0: 2 bytes scale + 16 bytes data
    
    # Generate data block by block to ensure valid FP16 scales
    data = []
    for _ in range(n_blocks):
        # Generate a valid FP16 scale (avoid NaN/Inf)
        scale = np.random.uniform(-10.0, 10.0)
        scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
        # Generate random packed data
        packed = np.random.randint(0, 256, 16, dtype=np.uint8)
        data.extend(scale_bytes.tolist())
        data.extend(packed.tolist())
    
    return torch.tensor(data, dtype=torch.uint8, device=xpu_device)


def reference_dequantize_q4_0(data: torch.Tensor, sequential: bool = False):
    """Reference implementation for Q4_0 dequantization."""
    data_cpu = data.cpu()
    n_blocks = data_cpu.numel() // 18
    output = torch.zeros(n_blocks * 32, dtype=torch.float16)
    
    for i in range(n_blocks):
        block = data_cpu[i * 18 : (i + 1) * 18]
        scale = block[:2].view(torch.float16).item()
        packed = block[2:].numpy().astype(int)  # Convert to int to avoid overflow
        
        out_block = output[i * 32 : (i + 1) * 32]
        
        for j in range(16):
            low = (packed[j] & 0x0F) - 8
            high = (packed[j] >> 4) - 8
            
            if sequential:
                out_block[j] = scale * low
                out_block[j + 16] = scale * high
            else:
                out_block[2 * j] = scale * low
                out_block[2 * j + 1] = scale * high
    
    return output.to(data.device)


class TestGGUFImport:
    """Tests for module import and availability."""
    
    def test_import(self):
        """Test that the module can be imported."""
        import omni_xpu_kernel
        assert omni_xpu_kernel is not None
    
    def test_is_available(self):
        """Test availability check."""
        import omni_xpu_kernel
        result = omni_xpu_kernel.is_available()
        assert isinstance(result, bool)


class TestGGUFDequantCorrectness:
    """Correctness tests for GGUF dequantization kernels."""
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_dequantize_q4_0_shape(self, q4_0_data):
        """Test output shape."""
        from omni_xpu_kernel import gguf
        
        output = gguf.dequantize_q4_0(q4_0_data, torch.float16)
        n_blocks = q4_0_data.numel() // 18
        assert output.shape == (n_blocks * 32,)
        assert output.dtype == torch.float16
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_dequantize_q4_0_correctness(self, q4_0_data):
        """Test correctness against reference."""
        from omni_xpu_kernel import gguf
        
        output = gguf.dequantize_q4_0(q4_0_data, torch.float16)
        reference = reference_dequantize_q4_0(q4_0_data, sequential=False)
        
        torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_dequantize_q4_0_comfyui_correctness(self, q4_0_data):
        """Test ComfyUI layout correctness."""
        from omni_xpu_kernel import gguf
        
        output = gguf.dequantize_q4_0_comfyui(q4_0_data, torch.float16)
        reference = reference_dequantize_q4_0(q4_0_data, sequential=True)
        
        torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)
    
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_dequantize_dtypes(self, q4_0_data):
        """Test different output dtypes."""
        from omni_xpu_kernel import gguf
        
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            output = gguf.dequantize_q4_0(q4_0_data, dtype)
            assert output.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
