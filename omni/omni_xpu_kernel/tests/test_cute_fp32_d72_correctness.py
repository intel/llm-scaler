"""FP64 oracles for the FP32-input vision attention entry point."""
import pytest
import torch

from omni_xpu_kernel import cute


@pytest.mark.parametrize("batch,heads,lq,lkv,scale", [
    (2, 3, 33, 65, 1.0),
    (1, 16, 129, 257, 1.0),
    (1, 2, 37, 67, 4.0),
])
def test_fp32_d72_qkv_views(batch, heads, lq, lkv, scale):
    if not torch.xpu.is_available() or not cute.supports_fp32_d72_bhld():
        pytest.skip("requires the BMG FP32 D72 sidecar")
    torch.manual_seed(731)
    q = (torch.randn(batch, lq, heads, 72, device="xpu") * scale).transpose(1, 2)
    packed = torch.randn(batch, lkv, 3, heads, 72, device="xpu") * scale
    k = packed[:, :, 1].transpose(1, 2)
    v = packed[:, :, 2].transpose(1, 2)
    out = cute.sdp_bhld_fp32_d72(q, k, v)
    qc, kc, vc = (t.cpu().double() for t in (q, k, v))
    ref = ((qc @ kc.transpose(-1, -2)) / 72**0.5).softmax(-1) @ vc
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu().double(), ref, atol=1e-4, rtol=1e-4)
