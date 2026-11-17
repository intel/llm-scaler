"""FP64 oracles for the FP32-input vision attention entry point."""
import pytest
import math
import torch
import torch.nn.functional as F

from omni_xpu_kernel import cute


def _require_kernel():
    if not torch.xpu.is_available() or not cute.supports_fp32_d72_bhld():
        pytest.skip("requires the BMG FP32 D72 sidecar")


def _reference(q, k, v):
    qc, kc, vc = (t.cpu().double() for t in (q, k, v))
    return ((qc @ kc.transpose(-1, -2)) / 72**0.5).softmax(-1) @ vc


@pytest.mark.parametrize("batch,heads,lq,lkv,scale", [
    (2, 3, 33, 65, 1.0),
    (1, 16, 129, 257, 1.0),
    (1, 2, 37, 67, 4.0),
])
def test_fp32_d72_qkv_views(batch, heads, lq, lkv, scale):
    _require_kernel()
    torch.manual_seed(731)
    q = (torch.randn(batch, lq, heads, 72, device="xpu") * scale).transpose(1, 2)
    packed = torch.randn(batch, lkv, 3, heads, 72, device="xpu") * scale
    k = packed[:, :, 1].transpose(1, 2)
    v = packed[:, :, 2].transpose(1, 2)
    out = cute.sdp_bhld_fp32_d72(q, k, v)
    ref = _reference(q, k, v)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu().double(), ref, atol=1e-4, rtol=1e-4)
    # The kernel replaces FP32 SDPA, so its error is held to the same order.
    sdpa = F.scaled_dot_product_attention(q, k, v)
    error = (out.cpu().double() - ref).abs().max()
    sdpa_error = (sdpa.cpu().double() - ref).abs().max()
    assert error <= 2 * sdpa_error, (float(error), float(sdpa_error))


@pytest.mark.parametrize("value", [3.4027e38, float("inf"), float("nan")])
def test_fp32_d72_extreme_values_follow_reference(value):
    _require_kernel()
    torch.manual_seed(5)
    # Zero queries give every key the same weight, so the extreme value
    # reaches each output row of its column.
    q = torch.zeros(1, 2, 17, 72, device="xpu")
    k = torch.randn(1, 2, 40, 72, device="xpu")
    v = torch.randn(1, 2, 40, 72, device="xpu")
    v[:, :, 5, 3] = value
    out = cute.sdp_bhld_fp32_d72(q, k, v).cpu().double()
    ref = _reference(q, k, v)
    assert torch.equal(out.isnan(), ref.isnan())
    assert torch.equal(out.isinf(), ref.isinf())
    finite = ref.isfinite()
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("length,value", [(32, 2.0**123), (65, -2.0**126)])
def test_fp32_d72_repeated_large_values_remain_finite(length, value):
    _require_kernel()
    q = torch.zeros(1, 2, 17, 72, device="xpu")
    k = torch.zeros(1, 2, length, 72, device="xpu")
    v = torch.full_like(k, value)
    out = cute.sdp_bhld_fp32_d72(q, k, v)
    assert out.isfinite().all()
    torch.testing.assert_close(out, torch.full_like(q, value), atol=0, rtol=1e-5)


def test_fp32_d72_tiny_weights_keep_large_v_contributions():
    _require_kernel()
    q = torch.zeros(1, 2, 17, 72, device="xpu")
    q[..., 0] = 1
    k = torch.zeros(1, 2, 32, 72, device="xpu")
    k[:, :, 1:, 0] = -134 * math.log(2) * math.sqrt(72)
    v = torch.full_like(k, 2.0**123)
    v[:, :, 0] = 0
    # Scaling P itself underflows these weights to zero in TF32, losing an
    # otherwise finite ~0.01514 contribution. Scaling V preserves it.
    out = cute.sdp_bhld_fp32_d72(q, k, v)
    torch.testing.assert_close(out.cpu().double(), _reference(q, k, v), atol=1e-4, rtol=1e-4)
