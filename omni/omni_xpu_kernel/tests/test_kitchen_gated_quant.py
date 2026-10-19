"""Gated quantizer correctness against dense FP64 activation/rotation references."""

import pytest
import torch

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")


def activation(x, act):
    first, second = x.chunk(2, dim=-1)
    if act == "situ":
        return (4 * torch.tanh(second / 4) * torch.sigmoid(second)) * (25 * torch.tanh(first / 25))
    return torch.nn.functional.silu(first) * second


def rotate(x, group):
    if not group:
        return x
    # Explicit dense H4 tensor products, independent of native butterfly indexing.
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                       [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < group:
        matrix = torch.kron(matrix, h4)
    matrix /= group ** 0.5
    return (x.reshape(-1, group) @ matrix).reshape_as(x)


def assert_reference(x, act, group, codes, scales):
    raw = x.cpu()
    exact = rotate(activation(raw.double(), act), group)
    expected_scale = (exact.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
    expected = (exact / expected_scale).round().clamp(-128, 127)
    torch.testing.assert_close(scales.cpu().double(), expected_scale, rtol=2e-5, atol=1e-35)
    assert codes.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert codes.dtype == torch.int8 and scales.dtype == torch.float32
    difference = (codes.cpu().double() - expected).abs()
    assert difference.max().item() <= 1
    # The CUDA contract requires a fused FP32 activation to be at least as
    # accurate as materializing the activation in the input dtype first.
    chain = rotate(activation(raw, act).double(), group).to(raw.dtype).double()
    chain_scale = (chain.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
    chain_codes = (chain / chain_scale).round().clamp(-128, 127)
    chain_error = (chain_codes - expected).abs().mean().item()
    assert difference.mean().item() <= max(chain_error * 1.05, 1e-4)


@pytest.mark.parametrize("act", ["situ", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group", [0, 64, 256])
@pytest.mark.parametrize("shape", [(1, 512), (17, 512), (2, 3, 2048), (1, 16384)])
def test_gated_quantizer_dense_reference(act, dtype, group, shape):
    torch.manual_seed(193)
    x = torch.randn(shape, device="xpu", dtype=dtype) * 3
    if act == "situ":
        x[..., :shape[-1] // 2] *= 20
        x[..., shape[-1] // 2:] *= 4
    x.flatten()[::127] = 2000
    x.flatten()[61::251] = -2000
    original = x.clone()
    codes, scales = kitchen.gated_quantize_int8(x, act, group)
    assert_reference(x, act, group, codes, scales)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_situ_zero_near_zero_and_strided_input(dtype):
    torch.manual_seed(197)
    x = torch.randn(4, 4096, device="xpu", dtype=dtype)[:, ::2]
    x *= 1e-6
    x[0].zero_()
    codes, scales = kitchen.gated_quantize_int8(x, "situ", 256)
    assert_reference(x, "situ", 256, codes, scales)
    assert codes[0].count_nonzero().item() == 0
    assert scales[0].item() > 0


def test_situ_graph_replays_new_inputs():
    x = torch.randn(3, 512, device="xpu", dtype=torch.bfloat16)
    for _ in range(3):
        kitchen.gated_quantize_int8(x, "situ", 256)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        codes, scales = kitchen.gated_quantize_int8(x, "situ", 256)
    for factor in [1e-6, 30, 0, 1]:
        x.normal_().mul_(factor)
        graph.replay()
        assert_reference(x, "situ", 256, codes, scales)


def test_gated_quantizer_empty_batch_and_rejections():
    x = torch.empty(0, 512, device="xpu", dtype=torch.bfloat16)
    codes, scales = kitchen.gated_quantize_int8(x, "situ", 256)
    assert codes.shape == (0, 256) and scales.shape == (0, 1)
    for shape, dtype, act, group, message in [
        ((2, 511), torch.bfloat16, "situ", 0, "even raw width"),
        ((2, 16386), torch.bfloat16, "situ", 0, "8192"),
        ((2, 512), torch.float32, "situ", 0, "BF16 or FP16"),
        ((2, 512), torch.bfloat16, "gelu_tanh", 0, "situ or swiglu"),
        ((2, 512), torch.bfloat16, "situ", 16, "0, 64 or 256"),
        ((2, 128), torch.bfloat16, "situ", 256, "divide"),
    ]:
        bad = torch.empty(shape, device="xpu", dtype=dtype)
        with pytest.raises(RuntimeError, match=message):
            kitchen.gated_quantize_int8(bad, act, group)
