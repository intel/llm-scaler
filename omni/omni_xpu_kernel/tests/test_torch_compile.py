"""PT2 contracts for native inference APIs; requires a matching installed XPU DSO."""

import pytest
import torch

from omni_xpu_kernel import cute, int8, norm

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


def _inputs(dtype=torch.bfloat16, noncontiguous=False):
    x = torch.randn((3, 128), device="xpu", dtype=dtype)
    y = torch.randn_like(x)
    if noncontiguous:
        x = x.t().contiguous().t()
        y = y.t().contiguous().t()
    weight = torch.randint(-127, 128, (64, 128), device="xpu", dtype=torch.int8)
    scale = torch.full((64, 1), 0.01, device="xpu", dtype=torch.float32)
    return x, y, weight, scale


def _case(name, dtype, noncontiguous):
    x, y, weight, scale = _inputs(dtype, noncontiguous)
    if name == "rms_norm":
        return norm.rms_norm, (torch.ones(128, device="xpu", dtype=dtype), x.contiguous()), {}
    if name == "layer_norm":
        return norm.layer_norm, (x,), {}
    if name == "fused_silu_mul":
        return int8.fused_silu_mul, (x, y), {}
    if name == "quantize_int8_rowwise":
        return int8.quantize_int8_rowwise, (x,), {}
    if name == "fused_silu_mul_quantize_rowwise":
        return int8.fused_silu_mul_quantize_rowwise, (x, y), {}
    if name == "fused_swiglu_quantize_rowwise":
        return int8.fused_swiglu_quantize_rowwise, (torch.cat((x, y), -1),), {}
    if name == "fused_gelu_tanh_quantize_rowwise":
        return int8.fused_gelu_tanh_quantize_rowwise, (x,), {}
    if name == "rotate_convrot":
        return int8.rotate_convrot, (x,), {"group_size": 16}
    if name == "int8_linear":
        return int8.int8_linear, (x, weight, scale), {"convrot": True, "convrot_groupsize": 16}
    if name == "int8_linear_shared_input":
        return int8.int8_linear_shared_input, (x, weight, scale, weight, scale), {
            "convrot": True, "convrot_groupsize": 16, "out_dtype": dtype,
        }
    if name == "int8_linear_prequantized":
        q, qs = int8.quantize_int8_rowwise(x)
        return int8.int8_linear_prequantized, (q, qs, weight, scale), {"out_dtype": dtype}
    raise AssertionError(name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("name", [
    "rms_norm", "layer_norm", "fused_silu_mul", "quantize_int8_rowwise",
    "fused_silu_mul_quantize_rowwise", "fused_swiglu_quantize_rowwise",
    "fused_gelu_tanh_quantize_rowwise", "rotate_convrot", "int8_linear",
    "int8_linear_shared_input", "int8_linear_prequantized",
])
def test_native_inference_fullgraph_and_operator_contract(name, dtype, noncontiguous):
    function, args, kwargs = _case(name, dtype, noncontiguous)
    with torch.inference_mode():
        expected = function(*args, **kwargs)
        compiled = torch.compile(function, backend="inductor", fullgraph=True)
        actual = compiled(*args, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Includes alias/mutation schema, FakeTensor shape/dtype/stride consistency,
    # and AOT dispatch checks against the real installed native implementation.
    torch.library.opcheck(getattr(torch.ops.omni_xpu, name).default, args, kwargs)


def test_dynamic_row_count_remains_one_graph():
    from torch._dynamo.testing import CompileCounterWithBackend

    backend = CompileCounterWithBackend("inductor")
    compiled = torch.compile(int8.quantize_int8_rowwise, backend=backend,
                             fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for rows in (2, 3, 5):
            x = torch.randn((rows, 128), device="xpu", dtype=torch.bfloat16)
            torch.testing.assert_close(compiled(x), int8.quantize_int8_rowwise(x), rtol=0, atol=0)
    assert backend.frame_count == 1


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("head_dim", [120, 128])
def test_cute_fake_preserves_bhld_output_layout(packed, head_dim):
    if not cute.is_available():
        pytest.skip("CUTE sidecar required")
    name = f"sdp_bhld_d{head_dim}"
    if not getattr(cute, f"supports_d{head_dim}_bhld")():
        pytest.skip("target does not provide this CUTE contract")
    q = torch.randn((1, 32, 4, head_dim), device="xpu", dtype=torch.bfloat16).transpose(1, 2)
    if packed:
        q = q.contiguous()
    function = getattr(cute, name)
    with torch.inference_mode():
        expected = function(q, q, q)
        actual = torch.compile(function, backend="inductor", fullgraph=True)(q, q, q)
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.library.opcheck(getattr(torch.ops.cute_fmha, name).default, (q, q, q))


def test_cute_blhd_fullgraph():
    if not cute.is_available():
        pytest.skip("CUTE sidecar required")
    q = torch.randn((1, 32, 4, 128), device="xpu", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = cute.sdp(q, q, q)
        actual = torch.compile(cute.sdp, backend="inductor", fullgraph=True)(q, q, q)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.library.opcheck(torch.ops.cute_fmha.sdp.default, (q, q, q))
