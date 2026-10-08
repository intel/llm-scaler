"""Regression tests for native fused scale-back bias validation."""

import pytest
import torch


@pytest.fixture(scope="module")
def scaleback():
    if not torch.xpu.is_available():
        pytest.skip("native fused scale-back requires XPU")

    from omni_xpu_kernel import int8

    native = int8._get_native()
    if native is None or not hasattr(native, "fused_scaleback"):
        pytest.skip("native fused scale-back is unavailable")
    return native.fused_scaleback


def _inputs(n):
    return (
        torch.zeros((2, n), device="xpu", dtype=torch.int32),
        torch.ones(2, device="xpu"),
        torch.ones(1, device="xpu"),
    )


@pytest.mark.parametrize("n", [64, 65])
@pytest.mark.parametrize("bias_size", ["empty", "scalar", "short", "long"])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_rejects_wrong_bias_length(scaleback, n, bias_size, bias_dtype):
    length = {"empty": 0, "scalar": 1, "short": n - 1, "long": n + 1}[bias_size]
    # Retain a full backing buffer so a regression with an unchanged dtype
    # can fail the assertion without reading outside this test's own storage.
    backing = torch.zeros(n + 1, device="xpu", dtype=bias_dtype)
    bias = backing[:length]

    message = f"bias must contain exactly N={n} elements, got numel={length}"
    dtype_code = 2 if bias_dtype == torch.bfloat16 else 0
    with pytest.raises(RuntimeError, match=message):
        scaleback(*_inputs(n), bias, dtype_code)


@pytest.mark.parametrize("n", [64, 65])
@pytest.mark.parametrize("dtype_code", [0, 1, 2])
def test_rejects_cpu_bias(scaleback, n, dtype_code):
    bias = torch.zeros(n, device="cpu")
    with pytest.raises(
        RuntimeError, match="bias must be on the same XPU device as gemm_result"
    ):
        scaleback(*_inputs(n), bias, dtype_code)


@pytest.mark.parametrize("n", [64, 65])
def test_rejects_other_xpu_bias(scaleback, n):
    count = torch.xpu.device_count()
    if count < 2:
        pytest.skip("requires two XPU devices")

    gemm, x_scale, w_scale = _inputs(n)
    other_index = (gemm.device.index + 1) % count
    bias = torch.zeros(n, device=f"xpu:{other_index}", dtype=torch.bfloat16)
    with pytest.raises(
        RuntimeError, match="bias must be on the same XPU device as gemm_result"
    ):
        scaleback(gemm, x_scale, w_scale, bias)


@pytest.mark.parametrize("n", [64, 65])
@pytest.mark.parametrize(
    "dtype_code,dtype",
    [(0, torch.float32), (1, torch.float16), (2, torch.bfloat16)],
)
@pytest.mark.parametrize("scalar_scale", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_reference(scaleback, n, dtype_code, dtype, scalar_scale, with_bias):
    # Exercise materialization and dtype conversion along with both dispatch
    # paths, scalar/channel scales, and the optional bias branch.
    gemm = torch.arange(2 * n, device="xpu", dtype=torch.int32).reshape(n, 2).t()
    x_scale = torch.tensor([[0.5], [0.25]], device="xpu", dtype=torch.float16)
    w_scale = (
        torch.tensor(0.125, device="xpu", dtype=torch.float16)
        if scalar_scale
        else (torch.arange(n, device="xpu", dtype=torch.float32) % 4 + 1) * 0.125
    )
    bias = (
        (torch.arange(2 * n, device="xpu", dtype=torch.float32) * 0.25)[::2]
        if with_bias else None
    )

    expected = gemm.float() * x_scale.float() * w_scale.float().reshape(1, -1)
    if bias is not None:
        expected = expected + bias.to(dtype).float()
    output = scaleback(gemm, x_scale, w_scale, bias, dtype_code)

    assert output.shape == (2, n)
    assert output.dtype == dtype
    assert output.device == gemm.device
    torch.testing.assert_close(output, expected.to(dtype), rtol=0, atol=0)
