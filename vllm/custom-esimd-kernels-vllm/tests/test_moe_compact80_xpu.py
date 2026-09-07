"""Contract, tail and concurrent-stream regressions for the compact80 ABI."""

import pytest
import torch
from test_moe_m1_asymmetric_v1_xpu import _dso_path
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def load_dso():
    torch.set_num_threads(4)
    torch.ops.load_library(str(_dso_path()))


@pytest.fixture(scope="module")
def weights():
    generator = torch.Generator().manual_seed(20260906)
    w = torch.randint(
        -128, 128, (512, 2560, 40), generator=generator, dtype=torch.int8
    ).to("xpu")
    scales = (torch.rand(512, 2560, 1, generator=generator) * 0.03).half().to("xpu")
    padded = torch.zeros(512, 2560, 64, dtype=torch.int8, device="xpu")
    padded[..., :40].copy_(w)
    torch.xpu.synchronize()
    return w, scales, padded


def make_inputs(m):
    generator = torch.Generator().manual_seed(m + 6)
    x = (torch.randn(m, 80, generator=generator) * 0.05).half().to("xpu")
    ids = torch.randint(0, 512, (m,), generator=generator)
    counts = torch.bincount(ids, minlength=512).int().to("xpu")
    return x, counts


def reference(x, counts, weights):
    _, scales, padded = weights
    a = torch.zeros(x.shape[0], 128, dtype=x.dtype, device=x.device)
    a[:, :80].copy_(x)
    out = torch.empty(x.shape[0], 2560, dtype=x.dtype, device=x.device)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=a,
        ptr_A_scale=None,
        ptr_B=padded,
        ptr_B_scale=scales,
        ptr_bias=None,
        ptr_D=out,
        rows_per_expert=counts,
        N=2560,
        K=128,
        num_experts=512,
    )
    return out


@pytest.mark.parametrize("m", [0, 1, 2, 15, 16, 17, 33, 128, 129, 1024, 4096])
def test_down_bitwise(m, weights):
    x, counts = make_inputs(m)
    out = torch.ops.moe_int4_ops.moe_compact80_down_grouped_gemm(
        x, weights[0], weights[1], counts
    )
    if not m:
        assert out.shape == (0, 2560)
        torch.xpu.synchronize()
        return
    expected = reference(x, counts, weights)
    torch.xpu.synchronize()
    assert torch.equal(out, expected)


def test_down_rejects_misaligned_input(weights):
    x = torch.empty(81, dtype=torch.float16, device="xpu")[1:].reshape(1, 80)
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="64-byte"):
        torch.ops.moe_int4_ops.moe_compact80_down_grouped_gemm(
            x, weights[0], weights[1], counts
        )


def test_down_rejects_wrong_scale_shape(weights):
    x, counts = make_inputs(1)
    with pytest.raises(RuntimeError, match="weight/scale shapes"):
        torch.ops.moe_int4_ops.moe_compact80_down_grouped_gemm(
            x, weights[0], weights[1][..., :0], counts
        )


def test_concurrent_streams(weights):
    inputs = [make_inputs(33), make_inputs(129)]
    expected = [reference(x, counts, weights) for x, counts in inputs]
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    outputs = [None, None]
    parent = torch.xpu.current_stream()
    for stream in streams:
        stream.wait_stream(parent)
    for _ in range(30):
        for i, stream in enumerate(streams):
            with torch.xpu.stream(stream):
                x, counts = inputs[i]
                outputs[i] = torch.ops.moe_int4_ops.moe_compact80_down_grouped_gemm(
                    x, weights[0], weights[1], counts
                )
    for stream in streams:
        parent.wait_stream(stream)
    torch.xpu.synchronize()
    for out, ref in zip(outputs, expected):
        assert torch.equal(out, ref)
