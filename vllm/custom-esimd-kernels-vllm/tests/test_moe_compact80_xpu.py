"""Contract, tail and concurrent-stream regressions for TP8/TP4 compact ABI."""

import pytest
import torch
from test_moe_m1_asymmetric_v1_xpu import _dso_path
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def load_dso():
    torch.set_num_threads(4)
    torch.ops.load_library(str(_dso_path()))


@pytest.fixture(scope="module", params=[80, 160])
def weights(request):
    k = request.param
    generator = torch.Generator().manual_seed(20260906)
    w = torch.randint(
        -128, 128, (512, 2560, k // 2), generator=generator, dtype=torch.int8
    ).to("xpu")
    groups = (k + 127) // 128
    scales = (
        (torch.rand(512, 2560, groups, generator=generator) * 0.03).half().to("xpu")
    )
    padded = torch.zeros(512, 2560, groups * 64, dtype=torch.int8, device="xpu")
    padded[..., : k // 2].copy_(w)
    torch.xpu.synchronize()
    return w, scales, padded


def make_inputs(m, k=80):
    generator = torch.Generator().manual_seed(m + 6)
    x = (torch.randn(m, k, generator=generator) * 0.05).half().to("xpu")
    ids = torch.randint(0, 512, (m,), generator=generator)
    counts = torch.bincount(ids, minlength=512).int().to("xpu")
    return x, counts


def reference(x, counts, weights):
    _, scales, padded = weights
    k = padded.shape[-1] * 2
    a = torch.zeros(x.shape[0], k, dtype=x.dtype, device=x.device)
    a[:, : x.shape[1]].copy_(x)
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
        K=k,
        num_experts=512,
    )
    return out


@pytest.mark.parametrize("m", [0, 1, 2, 15, 16, 17, 33, 128, 129, 1024, 4096])
def test_down_bitwise(m, weights):
    k = weights[0].shape[-1] * 2
    x, counts = make_inputs(m, k)
    out = getattr(torch.ops.moe_int4_ops, f"moe_compact{k}_down_grouped_gemm")(
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
    k = weights[0].shape[-1] * 2
    x = torch.empty(k + 1, dtype=torch.float16, device="xpu")[1:].reshape(1, k)
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="64-byte"):
        getattr(torch.ops.moe_int4_ops, f"moe_compact{k}_down_grouped_gemm")(
            x, weights[0], weights[1], counts
        )


def test_down_rejects_wrong_scale_shape(weights):
    k = weights[0].shape[-1] * 2
    x, counts = make_inputs(1, k)
    with pytest.raises(RuntimeError, match="weight/scale shapes"):
        getattr(torch.ops.moe_int4_ops, f"moe_compact{k}_down_grouped_gemm")(
            x, weights[0], weights[1][..., :0], counts
        )


def test_concurrent_streams(weights):
    k = weights[0].shape[-1] * 2
    inputs = [make_inputs(33, k), make_inputs(129, k)]
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
                outputs[i] = getattr(
                    torch.ops.moe_int4_ops, f"moe_compact{k}_down_grouped_gemm"
                )(x, weights[0], weights[1], counts)
    for stream in streams:
        parent.wait_stream(stream)
    torch.xpu.synchronize()
    for out, ref in zip(outputs, expected):
        assert torch.equal(out, ref)
