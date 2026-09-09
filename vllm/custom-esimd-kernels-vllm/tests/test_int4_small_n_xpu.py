"""Independent Q4_0 golden and async contract for the TP4/TP8 GDN projection."""

import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load():
    dso = os.getenv("SMALL_N_DSO")
    if not dso:
        pytest.skip("设置SMALL_N_DSO以显式启用XPU kernel测试")
    if not torch.xpu.is_available():
        pytest.skip("需要可用XPU")
    torch.ops.load_library(dso)
    torch.set_num_threads(4)


def case(m, n, amplitude=0.3):
    g = torch.Generator().manual_seed(24012 + m * 31 + n)
    x = (torch.randn(m, 2560, generator=g) * amplitude).half()
    weight = torch.randint(256, (n, 1280), generator=g, dtype=torch.uint8)
    scale = ((torch.rand(n, 20, generator=g) - 0.5) * 0.08).half()
    return x, weight, scale


def reference(args):
    x, w, s = args
    raw = w.int()
    values = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).float() - 8
    # Q4_0 uses unsigned nibbles minus 8, not MoE's signed-S4 encoding.
    dequant = (values * s.float().repeat_interleave(128, -1)).half().float()
    return (x.float() @ dequant.t()).half()


def call(args, output=None):
    x, w, s = args
    if output is None:
        output = torch.empty(x.shape[0], w.shape[0], device="xpu", dtype=torch.float16)
    result = torch.ops.custom_esimd_kernels_vllm.esimd_gemm_int4_small_n_v1(
        x, w, s, output
    )
    assert result.data_ptr() == output.data_ptr()
    return result


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("n", [12, 24])
@pytest.mark.parametrize("amplitude", [0.03, 0.3, 1.5])
def test_independent_dequant_and_fp32_dot(m, n, amplitude):
    args = case(m, n, amplitude)
    expected = reference(args)
    actual = call(tuple(t.to("xpu") for t in args))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=1e-3)


def test_two_streams_producer_and_consumer_ordering():
    parent = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    data = [case(8, n) for n in (12, 24)]
    device = [tuple(t.to("xpu") for t in args) for args in data]
    for s in streams:
        s.wait_stream(parent)
    pending = []
    for m in (2, 7, 4, 8):
        for s, cpu, gpu in zip(streams, data, device):
            with torch.xpu.stream(s):
                x = gpu[0][:m].clone().mul_(0.5)
                y = call((x, gpu[1], gpu[2]))
                pending.append(
                    (y.clone(), reference(((cpu[0][:m] * 0.5).half(), *cpu[1:])))
                )
    for s in streams:
        parent.wait_stream(s)
    for actual, expected in pending:
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "bad", ["rows", "n", "k", "scale", "dtype", "stride", "lazy", "alias", "offset"]
)
def test_reject_before_writing_output(bad):
    args = [t.to("xpu") for t in case(4, 24)]
    output = torch.full((4, 24), 17.0, dtype=torch.float16, device="xpu")
    if bad == "rows":
        args[0] = args[0][:1]
    elif bad == "n":
        args[1] = args[1][:16]
    elif bad == "k":
        args[0] = args[0][:, :1280].contiguous()
    elif bad == "scale":
        args[2] = args[2][:, :1].contiguous()
    elif bad == "dtype":
        args[0] = args[0].float()
    elif bad == "stride":
        args[0] = torch.empty((4, 5120), dtype=torch.float16, device="xpu")[:, ::2]
    elif bad == "lazy":
        args[0] = torch._neg_view(args[0])
    elif bad == "alias":
        output = args[0].flatten()[:96].view(4, 24)
    elif bad == "offset":
        output = torch.full((97,), 17.0, dtype=torch.float16, device="xpu")[1:].view(
            4, 24
        )
    saved = output.clone()
    with pytest.raises(RuntimeError):
        call(args, output)
    torch.testing.assert_close(output, saved, rtol=0, atol=0)
