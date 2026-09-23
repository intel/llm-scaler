"""PLE multi-row reductions preserve the generic ABI and M1 numerical path."""

import os

import pytest
import torch


@pytest.fixture(scope="module")
def device():
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    torch.ops.load_library(os.environ["HC_CHAIN_DSO"])
    assert torch._C._jit_get_schemas_for_operator(
        "custom_esimd_kernels_vllm::ple_grouped_norm"
    )
    return torch.device("xpu:0")


def reference(x, weight, eps=1e-6, group_size=2560):
    grouped = x.float().reshape(-1, x.shape[-1] // group_size, group_size)
    result = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return (result.reshape_as(x) * (1 + weight.float())).half()


def inputs(rows, device, scale=0.25, offset=0, width=10240):
    generator = torch.Generator(device="cpu").manual_seed(38300 + rows)
    def value(shape, multiplier):
        source = (torch.randn(shape, generator=generator) * multiplier).half()
        storage = torch.full((source.numel() + offset + 2,), 17.0,
                             device=device, dtype=torch.float16)
        result = storage[offset:offset + source.numel()].view(shape)
        result.copy_(source)
        return storage, result
    xb, x = value((rows, width), scale)
    wb, weight = value((width,), 0.1)
    ob, out = value((rows, width), 0.1)
    return (x, weight, out), (xb, wb, ob)


def operation(*args):
    return torch.ops.custom_esimd_kernels_vllm.ple_grouped_norm(*args)


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("scale", (0.001, 0.25, 8.0))
@pytest.mark.parametrize("offset", (0, 2))
def test_multi_score_matches_torch_and_m1(device, rows, scale, offset):
    args, backing = inputs(rows, device, scale=scale, offset=offset)
    key = args[0]
    query = key.flip(1).contiguous()
    output = torch.full((rows, 4), -1., device=device, dtype=torch.float16)
    op = torch.ops.custom_esimd_kernels_vllm.ple_score_gate
    snapshots = [x.clone() for x in (key, query)]
    op(key, query, output, 2560)
    rowwise = torch.empty_like(output)
    for row in range(rows):
        op(key[row:row + 1], query[row:row + 1], rowwise[row:row + 1], 2560)
    torch.testing.assert_close(output, rowwise, rtol=0, atol=0)
    score = (key.float() * query.float()).reshape(rows, 4, 2560).sum(-1)
    score = score / 2560**0.5
    expected = torch.sigmoid(
        torch.sign(score) * score.abs().clamp_min(1e-6).sqrt()
    ).half()
    torch.testing.assert_close(output, expected, rtol=0.002, atol=0.0005)
    for actual, snapshot in zip((key, query), snapshots):
        torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("rows", (1, 5, 8, 9))
def test_score_zero_clamp_nan_and_odd_offset_fallback(device, rows):
    key = torch.zeros((rows, 10240), device=device, dtype=torch.float16)
    query = torch.ones_like(key)
    key[:, 0] = 1e-6
    key[:, 2560] = -1e-6
    key[:, 7680] = float("nan")
    odd = torch.empty(key.numel() + 1, device=device, dtype=key.dtype)[1:]
    odd = odd.view_as(key)
    odd.copy_(key)
    outputs = [torch.empty((rows, 4), device=device, dtype=key.dtype) for _ in range(2)]
    op = torch.ops.custom_esimd_kernels_vllm.ple_score_gate
    op(key, query, outputs[0], 2560)
    op(odd, query, outputs[1], 2560)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert torch.equal(outputs[0][:, 2], torch.full_like(outputs[0][:, 2], 0.5))


def test_multi_score_current_stream_and_disjoint_inflight_outputs(device):
    current = torch.xpu.current_stream(device)
    records = []
    for rows in (2, 5, 8):
        stream = torch.xpu.Stream(device=device)
        stream.wait_stream(current)
        with torch.xpu.stream(stream):
            key = torch.randn((rows, 10240), device=device, dtype=torch.float16)
            query = torch.randn_like(key)
            output = torch.empty((rows, 4), device=device, dtype=key.dtype)
            expected = torch.empty_like(output)
            op = torch.ops.custom_esimd_kernels_vllm.ple_score_gate
            op(key, query, output, 2560)
            for row in range(rows):
                op(key[row:row + 1], query[row:row + 1], expected[row:row + 1], 2560)
            records.append((stream, key, query, output, expected))
    for stream, key, query, output, expected in records:
        stream.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("scale", (0.001, 0.25, 8.0))
@pytest.mark.parametrize("offset", (0, 2))
def test_multi_matches_torch_and_old_m1_rowwise(device, rows, scale, offset):
    args, backing = inputs(rows, device, scale, offset)
    snapshots = [x.clone() for x in backing]
    x, weight, out = args
    expected = reference(x, weight)
    assert operation(*args, 1e-6, 2560) is None
    torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
    old = torch.empty_like(out)
    for row in range(rows):
        operation(x[row:row+1], weight, old[row:row+1], 1e-6, 2560)
    torch.testing.assert_close(out, old, rtol=0, atol=0)
    for actual, snapshot in zip(backing[:2], snapshots[:2]):
        torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(backing[2][:offset], snapshots[2][:offset], rtol=0, atol=0)
    torch.testing.assert_close(backing[2][-2:], snapshots[2][-2:], rtol=0, atol=0)


@pytest.mark.parametrize("rows,offset,width,group_size", [
    (1, 0, 10240, 2560), (9, 0, 10240, 2560),
    (5, 1, 10240, 2560), (5, 0, 5120, 2560), (5, 0, 10240, 1280),
])
def test_generic_and_m1_contracts_remain_available(device, rows, offset, width, group_size):
    args, _ = inputs(rows, device, offset=offset, width=width)
    operation(*args, 1e-6, group_size)
    torch.testing.assert_close(args[2], reference(*args[:2], group_size=group_size),
                               rtol=0.002, atol=0.002)


def test_multidimensional_leading_batch_is_flattened(device):
    args, _ = inputs(8, device)
    x, weight, out = args
    operation(x.view(2, 4, 10240), weight, out.view(2, 4, 10240), 1e-6, 2560)
    torch.testing.assert_close(out, reference(x, weight), rtol=0.002, atol=0.002)


def test_current_stream_and_two_streams(device):
    current = torch.xpu.current_stream(device)
    streams = [torch.xpu.Stream(device=device) for _ in range(2)]
    records = []
    for index, stream in enumerate(streams):
        stream.wait_stream(current)
        with torch.xpu.stream(stream):
            args, _ = inputs(index + 5, device)
            expected = reference(*args[:2])
            operation(*args, 1e-6, 2560)
            consumed = args[2].float().square()
            records.append((args[2], expected, consumed))
    for stream in streams:
        current.wait_stream(stream)
    for out, expected, consumed in records:
        torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
        torch.testing.assert_close(consumed, out.float().square(), rtol=0, atol=0)


@pytest.mark.parametrize("rows,offset,native_multi", [(5, 0, True), (5, 1, False), (9, 0, False)])
def test_generic_abi_reaches_intended_kernel(device, rows, offset, native_multi):
    args, _ = inputs(rows, device, offset=offset)
    operation(*args, 1e-6, 2560)
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.XPU]) as prof:
        operation(*args, 1e-6, 2560)
        torch.xpu.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.XPU]
    assert len(kernels) == 1, [e.name for e in kernels]
    assert ("HcGroupedNormMultiMKernel" in kernels[0].name) == native_multi


@pytest.mark.parametrize("index", range(3))
def test_invalid_dtype_rejected_before_output_write(device, index):
    args, _ = inputs(5, device)
    args = list(args)
    args[index] = args[index].float()
    snapshots = [x.clone() for x in args]
    with pytest.raises(RuntimeError):
        operation(*args, 1e-6, 2560)
    for actual, expected in zip(args, snapshots):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
