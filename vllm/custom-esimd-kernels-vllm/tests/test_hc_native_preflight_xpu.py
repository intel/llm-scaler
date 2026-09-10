"""HC live-metadata preflight: no cached weights, no submit on rejection."""

import pytest
import torch
from test_hc_host_chain_xpu import (
    EPS,
    _make_inputs,
    _make_outputs,
    _run_four_existing_ops,
)
from test_hc_host_chain_xpu import (
    _load_canonical_main as _load_canonical_main,  # noqa: PLC0414 - pytest fixture
)
from test_hc_host_chain_xpu import (
    device as device,  # noqa: PLC0414 - pytest fixture
)


def workspace():
    value = torch.classes.custom_esimd_kernels_vllm.HCWorkspace()
    assert callable(value.try_run), "test requires the new build-only DSO"
    return value


def expected(inputs):
    outputs = _make_outputs(inputs[0].device)
    _run_four_existing_ops(inputs, outputs)
    return outputs[0], outputs[4], outputs[2][:, 320:324]


def exact(actual, reference):
    assert actual is not None
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_live_parameter_rebinding_and_recurrent_alias(device):
    args = list(_make_inputs(device))
    args[4] = torch.nn.Parameter(args[4], requires_grad=False)
    ws = workspace()
    exact(ws.try_run(*args, EPS), expected(args))
    torch.xpu.synchronize()
    # set_data does not have to increment the version counter. A cached view
    # would use the previous allocation and fail this exact comparison.
    old = args[4].detach()
    args[4].data = args[4].detach().clone().mul_(0.5)
    assert args[4].data_ptr() != old.data_ptr()
    actual = ws.try_run(*args, EPS)
    exact(actual, expected(args))
    recurrent = (actual[0], args[1], actual[2], *args[3:])
    result = ws.try_run(*recurrent, EPS)
    exact(result, expected(recurrent))
    assert result[0].data_ptr() != actual[0].data_ptr()


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "cpu", "alignment", "negative", "strides"]
)
def test_rejects_before_allocation_and_preserves_outputs(device, index, kind):
    args = list(_make_inputs(device))
    ws = workspace()
    borrowed = ws.try_run(*args, EPS)
    snapshots = [x.clone() for x in borrowed]
    t = args[index]
    if kind == "shape":
        args[index] = t.flatten()[:1]
    elif kind == "dtype":
        args[index] = t.float()
    elif kind == "cpu":
        args[index] = t.cpu()
    elif kind == "alignment":
        args[index] = torch.empty(t.numel() + 1, dtype=t.dtype, device=device)[1:].view(
            t.shape
        )
    elif kind == "strides":
        args[index] = torch.empty((t.numel(), 2), dtype=t.dtype, device=device)[
            :, 0
        ].view(t.shape)
    else:
        args[index] = torch._neg_view(t)
    torch.xpu.synchronize()
    allocated = torch.xpu.memory_allocated()
    assert ws.try_run(*args, EPS) is None
    assert torch.xpu.memory_allocated() == allocated
    exact(borrowed, snapshots)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan"), 1e-100])
def test_invalid_eps_is_pre_submit_rejection(device, eps):
    assert workspace().try_run(*_make_inputs(device), eps) is None


def test_current_stream_producer_consumer_and_reuse(device):
    args = _make_inputs(device)
    reference = expected(args)
    parent = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    ws = workspace()
    outputs = []
    keep_alive = []
    for stream in streams:
        stream.wait_stream(parent)
        with torch.xpu.stream(stream):
            local = [torch.empty_like(x) for x in args]
            keep_alive.append(local)
            for _ in range(8):
                for target, source in zip(local, args):
                    target.copy_(source)
                result = ws.try_run(*local, EPS)
                outputs.append(tuple(x.clone() for x in result))
        parent.wait_stream(stream)
    for result in outputs:
        exact(result, reference)
