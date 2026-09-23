"""Independent W8A16 reference, native I=80/160 and asynchronous ownership.

QWEN38_FP8_MOE_SO selects an isolated build without loading the installed package.
"""
import importlib
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def op():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.set_num_threads(1)
    path = os.getenv("QWEN38_FP8_MOE_SO")
    if path:
        spec = importlib.util.spec_from_file_location("qwen38_fp8_moe_ops", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("custom_esimd_kernels_vllm.qwen38_fp8_moe_ops")


@pytest.fixture(scope="module", params=[80, 160])
def weights(request):
    i = request.param
    g = torch.Generator().manual_seed(113+i)

    def rand(*shape):
        return torch.randn(shape, generator=g)

    def fp8(*shape):
        # Includes E4M3 subnormals; the reference does not flush them.
        return (rand(*shape)*12).to(torch.float8_e4m3fn)

    cpu = [
        fp8(512, 2560, 2*i), torch.rand(512, generator=g)*0.001+0.001,
        fp8(512, i, 2560), torch.rand(512, generator=g)*0.001+0.001,
        fp8(2*i, 2560), torch.tensor([0.0015]),
        fp8(2560, i), torch.tensor([0.0015]),
        (rand(1, 2560)*0.02).half(),
    ]
    router = (rand(512, 2560)*0.02).half()
    return cpu, router, [w.to("xpu") for w in cpu], router.to("xpu")


def reference(x, rw, w):
    x = x.float()
    logits = (x @ rw.float().T).half().float()
    selected = torch.argsort(logits, descending=True, stable=True)[:, :10]
    probs = torch.softmax(logits.gather(1, selected), dim=-1)
    m, h = x.shape
    i = w[0].shape[-1]//2
    output = torch.zeros(m, h)
    for t in range(m):
        for r in range(10):
            e = selected[t, r].item()
            projected = (x[t] @ w[0][e].float()*w[1][e]).half().float()
            intermediate = (F.silu(projected[:i])*projected[i:]).half().float()
            part = (intermediate @ w[2][e].float()*w[3][e]*probs[t, r]).half()
            output[t] += part.float()
    output = output.half().float()
    gu = (x @ w[4].float().T*w[5]).half().float()
    intermediate = (F.silu(gu[:, :i])*gu[:, i:]).half().float()
    shared = (intermediate @ w[6].float().T*w[7]).half()
    gate = torch.sigmoid((x @ w[8].float().T).half().float()).half()
    return (output+(shared*gate).float()).half()


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 128])
def test_native_width_against_independent_fp32(op, weights, m):
    cpu, rw, device, drw = weights
    x = torch.randn(m, 2560, generator=torch.Generator().manual_seed(72+m)).half()
    expected = reference(x, rw, cpu)
    actual = op.Workspace().try_run(x.to("xpu"), drw, device).cpu()
    assert torch.isfinite(actual).all()
    # FP32 reduction trees may cross FP16 rounding boundaries. This is a tight
    # absolute error budget for O(0.05) outputs, not a token-equality assertion.
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=0.01)


def test_retained_outputs_alternating_shapes_and_streams(op, weights):
    cpu, rw, device, drw = weights
    workspace = op.Workspace()
    streams = [torch.xpu.current_stream(), torch.xpu.Stream()]
    streams[1].wait_stream(streams[0])
    retained = []
    for step, m in enumerate([1, 4, 2, 8, 1, 3, 1]):
        x = torch.randn(m, 2560, generator=torch.Generator().manual_seed(29+step)).half()
        with torch.xpu.stream(streams[step % 2]):
            dx = x.to("xpu")
            out = workspace.try_run(dx, drw, device)
            retained.append((out, reference(x, rw, cpu)))
    torch.xpu.synchronize()
    assert len({out.data_ptr() for out, _ in retained}) == len(retained)
    for out, expected in retained:
        torch.testing.assert_close(out.cpu(), expected, atol=3e-4, rtol=0.01)


def test_ineligible_call_does_not_change_live_output(op, weights):
    _, _, device, drw = weights
    workspace = op.Workspace()
    x = torch.randn(1, 2560, device="xpu", dtype=torch.float16)
    out = workspace.try_run(x, drw, device)
    snapshot = out.clone()
    bad = list(device)
    bad[7] = bad[7].double()
    assert workspace.try_run(x, drw, bad) is None
    assert workspace.try_run(x[:, ::2], drw, device) is None
    assert workspace.try_run(x, drw, device[:-1]) is None
    unaligned = torch.empty(2561, device="xpu", dtype=torch.float16)[1:].view(1, 2560)
    assert workspace.try_run(unaligned, drw, device) is None
    torch.testing.assert_close(out, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("grouped", [0, 1])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_router_does_not_index_past_experts(op, weights, m, grouped, bad_value):
    _, _, device, drw = weights
    workspace = op.Workspace(grouped_up=grouped)
    x = torch.ones(m, 2560, device="xpu", dtype=torch.float16)
    bad_router = torch.full_like(drw, bad_value)
    invalid = workspace.try_run(x, bad_router, device)
    # Non-finite logits need not produce finite output, but cannot corrupt the
    # device or the next valid call. In particular, do not hide NaNs as zeros.
    torch.xpu.synchronize()
    assert torch.isnan(invalid).all()
    recovered = workspace.try_run(x, drw, device)
    reference_out = op.Workspace(grouped_up=0).try_run(x, drw, device)
    torch.testing.assert_close(recovered, reference_out, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("pattern", ["random", "correlated", "identical"])
def test_grouped_up_matches_route_order_and_rounding(op, weights, m, pattern):
    cpu, rw, device, drw = weights
    x = torch.randn(m, 2560, generator=torch.Generator().manual_seed(410+m)).half()
    if pattern == "identical":
        x[:] = x[0].clone()
    elif pattern == "correlated":
        for row in range(1, m, 2):
            x[row] = (0.8*x[row-1].float() + 0.6*x[row].float()).half()
    dx = x.to("xpu")
    plain = op.Workspace(grouped_up=0).try_run(dx, drw, device).cpu()
    grouped = op.Workspace(grouped_up=1).try_run(dx, drw, device).cpu()
    torch.testing.assert_close(grouped, reference(x, rw, cpu), atol=3e-4, rtol=0.01)
    torch.testing.assert_close(grouped, plain, atol=0, rtol=0)


def test_grouped_async_owner_and_environment(op, weights, monkeypatch):
    monkeypatch.setenv("VLLM_XPU_QWEN38_FP8_MOE_GROUPED_UP", "1")
    cpu, rw, device, drw = weights
    workspace = op.Workspace()
    streams = [torch.xpu.current_stream(), torch.xpu.Stream()]
    streams[1].wait_stream(streams[0])
    retained = []
    for step, m in enumerate([1, 5, 2, 8, 3, 1]):
        x = torch.randn(m, 2560, generator=torch.Generator().manual_seed(441+step)).half()
        with torch.xpu.stream(streams[step % 2]):
            out = workspace.try_run(x.to("xpu"), drw, device)
            retained.append((out, reference(x, rw, cpu)))
    torch.xpu.synchronize()
    assert len({out.data_ptr() for out, _ in retained}) == len(retained)
    for out, expected in retained:
        torch.testing.assert_close(out.cpu(), expected, atol=3e-4, rtol=0.01)
    monkeypatch.setenv("VLLM_XPU_QWEN38_FP8_MOE_GROUPED_UP", "invalid")
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        op.Workspace()
