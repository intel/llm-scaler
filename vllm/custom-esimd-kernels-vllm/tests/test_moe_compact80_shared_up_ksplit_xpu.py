"""compact80 shared UP K-split4 候选；只由主线程授权后串行运行。

使用 MOE_INT4_DSO 指定新 DSO。运行解释器约定：
uv run --no-project /opt/venv/bin/python -m pytest <本文件>
不自动编译/安装，不计时；M2..6默认启用，完整 MoE 测试临时切换开关。
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

H, I = 2560, 80
SWITCH = "VLLM_XPU_MOE_COMPACT80_SHARED_UP_KSPLIT4"
# 在实测之前固定门槛；K-split FP32 归约不承诺 FP16 bitexact。
UP_ATOL, UP_RTOL = 2e-4, 1e-3
GATE_ATOL, GATE_RTOL = 2e-6, 2e-5


@pytest.fixture(scope="module", autouse=True)
def load_dso():
    if not torch.xpu.is_available():
        pytest.skip("需要 XPU")
    configured = os.environ.get("MOE_INT4_DSO")
    if configured:
        dso = Path(configured)
    else:
        package = Path(__file__).parents[1] / "python/custom_esimd_kernels_vllm"
        matches = tuple(package.glob("moe_int4_ops*.so"))
        if len(matches) != 1:
            raise RuntimeError("请用 MOE_INT4_DSO 指定唯一的新 DSO")
        dso = matches[0]
    torch.ops.load_library(str(dso))
    # 新符号缺失应失败，不能把旧 DSO 静默当成 skip/pass。
    assert torch._C._jit_get_schemas_for_operator(
        "moe_int4_ops::_moe_compact80_shared_up_ab_out_v1"
    )


def inputs(m, scale=0.3, seed=20260908, offset=0):
    generator = torch.Generator().manual_seed(seed + m)

    def make(shape, amplitude):
        cpu = (torch.randn(shape, generator=generator) * amplitude).half()
        base = torch.empty(cpu.numel() + offset, device="xpu", dtype=torch.float16)
        value = base[offset:].view(shape)
        value.copy_(cpu)
        return value

    return make((m, H), scale), make((2 * I, H), 0.12), make((1, H), 0.04)


def outputs(m):
    return (
        torch.full((m, I), 17.0, device="xpu", dtype=torch.float16),
        torch.full((m, 1), -19.0, device="xpu", dtype=torch.float32),
    )


def call(args, out, candidate):
    pointers = tuple(t.data_ptr() for t in out)
    result = torch.ops.moe_int4_ops._moe_compact80_shared_up_ab_out_v1(
        *args, *out, candidate
    )
    assert result is None
    assert tuple(t.data_ptr() for t in out) == pointers
    return out


def reference(args):
    # 独立数学：不能先把 linear 或 SiLU 结果转 FP16。
    x, weight, gate_weight = args
    projected = F.linear(x.float(), weight.float())
    intermediate = (F.silu(projected[:, :I]) * projected[:, I:]).half()
    gate = torch.sigmoid(F.linear(x.float(), gate_weight.float()))
    return intermediate, gate


def assert_stage_close(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], atol=UP_ATOL, rtol=UP_RTOL)
    torch.testing.assert_close(actual[1], expected[1], atol=GATE_ATOL, rtol=GATE_RTOL)


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("scale", [0.03, 0.3, 1.5])
@pytest.mark.parametrize("offset", [0, 8])
def test_shared_up_independent_reference_and_old_path(m, scale, offset):
    args = inputs(m, scale, offset=offset)
    saved = [value.clone() for value in args]
    expected = reference(args)
    old = call(args, outputs(m), False)
    candidate = call(args, outputs(m), True)
    torch.xpu.synchronize()
    assert_stage_close(old, expected)
    assert_stage_close(candidate, expected)
    assert_stage_close(candidate, old)
    for value, snapshot in zip(args, saved):
        assert torch.equal(value, snapshot)


@pytest.mark.parametrize("m", range(2, 9))
def test_split_boundaries_last_column_and_signed_gate(m):
    args = inputs(m)
    x, weight, gate_weight = args
    x.zero_()
    edges = [0, 639, 640, 1279, 1280, 1919, 1920, 2559]
    for row in range(m):
        x[row, edges] = torch.tensor(
            [1, -2, 3, -4, 5, -6, 7, -8], device="xpu", dtype=x.dtype
        ) * (row + 1) / 8
    # shared gate 既覆盖正负，又保持 intermediate 的 gate/up 权重独立。
    gate_weight.zero_()
    gate_weight[0, edges] = 0.125
    x[1::2].neg_()
    weight[I - 1].fill_(0.03125)
    weight[-1].fill_(-0.0625)
    expected = reference(args)
    actual = call(args, outputs(m), True)
    torch.xpu.synchronize()
    assert_stage_close(actual, expected)
    assert (actual[1] < 0.5).any() and (actual[1] > 0.5).any()


def test_two_streams_producers_consumers_reuse_and_feedback():
    parent = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    base = [inputs(5, seed=41), inputs(8, seed=73)]
    records = []
    keepalive = []
    for stream in streams:
        stream.wait_stream(parent)
    for iteration in range(4):
        for index, stream in enumerate(streams):
            with torch.xpu.stream(stream):
                m = [2, 5, 3, 5][iteration] if index == 0 else [8, 4, 7, 6][iteration]
                # producer 与消费计算在同一非默认流，无中途 host synchronize。
                x = base[index][0][:m].clone().mul_(0.5)
                args = (x, base[index][1], base[index][2])
                out = outputs(m)
                expected = reference(args)
                call(args, out, True)
                records.append((tuple(t.clone() for t in out), expected))
                feedback = (out[0].repeat(1, H // I) * 0.03, *args[1:])
                feedback_expected = reference(feedback)
                call(feedback, out, True)  # 在消费旧输出之后复用同一输出 allocation。
                records.append((tuple(t.clone() for t in out), feedback_expected))
                keepalive.extend((args, out, feedback))
    for stream in streams:
        parent.wait_stream(stream)
    torch.xpu.synchronize()
    for actual, expected in records:
        assert_stage_close(actual, expected)


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("bad", [
    "dtype", "weight_shape", "weight_stride", "offset", "cpu_weight",
    "lazy_input", "lazy_output", "lazy_weight", "lazy_gate_weight",
    "lazy_gate_output", "lazy_both_outputs", "output_resize", "output_set",
    "input_alias", "output_alias", "dlpack_alias", "dlpack_offset",
])
def test_bad_contracts_leave_all_storage_unchanged(candidate, bad):
    m = 5
    args, out = list(inputs(m)), list(outputs(m))
    backing = []
    if bad == "dtype":
        args[0] = args[0].float()
    elif bad == "weight_shape":
        args[1] = args[1][:-1]
    elif bad == "weight_stride":
        args[1] = args[1].t().contiguous().t()
    elif bad in ("offset", "dlpack_offset"):
        owner = torch.empty(m * H + 1, dtype=torch.float16, device="xpu")
        owner.fill_(0.125)
        view = owner[1:].view(m, H)
        args[0] = torch.utils.dlpack.from_dlpack(view) if bad == "dlpack_offset" else view
        backing.append(owner)
    elif bad == "cpu_weight":
        args[2] = args[2].cpu()
    elif bad == "lazy_input":
        args[0] = torch._neg_view(args[0])
    elif bad == "lazy_output":
        out[0] = torch._neg_view(out[0])
    elif bad == "lazy_weight":
        args[1] = torch._neg_view(args[1])
    elif bad == "lazy_gate_weight":
        args[2] = torch._neg_view(args[2])
    elif bad == "lazy_gate_output":
        out[1] = torch._neg_view(out[1])
    elif bad == "lazy_both_outputs":
        out = [torch._neg_view(t) for t in out]
    elif bad == "output_resize":
        out[0].resize_(m, I - 1)
    elif bad == "output_set":
        owner = torch.full((m * I + 1,), 3.0, dtype=torch.float16, device="xpu")
        out[0].set_(owner.untyped_storage(), 1, (m, I), (I, 1))
        backing.append(owner)
    elif bad in ("input_alias", "dlpack_alias"):
        view = args[0].view(-1)[:m * I].view(m, I)
        out[0] = torch.utils.dlpack.from_dlpack(view) if bad == "dlpack_alias" else view
    elif bad == "output_alias":
        out[1] = out[0].view(torch.float32).view(-1)[:m].view(m, 1)
    tracked = args + out + backing
    snapshots = [t.clone() for t in tracked]
    torch.xpu.synchronize()
    # 不接受通用 Neg fallback 在 kernel 执行后抛出的回写/栈错误。
    # 必须由本 op 的原始 tensor preflight 拒绝，且所有 storage 保持不变。
    message = "compact80 shared UP A/B rejects lazy neg/conj tensors" if bad.startswith("lazy_") else None
    with pytest.raises(RuntimeError, match=message):
        call(args, out, candidate)
    torch.xpu.synchronize()
    for tensor, snapshot in zip(tracked, snapshots):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("m", [0, 1, 9])
def test_stage_rejects_out_of_range_m(m):
    args, out = inputs(m), outputs(m)
    with pytest.raises(RuntimeError, match="2..8 tokens"):
        call(args, out, True)


@pytest.fixture(scope="module")
def full_inputs():
    # 复用既有 compact80 全量随机 INT4 权重 fixture 和独立 dequant oracle。
    from test_moe_compact80_grouped_xpu import build_full_inputs

    return build_full_inputs()


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("grouped", [False, True], ids=["non_grouped", "grouped"])
def test_full_compact80_switch_and_independent_oracle(m, grouped, full_inputs, monkeypatch):
    from test_moe_compact80_grouped_xpu import (
        OUTPUT_ATOL, _grouped_op, _legacy_compact_multi_op, _make_logits, _reference,
    )

    operation = _grouped_op if grouped else _legacy_compact_multi_op
    logits_cpu = _make_logits(m, variant=2)
    logits = logits_cpu.to("xpu")
    expected = _reference(full_inputs, m, logits_cpu).to("xpu")
    out = torch.empty((m, H), device="xpu", dtype=torch.float16)
    snapshot = full_inputs.x.clone()
    monkeypatch.setenv(SWITCH, "0")
    returned = operation(full_inputs, m, logits, out)
    assert returned.data_ptr() == out.data_ptr()
    old = out.clone()
    monkeypatch.setenv(SWITCH, "1")
    returned = operation(full_inputs, m, logits, out)
    assert returned.data_ptr() == out.data_ptr()
    candidate = out.clone()
    monkeypatch.setenv(SWITCH, "0")
    operation(full_inputs, m, logits, out)
    torch.xpu.synchronize()
    assert torch.equal(old, out)
    monkeypatch.delenv(SWITCH, raising=False)
    operation(full_inputs, m, logits, out)
    torch.xpu.synchronize()
    assert torch.equal(out, candidate if m <= 6 else old)
    assert torch.equal(full_inputs.x, snapshot)
    for actual in (old, candidate):
        torch.testing.assert_close(actual, expected, rtol=0, atol=OUTPUT_ATOL)
    torch.testing.assert_close(candidate, old, rtol=0, atol=OUTPUT_ATOL)


def test_full_m1_ignores_candidate_switch(full_inputs, monkeypatch):
    from test_moe_compact80_grouped_xpu import _m1_op, _make_logits

    logits = _make_logits(1).to("xpu")
    out = torch.empty((1, H), device="xpu", dtype=torch.float16)
    monkeypatch.delenv(SWITCH, raising=False)
    old = _m1_op(full_inputs, logits, out).clone()
    monkeypatch.setenv(SWITCH, "1")
    _m1_op(full_inputs, logits, out)
    torch.xpu.synchronize()
    assert torch.equal(old, out)


def test_full_m9_ignores_candidate_switch(full_inputs, monkeypatch):
    from dataclasses import replace
    from test_moe_compact80_grouped_xpu import _legacy_compact_multi_op, _make_logits

    data = replace(full_inputs, x=full_inputs.x[:1].repeat(9, 1))
    logits = _make_logits(9).to("xpu")
    out = torch.empty((9, H), device="xpu", dtype=torch.float16)
    monkeypatch.delenv(SWITCH, raising=False)
    old = _legacy_compact_multi_op(data, 9, logits, out).clone()
    monkeypatch.setenv(SWITCH, "1")
    _legacy_compact_multi_op(data, 9, logits, out)
    torch.xpu.synchronize()
    assert torch.equal(old, out)
