"""HCMultiMWorkspaceV1 原生测试；由 main 构建后运行，本任务不执行。

HC_CHAIN_DSO 指定 canonical main DSO；不导入 vLLM/Python 集成层。
独立 Torch 完整链路 golden 与当前三算子逐步 exact 对拍同时作为门禁。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


WIDTH, HIDDEN, RANK, DOWN_WIDTH = 10240, 2560, 320, 336
EPS = 1.0e-6
COMBINE = "hc_combine_norm_multi_m_strided_v1"
DOWN = "esimd_hc_down_fp16_multi_m_out_v1"
UP = "esimd_hc_up_gate_mix_multi_m_v1"


@pytest.fixture(scope="module")
def device():
    if not torch.xpu.is_available():
        pytest.skip("XPU 不可用")
    configured = os.environ.get("HC_CHAIN_DSO")
    if configured:
        dso = Path(configured)
        if not dso.is_file():
            pytest.fail(f"HC_CHAIN_DSO 不存在: {dso}")
        torch.ops.load_library(str(dso))
    else:
        import custom_esimd_kernels_vllm  # noqa: F401

    for name in (COMBINE, DOWN, UP):
        schema = f"custom_esimd_kernels_vllm::{name}"
        if not torch._C._jit_get_schemas_for_operator(schema):
            pytest.fail(f"canonical main DSO 未注册 {schema}")
    # 新 ABI 缺失是失败，不静默 skip/退回旧 M1 workspace。
    try:
        torch.classes.custom_esimd_kernels_vllm.HCMultiMWorkspaceV1
    except RuntimeError as exc:
        pytest.fail(f"canonical main DSO 缺少 HCMultiMWorkspaceV1: {exc}")
    return torch.device("xpu:0")


def _workspace():
    return torch.classes.custom_esimd_kernels_vllm.HCMultiMWorkspaceV1()


def _op(name):
    return getattr(torch.ops.custom_esimd_kernels_vllm, name)


def _rand(shape, device, generator, scale):
    return (
        torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        * scale
    ).half()


@pytest.fixture(scope="module")
def weights(device):
    generator = torch.Generator(device=device).manual_seed(9100)
    return (
        _rand((WIDTH,), device, generator, 0.03),
        _rand((DOWN_WIDTH, WIDTH), device, generator, 1.0 / math.sqrt(WIDTH)),
        _rand((WIDTH, RANK), device, generator, 1.0 / math.sqrt(RANK)),
    )


def _inputs(device, weights, rows=5, seed=9101, scale=0.25):
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = _rand((rows, WIDTH), device, generator, scale)
    block = _rand((rows, HIDDEN), device, generator, scale)
    merged = _rand((rows, DOWN_WIDTH), device, generator, 0.5)
    injection = merged[:, RANK:RANK + 4]
    assert injection.stride() == (DOWN_WIDTH, 1)
    assert injection.storage_offset() == RANK
    return hidden, block, injection, *weights


def _torch_reference(args, eps=EPS):
    hidden, block, injection, norm_weight, down_weight, up_weight = args
    rows = hidden.shape[0]
    # The established native combine contracts FP32 multiply+add into FMA.
    # Use an independent Torch FMA oracle, retaining the FP16 boundary before
    # RMSNorm. Separate Torch multiply/add has another FP32 rounding point and
    # can cross a half-way boundary (seed9203, scale1.5); do not change the old
    # kernel or relax tolerances to conceal that arithmetic distinction.
    combined = torch.addcmul(
        hidden.float().view(rows, 4, HIDDEN),
        block.float().unsqueeze(1),
        (2.0 * torch.sigmoid(injection.float() / 4.0)).unsqueeze(-1),
    ).half()
    inverse_rms = torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + eps)
    normed = (
        combined.float() * inverse_rms
        * (1.0 + norm_weight.float().view(1, 4, HIDDEN))
    ).half().view(rows, WIDTH)
    linear = F.linear(normed, down_weight).half()
    scaled = (linear[:, :RANK] / 4.0).half()
    activated = F.silu(scaled.float()).half()
    gate = F.linear(activated, up_weight).half()
    mixed = (
        (torch.sigmoid(gate.float()) * normed.float())
        .view(rows, 4, HIDDEN).mean(1).half()
    )
    return combined.view(rows, WIDTH), mixed, linear[:, RANK:RANK + 4]


def _sequential(args, eps=EPS):
    hidden, block, injection, norm_weight, down_weight, up_weight = args
    rows = hidden.shape[0]
    combined = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    down = torch.empty((rows, DOWN_WIDTH), device=hidden.device, dtype=torch.float16)
    mixed = torch.empty((rows, HIDDEN), device=hidden.device, dtype=torch.float16)
    assert _op(COMBINE)(
        hidden, block, injection, norm_weight, combined, normed, eps
    ) is None
    assert _op(DOWN)(normed, down_weight, down) is None
    assert _op(UP)(down[:, :RANK], up_weight, normed, mixed) is None
    return combined, mixed, down[:, RANK:RANK + 4]


def _exact(actual, expected):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def _golden(actual, expected):
    # 沿用现有 multi-M combine+norm 独立 golden 的 2e-3 绝对门槛；
    # 完整链路有 norm 与 projection 的 FP16 边界，另要求逐算子 bit-exact。
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=2.0e-3)


def _pointers(outputs):
    return tuple(t.data_ptr() for t in outputs)


def _check_outputs(outputs, rows, device):
    assert len(outputs) == 3
    for value, width in zip(outputs, (WIDTH, HIDDEN, 4)):
        assert value.shape == (rows, width)
        assert value.device == device and value.dtype == torch.float16
        assert value.data_ptr() % 4 == 0
    assert outputs[0].is_contiguous() and outputs[1].is_contiguous()
    assert outputs[2].stride() == (DOWN_WIDTH, 1)
    assert outputs[2].storage_offset() == RANK


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("scale", (0.25, 1.5))
def test_full_chain_torch_and_sequential_golden(device, weights, rows, scale):
    args = _inputs(device, weights, rows, seed=9200 + rows, scale=scale)
    snapshots = tuple(t.clone() for t in args)
    expected = _torch_reference(args)
    sequential = _sequential(args)
    workspace = _workspace()
    actual = workspace.run(*args, EPS)
    _check_outputs(actual, rows, device)
    _exact(actual, sequential)
    _golden(actual, expected)
    _exact(args, snapshots)
    repeated = workspace.run(*args, EPS)
    assert _pointers(repeated) == _pointers(actual)
    _exact(repeated, sequential)


def test_reuses_each_m_without_overwriting_other_m(device, weights):
    workspace = _workspace()
    borrowed, snapshots = {}, {}
    for iteration, rows in enumerate((*range(2, 9), *range(8, 1, -1))):
        args = _inputs(device, weights, rows, seed=9300 + iteration)
        actual = workspace.run(*args, EPS)
        if rows in borrowed:
            assert _pointers(actual) == _pointers(borrowed[rows])
        for other_rows in borrowed:
            if other_rows != rows:
                _exact(borrowed[other_rows], snapshots[other_rows])
        _exact(actual, _sequential(args))
        borrowed[rows] = actual
        snapshots[rows] = tuple(t.clone() for t in actual)
    # 所有返回张量仍存活时，不同 M 不能共享工作区 allocation。
    for index in range(3):
        assert len({values[index].data_ptr() for values in borrowed.values()}) == 7


def test_two_streams_reuse_and_cross_stream_consumer(device, weights):
    workspace = _workspace()
    current = torch.xpu.current_stream(device)
    streams = (torch.xpu.Stream(device=device), torch.xpu.Stream(device=device))
    for stream in streams:
        stream.wait_stream(current)
    records = []
    for index, stream in enumerate(streams):
        with torch.xpu.stream(stream):
            args = _inputs(device, weights, rows=5, seed=9400 + index)
            expected = _sequential(args)
            golden = _torch_reference(args)
            first = workspace.run(*args, EPS)
            repeated = workspace.run(*args, EPS)
            assert _pointers(first) == _pointers(repeated)
            consumed = repeated[1].float().mul(1.25).add(0.375)
            records.append((args, repeated, expected, golden, consumed))
    assert set(_pointers(records[0][1])).isdisjoint(_pointers(records[1][1]))

    # producer 的输出跨 stream 回馈；依赖完全由设备事件建立，无中途 host sync。
    streams[1].wait_stream(streams[0])
    with torch.xpu.stream(streams[1]):
        source = records[0][1]
        recurrent_args = (source[0], source[1], source[2], *weights)
        recurrent_expected = _sequential(recurrent_args)
        recurrent = workspace.run(*recurrent_args, EPS)
        recurrent_consumed = recurrent[1].float().square()
    for stream in streams:
        current.wait_stream(stream)
    # 至此全部提交完成；stream1 的第二次 run 已复用它自己的借出缓冲区。
    _exact(records[0][1], records[0][2])
    _golden(records[0][1], records[0][3])
    for _, _, expected, _, consumed in records:
        torch.testing.assert_close(
            consumed, expected[1].float().mul(1.25).add(0.375), rtol=0, atol=0
        )
    _exact(recurrent, recurrent_expected)
    torch.testing.assert_close(
        recurrent_consumed, recurrent_expected[1].float().square(), rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", (2, 5, 8))
def test_feeding_borrowed_outputs_back_allocates_safe_scratch(device, weights, rows):
    workspace = _workspace()
    args = _inputs(device, weights, rows, seed=9500 + rows)
    previous = workspace.run(*args, EPS)
    recurrent_args = (*previous, *weights)
    snapshots = tuple(t.clone() for t in recurrent_args)
    expected = _sequential(recurrent_args)
    golden = _torch_reference(recurrent_args)
    actual = workspace.run(*recurrent_args, EPS)
    assert set(_pointers(actual)).isdisjoint(_pointers(previous))
    _exact(actual, expected)
    _golden(actual, golden)
    _exact(recurrent_args, snapshots)


def _prime_sentinels(workspace, args):
    outputs = workspace.run(*args, EPS)
    # 防止部分提交仅重算相同数据而误过 unchanged 检查。
    for index, output in enumerate(outputs):
        output.fill_(7.0 + index)
    return outputs


def _rejected(workspace, args, borrowed, eps=EPS):
    watched = (*args, *borrowed)
    snapshots = tuple(t.clone() for t in watched)
    with pytest.raises(RuntimeError):
        workspace.run(*args, eps)
    _exact(watched, snapshots)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize(
    "violation", ("dtype", "cpu", "shape", "width", "stride", "offset")
)
def test_invalid_inputs_preserve_primed_outputs(device, weights, index, violation):
    workspace = _workspace()
    args = list(_inputs(device, weights))
    borrowed = _prime_sentinels(workspace, args)
    value = args[index]
    if violation == "dtype":
        args[index] = value.float()
    elif violation == "cpu":
        args[index] = value.cpu()
    elif violation == "shape":
        args[index] = value.reshape(-1) if value.ndim == 2 else value.unsqueeze(0)
    elif violation == "width":
        args[index] = value[..., :-2].contiguous()
    elif violation == "stride":
        backing = torch.full(
            (*value.shape[:-1], value.shape[-1] * 2), 0.25,
            device=device, dtype=torch.float16,
        )
        args[index] = backing[..., ::2]
        args[index].copy_(value)
    else:
        backing = torch.full((value.numel() + 1,), 0.25, device=device, dtype=value.dtype)
        args[index] = backing[1:].view_as(value)
        args[index].copy_(value)
    _rejected(workspace, args, borrowed)


@pytest.mark.parametrize("rows", (0, 1, 9))
def test_invalid_m_preserves_existing_cache(device, weights, rows):
    workspace = _workspace()
    borrowed = _prime_sentinels(workspace, _inputs(device, weights))
    _rejected(workspace, _inputs(device, weights, rows), borrowed)


@pytest.mark.parametrize("eps", (0.0, -1.0e-6, 1.0e-300, float("inf"), float("nan")))
def test_invalid_eps_preserves_primed_outputs(device, weights, eps):
    workspace = _workspace()
    args = _inputs(device, weights)
    _rejected(workspace, args, _prime_sentinels(workspace, args), eps)


@pytest.mark.parametrize("stride", (0, 3))
def test_invalid_injection_rows_preserve_outputs(device, weights, stride):
    workspace = _workspace()
    args = list(_inputs(device, weights))
    borrowed = _prime_sentinels(workspace, args)
    backing = torch.full((20,), 0.25, device=device, dtype=torch.float16)
    args[2] = backing.as_strided((5, 4), (stride, 1))
    _rejected(workspace, args, borrowed)


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("mutation", ("resize", "set_shape", "set_stride"))
def test_cached_output_metadata_rejected_before_first_submit(
    device, weights, index, mutation
):
    workspace = _workspace()
    args = _inputs(device, weights)
    borrowed = _prime_sentinels(workspace, args)
    target = borrowed[index]
    old_owner = target.detach()
    if mutation == "resize":
        # 元素数不变，不释放/重分配尚可能被 GPU 使用的 storage。
        target.resize_(1, target.numel())
    elif mutation == "set_shape":
        target.set_(torch.full(
            (5, target.shape[1] - 2), 11.0, device=device, dtype=torch.float16
        ))
    else:
        backing = torch.full(
            (5, target.shape[1] * 2), 11.0, device=device, dtype=target.dtype
        )
        target.set_(backing[:, ::2])
    _rejected(workspace, args, borrowed)
    assert old_owner.shape == (5, (WIDTH, HIDDEN)[index])


@pytest.mark.parametrize("mutation", ("resize", "set"))
def test_returned_injection_metadata_does_not_poison_next_narrow(device, weights, mutation):
    workspace = _workspace()
    args = _inputs(device, weights)
    borrowed = workspace.run(*args, EPS)
    pointers = _pointers(borrowed)
    if mutation == "resize":
        borrowed[2].resize_(1, 20)
    else:
        borrowed[2].set_(torch.full((5, 4), 13.0, device=device, dtype=torch.float16))
    actual = workspace.run(*args, EPS)
    assert actual[2] is not borrowed[2]
    assert _pointers(actual) == pointers
    _check_outputs(actual, 5, device)
    _exact(actual, _sequential(args))
    if mutation == "set":
        torch.testing.assert_close(
            borrowed[2], torch.full_like(borrowed[2], 13.0), rtol=0, atol=0
        )


@pytest.mark.parametrize("index", range(6))
def test_dlpack_offset_zero_but_misaligned_pointer_rejected(device, weights, index):
    workspace = _workspace()
    args = list(_inputs(device, weights))
    borrowed = _prime_sentinels(workspace, args)
    value = args[index]
    backing = torch.full((value.numel() + 1,), 0.25, device=device, dtype=value.dtype)
    odd = backing[1:].view_as(value)
    odd.copy_(value)
    args[index] = torch.from_dlpack(odd)
    assert args[index].storage_offset() == 0
    assert args[index].data_ptr() % 4 == 2
    _rejected(workspace, args, borrowed)


@pytest.mark.parametrize("index", (0, 1))
def test_cached_dlpack_misaligned_pointer_rejected(device, weights, index):
    workspace = _workspace()
    args = _inputs(device, weights)
    borrowed = _prime_sentinels(workspace, args)
    value = borrowed[index]
    old_owner = value.detach()
    backing = torch.full((value.numel() + 1,), 15.0, device=device, dtype=value.dtype)
    odd = torch.from_dlpack(backing[1:].view_as(value))
    value.set_(odd)
    assert value.storage_offset() == 0 and value.data_ptr() % 4 == 2
    _rejected(workspace, args, borrowed)
    assert old_owner.shape == (5, (WIDTH, HIDDEN)[index])


def test_cached_mixed_dlpack_alias_of_block_reallocates_without_input_mutation(
    device, weights
):
    workspace = _workspace()
    args = _inputs(device, weights)
    borrowed = workspace.run(*args, EPS)
    # 保留初次 submission 的输出 allocation，避免 set_ 提前释放它。
    old_owner = borrowed[1].detach()
    wrapped_block = torch.from_dlpack(args[1])
    assert not torch._C._is_alias_of(wrapped_block, args[1])
    borrowed[1].set_(wrapped_block)
    assert borrowed[1].data_ptr() == args[1].data_ptr()
    snapshots = tuple(t.clone() for t in args)
    expected = _sequential(args)
    actual = workspace.run(*args, EPS)
    assert actual[1].data_ptr() != args[1].data_ptr()
    _exact(args, snapshots)
    _exact(actual, expected)
    assert old_owner.shape == (5, HIDDEN)


def test_dlpack_wrapped_feedback_is_detected_by_physical_ranges(device, weights):
    workspace = _workspace()
    borrowed = workspace.run(*_inputs(device, weights), EPS)
    feedback = tuple(torch.from_dlpack(t) for t in borrowed)
    for wrapped, original in zip(feedback, borrowed):
        assert not torch._C._is_alias_of(wrapped, original)
    args = (*feedback, *weights)
    snapshots = tuple(t.clone() for t in args)
    expected = _sequential(args)
    actual = workspace.run(*args, EPS)
    assert set(_pointers(actual)).isdisjoint(_pointers(borrowed))
    _exact(actual, expected)
    _exact(args, snapshots)
