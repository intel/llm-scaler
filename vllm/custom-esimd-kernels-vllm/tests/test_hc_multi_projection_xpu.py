"""HC M2..8 独立 projection ABI 测试；编写阶段不运行。

由 main 统一构建后运行。可用 HC_MULTI_PROJECTION_DSO（或 HC_CHAIN_DSO）
指定 canonical main DSO；不导入 vLLM，不依赖待新增的 Python wrapper。
Golden 独立使用 Torch F.linear，显式保留各 FP16 舍入边界。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


DOWN = "esimd_hc_down_fp16_multi_m_out_v1"
UP = "esimd_hc_up_gate_mix_multi_m_v1"
OLD_DOWN = "esimd_hc_down_fp16_out"
OLD_UP = "esimd_hc_up_gate_mix_m1_v1"
WIDTH, RANK, HIDDEN = 10240, 320, 2560
# 预先固定，不根据运行结果放宽；另有逐行 M1 的 bit-exact 检查。
RTOL, ATOL = 2.0e-3, 2.0e-4


@pytest.fixture(scope="module")
def device():
    if not torch.xpu.is_available():
        pytest.skip("XPU 不可用")
    configured = os.environ.get("HC_MULTI_PROJECTION_DSO") or os.environ.get(
        "HC_CHAIN_DSO"
    )
    if configured:
        dso = Path(configured)
        if not dso.is_file():
            pytest.fail(f"找不到指定 DSO: {dso}")
        torch.ops.load_library(str(dso))
    else:
        import custom_esimd_kernels_vllm  # noqa: F401

    for name in (DOWN, UP, OLD_DOWN, OLD_UP):
        schema = f"custom_esimd_kernels_vllm::{name}"
        if not torch._C._jit_get_schemas_for_operator(schema):
            pytest.fail(f"canonical DSO 未注册 {schema}，请由 main 统一构建")
    return torch.device("xpu:0")


def _op(name):
    return getattr(torch.ops.custom_esimd_kernels_vllm, name)


def _rand(shape, device, generator, scale):
    return (
        torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
        * scale
    ).half()


@pytest.fixture(scope="module")
def weights(device):
    generator = torch.Generator(device=device).manual_seed(38081)
    down = _rand((336, WIDTH), device, generator, 1.0 / math.sqrt(WIDTH))
    up = _rand((WIDTH, RANK), device, generator, 1.0 / math.sqrt(RANK))
    return down, up


def _offset_copy(value, offset):
    storage = torch.full(
        (value.numel() + offset + 8,), -7.0, dtype=value.dtype, device=value.device
    )
    view = storage[offset : offset + value.numel()].view_as(value)
    view.copy_(value)
    return storage, view


def _down_reference(x, weight):
    linear = F.linear(x, weight).to(torch.float16)
    scaled = (linear[:, :RANK] / 4.0).to(torch.float16)
    result = linear.clone()
    result[:, :RANK] = F.silu(scaled.float()).to(torch.float16)
    return result


def _up_reference(x, weight, normed):
    # 绝不能用 FP32 gate 直接 sigmoid，也不能先 FP16 sigmoid/multiply。
    gate = F.linear(x[:, :RANK], weight).to(torch.float16)
    branches = torch.sigmoid(gate.float()) * normed.float()
    return branches.view(x.shape[0], 4, HIDDEN).mean(dim=1).half()


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def _unchanged(values, snapshots):
    for value, snapshot in zip(values, snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("columns", (320, 336))
@pytest.mark.parametrize("scale", (0.25, 1.5))
def test_down_torch_golden_and_caller_storage(device, weights, rows, columns, scale):
    generator = torch.Generator(device=device).manual_seed(8100 + rows + columns)
    # 所有参数均有非零、仅 4-byte aligned 的合法 storage offset。
    xb, x = _offset_copy(_rand((rows, WIDTH), device, generator, scale), 2)
    wb, weight = _offset_copy(weights[0][:columns], 2)
    ob, output = _offset_copy(
        torch.full((rows, columns), -5.0, device=device, dtype=torch.float16), 2
    )
    before = (xb.clone(), wb.clone(), ob.clone())
    pointer, identity = output.data_ptr(), id(output)
    expected = _down_reference(x, weight)
    assert _op(DOWN)(x, weight, output) is None
    assert output.data_ptr() == pointer and id(output) == identity
    _close(output[:, :RANK], expected[:, :RANK])
    if columns == 336:
        # 后 16 列必须保持 linear，不能被 SiLU 覆盖。
        _close(output[:, RANK:], expected[:, RANK:])
    _unchanged((xb, wb), before[:2])
    _unchanged((ob[:2], ob[-8:]), (before[2][:2], before[2][-8:]))


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("scale", (0.25, 1.5))
@pytest.mark.parametrize(
    "columns,stride,offset",
    ((320, 320, 0), (336, 336, 0), (320, 336, 0),
     (320, 336, 2), (336, 352, 16), (320, 322, 2)),
)
def test_up_torch_golden_strided_input(
    device, weights, rows, scale, columns, stride, offset
):
    generator = torch.Generator(device=device).manual_seed(8200 + rows + columns)
    backing = torch.full(
        (offset + rows * stride + 8,), 19.0, dtype=torch.float16, device=device
    )
    x = backing.as_strided((rows, columns), (stride, 1), offset)
    raw = _rand((rows, RANK), device, generator, 4.0 * scale)
    x[:, :RANK].copy_(F.silu((raw / 4.0).half().float()).half())
    # N336 的尾列/padding 是不同的非零值，up 必须完全忽略它们。
    nb, normed = _offset_copy(_rand((rows, WIDTH), device, generator, 1.0), 2)
    wb, weight = _offset_copy(weights[1], 2)
    ob, output = _offset_copy(
        torch.full((rows, HIDDEN), -5.0, dtype=torch.float16, device=device), 2
    )
    snapshots = tuple(t.clone() for t in (backing, nb, wb, ob))
    pointer, identity = output.data_ptr(), id(output)
    expected = _up_reference(x, weight, normed)
    assert _op(UP)(x, weight, normed, output) is None
    assert x.stride() == (stride, 1) and x.storage_offset() == offset
    assert output.data_ptr() == pointer and id(output) == identity
    _close(output, expected)
    _unchanged((backing, nb, wb), snapshots[:3])
    _unchanged((ob[:2], ob[-8:]), (snapshots[3][:2], snapshots[3][-8:]))


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("columns", (320, 336))
def test_multi_matches_unchanged_m1_rowwise(device, weights, rows, columns):
    generator = torch.Generator(device=device).manual_seed(8300 + rows)
    normed = _rand((rows, WIDTH), device, generator, 1.0)
    down_weight, up_weight = weights[0][:columns], weights[1]
    down = torch.empty((rows, columns), device=device, dtype=torch.float16)
    old_down = torch.empty_like(down)
    output = torch.empty((rows, HIDDEN), device=device, dtype=torch.float16)
    old_output = torch.empty_like(output)
    _op(DOWN)(normed, down_weight, down)
    _op(UP)(down, up_weight, normed, output)
    for row in range(rows):
        s = slice(row, row + 1)
        assert _op(OLD_DOWN)(normed[s], down_weight, old_down[s]) is None
        # 单行 [:320] contiguous，保持旧 M1 对 [1,320] 的严格 ABI。
        assert _op(OLD_UP)(old_down[s, :RANK], up_weight, normed[s], old_output[s]) is None
    _unchanged((down, output), (old_down, old_output))
    # M1 不作为唯一 golden；独立 Torch reference 必须同时成立。
    golden_down = _down_reference(normed, down_weight)
    _close(down, golden_down)
    _close(output, _up_reference(golden_down, up_weight, normed))


@pytest.mark.parametrize("columns", (320, 336))
def test_two_streams_and_producer_consumer_without_mid_host_sync(device, weights, columns):
    current = torch.xpu.current_stream(device)
    streams = (torch.xpu.Stream(device=device), torch.xpu.Stream(device=device))
    for stream in streams:
        stream.wait_stream(current)  # 只依赖共享 weights 的初始化。
    records = []
    for index, stream in enumerate(streams):
        with torch.xpu.stream(stream):
            rows = (5, 8)[index]
            generator = torch.Generator(device=device).manual_seed(8400 + index)
            source = _rand((rows, WIDTH), device, generator, 1.0)
            normed = source.mul(0.75).add(0.125)  # GPU producer。
            down = torch.empty((rows, columns), device=device, dtype=torch.float16)
            output = torch.empty((rows, HIDDEN), device=device, dtype=torch.float16)
            pointer = (down.data_ptr(), output.data_ptr())
            expected_down = _down_reference(normed, weights[0][:columns])
            expected = _up_reference(expected_down, weights[1], normed)
            for _ in range(4):
                assert _op(DOWN)(normed, weights[0][:columns], down) is None
                # 生产 [M,336][:,:320] view，不能在传入 op 前 contiguous。
                assert _op(UP)(down[:, :RANK], weights[1], normed, output) is None
            consumed = output.float().mul(1.25).add(0.375)
            records.append((source, normed, down, output, consumed, expected, pointer))

    # 显式跨 stream producer -> consumer，仅设备依赖，不做中途 host sync。
    streams[1].wait_stream(streams[0])
    with torch.xpu.stream(streams[1]):
        first = records[0]
        cross_output = torch.empty_like(first[3])
        _op(UP)(first[2], weights[1], first[1], cross_output)
        cross_consumed = cross_output.float().mul(1.25).add(0.375)
    for stream in streams:
        current.wait_stream(stream)
    # 全部提交完成后才进入有 host 同步的 assertion。
    for _, _, down, output, consumed, expected, pointer in records:
        assert (down.data_ptr(), output.data_ptr()) == pointer
        _close(output, expected)
        _close(consumed, expected.float().mul(1.25).add(0.375))
    _close(cross_consumed, records[0][5].float().mul(1.25).add(0.375))


def _contract_args(kind, device, weights, rows=5):
    generator = torch.Generator(device=device).manual_seed(8500 + rows)
    if kind == DOWN:
        return [
            _rand((rows, WIDTH), device, generator, 1.0), weights[0],
            torch.full((rows, 336), -5.0, dtype=torch.float16, device=device),
        ]
    return [
        _rand((rows, 336), device, generator, 0.25), weights[1],
        _rand((rows, WIDTH), device, generator, 1.0),
        torch.full((rows, HIDDEN), -5.0, dtype=torch.float16, device=device),
    ]


def _rejected(kind, args):
    snapshots = tuple(t.clone() for t in args)
    with pytest.raises((RuntimeError, NotImplementedError)):
        _op(kind)(*args)
    # 提交前拒绝不应留下任何写入；assert 等待当前 stream 后检查所有参数。
    _unchanged(args, snapshots)


@pytest.mark.parametrize("kind", (DOWN, UP))
@pytest.mark.parametrize("rows", (0, 1, 9))
def test_rejects_unsupported_m(device, weights, kind, rows):
    _rejected(kind, _contract_args(kind, device, weights, rows))


@pytest.mark.parametrize(
    "kind,index", [(kind, i) for kind, count in ((DOWN, 3), (UP, 4)) for i in range(count)]
)
@pytest.mark.parametrize("violation", ("dtype", "cpu", "shape", "inner_stride", "offset"))
def test_rejects_each_tensor_bad_contract(device, weights, kind, index, violation):
    args = _contract_args(kind, device, weights)
    value = args[index]
    if violation == "dtype":
        args[index] = value.float()
    elif violation == "cpu":
        args[index] = value.cpu()
    elif violation == "shape":
        args[index] = value.reshape(-1)
    elif violation == "inner_stride":
        backing = torch.empty(
            (value.shape[0], value.shape[1] * 2), device=device, dtype=value.dtype
        )
        args[index] = backing[:, ::2]
        args[index].copy_(value)
    else:
        _, args[index] = _offset_copy(value, 1)
    _rejected(kind, args)


@pytest.mark.parametrize(
    "kind,index", [(kind, i) for kind, count in ((DOWN, 3), (UP, 4)) for i in range(count)]
)
def test_rejects_wrong_dimensions(device, weights, kind, index):
    args = _contract_args(kind, device, weights)
    # 保持 2D / contiguous，防止仅测试到 rank/stride 拒绝。
    args[index] = args[index][:, :-2].contiguous()
    _rejected(kind, args)


@pytest.mark.parametrize(
    "kind,index", ((DOWN, 0), (DOWN, 1), (DOWN, 2), (UP, 1), (UP, 2), (UP, 3))
)
def test_rejects_padded_contiguous_required_tensor(device, weights, kind, index):
    args = _contract_args(kind, device, weights)
    value = args[index]
    backing = torch.full(
        (value.shape[0], value.shape[1] + 2), 0.5, device=device, dtype=value.dtype
    )
    args[index] = backing[:, :value.shape[1]]
    args[index].copy_(value)
    _rejected(kind, args)


@pytest.mark.parametrize("stride", (0, 318, 335, 337))
def test_rejects_up_overlapping_or_unaligned_rows(device, weights, stride):
    args = _contract_args(UP, device, weights)
    backing = torch.full((5 * 352,), 0.25, device=device, dtype=torch.float16)
    args[0] = backing.as_strided((5, 336), (stride, 1))
    _rejected(UP, args)


@pytest.mark.parametrize(
    "kind,index", [(kind, i) for kind, count in ((DOWN, 2), (UP, 3)) for i in range(count)]
)
@pytest.mark.parametrize("separate_storage", (False, True))
@pytest.mark.parametrize("shift", (0, 2))
def test_rejects_output_alias_of_every_input(
    device, weights, kind, index, separate_storage, shift
):
    args = _contract_args(kind, device, weights)
    value, output = args[index], args[-1]
    backing = torch.full(
        (max(value.numel(), output.numel() + shift) + 8,), 0.25,
        device=device, dtype=torch.float16,
    )
    args[index] = backing[:value.numel()].view_as(value)
    args[index].copy_(value)
    args[-1] = backing[shift : shift + output.numel()].view_as(output)
    if separate_storage:
        # 独立 StorageImpl 包装相同物理地址，不能只依赖 is_alias_of。
        args[-1] = torch.from_dlpack(args[-1])
    _rejected(kind, args)


def test_rejects_physical_overlap_with_later_strided_input_row(device, weights):
    args = _contract_args(UP, device, weights)
    backing = torch.full((5 * HIDDEN + 336,), 0.25, device=device, dtype=torch.float16)
    args[0] = backing.as_strided((5, RANK), (336, 1))
    # output 起点落在第二行；独立 storage，覆盖真实生产 view 的物理区间。
    args[-1] = torch.from_dlpack(backing[336:].view(5, HIDDEN))
    _rejected(UP, args)


@pytest.mark.parametrize("kind", (DOWN, UP))
def test_rejects_all_cpu_and_other_xpu(device, weights, kind):
    args = _contract_args(kind, device, weights)
    _rejected(kind, [t.cpu() for t in args])
    if torch.xpu.device_count() < 2:
        pytest.skip("跨 XPU 合约测试需要至少两张可见卡")
    for index in range(len(args)):
        mixed = list(args)
        mixed[index] = mixed[index].to("xpu:1")
        _rejected(kind, mixed)


@pytest.mark.parametrize("kind,old", ((DOWN, OLD_DOWN), (UP, OLD_UP)))
def test_old_m1_entry_still_rejects_m2(device, weights, kind, old):
    args = _contract_args(kind, device, weights, rows=2)
    if kind == UP:
        args[0] = args[0][:, :RANK].contiguous()
    _rejected(old, args)


def test_old_up_still_rejects_336_input(device, weights):
    _rejected(OLD_UP, _contract_args(UP, device, weights, rows=1))
