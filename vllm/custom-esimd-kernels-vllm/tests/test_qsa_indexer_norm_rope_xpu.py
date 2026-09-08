"""Accuracy and ABI checks for QSA indexer norm+RoPE."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch


def _xpu_available() -> bool:
    try:
        return torch.xpu.is_available() and torch.xpu.device_count() > 0
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA validation requires an XPU"
)


def _load_qsa_extension():
    configured = os.environ.get("QSA_DSO")
    if configured:
        spec = importlib.util.spec_from_file_location("qsa_ops", configured)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSA_DSO: {configured}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    package_dir = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "custom_esimd_kernels_vllm"
    )
    candidates = sorted(package_dir.glob("qsa_ops*.so"))
    if len(candidates) == 1:
        spec = importlib.util.spec_from_file_location("qsa_ops", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSA extension: {candidates[0]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("custom_esimd_kernels_vllm.qsa_ops")


@pytest.fixture(scope="module")
def qsa_ops():
    return _load_qsa_extension()


def _reference(input_tensor, weight, positions, cache, mrope):
    values = input_tensor.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    values = values * (1.0 + weight.float())
    if mrope:
        pos_t, pos_h, pos_w = positions[:, 0].tolist()
    else:
        pos_t = pos_h = pos_w = int(positions[0].item())
    cos = cache[pos_t, :32].float()
    sin = cache[pos_t, 32:].float()
    if mrope:
        h_cos = cache[pos_h, :32].float()
        h_sin = cache[pos_h, 32:].float()
        w_cos = cache[pos_w, :32].float()
        w_sin = cache[pos_w, 32:].float()
        cos = cos.clone()
        sin = sin.clone()
        cos[1::3] = h_cos[1::3]
        sin[1::3] = h_sin[1::3]
        cos[2::3] = w_cos[2::3]
        sin[2::3] = w_sin[2::3]
    first = values[..., :32].clone()
    second = values[..., 32:64].clone()
    values[..., :32] = first * cos - second * sin
    values[..., 32:64] = second * cos + first * sin
    return values.to(input_tensor.dtype)


def _make_case(dtype=torch.int64, heads=4):
    input_tensor = torch.linspace(-1.0, 1.0, heads * 128, dtype=torch.float16)
    input_tensor = input_tensor.reshape(1, heads, 128).to("xpu")
    weight = torch.linspace(-0.25, 0.25, 128, dtype=torch.float16).to("xpu")
    positions_cpu = torch.tensor([[1], [2], [3]], dtype=dtype)
    positions = positions_cpu.to("xpu")
    cache = torch.zeros((8, 64), dtype=torch.float16)
    for position in range(8):
        angles = torch.linspace(0.01, 0.32, 32) * (position + 1)
        cache[position, :32] = torch.cos(angles)
        cache[position, 32:] = torch.sin(angles)
    return input_tensor, weight, positions, cache.to("xpu"), positions_cpu


def test_qsa_indexer_norm_rope_exposes_v1_capability(qsa_ops):
    assert qsa_ops.qsa_indexer_postprocess_abi_version == 1
    assert qsa_ops.qsa_indexer_head_dim == 128
    assert qsa_ops.qsa_indexer_rotary_dim == 64
    assert qsa_ops.qsa_indexer_mrope_interleaved == 1
    assert qsa_ops.qsa_indexer_gemma_weight_plus_one == 1
    assert qsa_ops.qsa_indexer_fp32_rms == 1
    assert callable(qsa_ops.qsa_indexer_norm_rope_v1)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_qsa_indexer_norm_rope_matches_mrope_reference(qsa_ops, position_dtype):
    input_tensor, weight, positions, cache, positions_cpu = _make_case(position_dtype)
    output = torch.empty_like(input_tensor)
    returned = qsa_ops.qsa_indexer_norm_rope_v1(
        input_tensor, output, weight, positions, cache, True, True
    )
    torch.xpu.synchronize()
    expected = _reference(
        input_tensor.cpu(), weight.cpu(), positions_cpu, cache.cpu(), True
    )
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_qsa_indexer_norm_rope_matches_1d_reference(qsa_ops):
    input_tensor, weight, _, cache, _ = _make_case()
    positions = torch.tensor([4], dtype=torch.int64, device="xpu")
    output = torch.empty_like(input_tensor)
    qsa_ops.qsa_indexer_norm_rope_v1(
        input_tensor, output, weight, positions, cache, False, True
    )
    torch.xpu.synchronize()
    expected = _reference(
        input_tensor.cpu(), weight.cpu(), positions.cpu(), cache.cpu(), False
    )
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_qsa_indexer_norm_rope_rejects_unproven_mrope(qsa_ops):
    input_tensor, weight, positions, cache, _ = _make_case()
    output = torch.empty_like(input_tensor)
    with pytest.raises(RuntimeError, match="position-bound proof"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, False
        )


def test_qsa_indexer_norm_rope_rejects_output_alias(qsa_ops):
    input_tensor, weight, positions, cache, _ = _make_case()
    with pytest.raises(RuntimeError, match="must not overlap"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, input_tensor, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_matches_k_geometry(qsa_ops):
    input_tensor, weight, positions, cache, positions_cpu = _make_case(heads=1)
    output = torch.empty_like(input_tensor)
    returned = qsa_ops.qsa_indexer_norm_rope_v1(
        input_tensor, output, weight, positions, cache, True, True
    )
    torch.xpu.synchronize()
    expected = _reference(
        input_tensor.cpu(), weight.cpu(), positions_cpu, cache.cpu(), True
    )
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_qsa_indexer_norm_rope_accepts_production_strided_mrope(qsa_ops):
    input_tensor, weight, _, cache, _ = _make_case()
    positions_cpu = torch.zeros((3, 4), dtype=torch.int64)
    positions_cpu[:, 0] = torch.tensor([1, 2, 3])
    positions = positions_cpu.to("xpu")[:, :1]
    assert positions.shape == (3, 1)
    assert positions.stride(1) == 1
    assert positions.stride(0) == 4
    output = torch.empty_like(input_tensor)
    qsa_ops.qsa_indexer_norm_rope_v1(
        input_tensor, output, weight, positions, cache, True, True
    )
    torch.xpu.synchronize()
    expected = _reference(
        input_tensor.cpu(), weight.cpu(), positions_cpu[:, :1], cache.cpu(), True
    )
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_qsa_indexer_norm_rope_rejects_unproven_plain_positions(qsa_ops):
    input_tensor, weight, _, cache, _ = _make_case()
    positions = torch.tensor([4], dtype=torch.int64, device="xpu")
    output = torch.empty_like(input_tensor)
    with pytest.raises(RuntimeError, match="position-bound proof"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, False, False
        )


def test_qsa_indexer_norm_rope_rejects_physical_dlpack_overlap(qsa_ops):
    from torch.utils.dlpack import from_dlpack, to_dlpack

    input_tensor, weight, positions, cache, _ = _make_case()
    aliased_output = from_dlpack(to_dlpack(input_tensor))
    assert aliased_output.data_ptr() == input_tensor.data_ptr()
    output = aliased_output
    with pytest.raises(RuntimeError, match="must not overlap"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_wrong_dtype(qsa_ops):
    input_tensor, weight, positions, cache, _ = _make_case()
    bad_input = input_tensor.float()
    output = torch.empty_like(bad_input)
    with pytest.raises(RuntimeError, match="must be float16"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            bad_input, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_wrong_input_shape(qsa_ops):
    _, weight, positions, cache, _ = _make_case()
    input_tensor = torch.empty((1, 2, 128), dtype=torch.float16, device="xpu")
    output = torch.empty_like(input_tensor)
    with pytest.raises(RuntimeError, match=r"input must have shape"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_wrong_device(qsa_ops):
    input_tensor, weight, positions, cache, _ = _make_case()
    input_cpu = input_tensor.cpu()
    output = torch.empty_like(input_cpu)
    with pytest.raises(RuntimeError, match=r"must share one XPU device"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_cpu, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_unaligned_activation(qsa_ops):
    _, weight, positions, cache, _ = _make_case()
    backing = torch.empty(4 * 128 + 1, dtype=torch.float16, device="xpu")
    input_tensor = backing[1:].view(1, 4, 128)
    if input_tensor.data_ptr() % 16 == 0:
        pytest.skip("allocator did not produce an unaligned view")
    output = torch.empty_like(input_tensor)
    with pytest.raises(RuntimeError, match=r"16-byte aligned"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_malformed_cache(qsa_ops):
    input_tensor, weight, positions, cache, _ = _make_case()
    malformed_cache = cache[:, :32]
    output = torch.empty_like(input_tensor)
    with pytest.raises(RuntimeError, match=r"shape \[max_position,64\]"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, malformed_cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_rows_above_capacity(qsa_ops):
    input_tensor = torch.empty((65, 4, 128), dtype=torch.float16, device="xpu")
    output = torch.empty_like(input_tensor)
    weight = torch.ones(128, dtype=torch.float16, device="xpu")
    positions = torch.ones((3, 65), dtype=torch.int64, device="xpu")
    cache = torch.zeros((8, 64), dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match=r"input must have shape"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_output_cache_overlap(qsa_ops):
    input_tensor, weight, positions, _, _ = _make_case()
    backing = torch.empty(512, dtype=torch.float16, device="xpu")
    output = backing.view(1, 4, 128)
    cache = backing.view(8, 64)
    with pytest.raises(RuntimeError, match=r"must not overlap"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, True
        )


def test_qsa_indexer_norm_rope_rejects_output_positions_overlap(qsa_ops):
    input_tensor, weight, _, cache, _ = _make_case()
    backing = torch.empty(512, dtype=torch.float16, device="xpu")
    output = backing.view(1, 4, 128)
    positions = backing[:12].view(torch.int64).view(3, 1)
    with pytest.raises(RuntimeError, match=r"must not overlap"):
        qsa_ops.qsa_indexer_norm_rope_v1(
            input_tensor, output, weight, positions, cache, True, True
        )


def _v2_reference(input_tensor, weight, positions, cache, mrope):
    """独立 eager Torch golden，显式保留 norm 和每个乘积的 FP16 舍入。"""
    values = input_tensor.float()
    normed = (
        values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        * (1.0 + weight.float())
    ).half()
    cos, sin = cache[positions.long()].chunk(2, dim=-1)
    if mrope:
        cos, sin = cos[0].clone(), sin[0].clone()
        selected = cache[positions.long()]
        cos[:, 1::3] = selected[1, :, :32][:, 1::3]
        sin[:, 1::3] = selected[1, :, 32:][:, 1::3]
        cos[:, 2::3] = selected[2, :, :32][:, 2::3]
        sin[:, 2::3] = selected[2, :, 32:][:, 2::3]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    first, second = normed[..., :32], normed[..., 32:64]
    first_cos = (first.float() * cos.float()).half()
    second_sin = (second.float() * sin.float()).half()
    second_cos = (second.float() * cos.float()).half()
    first_sin = (first.float() * sin.float()).half()
    return torch.cat((
        (first_cos.float() - second_sin.float()).half(),
        (second_cos.float() + first_sin.float()).half(),
        normed[..., 64:],
    ), dim=-1)


def _make_v2_case(rows, heads=4, layout="axis_padded", dtype=torch.int64, scale=0.3):
    generator = torch.Generator().manual_seed(9041 + rows + heads)
    # Q 的真实输入来自 [M,640].split((512,128))，不拷成 contiguous。
    if heads == 4:
        backing = torch.randn(rows, 640, generator=generator).half().to("xpu")
        backing.mul_(scale)
        input_tensor = backing.split((512, 128), dim=-1)[0].view(rows, heads, 128)
        assert input_tensor.stride() == (640, 128, 1)
    else:
        backing = (torch.randn(rows, 1, 128, generator=generator) * scale).half().to("xpu")
        input_tensor = backing
    weight = (torch.randn(128, generator=generator) * 0.2).half().to("xpu")
    angles = torch.arange(64).float()[:, None] * torch.linspace(0.013, 0.49, 32)
    cache = torch.cat((angles.cos(), angles.sin()), dim=1).half().to("xpu")
    values = (torch.arange(3 * rows).view(3, rows) * 7 + 1) % 64
    # cache[positions] 支持合法负索引；也覆盖首、末 cache row。
    values[:, 0] = torch.tensor([0, 63, -64])
    values[:, -1] = torch.tensor([-1, 0, 0])
    values = values.to(dtype)
    if layout == "plain":
        positions = values[0].to("xpu")
    elif layout == "axis_padded":
        position_backing = torch.zeros(3, 4096, dtype=dtype, device="xpu")
        positions = position_backing[:, 3:3 + rows]
        positions.copy_(values)
        assert positions.stride() == (4096, 1)
    else:
        positions = values.t().contiguous().to("xpu").t()
        assert positions.stride() == (1, 3)
    return input_tensor, weight, positions, cache, backing


def _call_v2(module, case, output=None, *, neox=True, fp32=False):
    x, weight, positions, cache, _ = case
    if output is None:
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    returned = module.qsa_indexer_norm_rope_v2(
        x, output, weight, positions, cache, positions.ndim == 2, neox, fp32
    )
    assert returned.data_ptr() == output.data_ptr()
    assert returned.is_contiguous()
    return output


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("layout", ["plain", "axis_padded", "column_strided"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("scale", [0.1, 1.5])
def test_qsa_v2_real_layouts_and_fp16_reference(qsa_ops, rows, heads, layout, dtype, scale):
    case = _make_v2_case(rows, heads, layout, dtype, scale)
    x, weight, positions, cache, backing = case
    saved = [t.clone() for t in (backing, weight, positions, cache)]
    expected = _v2_reference(x, weight, positions, cache, positions.ndim == 2)
    actual = _call_v2(qsa_ops, case)
    torch.xpu.synchronize()
    # 不沿用 v1 的 2e-2 门槛；随机 RMS 归约允许少量 FP32 舍入差异。
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    for tensor, snapshot in zip((backing, weight, positions, cache), saved):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("rows", [2, 5, 8])
@pytest.mark.parametrize("layout", ["plain", "axis_padded", "column_strided"])
def test_qsa_v2_preserves_each_half_product_bitexact(qsa_ops, rows, layout):
    case = _make_v2_case(rows, 4, layout)
    x, weight, positions, cache, _ = case
    x.fill_(1)
    x[..., 1::2] = -1
    weight.copy_((torch.arange(128, device="xpu").float() / 256).half())
    # variance=1，norm FP16 可精确确定；这样 bitexact 检查隔离 RoPE 舍入。
    expected = _v2_reference(x, weight, positions, cache, positions.ndim == 2)
    actual = _call_v2(qsa_ops, case)
    torch.xpu.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rows", [2, 5, 8])
def test_qsa_v2_half_product_subnormal_and_even_ties(qsa_ops, rows):
    case = _make_v2_case(rows, 4, "plain")
    x, weight, positions, cache, _ = case
    x.fill_(1)
    weight.fill_(-1)
    # variance=1；这些 norm 值都应在 FP16 边界舍入回精确 dyadic 值。
    norm_values = torch.tensor(
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 1023.5, 1.5, 1.5],
        dtype=torch.float32, device="xpu",
    )
    weight[:norm_values.numel()].copy_((norm_values - 1).half())
    positions.zero_()
    cache.zero_()
    smallest_half = 2.0 ** -24
    cache[0, :9] = smallest_half
    # 前九列覆盖下溢、ties-to-even、带符号结果与 subnormal/normal 边界。
    # 最后两列覆盖普通数尾数 tie 的奇/偶两种方向。
    cache[0, 9] = 1 + 2.0 ** -10
    cache[0, 10] = 1 + 3 * 2.0 ** -10
    expected = _v2_reference(x, weight, positions, cache, False)
    actual = _call_v2(qsa_ops, case)
    torch.xpu.synchronize()
    expected_units = torch.tensor(
        [0, 2, 2, 4, 0, -2, -2, -4, 1024], dtype=torch.float32, device="xpu"
    )
    assert torch.equal(expected[0, 0, :9].float() / smallest_half, expected_units)
    assert torch.equal(actual, expected)


def test_qsa_v2_half_product_overflow_boundary(qsa_ops):
    case = _make_v2_case(5, 1, "column_strided")
    x, weight, positions, cache, _ = case
    x.fill_(1)
    weight.fill_(-1)
    weight[:3] = 0.5  # norm FP16 = 1.5
    positions.zero_()
    cache.zero_()
    cache[0, :3] = torch.tensor(
        [43648, 43680, -43680], dtype=torch.float16, device="xpu"
    )
    # 43648*1.5=65472（有限）；43680*1.5=65520（FP16 溢出 tie）。
    expected = _v2_reference(x, weight, positions, cache, True)
    actual = _call_v2(qsa_ops, case)
    torch.xpu.synchronize()
    assert torch.isfinite(expected[..., 0]).all()
    assert torch.isposinf(expected[..., 1]).all()
    assert torch.isneginf(expected[..., 2]).all()
    assert torch.equal(actual, expected)


def test_qsa_v2_current_stream_row_indices_and_output_reuse(qsa_ops):
    parent = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    records, keepalive = [], []
    for stream in streams:
        stream.wait_stream(parent)
    for index, stream in enumerate(streams):
        with torch.xpu.stream(stream):
            for rows in ([5, 2, 5, 8] if index == 0 else [8, 3, 8, 6]):
                case = _make_v2_case(rows, 4 if index == 0 else 1, "column_strided")
                x, weight, positions, cache, _ = case
                # positions 和 activations 的 producer 均位于当前非默认流。
                x.mul_(0.5)
                positions.copy_((positions + 3).remainder(cache.shape[0]))
                expected = _v2_reference(x, weight, positions, cache, True)
                out = _call_v2(qsa_ops, case)
                records.append((out.clone(), expected))
                feedback = (out, weight, positions, cache, out)
                second = _call_v2(qsa_ops, feedback)
                records.append((second.clone(), _v2_reference(out, weight, positions, cache, True)))
                keepalive.extend((case, out, second))
    for stream in streams:
        parent.wait_stream(stream)
    torch.xpu.synchronize()
    for actual, expected in records:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bad", [
    "dtype", "weight_shape", "inner_stride", "input_offset", "position_stride",
    "output_stride", "output_offset", "output_alias", "dlpack_alias",
    "lazy_input", "lazy_output", "neox", "fp32", "device",
])
def test_qsa_v2_metadata_rejected_before_output_write(qsa_ops, bad):
    case = list(_make_v2_case(5))
    x, weight, positions, cache, backing = case
    output = torch.full(x.shape, 19.0, dtype=torch.float16, device="xpu")
    neox, fp32 = True, False
    if bad == "dtype":
        case[0] = x.float()
    elif bad == "weight_shape":
        case[1] = weight[:-1]
    elif bad == "inner_stride":
        case[0] = x.transpose(1, 2).contiguous().transpose(1, 2)
    elif bad == "input_offset":
        base = torch.full((5 * 512 + 1,), 0.2, dtype=torch.float16, device="xpu")
        case[0] = base[1:].view(5, 4, 128)
    elif bad == "position_stride":
        case[2] = positions[:1].expand(3, 5)
    elif bad == "output_stride":
        output = torch.full_like(backing, 19.0)[:, :512].view(5, 4, 128)
    elif bad == "output_offset":
        output = torch.full((5 * 512 + 1,), 19.0, dtype=x.dtype, device="xpu")[1:].view(5, 4, 128)
    elif bad in ("output_alias", "dlpack_alias"):
        output = backing.view(-1)[:5 * 512].view(5, 4, 128)
        if bad == "dlpack_alias":
            output = torch.utils.dlpack.from_dlpack(output)
    elif bad == "lazy_input":
        case[0] = torch._neg_view(x)
    elif bad == "lazy_output":
        output = torch._neg_view(output)
    elif bad == "neox":
        neox = False
    elif bad == "fp32":
        fp32 = True
    elif bad == "device":
        case[1] = weight.cpu()
    tracked = [*case, output]
    saved = [t.clone() for t in tracked]
    torch.xpu.synchronize()
    with pytest.raises(RuntimeError):
        _call_v2(qsa_ops, case, output, neox=neox, fp32=fp32)
    torch.xpu.synchronize()
    for tensor, snapshot in zip(tracked, saved):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("rows", [0, 1, 9])
def test_qsa_v2_rejects_rows_outside_multi_contract(qsa_ops, rows):
    x = torch.zeros((rows, 4, 128), dtype=torch.float16, device="xpu")
    weight = torch.zeros(128, dtype=torch.float16, device="xpu")
    positions = torch.zeros(rows, dtype=torch.int64, device="xpu")
    cache = torch.zeros(4, 64, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="2..8 rows"):
        _call_v2(qsa_ops, (x, weight, positions, cache, x))
