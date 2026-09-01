"""Accuracy and ABI checks for QSA indexer norm+RoPE."""

from __future__ import annotations

import importlib
import importlib.util
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
