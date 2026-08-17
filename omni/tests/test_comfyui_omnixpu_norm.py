"""Portable control-flow tests for native RMSNorm eligibility."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"
_ADAPTERS = _PLUGIN / "adapters"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_patch(monkeypatch):
    package_name = "omnixpu_norm_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    adapters = types.ModuleType(f"{package_name}.adapters")
    adapters.__path__ = [str(_ADAPTERS)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(
        sys.modules, "comfy.model_management", model_management
    )
    return _load_module(
        f"{adapters.__name__}.norm", _ADAPTERS / "norm.py"
    )


class _FakeTensor:
    def __init__(self, hidden_size: int):
        self.shape = (1, 4205, 28, hidden_size)
        self.ndim = 4
        self.is_xpu = True
        self.reshaped_to = None

    def is_contiguous(self):
        return True

    def reshape(self, *shape):
        self.reshaped_to = shape
        return self


class _FakeSplitQKVTensor:
    def __init__(self, length: int = 4096):
        self.shape = (1, length, 30, 128)
        self.ndim = 4
        self.is_xpu = True
        self.reshaped_to = None
        self.materialized = False
        self._stride = (length * 11520, 11520, 128, 1)

    def is_contiguous(self):
        return self.materialized

    def stride(self, dimension):
        return self._stride[dimension]

    def contiguous(self):
        self.materialized = True
        return self

    def reshape(self, *shape):
        self.reshaped_to = shape
        return self


class _FakeSeedVRTensor:
    def __init__(self, shape=(4, 128, 512, 512), *, interleaved=True):
        self.shape = shape
        self.ndim = 4
        self.is_xpu = True
        self.dtype = torch.float16
        self.device = torch.device("xpu:0")
        batch, channels, height, width = shape
        spatial = height * width
        self._stride = (
            (spatial, batch * spatial, width, 1)
            if interleaved
            else (channels * spatial, spatial, width, 1)
        )

    def stride(self):
        return self._stride


class _FakeSeedVRParameter:
    def __init__(self, channels=128, *, dtype=torch.float16):
        self.device = torch.device("xpu:0")
        self.dtype = dtype
        self.ndim = 1
        self._channels = channels

    def is_contiguous(self):
        return True

    def numel(self):
        return self._channels


def test_h120_requires_native_capability(monkeypatch):
    patch = _load_patch(monkeypatch)
    patch._omni_norm = object()
    value = _FakeTensor(120)

    patch._allow_h120_rms = False
    assert patch._rms_input_2d(value) is None

    patch._allow_h120_rms = True
    assert patch._rms_input_2d(value) is value
    assert value.reshaped_to == (-1, 120)


def test_h120_targets_are_explicit(monkeypatch):
    patch = _load_patch(monkeypatch)

    assert patch._target_supports_h120("ptl-h")
    assert patch._target_supports_h120("bmg")
    assert not patch._target_supports_h120("unknown")


def test_noncontiguous_rms_targets_are_explicit(monkeypatch):
    patch = _load_patch(monkeypatch)

    assert patch._target_supports_noncontiguous_rms("ptl-h")
    assert patch._target_supports_noncontiguous_rms("bmg")
    assert not patch._target_supports_noncontiguous_rms("unknown")


def test_group_norm_target_is_bmg_only(monkeypatch):
    patch = _load_patch(monkeypatch)

    assert patch._target_supports_group_norm("bmg")
    assert not patch._target_supports_group_norm("ptl-h")
    assert not patch._target_supports_group_norm("unknown")


def test_group_norm_shapes_are_exact(monkeypatch):
    patch = _load_patch(monkeypatch)

    assert (1, 512, 128, 128) in patch._bmg_group_norm_shapes
    assert (1, 128, 1024, 1024) in patch._bmg_group_norm_shapes
    assert (1, 512, 64, 64) not in patch._bmg_group_norm_shapes
    assert len(patch._bmg_group_norm_shapes) == 6


def test_seedvr_group_norm_shapes_and_layout_are_exact(monkeypatch):
    patch = _load_patch(monkeypatch)

    assert patch._seedvr_group_norm_shapes == {
        (4, 128, 512, 512),
        (4, 256, 256, 256),
        (4, 256, 512, 512),
        (4, 512, 256, 256),
        (2, 512, 128, 128),
    }
    assert patch._is_seedvr_group_norm_layout(_FakeSeedVRTensor())
    assert not patch._is_seedvr_group_norm_layout(
        _FakeSeedVRTensor(interleaved=False)
    )
    assert not patch._is_seedvr_group_norm_layout(
        _FakeSeedVRTensor(shape=(1, 128, 512, 512))
    )


def test_seedvr_group_norm_route_requires_full_contract(monkeypatch):
    patch = _load_patch(monkeypatch)
    value = _FakeSeedVRTensor()
    weight = _FakeSeedVRParameter()
    bias = _FakeSeedVRParameter()

    patch._allow_seedvr_group_norm = True
    assert patch._can_use_seedvr_group_norm(
        value, 32, weight, bias, 1e-6
    )
    assert not patch._can_use_seedvr_group_norm(
        value, 16, weight, bias, 1e-6
    )
    assert not patch._can_use_seedvr_group_norm(
        value,
        32,
        _FakeSeedVRParameter(dtype=torch.bfloat16),
        bias,
        1e-6,
    )
    assert not patch._can_use_seedvr_group_norm(
        _FakeSeedVRTensor(interleaved=False), 32, weight, bias, 1e-6
    )


def test_split_qkv_rms_materializes_only_when_enabled(monkeypatch):
    patch = _load_patch(monkeypatch)
    patch._omni_norm = object()
    value = _FakeSplitQKVTensor()

    patch._allow_noncontiguous_rms = False
    assert patch._rms_input_2d(value) is None
    assert not value.materialized

    patch._allow_noncontiguous_rms = True
    assert patch._rms_input_2d(value) is value
    assert value.materialized
    assert value.reshaped_to == (-1, 128)


def test_existing_multiple_of_32_route_is_unchanged(monkeypatch):
    patch = _load_patch(monkeypatch)
    patch._omni_norm = object()
    patch._allow_h120_rms = False
    value = _FakeTensor(3360)

    assert patch._rms_input_2d(value) is value
    assert value.reshaped_to == (-1, 3360)


def test_other_non_multiple_of_32_hidden_size_stays_on_fallback(monkeypatch):
    patch = _load_patch(monkeypatch)
    patch._omni_norm = object()
    patch._allow_h120_rms = True

    assert patch._rms_input_2d(_FakeTensor(121)) is None


def test_weightless_rms_chunking_preserves_torch_math(monkeypatch):
    patch = _load_patch(monkeypatch)
    value = torch.randn(7, 4, generator=torch.Generator().manual_seed(43))
    expected = torch.nn.functional.rms_norm(value, (4,), None, 1e-6)
    original = torch.nn.functional.rms_norm
    calls = []

    def record(input, normalized_shape, weight=None, eps=None):
        calls.append(input.shape[0])
        return original(input, normalized_shape, weight, eps)

    monkeypatch.setattr(torch.nn.functional, "rms_norm", record)
    monkeypatch.setattr(patch, "_RMS_CHUNK_INPUT_BYTES", 2 * 4 * 4)

    actual = patch._chunked_weightless_rms_norm(value, (4,), 1e-6)

    assert calls == [2, 2, 2, 1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
