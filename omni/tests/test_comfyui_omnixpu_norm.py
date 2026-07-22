"""Portable control-flow tests for native RMSNorm eligibility."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"


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
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
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
        f"{patches.__name__}.patch_norm", _PATCHES / "patch_norm.py"
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


def test_h120_requires_ptl_capability(monkeypatch):
    patch = _load_patch(monkeypatch)
    patch._omni_norm = object()
    value = _FakeTensor(120)

    patch._allow_h120_rms = False
    assert patch._rms_input_2d(value) is None

    patch._allow_h120_rms = True
    assert patch._rms_input_2d(value) is value
    assert value.reshaped_to == (-1, 120)


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
