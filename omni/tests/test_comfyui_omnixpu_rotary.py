"""Portable routing tests for the model-specific LTX RoPE adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
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


class _Tensor:
    def __init__(self, shape, stride, dtype="torch.bfloat16"):
        self.shape = shape
        self._stride = stride
        self.dtype = dtype
        self.device = "xpu:0"

    def stride(self):
        return self._stride


class _Rotary:
    def __init__(self):
        self.supported = True
        self.failure = None
        self.calls = 0
        self.output = object()

    def supports_ltx_split_rope_direct(self):
        return True

    def ltx_split_rope_direct_supported(self, _input, _cos, _sin):
        return self.supported

    def apply_ltx_split_rope_direct(self, _input, _cos, _sin):
        self.calls += 1
        if self.failure is not None:
            if isinstance(self.failure, BaseException):
                raise self.failure
            raise RuntimeError(self.failure)
        return self.output


def _load_patch(monkeypatch, *, target="bmg"):
    package_name = "omnixpu_rotary_test"
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

    rotary = _Rotary()
    probe = types.SimpleNamespace(rotary=rotary)
    monkeypatch.setitem(sys.modules, "ComfyUI-OmniXPU.probe", probe)
    omni_package = types.ModuleType("omni_xpu_kernel")
    omni_package.__xpu_target__ = target
    monkeypatch.setitem(sys.modules, "omni_xpu_kernel", omni_package)

    calls = []

    def original(input_tensor, cos, sin):
        calls.append((input_tensor, cos, sin))
        return "fallback"

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    lightricks = types.ModuleType("comfy.ldm.lightricks")
    lightricks.__path__ = []
    model = types.ModuleType("comfy.ldm.lightricks.model")
    model.apply_split_rotary_emb = original
    comfy.ldm = ldm
    ldm.lightricks = lightricks
    lightricks.model = model
    for module in (comfy, ldm, lightricks, model):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    patch = _load_module(
        f"{adapters.__name__}.rotary", _ADAPTERS / "rotary.py"
    )
    return patch, rotary, model, calls


def _inputs():
    input_tensor = _Tensor(
        (2, 3520, 4096),
        (3520 * 4096, 4096, 1),
    )
    frequency_stride = (3520 * 32 * 64, 64, 32 * 64, 1)
    cos = _Tensor((2, 32, 3520, 64), frequency_stride)
    sin = _Tensor((2, 32, 3520, 64), frequency_stride)
    return input_tensor, cos, sin


def test_supported_contract_routes_directly(monkeypatch):
    patch, rotary, model, calls = _load_patch(monkeypatch)
    assert patch.apply() == (True, "")

    assert model.apply_split_rotary_emb(*_inputs()) is rotary.output
    assert rotary.calls == 1
    assert not calls
    assert patch.get_stats() == {
        "routed": 1,
        "fallback": 0,
        "quarantined_contracts": 0,
    }


def test_unsupported_contract_preserves_comfyui_fallback(monkeypatch):
    patch, rotary, model, calls = _load_patch(monkeypatch)
    rotary.supported = False
    assert patch.apply() == (True, "")

    assert model.apply_split_rotary_emb(*_inputs()) == "fallback"
    assert rotary.calls == 0
    assert len(calls) == 1


def test_runtime_failure_quarantines_exact_contract(monkeypatch):
    patch, rotary, model, calls = _load_patch(monkeypatch)
    rotary.failure = "candidate failure"
    assert patch.apply() == (True, "")

    assert model.apply_split_rotary_emb(*_inputs()) == "fallback"
    assert model.apply_split_rotary_emb(*_inputs()) == "fallback"
    assert rotary.calls == 1
    assert len(calls) == 2
    assert patch.get_stats()["quarantined_contracts"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        torch.OutOfMemoryError("synthetic rotary XPU OOM"),
        RuntimeError("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"),
    ],
)
def test_fatal_runtime_failure_does_not_enter_fallback(
    monkeypatch, failure
):
    patch, rotary, model, calls = _load_patch(monkeypatch)
    rotary.failure = failure
    assert patch.apply() == (True, "")

    with pytest.raises(type(failure)) as raised:
        model.apply_split_rotary_emb(*_inputs())

    assert raised.value is failure
    assert rotary.calls == 1
    assert not calls
    assert patch.get_stats() == {
        "routed": 0,
        "fallback": 0,
        "quarantined_contracts": 0,
    }


def test_non_bmg_package_does_not_patch(monkeypatch):
    patch, _rotary, model, _calls = _load_patch(
        monkeypatch, target="ptl-h"
    )
    original = model.apply_split_rotary_emb

    ok, reason = patch.apply()
    assert not ok
    assert "BMG-only" in reason
    assert model.apply_split_rotary_emb is original
