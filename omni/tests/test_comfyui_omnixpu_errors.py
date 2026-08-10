"""Portable tests for accelerator error classification and FP8 fallback."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_ADAPTERS = _PLUGIN / "adapters"
_PATCHES = _PLUGIN / "patches"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_test_package(monkeypatch):
    package_name = "omnixpu_errors_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    adapters = types.ModuleType(f"{package_name}.adapters")
    adapters.__path__ = [str(_ADAPTERS)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")
    return package_name


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("XPU out of memory"),
        RuntimeError("PI_ERROR_OUT_OF_RESOURCES"),
        RuntimeError("ZE_RESULT_ERROR_DEVICE_LOST"),
    ],
)
def test_fatal_accelerator_errors_are_classified(monkeypatch, error):
    package_name = _install_test_package(monkeypatch)
    errors = _load_module(
        f"{package_name}.adapters.errors",
        _ADAPTERS / "errors.py",
    )

    assert errors.is_fatal_accelerator_error(error)


def test_wrapped_oom_is_classified(monkeypatch):
    package_name = _install_test_package(monkeypatch)
    errors = _load_module(
        f"{package_name}.adapters.errors",
        _ADAPTERS / "errors.py",
    )
    wrapped = RuntimeError("custom op failed")
    wrapped.__cause__ = torch.OutOfMemoryError("allocation failed")

    assert errors.is_fatal_accelerator_error(wrapped)


def test_ordinary_runtime_error_remains_recoverable(monkeypatch):
    package_name = _install_test_package(monkeypatch)
    errors = _load_module(
        f"{package_name}.adapters.errors",
        _ADAPTERS / "errors.py",
    )

    assert not errors.is_fatal_accelerator_error(
        RuntimeError("unsupported alignment")
    )


def test_fp8_oom_does_not_enter_original_fallback(monkeypatch):
    package_name = _install_test_package(monkeypatch)
    fallback_calls = []
    uncast_calls = []

    class FakeTensor:
        dtype = torch.float16
        shape = (2, 8)
        ndim = 2
        is_xpu = True
        device = types.SimpleNamespace(type="xpu")

    class FakeWeight:
        dtype = torch.float8_e4m3fn
        shape = (4, 8)

    model_management = types.ModuleType("comfy.model_management")
    model_management.lora_compute_dtype = lambda device: torch.float16
    comfy_ops = types.ModuleType("comfy.ops")

    def original_fp8_linear(module, input_tensor):
        fallback_calls.append((module, input_tensor))
        return "fallback"

    comfy_ops.fp8_linear = original_fp8_linear
    comfy_ops.cast_bias_weight = lambda *args, **kwargs: (
        FakeWeight(),
        None,
        None,
    )
    comfy_ops.uncast_bias_weight = lambda *args: uncast_calls.append(args)
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    comfy.ops = comfy_ops
    probe = types.SimpleNamespace(
        linear_fp8=lambda *args: (_ for _ in ()).throw(
            torch.OutOfMemoryError("synthetic FP8 XPU OOM")
        )
    )
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.ops", comfy_ops)
    monkeypatch.setitem(sys.modules, "ComfyUI-OmniXPU.probe", probe)

    fp8_gemm = _load_module(
        f"{package_name}.adapters.fp8_gemm",
        _ADAPTERS / "fp8_gemm.py",
    )
    assert fp8_gemm.apply() == (True, None)
    monkeypatch.setattr(
        fp8_gemm,
        "_prepare_scale",
        lambda scale, weight, input_tensor: scale,
    )
    module = types.SimpleNamespace(
        weight=FakeWeight(),
        scale_weight=torch.ones((), dtype=torch.float32),
    )

    with pytest.raises(torch.OutOfMemoryError, match="synthetic FP8"):
        comfy_ops.fp8_linear(module, FakeTensor())

    assert fallback_calls == []
    assert uncast_calls == []
