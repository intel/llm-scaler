"""Tests for the OmniXPU allocator prestartup policy."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[1] / "ComfyUI-OmniXPU" / "prestartup_script.py"
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeXpu:
    def __init__(self, *, available: bool = True):
        self.available = available
        self.requested = []
        self.fraction = 1.0

    def is_available(self):
        return self.available

    def set_per_process_memory_fraction(self, fraction):
        self.requested.append(fraction)
        self.fraction = fraction

    def get_per_process_memory_fraction(self):
        return self.fraction


def _install_fake_torch(monkeypatch, xpu):
    torch = types.ModuleType("torch")
    torch.xpu = xpu
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_unset_fraction_does_not_touch_torch_outside_windows(monkeypatch):
    monkeypatch.delenv("OMNIXPU_XPU_MEMORY_FRACTION", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    fake_xpu = _FakeXpu()
    _install_fake_torch(monkeypatch, fake_xpu)

    _load_script("omnixpu_prestartup_unset")

    assert fake_xpu.requested == []


def test_windows_defaults_to_point_99(monkeypatch, caplog):
    monkeypatch.delenv("OMNIXPU_XPU_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("OMNIXPU_ENABLE", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    fake_xpu = _FakeXpu()
    _install_fake_torch(monkeypatch, fake_xpu)

    with caplog.at_level("INFO", logger="ComfyUI-OmniXPU"):
        _load_script("omnixpu_prestartup_windows_default")

    assert fake_xpu.requested == [0.99]
    assert "requested=0.99 actual=0.99 source=Windows default" in caplog.text


@pytest.mark.parametrize("value", ["", "0", "off", "false", "disabled"])
def test_environment_can_disable_windows_default(monkeypatch, value):
    monkeypatch.setenv("OMNIXPU_XPU_MEMORY_FRACTION", value)
    monkeypatch.delenv("OMNIXPU_ENABLE", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    fake_xpu = _FakeXpu()
    _install_fake_torch(monkeypatch, fake_xpu)

    _load_script(f"omnixpu_prestartup_disabled_{value or 'empty'}")

    assert fake_xpu.requested == []


def test_master_switch_disables_prestartup_policy(monkeypatch):
    monkeypatch.delenv("OMNIXPU_XPU_MEMORY_FRACTION", raising=False)
    monkeypatch.setenv("OMNIXPU_ENABLE", "0")
    monkeypatch.setattr(sys, "platform", "win32")
    fake_xpu = _FakeXpu()
    _install_fake_torch(monkeypatch, fake_xpu)

    _load_script("omnixpu_prestartup_master_disabled")

    assert fake_xpu.requested == []


def test_fraction_is_applied_and_read_back(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_XPU_MEMORY_FRACTION", " 0.99 ")
    monkeypatch.delenv("OMNIXPU_ENABLE", raising=False)
    fake_xpu = _FakeXpu()
    _install_fake_torch(monkeypatch, fake_xpu)

    with caplog.at_level("INFO", logger="ComfyUI-OmniXPU"):
        module = _load_script("omnixpu_prestartup_applied")

    assert fake_xpu.requested == [0.99]
    assert module.apply_xpu_memory_fraction() == pytest.approx(0.99)
    assert fake_xpu.requested == [0.99, 0.99]
    assert "requested=0.99 actual=0.99 source=environment" in caplog.text


@pytest.mark.parametrize(
    "value", ["not-a-number", "-0.1", "1.01", "nan", "inf"]
)
def test_invalid_fraction_fails_before_touching_torch(monkeypatch, value):
    monkeypatch.setenv("OMNIXPU_XPU_MEMORY_FRACTION", value)
    monkeypatch.delenv("OMNIXPU_ENABLE", raising=False)
    fake_xpu = _FakeXpu()
    _install_fake_torch(monkeypatch, fake_xpu)

    with pytest.raises(SystemExit, match="expected"):
        _load_script(f"omnixpu_prestartup_invalid_{value.replace('.', '_')}")

    assert fake_xpu.requested == []


def test_explicit_fraction_requires_available_xpu(monkeypatch):
    monkeypatch.setenv("OMNIXPU_XPU_MEMORY_FRACTION", "0.99")
    monkeypatch.delenv("OMNIXPU_ENABLE", raising=False)
    _install_fake_torch(monkeypatch, _FakeXpu(available=False))

    with pytest.raises(SystemExit, match="torch.xpu is unavailable"):
        _load_script("omnixpu_prestartup_no_xpu")
