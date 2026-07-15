"""Portable tests for ComfyUI-OmniXPU patch tracing."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest
import torch

_DEBUG_PATH = Path(__file__).parents[1] / "ComfyUI-OmniXPU" / "patches" / "debug.py"
if not _DEBUG_PATH.is_file():
    pytest.skip(
        f"OmniXPU debug helper is unavailable: {_DEBUG_PATH}", allow_module_level=True
    )

_SPEC = importlib.util.spec_from_file_location("omnixpu_debug_test_module", _DEBUG_PATH)
if _SPEC is None or _SPEC.loader is None:
    pytest.skip("Unable to load the OmniXPU debug helper", allow_module_level=True)

_DEBUG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEBUG)

_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()


def test_debug_flag_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("OMNIXPU_DEBUG", raising=False)
    assert not _DEBUG.debug_enabled()


def test_debug_flag_accepts_common_true_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("OMNIXPU_DEBUG", value)
        assert _DEBUG.debug_enabled()


def test_disabled_trace_does_not_wrap(monkeypatch):
    monkeypatch.setenv("OMNIXPU_DEBUG", "0")

    def operation(x):
        return x

    assert _DEBUG.trace_patch("operation", ("x",))(operation) is operation


def test_enabled_trace_ignores_cpu_calls_by_default(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")

    @_DEBUG.trace_patch("sample", ("x",))
    def operation(x):
        return x

    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        operation(torch.zeros((2, 3)))
    assert not caplog.records


def test_enabled_trace_formats_tensor_metadata(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")

    @_DEBUG.trace_patch("sample", ("x", "nested"), xpu_only=False)
    def operation(x, nested):
        return x

    x = torch.zeros((2, 3), dtype=torch.bfloat16)
    nested = {"weight": torch.ones((4, 3), dtype=torch.int8)}
    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        assert operation(x, nested) is x

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "[OmniXPU DEBUG] patch=sample" in message
    assert "x(shape=(2, 3), dtype=torch.bfloat16, device=cpu)" in message
    assert "nested['weight'](shape=(4, 3), dtype=torch.int8, device=cpu)" in message


@pytest.mark.skipif(not _XPU_AVAILABLE, reason="XPU is unavailable")
def test_enabled_trace_logs_xpu_calls(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")

    @_DEBUG.trace_patch("xpu_sample", ("x",))
    def operation(x):
        return x

    x = torch.zeros((2, 3), device="xpu", dtype=torch.bfloat16)
    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        assert operation(x) is x

    message = caplog.records[0].getMessage()
    assert "patch=xpu_sample" in message
    assert "shape=(2, 3)" in message
    assert "device=xpu:0" in message
