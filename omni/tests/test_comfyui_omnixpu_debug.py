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
    monkeypatch.delenv("OMNIXPU_DEBUG_VERBOSE", raising=False)
    assert not _DEBUG.debug_enabled()
    assert not _DEBUG.verbose_debug_enabled()


def test_debug_flag_accepts_common_true_values(monkeypatch):
    monkeypatch.delenv("OMNIXPU_DEBUG_VERBOSE", raising=False)
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("OMNIXPU_DEBUG", value)
        assert _DEBUG.debug_enabled()


def test_verbose_flag_enables_kernel_and_dispatch_tracing(monkeypatch):
    monkeypatch.delenv("OMNIXPU_DEBUG", raising=False)
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "1")
    assert _DEBUG.debug_enabled()
    assert _DEBUG.verbose_debug_enabled()


def test_disabled_trace_does_not_wrap(monkeypatch):
    monkeypatch.setenv("OMNIXPU_DEBUG", "0")
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "0")

    def operation(x):
        return x

    assert _DEBUG.trace_patch("operation", ("x",))(operation) is operation


def test_kernel_debug_does_not_install_verbose_dispatch_wrapper(monkeypatch):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "0")

    def operation(x):
        return x

    decorated = _DEBUG.trace_patch(
        "operation", ("x",), stage="dispatch", verbose_only=True
    )(operation)
    assert decorated is operation


def test_enabled_trace_ignores_cpu_calls_by_default(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "0")

    @_DEBUG.trace_patch("sample", ("x",))
    def operation(x):
        return x

    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        operation(torch.zeros((2, 3)))
    assert not caplog.records


def test_enabled_trace_formats_tensor_metadata(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "0")

    @_DEBUG.trace_patch(
        "sample",
        ("x", "nested"),
        xpu_only=False,
        details={"backend": "omni_xpu"},
    )
    def operation(x, nested):
        return x

    x = torch.zeros((2, 3), dtype=torch.bfloat16)
    nested = {"weight": torch.ones((4, 3), dtype=torch.int8)}
    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        assert operation(x, nested) is x

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "[OmniXPU DEBUG] stage=kernel op=sample backend=omni_xpu" in message
    assert "x(shape=(2, 3), dtype=torch.bfloat16, device=cpu)" in message
    assert "nested['weight'](shape=(4, 3), dtype=torch.int8, device=cpu)" in message


def test_verbose_trace_logs_dispatch_metadata(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "0")
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "1")

    @_DEBUG.trace_patch(
        "mixed_precision.Linear",
        ("input",),
        xpu_only=False,
        stage="dispatch",
        details={
            "quant_format": "int8_tensorwise",
            "layout": "TensorWiseINT8Layout",
        },
        verbose_only=True,
    )
    def operation(input):
        return input

    x = torch.zeros((2, 3), dtype=torch.bfloat16)
    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        assert operation(x) is x

    message = caplog.records[0].getMessage()
    assert "stage=dispatch op=mixed_precision.Linear" in message
    assert "quant_format=int8_tensorwise" in message
    assert "layout=TensorWiseINT8Layout" in message


@pytest.mark.skipif(not _XPU_AVAILABLE, reason="XPU is unavailable")
def test_enabled_trace_logs_xpu_calls(monkeypatch, caplog):
    monkeypatch.setenv("OMNIXPU_DEBUG", "1")
    monkeypatch.setenv("OMNIXPU_DEBUG_VERBOSE", "0")

    @_DEBUG.trace_patch("xpu_sample", ("x",))
    def operation(x):
        return x

    x = torch.zeros((2, 3), device="xpu", dtype=torch.bfloat16)
    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        assert operation(x) is x

    message = caplog.records[0].getMessage()
    assert "stage=kernel op=xpu_sample" in message
    assert "shape=(2, 3)" in message
    assert "device=xpu:0" in message
