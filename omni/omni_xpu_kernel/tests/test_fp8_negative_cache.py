"""CPU-only coverage for the Python FP8 negative-cache wrapper."""

import logging

import pytest
import torch


class _UnsupportedNative:
    def __init__(self):
        self.calls = 0

    def onednn_w8a16_fp8(self, x, weight, scales, bias):
        self.calls += 1
        raise RuntimeError(
            "OMNI_FP8_PRIMITIVE_UNSUPPORTED:new: device=xpu:0 M=4 K=8 N=2"
        )

    def fp8_cache_clear(self):
        return None

    def fp8_failure_cache_stats(self):
        return 1, 0, 1


def _case():
    return (
        torch.zeros((4, 8), dtype=torch.float16),
        torch.zeros((2, 8), dtype=torch.float8_e4m3fn),
        torch.ones((2,), dtype=torch.float32),
    )


def test_try_fp8_logs_once_and_short_circuits_native(monkeypatch, caplog):
    from omni_xpu_kernel import linear

    native = _UnsupportedNative()
    monkeypatch.setattr(linear, "_get_native", lambda: native)
    linear.fp8_cache_clear()

    with caplog.at_level(logging.WARNING, logger="omni_xpu_kernel.fp8"):
        assert linear.try_onednn_w8a16_fp8(*_case()) is None
        assert linear.try_onednn_w8a16_fp8(*_case()) is None

    assert native.calls == 1
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "input_shape=(4, 8)" in warnings[0].message
    assert "weight_shape=(2, 8)" in warnings[0].message
    assert linear.fp8_failure_cache_stats() == {
        "failures": 1,
        "negative_hits": 1,
        "size": 1,
    }


def test_try_fp8_does_not_suppress_other_runtime_errors(monkeypatch):
    from omni_xpu_kernel import linear

    class _BrokenNative(_UnsupportedNative):
        def onednn_w8a16_fp8(self, x, weight, scales, bias):
            raise RuntimeError("device lost")

    native = _BrokenNative()
    monkeypatch.setattr(linear, "_get_native", lambda: native)
    linear.fp8_cache_clear()

    with pytest.raises(RuntimeError, match="device lost"):
        linear.try_onednn_w8a16_fp8(*_case())
