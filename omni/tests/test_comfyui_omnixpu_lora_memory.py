"""Tests for the ComfyUI-OmniXPU LoRA memory budget adapter."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


_MODULE = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "adapters"
    / "lora_memory.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("omnixpu_lora_memory_test", _MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, numel, element_size):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakePatch:
    def __init__(self, weights):
        self.weights = weights


class FakePatcher:
    def __init__(self, patches=None, attachments=None, additional=None):
        self.patches = patches or {}
        self.attachments = attachments or {}
        self.additional = additional or []

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def get_attachment(self, key):
        return self.attachments.get(key)

    def model_patches_models(self):
        return self.additional


def _patcher():
    first = FakeTensor(3, 4)
    second = FakeTensor(513, 2)
    patch = FakePatch((first, second, None))
    return FakePatcher(
        patches={
            "block.weight": [(1.0, patch, 1.0, None, None)],
            "shared.weight": [(1.0, patch, 1.0, None, None)],
        }
    )


def test_patch_stats_count_unique_tensors_and_aligned_layer_peak():
    monitor = _load_module()

    stats = monitor._patcher_stats(_patcher())

    assert stats == {
        "payload": 1038,
        "staging": 3 * 1024,
        "tensors": 2,
        "peak": 3 * 1024,
    }


def test_lora_loader_caches_attached_tensor_budget(caplog):
    monitor = _load_module()
    before = FakePatcher()
    after = _patcher()
    comfy_sd = types.SimpleNamespace(
        load_lora_for_models=lambda model, clip, lora, strength_model, strength_clip: (
            after,
            None,
        )
    )
    monitor._patch_lora_loader(comfy_sd)

    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        result = comfy_sd.load_lora_for_models(
            before, None, {"a": FakeTensor(513, 2)}, 1.0, 0.0
        )

    budget = monitor._entries_stats(monitor._get_budget_entries(after))
    assert result == (after, None)
    assert budget == {"payload": 1038, "staging": 3 * 1024, "tensors": 2}
    assert "LoRA memory budget cached" in caplog.text
    assert "model_added=0.0MiB" in caplog.text


def test_stacked_lora_extends_inherited_budget_without_rescanning_at_load():
    monitor = _load_module()
    previous_tensor = FakeTensor(512, 2)
    added_tensor = FakeTensor(513, 2)
    before = FakePatcher(
        patches={"old.weight": [(1.0, previous_tensor, 1.0, None, None)]}
    )
    monitor._set_budget_entries(
        before, {id(previous_tensor): monitor._tensor_size(previous_tensor)}
    )
    after = FakePatcher(
        patches={
            "old.weight": [(1.0, previous_tensor, 1.0, None, None)],
            "new.weight": [(1.0, added_tensor, 1.0, None, None)],
        },
        attachments=dict(before.attachments),
    )
    comfy_sd = types.SimpleNamespace(
        load_lora_for_models=lambda *args, **kwargs: (after, None)
    )
    monitor._patch_lora_loader(comfy_sd)

    comfy_sd.load_lora_for_models(
        before, None, {"new": added_tensor}, 1.0, 0.0
    )

    budget = monitor._entries_stats(monitor._get_budget_entries(after))
    assert budget == {
        "payload": 2050,
        "staging": 3 * 1024,
        "tensors": 2,
    }
    assert monitor._entries_stats(monitor._get_budget_entries(before)) == {
        "payload": 1024,
        "staging": 1024,
        "tensors": 1,
    }


def test_model_loader_uses_cached_budget_and_adjusts_explicit_minimum(
    monkeypatch, caplog
):
    monitor = _load_module()
    patcher = _patcher()
    entries = monitor._patcher_entries(patcher)
    monitor._set_budget_entries(patcher, entries)
    calls = []

    def load_models_gpu(models, *args, **kwargs):
        calls.append((models, args, kwargs))
        return "loaded"

    model_management = types.SimpleNamespace(load_models_gpu=load_models_gpu)
    monkeypatch.setattr(
        monitor,
        "_patcher_entries",
        lambda patcher: pytest.fail("model load must not rescan patch tensors"),
    )
    monkeypatch.setattr(
        monitor,
        "_xpu_snapshot",
        lambda device=None: {
            "allocated": 10,
            "reserved": 20,
            "free": 30,
            "total": 40,
        },
    )
    monitor._patch_model_loader(model_management)
    models = [patcher]

    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        result = model_management.load_models_gpu(
            models,
            memory_required=4096,
            minimum_memory_required=2048,
            force_full_load=True,
        )

    assert result == "loaded"
    assert calls == [
        (
            models,
            (),
            {
                "memory_required": 4096 + 3 * 1024,
                "minimum_memory_required": 2048 + 3 * 1024,
                "force_full_load": True,
            },
        )
    ]
    assert "LoRA model budget applied" in caplog.text
    assert "LoRA model load complete" in caplog.text


def test_model_loader_keeps_default_minimum_and_deduplicates_shared_tensors():
    monitor = _load_module()
    shared = FakeTensor(513, 2)
    entry = {id(shared): monitor._tensor_size(shared)}
    first = FakePatcher()
    second = FakePatcher()
    monitor._set_budget_entries(first, entry)
    monitor._set_budget_entries(second, entry)
    calls = []

    def load_models_gpu(models, *args, **kwargs):
        calls.append((models, args, kwargs))
        return "loaded"

    model_management = types.SimpleNamespace(load_models_gpu=load_models_gpu)
    monitor._patch_model_loader(model_management)
    models = [first, second]

    result = model_management.load_models_gpu(models, 4096)

    assert result == "loaded"
    assert calls == [(models, (4096 + 2 * 1024,), {})]


def test_apply_does_not_wrap_dynamic_staging_without_trace(monkeypatch):
    monitor = _load_module()
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.sd = types.ModuleType("comfy.sd")
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.ops = types.ModuleType("comfy.ops")
    comfy.sd.load_lora_for_models = lambda *args, **kwargs: (None, None)
    comfy.model_management.load_models_gpu = lambda *args, **kwargs: None
    comfy.ops.cast_modules_with_vbar = lambda *args, **kwargs: None
    original_dynamic_cast = comfy.ops.cast_modules_with_vbar
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy.sd)
    monkeypatch.setitem(
        sys.modules,
        "comfy.model_management",
        comfy.model_management,
    )
    monkeypatch.setitem(sys.modules, "comfy.ops", comfy.ops)
    monkeypatch.delenv("OMNIXPU_LORA_MEMORY_TRACE", raising=False)

    applied, message = monitor.apply()

    assert applied
    assert message == "whole-LoRA model budgets cached"
    assert comfy.ops.cast_modules_with_vbar is original_dynamic_cast


def test_dynamic_staging_monitor_logs_oom_and_reraises(monkeypatch, caplog):
    monitor = _load_module()

    def cast_modules_with_vbar(modules, dtype, device, bias_dtype, non_blocking):
        raise MemoryError("synthetic OOM")

    comfy_ops = types.SimpleNamespace(
        cast_modules_with_vbar=cast_modules_with_vbar
    )
    monkeypatch.setattr(monitor, "_xpu_snapshot", lambda device=None: None)
    monitor._patch_dynamic_cast(comfy_ops)
    lowvram_patch = types.SimpleNamespace(memory_required=lambda: 2 * 1024**2)
    module = types.SimpleNamespace(
        weight_lowvram_function=lowvram_patch,
        bias_lowvram_function=None,
    )

    with caplog.at_level(logging.INFO, logger="ComfyUI-OmniXPU"):
        with pytest.raises(MemoryError, match="synthetic OOM"):
            comfy_ops.cast_modules_with_vbar(
                [module], None, "xpu:0", None, False
            )

    assert "LoRA XPU staging new_peak: patch=2.0MiB" in caplog.text
    assert "LoRA XPU staging failed: patch=2.0MiB" in caplog.text
