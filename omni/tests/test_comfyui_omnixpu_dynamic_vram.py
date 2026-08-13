"""Tests for Windows XPU DynamicVRAM boundary reclaim."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


_MODULE = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "adapters"
    / "dynamic_vram.py"
)
_MIB = 1024**2


def _load_module(monkeypatch):
    name = "omnixpu_dynamic_vram_test"
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _Device:
    def __init__(self, device_type, index=0):
        self.type = device_type
        self.index = index

    def __hash__(self):
        return hash((self.type, self.index))

    def __eq__(self, other):
        return (self.type, self.index) == (other.type, other.index)

    def __str__(self):
        return f"{self.type}:{self.index}"


class _Patcher:
    def __init__(self, name, device, loaded=0, dynamic=True, children=()):
        self.name = name
        self.load_device = device
        self.offload_device = _Device("cpu")
        self.loaded = loaded
        self.dynamic = dynamic
        self.children = children
        self.attachments = {}
        self.free_calls = []
        self.release = None

    def model_patches_models(self):
        return self.children

    def is_dynamic(self):
        return self.dynamic

    def loaded_size(self):
        return self.loaded

    def partially_unload(self, device, memory_to_free):
        assert device == self.offload_device
        freed = min(self.loaded, memory_to_free)
        self.loaded -= freed
        self.free_calls.append(memory_to_free)
        self.release(freed)
        return freed


class _LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device

    def is_dead(self):
        return False


def _model_management(models, free, reserve=4 * 1024 * _MIB):
    state = {"free": free}

    def release(amount):
        state["free"] += amount

    for model in models:
        model.release = release

    calls = []

    def original(requested, *args, **kwargs):
        calls.append((requested, args, kwargs, state["free"]))
        return "loaded"

    management = SimpleNamespace(
        current_loaded_models=[_LoadedModel(model) for model in models],
        extra_reserved_memory=lambda: reserve,
        get_free_memory=lambda device: state["free"],
        load_models_gpu=original,
        minimum_inference_memory=lambda: reserve + 800 * _MIB,
    )
    return management, calls


def test_minimum_target_includes_reserved_vram(monkeypatch):
    dynamic_vram = _load_module(monkeypatch)
    management, _ = _model_management([], 0)

    target = dynamic_vram._minimum_memory_target(
        management, (), {"memory_required": 20 * 1024 * _MIB}
    )
    minimum_target = dynamic_vram._minimum_memory_target(
        management,
        (),
        {
            "memory_required": 20 * 1024 * _MIB,
            "minimum_memory_required": 16 * 1024 * _MIB,
        },
    )

    assert target == 24 * 1024 * _MIB
    assert minimum_target == 20 * 1024 * _MIB


def test_boundary_trim_reclaims_inactive_before_requested_model(monkeypatch):
    dynamic_vram = _load_module(monkeypatch)
    device = _Device("xpu")
    requested = _Patcher("requested", device, loaded=5 * 1024 * _MIB)
    inactive = _Patcher("inactive", device, loaded=8 * 1024 * _MIB)
    management, calls = _model_management(
        [requested, inactive], free=10 * 1024 * _MIB
    )
    dynamic_vram._patch_model_loader(management)

    result = management.load_models_gpu(
        [requested], minimum_memory_required=16 * 1024 * _MIB
    )

    assert result == "loaded"
    assert inactive.loaded == 0
    assert requested.loaded == 3 * 1024 * _MIB
    assert management.get_free_memory(device) == 20 * 1024 * _MIB
    assert len(inactive.free_calls) == 1
    assert len(requested.free_calls) == 1
    assert calls[0][3] == 20 * 1024 * _MIB


def test_boundary_trim_stops_after_inactive_models_meet_target(monkeypatch):
    dynamic_vram = _load_module(monkeypatch)
    device = _Device("xpu")
    requested = _Patcher("requested", device, loaded=5 * 1024 * _MIB)
    inactive = _Patcher("inactive", device, loaded=12 * 1024 * _MIB)
    management, _ = _model_management(
        [requested, inactive], free=10 * 1024 * _MIB
    )
    dynamic_vram._patch_model_loader(management)

    management.load_models_gpu(
        [requested], minimum_memory_required=16 * 1024 * _MIB
    )

    assert inactive.loaded == 2 * 1024 * _MIB
    assert requested.loaded == 5 * 1024 * _MIB
    assert requested.free_calls == []


def test_boundary_trim_skips_mixed_and_non_xpu_requests(monkeypatch):
    dynamic_vram = _load_module(monkeypatch)
    xpu = _Device("xpu")
    cpu = _Device("cpu")
    dynamic = _Patcher("dynamic", xpu, loaded=8 * 1024 * _MIB)
    static = _Patcher("static", xpu, dynamic=False)
    cpu_dynamic = _Patcher("cpu", cpu, loaded=8 * 1024 * _MIB)
    management, _ = _model_management(
        [dynamic, cpu_dynamic], free=10 * 1024 * _MIB
    )
    dynamic_vram._patch_model_loader(management)

    management.load_models_gpu(
        [dynamic, static], minimum_memory_required=16 * 1024 * _MIB
    )
    management.load_models_gpu(
        [cpu_dynamic], minimum_memory_required=16 * 1024 * _MIB
    )

    assert dynamic.free_calls == []
    assert cpu_dynamic.free_calls == []


def test_lora_budget_reaches_boundary_trim_before_core_loader(monkeypatch):
    dynamic_vram = _load_module(monkeypatch)
    lora_name = "omnixpu_dynamic_vram_lora_test"
    spec = importlib.util.spec_from_file_location(
        lora_name,
        _MODULE.with_name("lora_memory.py"),
    )
    assert spec is not None and spec.loader is not None
    lora_memory = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, lora_name, lora_memory)
    spec.loader.exec_module(lora_memory)
    monkeypatch.setattr(lora_memory, "_xpu_snapshot", lambda device=None: None)

    device = _Device("xpu")
    requested = _Patcher("requested", device, loaded=5 * 1024 * _MIB)
    inactive = _Patcher("inactive", device, loaded=8 * 1024 * _MIB)
    management, calls = _model_management(
        [requested, inactive], free=10 * 1024 * _MIB
    )
    lora_tensor = object()
    lora_memory._set_budget_entries(
        requested, {id(lora_tensor): (2 * 1024 * _MIB, 2 * 1024 * _MIB)}
    )

    dynamic_vram._patch_model_loader(management)
    lora_memory._patch_model_loader(management)
    management.load_models_gpu(
        [requested], minimum_memory_required=14 * 1024 * _MIB
    )

    assert management.get_free_memory(device) == 20 * 1024 * _MIB
    assert calls[0][2]["minimum_memory_required"] == 16 * 1024 * _MIB
    assert calls[0][3] == 20 * 1024 * _MIB
