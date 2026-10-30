"""Tests for the discrete-XPU DynamicVRAM RAM staging adapter."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest


_ADAPTER = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "adapters"
    / "xpu_ram_staging.py"
)
_GIB = 1024**3
_BUDGET_SOURCE = """
if not DISABLE_PINNED_MEMORY:
    if is_nvidia() or is_amd():
        if WINDOWS:
            MAX_PINNED_MEMORY = ram * 0.40
"""
_REGISTRATION_SOURCE = "if torch.cuda.cudart().cudaHostRegister(ptr, size, 1) == 0:\n"


def _load_adapter():
    name = "omnixpu_xpu_ram_staging_test"
    spec = importlib.util.spec_from_file_location(name, _ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _no_cuda():
    raise AssertionError("Torch not compiled with CUDA enabled")


def _fakes(*, integrated=0, windows=True, disabled=False, budget=-1, cgroup=None):
    torch_module = types.SimpleNamespace(
        device=lambda kind: kind,
        cuda=types.SimpleNamespace(is_available=lambda: False, cudart=_no_cuda),
        xpu=types.SimpleNamespace(
            get_device_properties=lambda device: types.SimpleNamespace(is_integrated_gpu=integrated)
        ),
    )
    model_management = types.SimpleNamespace(
        is_intel_xpu=lambda: True,
        DISABLE_PINNED_MEMORY=disabled,
        MAX_PINNED_MEMORY=budget,
        WINDOWS=windows,
        get_total_memory=lambda device: 64 * _GIB,
        get_torch_device=lambda: "xpu:0",
        get_disk_swap_total=lambda: 8 * _GIB,
        comfy=types.SimpleNamespace(
            system_memory=types.SimpleNamespace(cgroup_memory_limit=lambda: cgroup)
        ),
    )
    return model_management, torch_module


def _sources(registration=_REGISTRATION_SOURCE):
    return (_BUDGET_SOURCE, [_BUDGET_SOURCE, registration])


def test_discrete_xpu_gets_the_windows_budget_and_registration_answers():
    adapter = _load_adapter()
    mm, torch_module = _fakes()

    assert adapter._enable(mm, _sources(), torch_module) == (True, None)
    assert mm.MAX_PINNED_MEMORY == pytest.approx(64 * _GIB * 0.40)
    cudart = torch_module.cuda.cudart()
    assert cudart.cudaHostRegister(1, 2, 1) == 0
    assert cudart.cudaHostUnregister(1) == 0
    with pytest.raises(AssertionError, match="not compiled with CUDA"):
        cudart.cudaMemcpy
    assert adapter._enable(mm, _sources(), torch_module) == (
        False,
        "XPU RAM staging is already enabled",
    )


@pytest.mark.parametrize(
    ("cgroup", "expected"),
    [(None, min(64 * 0.90, 64 - 4, 64 + 8 - 16)), (32 * _GIB, min(64 * 0.90, 64 - 4, 64 - 16))],
)
def test_linux_budget_follows_comfyui_swap_rules(cgroup, expected):
    adapter = _load_adapter()
    mm, torch_module = _fakes(windows=False, cgroup=cgroup)

    assert adapter._enable(mm, _sources(), torch_module) == (True, None)
    assert mm.MAX_PINNED_MEMORY == pytest.approx(expected * _GIB)


@pytest.mark.parametrize(
    ("fake_options", "reason"),
    [
        ({"integrated": 1}, "integrated XPU memory is system RAM"),
        ({"integrated": None}, "Torch does not report whether the XPU is integrated"),
        ({"disabled": True}, "pinned memory disabled by --disable-pinned-memory"),
        ({"budget": 1024}, "ComfyUI already stages weights in RAM on this backend"),
    ],
)
def test_upstream_behavior_is_kept_when_staging_does_not_apply(fake_options, reason):
    adapter = _load_adapter()
    mm, torch_module = _fakes(**fake_options)
    budget = mm.MAX_PINNED_MEMORY

    assert adapter._enable(mm, _sources(), torch_module) == (False, reason)
    assert mm.MAX_PINNED_MEMORY == budget
    assert torch_module.cuda.cudart is _no_cuda


def test_unknown_cudart_usage_is_rejected():
    adapter = _load_adapter()
    mm, torch_module = _fakes()
    sources = _sources(_REGISTRATION_SOURCE + "torch.cuda.cudart().cudaMemcpy(a, b, n, k)\n")

    assert adapter._enable(mm, sources, torch_module) == (
        False,
        "unsupported cudart calls: cudaMemcpy",
    )
    assert torch_module.cuda.cudart is _no_cuda


def test_installed_comfyui_matches_the_contract():
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = True
    model_management = pytest.importorskip("comfy.model_management")
    pinned_memory = pytest.importorskip("comfy.pinned_memory")
    model_patcher = pytest.importorskip("comfy.model_patcher")
    adapter = _load_adapter()

    sources = [inspect.getsource(m) for m in (model_management, pinned_memory, model_patcher)]
    assert adapter._contract_violation(sources[0], sources) is None
