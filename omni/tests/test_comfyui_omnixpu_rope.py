"""Portable control-flow tests for the PTL Z-Image pair-RoPE route."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(
        self,
        *,
        length=1088,
        heads=30,
        dim=128,
        dtype=torch.bfloat16,
        contiguous=True,
        device=None,
    ):
        self.shape = (1, length, heads, dim)
        self.ndim = 4
        self.dtype = dtype
        self.device = device or types.SimpleNamespace(type="xpu")
        self.is_xpu = self.device.type == "xpu"
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class _FakeFreqs(_FakeTensor):
    def __init__(
        self,
        *,
        length=1088,
        dtype=torch.float32,
        contiguous=True,
        device=None,
    ):
        super().__init__(
            length=length,
            dtype=dtype,
            contiguous=contiguous,
            device=device,
        )
        self.shape = (1, length, 1, 64, 2, 2)
        self.ndim = 6


def _load_patch(
    monkeypatch,
    *,
    target="ptl-h",
    torch_version="2.11.0+xpu",
    enabled=True,
):
    monkeypatch.setattr(torch, "__version__", torch_version)
    if enabled:
        monkeypatch.delenv("OMNIXPU_ZIMAGE_ROPE_PAIR", raising=False)
    else:
        monkeypatch.setenv("OMNIXPU_ZIMAGE_ROPE_PAIR", "0")

    package_name = "omnixpu_rope_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")

    calls = []

    class Rotary:
        @staticmethod
        def apply_kitchen_rope(q, k, freqs):
            calls.append("kitchen_pair")
            return "q-pair", "k-pair"

    def original_apply_rope(q, k, freqs):
        calls.append("original_pair")
        return "q-original", "k-original"

    def original_apply_rope1(x, freqs):
        calls.append("original_single")
        return x

    flux_math = types.ModuleType("comfy.ldm.flux.math")
    flux_math._apply_rope1 = original_apply_rope1
    flux_math.apply_rope1 = original_apply_rope1
    flux_math.apply_rope = original_apply_rope

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    flux = types.ModuleType("comfy.ldm.flux")
    flux.__path__ = []
    comfy.ldm = ldm
    ldm.flux = flux
    flux.math = flux_math

    omni = types.ModuleType("omni_xpu_kernel")
    omni.__xpu_target__ = target
    probe = types.ModuleType("ComfyUI-OmniXPU.probe")
    probe.rotary = Rotary
    for name, module in (
        ("comfy", comfy),
        ("comfy.ldm", ldm),
        ("comfy.ldm.flux", flux),
        ("comfy.ldm.flux.math", flux_math),
        ("omni_xpu_kernel", omni),
        ("ComfyUI-OmniXPU.probe", probe),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    patch = _load_module(
        f"{patches.__name__}.patch_rope",
        _PATCHES / "patch_rope.py",
    )
    assert patch.apply() == (True, None)
    return patch, flux_math, calls


@pytest.mark.parametrize("length", [64, 1024, 1088])
def test_ptl_torch211_zimage_pair_uses_kitchen(monkeypatch, length):
    _patch, flux_math, calls = _load_patch(monkeypatch)
    device = types.SimpleNamespace(type="xpu")
    q = _FakeTensor(length=length, device=device)
    k = _FakeTensor(length=length, device=device)
    freqs = _FakeFreqs(length=length, device=device)
    assert flux_math.apply_rope(q, k, freqs) == ("q-pair", "k-pair")
    assert calls == ["kitchen_pair"]


@pytest.mark.parametrize(
    ("target", "torch_version", "length", "heads", "dtype", "enabled"),
    [
        ("bmg", "2.11.0+xpu", 1088, 30, torch.bfloat16, True),
        ("ptl-h", "2.10.0+xpu", 1088, 30, torch.bfloat16, True),
        ("ptl-h", "2.12.0+xpu", 1088, 30, torch.bfloat16, True),
        ("ptl-h", "2.11.0+xpu", 4096, 30, torch.bfloat16, True),
        ("ptl-h", "2.11.0+xpu", 1088, 24, torch.bfloat16, True),
        ("ptl-h", "2.11.0+xpu", 1088, 30, torch.float16, True),
        ("ptl-h", "2.11.0+xpu", 1088, 30, torch.bfloat16, False),
    ],
)
def test_unvalidated_pair_keeps_existing_route(
    monkeypatch, target, torch_version, length, heads, dtype, enabled
):
    patch, _flux_math, calls = _load_patch(
        monkeypatch,
        target=target,
        torch_version=torch_version,
        enabled=enabled,
    )
    device = types.SimpleNamespace(type="xpu")
    q = _FakeTensor(
        length=length,
        heads=heads,
        dtype=dtype,
        device=device,
    )
    k = _FakeTensor(
        length=length,
        heads=heads,
        dtype=dtype,
        device=device,
    )
    freqs = _FakeFreqs(length=length, device=device)
    assert not patch._use_ptl_zimage_pair(q, k, freqs)
    assert calls == []
