"""Portable control-flow tests for platform/workflow attention routing."""

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
        seq=1088,
        heads=30,
        dim_head=128,
        dtype=torch.bfloat16,
        pre_shaped=True,
    ):
        if pre_shaped:
            self.shape = (1, heads, seq, dim_head)
        else:
            self.shape = (1, seq, heads * dim_head)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="xpu")

    def view(self, *shape):
        return self

    def contiguous(self):
        return self

    def reshape(self, *shape):
        return self

    def permute(self, *dims):
        return self

    def __ne__(self, other):
        return types.SimpleNamespace(any=lambda: False)


def _load_patch(
    monkeypatch,
    *,
    target="ptl-h",
    torch_version="2.11.0+xpu",
    backend="auto",
):
    monkeypatch.setenv("OMNI_ATTN_BACKEND", backend)
    monkeypatch.setattr(torch, "__version__", torch_version)
    package_name = "omnixpu_attention_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")

    calls = []

    def torch_attention(q, k, v, heads, **kwargs):
        calls.append("torch")
        return "torch-output"

    def original_attention(*args, **kwargs):
        return None

    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.wrap_attn = lambda fn: fn
    attention.attention_basic = original_attention
    attention.attention_pytorch = torch_attention
    attention.optimized_attention = original_attention
    attention.optimized_attention_masked = original_attention

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    modules = types.ModuleType("comfy.ldm.modules")
    modules.__path__ = []
    comfy.ldm = ldm
    ldm.modules = modules
    modules.attention = attention

    class Cute:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def sdp(q, k, v):
            calls.append("cute")
            return q

    omni = types.ModuleType("omni_xpu_kernel")
    omni.__xpu_target__ = target
    omni.__path__ = []
    omni.cute = Cute

    probe = types.ModuleType("ComfyUI-OmniXPU.probe")
    probe.sdp = Cute
    for name, module in (
        ("comfy", comfy),
        ("comfy.ldm", ldm),
        ("comfy.ldm.modules", modules),
        ("comfy.ldm.modules.attention", attention),
        ("omni_xpu_kernel", omni),
        ("ComfyUI-OmniXPU.probe", probe),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    patch = _load_module(
        f"{patches.__name__}.patch_attention",
        _PATCHES / "patch_attention.py",
    )
    assert patch.apply() == (True, None)
    return patch, attention, calls


@pytest.mark.parametrize("seq", [64, 1024, 1088])
def test_ptl_auto_torch211_zimage_shape_uses_torch(monkeypatch, seq):
    patch, attention, calls = _load_patch(monkeypatch)
    tensor = _FakeTensor(seq=seq)
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["torch_sdpa"] == 1
    assert patch.get_stats()["fallback"] == 0


def test_explicit_cute_does_not_apply_auto_route(monkeypatch):
    _, attention, calls = _load_patch(monkeypatch, backend="cute")
    tensor = _FakeTensor()
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]


@pytest.mark.parametrize(
    ("target", "torch_version", "tensor", "heads"),
    [
        ("bmg", "2.11.0+xpu", _FakeTensor(), 30),
        ("ptl-h", "2.10.0+xpu", _FakeTensor(), 30),
        ("ptl-h", "2.12.0+xpu", _FakeTensor(), 30),
        ("ptl-h", "2.11.0+xpu", _FakeTensor(heads=24), 24),
        ("ptl-h", "2.11.0+xpu", _FakeTensor(seq=4096), 30),
        (
            "ptl-h",
            "2.11.0+xpu",
            _FakeTensor(dtype=torch.float16),
            30,
        ),
    ],
)
def test_unvalidated_auto_shapes_keep_cute(
    monkeypatch, target, torch_version, tensor, heads
):
    patch, attention, calls = _load_patch(
        monkeypatch, target=target, torch_version=torch_version
    )
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=heads, skip_reshape=True
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert patch.get_stats()["torch_sdpa"] == 0


@pytest.mark.parametrize(
    ("tensor", "kwargs"),
    [
        (_FakeTensor(pre_shaped=False), {}),
        (_FakeTensor(), {"skip_reshape": True, "skip_output_reshape": True}),
    ],
)
def test_unvalidated_layouts_keep_cute(monkeypatch, tensor, kwargs):
    patch, attention, calls = _load_patch(monkeypatch)
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, **kwargs
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert patch.get_stats()["torch_sdpa"] == 0
