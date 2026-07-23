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
_ADAPTERS = _PLUGIN / "adapters"


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
        stride=None,
    ):
        if pre_shaped:
            self.shape = (1, heads, seq, dim_head)
            self._stride = stride or (
                heads * seq * dim_head,
                dim_head,
                heads * dim_head,
                1,
            )
        else:
            self.shape = (1, seq, heads * dim_head)
            self._stride = stride or (seq * heads * dim_head, heads * dim_head, 1)
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

    def transpose(self, *dims):
        return self

    def stride(self):
        return self._stride

    def __ne__(self, other):
        return types.SimpleNamespace(any=lambda: False)


def _load_patch(
    monkeypatch,
    *,
    target="ptl-h",
    torch_version="2.11.0+xpu",
    backend="auto",
    d120_capable=True,
):
    monkeypatch.setenv("OMNI_ATTN_BACKEND", backend)
    monkeypatch.setattr(torch, "__version__", torch_version)
    package_name = "omnixpu_attention_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    adapters = types.ModuleType(f"{package_name}.adapters")
    adapters.__path__ = [str(_ADAPTERS)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
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

        @staticmethod
        def supports_d120_bhld():
            return d120_capable

        @staticmethod
        def sdp_bhld_d120(q, k, v):
            calls.append("cute_d120")
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
        f"{adapters.__name__}.attention",
        _ADAPTERS / "attention.py",
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


def test_ptl_auto_torch211_krea2_shape_uses_torch(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch)
    tensor = _FakeTensor(seq=4192, heads=48)
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=48, skip_reshape=True
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
        ("ptl-h", "2.11.0+xpu", _FakeTensor(seq=4191, heads=48), 48),
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


@pytest.mark.parametrize("seq", [4096, 4205])
def test_ptl_auto_boogu_d120_uses_strided_cute(monkeypatch, seq):
    patch, attention, calls = _load_patch(monkeypatch)
    tensor = _FakeTensor(
        seq=seq,
        heads=28,
        dim_head=120,
        dtype=torch.float16,
    )
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=28, skip_reshape=True
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_d120"]
    assert patch.get_stats()["esimd"] == 1
    assert patch.get_stats()["fallback"] == 0


@pytest.mark.parametrize(
    ("target", "torch_version", "backend", "d120_capable", "seq"),
    [
        ("bmg", "2.11.0+xpu", "auto", True, 4096),
        ("ptl-h", "2.10.0+xpu", "auto", True, 4096),
        ("ptl-h", "2.12.0+xpu", "auto", True, 4096),
        ("ptl-h", "2.11.0+xpu", "cute", True, 4096),
        ("ptl-h", "2.11.0+xpu", "auto", False, 4096),
        ("ptl-h", "2.11.0+xpu", "auto", True, 109),
    ],
)
def test_unvalidated_boogu_d120_keeps_torch_fallback(
    monkeypatch, target, torch_version, backend, d120_capable, seq
):
    patch, attention, calls = _load_patch(
        monkeypatch,
        target=target,
        torch_version=torch_version,
        backend=backend,
        d120_capable=d120_capable,
    )
    tensor = _FakeTensor(
        seq=seq,
        heads=28,
        dim_head=120,
        dtype=torch.float16,
    )
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=28, skip_reshape=True
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["fallback"] == 1


def test_boogu_d120_rejects_unvalidated_tensor_contract(monkeypatch):
    _, attention, calls = _load_patch(monkeypatch)
    tensor = _FakeTensor(
        seq=4096,
        heads=28,
        dim_head=120,
        dtype=torch.float16,
        stride=(13762560, 491520, 1, 4096),
    )
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=28, skip_reshape=True
    )
    assert result == "torch-output"
    assert calls == ["torch"]

    calls.clear()
    q = _FakeTensor(
        seq=4096,
        heads=28,
        dim_head=120,
        dtype=torch.float16,
    )
    k = _FakeTensor(
        seq=4096,
        heads=28,
        dim_head=120,
        dtype=torch.bfloat16,
    )
    result = attention.optimized_attention(
        q, k, q, heads=28, skip_reshape=True
    )
    assert result == "torch-output"
    assert calls == ["torch"]
