"""Portable control-flow tests for platform/workflow attention routing."""

from __future__ import annotations

import ast
import functools
import importlib.util
import hashlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"
_ADAPTERS = _PLUGIN / "adapters"


def _pinned_comfy_wrap_attn():
    key = "OMNIXPU_TEST_COMFY_ATTENTION_SOURCE"
    if key not in os.environ:
        pytest.skip(f"set {key} to the pinned Comfy attention.py for this integration check")
    source = Path(os.environ[key])
    if not source.is_file():
        pytest.fail(f"explicit pinned Comfy source is missing: {source}")
    contents = source.read_bytes()
    if hashlib.sha256(contents).hexdigest() != "9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960":
        pytest.fail("explicit Comfy attention.py is not pinned 73c9 source")
    module = ast.parse(contents.decode("utf-8"), filename=str(source))
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == "wrap_attn"]
    if len(functions) != 1:
        pytest.fail("pinned Comfy source does not define exactly one wrap_attn")
    namespace = {
        "functools": functools,
        # This focused fixture calls wrap_attn with plain fake tensors, so its
        # container branch must remain false without importing Comfy runtime.
        "AttentionTensorContainer": type("AttentionTensorContainer", (), {}),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["wrap_attn"]


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
        batch=1,
        non_finite=False,
        storage_offset=0,
    ):
        if pre_shaped:
            self.shape = (batch, heads, seq, dim_head)
            self._stride = stride or (
                heads * seq * dim_head,
                dim_head,
                heads * dim_head,
                1,
            )
        else:
            self.shape = (batch, seq, heads * dim_head)
            self._stride = stride or (seq * heads * dim_head, heads * dim_head, 1)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="xpu")
        self.non_finite = non_finite
        self._storage_offset = storage_offset
        self._non_finite_checks = []

    @classmethod
    def _with_metadata(cls, source, shape, stride):
        tensor = object.__new__(cls)
        tensor.shape = tuple(shape)
        tensor._stride = tuple(stride)
        tensor.dtype = source.dtype
        tensor.device = source.device
        tensor.non_finite = source.non_finite
        tensor._storage_offset = source._storage_offset
        tensor._non_finite_checks = source._non_finite_checks
        return tensor

    @staticmethod
    def _contiguous_stride(shape):
        stride = []
        value = 1
        for size in reversed(shape):
            stride.append(value)
            value *= size
        return tuple(reversed(stride))

    def view(self, *shape):
        shape = list(shape)
        if shape.count(-1) == 1:
            known = 1
            for size in shape:
                if size != -1:
                    known *= size
            numel = 1
            for size in self.shape:
                numel *= size
            shape[shape.index(-1)] = numel // known
        return self._with_metadata(
            self,
            shape,
            self._contiguous_stride(shape),
        )

    def contiguous(self):
        return self._with_metadata(
            self,
            self.shape,
            self._contiguous_stride(self.shape),
        )

    def reshape(self, *shape):
        return self.view(*shape)

    def permute(self, *dims):
        return self._with_metadata(
            self,
            (self.shape[index] for index in dims),
            (self._stride[index] for index in dims),
        )

    def transpose(self, dim0, dim1):
        dims = list(range(len(self.shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("the attention fixture supports batch slices only")
        start, stop, step = key.indices(self.shape[0])
        if step != 1:
            raise ValueError("the attention fixture requires unit batch steps")
        result = self._with_metadata(
            self, (stop - start, *self.shape[1:]), self._stride,
        )
        result._storage_offset += start * self._stride[0]
        return result

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._storage_offset

    def as_strided(self, shape, stride, storage_offset=None):
        tensor = self._with_metadata(self, shape, stride)
        if storage_offset is not None:
            tensor._storage_offset = storage_offset
        return tensor

    def __ne__(self, other):
        self._non_finite_checks.append(True)
        return types.SimpleNamespace(any=lambda: self.non_finite)

    @property
    def non_finite_checks(self):
        return len(self._non_finite_checks)


class _FakeMask:
    def __init__(self, q, kv, *, dtype=torch.bool, contiguous=True, rank=2):
        self.shape = (q, kv) if rank == 2 else (1, 1, q, kv)
        self.ndim = rank
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="xpu")
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def _load_patch(
    monkeypatch,
    *,
    target="ptl-h",
    torch_version="2.11.0+xpu",
    backend="auto",
    platform=None,
    expected_apply=True,
    d120_capable=True,
    d128_bhld_capable=True,
    d128_bhld_error=None,
    h3_vae_d64_capable=True,
    h3_vae_d64_error=None,
    wrap_attn=None,
):
    if backend is None:
        monkeypatch.delenv("OMNI_ATTN_BACKEND", raising=False)
    else:
        monkeypatch.setenv("OMNI_ATTN_BACKEND", backend)
    if platform is not None:
        monkeypatch.setattr(sys, "platform", platform)
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
    attention.wrap_attn = wrap_attn or (lambda fn: fn)
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
        def supports_wan22_cross():
            return True

        @staticmethod
        def sdp_wan22_cross(q, k, v):
            calls.append("cute_wan22_cross")
            return q

        @staticmethod
        def supports_d128_bhld():
            return d128_bhld_capable

        @staticmethod
        def sdp_bhld_d128(q, k, v):
            calls.append("cute_d128_bhld")
            assert len(q.shape) == 4
            assert len(k.shape) == 4
            assert len(v.shape) == 4
            assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
            if d128_bhld_error is not None:
                raise d128_bhld_error
            return q

        @staticmethod
        def supports_minimax_h3_vae_d64():
            return h3_vae_d64_capable

        @staticmethod
        def sdp_minimax_h3_vae_d64(q, k, v):
            calls.append("cute_h3_vae_d64")
            if h3_vae_d64_error is not None:
                raise h3_vae_d64_error
            return q

        @staticmethod
        def supports_d120_bhld():
            return d120_capable

        @staticmethod
        def sdp_bhld_d120(q, k, v):
            calls.append("cute_d120")
            return q

    class Esimd:
        @staticmethod
        def sdp(q, k, v):
            calls.append("esimd")
            return q

    omni = types.ModuleType("omni_xpu_kernel")
    omni.__xpu_target__ = target
    omni.__path__ = []
    omni.cute = Cute

    probe = types.ModuleType("ComfyUI-OmniXPU.probe")
    probe.sdp = Esimd
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
    result = patch.apply()
    if expected_apply:
        assert result == (True, None)
    else:
        assert result[0] is False
    return patch, attention, calls


@pytest.mark.parametrize(
    ("target", "torch_version", "expected"),
    [
        ("ptl-h", "2.10.0+xpu", False),
        ("ptl-h", "2.11.0+xpu", True),
        ("ptl-h", "2.12.0+xpu", True),
        ("ptl-h", "2.13.0+xpu", False),
        ("bmg", "2.10.0+xpu", False),
        ("bmg", "2.11.0+xpu", True),
        ("bmg", "2.12.0+xpu", True),
        ("bmg", "2.13.0+xpu", True),
        ("bmg", "2.14.0+xpu", False),
        ("bmg", "3.0.0+xpu", False),
        ("unknown", "2.13.0+xpu", False),
        ("bmg", "invalid", False),
    ],
)
def test_versioned_routes_use_explicit_target_matrix(
    monkeypatch, target, torch_version, expected
):
    patch, _, _ = _load_patch(
        monkeypatch, target=target, torch_version=torch_version
    )

    assert patch._torch_supports_versioned_routes() is expected


def test_windows_defaults_to_unpatched_torch_sdpa(monkeypatch):
    patch, attention, calls = _load_patch(
        monkeypatch,
        backend=None,
        platform="win32",
        expected_apply=False,
    )
    tensor = _FakeTensor()
    assert attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=30,
        skip_reshape=True,
    ) is None
    assert calls == []
    assert patch.get_stats()["policy"] == "torch"
    assert patch.get_stats()["backend"] == "torch"


def test_non_windows_keeps_auto_as_default(monkeypatch):
    patch, attention, calls = _load_patch(
        monkeypatch,
        backend=None,
        platform="linux",
    )
    tensor = _FakeTensor(seq=4096)
    result = attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=30,
        skip_reshape=True,
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert patch.get_stats()["policy"] == "auto"
    assert patch.get_stats()["esimd"] == 0


@pytest.mark.parametrize("torch_version", ["2.11.0+xpu", "2.12.0+xpu"])
@pytest.mark.parametrize("seq", [64, 1024, 1088])
def test_ptl_auto_validated_torch_zimage_shape_uses_torch(
    monkeypatch, torch_version, seq
):
    patch, attention, calls = _load_patch(
        monkeypatch, torch_version=torch_version
    )
    tensor = _FakeTensor(seq=seq)
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["torch_sdpa"] == 1
    assert patch.get_stats()["fallback"] == 0


@pytest.mark.parametrize("torch_version", ["2.11.0+xpu", "2.12.0+xpu"])
def test_ptl_auto_validated_torch_krea2_shape_uses_torch(
    monkeypatch, torch_version
):
    patch, attention, calls = _load_patch(
        monkeypatch, torch_version=torch_version
    )
    tensor = _FakeTensor(seq=4192, heads=48)
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=48, skip_reshape=True
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["torch_sdpa"] == 1
    assert patch.get_stats()["fallback"] == 0


def test_explicit_cute_does_not_apply_auto_route(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, backend="cute")
    tensor = _FakeTensor()
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["esimd"] == 0


def test_ptl_dispatch_does_not_probe_bmg_capabilities(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="ptl-h")

    def unexpected_bmg_probe(*args, **kwargs):
        raise AssertionError("PTL dispatch reached a BMG-only capability probe")

    monkeypatch.setattr(
        patch,
        "_prepare_bmg_d128_bhld_cute",
        unexpected_bmg_probe,
    )
    monkeypatch.setattr(
        patch,
        "_use_bmg_wan22_cute_cross",
        unexpected_bmg_probe,
    )

    tensor = _FakeTensor(seq=4096)
    result = attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=30,
        skip_reshape=True,
    )

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert patch.get_stats()["cute"] == 1


def test_esimd_is_selected_only_when_explicitly_requested(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, backend="esimd")
    tensor = _FakeTensor()
    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["esimd"]
    assert patch.get_stats()["cute"] == 0
    assert patch.get_stats()["esimd"] == 1


@pytest.mark.parametrize(
    "torch_version", ["2.11.0+xpu", "2.12.0+xpu", "2.13.0+xpu"]
)
def test_bmg_wan22_t2v_turbo_720p_cross_uses_cute(
    monkeypatch, torch_version
):
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
        torch_version=torch_version,
    )
    q = _FakeTensor(
        seq=75600,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=512,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )
    result = attention.optimized_attention(
        q,
        kv,
        kv,
        heads=40,
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_wan22_cross"]
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {
        "wan22_t2v_turbo_720p_cross": 1
    }


def test_bmg_wan22_cute_skips_output_scan_by_default(monkeypatch):
    monkeypatch.delenv("OMNIXPU_VALIDATE_ATTENTION_OUTPUT", raising=False)
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(
        seq=75600,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
        non_finite=True,
    )
    kv = _FakeTensor(
        seq=512,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )

    result = attention.optimized_attention(q, kv, kv, heads=40)

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_wan22_cross"]
    assert q.non_finite_checks == 0
    assert patch.get_stats()["fallback"] == 0


def test_bmg_wan22_diagnostic_output_scan_falls_back(monkeypatch):
    monkeypatch.setenv("OMNIXPU_VALIDATE_ATTENTION_OUTPUT", "1")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(
        seq=75600,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
        non_finite=True,
    )
    kv = _FakeTensor(
        seq=512,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )

    result = attention.optimized_attention(q, kv, kv, heads=40)

    assert result == "torch-output"
    assert calls == ["cute_wan22_cross", "torch"]
    assert q.non_finite_checks == 1
    assert patch.get_stats()["reasons"] == {"output_non_finite": 1}


def test_generic_cute_skips_output_scan_by_default(monkeypatch):
    monkeypatch.delenv("OMNIXPU_VALIDATE_ATTENTION_OUTPUT", raising=False)
    patch, attention, calls = _load_patch(monkeypatch, backend="cute")
    tensor = _FakeTensor(dtype=torch.float16, non_finite=True)

    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert tensor.non_finite_checks == 0
    assert patch.get_stats()["fallback"] == 0


def test_explicit_esimd_non_finite_output_still_falls_back(monkeypatch):
    monkeypatch.delenv("OMNIXPU_VALIDATE_ATTENTION_OUTPUT", raising=False)
    patch, attention, calls = _load_patch(monkeypatch, backend="esimd")
    tensor = _FakeTensor(dtype=torch.float16, non_finite=True)

    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=30, skip_reshape=True
    )

    assert result == "torch-output"
    assert calls == ["esimd", "torch"]
    assert tensor.non_finite_checks == 1
    assert patch.get_stats()["reasons"] == {"output_non_finite": 1}


@pytest.mark.parametrize(
    ("dtype", "q_len", "kv_len", "route"),
    [
        (
            torch.float16,
            44550,
            512,
            "animate2_b1_fp16_d128_q44550_kv512_cross",
        ),
        (
            torch.bfloat16,
            2025,
            44550,
            "animate2_b1_bf16_d128_q2025_kv44550_cross",
        ),
        (
            torch.bfloat16,
            75600,
            512,
            "animate2_b1_bf16_d128_q75600_kv512_cross",
        ),
    ],
)
def test_bmg_animate2_cross_uses_cute_by_default(
    monkeypatch, dtype, q_len, kv_len, route
):
    monkeypatch.delenv("OMNIXPU_ANIMATE2_CROSS", raising=False)
    monkeypatch.delenv("OMNIXPU_ANIMATE2_SHAPES", raising=False)
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(
        seq=q_len,
        heads=40,
        dtype=dtype,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=kv_len,
        heads=40,
        dtype=dtype,
        pre_shaped=False,
    )

    result = attention.optimized_attention(q, kv, kv, heads=40)

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_d128_bhld"]
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {route: 1}


def test_bmg_animate2_cross_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNIXPU_ANIMATE2_CROSS", "0")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(
        seq=44550,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=512,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )

    result = attention.optimized_attention(q, kv, kv, heads=40)

    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["routes"] == {}


def test_bmg_animate2_shape_allowlist_can_bisect_routes(monkeypatch):
    monkeypatch.delenv("OMNIXPU_ANIMATE2_CROSS", raising=False)
    monkeypatch.setenv("OMNIXPU_ANIMATE2_SHAPES", "2025:*,*:257")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(
        seq=44550,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=512,
        heads=40,
        dtype=torch.float16,
        pre_shaped=False,
    )

    result = attention.optimized_attention(q, kv, kv, heads=40)

    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["routes"] == {}


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "kv_len", "dtype", "expected"),
    [
        (1, 40, 256, 128, torch.float16, True),
        (2, 40, 256, 128, torch.float16, False),
        (1, 32, 256, 128, torch.float16, False),
        (1, 40, 256, 256, torch.float16, False),
        (1, 40, 255, 128, torch.float16, False),
        (1, 40, 256, 127, torch.float16, False),
        (1, 40, 256, 128, torch.float32, False),
        (1, 40, 75600, 512, torch.float16, False),
        (1, 40, 75600, 512, torch.bfloat16, True),
    ],
)
def test_animate2_cute_shape_contract(
    monkeypatch, batch, heads, q_len, kv_len, dtype, expected
):
    monkeypatch.delenv("OMNIXPU_ANIMATE2_CROSS", raising=False)
    monkeypatch.delenv("OMNIXPU_ANIMATE2_SHAPES", raising=False)
    patch, _, _ = _load_patch(monkeypatch, target="bmg")

    assert patch._is_animate2_cute_shape(
        batch,
        heads,
        128,
        q_len,
        kv_len,
        dtype,
    ) is expected


@pytest.mark.parametrize(
    ("q_len", "kv_len", "pre_shaped", "route"),
    [
        (14080, 1024, False, "bmg_b1_bf16_d128_kv1024_cross"),
        (768, 768, False, "bmg_b2_bf16_d128_self"),
        (769, 769, False, "bmg_b2_bf16_d128_self"),
        (1024, 1024, False, "bmg_b2_bf16_d128_self"),
        (3520, 3520, False, "bmg_b2_bf16_d128_self"),
        (7041, 7041, False, "bmg_b2_bf16_d128_self"),
        (14080, 14080, False, "bmg_b2_bf16_d128_self"),
        (28160, 28160, True, "bmg_b2_bf16_d128_self"),
        (1025, 1024, False, "bmg_b2_bf16_d128_kv1024_cross"),
        (2049, 1024, False, "bmg_b2_bf16_d128_kv1024_cross"),
        (3520, 1024, False, "bmg_b2_bf16_d128_kv1024_cross"),
        (14080, 1024, False, "bmg_b2_bf16_d128_kv1024_cross"),
    ],
)
@pytest.mark.parametrize(
    "torch_version", ["2.11.0+xpu", "2.12.0+xpu", "2.13.0+xpu"]
)
def test_bmg_attention_open_ended_domain_uses_general_bhld_cute(
    monkeypatch, q_len, kv_len, pre_shaped, route, torch_version
):
    batch = 1 if route.startswith("bmg_b1_") else 2
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
        torch_version=torch_version,
    )
    q = _FakeTensor(
        seq=q_len,
        heads=32,
        batch=batch,
        pre_shaped=pre_shaped,
    )
    kv = _FakeTensor(
        seq=kv_len,
        heads=32,
        batch=batch,
        pre_shaped=pre_shaped,
    )
    result = attention.optimized_attention(
        q,
        kv,
        kv,
        heads=32,
        skip_reshape=pre_shaped,
    )
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_d128_bhld"]
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {route: 1}


def test_bmg_b1_self_keeps_legacy_cute_route(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    tensor = _FakeTensor(seq=14080, heads=32, batch=1)

    result = attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=32,
        skip_reshape=True,
    )

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["routes"] == {}


@pytest.mark.parametrize(
    ("q_len", "kv_len", "pre_shaped"),
    [
        (2048, 2049, False),
        (2049, 4099, True),
        (4032, 4040, False),
        (4032, 8078, True),
        (4032, 12188, False),
        (4097, 8222, True),
        (2048, (2 ** 31 - 1) // 4096, True),
    ],
)
def test_bmg_b1_dense_prefix_rectangular_uses_cute(
    monkeypatch, q_len, kv_len, pre_shaped
):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=q_len, heads=32, pre_shaped=pre_shaped)
    kv = _FakeTensor(seq=kv_len, heads=32, pre_shaped=pre_shaped)

    result = attention.optimized_attention(
        q, kv, kv, heads=32, skip_reshape=pre_shaped
    )

    assert isinstance(result, _FakeTensor)
    assert result.shape == (1, q_len, 32 * 128)
    assert calls == ["cute_d128_bhld"]
    assert patch.get_stats()["routes"] == {
        "bmg_b1_bf16_d128_prefix_rectangular": 1
    }


def test_bmg_b1_dense_prefix_rectangular_keeps_bhld_output_request(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=2049, heads=32)
    kv = _FakeTensor(seq=6150, heads=32)

    result = attention.optimized_attention(
        q, kv, kv, heads=32, skip_reshape=True, skip_output_reshape=True
    )

    assert result.shape == (1, 32, 2049, 128)
    assert calls == ["cute_d128_bhld"]
    assert patch.get_stats()["routes"] == {
        "bmg_b1_bf16_d128_prefix_rectangular": 1
    }


def test_bmg_b1_prefix_segment_normalizes_unused_batch_stride(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(
        seq=2049, heads=32, pre_shaped=False,
        stride=(2200 * 4096, 4096, 1), storage_offset=151 * 4096,
    )
    kv = _FakeTensor(seq=2050, heads=32, pre_shaped=False)
    prepared = []

    def observe(q_bhld, k_bhld, v_bhld):
        prepared.append((q_bhld, k_bhld, v_bhld))
        return q_bhld

    monkeypatch.setattr(patch._backend_sdp, "sdp_bhld_d128", observe)
    result = attention.optimized_attention(q, kv, kv, heads=32)

    assert result.shape == (1, 2049, 4096)
    assert calls == []
    assert len(prepared) == 1
    q_bhld, k_bhld, v_bhld = prepared[0]
    assert q_bhld.stride() == (2049 * 4096, 128, 4096, 1)
    assert q_bhld.storage_offset() == q.storage_offset()
    assert k_bhld.stride() == v_bhld.stride() == (2050 * 4096, 128, 4096, 1)
    assert patch.get_stats()["routes"] == {
        "bmg_b1_bf16_d128_prefix_rectangular": 1
    }


def test_bmg_b1_prefix_segment_cpu_tensor_view_has_no_copy(monkeypatch):
    patch, _, _ = _load_patch(monkeypatch, target="bmg")
    width = 32 * 128
    backing = torch.arange(23 * width, dtype=torch.float32).view(1, 23, width)
    segment = backing[:, 7:17]
    assert segment.stride(0) == 23 * width
    assert segment.storage_offset() == 7 * width

    for source, pre_shaped in (
        (segment, False),
        (segment.view(1, 10, 32, 128).transpose(1, 2), True),
    ):
        view = patch._b1_dense_segment_bhld(source, 10, 32, 128, pre_shaped)
        expected = segment.view(1, 10, 32, 128).transpose(1, 2)
        assert view.shape == (1, 32, 10, 128)
        assert view.stride() == (10 * width, 128, width, 1)
        assert view.storage_offset() == source.storage_offset()
        assert view.data_ptr() == source.data_ptr()
        assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert torch.equal(view, expected)

    assert patch._b1_dense_segment_bhld(segment[:, ::2], 5, 32, 128, False) is None
    packed = torch.empty(1, 32, 23, 128)[:, :, 7:17]
    assert patch._b1_dense_segment_bhld(packed, 10, 32, 128, True) is None


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU unavailable")
def test_bmg_b1_prefix_segment_xpu_dispatch_preserves_source_storage(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    width = 32 * 128
    backing = torch.randn((1, 2055, width), device="xpu", dtype=torch.bfloat16)
    q = backing[:, 6:2055]
    k = torch.randn((1, 2057, width), device="xpu", dtype=torch.bfloat16)[:, 7:]
    v = torch.randn((1, 2059, width), device="xpu", dtype=torch.bfloat16)[:, 9:]
    prepared = []

    def observe(q_bhld, k_bhld, v_bhld):
        prepared.append((q_bhld, k_bhld, v_bhld))
        return q_bhld

    monkeypatch.setattr(patch._backend_sdp, "sdp_bhld_d128", observe)
    out = attention.optimized_attention(q, k, v, heads=32)

    assert calls == []
    assert len(prepared) == 1
    for source, tensor in zip((q, k, v), prepared[0]):
        assert tensor.data_ptr() == source.data_ptr()
        assert tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert tensor.storage_offset() == source.storage_offset()
    assert prepared[0][0].stride() == (2049 * width, 128, width, 1)
    assert out.shape == (1, 2049, width)
    assert out.data_ptr() == q.data_ptr()
    assert patch.get_stats()["routes"] == {
        "bmg_b1_bf16_d128_prefix_rectangular": 1
    }


@pytest.mark.parametrize(
    ("q_len", "kv_len", "batch", "kwargs"),
    [
        (1024, 2048, 1, {}),
        (1025, 2050, 1, {}),
        (2047, 4095, 1, {}),
        (1024, 1024, 1, {"mask": _FakeTensor(seq=1024, heads=32)}),
        (2048, 4096, 1, {"mask": _FakeTensor(seq=2048, heads=32)}),
        (2048, 4096, 1, {"attn_precision": "fp32"}),
        (2048, 4096, 1, {"dropout_p": 0.1}),
        (2048, 4096, 1, {"is_causal": True}),
        (2048, 4096, 1, {"enable_gqa": True}),
        (2048, 4096, 1, {"scale": 0.5}),
        (4032, 12188, 2, {}),
        (2048, (2 ** 31 - 1) // 4096 + 1, 1, {}),
    ],
)
def test_bmg_b1_prefix_rectangular_unsupported_semantics_keep_fallback(
    monkeypatch, q_len, kv_len, batch, kwargs
):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=q_len, heads=32, batch=batch)
    kv = _FakeTensor(seq=kv_len, heads=32, batch=batch)

    result = attention.optimized_attention(
        q, kv, kv, heads=32, skip_reshape=True, **kwargs
    )

    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["routes"] == {}


def test_bmg_b1_prefix_rectangular_requires_dense_layout_and_no_grad(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=2049, heads=32)
    kv = _FakeTensor(seq=8193, heads=32)
    q.requires_grad = True

    assert attention.optimized_attention(
        q, kv, kv, heads=32, skip_reshape=True
    ) == "torch-output"

    q.requires_grad = False
    q._stride = (2049 * 4096, 128, 8192, 1)
    assert attention.optimized_attention(
        q, kv, kv, heads=32, skip_reshape=True
    ) == "torch-output"

    assert calls == ["torch", "torch"]
    assert patch.get_stats()["routes"] == {}


@pytest.mark.parametrize("variant", ["1", "4"])
@pytest.mark.parametrize(
    ("q_len", "kv_len"),
    [(335, 335), (8, 8), (38, 4078), (6, 4046),
     (78, 8156), (6, 8084), (75, 10659)],
)
def test_bmg_masked_d128_opt_in_uses_actual_mask_route(
    monkeypatch, variant, q_len, kv_len,
):
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/sidecar.so")
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "a" * 64)
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", variant)
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=q_len, heads=32, pre_shaped=False)
    kv = _FakeTensor(seq=kv_len, heads=32, pre_shaped=False)
    mask = _FakeMask(q_len, kv_len)
    prepared = []

    def sidecar(values, actual_mask):
        prepared.append((values, actual_mask))
        return values[1]

    monkeypatch.setattr(patch, "_run_experimental_masked_d128", sidecar)
    result = attention.optimized_attention(q, kv, kv, heads=32, mask=mask)
    assert result.shape == (1, q_len, 4096)
    assert calls == [] and len(prepared) == 1
    assert prepared[0][0][0] == variant and prepared[0][1] is mask
    assert patch.get_stats()["routes"] == {
        f"bmg_b1_bf16_d128_masked_split{variant}": 1
    }


@pytest.mark.parametrize("variant", ["1", "4", "direct"])
def test_bmg_masked_d128_accepts_pinned_comfy_wrap_marker_only(monkeypatch, variant):
    wrap_attn = _pinned_comfy_wrap_attn()
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/sidecar.so")
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "a" * 64)
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", variant)
    patch, attention, calls = _load_patch(
        monkeypatch, target="bmg", wrap_attn=wrap_attn,
    )
    q = _FakeTensor(seq=335, heads=32, pre_shaped=False)
    kv = _FakeTensor(seq=335, heads=32, pre_shaped=False)
    mask = _FakeMask(335, 335)
    selected = []
    monkeypatch.setattr(patch, "_run_experimental_masked_d128",
                        lambda prepared, actual_mask: selected.append(prepared) or prepared[1])
    result = attention.optimized_attention(q, kv, kv, heads=32, mask=mask)
    assert result.shape == (1, 335, 4096)
    assert len(selected) == 1 and selected[0][0] == variant and calls == []
    route = ("bmg_b1_bf16_d128_masked_direct" if variant == "direct"
             else f"bmg_b1_bf16_d128_masked_split{variant}")
    assert patch.get_stats()["routes"] == {route: 1}

    # A caller-supplied false marker is not a trusted wrapper marker; the
    # actual Comfy wrapper only inserts True when the key is absent.
    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=mask, _inside_attn_wrapper=False,
    ) == "torch-output"
    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=mask, unknown_semantics=True,
    ) == "torch-output"
    assert len(selected) == 1 and calls == ["torch", "torch"]


def test_bmg_masked_d128_direct_is_explicit_and_keeps_other_masks_on_torch(monkeypatch):
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/direct-sidecar.so")
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "b" * 64)
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", "direct")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=335, heads=32, pre_shaped=False)
    kv = _FakeTensor(seq=335, heads=32, pre_shaped=False)
    mask = _FakeMask(335, 335)
    selected = []
    monkeypatch.setattr(patch, "_run_experimental_masked_d128",
                        lambda prepared, actual_mask: selected.append(prepared) or prepared[1])
    result = attention.optimized_attention(q, kv, kv, heads=32, mask=mask)
    assert result.shape == (1, 335, 4096)
    assert len(selected) == 1 and selected[0][0] == "direct"
    assert patch.get_stats()["routes"] == {"bmg_b1_bf16_d128_masked_direct": 1}
    assert calls == []

    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=_FakeMask(335, 335, dtype=torch.float32),
    ) == "torch-output"
    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=mask, scale=0.1,
    ) == "torch-output"
    assert len(selected) == 1 and calls == ["torch", "torch"]


def test_bmg_masked_d128_opt_in_preserves_segment_offset_and_bhld_output(monkeypatch):
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/sidecar.so")
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "a" * 64)
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", "4")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=75, heads=32, pre_shaped=False,
                    stride=(11000 * 4096, 4096, 1), storage_offset=10584 * 4096)
    kv = _FakeTensor(seq=10659, heads=32, pre_shaped=False)
    seen = []
    monkeypatch.setattr(patch, "_run_experimental_masked_d128",
                        lambda values, mask: seen.append(values) or values[1])
    result = attention.optimized_attention(
        q, kv, kv, heads=32, mask=_FakeMask(75, 10659),
        skip_output_reshape=True,
    )
    assert result.shape == (1, 32, 75, 128) and calls == []
    assert seen[0][1].storage_offset() == q.storage_offset()
    assert seen[0][1].stride() == (75 * 4096, 128, 4096, 1)


@pytest.mark.parametrize("bad_mask", [
    {"dtype": torch.float32}, {"contiguous": False}, {"rank": 4},
    {"q": 74}, {"kv": 10658},
])
def test_bmg_masked_d128_nonbool_or_broadcast_mask_keeps_torch(
    monkeypatch, bad_mask,
):
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/sidecar.so")
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "a" * 64)
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", "4")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=75, heads=32)
    kv = _FakeTensor(seq=10659, heads=32)
    mask = _FakeMask(bad_mask.get("q", 75), bad_mask.get("kv", 10659),
                     **{k: v for k, v in bad_mask.items() if k not in ("q", "kv")})
    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=mask, skip_reshape=True,
    ) == "torch-output"
    assert calls == ["torch"] and patch.get_stats()["routes"] == {}


def test_bmg_masked_d128_missing_opt_in_and_other_semantics_keep_torch(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=75, heads=32)
    kv = _FakeTensor(seq=10659, heads=32)
    mask = _FakeMask(75, 10659)
    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=mask, skip_reshape=True,
    ) == "torch-output"
    for key, value in (("scale", 0.5), ("dropout_p", 0.1),
                       ("is_causal", True), ("enable_gqa", True)):
        monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/sidecar.so")
        monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "a" * 64)
        monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", "1")
        assert attention.optimized_attention(
            q, kv, kv, heads=32, mask=mask, skip_reshape=True, **{key: value},
        ) == "torch-output"
    assert calls == ["torch"] * 5 and patch.get_stats()["routes"] == {}


def test_bmg_masked_d128_grad_and_mask_address_limit_fall_back(monkeypatch):
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_DSO", "/new/sidecar.so")
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_SHA256", "a" * 64)
    monkeypatch.setenv("OMNIXPU_EXPERIMENTAL_MASKED_D128_PARTITIONS", "1")
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    q = _FakeTensor(seq=75, heads=32)
    kv = _FakeTensor(seq=10659, heads=32)
    q.requires_grad = True
    assert attention.optimized_attention(
        q, kv, kv, heads=32, mask=_FakeMask(75, 10659), skip_reshape=True,
    ) == "torch-output"
    q.requires_grad = False
    large_q = _FakeTensor(seq=40000, heads=32)
    large_kv = _FakeTensor(seq=60000, heads=32)
    assert attention.optimized_attention(
        large_q, large_kv, large_kv, heads=32,
        mask=_FakeMask(40000, 60000), skip_reshape=True,
    ) == "torch-output"
    assert calls == ["torch", "torch"] and patch.get_stats()["routes"] == {}


@pytest.mark.parametrize(
    ("seq", "qk_stride", "v_stride"),
    [
        (31, (7168, 128, 21504, 1), (7168, 128, 21504, 1)),
        (255, (7168, 128, 21504, 1), (7168, 128, 21504, 1)),
        (388, (7168, 128, 7168, 1), (7168, 128, 21504, 1)),
        (1025, (7168, 128, 21504, 1), (7168, 128, 21504, 1)),
        (15787, (7168, 128, 21504, 1), (7168, 128, 21504, 1)),
    ],
)
@pytest.mark.parametrize(
    "torch_version", ["2.11.0+xpu", "2.12.0+xpu", "2.13.0+xpu"]
)
def test_bmg_minimax_h3_h56_uses_direct_qkv_bhld_cute(
    monkeypatch, seq, qk_stride, v_stride, torch_version
):
    patch, attention, calls = _load_patch(
        monkeypatch, target="bmg", torch_version=torch_version
    )
    q = _FakeTensor(seq=seq, heads=56, stride=qk_stride)
    k = _FakeTensor(seq=seq, heads=56, stride=qk_stride)
    v = _FakeTensor(seq=seq, heads=56, stride=v_stride)

    result = attention.optimized_attention(
        q,
        k,
        v,
        heads=56,
        skip_reshape=True,
    )

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_d128_bhld"]
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {
        "minimax_h3_h56_bf16_d128_qkv_bhld": 1
    }


@pytest.mark.parametrize(
    ("seq", "stride"),
    [
        (30, (7168, 128, 21504, 1)),
        (15787, (7168, 128, 7169, 1)),
    ],
)
def test_bmg_minimax_h3_h56_rejects_unvalidated_contract(
    monkeypatch, seq, stride
):
    _, attention, calls = _load_patch(monkeypatch, target="bmg")
    tensor = _FakeTensor(seq=seq, heads=56, stride=stride)

    result = attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=56,
        skip_reshape=True,
    )

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute"]


@pytest.mark.parametrize(
    ("target", "torch_version", "capable", "q_len", "kv_len"),
    [
        ("ptl-h", "2.11.0+xpu", True, 3520, 3520),
        ("bmg", "2.10.0+xpu", True, 3520, 3520),
        ("bmg", "2.14.0+xpu", True, 3520, 3520),
        ("bmg", "2.11.0+xpu", False, 3520, 3520),
        ("bmg", "2.11.0+xpu", True, 767, 767),
        ("bmg", "2.11.0+xpu", True, 1023, 1024),
        ("bmg", "2.11.0+xpu", True, 3520, 1023),
    ],
)
def test_unqualified_b2_bhld_attention_keeps_torch(
    monkeypatch, target, torch_version, capable, q_len, kv_len
):
    patch, attention, calls = _load_patch(
        monkeypatch,
        target=target,
        torch_version=torch_version,
        d128_bhld_capable=capable,
    )
    q = _FakeTensor(seq=q_len, heads=32, batch=2)
    kv = _FakeTensor(seq=kv_len, heads=32, batch=2)
    result = attention.optimized_attention(
        q,
        kv,
        kv,
        heads=32,
        skip_reshape=True,
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["cute"] == 0
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["fallback"] == 1


def test_bmg_b2_auto_rejects_non_dense_bhld_layout(monkeypatch):
    patch, attention, calls = _load_patch(monkeypatch, target="bmg")
    seq = 3520
    heads = 32
    dim_head = 128
    tensor = _FakeTensor(
        seq=seq,
        heads=heads,
        batch=2,
        stride=(13762560, 491520, 1, 4096),
    )
    result = attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=heads,
        skip_reshape=True,
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["routes"] == {}
    assert patch.get_stats()["fallback"] == 1


def test_bmg_b2_runtime_error_warns_falls_back_and_quarantines(
    monkeypatch, caplog
):
    error = RuntimeError("synthetic CUTE failure")
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
        d128_bhld_error=error,
    )
    q = _FakeTensor(
        seq=3520,
        heads=32,
        batch=2,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=1024,
        heads=32,
        batch=2,
        pre_shaped=False,
    )

    first = attention.optimized_attention(q, kv, kv, heads=32)
    second = attention.optimized_attention(q, kv, kv, heads=32)

    assert first == "torch-output"
    assert second == "torch-output"
    assert calls == ["cute_d128_bhld", "torch", "torch"]
    assert patch.get_stats()["quarantined_contracts"] == 1
    assert patch.get_stats()["reasons"]["cute_runtime_error"] == 1
    assert "OMNI_ATTN_BACKEND=torch" in caplog.text
    assert "synthetic CUTE failure" in caplog.text


def test_bmg_b2_out_of_memory_is_not_retried_or_quarantined(monkeypatch):
    error = torch.OutOfMemoryError("synthetic XPU out of memory")
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
        d128_bhld_error=error,
    )
    q = _FakeTensor(
        seq=3520,
        heads=32,
        batch=2,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=1024,
        heads=32,
        batch=2,
        pre_shaped=False,
    )

    with pytest.raises(torch.OutOfMemoryError, match="synthetic XPU"):
        attention.optimized_attention(q, kv, kv, heads=32)

    assert calls == ["cute_d128_bhld"]
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["quarantined_contracts"] == 0


def test_auto_unsupported_cross_uses_torch_not_esimd(monkeypatch):
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
    )
    q = _FakeTensor(
        seq=4096,
        heads=39,
        dtype=torch.float16,
        pre_shaped=False,
    )
    kv = _FakeTensor(
        seq=512,
        heads=39,
        dtype=torch.float16,
        pre_shaped=False,
    )
    result = attention.optimized_attention(
        q,
        kv,
        kv,
        heads=39,
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["cute"] == 0
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["fallback"] == 1


def test_auto_d64_uses_torch_not_esimd(monkeypatch):
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
    )
    tensor = _FakeTensor(
        seq=4096,
        heads=40,
        dim_head=64,
        dtype=torch.float16,
        pre_shaped=False,
    )
    result = attention.optimized_attention(
        tensor,
        tensor,
        tensor,
        heads=40,
    )
    assert result == "torch-output"
    assert calls == ["torch"]
    assert patch.get_stats()["cute"] == 0
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["fallback"] == 1


@pytest.mark.parametrize(
    "torch_version", ["2.11.0+xpu", "2.12.0+xpu", "2.13.0+xpu"]
)
@pytest.mark.parametrize("seq", [6, 12, 261, 453, 901, 1797])
def test_bmg_minimax_h3_video_vae_d64_uses_structural_cute(
    monkeypatch, torch_version, seq
):
    patch, attention, calls = _load_patch(
        monkeypatch, target="bmg", torch_version=torch_version
    )
    q = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 2048, 64, 2048, 1),
    )
    k = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 2048, 64, 2048, 1),
    )
    v = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 6144, 192, 6144, 1),
    )

    result = attention.optimized_attention(
        q, k, v, heads=32, skip_reshape=True
    )

    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_h3_vae_d64"]
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {
        "minimax_h3_video_vae_fp16_d64": 1
    }


@pytest.mark.parametrize("batch", [2, 3, 4])
def test_bmg_minimax_h3_video_vae_d64_batched_packed_qkv(monkeypatch, batch):
    patch, attention, calls = _load_patch(
        monkeypatch, target="bmg", torch_version="2.13.0+xpu",
    )
    seq = 1797
    packed_stride = (seq * 6144, 192, 6144, 1)
    q = _FakeTensor(
        batch=batch, seq=seq, heads=32, dim_head=64,
        dtype=torch.float16, stride=packed_stride,
    )
    k = _FakeTensor(
        batch=batch, seq=seq, heads=32, dim_head=64,
        dtype=torch.float16, stride=packed_stride,
    )
    v = _FakeTensor(
        batch=batch, seq=seq, heads=32, dim_head=64,
        dtype=torch.float16, stride=packed_stride,
    )

    def cat_batches(values, dim=0):
        assert dim == 0 and len(values) == batch
        shape = (sum(value.shape[0] for value in values), *values[0].shape[1:])
        return _FakeTensor._with_metadata(
            values[0], shape, _FakeTensor._contiguous_stride(shape),
        )

    monkeypatch.setattr(torch, "cat", cat_batches)
    result = attention.optimized_attention(q, k, v, heads=32, skip_reshape=True)
    assert result.shape == (batch, seq, 2048)
    assert calls == ["cute_h3_vae_d64"] * batch
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {
        "minimax_h3_video_vae_fp16_d64": 1,
    }


def test_bmg_minimax_h3_video_vae_d64_packed_qk_materialization(monkeypatch):
    patch, _, _ = _load_patch(monkeypatch, target="bmg")
    seq = 261
    qkv = torch.randn(4, seq, 32, 192)
    q, k, v = (part.transpose(1, 2) for part in qkv.chunk(3, dim=-1))
    assert patch._is_minimax_h3_vae_d64_bhld(q, seq, 4, value=False)
    assert patch._is_minimax_h3_vae_d64_bhld(k, seq, 4, value=False)
    assert patch._is_minimax_h3_vae_d64_bhld(v, seq, 4, value=True)
    prepared = patch._minimax_h3_vae_d64_qk_for_native(q[:1])
    assert prepared.stride() == (seq * 2048, 64, 2048, 1)
    assert torch.equal(prepared, q[:1])
    assert not torch._C._is_alias_of(prepared, q)


def test_bmg_minimax_h3_video_vae_d64_rejects_wrong_v_stride(monkeypatch):
    _, attention, calls = _load_patch(monkeypatch, target="bmg")
    tensor = _FakeTensor(
        seq=1797,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(3680256, 64, 2048, 1),
    )

    result = attention.optimized_attention(
        tensor, tensor, tensor, heads=32, skip_reshape=True
    )

    assert result == "torch-output"
    assert calls == ["torch"]


def test_bmg_minimax_h3_video_vae_d64_rejects_non_tile_sequence(
    monkeypatch,
):
    _, attention, calls = _load_patch(monkeypatch, target="bmg")
    seq = 5
    q = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 2048, 64, 2048, 1),
    )
    v = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 6144, 192, 6144, 1),
    )

    result = attention.optimized_attention(
        q, q, v, heads=32, skip_reshape=True
    )

    assert result == "torch-output"
    assert calls == ["torch"]


def test_bmg_minimax_h3_video_vae_d64_quarantines_runtime_failure(
    monkeypatch,
):
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
        h3_vae_d64_error=RuntimeError("candidate failed"),
    )
    seq = 453
    q = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 2048, 64, 2048, 1),
    )
    v = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 6144, 192, 6144, 1),
    )

    for _ in range(2):
        result = attention.optimized_attention(
            q, q, v, heads=32, skip_reshape=True
        )
        assert result == "torch-output"

    assert calls == ["cute_h3_vae_d64", "torch", "torch"]
    assert patch.get_stats()["fallback"] == 2


def test_bmg_minimax_h3_video_vae_d64_device_oom_is_not_retried(
    monkeypatch,
):
    error = RuntimeError("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY")
    patch, attention, calls = _load_patch(
        monkeypatch,
        target="bmg",
        h3_vae_d64_error=error,
    )
    seq = 453
    q = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 2048, 64, 2048, 1),
    )
    v = _FakeTensor(
        seq=seq,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
        stride=(seq * 6144, 192, 6144, 1),
    )

    with pytest.raises(RuntimeError, match="OUT_OF_DEVICE_MEMORY"):
        attention.optimized_attention(
            q, q, v, heads=32, skip_reshape=True
        )

    assert calls == ["cute_h3_vae_d64"]
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["quarantined_contracts"] == 0


def test_attention_contract_trace_records_exact_layout_once(
    monkeypatch, caplog
):
    caplog.set_level("INFO", logger="ComfyUI-OmniXPU")
    monkeypatch.setenv("OMNI_ATTN_TRACE_CONTRACTS", "1")
    _, attention, calls = _load_patch(monkeypatch, target="bmg")
    tensor = _FakeTensor(
        seq=1797,
        heads=32,
        dim_head=64,
        dtype=torch.float16,
    )

    for _ in range(2):
        result = attention.optimized_attention(
            tensor,
            tensor,
            tensor,
            heads=32,
            skip_reshape=True,
        )
        assert result == "torch-output"

    assert calls == ["torch", "torch"]
    assert caplog.text.count("[OmniXPU] attention contract:") == 1
    assert "heads=32" in caplog.text
    assert "q=(1, 32, 1797, 64)/(3680256, 64, 2048, 1)/blhd_backed" in caplog.text


@pytest.mark.parametrize(
    ("target", "torch_version", "tensor", "heads"),
    [
        ("bmg", "2.11.0+xpu", _FakeTensor(), 30),
        ("ptl-h", "2.10.0+xpu", _FakeTensor(), 30),
        ("ptl-h", "2.13.0+xpu", _FakeTensor(), 30),
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
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["esimd"] == 0
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
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["torch_sdpa"] == 0


@pytest.mark.parametrize(
    ("target", "torch_version"),
    [
        ("ptl-h", "2.11.0+xpu"),
        ("ptl-h", "2.12.0+xpu"),
        ("bmg", "2.11.0+xpu"),
        ("bmg", "2.12.0+xpu"),
        ("bmg", "2.13.0+xpu"),
    ],
)
@pytest.mark.parametrize("seq", [4096, 4205])
def test_validated_auto_boogu_d120_uses_strided_cute(
    monkeypatch, target, torch_version, seq
):
    patch, attention, calls = _load_patch(
        monkeypatch, target=target, torch_version=torch_version
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
    assert isinstance(result, _FakeTensor)
    assert calls == ["cute_d120"]
    assert patch.get_stats()["cute"] == 1
    assert patch.get_stats()["esimd"] == 0
    assert patch.get_stats()["fallback"] == 0
    assert patch.get_stats()["routes"] == {"boogu_cute_d120_bhld": 1}


@pytest.mark.parametrize(
    ("target", "torch_version", "backend", "d120_capable", "seq"),
    [
        ("ptl-h", "2.10.0+xpu", "auto", True, 4096),
        ("ptl-h", "2.13.0+xpu", "auto", True, 4096),
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
