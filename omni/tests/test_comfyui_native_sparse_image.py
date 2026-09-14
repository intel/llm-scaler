"""Host fixtures for native sparse image admission; no XPU execution."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installed(monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "tools" / "validate_comfyui_image.py"
    spec = importlib.util.spec_from_file_location("native_sparse_image_test", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    monkeypatch.setattr(validator, "COMFYUI_ROOT", tmp_path / "ComfyUI")
    package = tmp_path / "cute"
    package.mkdir()
    library = package / "cute_fmha_torch.so"
    library.write_bytes(b"native package fixture")
    native = SimpleNamespace(is_available=lambda: True)
    cute = SimpleNamespace(
        __file__=str(package / "__init__.py"),
        _find_extension=lambda: str(library), sol_attn_v2=native,
    )
    kitchen = SimpleNamespace(sol_attn_is_available=lambda device: device == "xpu")
    upstream = SimpleNamespace(
        __file__=str(validator.COMFYUI_ROOT / "comfy_extras/nodes_sparse_attention.py"),
        BlockSparseAttention=SimpleNamespace(
            define_schema=lambda: SimpleNamespace(node_id="BlockSparseAttention")
        ),
    )
    adapter = SimpleNamespace(apply=lambda: (True, ""), _upstream_module=lambda: upstream)
    adapter_path = tmp_path / "sparse_attention.py"
    adapter_path.write_text("from _test_sparse_adapter import apply, _upstream_module\n")
    for name, module in (
        ("torch", SimpleNamespace(device=lambda name: name)),
        ("comfy_kitchen", kitchen),
        ("omni_xpu_kernel", SimpleNamespace(cute=cute)),
        ("omni_xpu_kernel.cute", cute),
        ("_test_sparse_adapter", adapter),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for switch in ("OMNIXPU_ENABLE", "OMNIXPU_SPARSE_ATTENTION", "SOL_ATTN_XPU_EXPERIMENTAL"):
        monkeypatch.delenv(switch, raising=False)
    return SimpleNamespace(
        validator=validator, native=native, cute=cute, kitchen=kitchen,
        upstream=upstream, adapter=adapter, adapter_path=adapter_path, library=library,
    )


def test_native_admission_succeeds_without_legacy_node_or_gate(installed):
    result = installed.validator.require_native_sparse_attention_backend(installed.adapter_path)
    assert result == {
        "node_id": "BlockSparseAttention",
        "backend": "kitchen-xpu/omni-cute",
        "library": str(installed.library),
        "library_sha256": hashlib.sha256(installed.library.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("switch", ["OMNIXPU_ENABLE", "OMNIXPU_SPARSE_ATTENTION"])
def test_disabled_adapter_cannot_pass_admission(installed, monkeypatch, switch):
    monkeypatch.setenv(switch, "0")
    with pytest.raises(RuntimeError, match=switch):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)


def test_legacy_only_native_api_cannot_pass(installed):
    installed.cute.supports_sol_attn = lambda: True
    installed.native.is_available = lambda: False
    with pytest.raises(RuntimeError, match="complete quantized"):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)


def test_missing_kitchen_route_cannot_pass(installed):
    installed.kitchen.sol_attn_is_available = lambda device: False
    with pytest.raises(RuntimeError, match="Kitchen native sparse"):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)


@pytest.mark.parametrize("kind", ["external", "missing", "empty"])
def test_unpackaged_or_missing_library_cannot_pass(installed, tmp_path, kind):
    path = tmp_path / "external.so"
    if kind == "external":
        path.write_bytes(b"unpackaged library")
    installed.cute._find_extension = lambda: "" if kind == "empty" else str(path)
    with pytest.raises(RuntimeError, match="DSO must belong"):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)


def test_missing_adapter_cannot_pass(installed, tmp_path):
    with pytest.raises(RuntimeError, match="adapter is missing"):
        installed.validator.require_native_sparse_attention_backend(tmp_path / "missing.py")


def test_incompatible_upstream_guard_cannot_pass(installed):
    installed.adapter.apply = lambda: (False, "unsupported eligibility")
    with pytest.raises(RuntimeError, match="unsupported eligibility"):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)


def test_custom_node_cannot_impersonate_builtin(installed):
    installed.upstream.__file__ = "/custom_nodes/replacement/nodes_sparse_attention.py"
    with pytest.raises(RuntimeError, match="native sparse node source"):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)


def test_wrong_node_schema_cannot_pass(installed):
    installed.upstream.BlockSparseAttention.define_schema = lambda: SimpleNamespace(node_id="LegacyNode")
    with pytest.raises(RuntimeError, match="native sparse node ID"):
        installed.validator.require_native_sparse_attention_backend(installed.adapter_path)
