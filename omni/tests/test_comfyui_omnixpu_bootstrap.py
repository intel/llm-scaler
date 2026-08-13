"""Portable ownership and bootstrap tests for ComfyUI-OmniXPU."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"


def _load_module(name: str, path: Path, *, package_path: Path | None = None):
    locations = [str(package_path)] if package_path is not None else None
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=locations
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_registry(monkeypatch):
    package_name = "omnixpu_bootstrap_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    monkeypatch.setitem(sys.modules, package_name, package)
    return _load_module(
        f"{package_name}.patches",
        _PLUGIN / "patches" / "__init__.py",
        package_path=_PLUGIN / "patches",
    )


def test_generic_kitchen_operations_are_not_custom_node_components(monkeypatch):
    patches = _load_registry(monkeypatch)
    names = {entry["name"] for entry in patches.get_components()}
    modules = {entry["module"] for entry in patches.get_components()}

    assert "rope" not in names
    assert "int8" not in names
    assert "fp8_neg_zero_fix" not in names
    assert all("patch_rope" not in module for module in modules)
    assert all("patch_int8.py" not in module for module in modules)
    assert all("patch_fp8_fix" not in module for module in modules)


def test_lora_memory_budget_is_registered_as_an_adapter(monkeypatch):
    patches = _load_registry(monkeypatch)
    components = {entry["name"]: entry for entry in patches.get_components()}

    assert components["lora_memory_budget"]["kind"] == "adapter"


def test_dynamic_vram_trim_runs_inside_lora_budget_wrapper(monkeypatch):
    patches = _load_registry(monkeypatch)
    names = [entry["name"] for entry in patches.get_components()]

    assert names.index("dynamic_vram_boundary_trim") < names.index(
        "lora_memory_budget"
    )


def test_legacy_global_fixes_default_to_disabled(monkeypatch):
    for name in (
        "OMNIXPU_ENABLE",
        "OMNIXPU_ATTENTION",
        "OMNIXPU_ROTARY",
        "OMNIXPU_NORM",
        "OMNIXPU_FP8_GEMM",
        "OMNIXPU_INT8_FFN",
        "OMNIXPU_DYNAMIC_VRAM_BOUNDARY_TRIM",
        "OMNIXPU_LORA_MEMORY",
        "OMNIXPU_INTERPOLATE_FIX",
        "OMNIXPU_MEDIAN_FIX",
    ):
        monkeypatch.delenv(name, raising=False)

    config = _load_module("omnixpu_bootstrap_config", _PLUGIN / "config.py").config
    assert config.attention
    assert config.rotary
    assert config.norm
    assert config.fp8_gemm
    assert config.int8_ffn
    assert config.dynamic_vram_boundary_trim
    assert config.lora_memory
    assert not config.interpolate_fix
    assert not config.median_fix
    assert not hasattr(config, "rope")
    assert not hasattr(config, "int8")
    assert not hasattr(config, "fp8_neg_zero_fix")


def test_disabled_components_are_reported_without_importing_modules(monkeypatch):
    patches = _load_registry(monkeypatch)
    cfg = types.SimpleNamespace(
        attention=False,
        rotary=False,
        norm=False,
        fp8_gemm=False,
        int8_ffn=False,
        dynamic_vram_boundary_trim=False,
        lora_memory=False,
        interpolate_fix=False,
        median_fix=False,
    )
    patches.apply_all_patches(cfg)

    assert all(entry["status"] == "skipped" for entry in patches.get_status())
    for entry in patches.get_components():
        module_name = entry["module"].removesuffix(".py").replace("/", ".")
        assert f"omnixpu_bootstrap_test.{module_name}" not in sys.modules


def test_component_loader_registers_parent_package_for_sibling_imports(
    monkeypatch, tmp_path
):
    patches = _load_registry(monkeypatch)
    package_name = "omnixpu_component_loader_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, package_name, package)

    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "__init__.py").write_text("", encoding="utf-8")
    (adapters / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    (adapters / "component.py").write_text(
        "from .helper import VALUE\n"
        "def apply():\n"
        "    return VALUE == 7, ''\n",
        encoding="utf-8",
    )

    component = patches.Component(
        "test_adapter", "test", "adapter", "test", "adapters/component.py"
    )
    apply_fn = patches._load_component(tmp_path, package_name, component)

    assert apply_fn() == (True, "")
    assert f"{package_name}.adapters" in sys.modules
