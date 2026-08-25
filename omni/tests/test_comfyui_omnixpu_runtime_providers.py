"""Lifecycle tests for co-installable OmniXPU runtime providers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


_RUNTIME = (
    Path(__file__).parents[1] / "ComfyUI-OmniXPU" / "runtime_bootstrap.py"
)
_DIAGNOSTICS = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "nodes"
    / "diagnostics.py"
)


@pytest.fixture
def runtime(monkeypatch):
    original_meta_path = list(sys.meta_path)
    module_name = f"omnixpu_runtime_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, _RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    monkeypatch.delenv("OMNIXPU_PROVIDER_BOOTSTRAP", raising=False)
    monkeypatch.delenv("OMNIXPU_ENABLE", raising=False)
    yield module
    sys.meta_path[:] = original_meta_path
    for name in tuple(sys.modules):
        if name in {"comfy_kitchen", "comfy_aimdo"} or name.startswith(
            ("comfy_kitchen.", "comfy_aimdo.")
        ):
            sys.modules.pop(name, None)


def _provider(runtime, tmp_path, provider_id, canonical_import, activation):
    vendor_root = tmp_path / provider_id / "vendor"
    canonical_root = vendor_root / canonical_import
    canonical_root.mkdir(parents=True)
    manifest = {
        "provider_id": provider_id,
        "activation": activation,
    }
    return runtime.RuntimeProvider(
        provider_id=provider_id,
        canonical_import=canonical_import,
        canonical_distribution=canonical_import.replace("_", "-"),
        version="1.0",
        vendor_root=vendor_root,
        canonical_root=canonical_root,
        manifest=manifest,
    )


def _install_official_aimdo(monkeypatch, tmp_path, *, dynamic=True):
    official_root = tmp_path / "official_aimdo"
    official_root.mkdir()
    package = types.ModuleType("comfy_aimdo")
    package.__path__ = [str(official_root)]
    control = types.ModuleType("comfy_aimdo.control")
    control.__package__ = "comfy_aimdo"
    control.__file__ = str(official_root / "control.py")
    control.lib = None
    control.devctxs = []
    control._xpu_allocator_ready = False
    control._torch_allocator = None
    control.marker = "official"
    monkeypatch.setitem(sys.modules, "comfy_aimdo", package)
    monkeypatch.setitem(sys.modules, "comfy_aimdo.control", control)

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    cli_args = types.ModuleType("comfy.cli_args")
    cli_args.args = types.SimpleNamespace(
        enable_dynamic_vram=dynamic,
        reserve_vram=1.5,
        disable_nvml_pressure=False,
    )
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli_args)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    return package, control


def test_kitchen_is_routed_only_when_canonical_import_occurs(
    runtime, monkeypatch, tmp_path
):
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_kitchen.xpu",
        "comfy_kitchen",
        {"strategy": "canonical_meta_path", "requires_dynamic_vram": False},
    )
    (provider.canonical_root / "feature.py").write_text(
        "VALUE = 'xpu-provider'\n", encoding="utf-8"
    )
    (provider.canonical_root / "__init__.py").write_text(
        "from .feature import VALUE\n", encoding="utf-8"
    )
    monkeypatch.delitem(sys.modules, "comfy_kitchen", raising=False)

    state = runtime.bootstrap(
        providers_override={provider.provider_id: provider},
        dynamic_vram_override=False,
    )

    assert "comfy_kitchen" not in sys.modules
    assert state["providers"][provider.provider_id]["status"] == "active"
    kitchen = importlib.import_module("comfy_kitchen")
    assert kitchen.VALUE == "xpu-provider"
    assert Path(kitchen.__file__).is_relative_to(provider.canonical_root)


def test_aimdo_overlay_preserves_module_identity_and_cli_policy(
    runtime, monkeypatch, tmp_path
):
    package, official_control = _install_official_aimdo(monkeypatch, tmp_path)
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )
    (provider.canonical_root / "control.py").write_text(
        "lib = None\n"
        "devctxs = []\n"
        "_xpu_allocator_ready = False\n"
        "_torch_allocator = None\n"
        "def init(**kwargs):\n"
        "    global lib, _xpu_allocator_ready, _torch_allocator, CALL\n"
        "    CALL = kwargs\n"
        "    lib = object()\n"
        "    _torch_allocator = object()\n"
        "    _xpu_allocator_ready = True\n"
        "    return True\n",
        encoding="utf-8",
    )

    state = runtime.bootstrap(providers_override={provider.provider_id: provider})

    assert sys.modules["comfy_aimdo.control"] is official_control
    assert not hasattr(official_control, "marker")
    assert Path(official_control.__file__) == provider.canonical_root / "control.py"
    assert official_control.CALL == {
        "implementation": "xpu",
        "simple_vram_headroom": int(1.5 * 1024**3),
        "nvml_pressure": True,
        "xpu_allocator_mode": "global",
    }
    assert package.__path__[0] == str(provider.canonical_root)
    assert state["providers"][provider.provider_id]["status"] == "active"


def test_aimdo_unwinds_official_pre_device_native_state(
    runtime, monkeypatch, tmp_path
):
    _, official_control = _install_official_aimdo(monkeypatch, tmp_path)
    official_control.lib = object()
    deinit_calls = []

    def deinit():
        deinit_calls.append(True)
        official_control.lib = None

    official_control.deinit = deinit
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )
    (provider.canonical_root / "control.py").write_text(
        "lib = None\n"
        "devctxs = []\n"
        "_xpu_allocator_ready = False\n"
        "_torch_allocator = None\n"
        "def init(**kwargs):\n"
        "    global lib, _xpu_allocator_ready, _torch_allocator\n"
        "    lib = object()\n"
        "    _torch_allocator = object()\n"
        "    _xpu_allocator_ready = True\n"
        "    return True\n",
        encoding="utf-8",
    )

    state = runtime.bootstrap(providers_override={provider.provider_id: provider})

    assert deinit_calls == [True]
    assert sys.modules["comfy_aimdo.control"] is official_control
    assert state["providers"][provider.provider_id]["status"] == "active"


def test_aimdo_refuses_official_device_state(runtime, monkeypatch, tmp_path):
    _, official_control = _install_official_aimdo(monkeypatch, tmp_path)
    official_control.lib = object()
    official_control.devctxs = [object()]
    official_control.deinit = lambda: pytest.fail("must not unwind device state")
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )
    (provider.canonical_root / "control.py").write_text(
        "raise AssertionError('must not execute')\n", encoding="utf-8"
    )

    state = runtime.bootstrap(providers_override={provider.provider_id: provider})

    assert sys.modules["comfy_aimdo.control"] is official_control
    assert official_control.devctxs
    assert state["providers"][provider.provider_id]["status"] == "skipped"
    assert "irreversible runtime state" in state["providers"][provider.provider_id][
        "reason"
    ]


def test_aimdo_reversible_failure_restores_official_module(
    runtime, monkeypatch, tmp_path
):
    package, official_control = _install_official_aimdo(monkeypatch, tmp_path)
    original_path = package.__path__
    original_file = official_control.__file__
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )
    (provider.canonical_root / "helper.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (provider.canonical_root / "control.py").write_text(
        "from . import helper\n"
        "lib = None\n"
        "devctxs = []\n"
        "_xpu_allocator_ready = False\n"
        "_torch_allocator = None\n"
        "def init(**kwargs):\n"
        "    return False\n",
        encoding="utf-8",
    )

    state = runtime.bootstrap(providers_override={provider.provider_id: provider})

    assert sys.modules["comfy_aimdo.control"] is official_control
    assert official_control.marker == "official"
    assert official_control.__file__ == original_file
    assert package.__path__ is original_path
    assert "comfy_aimdo.helper" not in sys.modules
    assert state["providers"][provider.provider_id]["status"] == "skipped"
    assert "initialization returned false" in state["providers"][provider.provider_id][
        "reason"
    ]


def test_aimdo_failure_after_native_state_is_fatal(runtime, monkeypatch, tmp_path):
    _install_official_aimdo(monkeypatch, tmp_path)
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )
    (provider.canonical_root / "control.py").write_text(
        "lib = None\n"
        "devctxs = []\n"
        "_xpu_allocator_ready = False\n"
        "_torch_allocator = None\n"
        "def init(**kwargs):\n"
        "    global lib\n"
        "    lib = object()\n"
        "    raise RuntimeError('after native load')\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="refusing unsafe fallback"):
        runtime.bootstrap(providers_override={provider.provider_id: provider})


def test_aimdo_is_not_overlaid_when_dynamic_vram_is_disabled(
    runtime, monkeypatch, tmp_path
):
    _, official_control = _install_official_aimdo(
        monkeypatch, tmp_path, dynamic=False
    )
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )
    (provider.canonical_root / "control.py").write_text(
        "raise AssertionError('must not execute')\n", encoding="utf-8"
    )

    state = runtime.bootstrap(providers_override={provider.provider_id: provider})

    assert sys.modules["comfy_aimdo.control"] is official_control
    assert official_control.marker == "official"
    assert state["providers"][provider.provider_id] == {
        "status": "skipped",
        "reason": "DynamicVRAM is disabled",
    }


def test_required_mode_refuses_a_partial_provider_set(
    runtime, monkeypatch, tmp_path
):
    monkeypatch.setenv("OMNIXPU_PROVIDER_BOOTSTRAP", "required")
    provider = _provider(
        runtime,
        tmp_path,
        "comfy_kitchen.xpu",
        "comfy_kitchen",
        {"strategy": "canonical_meta_path", "requires_dynamic_vram": False},
    )

    with pytest.raises(SystemExit, match="missing=.*comfy_aimdo.xpu"):
        runtime.bootstrap(providers_override={provider.provider_id: provider})


def test_required_mode_refuses_disabled_dynamic_vram(runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIXPU_PROVIDER_BOOTSTRAP", "required")
    kitchen = _provider(
        runtime,
        tmp_path,
        "comfy_kitchen.xpu",
        "comfy_kitchen",
        {"strategy": "canonical_meta_path", "requires_dynamic_vram": False},
    )
    (kitchen.canonical_root / "__init__.py").write_text("", encoding="utf-8")
    aimdo = _provider(
        runtime,
        tmp_path,
        "comfy_aimdo.xpu",
        "comfy_aimdo",
        {
            "strategy": "canonical_control_overlay",
            "requires_dynamic_vram": True,
            "allocator_modes": {"linux": ["global"], "win32": ["native_hook"]},
        },
    )

    with pytest.raises(SystemExit, match="requires DynamicVRAM"):
        runtime.bootstrap(
            providers_override={
                kitchen.provider_id: kitchen,
                aimdo.provider_id: aimdo,
            },
            dynamic_vram_override=False,
        )


@pytest.mark.parametrize(
    ("compatible_versions", "vendored_hash", "expected_error"),
    (
        (["0.2.31"], hashlib.sha256(b"different").hexdigest(), "hash mismatch"),
        (["0.2.30"], None, "is incompatible"),
    ),
)
def test_discovery_rejects_incompatible_or_tampered_provider(
    runtime,
    monkeypatch,
    tmp_path,
    compatible_versions,
    vendored_hash,
    expected_error,
):
    provider_root = tmp_path / "site"
    vendor_root = provider_root / "comfy_kitchen_xpu_runtime" / "_vendor"
    canonical_root = vendor_root / "comfy_kitchen"
    canonical_root.mkdir(parents=True)
    vendored = canonical_root / "__init__.py"
    vendored.write_text("VALUE = 1\n", encoding="utf-8")
    relative = vendored.relative_to(provider_root).as_posix()
    manifest = {
        "schema_version": 1,
        "provider_id": "comfy_kitchen.xpu",
        "provider_distribution": {
            "name": "comfy-kitchen-xpu-runtime",
            "version": "0.2.31",
        },
        "provider_package": "comfy_kitchen_xpu_runtime",
        "canonical_distribution": {
            "name": "comfy-kitchen",
            "compatible_versions": compatible_versions,
        },
        "canonical_import": "comfy_kitchen",
        "source": {
            "repository": "https://github.com/xiangyuT/comfy-kitchen-xpu.git",
            "revision": "a" * 40,
            "distribution": "comfy-kitchen",
            "version": "0.2.31",
            "wheel_sha256": "b" * 64,
        },
        "runtime": {
            "torch_version": "2.13.0+xpu",
            "platforms": [sys.platform],
            "xpu_targets": ["bmg"],
        },
        "activation": {
            "strategy": "canonical_meta_path",
            "requires_dynamic_vram": False,
        },
        "vendor_root": "comfy_kitchen_xpu_runtime/_vendor",
        "vendored_files": {
            relative: vendored_hash or hashlib.sha256(vendored.read_bytes()).hexdigest()
        },
    }

    class FakeDistribution:
        version = "0.2.31"
        metadata = {"Name": "comfy-kitchen-xpu-runtime"}
        files = tuple(PurePath for PurePath in (Path(relative),))

        @staticmethod
        def locate_file(path):
            return provider_root / path

    class FakeEntryPoint:
        name = "comfy_kitchen.xpu"
        dist = FakeDistribution()

        @staticmethod
        def load():
            return lambda: manifest

    monkeypatch.setattr(runtime, "_entry_points", lambda: (FakeEntryPoint(),))
    monkeypatch.setattr(
        runtime.importlib.metadata, "version", lambda name: "0.2.31"
    )
    monkeypatch.setattr(
        runtime, "_torch_version_without_import", lambda: "2.13.0+xpu"
    )
    monkeypatch.setenv("OMNI_IMAGE_XPU_TARGET", "bmg")

    providers, errors = runtime.discover_providers()

    assert providers == {}
    assert len(errors) == 1
    assert expected_error in errors[0]


def test_discovery_is_fatal_if_provider_metadata_imports_torch(
    runtime, monkeypatch
):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    class FakeEntryPoint:
        name = "comfy_kitchen.xpu"

        @staticmethod
        def load():
            monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
            return lambda: {}

    monkeypatch.setattr(runtime, "_entry_points", lambda: (FakeEntryPoint(),))

    with pytest.raises(SystemExit, match="unsafe allocator fallback"):
        runtime.discover_providers()


def test_diagnostics_reports_provider_activation_and_rejection(monkeypatch):
    runtime = types.SimpleNamespace(
        get_state=lambda: {
            "status": "active",
            "mode": "auto",
            "providers": {
                "comfy_kitchen.xpu": {"status": "active", "reason": ""},
                "comfy_aimdo.xpu": {
                    "status": "skipped",
                    "reason": "DynamicVRAM is disabled",
                },
            },
            "errors": ["duplicate provider entry point: comfy_kitchen.xpu"],
        }
    )
    monkeypatch.setitem(
        sys.modules, "_comfyui_omnixpu_runtime_bootstrap", runtime
    )
    spec = importlib.util.spec_from_file_location(
        "omnixpu_provider_diagnostics_test", _DIAGNOSTICS
    )
    assert spec is not None and spec.loader is not None
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)

    status = diagnostics.OmniXPUStatus().get_status()[0]

    assert "runtime providers: active (mode=auto)" in status
    assert "comfy_kitchen.xpu: active" in status
    assert "comfy_aimdo.xpu: skipped (DynamicVRAM is disabled)" in status
    assert "rejected: duplicate provider entry point" in status
