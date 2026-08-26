"""Early, conditional routing for co-installable Intel XPU runtime providers."""

from __future__ import annotations

import copy
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


ENTRY_POINT_GROUP = "comfyui_omnixpu.runtime_providers"
BOOTSTRAP_ENV = "OMNIXPU_PROVIDER_BOOTSTRAP"
MASTER_ENV = "OMNIXPU_ENABLE"
TARGET_ENV = "OMNI_IMAGE_XPU_TARGET"
VALID_MODES = frozenset(("auto", "off", "required"))
KNOWN_PROVIDERS = frozenset(("comfy_aimdo.xpu", "comfy_kitchen.xpu"))
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOG = logging.getLogger("ComfyUI-OmniXPU")
_PROVIDER_CONTRACTS = {
    "comfy_kitchen.xpu": {
        "provider_distribution": "comfy-kitchen-xpu-runtime",
        "provider_package": "comfy_kitchen_xpu_runtime",
        "canonical_distribution": "comfy-kitchen",
        "canonical_import": "comfy_kitchen",
        "source_repository": "https://github.com/xiangyuT/comfy-kitchen-xpu.git",
        "activation_strategy": "canonical_meta_path",
        "requires_dynamic_vram": False,
    },
    "comfy_aimdo.xpu": {
        "provider_distribution": "comfy-aimdo-xpu-runtime",
        "provider_package": "comfy_aimdo_xpu_runtime",
        "canonical_distribution": "comfy-aimdo",
        "canonical_import": "comfy_aimdo",
        "source_repository": "https://github.com/xiangyuT/comfy-aimdo-xpu.git",
        "activation_strategy": "canonical_control_overlay",
        "requires_dynamic_vram": True,
    },
}


@dataclass(frozen=True)
class RuntimeProvider:
    provider_id: str
    canonical_import: str
    canonical_distribution: str
    version: str
    vendor_root: Path
    canonical_root: Path
    manifest: dict[str, Any]


class _CanonicalPackageFinder(importlib.abc.MetaPathFinder):
    """Route exactly one canonical top-level package to a provider tree."""

    def __init__(self, provider: RuntimeProvider):
        self.provider = provider

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.provider.canonical_import:
            return None
        package_root = self.provider.canonical_root
        initializer = package_root / "__init__.py"
        if not initializer.is_file():
            raise ImportError(
                f"provider {self.provider.provider_id} has no {initializer}"
            )
        return importlib.util.spec_from_file_location(
            fullname,
            initializer,
            submodule_search_locations=[str(package_root)],
        )


_FINDERS: dict[str, _CanonicalPackageFinder] = {}
_STATE: dict[str, Any] = {
    "status": "not-run",
    "mode": None,
    "providers": {},
    "errors": [],
}


def get_state() -> dict[str, Any]:
    """Return a diagnostics-safe snapshot of process-global bootstrap state."""

    return copy.deepcopy(_STATE)


def _set_provider_state(provider_id: str, status: str, reason: str = "") -> None:
    _STATE["providers"][provider_id] = {"status": status, "reason": reason}


def _mode() -> str:
    if os.environ.get(MASTER_ENV, "1") == "0":
        return "off"
    mode = os.environ.get(BOOTSTRAP_ENV, "auto").strip().lower()
    if mode not in VALID_MODES:
        choices = ", ".join(sorted(VALID_MODES))
        raise RuntimeError(f"{BOOTSTRAP_ENV} must be one of {choices}, got {mode!r}")
    return mode


def _torch_version_without_import() -> str:
    loaded = sys.modules.get("torch")
    if loaded is not None:
        return str(getattr(loaded, "__version__", ""))
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError) as exc:
        raise RuntimeError(f"cannot locate PyTorch: {exc}") from exc
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the PyTorch package directory")
    for location in spec.submodule_search_locations:
        version_file = Path(location) / "version.py"
        if not version_file.is_file():
            continue
        version_spec = importlib.util.spec_from_file_location(
            "_omnixpu_torch_version", version_file
        )
        if version_spec is None or version_spec.loader is None:
            continue
        module = importlib.util.module_from_spec(version_spec)
        version_spec.loader.exec_module(module)
        version = str(getattr(module, "__version__", ""))
        if version:
            return version
    raise RuntimeError("PyTorch version.py did not expose __version__")


def _entry_points():
    points = importlib.metadata.entry_points()
    if hasattr(points, "select"):
        return tuple(points.select(group=ENTRY_POINT_GROUP))
    return tuple(points.get(ENTRY_POINT_GROUP, ()))


def _safe_relative_path(raw_value: object, *, field: str) -> PurePosixPath:
    if not isinstance(raw_value, str) or not raw_value:
        raise RuntimeError(f"provider manifest {field} must be a non-empty path")
    path = PurePosixPath(raw_value)
    if path.is_absolute() or ".." in path.parts or "\\" in raw_value:
        raise RuntimeError(f"provider manifest {field} is unsafe: {raw_value!r}")
    return path


def _distribution_name(distribution) -> str:
    return str(distribution.metadata.get("Name", ""))


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _validate_manifest(entry_point, manifest: object) -> RuntimeProvider:
    if not isinstance(manifest, dict):
        raise RuntimeError("provider entry point did not return a manifest object")
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported provider manifest schema")
    provider_id = str(manifest.get("provider_id", ""))
    if provider_id != entry_point.name or provider_id not in KNOWN_PROVIDERS:
        raise RuntimeError(
            f"provider id {provider_id!r} does not match entry point {entry_point.name!r}"
        )
    contract = _PROVIDER_CONTRACTS[provider_id]

    provider_distribution = manifest.get("provider_distribution")
    canonical_distribution = manifest.get("canonical_distribution")
    runtime = manifest.get("runtime")
    source = manifest.get("source")
    if not all(
        isinstance(item, dict)
        for item in (provider_distribution, canonical_distribution, runtime, source)
    ):
        raise RuntimeError("provider manifest is missing required mapping fields")

    installed_provider_name = _distribution_name(entry_point.dist)
    installed_provider_version = str(entry_point.dist.version)
    expected_provider_name = str(provider_distribution.get("name", ""))
    expected_provider_version = str(provider_distribution.get("version", ""))
    if _normalize_distribution(expected_provider_name) != _normalize_distribution(
        contract["provider_distribution"]
    ):
        raise RuntimeError("provider distribution is not the registered contract")
    if manifest.get("provider_package") != contract["provider_package"]:
        raise RuntimeError("provider package is not the registered contract")
    if _normalize_distribution(installed_provider_name) != _normalize_distribution(
        expected_provider_name
    ):
        raise RuntimeError("provider distribution name does not match its manifest")
    if installed_provider_version != expected_provider_version:
        raise RuntimeError("provider distribution version does not match its manifest")

    canonical_name = str(canonical_distribution.get("name", ""))
    if _normalize_distribution(canonical_name) != _normalize_distribution(
        contract["canonical_distribution"]
    ):
        raise RuntimeError("canonical distribution is not the registered contract")
    compatible_versions = canonical_distribution.get("compatible_versions")
    if not isinstance(compatible_versions, list) or not all(
        isinstance(value, str) for value in compatible_versions
    ):
        raise RuntimeError("canonical compatible_versions must be a string list")
    try:
        official_version = importlib.metadata.version(canonical_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"official distribution {canonical_name!r} is not installed"
        ) from exc
    if official_version not in compatible_versions:
        raise RuntimeError(
            f"official {canonical_name} {official_version} is incompatible; "
            f"provider accepts {compatible_versions}"
        )

    torch_version = _torch_version_without_import()
    if not torch_version.endswith("+xpu"):
        raise RuntimeError(f"PyTorch {torch_version!r} is not an XPU build")
    if torch_version != runtime.get("torch_version"):
        raise RuntimeError(
            f"PyTorch {torch_version} does not match provider "
            f"{runtime.get('torch_version')}"
        )
    platforms = runtime.get("platforms")
    if not isinstance(platforms, list) or sys.platform not in platforms:
        raise RuntimeError(f"provider does not support platform {sys.platform}")
    target = os.environ.get(TARGET_ENV)
    targets = runtime.get("xpu_targets")
    if target and (not isinstance(targets, list) or target not in targets):
        raise RuntimeError(f"provider does not support XPU target {target}")

    revision = str(source.get("revision", ""))
    if not _REVISION_PATTERN.fullmatch(revision):
        raise RuntimeError("provider source revision is not a full lowercase Git SHA")
    canonical_import = str(manifest.get("canonical_import", ""))
    if canonical_import != contract["canonical_import"]:
        raise RuntimeError("canonical import is not the registered contract")
    if source.get("repository") != contract["source_repository"]:
        raise RuntimeError("provider source repository is not the registered contract")
    if _normalize_distribution(str(source.get("distribution", ""))) != (
        _normalize_distribution(canonical_name)
    ):
        raise RuntimeError("provider source distribution does not match canonical")
    if source.get("version") != expected_provider_version:
        raise RuntimeError("provider source version does not match provider version")
    if not _SHA256_PATTERN.fullmatch(str(source.get("wheel_sha256", ""))):
        raise RuntimeError("provider source wheel SHA256 is invalid")
    activation = manifest.get("activation")
    if not isinstance(activation, dict):
        raise RuntimeError("provider activation contract is missing")
    if activation.get("strategy") != contract["activation_strategy"]:
        raise RuntimeError("provider activation strategy is not registered")
    if activation.get("requires_dynamic_vram") is not contract[
        "requires_dynamic_vram"
    ]:
        raise RuntimeError("provider DynamicVRAM requirement is not registered")

    distribution_files = tuple(entry_point.dist.files or ())
    conflicting_prefix = f"{canonical_import}/"
    if any(str(path).startswith(conflicting_prefix) for path in distribution_files):
        raise RuntimeError(
            f"provider distribution illegally owns top-level {canonical_import} files"
        )

    vendor_relative = _safe_relative_path(
        manifest.get("vendor_root"), field="vendor_root"
    )
    vendor_root = Path(entry_point.dist.locate_file(vendor_relative)).resolve()
    if not vendor_root.is_dir():
        raise RuntimeError(f"provider vendor root is missing: {vendor_root}")
    canonical_root = vendor_root / canonical_import
    if not canonical_root.is_dir():
        raise RuntimeError(f"provider canonical root is missing: {canonical_root}")

    hashes = manifest.get("vendored_files")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("provider manifest has no vendored file hashes")
    for raw_path, expected_hash in hashes.items():
        relative = _safe_relative_path(raw_path, field="vendored_files key")
        path = Path(entry_point.dist.locate_file(relative)).resolve()
        try:
            path.relative_to(vendor_root)
        except ValueError as exc:
            raise RuntimeError(f"vendored file escapes provider root: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"vendored provider file is missing: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"vendored provider file hash mismatch: {path}")

    return RuntimeProvider(
        provider_id=provider_id,
        canonical_import=canonical_import,
        canonical_distribution=canonical_name,
        version=expected_provider_version,
        vendor_root=vendor_root,
        canonical_root=canonical_root,
        manifest=manifest,
    )


def discover_providers() -> tuple[dict[str, RuntimeProvider], list[str]]:
    """Discover and verify providers without importing PyTorch."""

    providers: dict[str, RuntimeProvider] = {}
    errors: list[str] = []
    torch_was_loaded = "torch" in sys.modules
    for entry_point in _entry_points():
        if entry_point.name not in KNOWN_PROVIDERS:
            continue
        if entry_point.name in providers:
            errors.append(f"duplicate provider entry point: {entry_point.name}")
            continue
        try:
            factory = entry_point.load()
            manifest = factory()
            if not torch_was_loaded and "torch" in sys.modules:
                raise SystemExit(
                    "[OmniXPU] provider metadata entry point imported PyTorch; "
                    "refusing an unsafe allocator fallback"
                )
            providers[entry_point.name] = _validate_manifest(entry_point, manifest)
        except Exception as exc:
            errors.append(f"{entry_point.name}: {exc}")
    return providers, errors


def _activate_kitchen(provider: RuntimeProvider) -> None:
    canonical = provider.canonical_import
    if canonical in sys.modules:
        raise RuntimeError(f"{canonical} was imported before provider routing")
    existing = _FINDERS.get(canonical)
    if existing is not None:
        if existing.provider == provider:
            return
        raise RuntimeError(f"another provider already routes {canonical}")
    finder = _CanonicalPackageFinder(provider)
    sys.meta_path.insert(0, finder)
    _FINDERS[canonical] = finder


def _dynamic_vram_request() -> tuple[bool, int | None, bool]:
    cli_args = sys.modules.get("comfy.cli_args")
    args = getattr(cli_args, "args", None)
    if args is None:
        return False, None, False
    enabled = bool(getattr(args, "enable_dynamic_vram", False))
    reserve_vram = getattr(args, "reserve_vram", None)
    headroom = None if reserve_vram is None else int(float(reserve_vram) * 1024**3)
    nvml_pressure = not bool(getattr(args, "disable_nvml_pressure", False))
    return enabled, headroom, nvml_pressure


def _aimdo_is_pristine(control: ModuleType) -> bool:
    return (
        getattr(control, "lib", None) is None
        and not getattr(control, "devctxs", ())
        and not getattr(control, "_xpu_allocator_ready", False)
        and getattr(control, "_torch_allocator", None) is None
    )


def _prepare_official_aimdo(control: ModuleType) -> None:
    """Unwind only AIMDO's reversible pre-device official initialization."""

    if _aimdo_is_pristine(control):
        return
    if (
        getattr(control, "devctxs", ())
        or getattr(control, "_xpu_allocator_ready", False)
        or getattr(control, "_torch_allocator", None) is not None
    ):
        raise RuntimeError("official AIMDO already owns irreversible runtime state")
    if getattr(control, "lib", None) is None:
        raise RuntimeError("official AIMDO state is not lifecycle-safe to unwind")
    deinit = getattr(control, "deinit", None)
    if not callable(deinit):
        raise RuntimeError("official AIMDO cannot unwind its pre-device native state")
    try:
        deinit()
    except Exception as exc:
        raise RuntimeError(
            f"official AIMDO pre-device unwind failed: {exc}"
        ) from exc
    if not _aimdo_is_pristine(control):
        raise RuntimeError("official AIMDO pre-device unwind left live runtime state")


def _activate_aimdo(
    provider: RuntimeProvider,
    *,
    simple_vram_headroom: int | None,
    nvml_pressure: bool,
) -> None:
    package = sys.modules.get("comfy_aimdo")
    control = sys.modules.get("comfy_aimdo.control")
    if package is None or control is None:
        raise RuntimeError("official comfy_aimdo.control is not loaded")
    if sys.platform == "linux" and "torch" in sys.modules:
        raise RuntimeError("PyTorch was imported before Linux allocator takeover")

    existing_modules = {
        name for name in sys.modules if name == "comfy_aimdo" or name.startswith("comfy_aimdo.")
    }
    unexpected = existing_modules - {"comfy_aimdo", "comfy_aimdo.control"}
    if unexpected:
        raise RuntimeError(
            "AIMDO submodules were imported before takeover: "
            + ", ".join(sorted(unexpected))
        )

    _prepare_official_aimdo(control)

    control_path = provider.canonical_root / "control.py"
    if not control_path.is_file():
        raise RuntimeError(f"provider control module is missing: {control_path}")
    package_path_snapshot = getattr(package, "__path__", None)
    control_snapshot = dict(control.__dict__)
    module_snapshot = set(sys.modules)
    irreversible = False
    try:
        package.__path__ = [str(provider.canonical_root), *list(package.__path__)]
        spec = importlib.util.spec_from_file_location("comfy_aimdo.control", control_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not create the AIMDO provider module spec")
        control.__dict__.clear()
        control.__dict__.update(
            {
                "__name__": "comfy_aimdo.control",
                "__package__": "comfy_aimdo",
                "__loader__": spec.loader,
                "__spec__": spec,
                "__file__": str(control_path),
                "__cached__": None,
            }
        )
        spec.loader.exec_module(control)

        allocator_modes = provider.manifest["activation"].get("allocator_modes", {})
        modes = allocator_modes.get(sys.platform, ())
        if len(modes) != 1:
            raise RuntimeError(
                f"provider does not select one allocator mode for {sys.platform}"
            )
        if sys.platform == "win32" and "torch" not in sys.modules:
            __import__("torch")
        initialized = control.init(
            implementation="xpu",
            simple_vram_headroom=simple_vram_headroom,
            nvml_pressure=nvml_pressure,
            xpu_allocator_mode=modes[0],
        )
        irreversible = not _aimdo_is_pristine(control)
        if not initialized:
            raise RuntimeError("AIMDO XPU provider initialization returned false")
    except Exception as exc:
        irreversible = irreversible or not _aimdo_is_pristine(control)
        if irreversible:
            raise SystemExit(
                "[OmniXPU] AIMDO provider failed after allocator/native state "
                f"became live; refusing unsafe fallback: {exc}"
            ) from exc
        control.__dict__.clear()
        control.__dict__.update(control_snapshot)
        package.__path__ = package_path_snapshot
        for name in set(sys.modules) - module_snapshot:
            if name.startswith("comfy_aimdo."):
                sys.modules.pop(name, None)
        raise


def bootstrap(
    *,
    dynamic_vram_override: bool | None = None,
    providers_override: dict[str, RuntimeProvider] | None = None,
) -> dict[str, Any]:
    """Discover providers and activate only lifecycle-safe XPU routes."""

    mode = _mode()
    _STATE.update({"status": "running", "mode": mode, "providers": {}, "errors": []})
    if mode == "off":
        _STATE["status"] = "disabled"
        return get_state()

    if providers_override is None:
        providers, errors = discover_providers()
    else:
        providers, errors = providers_override, []
    _STATE["errors"].extend(errors)
    if errors:
        for error in errors:
            _LOG.warning("[OmniXPU] runtime provider rejected: %s", error)
    if mode == "required" and (errors or KNOWN_PROVIDERS - providers.keys()):
        missing = sorted(KNOWN_PROVIDERS - providers.keys())
        detail = "; ".join([*errors, f"missing={missing}" if missing else ""])
        raise SystemExit(f"[OmniXPU] required runtime providers unavailable: {detail}")

    kitchen = providers.get("comfy_kitchen.xpu")
    if kitchen is not None:
        try:
            _activate_kitchen(kitchen)
            _set_provider_state(kitchen.provider_id, "active")
        except Exception as exc:
            _set_provider_state(kitchen.provider_id, "skipped", str(exc))
            if mode == "required":
                raise SystemExit(f"[OmniXPU] Kitchen provider activation failed: {exc}")
            _LOG.warning("[OmniXPU] Kitchen provider skipped: %s", exc)

    dynamic_vram, headroom, nvml_pressure = _dynamic_vram_request()
    if dynamic_vram_override is not None:
        dynamic_vram = dynamic_vram_override
    aimdo = providers.get("comfy_aimdo.xpu")
    if aimdo is not None and not dynamic_vram:
        _set_provider_state(aimdo.provider_id, "skipped", "DynamicVRAM is disabled")
        if mode == "required":
            raise SystemExit(
                "[OmniXPU] AIMDO provider activation requires DynamicVRAM"
            )
    elif aimdo is not None:
        try:
            _activate_aimdo(
                aimdo,
                simple_vram_headroom=headroom,
                nvml_pressure=nvml_pressure,
            )
            _set_provider_state(aimdo.provider_id, "active")
        except SystemExit:
            raise
        except Exception as exc:
            _set_provider_state(aimdo.provider_id, "skipped", str(exc))
            if mode == "required":
                raise SystemExit(f"[OmniXPU] AIMDO provider activation failed: {exc}")
            _LOG.warning("[OmniXPU] AIMDO provider skipped: %s", exc)

    active = any(
        state["status"] == "active" for state in _STATE["providers"].values()
    )
    _STATE["status"] = "active" if active else "no-provider-active"
    return get_state()


__all__ = [
    "RuntimeProvider",
    "bootstrap",
    "discover_providers",
    "get_state",
]
