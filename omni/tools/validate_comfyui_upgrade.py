#!/usr/bin/env python3
"""Validate ComfyUI/official-package upgrades in a disposable Omni container.

This check intentionally mutates the running container's writable layer. Run
it only in a disposable ``docker run --rm`` container built from the focused
image. It proves that normal ComfyUI requirements and official Kitchen/AIMDO
installation cannot overwrite the separately named XPU runtime providers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path


COMFYUI_ROOT = Path("/llm/ComfyUI")
UPDATE_SCRIPT = Path("/llm/tools/update_comfyui.sh")
RUNTIME_CONSTRAINTS = Path("/llm/manifests/omni-runtime-constraints.txt")
PROVIDER_DISTRIBUTIONS = (
    "comfy-aimdo-xpu-runtime",
    "comfy-kitchen-xpu-runtime",
)
OFFICIAL_DISTRIBUTIONS = ("comfy-aimdo", "comfy-kitchen")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution_snapshot(name: str) -> dict[str, object]:
    """Hash every RECORD-owned file of an installed distribution."""

    distribution = importlib.metadata.distribution(name)
    files = distribution.files
    if not files:
        raise RuntimeError(f"distribution {name!r} has no installed file record")
    owned: dict[str, str] = {}
    for relative in sorted(files, key=str):
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            raise RuntimeError(f"{name} owned file is missing: {path}")
        owned[str(relative)] = _sha256(path)
    return {
        "name": str(distribution.metadata["Name"]),
        "version": str(distribution.version),
        "files": owned,
    }


def provider_snapshot() -> dict[str, dict[str, object]]:
    return {name: distribution_snapshot(name) for name in PROVIDER_DISTRIBUTIONS}


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=environment)


def require_runtime_constraints() -> None:
    if os.environ.get("PIP_CONSTRAINT") != str(RUNTIME_CONSTRAINTS):
        raise RuntimeError(
            "PIP_CONSTRAINT must protect the focused image runtime during upgrades"
        )
    if not RUNTIME_CONSTRAINTS.is_file():
        raise RuntimeError(f"runtime constraints are missing: {RUNTIME_CONSTRAINTS}")
    constraints = RUNTIME_CONSTRAINTS.read_text(encoding="utf-8")
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "omni_xpu_kernel",
        *PROVIDER_DISTRIBUTIONS,
    ):
        normalized = name.replace("-", "[-_]")
        if not re.search(rf"^{normalized}==[^=\s]+$", constraints, re.MULTILINE | re.I):
            raise RuntimeError(f"runtime constraint is missing for {name}")


_ACTIVATION_PROBE = r"""
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import sys

import comfy_aimdo.control as official_control

official_preinit = official_control.init()
if "torch" in sys.modules:
    raise RuntimeError("official AIMDO imported Torch before provider bootstrap")
runtime_path = pathlib.Path(
    "/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py"
)
spec = importlib.util.spec_from_file_location(
    "_comfyui_omnixpu_runtime_bootstrap", runtime_path
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load runtime bootstrap from {runtime_path}")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)
providers, errors = runtime.discover_providers()
probe_mode = os.environ["OMNIXPU_UPGRADE_PROBE_MODE"]
if probe_mode == "required" and errors:
    raise RuntimeError("provider discovery failed: " + "; ".join(errors))
state = runtime.bootstrap(
    providers_override=providers,
    dynamic_vram_override=True,
)
expected_provider_ids = {"comfy_aimdo.xpu", "comfy_kitchen.xpu"}
if probe_mode == "required" and set(providers) != expected_provider_ids:
    raise RuntimeError(f"required provider set is incomplete: {providers}")
for provider_id in providers:
    if state["providers"].get(provider_id, {}).get("status") != "active":
        raise RuntimeError(f"provider did not activate: {provider_id}: {state}")
if sys.modules.get("comfy_aimdo.control") is not official_control:
    raise RuntimeError("AIMDO provider replaced the official module object")

import comfy_kitchen

aimdo_provider = providers.get("comfy_aimdo.xpu")
kitchen_provider = providers.get("comfy_kitchen.xpu")
aimdo_path = pathlib.Path(official_control.__file__).resolve()
kitchen_path = pathlib.Path(comfy_kitchen.__file__).resolve()
if aimdo_provider is None:
    if not any("comfy_aimdo.xpu" in error and "incompatible" in error for error in errors):
        raise RuntimeError(f"AIMDO provider vanished without incompatibility: {errors}")
    if "comfy_aimdo_xpu_runtime" in str(aimdo_path):
        raise RuntimeError("incompatible AIMDO provider remained routed")
elif not aimdo_path.is_relative_to(aimdo_provider.canonical_root):
    raise RuntimeError("AIMDO control did not route to its provider")
if kitchen_provider is None:
    if not any("comfy_kitchen.xpu" in error and "incompatible" in error for error in errors):
        raise RuntimeError(f"Kitchen provider vanished without incompatibility: {errors}")
    if "comfy_kitchen_xpu_runtime" in str(kitchen_path):
        raise RuntimeError("incompatible Kitchen provider remained routed")
elif not kitchen_path.is_relative_to(kitchen_provider.canonical_root):
    raise RuntimeError("Kitchen did not route to its provider")
print(json.dumps({
    "official": {
        "comfy-aimdo": importlib.metadata.version("comfy-aimdo"),
        "comfy-kitchen": importlib.metadata.version("comfy-kitchen"),
    },
    "providers": {
        key: value.manifest["source"]["revision"]
        for key, value in sorted(providers.items())
    },
    "state": state,
    "errors": errors,
    "official_preinit": bool(official_preinit),
}, sort_keys=True))
"""


def activation_probe(mode: str) -> dict[str, object]:
    if mode not in {"auto", "required"}:
        raise ValueError(f"invalid activation probe mode: {mode}")
    environment = os.environ.copy()
    environment.update(
        {
            "OMNIXPU_PROVIDER_BOOTSTRAP": mode,
            "OMNIXPU_UPGRADE_PROBE_MODE": mode,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/tmp/omnixpu-upgrade-pycache",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", _ACTIVATION_PROBE],
        check=True,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("runtime provider activation probe produced no output")
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-revision", required=True)
    parser.add_argument(
        "--kitchen-version",
        default=os.environ.get("OMNI_COMFY_KITCHEN_VERSION"),
    )
    parser.add_argument(
        "--aimdo-version",
        default=os.environ.get("OMNI_COMFY_AIMDO_VERSION"),
    )
    parser.add_argument("--upgrade-kitchen-version", required=True)
    parser.add_argument("--upgrade-aimdo-version", required=True)
    args = parser.parse_args()

    if not _REVISION_PATTERN.fullmatch(args.comfyui_revision):
        parser.error("--comfyui-revision must be a full lowercase Git commit")
    if not args.kitchen_version or not args.aimdo_version:
        parser.error("official Kitchen and AIMDO versions must be explicit")
    if not COMFYUI_ROOT.joinpath(".git").exists() or not UPDATE_SCRIPT.is_file():
        raise RuntimeError("this check must run inside the focused ComfyUI image")

    require_runtime_constraints()
    before = provider_snapshot()
    official_before = {
        name: importlib.metadata.version(name) for name in OFFICIAL_DISTRIBUTIONS
    }

    update_environment = os.environ.copy()
    update_environment["COMFYUI_UPGRADE_REF"] = args.comfyui_revision
    run(["bash", str(UPDATE_SCRIPT)], environment=update_environment)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            f"comfy-kitchen=={args.upgrade_kitchen_version}",
            f"comfy-aimdo=={args.upgrade_aimdo_version}",
        ]
    )

    after_upgrade = provider_snapshot()
    if after_upgrade != before:
        raise RuntimeError(
            "official package or ComfyUI upgrade changed provider-owned files"
        )
    upgraded_official = {
        name: importlib.metadata.version(name) for name in OFFICIAL_DISTRIBUTIONS
    }
    expected_upgraded_official = {
        "comfy-aimdo": args.upgrade_aimdo_version,
        "comfy-kitchen": args.upgrade_kitchen_version,
    }
    if upgraded_official != expected_upgraded_official:
        raise RuntimeError(
            f"official upgrade versions changed unexpectedly: {upgraded_official}"
        )
    auto_probe = activation_probe("auto")
    if provider_snapshot() != before:
        raise RuntimeError("auto fallback probe changed provider-owned files")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            f"comfy-kitchen=={args.kitchen_version}",
            f"comfy-aimdo=={args.aimdo_version}",
        ]
    )

    after_restore = provider_snapshot()
    if after_restore != before:
        raise RuntimeError(
            "restoring compatible official packages changed provider-owned files"
        )
    official_after = {
        name: importlib.metadata.version(name) for name in OFFICIAL_DISTRIBUTIONS
    }
    expected_official = {
        "comfy-aimdo": args.aimdo_version,
        "comfy-kitchen": args.kitchen_version,
    }
    if official_after != expected_official:
        raise RuntimeError(
            f"official package versions changed unexpectedly: {official_after}"
        )
    actual_comfyui_revision = subprocess.check_output(
        ["git", "-C", str(COMFYUI_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_comfyui_revision != args.comfyui_revision:
        raise RuntimeError(
            f"ComfyUI upgrade selected {actual_comfyui_revision}, "
            f"expected {args.comfyui_revision}"
        )

    required_probe = activation_probe("required")
    after_probe = provider_snapshot()
    if after_probe != before:
        raise RuntimeError("provider activation changed provider-owned files")
    run([sys.executable, "-m", "pip", "check"])

    print(
        json.dumps(
            {
                "status": "passed",
                "comfyui_revision": actual_comfyui_revision,
                "official_before": official_before,
                "official_upgraded": upgraded_official,
                "official_after": official_after,
                "provider_versions": {
                    name: snapshot["version"] for name, snapshot in before.items()
                },
                "provider_file_counts": {
                    name: len(snapshot["files"]) for name, snapshot in before.items()
                },
                "auto_activation": auto_probe,
                "required_activation": required_probe,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
