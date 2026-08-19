#!/usr/bin/env python3
"""Acceptance checks for a built ComfyUI-focused Omni image.

Run this inside the final container, after exposing ``/dev/dri``.  These checks
intentionally live outside the Dockerfile: image construction has no XPU
device and should not encode release-policy assertions in cached build layers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from runpy import run_path

from packaging.version import Version


REQUIRED_KITCHEN_CAPABILITIES = {
    "dequantize_gguf",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "int8_linear",
    "mm_int8",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "svdquant_w4a16_linear",
}

PINNED_CHECKOUTS = {
    "ComfyUI": (
        Path("/llm/ComfyUI"),
        "OMNI_COMFYUI_REVISION",
    ),
    "Kitchen": (
        Path("/llm/comfy-kitchen-xpu"),
        "OMNI_COMFY_KITCHEN_REVISION",
    ),
    "Comfy AIMDO": (
        Path("/llm/comfy-aimdo-xpu"),
        "OMNI_COMFY_AIMDO_REVISION",
    ),
    "GGUF custom node": (
        Path("/llm/ComfyUI/custom_nodes/ComfyUI-GGUF-XPU"),
        "OMNI_COMFY_GGUF_REVISION",
    ),
    "combined Nunchaku custom node/runtime": (
        Path("/llm/ComfyUI/custom_nodes/ComfyUI-nunchaku-XPU"),
        "OMNI_COMFY_NUNCHAKU_REVISION",
    ),
}

GGUF_DEPENDENCIES = {
    "gguf": "gguf",
    "sentencepiece": "sentencepiece",
    "protobuf": "google.protobuf",
}

COMFYUI_PACKAGE_ENVIRONMENT = {
    "comfyui-frontend-package": "OMNI_COMFYUI_FRONTEND_VERSION",
    "comfyui-workflow-templates": "OMNI_COMFYUI_WORKFLOW_TEMPLATES_VERSION",
    "comfyui-manager": "OMNI_COMFYUI_MANAGER_VERSION",
}

REQUIRED_MINIMAX_H3_TEMPLATES = {
    "api_minimax_h3_flf2v.json",
    "api_minimax_h3_r2v.json",
    "api_minimax_h3_t2v.json",
    "video_minimax_h3_i2v.json",
    "video_minimax_h3_r2v.json",
    "video_minimax_h3_t2v.json",
}
PINNED_MINIMAX_H3_TEMPLATE_HASHES = {
    "video_minimax_h3_t2v.json": (
        "31ab33fdb053a7834cc866bd7aa08b887518fc656e4a796c89779c6b5e1786e6"
    ),
}

COMFYUI_ROOT = Path("/llm/ComfyUI")
COMFYUI_DATABASE_DIRECTORY = COMFYUI_ROOT / "user"
AIMDO_SOURCE_ROOT = Path("/llm/comfy-aimdo-xpu")
AIMDO_REQUIRED_XPU_TESTS = {
    "test_xpu_backend.py",
    "test_xpu_comfyui_opt_in.py",
}
ONEDNN_PROVENANCE_MANIFEST = Path("/llm/manifests/onednn-runtime.env")
ONEDNN_PATCH = Path(
    "/llm/patches/onednn-v3.11.2-enable-bf16-int4-dequantization.patch"
)
ONEDNN_RUNTIME_LIBRARY = Path("/opt/venv/lib/libdnnl.so.3.11")
ONEDNN_RUNTIME_LINK = Path("/opt/venv/lib/libdnnl.so.3")
ONEDNN_PROVENANCE_FIELDS = {
    "schema_version",
    "package_version",
    "source_repository",
    "source_revision",
    "patch_sha256",
    "library_path",
    "library_sha256",
}


def require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def require_full_revision(label: str, revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(
            f"{label} must be a full 40-character Git commit, got {revision!r}"
        )


def require_checkout_revision(label: str, path: Path, expected: str) -> None:
    require_full_revision(f"{label} revision", expected)
    if not path.is_dir():
        raise RuntimeError(f"{label} checkout is missing: {path}")
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    require_equal(f"{label} checkout revision", completed.stdout.strip(), expected)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_onednn_runtime_provenance(
    *,
    expected_package_version: str,
    expected_source_repository: str,
    expected_source_revision: str,
    expected_patch_sha256: str,
    manifest_path: Path = ONEDNN_PROVENANCE_MANIFEST,
    patch_path: Path = ONEDNN_PATCH,
    runtime_library: Path = ONEDNN_RUNTIME_LIBRARY,
    runtime_link: Path | None = ONEDNN_RUNTIME_LINK,
) -> dict[str, str]:
    """Require the patched oneDNN source, patch, and DSO to match the image."""

    require_full_revision("oneDNN source revision", expected_source_revision)
    if re.fullmatch(r"[0-9a-f]{64}", expected_patch_sha256) is None:
        raise RuntimeError(
            "oneDNN patch SHA256 must be 64 lowercase hexadecimal characters, "
            f"got {expected_patch_sha256!r}"
        )
    if not manifest_path.is_file():
        raise RuntimeError(f"oneDNN provenance manifest is missing: {manifest_path}")

    provenance = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        key, separator, value = raw_line.partition("=")
        if not separator or not key or not value:
            raise RuntimeError(
                f"invalid oneDNN provenance line {line_number}: {raw_line!r}"
            )
        if key in provenance:
            raise RuntimeError(f"duplicate oneDNN provenance field: {key}")
        provenance[key] = value

    require_equal(
        "oneDNN provenance fields",
        ",".join(sorted(provenance)),
        ",".join(sorted(ONEDNN_PROVENANCE_FIELDS)),
    )
    expected_values = {
        "schema_version": "1",
        "package_version": expected_package_version,
        "source_repository": expected_source_repository,
        "source_revision": expected_source_revision,
        "patch_sha256": expected_patch_sha256,
        "library_path": str(runtime_library),
    }
    for key, expected in expected_values.items():
        require_equal(f"oneDNN provenance {key}", provenance[key], expected)

    if not patch_path.is_file():
        raise RuntimeError(f"oneDNN patch is missing: {patch_path}")
    require_equal(
        "oneDNN patch SHA256",
        file_sha256(patch_path),
        expected_patch_sha256,
    )
    if not runtime_library.is_file():
        raise RuntimeError(f"patched oneDNN runtime is missing: {runtime_library}")
    require_equal(
        "oneDNN runtime SHA256",
        file_sha256(runtime_library),
        provenance["library_sha256"],
    )
    if runtime_link is not None:
        if not runtime_link.is_symlink():
            raise RuntimeError(f"oneDNN runtime link is missing: {runtime_link}")
        require_equal(
            "oneDNN runtime link target",
            str(runtime_link.resolve(strict=True)),
            str(runtime_library.resolve(strict=True)),
        )
    return provenance


def add_comfyui_to_import_path() -> None:
    """Make integrated packages importable from the runner's /tmp cwd."""
    comfyui_root = str(COMFYUI_ROOT)
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)


def require_torch_matched_oneapi_runtime() -> dict[str, list[str]]:
    """Require the loaded SYCL/UR libraries to come from the Torch venv."""
    loaded = Path("/proc/self/maps").read_text(encoding="utf-8")
    expected_root = Path("/opt/venv/lib")
    resolved = {}
    for library in ("libsycl.so", "libur_loader.so"):
        paths = sorted(
            {
                line.rsplit(maxsplit=1)[-1]
                for line in loaded.splitlines()
                if library in line and line.rsplit(maxsplit=1)[-1].startswith("/")
            }
        )
        if not paths:
            raise RuntimeError(f"{library} is not loaded")
        unexpected = [
            path
            for path in paths
            if not Path(path).is_relative_to(expected_root)
        ]
        if unexpected:
            raise RuntimeError(
                f"{library} must load from {expected_root}, got: "
                + ", ".join(unexpected)
            )
        resolved[library] = paths
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-no-xpu",
        action="store_true",
        help="check package identity and imports without requiring a GPU",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow a development image built from a dirty llm-scaler worktree",
    )
    args = parser.parse_args()

    add_comfyui_to_import_path()

    import torch
    import comfy_aimdo.control
    import comfy_kitchen
    import comfyui_manager
    import nunchaku_torch
    import omni_xpu_kernel
    import torchaudio
    from omni_xpu_kernel import _version as kernel_version
    from omni_xpu_kernel import gguf as omni_gguf

    expected_image = os.environ["OMNI_IMAGE_VERSION"]
    expected_target = os.environ["OMNI_IMAGE_XPU_TARGET"]
    expected_torch = os.environ["OMNI_TORCH_VERSION"]
    expected_torchvision = os.environ["OMNI_TORCHVISION_VERSION"]
    expected_torchaudio = os.environ["OMNI_TORCHAUDIO_VERSION"]
    expected_onednn = os.environ["OMNI_ONEDNN_VERSION"]
    expected_onednn_source_repository = os.environ[
        "OMNI_ONEDNN_SOURCE_REPOSITORY"
    ]
    expected_onednn_source_revision = os.environ[
        "OMNI_ONEDNN_SOURCE_REVISION"
    ]
    expected_onednn_patch_sha256 = os.environ["OMNI_ONEDNN_PATCH_SHA256"]
    expected_comfyui = os.environ["OMNI_COMFYUI_VERSION"]
    expected_kitchen = os.environ["OMNI_COMFY_KITCHEN_VERSION"]
    expected_aimdo = os.environ["OMNI_COMFY_AIMDO_VERSION"]
    expected_aimdo_revision = os.environ["OMNI_COMFY_AIMDO_REVISION"]
    expected_nunchaku = os.environ["OMNI_COMFY_NUNCHAKU_VERSION"]
    source_revision = os.environ["OMNI_LLM_SCALER_SOURCE_REVISION"]
    source_dirty = os.environ["OMNI_LLM_SCALER_SOURCE_DIRTY"]

    require_equal("image version", kernel_version.__image_version__, expected_image)
    require_equal("Torch version", torch.__version__, expected_torch)
    require_equal(
        "torchvision distribution version",
        importlib.metadata.version("torchvision"),
        expected_torchvision,
    )
    require_equal(
        "torchaudio distribution version",
        importlib.metadata.version("torchaudio"),
        expected_torchaudio,
    )
    require_equal(
        "oneDNN distribution version",
        importlib.metadata.version("onednn"),
        expected_onednn,
    )
    onednn_provenance = require_onednn_runtime_provenance(
        expected_package_version=expected_onednn,
        expected_source_repository=expected_onednn_source_repository,
        expected_source_revision=expected_onednn_source_revision,
        expected_patch_sha256=expected_onednn_patch_sha256,
    )
    oneapi_runtime_libraries = require_torch_matched_oneapi_runtime()
    require_equal("kernel package target", omni_xpu_kernel.__xpu_target__, expected_target)
    require_equal("kernel AOT target", omni_xpu_kernel.core_aot_target(), expected_target)
    require_full_revision("llm-scaler source revision", source_revision)
    if not args.allow_dirty_source:
        require_equal("llm-scaler source dirty", source_dirty, "false")
    for label, (path, environment_variable) in PINNED_CHECKOUTS.items():
        require_checkout_revision(label, path, os.environ[environment_variable])

    require_equal(
        "Comfy AIMDO distribution version",
        importlib.metadata.version("comfy-aimdo"),
        expected_aimdo,
    )
    require_equal(
        "Comfy AIMDO detected backend",
        str(comfy_aimdo.control.detect_vendor()),
        "xpu",
    )
    aimdo_xpu_library = Path(comfy_aimdo.control.__file__).with_name(
        "aimdo_xpu.so"
    )
    if not aimdo_xpu_library.is_file():
        raise RuntimeError(
            f"Comfy AIMDO XPU library is missing: {aimdo_xpu_library}"
        )
    missing_aimdo_tests = sorted(
        name
        for name in AIMDO_REQUIRED_XPU_TESTS
        if not (AIMDO_SOURCE_ROOT / "tests" / name).is_file()
    )
    if missing_aimdo_tests:
        raise RuntimeError(
            "Comfy AIMDO installed-image XPU tests are missing: "
            + ", ".join(missing_aimdo_tests)
        )

    comfyui_version = run_path("/llm/ComfyUI/comfyui_version.py")["__version__"]
    require_equal("ComfyUI version", comfyui_version, expected_comfyui)
    require_equal(
        "Kitchen module version",
        comfy_kitchen.__version__,
        expected_kitchen,
    )
    require_equal(
        "Kitchen distribution version",
        importlib.metadata.version("comfy-kitchen"),
        expected_kitchen,
    )
    comfyui_dependency_versions = {}
    for distribution_name, environment_variable in (
        COMFYUI_PACKAGE_ENVIRONMENT.items()
    ):
        actual = importlib.metadata.version(distribution_name)
        expected = os.environ[environment_variable]
        require_equal(f"{distribution_name} version", actual, expected)
        comfyui_dependency_versions[distribution_name] = actual
    if not comfyui_manager.__file__:
        raise RuntimeError("comfyui_manager package has no importable source")

    template_distribution = importlib.metadata.distribution(
        "comfyui-workflow-templates-json"
    )
    template_root = Path(
        template_distribution.locate_file(
            "comfyui_workflow_templates_json/templates"
        )
    )
    missing_templates = sorted(
        name for name in REQUIRED_MINIMAX_H3_TEMPLATES
        if not (template_root / name).is_file()
    )
    if missing_templates:
        raise RuntimeError(
            "ComfyUI workflow template package is missing MiniMax H3 files: "
            + ", ".join(missing_templates)
        )
    h3_template_hashes = {}
    for name in sorted(REQUIRED_MINIMAX_H3_TEMPLATES):
        path = template_root / name
        json.loads(path.read_text(encoding="utf-8"))
        h3_template_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected_hash in PINNED_MINIMAX_H3_TEMPLATE_HASHES.items():
        require_equal(
            f"MiniMax H3 official template hash ({name})",
            h3_template_hashes[name],
            expected_hash,
        )

    dependency_manifest = Path("/llm/manifests/comfyui-python-freeze.txt")
    if not dependency_manifest.is_file() or not dependency_manifest.read_text(
        encoding="utf-8"
    ).strip():
        raise RuntimeError(
            "complete ComfyUI Python dependency manifest is missing or empty"
        )
    if not COMFYUI_DATABASE_DIRECTORY.is_dir():
        raise RuntimeError(
            "ComfyUI default database directory is missing: "
            f"{COMFYUI_DATABASE_DIRECTORY}"
        )
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".comfyui-db-contract-",
            dir=COMFYUI_DATABASE_DIRECTORY,
        ):
            pass
    except OSError as error:
        raise RuntimeError(
            "ComfyUI default database directory is not writable: "
            f"{COMFYUI_DATABASE_DIRECTORY}"
        ) from error
    require_equal(
        "ComfyUI-nunchaku-XPU distribution version",
        importlib.metadata.version("ComfyUI-nunchaku-XPU"),
        expected_nunchaku,
    )
    nunchaku_distribution = importlib.metadata.distribution("ComfyUI-nunchaku-XPU")
    if not any(
        str(file).startswith("nunchaku_torch/")
        for file in (nunchaku_distribution.files or ())
    ):
        raise RuntimeError(
            "ComfyUI-nunchaku-XPU distribution does not contain the bundled "
            "nunchaku_torch runtime"
        )
    try:
        standalone_nunchaku = importlib.metadata.version("nunchaku-torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "standalone nunchaku-torch distribution must be absent, got "
            f"{standalone_nunchaku!r}"
        )
    if "/llm/nunchaku-torch/" in nunchaku_torch.__file__:
        raise RuntimeError(
            "nunchaku_torch must come from the combined custom-node "
            f"distribution, got {nunchaku_torch.__file__!r}"
        )
    if Path("/llm/nunchaku-torch").exists():
        raise RuntimeError("standalone /llm/nunchaku-torch checkout must be absent")
    require_equal(
        "kernel Torch ABI",
        omni_xpu_kernel.__torch_version__,
        torch.__version__.split("+", 1)[0],
    )
    for function_name in (
        "dequantize_q4_0",
        "dequantize_q4_1",
        "dequantize_q8_0",
        "dequantize_q4_k",
        "dequantize_q6_k",
    ):
        if not callable(getattr(omni_gguf, function_name, None)):
            raise RuntimeError(
                f"Omni GGUF API is missing callable {function_name!r}"
            )

    dependency_versions = {}
    for distribution_name, module_name in GGUF_DEPENDENCIES.items():
        dependency_versions[distribution_name] = importlib.metadata.version(
            distribution_name
        )
        importlib.import_module(module_name)
    if Version(dependency_versions["gguf"]) < Version("0.13.0"):
        raise RuntimeError(
            "GGUF dependency must satisfy gguf>=0.13.0, got "
            f"{dependency_versions['gguf']!r}"
        )

    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
    )

    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    if not xpu_available:
        if args.allow_no_xpu:
            print("Package checks passed; XPU checks skipped (--allow-no-xpu).")
            return
        raise RuntimeError(
            "PyTorch XPU is unavailable; run the container with --device=/dev/dri"
        )

    backend = comfy_kitchen.list_backends()["xpu"]
    if not backend["available"]:
        raise RuntimeError(f"Kitchen XPU backend is unavailable: {backend}")

    capabilities = set(backend["capabilities"])
    missing = REQUIRED_KITCHEN_CAPABILITIES - capabilities
    if missing:
        raise RuntimeError(
            "Kitchen XPU backend is missing required capabilities: "
            + ", ".join(sorted(missing))
        )

    audio = torch.linspace(-1.0, 1.0, 1600, device="xpu").unsqueeze(0)
    resampled_audio = torchaudio.functional.resample(audio, 16000, 24000)
    torch.xpu.synchronize()
    if tuple(resampled_audio.shape) != (1, 2400):
        raise RuntimeError(
            "torchaudio XPU resample returned unexpected shape: "
            f"{tuple(resampled_audio.shape)}"
        )
    if not bool(torch.isfinite(resampled_audio).all().item()):
        raise RuntimeError("torchaudio XPU resample returned non-finite values")

    device_name = torch.xpu.get_device_name(0)
    print(
        "ComfyUI image acceptance passed: "
        f"image={expected_image}, target={expected_target}, "
        f"source={source_revision[:12]}, dirty={source_dirty}, "
        f"base={os.environ['OMNI_BASE_IMAGE']}, torch={torch.__version__}, "
        f"torchvision={expected_torchvision}, torchaudio={expected_torchaudio}, "
        f"onednn={expected_onednn}@{expected_onednn_source_revision[:12]}, "
        f"onednn_dso={onednn_provenance['library_sha256'][:12]}, "
        f"oneapi_runtime={oneapi_runtime_libraries}, "
        f"comfyui={expected_comfyui}, "
        f"frontend={comfyui_dependency_versions['comfyui-frontend-package']}, "
        "templates="
        f"{comfyui_dependency_versions['comfyui-workflow-templates']}, "
        f"manager={comfyui_dependency_versions['comfyui-manager']}, "
        f"kitchen={expected_kitchen}, "
        f"aimdo={expected_aimdo}@{expected_aimdo_revision[:12]}, "
        f"gguf={dependency_versions['gguf']}, nunchaku={expected_nunchaku}, "
        f"xpu={device_name!r}, kitchen_capabilities={len(capabilities)}, "
        f"h3_templates={len(h3_template_hashes)}"
    )


if __name__ == "__main__":
    main()
