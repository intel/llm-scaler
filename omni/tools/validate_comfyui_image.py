#!/usr/bin/env python3
"""Acceptance checks for a built ComfyUI-focused Omni image.

Run this inside the final container, after exposing ``/dev/dri``.  These checks
intentionally live outside the Dockerfile: image construction has no XPU
device and should not encode release-policy assertions in cached build layers.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

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
    "Kitchen": (
        Path("/llm/comfy-kitchen-xpu"),
        "OMNI_COMFY_KITCHEN_REVISION",
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

    import torch
    import comfy_kitchen
    import nunchaku_torch
    import omni_xpu_kernel
    from omni_xpu_kernel import _version as kernel_version
    from omni_xpu_kernel import gguf as omni_gguf

    expected_image = os.environ["OMNI_IMAGE_VERSION"]
    expected_target = os.environ["OMNI_IMAGE_XPU_TARGET"]
    expected_kitchen = os.environ["OMNI_COMFY_KITCHEN_VERSION"]
    expected_nunchaku = os.environ["OMNI_COMFY_NUNCHAKU_VERSION"]
    source_revision = os.environ["OMNI_LLM_SCALER_SOURCE_REVISION"]
    source_dirty = os.environ["OMNI_LLM_SCALER_SOURCE_DIRTY"]

    require_equal("image version", kernel_version.__image_version__, expected_image)
    require_equal("kernel package target", omni_xpu_kernel.__xpu_target__, expected_target)
    require_equal("kernel AOT target", omni_xpu_kernel.core_aot_target(), expected_target)
    require_full_revision("llm-scaler source revision", source_revision)
    if not args.allow_dirty_source:
        require_equal("llm-scaler source dirty", source_dirty, "false")
    for label, (path, environment_variable) in PINNED_CHECKOUTS.items():
        require_checkout_revision(label, path, os.environ[environment_variable])

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

    device_name = torch.xpu.get_device_name(0)
    print(
        "ComfyUI image acceptance passed: "
        f"image={expected_image}, target={expected_target}, "
        f"source={source_revision[:12]}, dirty={source_dirty}, "
        f"torch={torch.__version__}, kitchen={expected_kitchen}, "
        f"gguf={dependency_versions['gguf']}, nunchaku={expected_nunchaku}, "
        f"xpu={device_name!r}, kitchen_capabilities={len(capabilities)}"
    )


if __name__ == "__main__":
    main()
