#!/usr/bin/env python3
"""Acceptance checks for a built ComfyUI-focused Omni image.

Run this inside the final container, after exposing ``/dev/dri``.  These checks
intentionally live outside the Dockerfile: image construction has no XPU
device and should not encode release-policy assertions in cached build layers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys


REQUIRED_KITCHEN_CAPABILITIES = {
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "int8_linear",
    "mm_int8",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
}


def require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


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
    import omni_xpu_kernel
    from omni_xpu_kernel import _version as kernel_version

    expected_image = os.environ["OMNI_IMAGE_VERSION"]
    expected_target = os.environ["OMNI_IMAGE_XPU_TARGET"]
    expected_kitchen = os.environ["OMNI_COMFY_KITCHEN_VERSION"]
    source_revision = os.environ["OMNI_LLM_SCALER_SOURCE_REVISION"]
    source_dirty = os.environ["OMNI_LLM_SCALER_SOURCE_DIRTY"]

    require_equal("image version", kernel_version.__image_version__, expected_image)
    require_equal("kernel package target", omni_xpu_kernel.__xpu_target__, expected_target)
    require_equal("kernel AOT target", omni_xpu_kernel.core_aot_target(), expected_target)
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise RuntimeError(
            "llm-scaler source revision must be a full 40-character Git commit, "
            f"got {source_revision!r}"
        )
    if not args.allow_dirty_source:
        require_equal("llm-scaler source dirty", source_dirty, "false")
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
        "kernel Torch ABI",
        omni_xpu_kernel.__torch_version__,
        torch.__version__.split("+", 1)[0],
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
        f"xpu={device_name!r}, kitchen_capabilities={len(capabilities)}"
    )


if __name__ == "__main__":
    main()
