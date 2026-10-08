"""llm-scaler release policy for its optional AIMDO native-owner integration.

Runtime admission works before Torch import. The optional post-build command
loads the sidecar's release export without installing its allocator. Release
versions are the compatibility identity; Torch library bytes are not inspected.
"""
from __future__ import annotations

import argparse
import ctypes
from importlib import metadata
from pathlib import Path


SUPPORTED_TORCH_RELEASES = frozenset({"2.14.0+xpu"})


class UnsupportedRelease(RuntimeError):
    pass


def validate_torch_release(version: str) -> None:
    if version not in SUPPORTED_TORCH_RELEASES:
        supported = ", ".join(sorted(SUPPORTED_TORCH_RELEASES))
        raise UnsupportedRelease(
            f"llm-scaler AIMDO diagnostic supports Torch release {supported}; got {version!r}"
        )


def validate_native_owner_release(
    path: str | Path, *, installed_version: str, declared_version: str
) -> None:
    """Check build provenance before packaging; Torch must already be imported."""
    validate_torch_release(installed_version)
    if declared_version != installed_version:
        raise UnsupportedRelease(
            f"AIMDO provider declares Torch release {declared_version!r}, "
            f"but the build environment has {installed_version!r}"
        )
    try:
        library = ctypes.CDLL(str(Path(path).resolve()))
        version = library.aimdo_full_proxy_torch_version
        version.argtypes = []
        version.restype = ctypes.c_char_p
        raw = version()
        built_version = raw.decode("utf-8") if raw is not None else None
    except (OSError, AttributeError, UnicodeDecodeError) as error:
        raise UnsupportedRelease(
            f"Cannot read AIMDO native-owner Torch release from {str(path)!r}: {error}"
        ) from error
    if built_version != installed_version:
        raise UnsupportedRelease(
            f"AIMDO native-owner was built for Torch release {built_version!r}; "
            f"the build environment and provider require {installed_version!r}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-version", help="declared build release; default: installed Torch metadata")
    parser.add_argument("--native-owner", type=Path, help="post-build sidecar to check before packaging")
    args = parser.parse_args()
    try:
        declared_version = args.torch_version or metadata.version("torch")
        validate_torch_release(declared_version)
        if args.native_owner is not None:
            import torch

            validate_native_owner_release(
                args.native_owner,
                installed_version=str(torch.__version__),
                declared_version=declared_version,
            )
    except (UnsupportedRelease, metadata.PackageNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
