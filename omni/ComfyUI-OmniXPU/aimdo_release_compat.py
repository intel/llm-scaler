"""llm-scaler release policy for its optional AIMDO native-owner integration.

Usable before Torch import and as a build-time command. Release versions are
the compatibility identity; this module does not inspect Torch library bytes.
"""
from __future__ import annotations

import argparse
from importlib import metadata


SUPPORTED_TORCH_RELEASES = frozenset({"2.14.0+xpu"})


class UnsupportedRelease(RuntimeError):
    pass


def validate_torch_release(version: str) -> None:
    if version not in SUPPORTED_TORCH_RELEASES:
        supported = ", ".join(sorted(SUPPORTED_TORCH_RELEASES))
        raise UnsupportedRelease(
            f"llm-scaler AIMDO diagnostic supports Torch release {supported}; got {version!r}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-version", help="declared build release; default: installed Torch metadata")
    args = parser.parse_args()
    try:
        validate_torch_release(args.torch_version or metadata.version("torch"))
    except (UnsupportedRelease, metadata.PackageNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
