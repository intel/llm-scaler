#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fail on new pip-check issues; permit only documented XPU metadata gaps."""
import argparse
import re
import subprocess
import sys
from importlib import metadata

# The pinned LMCache wheel declares optional backends not enabled by this image.
MISSING = {
    "lmcache": {
        "aiofile", "awscrt", "cufile-python", "google-api-core",
        "google-cloud-bigtable", "nvtx", "pytest", "redis",
    },
}

# Exact official Dockerfile.xpu replacements, not a general conflict exemption.
# Both the required old pin and the installed replacement must match.
TORCH_RUNTIME_REPLACEMENTS = {
    "intel-cmplr-lib-rt": ("2026.1.0", "2026.1.1"),
    "intel-cmplr-lib-ur": ("2026.1.0", "2026.1.1"),
    "intel-cmplr-lic-rt": ("2026.1.0", "2026.1.1"),
    "intel-sycl-rt": ("2026.1.0", "2026.1.1"),
    "dpcpp-cpp-rt": ("2026.1.0", "2026.1.1"),
    "intel-opencl-rt": ("2026.1.0", "2026.1.1"),
    "intel-openmp": ("2026.1.0", "2026.1.1"),
    "oneccl": ("2022.1.1", "2022.1.2"),
    "oneccl-devel": ("2022.1.1", "2022.1.2"),
    "intel-pti": ("1.0.1", "1.1.0"),
}


def allowed_issue(line):
    match = re.fullmatch(r"(\S+) (\S+) requires (\S+), which is not installed\.", line)
    if match:
        owner, version, dependency = match.groups()
        # Exceptions apply to this exact stack, not arbitrary upgraded wheels.
        expected = {"lmcache": "0.4.7"}
        return version == expected.get(owner) and dependency in MISSING.get(owner, ())
    match = re.fullmatch(
        r"torch 2\.14\.0\+xpu has requirement ([\w-]+)==([^,;]+)"
        r'(?:; platform_system == "Linux" and platform_machine == "x86_64")?, '
        r"but you have ([\w-]+) ([^ .]+(?:\.[^ .]+)*)\.", line)
    if match:
        required_name, required_version, installed_name, installed_version = match.groups()
        return (required_name == installed_name and
                TORCH_RUNTIME_REPLACEMENTS.get(required_name) ==
                (required_version, installed_version))
    # PyPI's decord wheel advertises manylinux2010, which pip 26 rejects on this
    # runtime even though importing and loading its native library succeeds.
    return line == "decord 0.6.0 is not supported on this platform"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", required=True, action="append")
    args = parser.parse_args()
    from packaging.requirements import Requirement

    failures = []
    for path in args.constraints:
        with open(path) as constraints:
            for line in constraints:
                line = line.split("#", 1)[0].strip().removesuffix("\\").strip()
                if not line or line.startswith("--hash="):
                    continue
                requirement = Requirement(line)
                try:
                    version = metadata.version(requirement.name)
                except metadata.PackageNotFoundError:
                    failures.append(f"Missing protected runtime dependency: {requirement.name}")
                    continue
                if version not in requirement.specifier:
                    failures.append(f"Protected runtime dependency: {requirement}, installed {version}")

    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            check=False, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line == "No broken requirements found.":
            continue
        if allowed_issue(line):
            print(f"Documented XPU metadata exception: {line}")
        else:
            failures.append(line)
    if result.returncode not in (0, 1) or result.stderr.strip():
        failures.append(f"pip check failed: {result.stderr.strip()}")
    if result.returncode and not result.stdout.strip():
        failures.append("pip check failed without diagnostics")
    if failures:
        raise SystemExit("Runtime dependency validation failed:\n" + "\n".join(failures))
    # Verify the native library behind the platform metadata exception.
    import decord  # noqa: F401
    print("Runtime dependency validation passed")


if __name__ == "__main__":
    main()
