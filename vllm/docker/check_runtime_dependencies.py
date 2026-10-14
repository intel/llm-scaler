#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fail on new pip-check issues; permit only documented XPU metadata gaps."""
import argparse
import importlib.metadata as metadata
import re
import subprocess
import sys

# Missing requirements ONLY: a present but incompatible version is never exempt.
# System oneCCL and triton-xpu replace these Python distributions. The pinned
# LMCache wheel still declares backends not enabled by this MP/filesystem image.
MISSING = {
    "torch": {"oneccl", "oneccl-devel"},
    "xgrammar": {"triton"},
    "lmcache": {
        "aiofile", "awscrt", "cufile-python", "google-api-core",
        "google-cloud-bigtable", "nvtx", "pytest", "redis",
    },
}


def allowed_issue(line):
    match = re.fullmatch(r"(\S+) (\S+) requires (\S+), which is not installed\.", line)
    if match:
        owner, version, dependency = match.groups()
        # Exceptions apply to this exact stack, not arbitrary upgraded wheels.
        expected = {"torch": "2.12.0+xpu", "lmcache": "0.4.7", "xgrammar": "0.2.8"}
        return version == expected.get(owner) and dependency in MISSING.get(owner, ())
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
                            capture_output=True, text=True)
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
