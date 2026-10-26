#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Describe, record and verify the native wheel set built by docker/Dockerfile.

inputs: print the build inputs (source revisions, patch and source hashes,
        runtime lock hash, Python/torch/arch) as JSON.
create: write manifest.json for a wheel directory.
verify: reject a wheel directory that does not match this build's inputs.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Distribution names of the wheels built by this Dockerfile, one wheel each.
DISTRIBUTIONS = (
    "custom_esimd_kernels_vllm",
    "kvweave",
    "lmcache",
    "vllm",
    "vllm_xpu_kernels",
)
REVISION_ARGS = (
    "VLLM_VERSION",
    "VLLM_XPU_KERNELS_COMMIT",
    "ONEDNN_COMMIT",
    "KVWEAVE_COMMIT",
    "LMCACHE_COMMIT",
    "PYTHON_VERSION",
    "ESIMD_TORCH_XPU_ARCH_LIST",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root):
    """Hash relative paths and contents of every file below root."""
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(sha256(path).encode() + b"\0")
    return digest.hexdigest()


def torch_version(constraints):
    for line in Path(constraints).read_text().splitlines():
        match = re.match(r"torch==(\S+)", line)
        if match:
            return match.group(1)
    raise SystemExit(f"No torch pin in {constraints}")


def build_inputs(args):
    missing = [name for name in REVISION_ARGS if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing build input environment: " + ", ".join(missing))
    return {
        "revisions": {name: os.environ[name] for name in REVISION_ARGS},
        "patches": {Path(p).name: sha256(p) for p in sorted(args.patch)},
        "custom_esimd_kernels_vllm_source": tree_sha256(args.esimd_source),
        "runtime_requirements": sha256(args.lock),
        "torch": torch_version(args.constraints),
    }


def flatten(inputs, prefix=""):
    flat = {}
    for key, value in inputs.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def wheel_distribution(path):
    # Wheel file names are {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    return path.name.split("-", 1)[0].lower()


def wheels(directory):
    found = {}
    for path in sorted(Path(directory).glob("*.whl")):
        found.setdefault(wheel_distribution(path), []).append(path)
    return found


def verify(manifest_path, inputs_path, directory):
    errors = []
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot read wheel manifest {manifest_path}: {error}")
    expected_inputs = flatten(json.loads(Path(inputs_path).read_text()))
    recorded_inputs = flatten(manifest.get("inputs", {}))
    for key in sorted(set(expected_inputs) | set(recorded_inputs)):
        if expected_inputs.get(key) != recorded_inputs.get(key):
            errors.append(f"Build input mismatch for {key}: wheels were built with "
                          f"{recorded_inputs.get(key)!r}, this build uses {expected_inputs.get(key)!r}")

    found = wheels(directory)
    recorded = manifest.get("wheels", {})
    for distribution in DISTRIBUTIONS:
        paths = found.pop(distribution, [])
        if len(paths) != 1:
            errors.append(f"Expected exactly one {distribution} wheel, found "
                          f"{[p.name for p in paths] or 'none'}")
            continue
        entry = recorded.get(paths[0].name)
        if entry is None:
            errors.append(f"{paths[0].name} is not listed in the manifest")
        elif entry != sha256(paths[0]):
            errors.append(f"{paths[0].name} does not match its manifest SHA256")
    for paths in found.values():
        errors.extend(f"Unexpected wheel: {p.name}" for p in paths)
    if errors:
        raise SystemExit("Wheel set validation failed:\n" + "\n".join(errors))
    print(f"Wheel set matches build inputs: {', '.join(sorted(recorded))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs")
    inputs.add_argument("--patch", action="append", required=True)
    inputs.add_argument("--esimd-source", required=True)
    inputs.add_argument("--lock", required=True)
    inputs.add_argument("--constraints", required=True)
    create = commands.add_parser("create")
    create.add_argument("--inputs", required=True)
    create.add_argument("--wheels", required=True)
    create.add_argument("--output", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--inputs", required=True)
    check.add_argument("--wheels", required=True)
    args = parser.parse_args()

    if args.command == "inputs":
        json.dump(build_inputs(args), sys.stdout, indent=2, sort_keys=True)
        print()
    elif args.command == "create":
        found = wheels(args.wheels)
        manifest = {
            "inputs": json.loads(Path(args.inputs).read_text()),
            "wheels": {p.name: sha256(p) for paths in found.values() for p in paths},
        }
        Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify(args.output, args.inputs, args.wheels)
    else:
        verify(Path(args.wheels) / "manifest.json", args.inputs, args.wheels)


if __name__ == "__main__":
    main()
