# SPDX-License-Identifier: Apache-2.0
"""Fetch exact source revisions and apply the checksum-verified patch series."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def prepare(manifest: Path, root: Path) -> None:
    metadata = json.loads(manifest.read_text())
    root.mkdir(parents=True, exist_ok=True)
    for source in metadata["sources"]:
        directory = root / source["name"]
        patches = []
        for entry in source["patches"]:
            patch = manifest.parent / entry["file"]
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {patch.name}")
            patches.append(patch)
        if directory.exists():
            # Never reset or overwrite an existing developer checkout.
            if git(directory, "rev-parse", "HEAD") != source["base_commit"]:
                raise ValueError(f"Unexpected source revision: {directory}")
            if git(directory, "write-tree") != source["patched_tree"]:
                raise ValueError(f"Existing checkout differs from recipe: {directory}")
            if git(directory, "diff", "--name-only") or git(
                directory, "ls-files", "--others", "--exclude-standard"
            ):
                raise ValueError(
                    f"Existing checkout has additional changes: {directory}"
                )
            continue
        directory.mkdir()
        git(directory, "init", "--quiet")
        git(directory, "remote", "add", "origin", source["repository"])
        git(directory, "fetch", "--quiet", "--depth=1", "origin", source["base_commit"])
        git(directory, "checkout", "--quiet", "--detach", source["base_commit"])
        for patch in patches:
            git(directory, "apply", "--check", "--index", str(patch))
            git(directory, "apply", "--index", str(patch))
        if git(directory, "write-tree") != source["patched_tree"]:
            raise ValueError(f"Patched source tree does not match: {source['name']}")
        print(f"Prepared {source['name']}: {source['base_commit']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.manifest.resolve(), arguments.root.resolve())
