#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Replace identical copies of a versioned shared library with symlinks.

Wheels cannot carry symlinks, so a library shipped as libfoo.so, libfoo.so.1
and libfoo.so.1.0 is installed as three full copies. Within each directory,
keep the most specific name and link the other names of the same library to
it when their contents are identical.
"""
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SONAME = re.compile(r"^(?P<stem>.+\.so)(\.[0-9A-Za-z]+)*$")


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main(root):
    saved = 0
    for directory, _, names in os.walk(root):
        groups = defaultdict(list)
        for name in names:
            path = Path(directory, name)
            match = SONAME.match(name)
            if match and path.is_file() and not path.is_symlink():
                groups[match["stem"]].append(path)
        for paths in groups.values():
            if len(paths) < 2:
                continue
            by_content = defaultdict(list)
            for path in paths:
                by_content[(path.stat().st_size, digest(path))].append(path)
            for (size, _), copies in by_content.items():
                if len(copies) < 2:
                    continue
                target = max(copies, key=lambda p: (len(p.name), p.name))
                for copy in copies:
                    if copy != target:
                        copy.unlink()
                        copy.symlink_to(target.name)
                        saved += size
    print(f"Replaced duplicate shared libraries with symlinks: {saved / 2**20:.0f} MiB")


if __name__ == "__main__":
    main(sys.argv[1])
