# SPDX-License-Identifier: Apache-2.0
"""Check the source requirements without importing or initializing an XPU."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement


def requirements(path: Path):
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            yield from requirements(path.parent / line[3:].strip())
        elif line and not line.startswith("--"):
            yield Requirement(line)


def check(source: Path) -> None:
    failures = []
    for requirement in requirements(source / "requirements/xpu.txt"):
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            failures.append(f"Missing {requirement}")
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            failures.append(f"{requirement}: installed {installed}")
    for name, expected in (
        ("torch", "2.14.0+xpu"),
        ("triton", "3.8.0+xpu"),
        ("vllm", "0.31.0+xpu"),
        ("vllm-xpu-kernels", "0.1.15.4"),
        ("transformers", "5.17.0"),
        ("tokenizers", "0.23.2"),
    ):
        actual = version(name)
        print(f"{name}={actual}")
        if actual != expected:
            failures.append(f"Expected {name}={expected}, found {actual}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    check(parser.parse_args().source)
