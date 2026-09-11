"""Incremental SYCL builds must incorporate changes made only to headers."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("icpx") is None or shutil.which("ninja") is None,
    reason="requires the oneAPI compiler and ninja",
)
def test_sycl_header_change_rebuilds_object(tmp_path):
    build_module = Path(__file__).resolve().parents[1] / "esimd_build_extention.py"
    spec = importlib.util.spec_from_file_location("esimd_build_for_test", build_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    header = tmp_path / "value.h"
    source = tmp_path / "probe.sycl"
    obj = tmp_path / "probe.o"
    header.write_text("#define HEADER_VALUE 7\n")
    source.write_text('#include "value.h"\nint main() { return HEADER_VALUE; }\n')
    module._write_ninja_file(
        path=str(tmp_path / "build.ninja"),
        cflags=[], post_cflags=[], cuda_cflags=[], cuda_post_cflags=[],
        cuda_dlink_post_cflags=[], sycl_cflags=["-fsycl"],
        sycl_post_cflags=[], sycl_dlink_post_cflags=[],
        sources=[str(source)], objects=[str(obj)], ldflags=[],
        library_target=None, with_cuda=False, with_sycl=True,
    )

    def build_and_run(expected):
        subprocess.run(["ninja", "-C", str(tmp_path)], check=True, capture_output=True)
        executable = tmp_path / f"probe_{expected}"
        subprocess.run(
            ["icpx", "-fsycl", str(obj), "-o", str(executable)],
            check=True, capture_output=True,
        )
        assert subprocess.run([str(executable)]).returncode == expected
        # Generated temporary SYCL headers must not cause perpetual rebuilds.
        dry_run = subprocess.check_output(
            ["ninja", "-C", str(tmp_path), "-n"], text=True
        )
        assert "no work to do" in dry_run

    build_and_run(7)
    header.write_text("#define HEADER_VALUE 9\n")
    build_and_run(9)
