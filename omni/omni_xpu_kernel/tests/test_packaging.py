import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from importlib.util import find_spec
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path("omni_xpu_kernel") / "lgrf_uni"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
ARTIFACT_NAME = f"lgrf_sdp{EXT_SUFFIX if isinstance(EXT_SUFFIX, str) and EXT_SUFFIX else '.so'}"


def has_packaging_prerequisites() -> bool:
    if shutil.which("icpx") is None:
        return False
    torch_spec = find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        return False
    torch_root = Path(torch_spec.origin).resolve().parent
    xpu_header = torch_root / "include" / "c10" / "xpu" / "impl" / "xpu_cmake_macros.h"
    return xpu_header.exists()


@pytest.mark.skipif(
    not has_packaging_prerequisites(),
    reason="Packaging test requires icpx and an XPU-enabled PyTorch build environment",
)
def test_lgrf_sdp_wheel_contains_sidecar_artifact():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        source_copy = temp_root / "src"
        dist_dir = temp_root / "dist"

        shutil.copytree(
            PROJECT_ROOT,
            source_copy,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                ".pytest_cache",
                "*.pyc",
                "*.pyo",
                "build",
                "dist",
                "*.egg-info",
            ),
        )

        stale_artifact = source_copy / PACKAGE_DIR / ARTIFACT_NAME
        stale_artifact.unlink(missing_ok=True)
        assert not stale_artifact.exists(), "test must start from a clean tree without a prebuilt sidecar"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-build-isolation",
                "--no-deps",
                "-w",
                str(dist_dir),
            ],
            cwd=source_copy,
            capture_output=True,
            text=True,
            env={**os.environ, "PIP_NO_INPUT": "1"},
        )

        assert result.returncode == 0, (
            "expected wheel build to succeed from a clean copy\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        wheels = list(dist_dir.glob("*.whl"))
        assert wheels, "expected pip wheel to produce a wheel artifact"

        expected_member = str(PACKAGE_DIR / ARTIFACT_NAME)
        with zipfile.ZipFile(wheels[0]) as wheel_zip:
            assert expected_member in wheel_zip.namelist(), (
                f"expected packaged wheel to contain {expected_member}, "
                "not just a source-tree sidecar artifact"
            )
