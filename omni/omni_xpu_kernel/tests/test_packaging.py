import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email.parser import Parser
from importlib.util import find_spec
from pathlib import Path
from runpy import run_path
from typing import Optional

import pytest
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path("omni_xpu_kernel") / "lgrf_uni"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
ARTIFACT_NAME = f"lgrf_sdp{EXT_SUFFIX if isinstance(EXT_SUFFIX, str) and EXT_SUFFIX else '.so'}"
VERSION_FILE = PROJECT_ROOT / "omni_xpu_kernel" / "_version.py"
SOURCE_VERSION = "0.1.0-b8"
PACKAGE_VERSION = "0.1.0b8"


def setup_metadata_env(*, require_cute: Optional[str]) -> dict[str, str]:
    env = {**os.environ, "PIP_NO_INPUT": "1"}
    env.pop("CUTLASS_SYCL_ROOT", None)
    if require_cute is None:
        env.pop("OMNI_XPU_REQUIRE_CUTE", None)
    else:
        env["OMNI_XPU_REQUIRE_CUTE"] = require_cute
    return env


def run_setup_name(*, require_cute: Optional[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=setup_metadata_env(require_cute=require_cute),
    )


def test_kernel_version_is_exposed_by_package_metadata():
    import omni_xpu_kernel

    version_module = run_path(str(VERSION_FILE))
    assert version_module["__version__"] == SOURCE_VERSION
    assert omni_xpu_kernel.__version__ == SOURCE_VERSION

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=setup_metadata_env(require_cute="0"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == SOURCE_VERSION
    assert str(Version(result.stdout.strip())) == PACKAGE_VERSION


def test_distribution_metadata_uses_normalized_b8_version(tmp_path):
    result = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=setup_metadata_env(require_cute="0"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    pkg_info = next(tmp_path.glob("*.egg-info/PKG-INFO"))
    metadata = Parser().parsestr(pkg_info.read_text(encoding="utf-8"))
    assert metadata["Version"] == PACKAGE_VERSION


def test_cute_is_required_by_default():
    result = run_setup_name(require_cute=None)

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "CUTE is required by default" in output


def test_core_only_build_requires_explicit_cute_opt_out():
    result = run_setup_name(require_cute="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "omni_xpu_kernel" in result.stdout


def test_extension_metadata_tracks_native_sources(monkeypatch):
    import setuptools

    captured = {}
    monkeypatch.delenv("CUTLASS_SYCL_ROOT", raising=False)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))

    run_path(str(PROJECT_ROOT / "setup.py"), run_name="__setup_metadata_test__")

    extensions = {extension.name: extension for extension in captured["ext_modules"]}
    main_sources = {Path(source).name for source in extensions["omni_xpu_kernel._C"].sources}
    assert all(
        not Path(source).is_absolute()
        for extension in extensions.values()
        for source in extension.sources
    )
    assert "bindings.cpp" in main_sources
    assert "kitchen_rope.cpp" in main_sources
    assert "svdq_dequant.cpp" in main_sources
    assert extensions["omni_xpu_kernel.lgrf_uni.lgrf_sdp"].sources


def test_default_build_accepts_complete_cutlass_tree(tmp_path):
    for required_dir in ("include", "tools/util/include", "examples/common", "applications"):
        (tmp_path / required_dir).mkdir(parents=True)
    env = setup_metadata_env(require_cute=None)
    env["CUTLASS_SYCL_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "omni_xpu_kernel" in result.stdout


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
            env=setup_metadata_env(require_cute="0"),
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
