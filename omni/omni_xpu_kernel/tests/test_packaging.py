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
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
IMAGE_VERSION = "0.1.0-b8-dev"
BASE_VERSION = "0.1.0b8.dev0"
SUPPORTED_TORCH_MINORS = ("2.10", "2.11", "2.12")
SUPPORTED_XPU_TARGETS = ("bmg", "ptl-h")
VERSION_NAMESPACE = run_path(str(VERSION_FILE))
TORCH_VERSION = VERSION_NAMESPACE["get_installed_torch_version"]()
TORCH_VERSION_TAG = VERSION_NAMESPACE["get_torch_tag"](TORCH_VERSION)
XPU_TARGET = VERSION_NAMESPACE["get_build_xpu_target"]()
XPU_TARGET_TAG = VERSION_NAMESPACE["get_xpu_target_tag"](XPU_TARGET)
PACKAGE_VERSION = f"{BASE_VERSION}+{TORCH_VERSION_TAG}.{XPU_TARGET_TAG}"
SOURCE_VERSION = PACKAGE_VERSION


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
    assert version_module["__image_version__"] == IMAGE_VERSION
    assert version_module["__base_version__"] == BASE_VERSION
    assert version_module["__supported_torch_minors__"] == SUPPORTED_TORCH_MINORS
    assert version_module["__supported_xpu_targets__"] == SUPPORTED_XPU_TARGETS
    assert version_module["__torch_version__"] == TORCH_VERSION
    assert version_module["__xpu_target__"] == XPU_TARGET
    assert version_module["__version__"] == SOURCE_VERSION
    assert "+" not in IMAGE_VERSION
    assert TORCH_VERSION_TAG == "torch" + "".join(TORCH_VERSION.split(".")[:2])
    assert str(Version(SOURCE_VERSION)) == SOURCE_VERSION
    assert omni_xpu_kernel.__torch_version__ == TORCH_VERSION
    assert omni_xpu_kernel.__xpu_target__ == XPU_TARGET
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


@pytest.mark.parametrize(
    ("torch_version", "public_version", "torch_minor", "torch_tag"),
    [
        ("2.10.0+xpu", "2.10.0", "2.10", "torch210"),
        ("2.11.0+xpu", "2.11.0", "2.11", "torch211"),
        ("2.12.0+xpu", "2.12.0", "2.12", "torch212"),
        ("2.12.1+xpu", "2.12.1", "2.12", "torch212"),
    ],
)
def test_supported_torch_minors_select_distinct_wheel_tags(
    torch_version, public_version, torch_minor, torch_tag
):
    assert VERSION_NAMESPACE["get_public_torch_version"](torch_version) == public_version
    assert VERSION_NAMESPACE["get_torch_minor"](torch_version) == torch_minor
    assert VERSION_NAMESPACE["get_torch_tag"](torch_version) == torch_tag
    assert VERSION_NAMESPACE["get_package_version"](torch_version, XPU_TARGET) == (
        f"{BASE_VERSION}+{torch_tag}.{XPU_TARGET_TAG}"
    )


@pytest.mark.parametrize(
    ("target", "target_tag"),
    [("bmg", "bmg"), ("ptl-h", "ptlh")],
)
def test_gpu_targets_select_distinct_wheel_tags(target, target_tag):
    package_version = VERSION_NAMESPACE["get_package_version"]("2.11.0+xpu", target)

    assert package_version == f"{BASE_VERSION}+torch211.{target_tag}"
    assert VERSION_NAMESPACE["get_xpu_target_from_package_version"](
        package_version
    ) == target


@pytest.mark.parametrize("target", ["ptl", "ptl-u", "pvc", "invalid"])
def test_unsupported_gpu_targets_are_rejected(target):
    with pytest.raises(RuntimeError, match="Unsupported OMNI_XPU_DEVICE"):
        VERSION_NAMESPACE["normalize_xpu_target"](target)


def test_installed_wheel_identity_comes_from_its_own_metadata(monkeypatch, tmp_path):
    class FakeDistribution:
        version = f"{BASE_VERSION}+torch210.ptlh"
        requires = ["torch==2.10.0", "onednn==2025.3.0"]
        files = [Path("omni_xpu_kernel-0.1.0.dist-info") / "RECORD"]

        @staticmethod
        def locate_file(path):
            return tmp_path / path

    get_build_info = VERSION_NAMESPACE["get_packaged_build_info"]
    monkeypatch.setitem(get_build_info.__globals__, "distribution", lambda name: FakeDistribution())
    packaged_version_file = tmp_path / "omni_xpu_kernel" / "_version.py"

    assert get_build_info(packaged_version_file) == (
        f"{BASE_VERSION}+torch210.ptlh",
        "2.10.0",
        "ptl-h",
    )
    # An unrelated installed wheel must not override a source checkout build.
    assert get_build_info(VERSION_FILE) is None


def test_inconsistent_installed_wheel_metadata_is_rejected(monkeypatch, tmp_path):
    class FakeDistribution:
        version = f"{BASE_VERSION}+torch212.ptlh"
        requires = ["torch==2.10.0"]
        files = [Path("omni_xpu_kernel-0.1.0.dist-info") / "RECORD"]

        @staticmethod
        def locate_file(path):
            return tmp_path / path

    get_build_info = VERSION_NAMESPACE["get_packaged_build_info"]
    monkeypatch.setitem(get_build_info.__globals__, "distribution", lambda name: FakeDistribution())
    packaged_version_file = tmp_path / "omni_xpu_kernel" / "_version.py"

    with pytest.raises(RuntimeError, match="metadata is inconsistent"):
        get_build_info(packaged_version_file)


@pytest.mark.parametrize("torch_version", ["2.9.1+xpu", "2.13.0+xpu", "invalid"])
def test_unsupported_torch_versions_are_rejected(torch_version):
    with pytest.raises(RuntimeError, match="Torch minor|Unsupported Torch version"):
        VERSION_NAMESPACE["get_torch_minor"](torch_version)


def test_distribution_metadata_uses_normalized_torch_version(tmp_path):
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
    assert f"torch=={TORCH_VERSION}" in metadata.get_all("Requires-Dist")


def test_setup_metadata_rejects_unknown_gpu_target():
    env = setup_metadata_env(require_cute="0")
    env["OMNI_XPU_DEVICE"] = "pvc"

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "Unsupported OMNI_XPU_DEVICE" in result.stdout + result.stderr


def test_setup_metadata_tags_ptl_h_target():
    env = setup_metadata_env(require_cute="0")
    env["OMNI_XPU_DEVICE"] = "ptl-h"

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"{BASE_VERSION}+{TORCH_VERSION_TAG}.ptlh"


def test_build_system_does_not_force_a_torch_environment():
    pyproject = PYPROJECT_FILE.read_text(encoding="utf-8")
    build_system = pyproject.split("[build-system]", 1)[1].split("\n[", 1)[0]
    assert "torch==" not in build_system
    assert "onednn" not in build_system
    assert 'dynamic = ["version", "dependencies"]' in pyproject
    assert "omni_xpu_kernel._version.__version__" not in pyproject


def test_cute_is_required_by_default():
    result = run_setup_name(require_cute=None)

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "CUTE is required by default" in output


def test_core_only_build_requires_explicit_cute_opt_out():
    result = run_setup_name(require_cute="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "omni_xpu_kernel" in result.stdout


def test_extension_metadata_tracks_native_sources(monkeypatch, tmp_path):
    import setuptools

    captured = {}
    for required_dir in ("include", "tools/util/include", "examples/common", "applications"):
        (tmp_path / required_dir).mkdir(parents=True)
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("CUTLASS_SYCL_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "1")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))

    setup_namespace = run_path(
        str(PROJECT_ROOT / "setup.py"), run_name="__setup_metadata_test__"
    )

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
    assert setup_namespace["BUILD_XPU_TARGET"] == XPU_TARGET
    assert setup_namespace["XPU_ARCH_MACRO"] == (
        "OMNI_XPU_ARCH_PTL_H" if XPU_TARGET == "ptl-h" else "OMNI_XPU_ARCH_BMG"
    )
    cute_dependencies = {
        Path(dependency).name
        for dependency in extensions["omni_xpu_kernel.cute.cute_fmha_torch"].depends
    }
    assert "cute_fmha_config.h" in cute_dependencies
    assert all(
        not Path(dependency).is_absolute()
        for extension in extensions.values()
        for dependency in extension.depends
    )
    assert extensions["omni_xpu_kernel.lgrf_uni.lgrf_sdp"].sources
    assert f"torch=={TORCH_VERSION}" in captured["install_requires"]
    assert captured["version"] == PACKAGE_VERSION
    assert any(
        requirement.startswith("onednn==2025.3.0;")
        for requirement in captured["install_requires"]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="ELF $ORIGIN is Linux-only")
def test_linux_runtime_search_paths_are_prefix_relative(monkeypatch):
    import setuptools

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.delenv("CUTLASS_SYCL_ROOT", raising=False)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(str(PROJECT_ROOT / "setup.py"), run_name="__rpath_test__")

    runtime_lib = Path(sys.prefix) / "lib"
    core_rpath = namespace["get_origin_rpath"]("omni_xpu_kernel._C", runtime_lib)
    sidecar_rpath = namespace["get_origin_rpath"](
        "omni_xpu_kernel.lgrf_uni.lgrf_sdp", runtime_lib
    )

    assert core_rpath.startswith("$ORIGIN/")
    assert sidecar_rpath.startswith("$ORIGIN/")
    assert sys.prefix not in core_rpath
    assert sys.prefix not in sidecar_rpath


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
            if sys.platform == "linux" and shutil.which("readelf") is not None:
                native_members = [
                    member for member in wheel_zip.namelist()
                    if member.endswith(".so")
                ]
                extract_root = temp_root / "extracted"
                for member in native_members:
                    wheel_zip.extract(member, extract_root)
                    dynamic = subprocess.run(
                        ["readelf", "-d", str(extract_root / member)],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                    assert "$ORIGIN" in dynamic
                    assert str(sys.prefix) not in dynamic
                    assert "/opt/intel" not in dynamic
