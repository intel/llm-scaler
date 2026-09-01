import hashlib
import os
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from runpy import run_path
from typing import Optional

import pytest
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "omni_xpu_kernel" / "_version.py"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
IMAGE_VERSION = "0.2.0-b2"
BASE_VERSION = "0.2.0b2"
SUPPORTED_TORCH_MINORS = ("2.10", "2.11", "2.12", "2.13")
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
        ("2.13.0+xpu", "2.13.0", "2.13", "torch213"),
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


def test_windows_onednn_contract_tracks_torch_minor(monkeypatch):
    import setuptools

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"), run_name="__windows_onednn_contract_test__"
    )
    get_contract = namespace["get_windows_onednn_contract"]

    assert get_contract("2.12.0+xpu") == ("2025.3.0", (3, 9, 1), "2025.3")
    assert get_contract("2.13.0+xpu") == ("2026.0.0", (3, 11, 2), "2026.0")


def test_windows_compile_env_adds_aot_companion_tools(monkeypatch, tmp_path):
    import setuptools

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"), run_name="__windows_compile_env_test__"
    )
    get_compile_env = namespace["get_compile_env"]
    monkeypatch.setitem(get_compile_env.__globals__, "IS_WINDOWS", True)

    oneapi_root = tmp_path / "oneAPI"
    compiler_bin = oneapi_root / "compiler" / "2026.0" / "bin"
    compiler_tools = compiler_bin / "compiler"
    compiler_lib = compiler_bin.parent / "lib"
    ocloc_bin = oneapi_root / "ocloc" / "2026.0" / "bin"
    compiler_tools.mkdir(parents=True)
    compiler_lib.mkdir(parents=True)
    ocloc_bin.mkdir(parents=True)
    compiler = compiler_bin / "icx.exe"
    compiler.write_bytes(b"")
    monkeypatch.setenv("PATH", os.fspath(tmp_path / "existing"))
    monkeypatch.setenv("LIB", os.fspath(tmp_path / "existing-lib"))

    env = get_compile_env(compiler=os.fspath(compiler))
    paths = env["PATH"].split(os.pathsep)

    assert paths[:2] == [
        os.fspath(compiler_tools.resolve()),
        os.fspath(ocloc_bin.resolve()),
    ]
    assert env["LIB"].split(os.pathsep)[:2] == [
        os.fspath(compiler_lib.resolve()),
        os.fspath(tmp_path / "existing-lib"),
    ]


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


@pytest.mark.parametrize("torch_version", ["2.9.1+xpu", "2.14.0+xpu", "invalid"])
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
    assert 'exclude = ["tests*", "scripts*", "benchmarks*"]' in pyproject
    assert "omni_xpu_kernel._version.__version__" not in pyproject


@pytest.mark.skipif(sys.platform == "win32", reason="Windows CUTE is opt-in")
def test_cute_is_required_by_default():
    result = run_setup_name(require_cute=None)

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "CUTE is required by default" in output


def test_explicit_cute_opt_out_builds_core_only():
    result = run_setup_name(require_cute="0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "omni_xpu_kernel" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CUTE default")
def test_windows_cute_enable_requires_a_complete_sycl_tla_tree():
    result = run_setup_name(require_cute="1")

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "CUTE was explicitly enabled on Windows" in output


def test_extension_metadata_tracks_native_sources(monkeypatch, tmp_path):
    import setuptools

    captured = {}
    for required_dir in ("include", "tools/util/include", "examples/common", "applications"):
        (tmp_path / required_dir).mkdir(parents=True)
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("CUTLASS_SYCL_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "OMNI_XPU_REQUIRE_CUTE",
        "0" if sys.platform == "win32" else "1",
    )
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
    assert "kitchen_rms_rope_sycl.cpp" in main_sources
    assert "group_norm_seedvr_bmg.cpp" in main_sources
    assert "cat_pad_bmg.cpp" in main_sources
    assert "svdq_dequant.cpp" in main_sources
    assert setup_namespace["BUILD_XPU_TARGET"] == XPU_TARGET
    assert setup_namespace["XPU_ARCH_MACRO"] == (
        "OMNI_XPU_ARCH_PTL_H" if XPU_TARGET == "ptl-h" else "OMNI_XPU_ARCH_BMG"
    )
    assert all(
        not package.startswith(("tests", "scripts", "benchmarks"))
        for package in captured["packages"]
    )
    if sys.platform == "win32":
        assert "omni_xpu_kernel.cute.cute_fmha_torch" not in extensions
    else:
        cute_extension = extensions[
            "omni_xpu_kernel.cute.cute_fmha_torch"
        ]
        cute_sources = {
            Path(source).name for source in cute_extension.sources
        }
        cute_dependencies = {
            Path(dependency).name
            for dependency in cute_extension.depends
        }
        assert cute_sources == {
            "cute_fmha_torch.cpp",
            "sol_attn_prepare.cpp",
            "sol_attn_torch.cpp",
        }
        assert "cute_fmha_config.h" in cute_dependencies
        assert "sol_attn_config.h" in cute_dependencies
        assert "sol_attn_mainloop.hpp" in cute_dependencies
    assert all(
        not Path(dependency).is_absolute()
        for extension in extensions.values()
        for dependency in extension.depends
    )
    assert extensions["omni_xpu_kernel.lgrf_uni.lgrf_sdp"].sources
    assert f"torch=={TORCH_VERSION}" in captured["install_requires"]
    assert captured["version"] == PACKAGE_VERSION
    onednn_requirements = [
        requirement
        for requirement in captured["install_requires"]
        if requirement.startswith("onednn==2026.0.0;")
    ]
    assert onednn_requirements == [
        "onednn==2026.0.0; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ]
    if sys.platform == "win32":
        assert captured["package_data"]["omni_xpu_kernel"] == [
            ".libs/*.dll",
            ".libs/onednn/*",
        ]


def test_windows_cute_extension_is_explicit_opt_in(monkeypatch, tmp_path):
    import platform
    import setuptools

    for required_dir in (
        "include",
        "tools/util/include",
        "examples/common",
        "applications",
    ):
        (tmp_path / required_dir).mkdir(parents=True)
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setenv("CUTLASS_SYCL_ROOT", str(tmp_path))

    def extension_names(require_cute):
        captured = {}
        if require_cute is None:
            monkeypatch.delenv("OMNI_XPU_REQUIRE_CUTE", raising=False)
        else:
            monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", require_cute)
        monkeypatch.setattr(
            setuptools, "setup", lambda **kwargs: captured.update(kwargs)
        )
        run_path(
            str(PROJECT_ROOT / "setup.py"),
            run_name=f"__windows_cute_metadata_{require_cute}__",
        )
        return {extension.name for extension in captured["ext_modules"]}

    extension = "omni_xpu_kernel.cute.cute_fmha_torch"
    assert extension not in extension_names(None)
    assert extension not in extension_names("0")
    assert extension in extension_names("1")


def test_windows_onednn_runtime_bundle_contains_notices_and_hash(
    monkeypatch, tmp_path
):
    import setuptools

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"), run_name="__onednn_bundle_test__"
    )
    find_runtime = namespace["find_windows_onednn_runtime"]
    bundle_runtime = namespace["bundle_windows_onednn_runtime"]
    monkeypatch.setitem(find_runtime.__globals__, "IS_WINDOWS", True)

    dnnl_root = tmp_path / "oneapi" / "dnnl" / "2026.0"
    lib_dir = dnnl_root / "lib"
    runtime = dnnl_root / "bin" / "dnnl.dll"
    notices = dnnl_root / "share" / "doc" / "dnnl"
    lib_dir.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    notices.mkdir(parents=True)
    (lib_dir / "dnnl.lib").write_bytes(b"import library")
    runtime_bytes = b"validated oneDNN runtime"
    runtime.write_bytes(runtime_bytes)
    (notices / "LICENSE").write_text("license\n", encoding="utf-8")
    (notices / "THIRD-PARTY-PROGRAMS").write_text(
        "third party\n", encoding="utf-8"
    )

    assert find_runtime(lib_dir) == runtime.resolve()
    output = tmp_path / "build" / "omni_xpu_kernel" / "_C.pyd"
    bundle_runtime(output, runtime)

    vendor = output.parent / ".libs"
    assert (vendor / "dnnl.dll").read_bytes() == runtime_bytes
    assert (vendor / "onednn" / "LICENSE").read_text(encoding="utf-8") == (
        "license\n"
    )
    version = (vendor / "onednn" / "VERSION").read_text(encoding="utf-8")
    expected_version = ".".join(
        map(str, namespace["get_validated_onednn_version"]())
    )
    assert f"oneDNN={expected_version}" in version
    assert hashlib.sha256(runtime_bytes).hexdigest() in version


def test_bmg_cute_overlay_patches_private_header_copy(monkeypatch, tmp_path):
    import setuptools

    cutlass_root = tmp_path / "sycl-tla"
    for required_dir in (
        "include",
        "tools/util/include",
        "examples/common",
        "applications",
    ):
        (cutlass_root / required_dir).mkdir(parents=True)
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("CUTLASS_SYCL_ROOT", str(cutlass_root))
    monkeypatch.setenv(
        "OMNI_XPU_REQUIRE_CUTE",
        "0" if sys.platform == "win32" else "1",
    )
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"), run_name="__cute_bmg_overlay_test__"
    )
    if sys.platform == "win32":
        for relative_path, patches in namespace[
            "WINDOWS_CUTE_HEADER_PATCHES"
        ].items():
            source = cutlass_root / "include" / relative_path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                "\n".join(original for original, _, _ in patches) + "\n",
                encoding="utf-8",
            )

    collective = (
        cutlass_root
        / "applications"
        / "flash_attention_v2"
        / "collective"
    )
    collective.mkdir(parents=True)
    original = namespace["BMG_CUTE_REMAINDER_MASK_ORIGINAL"]
    replacement = namespace["BMG_CUTE_REMAINDER_MASK_REPLACEMENT"]
    source_header = collective / "xe_fmha_fwd_mainloop.hpp"
    source_header.write_text(f"prefix\n{original}suffix\n", encoding="utf-8")
    (collective / "fmha_fusion.hpp").write_text(
        "fusion sentinel\n", encoding="utf-8"
    )

    overlay = namespace["prepare_bmg_cute_include_overlay"](
        cutlass_root, tmp_path / "build"
    )
    overlay_collective = (
        overlay / "flash_attention_v2" / "collective"
    )
    patched = (
        overlay_collective / "xe_fmha_fwd_mainloop.hpp"
    ).read_text(encoding="utf-8")

    assert replacement in patched
    assert original not in patched
    assert original in source_header.read_text(encoding="utf-8")
    assert (
        overlay_collective / "fmha_fusion.hpp"
    ).read_text(encoding="utf-8") == "fusion sentinel\n"


def _write_synthetic_bmg_cute_headers(namespace, cutlass_root):
    collective = (
        cutlass_root
        / "applications"
        / "flash_attention_v2"
        / "collective"
    )
    collective.mkdir(parents=True, exist_ok=True)
    (collective / "xe_fmha_fwd_mainloop.hpp").write_text(
        namespace["BMG_CUTE_REMAINDER_MASK_ORIGINAL"], encoding="utf-8"
    )
    (collective / "fmha_fusion.hpp").write_text(
        "fusion sentinel\n", encoding="utf-8"
    )
    for relative_path, patches in namespace[
        "WINDOWS_CUTE_HEADER_PATCHES"
    ].items():
        source = cutlass_root / "include" / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "\n".join(original for original, _, _ in patches) + "\n",
            encoding="utf-8",
        )


def test_windows_cute_overlay_is_private_and_fail_closed(monkeypatch, tmp_path):
    import setuptools

    cutlass_root = tmp_path / "sycl-tla"
    for required_dir in (
        "include",
        "tools/util/include",
        "examples/common",
        "applications",
    ):
        (cutlass_root / required_dir).mkdir(parents=True)
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"),
        run_name="__windows_cute_overlay_test__",
    )
    _write_synthetic_bmg_cute_headers(namespace, cutlass_root)
    prepare_overlay = namespace["prepare_bmg_cute_include_overlay"]
    monkeypatch.setitem(prepare_overlay.__globals__, "IS_WINDOWS", True)

    overlay = prepare_overlay(cutlass_root, tmp_path / "build")
    for relative_path, patches in namespace[
        "WINDOWS_CUTE_HEADER_PATCHES"
    ].items():
        source = cutlass_root / "include" / relative_path
        source_text = source.read_text(encoding="utf-8")
        overlay_text = (overlay / relative_path).read_text(encoding="utf-8")
        for original, replacement, _ in patches:
            assert original in source_text
            assert replacement in overlay_text
            if original not in replacement:
                assert original not in overlay_text

    drifted = (
        cutlass_root / "include" / "cute" / "atom" / "copy_traits_xe_2d.hpp"
    )
    drifted.write_text("upstream changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="matches=0"):
        prepare_overlay(cutlass_root, tmp_path / "drifted-build")


def test_windows_cute_compile_command_uses_validated_flags(monkeypatch, tmp_path):
    import setuptools

    cutlass_root = tmp_path / "sycl-tla"
    for required_dir in (
        "include",
        "tools/util/include",
        "examples/common",
        "applications",
    ):
        (cutlass_root / required_dir).mkdir(parents=True)
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("CUTLASS_SYCL_ROOT", str(cutlass_root))
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"),
        run_name="__windows_cute_command_test__",
    )
    _write_synthetic_bmg_cute_headers(namespace, cutlass_root)

    build_extension = namespace["ICPXBuildExt"].build_extension
    build_globals = build_extension.__globals__
    monkeypatch.setitem(build_globals, "IS_WINDOWS", True)
    monkeypatch.setitem(build_globals, "BUILD_XPU_TARGET", "bmg")
    monkeypatch.setitem(build_globals, "XPU_ARCH_MACRO", "OMNI_XPU_ARCH_BMG")
    monkeypatch.setitem(build_globals, "get_icpx_path", lambda: "C:/fake/icx.exe")
    monkeypatch.setitem(build_globals, "validate_torch_build", lambda *args: None)
    fake_onednn = tmp_path / "onednn"
    monkeypatch.setitem(
        build_globals,
        "get_onednn_paths",
        lambda: (
            fake_onednn / "include",
            fake_onednn / "lib",
            fake_onednn / "lib" / "dnnl.lib",
            fake_onednn / "bin" / "dnnl.dll",
            "test",
        ),
    )
    monkeypatch.setitem(
        build_globals, "get_runtime_library_dir", lambda: tmp_path / "runtime"
    )
    monkeypatch.setitem(
        build_globals,
        "get_torch_runtime_library_dir",
        lambda: tmp_path / "torch-runtime",
    )
    monkeypatch.setitem(
        build_globals, "find_onednn_notice_dir", lambda runtime: None
    )
    monkeypatch.setitem(
        build_globals,
        "get_compile_env",
        lambda include, compiler="": {},
    )

    commands = []

    def capture_compile(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture_compile)
    command = namespace["ICPXBuildExt"](setuptools.Distribution())
    command.build_temp = str(tmp_path / "build")
    monkeypatch.setattr(
        command,
        "get_ext_fullpath",
        lambda name: str(tmp_path / "cute_fmha_torch.cp313-win_amd64.pyd"),
    )
    extension = namespace["ICPXExtension"](
        "omni_xpu_kernel.cute.cute_fmha_torch",
        sourcedir=str(PROJECT_ROOT),
    )
    command.build_extension(extension)

    assert len(commands) == 1
    compile_command = commands[0]
    assert compile_command[:2] == ["C:/fake/icx.exe", "-fsycl"]
    assert "/MD" in compile_command
    assert "/LD" in compile_command
    assert "-Xsycl-target-backend=spir64_gen" in compile_command
    assert "-device bmg-g31" in compile_command
    assert "-fno-sycl-instrument-device-code" in compile_command
    assert "-DCUTLASS_ENABLE_SYCL" in compile_command
    assert "-DSYCL_INTEL_TARGET" in compile_command
    assert "-DOMNI_XPU_ARCH_BMG=1" in compile_command
    assert any(
        argument.startswith("/I") and "cute_bmg_include_overlay" in argument
        for argument in compile_command
    )
    assert "torch_xpu.lib" in compile_command
    assert any(argument.endswith("sol_attn_prepare.cpp") for argument in compile_command)
    assert any(argument.endswith("sol_attn_torch.cpp") for argument in compile_command)
    assert compile_command.index("/link") > compile_command.index(
        str(
            PROJECT_ROOT
            / "omni_xpu_kernel"
            / "cute"
            / "cute_fmha_torch.cpp"
        )
    )


def test_cute_loader_finds_windows_pyd(monkeypatch, tmp_path):
    namespace = run_path(
        str(PROJECT_ROOT / "omni_xpu_kernel" / "cute" / "__init__.py"),
        run_name="__cute_loader_test__",
    )
    package_dir = tmp_path / "omni_xpu_kernel" / "cute"
    package_dir.mkdir(parents=True)
    extension = package_dir / "cute_fmha_torch.cp313-win_amd64.pyd"
    extension.write_bytes(b"test extension")
    find_extension = namespace["_find_extension"]
    monkeypatch.setitem(
        find_extension.__globals__, "__file__", str(package_dir / "__init__.py")
    )
    monkeypatch.setitem(
        find_extension.__globals__,
        "EXTENSION_SUFFIXES",
        [".cp313-win_amd64.pyd", ".pyd"],
    )
    monkeypatch.delenv("OMNI_CUTE_FMHA_SO", raising=False)

    assert Path(find_extension()) == extension
    assert Path(namespace["_find_so"]()) == extension


@pytest.mark.parametrize("target", SUPPORTED_XPU_TARGETS)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux core AOT command")
def test_linux_core_compile_command_is_aot_for_every_supported_target(
    monkeypatch, tmp_path, target
):
    import setuptools

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.delenv("CUTLASS_SYCL_ROOT", raising=False)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(
        str(PROJECT_ROOT / "setup.py"), run_name="__core_aot_command_test__"
    )

    build_extension = namespace["ICPXBuildExt"].build_extension
    build_globals = build_extension.__globals__
    target_macro = (
        "OMNI_XPU_ARCH_PTL_H" if target == "ptl-h" else "OMNI_XPU_ARCH_BMG"
    )
    monkeypatch.setitem(build_globals, "BUILD_XPU_TARGET", target)
    monkeypatch.setitem(build_globals, "XPU_ARCH_MACRO", target_macro)
    monkeypatch.setitem(build_globals, "get_icpx_path", lambda: "/fake/icpx")

    commands = []

    def capture_compile(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture_compile)

    command = namespace["ICPXBuildExt"](setuptools.Distribution())
    monkeypatch.setattr(
        command,
        "get_ext_fullpath",
        lambda name: str(tmp_path / f"{name.rsplit('.', 1)[-1]}.so"),
    )
    extension = namespace["ICPXExtension"](
        "omni_xpu_kernel._C", sourcedir=str(PROJECT_ROOT)
    )
    command.build_extension(extension)

    assert len(commands) == 1
    compile_command = commands[0]
    assert compile_command[:2] == ["/fake/icpx", "-fsycl"]
    assert "-fsycl-esimd-force-stateless-mem" in compile_command
    assert compile_command.count("-fsycl-targets=spir64_gen") == 1
    backend_index = compile_command.index("-Xsycl-target-backend")
    assert compile_command[backend_index + 1] == f"-device {target}"
    assert "-DOMNI_XPU_CORE_AOT=1" in compile_command
    assert f"-D{target_macro}=1" in compile_command


@pytest.mark.skipif(sys.platform != "linux", reason="ELF $ORIGIN is Linux-only")
def test_linux_runtime_search_paths_are_prefix_relative(monkeypatch):
    import setuptools

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.delenv("CUTLASS_SYCL_ROOT", raising=False)
    monkeypatch.setenv("OMNI_XPU_REQUIRE_CUTE", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = run_path(str(PROJECT_ROOT / "setup.py"), run_name="__rpath_test__")

    runtime_lib = namespace["get_runtime_library_dir"]()
    torch_runtime_lib = namespace["get_torch_runtime_library_dir"]()
    core_rpath = namespace["get_origin_rpath"]("omni_xpu_kernel._C", runtime_lib)
    torch_rpath = namespace["get_origin_rpath"](
        "omni_xpu_kernel._C", torch_runtime_lib
    )
    sidecar_rpath = namespace["get_origin_rpath"](
        "omni_xpu_kernel.lgrf_uni.lgrf_sdp", runtime_lib
    )

    assert core_rpath.startswith("$ORIGIN/")
    assert torch_rpath == "$ORIGIN/../torch/lib"
    assert sidecar_rpath.startswith("$ORIGIN/")
    assert sys.prefix not in core_rpath
    assert sys.prefix not in sidecar_rpath

    external = Path("/opt/intel/oneapi/dnnl/latest/lib")
    assert namespace["get_origin_rpath"](
        "omni_xpu_kernel._C", external
    ) == external.resolve().as_posix()


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
