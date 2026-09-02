"""Static and CPU-only contracts for custom ESIMD DSO ownership."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETUP_PLE_ONLY = ROOT / "setup_ple_only.py"
SETUP_GEMV_ONLY = ROOT / "setup_gemv_only.py"
SETUP_SYCL = ROOT / "setup_sycl.py"
ESIMD_UTILS = (
    ROOT.parents[2]
    / "applications.ai.gpu.llm-scaler-vllm"
    / "vllm/model_executor/layers/esimd_utils.py"
)


def _load_esimd_utils():
    spec = importlib.util.spec_from_file_location(
        "esimd_utils_owner_contract", ESIMD_UTILS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_call(path: Path) -> ast.Call:
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(calls) == 1
    return calls[0]


def _keyword(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"missing setup keyword: {name}")


def _extension_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "SyclExtension":
            continue
        value = _keyword(node, "name")
        assert isinstance(value, ast.Constant) and isinstance(value.value, str)
        names.append(value.value)
    return names


def test_ple_only_is_dso_artifact_without_production_package() -> None:
    source = SETUP_PLE_ONLY.read_text()
    call = _setup_call(SETUP_PLE_ONLY)
    packages = _keyword(call, "packages")

    assert "find_packages" not in source
    assert isinstance(packages, ast.List) and not packages.elts
    assert isinstance(_keyword(call, "py_modules"), ast.List)
    assert not any(
        isinstance(keyword.value, ast.Dict) and keyword.arg == "package_dir"
        for keyword in call.keywords
    )
    assert _extension_names(SETUP_PLE_ONLY) == ["ple_ops"]
    assert "$ORIGIN/torch/lib" in source
    assert "$ORIGIN/../../torch/lib" not in source
    assert "esimd_kernel_ple.sycl" in source
    assert "torch_extension_ple.cc" in source


def test_build_only_variants_cannot_install_or_build_canonical_package() -> None:
    for path in (SETUP_GEMV_ONLY, SETUP_SYCL):
        source = path.read_text()
        call = _setup_call(path)
        packages = _keyword(call, "packages")
        assert "find_packages" not in source
        assert isinstance(packages, ast.List) and not packages.elts
        assert isinstance(_keyword(call, "py_modules"), ast.List)
        assert not any(
            isinstance(keyword.value, ast.Dict) and keyword.arg == "package_dir"
            for keyword in call.keywords
        )
        names = _extension_names(path)
        assert names
        assert all("custom_esimd_kernels_vllm." not in name for name in names)
        assert all(name.endswith("_only") for name in names)
        assert "$ORIGIN/torch/lib" in source
        assert "$ORIGIN/../../torch/lib" not in source
        assert 'os.environ.setdefault("TORCH_XPU_ARCH_LIST", "bmg-g31")' in source
        if path == SETUP_SYCL:
            assert 'extension_name="qsa_ops_sycl_only"' in source
            assert "custom_esimd_kernels_vllm.qsa_ops" not in source
        assert "esimd_kernel_ple.sycl" not in source
        assert "torch_extension_ple.cc" not in source


def test_main_prefix_does_not_match_gemm_variant(tmp_path: Path) -> None:
    main = tmp_path / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
    gemm = tmp_path / "custom_esimd_kernels_gemm.cpython-312-x86_64-linux-gnu.so"
    main.touch()
    gemm.touch()

    module = _load_esimd_utils()
    assert module._extension_candidates(tmp_path, "custom_esimd_kernels") == [main]
    assert module._extension_candidates(
        tmp_path, "custom_esimd_kernels_gemm"
    ) == [gemm]


def test_ambiguous_exact_prefix_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "custom_esimd_kernels.cpython-311-x86_64-linux-gnu.so"
    second = tmp_path / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
    first.touch()
    second.touch()

    module = _load_esimd_utils()
    with pytest.raises(ImportError, match="ambiguous"):
        module._require_single_extension(tmp_path, "custom_esimd_kernels")


def test_loader_rejects_standalone_ple_and_unknown_prefix() -> None:
    module = _load_esimd_utils()

    with pytest.raises(ImportError, match="standalone-only"):
        module.load_esimd_library("ple_ops")
    with pytest.raises(ValueError, match="unsupported"):
        module.load_esimd_library("not_a_custom_extension")


def test_default_loader_loads_canonical_main_before_gemm(tmp_path: Path, monkeypatch) -> None:
    main = tmp_path / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
    gemm = tmp_path / "custom_esimd_kernels_gemm.cpython-312-x86_64-linux-gnu.so"
    main.touch()
    gemm.touch()

    module = _load_esimd_utils()
    loaded: list[Path] = []
    monkeypatch.setattr(module, "_find_esimd_package_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "_dispatcher_has_schema", lambda schema: False)
    monkeypatch.setattr(
        module.torch.ops, "load_library", lambda path: loaded.append(Path(path))
    )
    module._ESIMD_LOADED = False

    module._load_esimd_extensions()

    assert loaded == [main, gemm]


def test_ple_standalone_loader_is_disabled_without_explicit_path(monkeypatch) -> None:
    module = _load_esimd_utils()
    monkeypatch.delenv("VLLM_XPU_QWEN38_PLE_STANDALONE_DSO", raising=False)
    assert module.load_qwen38_ple_standalone_library() is False


def test_ple_standalone_loader_rejects_relative_or_missing_path(tmp_path: Path) -> None:
    module = _load_esimd_utils()
    with pytest.raises(ImportError, match="absolute"):
        module.load_qwen38_ple_standalone_library("ple_ops.so")
    with pytest.raises(ImportError, match="does not exist"):
        module.load_qwen38_ple_standalone_library(str(tmp_path / "missing.so"))


def test_ple_standalone_loader_requires_all_short_conv_schemas(
    tmp_path: Path, monkeypatch
) -> None:
    dso = tmp_path / "ple_ops.so"
    dso.touch()
    module = _load_esimd_utils()
    loaded: list[Path] = []
    monkeypatch.setattr(module, "_dispatcher_has_schema", lambda schema: False)
    monkeypatch.setattr(
        module.torch.ops, "load_library", lambda path: loaded.append(Path(path))
    )
    with pytest.raises(ImportError, match="missing schemas"):
        module.load_qwen38_ple_standalone_library(dso)
    assert loaded == [dso]


def test_ple_standalone_loader_refuses_existing_dispatcher_owner(
    tmp_path: Path, monkeypatch
) -> None:
    dso = tmp_path / "ple_ops.so"
    dso.touch()
    module = _load_esimd_utils()
    monkeypatch.setattr(
        module, "_dispatcher_has_schema", lambda schema: schema == module._PLE_SCHEMA
    )
    with pytest.raises(ImportError, match="already owned"):
        module.load_qwen38_ple_standalone_library(dso)


def test_production_short_conv_candidate_is_explicit_and_stateful() -> None:
    ple_layer = (
        ROOT.parents[2]
        / "applications.ai.gpu.llm-scaler-vllm"
        / "vllm/models/qwen3_8_flash_next/xpu/ple_layer.py"
    ).read_text()
    assert '"VLLM_XPU_QWEN38_PLE_STANDALONE_DSO"' in ple_layer
    assert '"ple_short_conv_decode"' in ple_layer
    assert '"ple_short_conv_prefill"' in ple_layer
    assert '"ple_short_conv_spec"' in ple_layer
    assert "existing stateful short-conv path" in ple_layer
    assert "indices == NULL_BLOCK_ID" in ple_layer
    assert "torch.full_like(indices, -1)" in ple_layer
    assert "self._standalone_state_indices" in ple_layer
    assert "True,\n                -1," in ple_layer
    null_id_source = (
        ROOT.parents[2]
        / "applications.ai.gpu.llm-scaler-vllm"
        / "vllm/v1/attention/backends/utils.py"
    ).read_text()
    assert "NULL_BLOCK_ID = 0" in null_id_source


def test_projection_schemas_declare_caller_owned_outputs() -> None:
    gemv_source = (ROOT / "csrc/xpu/torch_extension.cc").read_text()
    gemm_source = (ROOT / "csrc/xpu/torch_extension_gemm.cc").read_text()

    assert (
        'esimd_gemv_fp16(Tensor input, Tensor weight, '
        'Tensor(a!) output) -> Tensor(a!)' in gemv_source
    )
    assert 'esimd_gemv_int4(Tensor input, Tensor weight, Tensor weight_scale, ' in gemv_source
    assert 'Tensor(a!) output) -> Tensor(a!)' in gemv_source
    assert 'esimd_gemm_int4_pgrp(Tensor input, Tensor weight, Tensor weight_scale, ' in gemm_source
    assert 'Tensor(a!) output) -> Tensor(a!)' in gemm_source
