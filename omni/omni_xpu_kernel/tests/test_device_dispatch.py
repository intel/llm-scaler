from pathlib import Path
import re
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

import omni_xpu_kernel
from omni_xpu_kernel import cute
from omni_xpu_kernel import device


TUNING_OVERRIDE_NAMES = (
    "OMNI_FP8_DEQUANT_ELEMENTS_PER_WI",
    "OMNI_FP8_QUANT_VEC",
    "OMNI_FP8_STOCHASTIC_ELEMENTS_PER_WORK_ITEM",
    "OMNI_CONVROT_DEQUANT_WG_SIZE",
    "OMNI_CONVROT_QUANT_WG_SIZE",
    "OMNI_INT8_DEQUANT_ELEMENTS_PER_WI",
    "OMNI_SILU_MUL_ELEMENTS_PER_WI",
    "OMNI_INT8_TENSORWISE_VEC",
    "OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE",
    "OMNI_KITCHEN_ROPE_PAIR_WG_SIZE",
    "OMNI_SVDQ_DEQUANT_GROUPS_PER_WI",
    "OMNI_SVDQ_QUANT_GROUPS_PER_WI",
    "OMNI_SVDQ_UNPACK_COLS_PER_WI",
    "OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION",
    "OMNI_SVDQ_UNPACK_WG_SIZE",
    "OMNI_RMS_NORM_H120_MODE",
    "OMNI_RMS_NORM_H128_BLOCK_SIZE",
    "OMNI_GROUP_NORM_BMG_TILE",
    "OMNI_GROUP_NORM_BMG_REDUCE_VECTOR",
    "OMNI_H3_RMS_ROPE_FAST_REDUCE",
    "OMNI_H3_RMS_ROPE_SLM_BF16",
    "OMNI_ROWQ_VECTOR_WIDTH_OVERRIDE",
    "OMNI_ROWQ_SUBGROUPS_PER_ROW_OVERRIDE",
)

TUNING_DEFAULTS = {
    "ptl-h": (
        256, 16, 8, 1, 1, 64, 1, 16, 1, 128, 60, 60, 3840, 64,
        32, 2, 32, 32768, 32, 0, 0, 0, 0,
    ),
    "bmg": (
        256, 8, 6, 1, 8, 32, 1, 8, 1, 32, 60, 60, 3840, 128,
        1, 2, 64, 32768, 32, 0, 0, 0, 0,
    ),
}


@pytest.mark.parametrize(
    ("device_id", "expected"),
    [
        (0xE210, "b60"),
        (0xE211, "b60"),
        (0xE20B, "b580"),
        (0xE212, "b50"),
        (0xE223, "b70"),
        (0xFFFF, "unknown"),
    ],
)
def test_python_device_classifier_uses_exact_ids(
    monkeypatch, device_id, expected
):
    native = SimpleNamespace(
        device=SimpleNamespace(
            classify_bmg_device_id=lambda value: {
                0xE210: "b60",
                0xE211: "b60",
                0xE20B: "b580",
                0xE212: "b50",
                0xE223: "b70",
            }.get(value, "unknown")
        )
    )
    monkeypatch.setattr(device, "_load_extension", lambda: native)

    assert device.classify_bmg_device_id(device_id) == expected


def test_device_info_and_sku_forward_to_native(monkeypatch):
    native = SimpleNamespace(
        device=SimpleNamespace(
            info=lambda index: {
                "index": index,
                "device_id": 0xE210,
                "physical_bmg_sku": "b60",
                "bmg_sku": "b60",
                "kernel_profile": "b60",
                "b580_policy_candidate": "none",
                "sku_forced": False,
                "performance_claim_allowed": True,
                "tuning_overrides": {
                    "OMNI_RMS_NORM_H120_MODE": 2,
                },
            },
            bmg_sku=lambda index: "b60" if index == 1 else "b70",
            physical_bmg_sku=lambda index: "b60" if index == 1 else "b70",
            kernel_profile=lambda index: "b60" if index == 1 else "b70",
            b580_policy_candidate=lambda index: "none",
        )
    )
    monkeypatch.setattr(device, "_load_extension", lambda: native)

    assert device.info(3) == {
        "index": 3,
        "device_id": 0xE210,
        "physical_bmg_sku": "b60",
        "bmg_sku": "b60",
        "kernel_profile": "b60",
        "b580_policy_candidate": "none",
        "sku_forced": False,
        "performance_claim_allowed": True,
        "tuning_overrides": {
            "OMNI_RMS_NORM_H120_MODE": 2,
        },
    }
    assert device.bmg_sku(1) == "b60"
    assert device.bmg_sku(0) == "b70"
    assert device.physical_bmg_sku(1) == "b60"
    assert device.physical_bmg_sku(0) == "b70"
    assert device.kernel_profile(1) == "b60"
    assert device.kernel_profile(0) == "b70"
    assert device.b580_policy_candidate(1) == "none"


def test_native_source_contract_covers_identity_profile_and_override():
    package_root = Path(__file__).resolve().parents[1]
    policy = (
        package_root / "omni_xpu_kernel/csrc/bmg_device_policy.h"
    ).read_text(encoding="utf-8")
    utilities = (
        package_root / "omni_xpu_kernel/csrc/device_utils.h"
    ).read_text(encoding="utf-8")
    warning = (
        package_root / "omni_xpu_kernel/csrc/bmg_device_warning.h"
    ).read_text(encoding="utf-8")

    for constant, value in (
        ("kArcB580", "0xE20B"),
        ("kArcProB50", "0xE212"),
        ("kArcProB60", "0xE211"),
        ("kArcProB70", "0xE223"),
    ):
        assert f"{constant} = {value}" in policy
    for sku in ("b580", "b50", "b60", "b70"):
        assert f'"{sku}"' in policy
    assert 'value != "generic"' in policy
    assert "resolve_bmg_selection" in policy
    assert 'std::getenv("OMNI_XPU_FORCE_SKU")' in utilities
    assert 'std::getenv("OMNI_XPU_B580_POLICY_CANDIDATE")' in utilities
    assert "warn_bmg_selection_once(device_id, selection)" in utilities
    assert "debug/prescreen only, performance_claim=false" in warning
    assert "SKU-specific kernel policy is unvalidated" in warning


def test_tuning_override_surface_is_centralized_and_exported():
    package_root = Path(__file__).resolve().parents[1]
    csrc_root = package_root / "omni_xpu_kernel/csrc"
    header = (csrc_root / "kernel_tuning_overrides.h").read_text(
        encoding="utf-8"
    )
    bindings = (csrc_root / "bindings.cpp").read_text(encoding="utf-8")

    defined_names = tuple(re.findall(r"^#ifndef (OMNI_[A-Z0-9_]+)$", header, re.M))
    assert defined_names == TUNING_OVERRIDE_NAMES
    assert 'result["tuning_overrides"]' in bindings
    compact_bindings = re.sub(r"\s+", "", bindings)
    for name in TUNING_OVERRIDE_NAMES:
        assert f"OMNI_EXPORT_TUNING_OVERRIDE({name});" in compact_bindings

    for source in csrc_root.glob("*.cpp"):
        assert "#ifndef OMNI_" not in source.read_text(encoding="utf-8")

    for source_name in (
        "bindings.cpp",
        "fp8_dequant_esimd.cpp",
        "fp8_quant.cpp",
        "group_norm_bmg.cpp",
        "int8_convrot_dequant_esimd.cpp",
        "int8_convrot_quant_esimd.cpp",
        "int8_dequantize_esimd.cpp",
        "int8_quantize_esimd.cpp",
        "int8_tensorwise_sycl.cpp",
        "kitchen_rms_rope_sycl.cpp",
        "kitchen_rope_sycl.cpp",
        "norm.cpp",
        "svdq_dequant.cpp",
    ):
        source = (csrc_root / source_name).read_text(encoding="utf-8")
        assert '#include "kernel_tuning_overrides.h"' in source


@pytest.mark.parametrize("target", ["ptl-h", "bmg"])
def test_tuning_override_defaults_compile_for_each_target(tmp_path, target):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the header contract")

    package_root = Path(__file__).resolve().parents[1]
    csrc_root = package_root / "omni_xpu_kernel/csrc"
    architecture = (
        "OMNI_XPU_ARCH_PTL_H" if target == "ptl-h" else "OMNI_XPU_ARCH_BMG"
    )
    assertions = "\n".join(
        f"static_assert({name} == {value});"
        for name, value in zip(TUNING_OVERRIDE_NAMES, TUNING_DEFAULTS[target])
    )
    source = f'#include "kernel_tuning_overrides.h"\n{assertions}\nint main() {{}}\n'
    executable = tmp_path / f"tuning_defaults_{target}"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-D{architecture}=1",
            "-I",
            str(csrc_root),
            "-x",
            "c++",
            "-",
            "-o",
            str(executable),
        ],
        input=source,
        text=True,
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_tuning_override_nondefaults_compile(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the header contract")

    package_root = Path(__file__).resolve().parents[1]
    csrc_root = package_root / "omni_xpu_kernel/csrc"
    overrides = {
        "OMNI_RMS_NORM_H120_MODE": 1,
        "OMNI_RMS_NORM_H128_BLOCK_SIZE": 16,
        "OMNI_GROUP_NORM_BMG_TILE": 8192,
        "OMNI_GROUP_NORM_BMG_REDUCE_VECTOR": 16,
        "OMNI_H3_RMS_ROPE_FAST_REDUCE": 1,
        "OMNI_H3_RMS_ROPE_SLM_BF16": 1,
        "OMNI_ROWQ_VECTOR_WIDTH_OVERRIDE": 8,
        "OMNI_ROWQ_SUBGROUPS_PER_ROW_OVERRIDE": 4,
    }
    assertions = "\n".join(
        f"static_assert({name} == {value});"
        for name, value in overrides.items()
    )
    source = f'#include "kernel_tuning_overrides.h"\n{assertions}\nint main() {{}}\n'
    executable = tmp_path / "tuning_nondefaults"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DOMNI_XPU_ARCH_BMG=1",
            *(f"-D{name}={value}" for name, value in overrides.items()),
            "-I",
            str(csrc_root),
            "-x",
            "c++",
            "-",
            "-o",
            str(executable),
        ],
        input=source,
        text=True,
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_device_independent_native_selection_compiles_and_runs(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the header contract")

    package_root = Path(__file__).resolve().parents[1]
    csrc_root = package_root / "omni_xpu_kernel/csrc"
    executable = tmp_path / "bmg_device_policy_test"
    source = textwrap.dedent(
        r"""
        #include "bmg_device_policy.h"
        #include "bmg_device_warning.h"
        #include "bmg_kernel_policy.h"
        #include <cassert>
        #include <iostream>
        #include <sstream>
        #include <stdexcept>
        #include <string>

        using namespace omni_xpu::device;

        int main() {
            static_assert(classify_bmg_device_id(kArcB580) == BmgSku::b580);
            static_assert(classify_bmg_device_id(kArcProB50) == BmgSku::b50);
            static_assert(classify_bmg_device_id(kArcProB60) == BmgSku::b60);
            static_assert(classify_bmg_device_id(kArcProB70) == BmgSku::b70);
            static_assert(classify_bmg_device_id(0xFFFF) == BmgSku::unknown);

            #define ASSERT_GENERIC_MATCHES_B70(field) \
                static_assert( \
                    GenericBmgKernelPolicy::field == B70KernelPolicy::field)
            ASSERT_GENERIC_MATCHES_B70(adaln_block_size);
            ASSERT_GENERIC_MATCHES_B70(adaln_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(int8_dequant_fp32_elements);
            ASSERT_GENERIC_MATCHES_B70(int8_dequant_fp32_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(int8_dequant_fp16_elements);
            ASSERT_GENERIC_MATCHES_B70(int8_dequant_fp16_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(int8_dequant_bf16_elements);
            ASSERT_GENERIC_MATCHES_B70(int8_dequant_bf16_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(int8_scaleback_elements);
            ASSERT_GENERIC_MATCHES_B70(int8_scaleback_work_group_rows);
            ASSERT_GENERIC_MATCHES_B70(int8_scaleback_work_group_cols);
            ASSERT_GENERIC_MATCHES_B70(convrot_g16_groups_per_dpas);
            ASSERT_GENERIC_MATCHES_B70(convrot_g16_work_items_per_row);
            ASSERT_GENERIC_MATCHES_B70(fp8_stochastic_elements);
            ASSERT_GENERIC_MATCHES_B70(svdq_dequant_groups);
            ASSERT_GENERIC_MATCHES_B70(svdq_dequant_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(svdq_quant_groups);
            ASSERT_GENERIC_MATCHES_B70(svdq_quant_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(svdq_smooth_elements);
            ASSERT_GENERIC_MATCHES_B70(svdq_smooth_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(svdq_convert_add_elements);
            ASSERT_GENERIC_MATCHES_B70(kitchen_rope_pairs_per_work_item);
            ASSERT_GENERIC_MATCHES_B70(kitchen_rope_work_group_size);
            ASSERT_GENERIC_MATCHES_B70(d120_l4205_v_tile);
            #undef ASSERT_GENERIC_MATCHES_B70

            const auto detected = resolve_bmg_selection(kArcB580, nullptr);
            assert(detected.physical_sku == BmgSku::b580);
            assert(detected.effective_sku == BmgSku::b580);
            assert(detected.kernel_profile == BmgKernelProfile::generic_bmg);
            assert(!detected.forced);
            assert(detected.b580_policy_candidate ==
                   B580PolicyCandidate::none);

            const auto candidate =
                resolve_bmg_selection(kArcB580, nullptr, "adaln");
            assert(candidate.physical_sku == BmgSku::b580);
            assert(candidate.effective_sku == BmgSku::b580);
            assert(candidate.kernel_profile == BmgKernelProfile::generic_bmg);
            assert(!candidate.forced);
            assert(candidate.b580_policy_candidate ==
                   B580PolicyCandidate::adaln);
            assert(b580_policy_candidate_name(
                       candidate.b580_policy_candidate) == "adaln");
            static_assert(
                B580AdalnCandidatePolicy::adaln_block_size ==
                B60KernelPolicy::adaln_block_size);
            static_assert(
                B580AdalnCandidatePolicy::int8_scaleback_elements ==
                GenericBmgKernelPolicy::int8_scaleback_elements);

            const auto forced = resolve_bmg_selection(kArcProB70, "b60");
            assert(forced.physical_sku == BmgSku::b70);
            assert(forced.effective_sku == BmgSku::b60);
            assert(forced.kernel_profile == BmgKernelProfile::b60);
            assert(forced.forced);

            const auto generic = resolve_bmg_selection(kArcProB70, "generic");
            assert(generic.physical_sku == BmgSku::b70);
            assert(generic.effective_sku == BmgSku::unknown);
            assert(generic.kernel_profile == BmgKernelProfile::generic_bmg);
            assert(generic.forced);

            bool rejected = false;
            try {
                (void)resolve_bmg_selection(kArcProB70, "not-a-sku");
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            assert(rejected);

            rejected = false;
            try {
                (void)resolve_bmg_selection(
                    kArcProB70, nullptr, "adaln");
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            assert(rejected);

            rejected = false;
            try {
                (void)resolve_bmg_selection(
                    kArcB580, "b60", "adaln");
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            assert(rejected);

            rejected = false;
            try {
                (void)resolve_bmg_selection(
                    kArcB580, nullptr, "not-a-candidate");
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            assert(rejected);

            std::ostringstream warnings;
            auto* previous = std::cerr.rdbuf(warnings.rdbuf());
            warn_bmg_selection_once(kArcB580, detected);
            warn_bmg_selection_once(kArcB580, detected);
            warn_bmg_selection_once(kArcProB70, forced);
            warn_bmg_selection_once(kArcProB70, forced);
            warn_bmg_selection_once(kArcB580, candidate);
            warn_bmg_selection_once(kArcB580, candidate);
            const auto normal_b70 = resolve_bmg_selection(kArcProB70, nullptr);
            warn_bmg_selection_once(kArcProB70, normal_b70);
            std::cerr.rdbuf(previous);

            const std::string text = warnings.str();
            const auto generic_at = text.find("uses kernel_profile=generic-bmg");
            assert(generic_at != std::string::npos);
            assert(text.find("uses kernel_profile=generic-bmg", generic_at + 1) ==
                   std::string::npos);
            const auto forced_at = text.find("OMNI_XPU_FORCE_SKU overrides");
            assert(forced_at != std::string::npos);
            assert(text.find("OMNI_XPU_FORCE_SKU overrides", forced_at + 1) ==
                   std::string::npos);
            assert(text.find("performance_claim=false") != std::string::npos);
            const auto candidate_at = text.find(
                "OMNI_XPU_B580_POLICY_CANDIDATE=adaln");
            assert(candidate_at != std::string::npos);
            assert(text.find(
                       "OMNI_XPU_B580_POLICY_CANDIDATE=adaln",
                       candidate_at + 1) == std::string::npos);
            return 0;
        }
        """
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(csrc_root),
            "-x",
            "c++",
            "-",
            "-o",
            str(executable),
        ],
        input=source,
        text=True,
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_native_info_keeps_physical_and_effective_identity_separate():
    package_root = Path(__file__).resolve().parents[1]
    bindings = (
        package_root / "omni_xpu_kernel/csrc/bindings.cpp"
    ).read_text(encoding="utf-8")

    for field in (
        'result["physical_bmg_sku"]',
        'result["bmg_sku"]',
        'result["sku_forced"]',
        'result["kernel_profile"]',
        'result["b580_policy_candidate"]',
        'result["performance_claim_allowed"]',
    ):
        assert field in bindings


def test_cute_sidecar_delegates_warning_ownership_to_core():
    package_root = Path(__file__).resolve().parents[1]
    sidecar = (
        package_root / "omni_xpu_kernel/cute/cute_fmha_torch.cpp"
    ).read_text(encoding="utf-8")
    wrapper = (
        package_root / "omni_xpu_kernel/cute/__init__.py"
    ).read_text(encoding="utf-8")

    assert "get_bmg_selection_unwarned(queue)" in sidecar
    assert "_prepare_bmg_policy_dispatch(q)" in wrapper
    assert "device.info(" in wrapper


def test_cute_warning_preparation_is_cached_by_device_and_override(
    monkeypatch,
):
    if cute is None:
        pytest.skip("CUTE Python wrapper is unavailable")

    calls = []
    tensor = SimpleNamespace(device=SimpleNamespace(index=2))
    monkeypatch.setattr(omni_xpu_kernel, "__xpu_target__", "bmg")
    monkeypatch.setattr(device, "info", lambda index: calls.append(index) or {})
    cute._prepared_bmg_policy_dispatches.clear()

    cute._prepare_bmg_policy_dispatch(tensor)
    cute._prepare_bmg_policy_dispatch(tensor)
    assert calls == [2]

    monkeypatch.setenv("OMNI_XPU_FORCE_SKU", "b60")
    cute._prepare_bmg_policy_dispatch(tensor)
    cute._prepare_bmg_policy_dispatch(tensor)
    assert calls == [2, 2]

    monkeypatch.delenv("OMNI_XPU_FORCE_SKU")
    monkeypatch.setenv("OMNI_XPU_B580_POLICY_CANDIDATE", "adaln")
    cute._prepare_bmg_policy_dispatch(tensor)
    cute._prepare_bmg_policy_dispatch(tensor)
    assert calls == [2, 2, 2]
    cute._prepared_bmg_policy_dispatches.clear()


def test_generic_bmg_policy_is_independent_from_b70():
    package_root = Path(__file__).resolve().parents[1]
    policy = (
        package_root / "omni_xpu_kernel/csrc/bmg_kernel_policy.h"
    ).read_text(encoding="utf-8")

    assert "struct GenericBmgKernelPolicy" in policy
    assert "using GenericBmgKernelPolicy = B70KernelPolicy" not in policy
    for source in (
        "int8_scaleback_esimd.cpp",
        "int8_quantize_esimd.cpp",
        "svdq_dequant.cpp",
        "svdq_fused_postproc.cpp",
    ):
        contents = (
            package_root / f"omni_xpu_kernel/csrc/{source}"
        ).read_text(encoding="utf-8")
        assert "GenericBmgKernelPolicy" in contents


def test_b580_candidate_axes_are_explicit_and_route_local():
    package_root = Path(__file__).resolve().parents[1]
    csrc_root = package_root / "omni_xpu_kernel/csrc"
    policy = (csrc_root / "bmg_device_policy.h").read_text(encoding="utf-8")
    kernel_policy = (csrc_root / "bmg_kernel_policy.h").read_text(
        encoding="utf-8"
    )

    routes = {
        "adaln": "adaln.cpp",
        "int8_dequant_fp32": "int8_dequantize_esimd.cpp",
        "int8_dequant_bf16": "int8_dequantize_esimd.cpp",
        "int8_scaleback": "int8_scaleback_esimd.cpp",
        "convrot_g16": "int8_quantize_esimd.cpp",
        "fp8_stochastic": "fp8_quant.cpp",
        "svdq_dequant": "svdq_dequant.cpp",
        "svdq_quant": "svdq_dequant.cpp",
        "svdq_smooth": "svdq_fused_postproc.cpp",
        "svdq_convert_add": "svdq_fused_postproc.cpp",
        "kitchen_rope": "kitchen_rope_sycl.cpp",
    }
    for candidate, source_name in routes.items():
        assert f"B580PolicyCandidate::{candidate}" in policy
        assert f"B580PolicyCandidate::{candidate}" in (
            csrc_root / source_name
        ).read_text(encoding="utf-8")

    cute_source = (
        package_root / "omni_xpu_kernel/cute/cute_fmha_torch.cpp"
    ).read_text(encoding="utf-8")
    assert "B580PolicyCandidate::d120_l4205_v_tile" in policy
    assert "B580PolicyCandidate::d120_l4205_v_tile" in cute_source
    assert "B580D120L4205CandidatePolicy" in kernel_policy
    assert "B580CandidateKernelPolicy" in kernel_policy
