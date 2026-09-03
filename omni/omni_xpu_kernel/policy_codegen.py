"""Validate the kernel-policy manifest and generate native constexpr headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = (
    PROJECT_ROOT
    / "omni_xpu_kernel"
    / "policies"
    / "kernel-policy-v1.json"
)
GENERATED_ROOT = PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "generated"
TUNING_HEADER_PATH = GENERATED_ROOT / "kernel_tuning_defaults_generated.h"
BMG_POLICY_HEADER_PATH = GENERATED_ROOT / "bmg_kernel_policy_generated.h"

EXPECTED_BUILD_TARGETS = ("bmg", "ptl-h")
EXPECTED_RUNTIME_POLICIES = ("b580", "b60", "b70", "generic-bmg")
EXPECTED_SKUS = ("b580", "b50", "b60", "b70")
EXPECTED_POLICY_STATUS = {
    "b580": ("experimental", False),
    "b60": ("experimental", False),
    "b70": ("stable", True),
    "generic-bmg": ("experimental-fallback", False),
}
EXPECTED_SKU_DISPATCH = {
    "b580": ("bmg-g21", "b580"),
    "b50": ("bmg-g21", "generic-bmg"),
    "b60": ("bmg-g21", "b60"),
    "b70": ("bmg-g31", "b70"),
}
EXPECTED_TUNING_PARAMETERS = (
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
EXPECTED_POLICY_PARAMETERS = (
    "adaln_block_size",
    "adaln_work_group_size",
    "int8_dequant_fp32_elements",
    "int8_dequant_fp32_work_group_size",
    "int8_dequant_fp16_elements",
    "int8_dequant_fp16_work_group_size",
    "int8_dequant_bf16_elements",
    "int8_dequant_bf16_work_group_size",
    "int8_scaleback_elements",
    "int8_scaleback_work_group_rows",
    "int8_scaleback_work_group_cols",
    "convrot_g16_groups_per_dpas",
    "convrot_g16_work_items_per_row",
    "fp8_stochastic_elements",
    "svdq_dequant_groups",
    "svdq_dequant_work_group_size",
    "svdq_quant_groups",
    "svdq_quant_work_group_size",
    "svdq_smooth_elements",
    "svdq_smooth_work_group_size",
    "svdq_convert_add_elements",
    "kitchen_rope_pairs_per_work_item",
    "kitchen_rope_work_group_size",
    "d120_l4205_v_tile",
    "h3_vae_d64_s1797_kv_tile",
)
EXPECTED_CANDIDATES = (
    "adaln",
    "int8-dequant-fp32",
    "int8-dequant-bf16",
    "int8-scaleback",
    "convrot-g16",
    "fp8-stochastic",
    "svdq-dequant",
    "svdq-quant",
    "svdq-smooth",
    "svdq-convert-add",
    "kitchen-rope",
    "d120-l4205-v-tile",
    "h3-vae-d64-s1797-kv-tile",
)


def _exact_keys(mapping, expected, label):
    actual = tuple(mapping)
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise RuntimeError(
            f"{label} keys differ: expected {list(expected)!r}, got {list(actual)!r}"
        )


def _validate_integer_map(mapping, expected, label):
    _exact_keys(mapping, expected, label)
    for name, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"{label}.{name} must be an integer")


def load_policy_manifest(path=MANIFEST_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _exact_keys(
        data,
        (
            "$schema",
            "schema_version",
            "kind",
            "build_tuning_profiles",
            "runtime_policies",
            "sku_profiles",
            "b580_candidates",
        ),
        "manifest",
    )
    if data["$schema"] != "kernel-policy-v1.schema.json":
        raise RuntimeError("unsupported kernel-policy schema reference")
    if data["schema_version"] != 1:
        raise RuntimeError("unsupported kernel-policy schema version")
    if data["kind"] != "omni_xpu_kernel_policy_manifest":
        raise RuntimeError("unsupported kernel-policy manifest kind")

    build_profiles = data["build_tuning_profiles"]
    _exact_keys(build_profiles, EXPECTED_BUILD_TARGETS, "build_tuning_profiles")
    for target, profile in build_profiles.items():
        _exact_keys(profile, ("policy_id", "parameters"), f"build profile {target}")
        if not isinstance(profile["policy_id"], str) or not profile["policy_id"]:
            raise RuntimeError(f"build profile {target} requires policy_id")
        _validate_integer_map(
            profile["parameters"],
            EXPECTED_TUNING_PARAMETERS,
            f"build profile {target}.parameters",
        )

    runtime_policies = data["runtime_policies"]
    _exact_keys(runtime_policies, EXPECTED_RUNTIME_POLICIES, "runtime_policies")
    for name, policy in runtime_policies.items():
        _exact_keys(
            policy,
            (
                "cpp_type",
                "policy_id",
                "status",
                "performance_claim_allowed",
                "parameters",
            ),
            f"runtime policy {name}",
        )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", policy["cpp_type"]):
            raise RuntimeError(f"invalid C++ type for runtime policy {name}")
        if not isinstance(policy["policy_id"], str) or not policy["policy_id"]:
            raise RuntimeError(f"runtime policy {name} requires policy_id")
        if not isinstance(policy["performance_claim_allowed"], bool):
            raise RuntimeError(
                f"runtime policy {name}.performance_claim_allowed must be boolean"
            )
        expected_status, expected_claim = EXPECTED_POLICY_STATUS[name]
        if (
            policy["status"] != expected_status
            or policy["performance_claim_allowed"] != expected_claim
        ):
            raise RuntimeError(
                f"runtime policy {name} must use status={expected_status!r}, "
                f"performance_claim_allowed={expected_claim!r}"
            )
        _validate_integer_map(
            policy["parameters"],
            EXPECTED_POLICY_PARAMETERS,
            f"runtime policy {name}.parameters",
        )

    sku_profiles = data["sku_profiles"]
    _exact_keys(sku_profiles, EXPECTED_SKUS, "sku_profiles")
    seen_device_ids = set()
    for sku, profile in sku_profiles.items():
        required = {"sku_id", "device_ids", "build_target", "effective_policy"}
        allowed = required | {"functional_evidence"}
        if set(profile) != required and set(profile) != allowed:
            raise RuntimeError(f"invalid fields for SKU profile {sku}")
        if profile["effective_policy"] not in runtime_policies:
            raise RuntimeError(f"SKU {sku} references an unknown runtime policy")
        expected_target, expected_policy = EXPECTED_SKU_DISPATCH[sku]
        if (
            profile["build_target"] != expected_target
            or profile["effective_policy"] != expected_policy
        ):
            raise RuntimeError(
                f"SKU {sku} must use build_target={expected_target!r}, "
                f"effective_policy={expected_policy!r}"
            )
        if not isinstance(profile["sku_id"], str) or not profile["sku_id"]:
            raise RuntimeError(f"SKU {sku} requires sku_id")
        if (
            not isinstance(profile["device_ids"], list)
            or not profile["device_ids"]
        ):
            raise RuntimeError(f"SKU {sku} requires Device IDs")
        for device_id in profile["device_ids"]:
            if (
                not isinstance(device_id, str)
                or len(device_id) != 6
                or not device_id.startswith("0x")
            ):
                raise RuntimeError(f"SKU {sku} has an invalid Device ID")
            int(device_id[2:], 16)
            if device_id in seen_device_ids:
                raise RuntimeError(f"duplicate Device ID {device_id}")
            seen_device_ids.add(device_id)
        evidence = profile.get("functional_evidence")
        if evidence is not None:
            _exact_keys(
                evidence,
                ("level", "scope", "performance_claim"),
                f"SKU {sku}.functional_evidence",
            )
            if (
                evidence["level"] != "image_passed"
                or not isinstance(evidence["scope"], str)
                or not evidence["scope"]
                or evidence["performance_claim"] is not False
            ):
                raise RuntimeError(
                    f"SKU {sku} has invalid functional evidence"
                )

    candidates = data["b580_candidates"]
    _exact_keys(candidates, EXPECTED_CANDIDATES, "b580_candidates")
    base_parameters = runtime_policies[
        sku_profiles["b580"]["effective_policy"]
    ]["parameters"]
    for name, candidate in candidates.items():
        _exact_keys(candidate, ("cpp_type", "overrides"), f"candidate {name}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", candidate["cpp_type"]):
            raise RuntimeError(f"invalid C++ type for candidate {name}")
        if not candidate["overrides"]:
            raise RuntimeError(f"candidate {name} requires at least one override")
        differs_from_base = False
        for field, value in candidate["overrides"].items():
            if field not in base_parameters:
                raise RuntimeError(f"candidate {name} has unknown field {field}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(f"candidate {name}.{field} must be an integer")
            differs_from_base = differs_from_base or value != base_parameters[field]
        if not differs_from_base:
            raise RuntimeError(
                f"candidate {name} does not differ from its base policy"
            )
    return data


def manifest_sha256(manifest):
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def render_tuning_header(manifest):
    profiles = manifest["build_tuning_profiles"]
    digest = manifest_sha256(manifest)
    lines = [
        "// Generated by policy_codegen.py from kernel-policy-v1.json.",
        "// Do not edit this file directly.",
        "#pragma once",
        "",
        "#if defined(OMNI_XPU_ARCH_BMG)",
    ]
    for target in ("bmg", "ptl-h"):
        if target != "bmg":
            lines.extend(["#elif defined(OMNI_XPU_ARCH_PTL_H)"])
        profile = profiles[target]
        lines.append(f'#define OMNI_BUILD_TUNING_POLICY_ID "{profile["policy_id"]}"')
        for name, value in profile["parameters"].items():
            suffix = name.removeprefix("OMNI_")
            lines.append(f"#define OMNI_MAINTAINED_{suffix} {value}")
    lines.extend(
        [
            "#else",
            '#error "Define exactly one supported XPU architecture"',
            "#endif",
            "",
        ]
    )
    for name in EXPECTED_TUNING_PARAMETERS:
        suffix = name.removeprefix("OMNI_")
        lines.extend(
            [
                f"#ifndef {name}",
                f"#define {name} OMNI_MAINTAINED_{suffix}",
                "#endif",
                "",
            ]
        )
    lines.extend(
        [
            "namespace omni_xpu {",
            "namespace tuning {",
            "",
            f'inline constexpr const char* policy_manifest_sha256 = "{digest}";',
            "inline constexpr const char* build_tuning_policy_id =",
            "    OMNI_BUILD_TUNING_POLICY_ID;",
            "",
            "inline constexpr bool is_candidate_build() {",
            "    return",
        ]
    )
    comparisons = []
    for name in EXPECTED_TUNING_PARAMETERS:
        suffix = name.removeprefix("OMNI_")
        comparisons.append(f"        {name} != OMNI_MAINTAINED_{suffix}")
    lines.append(" ||\n".join(comparisons) + ";")
    lines.extend(
        [
            "}",
            "",
            "}  // namespace tuning",
            "}  // namespace omni_xpu",
            "",
        ]
    )
    return "\n".join(lines)


def render_bmg_policy_header(manifest):
    policies = manifest["runtime_policies"]
    sku_profiles = manifest["sku_profiles"]
    digest = manifest_sha256(manifest)
    lines = [
        "// Generated by policy_codegen.py from kernel-policy-v1.json.",
        "// Do not edit this file directly.",
        "#pragma once",
        "",
        "namespace omni_xpu {",
        "namespace device {",
        "",
    ]
    for name in EXPECTED_RUNTIME_POLICIES:
        policy = policies[name]
        lines.append(f'struct {policy["cpp_type"]} {{')
        for field, value in policy["parameters"].items():
            lines.append(f"    static constexpr int {field} = {value};")
        lines.extend(["};", ""])

    b580_base = policies[sku_profiles["b580"]["effective_policy"]]["cpp_type"]
    for candidate in manifest["b580_candidates"].values():
        lines.append(f'struct {candidate["cpp_type"]} : {b580_base} {{')
        for field, value in candidate["overrides"].items():
            lines.append(f"    static constexpr int {field} = {value};")
        lines.extend(["};", ""])

    profile_cases = {
        "b580": policies["b580"],
        "b60": policies["b60"],
        "b70": policies["b70"],
        "generic_bmg": policies["generic-bmg"],
    }
    lines.extend(
        [
            f'inline constexpr const char* policy_manifest_sha256 = "{digest}";',
            "",
            "inline constexpr const char* kernel_policy_id(",
            "        BmgKernelProfile profile) {",
            "    switch (profile) {",
        ]
    )
    for enum_name, policy in profile_cases.items():
        lines.extend(
            [
                f"        case BmgKernelProfile::{enum_name}:",
                f'            return "{policy["policy_id"]}";',
            ]
        )
    lines.extend(["    }", '    return "unknown";', "}", ""])
    lines.extend(
        [
            "inline constexpr const char* kernel_policy_status(",
            "        BmgKernelProfile profile) {",
            "    switch (profile) {",
        ]
    )
    for enum_name, policy in profile_cases.items():
        lines.extend(
            [
                f"        case BmgKernelProfile::{enum_name}:",
                f'            return "{policy["status"]}";',
            ]
        )
    lines.extend(["    }", '    return "unknown";', "}", ""])
    lines.extend(
        [
            "inline constexpr bool kernel_policy_performance_claim_allowed(",
            "        BmgKernelProfile profile) {",
            "    switch (profile) {",
        ]
    )
    for enum_name, policy in profile_cases.items():
        value = "true" if policy["performance_claim_allowed"] else "false"
        lines.extend(
            [
                f"        case BmgKernelProfile::{enum_name}:",
                f"            return {value};",
            ]
        )
    lines.extend(["    }", "    return false;", "}", ""])

    lines.extend(
        [
            "inline constexpr const char* bmg_sku_profile_id(BmgSku sku) {",
            "    switch (sku) {",
        ]
    )
    for sku in EXPECTED_SKUS:
        lines.extend(
            [
                f"        case BmgSku::{sku}:",
                f'            return "{sku_profiles[sku]["sku_id"]}";',
            ]
        )
    lines.extend(["        default:", '            return "bmg-unknown";', "    }", "}", ""])

    lines.extend(
        [
            "inline constexpr const char* bmg_sku_build_target(BmgSku sku) {",
            "    switch (sku) {",
        ]
    )
    for sku in EXPECTED_SKUS:
        lines.extend(
            [
                f"        case BmgSku::{sku}:",
                f'            return "{sku_profiles[sku]["build_target"]}";',
            ]
        )
    lines.extend(["        default:", '            return "unknown";', "    }", "}", ""])
    lines.extend(["}  // namespace device", "}  // namespace omni_xpu", ""])
    return "\n".join(lines)


def generated_files(manifest):
    return {
        TUNING_HEADER_PATH: render_tuning_header(manifest),
        BMG_POLICY_HEADER_PATH: render_bmg_policy_header(manifest),
    }


def write_generated_headers(manifest=None):
    manifest = load_policy_manifest() if manifest is None else manifest
    for path, text in generated_files(manifest).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def verify_generated_headers(manifest=None):
    manifest = load_policy_manifest() if manifest is None else manifest
    stale = []
    for path, expected in generated_files(manifest).items():
        actual = path.read_text(encoding="utf-8") if path.is_file() else None
        if actual != expected:
            stale.append(path.relative_to(PROJECT_ROOT).as_posix())
    if stale:
        raise RuntimeError(
            "generated kernel-policy headers are stale; run "
            f"{Path(__file__).name}: {', '.join(stale)}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_policy_manifest()
    if args.check:
        verify_generated_headers(manifest)
    else:
        write_generated_headers(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
