"""Contracts for generated build defaults and per-SKU runtime policies."""

from pathlib import Path
from runpy import run_path

import pytest

from omni_xpu_kernel import device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEGEN = run_path(str(PROJECT_ROOT / "policy_codegen.py"))


def test_checked_in_policy_headers_match_manifest():
    manifest = CODEGEN["load_policy_manifest"]()

    CODEGEN["verify_generated_headers"](manifest)


def test_b580_defaults_have_independent_experimental_policy_identity():
    manifest = device.policy_manifest()
    b580_profile = manifest["sku_profiles"]["b580"]
    b580 = manifest["runtime_policies"]["b580"]
    b70 = manifest["runtime_policies"]["b70"]

    assert b580_profile["effective_policy"] == "b580"
    assert b580["cpp_type"] == "B580KernelPolicy"
    assert b580["policy_id"] == "b580-v1"
    assert b580["status"] == "experimental"
    assert b580["performance_claim_allowed"] is False
    # The accepted values currently match the maintained defaults, but remain
    # copied into a B580-owned record so later SKU tuning cannot drift by alias.
    assert b580["parameters"] == b70["parameters"]
    assert b580_profile["functional_evidence"]["level"] == "functional_pass"
    assert b580_profile["functional_evidence"]["performance_claim"] is False


def test_only_b70_runtime_policy_is_stable_and_performance_eligible():
    policies = device.policy_manifest()["runtime_policies"]

    assert {
        name
        for name, policy in policies.items()
        if policy["status"] == "stable"
    } == {"b70"}
    assert {
        name
        for name, policy in policies.items()
        if policy["performance_claim_allowed"]
    } == {"b70"}


def test_policy_defaults_resolve_each_sku_without_aliasing_mutable_data():
    b580 = device.policy_defaults(" B580 ")
    b70 = device.policy_defaults("b70")
    b50 = device.policy_defaults("b50")

    assert b580["effective_policy"] == "b580"
    assert b580["sku_profile_id"] == "bmg-b580"
    assert b580["build_target"] == "bmg-g21"
    assert b580["support_status"] == "experimental"
    assert b580["performance_claim_allowed"] is False
    assert b580["parameters"] == b70["parameters"]
    assert b580["parameters"] is not b70["parameters"]
    assert b50["effective_policy"] == "generic-bmg"

    b580["parameters"]["adaln_block_size"] = -1
    assert device.policy_defaults("b580")["parameters"][
        "adaln_block_size"
    ] == 32

    with pytest.raises(ValueError, match="Unknown BMG SKU"):
        device.policy_defaults("b90")
