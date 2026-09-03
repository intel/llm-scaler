"""Exact Intel BMG device identity used by native kernel dispatch."""

from __future__ import annotations

import json
from importlib.resources import files

from . import _load_extension


_POLICY_MANIFEST_RESOURCE = "policies/kernel-policy-v1.json"


def policy_manifest() -> dict[str, object]:
    """Return the packaged build/runtime policy manifest.

    A fresh object is decoded for every call so callers cannot mutate process-
    global policy metadata. Native dispatch remains owned by the generated C++
    policy compiled into the extension.
    """

    resource = files("omni_xpu_kernel").joinpath(_POLICY_MANIFEST_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


def policy_defaults(sku: str) -> dict[str, object]:
    """Return the separately recorded default runtime policy for one BMG SKU."""

    normalized = str(sku).strip().lower()
    manifest = policy_manifest()
    profiles = manifest["sku_profiles"]
    if normalized not in profiles:
        supported = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown BMG SKU {sku!r}; expected one of: {supported}"
        )
    profile = profiles[normalized]
    policy_name = profile["effective_policy"]
    policy = manifest["runtime_policies"][policy_name]
    return {
        "sku": normalized,
        "sku_profile_id": profile["sku_id"],
        "build_target": profile["build_target"],
        "effective_policy": policy_name,
        "policy_id": policy["policy_id"],
        "support_status": policy["status"],
        "performance_claim_allowed": policy[
            "performance_claim_allowed"
        ],
        "parameters": dict(policy["parameters"]),
    }


def classify_bmg_device_id(device_id: int) -> str:
    """Map an exact PCI Product Device ID to a supported BMG SKU identity.

    The B60 kernel profile intentionally covers both the validated G21 E210
    device and the public Arc Pro B60 E211 product ID.
    """

    return str(_load_extension().device.classify_bmg_device_id(device_id))


def info(index: int = 0) -> dict[str, object]:
    """Return identity, policy, and exact compiled tuning-control values."""

    return dict(_load_extension().device.info(index))


def bmg_sku(index: int = 0) -> str:
    """Return the effective BMG SKU after an optional debug override."""

    return str(_load_extension().device.bmg_sku(index))


def physical_bmg_sku(index: int = 0) -> str:
    """Return the exact physical BMG SKU; debug overrides never change it."""

    return str(_load_extension().device.physical_bmg_sku(index))


def kernel_profile(index: int = 0) -> str:
    """Return the effective native kernel profile for one Torch XPU device."""

    return str(_load_extension().device.kernel_profile(index))


def b580_policy_candidate(index: int = 0) -> str:
    """Return the active development-only B580 policy candidate axis."""

    return str(_load_extension().device.b580_policy_candidate(index))


__all__ = [
    "b580_policy_candidate",
    "bmg_sku",
    "classify_bmg_device_id",
    "info",
    "kernel_profile",
    "policy_defaults",
    "policy_manifest",
    "physical_bmg_sku",
]
