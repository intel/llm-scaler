"""Exact Intel BMG device identity used by native kernel dispatch."""

from __future__ import annotations

from . import _load_extension


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
    "physical_bmg_sku",
]
