"""Exact Intel BMG device identity used by native kernel dispatch."""

from __future__ import annotations

from . import _load_extension


def classify_bmg_device_id(device_id: int) -> str:
    """Map an exact PCI Product Device ID to ``b60``, ``b70`` or ``unknown``.

    The B60 kernel profile intentionally covers both the validated G21 E210
    device and the public Arc Pro B60 E211 product ID.
    """

    return str(_load_extension().device.classify_bmg_device_id(device_id))


def info(index: int = 0) -> dict[str, object]:
    """Return native identity and the selected runtime kernel profile."""

    return dict(_load_extension().device.info(index))


def bmg_sku(index: int = 0) -> str:
    """Return the BMG kernel profile selected for one Torch XPU device."""

    return str(_load_extension().device.bmg_sku(index))


__all__ = ["bmg_sku", "classify_bmg_device_id", "info"]
