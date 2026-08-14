"""Apply opt-in XPU allocator policy before ComfyUI imports torch."""

from __future__ import annotations

import logging
import math
import os
from typing import NoReturn


_ENV_NAME = "OMNIXPU_XPU_MEMORY_FRACTION"
_MASTER_ENV_NAME = "OMNIXPU_ENABLE"
_LOG = logging.getLogger("ComfyUI-OmniXPU")


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"[OmniXPU] {_ENV_NAME}: {message}")


def _parse_fraction(raw_value: str) -> float:
    try:
        fraction = float(raw_value)
    except ValueError:
        _fail(f"expected a number in (0, 1], got {raw_value!r}")

    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        _fail(f"expected a finite number in (0, 1], got {raw_value!r}")
    return fraction


def apply_xpu_memory_fraction_from_env() -> float | None:
    """Apply the requested caching-allocator fraction, if explicitly set."""

    raw_value = os.environ.get(_ENV_NAME)
    if raw_value is None or not raw_value.strip():
        return None

    if os.environ.get(_MASTER_ENV_NAME, "1") == "0":
        _LOG.info(
            "[OmniXPU] XPU allocator memory fraction skipped because %s=0",
            _MASTER_ENV_NAME,
        )
        return None

    fraction = _parse_fraction(raw_value.strip())

    try:
        import torch
    except Exception as exc:
        _fail(f"could not import torch: {exc}")

    xpu = getattr(torch, "xpu", None)
    try:
        available = xpu is not None and xpu.is_available()
    except Exception as exc:
        _fail(f"could not query torch.xpu availability: {exc}")
    if not available:
        _fail("was set, but torch.xpu is unavailable")

    setter = getattr(xpu, "set_per_process_memory_fraction", None)
    if not callable(setter):
        _fail("is unsupported by this PyTorch build")

    try:
        setter(fraction)
        getter = getattr(xpu, "get_per_process_memory_fraction", None)
        actual = float(getter()) if callable(getter) else fraction
    except Exception as exc:
        _fail(f"could not apply {fraction}: {exc}")

    _LOG.info(
        "[OmniXPU] XPU allocator memory fraction applied during prestartup: "
        "requested=%.12g actual=%.12g",
        fraction,
        actual,
    )
    return actual


apply_xpu_memory_fraction_from_env()
