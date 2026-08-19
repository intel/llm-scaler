"""Classify accelerator failures that must not enter an eager fallback path."""

from __future__ import annotations

import torch


_FATAL_ACCELERATOR_ERROR_MARKERS = (
    "out of memory",
    "insufficient device memory",
    "cannot allocate memory",
    "can't allocate memory",
    "pi_error_out_of_resources",
    "pi_error_device_lost",
    "ur_result_error_out_of_device_memory",
    "ur_result_error_out_of_host_memory",
    "ur_result_error_out_of_resources",
    "ur_result_error_device_lost",
    "ze_result_error_out_of_device_memory",
    "ze_result_error_out_of_host_memory",
    "ze_result_error_device_lost",
)


def is_fatal_accelerator_error(error: BaseException) -> bool:
    """Return true when retrying another backend would be unsafe.

    XPU allocation failures can surface as ``torch.OutOfMemoryError`` or as a
    generic ``RuntimeError`` carrying a Level Zero/PI/UR status.  Walk wrapped
    exceptions as well because some PyTorch and custom-op boundaries retain the
    accelerator error only as ``__cause__`` or ``__context__``.
    """

    oom_type = getattr(torch, "OutOfMemoryError", None)
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)

        if oom_type is not None and isinstance(current, oom_type):
            return True
        message = str(current).lower()
        if any(marker in message for marker in _FATAL_ACCELERATOR_ERROR_MARKERS):
            return True

        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return False
