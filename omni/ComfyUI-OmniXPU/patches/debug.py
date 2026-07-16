"""Opt-in kernel and dispatch tracing for ComfyUI-OmniXPU patches."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeVar, cast

import torch

_ENV_NAME = "OMNIXPU_DEBUG"
_VERBOSE_ENV_NAME = "OMNIXPU_DEBUG_VERBOSE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_WRAPPED_ATTR = "__omnixpu_debug_patch__"

log = logging.getLogger("ComfyUI-OmniXPU")

_F = TypeVar("_F", bound=Callable[..., Any])


def debug_enabled() -> bool:
    """Return whether kernel tracing was requested for this process."""
    return (
        os.environ.get(_ENV_NAME, "").strip().lower() in _TRUE_VALUES
        or verbose_debug_enabled()
    )


def verbose_debug_enabled() -> bool:
    """Return whether dispatch-level tracing was requested for this process."""
    return os.environ.get(_VERBOSE_ENV_NAME, "").strip().lower() in _TRUE_VALUES


def _tensor_descriptions(value: Any, name: str) -> tuple[list[str], bool]:
    if isinstance(value, torch.Tensor):
        description = f"{name}(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
        return [description], value.device.type == "xpu"
    if isinstance(value, Mapping):
        descriptions = []
        has_xpu = False
        for key, item in value.items():
            nested, nested_has_xpu = _tensor_descriptions(item, f"{name}[{key!r}]")
            descriptions.extend(nested)
            has_xpu = has_xpu or nested_has_xpu
        return descriptions, has_xpu
    if isinstance(value, (tuple, list)):
        descriptions = []
        has_xpu = False
        for index, item in enumerate(value):
            nested, nested_has_xpu = _tensor_descriptions(item, f"{name}[{index}]")
            descriptions.extend(nested)
            has_xpu = has_xpu or nested_has_xpu
        return descriptions, has_xpu
    return [], False


def _format_tensor_inputs(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    param_names: tuple[str, ...],
) -> tuple[str, bool]:
    descriptions = []
    has_xpu = False
    for index, value in enumerate(args):
        name = param_names[index] if index < len(param_names) else f"arg{index}"
        nested, nested_has_xpu = _tensor_descriptions(value, name)
        descriptions.extend(nested)
        has_xpu = has_xpu or nested_has_xpu
    for name, value in kwargs.items():
        nested, nested_has_xpu = _tensor_descriptions(value, name)
        descriptions.extend(nested)
        has_xpu = has_xpu or nested_has_xpu
    return (", ".join(descriptions) if descriptions else "none"), has_xpu


def log_debug_event(
    stage: str,
    op_name: str,
    tensors: Mapping[str, Any],
    *,
    details: Mapping[str, Any] | None = None,
    xpu_only: bool = True,
    verbose_only: bool = False,
) -> None:
    """Log one debug event without evaluating or synchronizing tensors."""
    if verbose_only:
        if not verbose_debug_enabled():
            return
    elif not debug_enabled():
        return

    tensor_text, has_xpu = _format_tensor_inputs(
        tuple(tensors.values()), {}, tuple(tensors.keys())
    )
    if tensor_text == "none" or (xpu_only and not has_xpu):
        return

    detail_text = ""
    if details:
        detail_text = " " + " ".join(
            f"{name}={value}" for name, value in details.items() if value is not None
        )
    log.info(
        "[OmniXPU DEBUG] stage=%s op=%s%s tensors=%s",
        stage,
        op_name,
        detail_text,
        tensor_text,
    )


def trace_patch(
    op_name: str,
    param_names: tuple[str, ...] = (),
    *,
    xpu_only: bool = True,
    stage: str = "kernel",
    details: Mapping[str, Any] | None = None,
    verbose_only: bool = False,
) -> Callable[[_F], _F]:
    """Trace a function call as a kernel or verbose dispatch event."""

    def decorator(func: _F) -> _F:
        enabled = verbose_debug_enabled() if verbose_only else debug_enabled()
        marker = (stage, op_name)
        if not enabled or getattr(func, _WRAPPED_ATTR, None) == marker:
            return func

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            tensors = {
                param_names[index] if index < len(param_names) else f"arg{index}": value
                for index, value in enumerate(args)
            }
            tensors.update(kwargs)
            log_debug_event(
                stage,
                op_name,
                tensors,
                details=details,
                xpu_only=xpu_only,
                verbose_only=verbose_only,
            )
            return func(*args, **kwargs)

        setattr(wrapped, _WRAPPED_ATTR, marker)
        return cast(_F, wrapped)

    return decorator


__all__ = [
    "debug_enabled",
    "log_debug_event",
    "trace_patch",
    "verbose_debug_enabled",
]
