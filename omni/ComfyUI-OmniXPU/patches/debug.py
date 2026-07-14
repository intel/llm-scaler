"""Opt-in tensor metadata tracing for ComfyUI-OmniXPU patches."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeVar, cast

import torch

_ENV_NAME = "OMNIXPU_DEBUG"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_WRAPPED_ATTR = "__omnixpu_debug_patch__"

log = logging.getLogger("ComfyUI-OmniXPU")

_F = TypeVar("_F", bound=Callable[..., Any])


def debug_enabled() -> bool:
    """Return whether patch tracing was requested for this process."""
    return os.environ.get(_ENV_NAME, "").strip().lower() in _TRUE_VALUES


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


def trace_patch(
    patch_name: str,
    param_names: tuple[str, ...] = (),
    *,
    xpu_only: bool = True,
) -> Callable[[_F], _F]:
    """Wrap a patch call when tracing was enabled before module import."""

    def decorator(func: _F) -> _F:
        if not debug_enabled() or getattr(func, _WRAPPED_ATTR, None) == patch_name:
            return func

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            tensors, has_xpu = _format_tensor_inputs(args, kwargs, param_names)
            if tensors != "none" and (has_xpu or not xpu_only):
                log.info("[OmniXPU DEBUG] patch=%s tensors=%s", patch_name, tensors)
            return func(*args, **kwargs)

        setattr(wrapped, _WRAPPED_ATTR, patch_name)
        return cast(_F, wrapped)

    return decorator


__all__ = ["debug_enabled", "trace_patch"]
