"""Reclaim DynamicVRAM residency at Windows XPU model boundaries."""

from __future__ import annotations

import functools
import logging
import sys


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_dynamic_vram_boundary_original__"


def _load_argument(args, kwargs, name, position, default):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _expanded_models(models):
    pending = list(models)
    expanded = []
    seen = set()
    while pending:
        model = pending.pop(0)
        identity = id(model)
        if identity in seen:
            continue
        seen.add(identity)
        expanded.append(model)
        pending.extend(model.model_patches_models())
    return expanded


def _minimum_memory_target(model_management, args, kwargs):
    memory_required = _load_argument(args, kwargs, "memory_required", 0, 0)
    minimum_required = _load_argument(
        args, kwargs, "minimum_memory_required", 2, None
    )
    reserved = model_management.extra_reserved_memory()
    inference = model_management.minimum_inference_memory()
    if minimum_required is None:
        return max(inference, memory_required + reserved)
    return max(inference, minimum_required + reserved)


def _trim_pass(model_management, device, target, requested, include_requested):
    reclaimed = 0
    trimmed = 0
    for loaded in reversed(model_management.current_loaded_models):
        model = loaded.model
        if loaded.device != device or loaded.is_dead() or not model.is_dynamic():
            continue
        is_requested = id(model) in requested
        if is_requested != include_requested:
            continue

        shortfall = target - model_management.get_free_memory(device)
        if shortfall <= 0:
            break
        if model.loaded_size() <= 0:
            continue

        freed = int(model.partially_unload(model.offload_device, shortfall))
        if freed > 0:
            reclaimed += freed
            trimmed += 1
    return reclaimed, trimmed


def _trim_dynamic_boundary(model_management, models, target):
    if not models or not all(model.is_dynamic() for model in models):
        return

    requested = {id(model) for model in models}
    devices = {model.load_device for model in models}
    for device in devices:
        if getattr(device, "type", None) != "xpu":
            continue

        free_before = int(model_management.get_free_memory(device))
        if free_before >= target:
            continue

        inactive_bytes, inactive_models = _trim_pass(
            model_management, device, target, requested, False
        )
        active_bytes, active_models = _trim_pass(
            model_management, device, target, requested, True
        )
        free_after = int(model_management.get_free_memory(device))
        log.info(
            "[OmniXPU] DynamicVRAM boundary trim: device=%s target=%.1fMiB "
            "free=%.1f->%.1fMiB reclaimed=%.1fMiB models=%d inactive/%d active",
            device,
            target / (1024 ** 2),
            free_before / (1024 ** 2),
            free_after / (1024 ** 2),
            (inactive_bytes + active_bytes) / (1024 ** 2),
            inactive_models,
            active_models,
        )


def _patch_model_loader(model_management):
    original = model_management.load_models_gpu
    if hasattr(original, _PATCH_MARKER):
        return

    @functools.wraps(original)
    def boundary_trimmed(models, *args, **kwargs):
        expanded = _expanded_models(models)
        target = _minimum_memory_target(model_management, args, kwargs)
        _trim_dynamic_boundary(model_management, expanded, target)
        return original(models, *args, **kwargs)

    setattr(boundary_trimmed, _PATCH_MARKER, original)
    model_management.load_models_gpu = boundary_trimmed


def apply():
    if sys.platform != "win32":
        return False, "Windows only"

    import comfy.model_management

    required = (
        "current_loaded_models",
        "extra_reserved_memory",
        "get_free_memory",
        "load_models_gpu",
        "minimum_inference_memory",
    )
    missing = [name for name in required if not hasattr(comfy.model_management, name)]
    if missing:
        return False, "missing ComfyUI hooks: " + ", ".join(missing)

    _patch_model_loader(comfy.model_management)
    return True, "Windows XPU DynamicVRAM boundary reclaim enabled"


__all__ = ["apply"]
