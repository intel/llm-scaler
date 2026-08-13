"""Cache whole-LoRA weight sizes on patchers and include them in load budgets."""

from __future__ import annotations

import functools
import logging
import os
from numbers import Real


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_lora_memory_original__"
_ATTACHMENT_KEY = "omnixpu_lora_memory_budget_v1"
_ATTACHMENT_VERSION = 1
_ALIGNMENT = 1024
_runtime_peak = 0


def _tensor_size(value):
    try:
        size = int(value.numel()) * int(value.element_size())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return size, (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _collect_entries(value, tensor_entries=None, seen_values=None):
    """Collect unique tensor identity -> (payload, aligned staging) entries."""
    if tensor_entries is None:
        tensor_entries = {}
    if seen_values is None:
        seen_values = set()

    tensor_size = _tensor_size(value)
    if tensor_size is not None:
        tensor_entries.setdefault(id(value), tensor_size)
        return tensor_entries

    identity = id(value)
    if identity in seen_values:
        return tensor_entries

    if isinstance(value, dict):
        seen_values.add(identity)
        values = value.values()
    elif isinstance(value, (list, tuple, set)):
        seen_values.add(identity)
        values = value
    elif hasattr(value, "weights"):
        seen_values.add(identity)
        values = (value.weights,)
    else:
        return tensor_entries

    for item in values:
        _collect_entries(item, tensor_entries, seen_values)
    return tensor_entries


def _entries_stats(entries):
    return {
        "payload": sum(value[0] for value in entries.values()),
        "staging": sum(value[1] for value in entries.values()),
        "tensors": len(entries),
    }


def _patcher_entries(patcher):
    patches = getattr(patcher, "patches", None)
    if not patches:
        return {}

    tensor_entries = {}
    seen_values = set()
    for patch_entries in patches.values():
        for patch_entry in patch_entries:
            try:
                value = patch_entry[1]
            except (IndexError, TypeError):
                continue
            _collect_entries(value, tensor_entries, seen_values)
    return tensor_entries


def _patcher_stats(patcher):
    entries = _patcher_entries(patcher)
    stats = _entries_stats(entries)
    stats["peak"] = 0

    patches = getattr(patcher, "patches", None)
    if not patches:
        return stats

    for patch_entries in patches.values():
        layer_entries = {}
        layer_seen = set()
        for patch_entry in patch_entries:
            try:
                value = patch_entry[1]
            except (IndexError, TypeError):
                continue
            _collect_entries(value, layer_entries, layer_seen)
        stats["peak"] = max(
            stats["peak"], _entries_stats(layer_entries)["staging"]
        )
    return stats


def _get_attachment(patcher):
    if patcher is None:
        return None
    getter = getattr(patcher, "get_attachment", None)
    if callable(getter):
        return getter(_ATTACHMENT_KEY)
    attachments = getattr(patcher, "attachments", None)
    if isinstance(attachments, dict):
        return attachments.get(_ATTACHMENT_KEY)
    return None


def _get_budget_entries(patcher):
    attachment = _get_attachment(patcher)
    if not isinstance(attachment, dict):
        return {}
    if attachment.get("version") != _ATTACHMENT_VERSION:
        return {}

    entries = {}
    for entry in attachment.get("entries", ()):
        try:
            identity, payload, staging = entry
            entries[int(identity)] = (int(payload), int(staging))
        except (TypeError, ValueError):
            continue
    return entries


def _set_budget_entries(patcher, entries):
    if patcher is None or not entries:
        return

    stats = _entries_stats(entries)
    attachment = {
        "version": _ATTACHMENT_VERSION,
        "entries": tuple(
            (identity, size[0], size[1])
            for identity, size in sorted(entries.items())
        ),
        "payload": stats["payload"],
        "staging": stats["staging"],
        "tensor_count": stats["tensors"],
    }
    setter = getattr(patcher, "set_attachments", None)
    if callable(setter):
        setter(_ATTACHMENT_KEY, attachment)
        return
    attachments = getattr(patcher, "attachments", None)
    if isinstance(attachments, dict):
        attachments[_ATTACHMENT_KEY] = attachment


def _cache_patcher_budget(before_patcher, after_patcher, before_entries):
    if after_patcher is None:
        empty = _entries_stats({})
        return empty, empty

    after_entries = _patcher_entries(after_patcher)
    added_entries = {
        identity: size
        for identity, size in after_entries.items()
        if identity not in before_entries
    }
    budget_entries = _get_budget_entries(after_patcher)
    if not budget_entries:
        budget_entries = _get_budget_entries(before_patcher)
    budget_entries.update(added_entries)
    _set_budget_entries(after_patcher, budget_entries)
    return _entries_stats(added_entries), _entries_stats(budget_entries)


def _iter_model_patchers(models):
    pending = list(models)
    seen = set()
    while pending:
        patcher = pending.pop(0)
        identity = id(patcher)
        if identity in seen:
            continue
        seen.add(identity)
        yield patcher

        additional = getattr(patcher, "model_patches_models", None)
        if callable(additional):
            additional_models = additional()
            if additional_models:
                pending.extend(additional_models)


def _models_budget_entries(models):
    entries = {}
    for patcher in _iter_model_patchers(models):
        for identity, size in _get_budget_entries(patcher).items():
            current = entries.get(identity)
            if current is None or size[1] > current[1]:
                entries[identity] = size
    return entries


def _is_memory_size(value):
    return isinstance(value, Real) and not isinstance(value, bool)


def _add_budget_to_load_args(args, kwargs, budget):
    new_args = list(args)
    new_kwargs = dict(kwargs)

    if "memory_required" in new_kwargs:
        requested = new_kwargs["memory_required"]
        new_kwargs["memory_required"] = requested + budget
    elif new_args:
        requested = new_args[0]
        new_args[0] = requested + budget
    else:
        requested = 0
        new_kwargs["memory_required"] = budget
    effective_requested = requested + budget

    minimum = None
    effective_minimum = None
    if "minimum_memory_required" in new_kwargs:
        minimum = new_kwargs["minimum_memory_required"]
        if _is_memory_size(minimum):
            effective_minimum = minimum + budget
            new_kwargs["minimum_memory_required"] = effective_minimum
    elif len(new_args) > 2:
        minimum = new_args[2]
        if _is_memory_size(minimum):
            effective_minimum = minimum + budget
            new_args[2] = effective_minimum

    return (
        tuple(new_args),
        new_kwargs,
        requested,
        effective_requested,
        minimum,
        effective_minimum,
    )


def _minimum_budget_text(minimum, effective_minimum):
    if minimum is None:
        return "default"
    if effective_minimum is None:
        return f"unchanged({type(minimum).__name__})"
    return f"{_mib(minimum):.1f}->{_mib(effective_minimum):.1f}MiB"


def _runtime_patch_bytes(modules):
    total = 0
    for module in modules:
        for param_key in ("weight", "bias"):
            patch = getattr(module, param_key + "_lowvram_function", None)
            if patch is not None:
                total += int(patch.memory_required())
    return total


def _xpu_snapshot(device=None):
    try:
        import torch

        if device is None:
            device = torch.xpu.current_device()
        stats = torch.xpu.memory_stats(device)
        free, total = torch.xpu.mem_get_info(device)
        return {
            "allocated": int(stats.get("allocated_bytes.all.current", 0)),
            "reserved": int(stats.get("reserved_bytes.all.current", 0)),
            "free": int(free),
            "total": int(total),
        }
    except (AttributeError, RuntimeError, TypeError):
        return None


def _mib(value):
    return value / (1024 * 1024)


def _snapshot_text(snapshot):
    if snapshot is None:
        return "xpu=unavailable"
    return (
        f"xpu_allocated={_mib(snapshot['allocated']):.1f}MiB "
        f"xpu_reserved={_mib(snapshot['reserved']):.1f}MiB "
        f"xpu_free={_mib(snapshot['free']):.1f}MiB"
    )


def _patch_lora_loader(comfy_sd):
    original = comfy_sd.load_lora_for_models
    if hasattr(original, _PATCH_MARKER):
        return

    @functools.wraps(original)
    def budgeted(model, clip, lora, strength_model, strength_clip, *args, **kwargs):
        source_stats = _entries_stats(_collect_entries(lora))
        model_before = _patcher_entries(model)
        clip_before_patcher = getattr(clip, "patcher", clip)
        clip_before = _patcher_entries(clip_before_patcher)

        result = original(
            model, clip, lora, strength_model, strength_clip, *args, **kwargs
        )

        model_added, model_budget = _cache_patcher_budget(
            model, result[0], model_before
        )
        result_clip_patcher = getattr(result[1], "patcher", result[1])
        clip_added, clip_budget = _cache_patcher_budget(
            clip_before_patcher, result_clip_patcher, clip_before
        )
        log.info(
            "[OmniXPU] LoRA memory budget cached: source=%.1fMiB/%d tensors "
            "model_added=%.1fMiB model_budget=%.1fMiB "
            "clip_added=%.1fMiB clip_budget=%.1fMiB",
            _mib(source_stats["staging"]),
            source_stats["tensors"],
            _mib(model_added["staging"]),
            _mib(model_budget["staging"]),
            _mib(clip_added["staging"]),
            _mib(clip_budget["staging"]),
        )
        return result

    setattr(budgeted, _PATCH_MARKER, original)
    comfy_sd.load_lora_for_models = budgeted


def _patch_model_loader(model_management):
    original = model_management.load_models_gpu
    if hasattr(original, _PATCH_MARKER):
        return

    @functools.wraps(original)
    def budgeted(models, *args, **kwargs):
        budget_stats = _entries_stats(_models_budget_entries(models))
        budget = budget_stats["staging"]
        if budget == 0:
            return original(models, *args, **kwargs)

        (
            load_args,
            load_kwargs,
            requested,
            effective_requested,
            minimum,
            effective_minimum,
        ) = _add_budget_to_load_args(args, kwargs, budget)
        before = _xpu_snapshot()
        log.info(
            "[OmniXPU] LoRA model budget applied: cached=%.1fMiB/%d tensors "
            "requested=%.1f->%.1fMiB minimum=%s %s",
            _mib(budget),
            budget_stats["tensors"],
            _mib(requested),
            _mib(effective_requested),
            _minimum_budget_text(minimum, effective_minimum),
            _snapshot_text(before),
        )
        try:
            result = original(models, *load_args, **load_kwargs)
        except Exception:
            log.exception(
                "[OmniXPU] LoRA model load failed: cached_budget=%.1fMiB %s",
                _mib(budget),
                _snapshot_text(_xpu_snapshot()),
            )
            raise
        log.info(
            "[OmniXPU] LoRA model load complete: %s",
            _snapshot_text(_xpu_snapshot()),
        )
        return result

    setattr(budgeted, _PATCH_MARKER, original)
    model_management.load_models_gpu = budgeted


def _patch_dynamic_cast(comfy_ops):
    original = comfy_ops.cast_modules_with_vbar
    if hasattr(original, _PATCH_MARKER):
        return

    trace = os.environ.get("OMNIXPU_LORA_MEMORY_TRACE", "0") != "0"

    @functools.wraps(original)
    def monitored(modules, *args, **kwargs):
        global _runtime_peak

        patch_bytes = _runtime_patch_bytes(modules)
        if patch_bytes == 0:
            return original(modules, *args, **kwargs)

        device = kwargs.get("device", args[1] if len(args) > 1 else None)
        new_peak = patch_bytes > _runtime_peak
        if new_peak:
            _runtime_peak = patch_bytes
        before = _xpu_snapshot(device) if trace or new_peak else None
        if trace or new_peak:
            log.info(
                "[OmniXPU] LoRA XPU staging%s: patch=%.1fMiB modules=%d %s",
                " new_peak" if new_peak else "",
                _mib(patch_bytes),
                len(modules),
                _snapshot_text(before),
            )
        try:
            result = original(modules, *args, **kwargs)
        except Exception:
            log.exception(
                "[OmniXPU] LoRA XPU staging failed: patch=%.1fMiB %s",
                _mib(patch_bytes),
                _snapshot_text(_xpu_snapshot(device)),
            )
            raise
        if trace:
            log.info(
                "[OmniXPU] LoRA XPU staging complete: patch=%.1fMiB %s",
                _mib(patch_bytes),
                _snapshot_text(_xpu_snapshot(device)),
            )
        return result

    setattr(monitored, _PATCH_MARKER, original)
    comfy_ops.cast_modules_with_vbar = monitored


def apply():
    import comfy.model_management
    import comfy.ops
    import comfy.sd

    required = (
        (comfy.sd, "load_lora_for_models"),
        (comfy.model_management, "load_models_gpu"),
    )
    missing = [name for module, name in required if not hasattr(module, name)]
    if missing:
        return False, "missing ComfyUI hooks: " + ", ".join(missing)

    _patch_lora_loader(comfy.sd)
    _patch_model_loader(comfy.model_management)
    runtime_trace = os.environ.get("OMNIXPU_LORA_MEMORY_TRACE", "0") != "0"
    if runtime_trace:
        if hasattr(comfy.ops, "cast_modules_with_vbar"):
            _patch_dynamic_cast(comfy.ops)
        else:
            log.warning(
                "[OmniXPU] LoRA runtime trace unavailable: "
                "missing cast_modules_with_vbar"
            )
    return True, (
        "whole-LoRA model budgets cached"
        + ("; XPU staging trace enabled" if runtime_trace else "")
    )


__all__ = ["apply"]
