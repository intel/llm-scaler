"""Adapt Qwen Image 2.1 prefix caching and its shared Wan cache on XPU."""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import os
import textwrap

import torch

log = logging.getLogger("ComfyUI-OmniXPU")
_MARKER = "__omnixpu_qwen_image21_cache_original__"
_WRAPPER_KEY = "omnixpu_qwen_image21_prefix_cache"
_PREFIX_PATCHES = ("post_input", "attn1_patch", "single_block")
# Preserve the legacy env name; unset or empty enables, and "0" disables.
_COPY_ENV = "OMNIXPU_EXPERIMENTAL_QWEN21_CACHE_COPY"
_COPY_MARKER = "__omnixpu_qwen21_cache_copy_original__"
_COPY_MIN_QUERY = 2048  # The existing BMG D128 prefix CUTE admission floor.
_MAX_ELEMENTS = (1 << 31) - 1


def _prefix_source_matches(function):
    """Accept the original attention math regardless of comments or formatting."""
    expected = ast.parse("""\
def prefix_cached_attention(prefix_k, prefix_v, transformer_options={}):
    def attn(q, k, v, heads):
        return optimized_attention(
            q.flatten(2),
            torch.cat([prefix_k, k], dim=1).flatten(2),
            torch.cat([prefix_v, v], dim=1).flatten(2),
            heads, transformer_options=transformer_options,
        )
    return attn
""")
    actual = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if len(actual.body) != 1 or not isinstance(actual.body[0], ast.FunctionDef):
        return False
    # The exported binding name is checked by the caller; the local name is
    # irrelevant to the attention math.
    actual.body[0].name = "prefix_cached_attention"
    return ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False)


def _copy_prefix_inputs(q, k, v, prefix_k, prefix_v, heads, options):
    """Admit source-bound cache hits on known BMG devices; otherwise use original."""
    if (os.environ.get(_COPY_ENV, "1") not in ("", "1") or not isinstance(options, dict)
            or os.environ.get("OMNI_ATTN_BACKEND", "auto").lower() not in ("auto", "cute")
            or options.get("optimized_attention_override")
            or torch.compiler.is_compiling() or heads != 32):
        return False
    try:
        patches = options.get("patches", {})
        replacements = options.get("patches_replace", {})
        if (not isinstance(patches, dict) or not isinstance(replacements, dict)
                or any(patches.get(name) for name in _PREFIX_PATCHES)
                or replacements.get("dit", {})):
            return False
        import omni_xpu_kernel
        if omni_xpu_kernel.__xpu_target__ != "bmg":
            return False
        tensors = (q, k, v, prefix_k, prefix_v)
        if (any(not isinstance(t, torch.Tensor) for t in tensors)
                or q.device.type != "xpu" or q.dtype != torch.bfloat16
                or any(t.device != q.device or t.dtype != q.dtype or t.requires_grad
                       or t.ndim != 4 or t.shape[0] != 1
                       or t.shape[2:] != (32, 128)
                       or t.stride()[1:] != (4096, 128, 1)
                       or t.data_ptr() % 64 for t in tensors)):
            return False
        current_len, prefix_len = k.shape[1], prefix_k.shape[1]
        if (current_len < _COPY_MIN_QUERY or prefix_len <= 0
                or q.shape[1] != current_len
                or v.shape != k.shape or prefix_v.shape != prefix_k.shape
                or (prefix_len + current_len) * 4096 > _MAX_ELEMENTS):
            return False
        index = q.device.index if q.device.index is not None else torch.xpu.current_device()
        device_id = getattr(torch.xpu.get_device_properties(index), "device_id", None)
        return (device_id is not None and
                omni_xpu_kernel.device.classify_bmg_device_id(int(device_id)) != "unknown")
    except (AttributeError, ImportError, TypeError, ValueError):
        return False


def _copy_prefix_factory(qwen):
    mode = os.environ.get(_COPY_ENV, "1")
    if mode not in ("", "0", "1"):
        return None, f"{_COPY_ENV} must be 0 or 1"
    if mode == "0":
        return None, None
    original = qwen.prefix_cached_attention
    if hasattr(original, _COPY_MARKER):
        return original, None
    if tuple(inspect.signature(original).parameters) != (
            "prefix_k", "prefix_v", "transformer_options"):
        return None, "unsupported prefix_cached_attention signature"
    if not _prefix_source_matches(original):
        return None, "unsupported prefix_cached_attention source contract"

    @functools.wraps(original)
    def factory(prefix_k, prefix_v, transformer_options={}):
        fallback = original(prefix_k, prefix_v, transformer_options)

        def attention(q, k, v, heads):
            if not _copy_prefix_inputs(q, k, v, prefix_k, prefix_v,
                                       heads, transformer_options):
                return fallback(q, k, v, heads)
            length = prefix_k.shape[1]
            shape = (1, length + k.shape[1], 32, 128)
            joined_k = torch.empty(shape, dtype=k.dtype, device=k.device)
            joined_v = torch.empty(shape, dtype=v.dtype, device=v.device)
            joined_k[:, :length].copy_(prefix_k)
            joined_k[:, length:].copy_(k)
            joined_v[:, :length].copy_(prefix_v)
            joined_v[:, length:].copy_(v)
            return qwen.optimized_attention(
                q.flatten(2), joined_k.flatten(2), joined_v.flatten(2), heads,
                transformer_options=transformer_options)

        return attention

    setattr(factory, _COPY_MARKER, original)
    return factory, None


def _rewrite(function, replacements, helpers=None):
    """Adapt exact upstream statements while retaining quantization/lifecycle code."""
    if function.__closure__:
        raise ValueError("cache method unexpectedly captures closure state")
    source = textwrap.dedent(inspect.getsource(function))
    if not source.startswith(f"def {function.__name__}("):
        raise ValueError("unsupported decorated cache method")
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"unsupported {function.__name__} cache contract")
        source = source.replace(before, after)
    namespace = dict(function.__globals__)
    namespace.update(helpers or {})
    exec(compile(source, inspect.getfile(function), "exec"), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


def _copy_pinned_cpu(tensor, device):
    try:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device=device, pin_memory=True)
    except torch.OutOfMemoryError:
        # Match upstream's pageable fallback when host registration cannot fit.
        log.warning("[OmniXPU] KV cache pinned allocation exhausted; using pageable RAM")
        return tensor.to(device, copy=True)
    # Cache fill completes before CPU key/slot reuse. Prefetch back to XPU uses
    # the upstream stream waits and staging buffers; Torch owns host lifetime.
    host.copy_(tensor, non_blocking=False)
    return host


def _guard_prefix_cache(executor, x, timesteps, context, ref_latents=None,
                        image_slots=None, transformer_options=None, **kwargs):
    options = {} if transformer_options is None else transformer_options
    patches = options.get("patches", {})
    unsafe = any(patches.get(name) for name in _PREFIX_PATCHES)
    unsafe = unsafe or bool(options.get("patches_replace", {}).get("dit", {}))
    model = executor.class_obj
    if x.device.type != "xpu" or not unsafe:
        return executor(x, timesteps, context, ref_latents, image_slots, options, **kwargs)

    enabled = model.prefix_cache_enabled
    model.reset_prefix_cache(False)
    try:
        return executor(x, timesteps, context, ref_latents, image_slots, options, **kwargs)
    finally:
        # Never restore a stale slot, including after an interrupted forward.
        model.reset_prefix_cache(enabled)


def apply():
    try:
        import comfy.ldm.qwen_image21.model as qwen
        import comfy.model_base as model_base
        import comfy.model_management as mm
        import comfy.model_patcher as model_patcher
        from comfy.patcher_extension import WrappersMP
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy", "comfy.ldm.qwen_image21", "comfy.ldm.qwen_image21.model"}:
            return False, "Qwen Image 2.1 is not available in this ComfyUI"
        raise

    if mm.get_torch_device().type != "xpu":
        return False, "Qwen Image 2.1 cache adapter requires XPU"

    try:
        cache = qwen.PoseBranchCache
        patcher = model_patcher.ModelPatcher
        originals = (cache.select, cache.put, patcher.__init__)
    except AttributeError as exc:
        return False, f"Qwen Image 2.1 cache API is unavailable: {exc}"
    if all(hasattr(method, _MARKER) for method in originals):
        try:
            copy_factory, copy_error = _copy_prefix_factory(qwen)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            return False, str(exc)
        if copy_error is not None:
            return False, copy_error
        if copy_factory is not None:
            qwen.prefix_cached_attention = copy_factory
        return True, "already patched"
    if any(hasattr(method, _MARKER) for method in originals):
        return False, "Qwen Image 2.1 cache adapter is partially applied"

    original_select, original_put, original_init = originals
    try:
        if tuple(inspect.signature(original_select).parameters) != ("self", "k", "create"):
            raise ValueError("unsupported PoseBranchCache.select signature")
        if inspect.signature(original_select).parameters["create"].default is not True:
            raise ValueError("unsupported PoseBranchCache.select default")
        if tuple(inspect.signature(original_put).parameters) != ("self", "i", "t"):
            raise ValueError("unsupported PoseBranchCache.put signature")
        forward = qwen.QwenImage21Transformer2DModel._forward
        if tuple(inspect.signature(forward).parameters) != (
            "self", "x", "timesteps", "context", "ref_latents", "image_slots",
            "transformer_options", "kwargs",
        ):
            raise ValueError("unsupported Qwen Image 2.1 forward signature")
        entry = inspect.getsource(qwen.QwenImage21Transformer2DModel.forward)
        body = inspect.getsource(forward)
        if ("WrappersMP.DIFFUSION_MODEL" not in entry
                or "self._forward" not in entry
                or "self.prefix_cache_enabled and prefix_len > 0 and not hooked" not in body
                or any(f'patches.get("{name}")' not in body for name in _PREFIX_PATCHES)):
            raise ValueError("unsupported Qwen Image 2.1 wrapper/cache contract")
        qwen_type = model_base.QwenImage21
        if not callable(getattr(patcher, "add_wrapper_with_key", None)):
            raise ValueError("ModelPatcher diffusion wrappers are unavailable")
        reset = inspect.getsource(qwen.QwenImage21Transformer2DModel.reset_prefix_cache)
        if "self.prefix_cache.free()" not in reset or "self.prefix_cache_enabled = enabled" not in reset:
            raise ValueError("unsupported Qwen Image 2.1 cache reset contract")
        select_xpu = _rewrite(original_select, (
            ("for s in self.slots:", "for index, s in enumerate(self.slots):"),
            ("self.slots.remove(s)", "self.slots.pop(index)"),
        ))
        put_xpu = _rewrite(original_put, (
            ("t = t.to(self.store_device, copy=True)",
             "t = _omnixpu_copy_pinned_cpu(t, self.store_device)"),
            ("if comfy.model_management.pin_memory(t):",
             "if not t.is_pinned() and comfy.model_management.pin_memory(t):"),
        ), {"_omnixpu_copy_pinned_cpu": _copy_pinned_cpu})
        copy_factory, copy_error = _copy_prefix_factory(qwen)
        if copy_error is not None:
            raise ValueError(copy_error)
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError) as exc:
        return False, str(exc)

    @functools.wraps(original_select)
    def select(self, k, create=True):
        if k.device.type != "xpu":
            return original_select(self, k, create=create)
        return select_xpu(self, k, create=create)

    @functools.wraps(original_put)
    def put(self, i, t):
        if (t.device.type != "xpu" or self.store_device.type != "cpu"
                or mm.args.disable_pinned_memory or mm.in_training or t.requires_grad):
            return original_put(self, i, t)
        return put_xpu(self, i, t)

    @functools.wraps(original_init)
    def init(self, model, load_device, offload_device, *args, **kwargs):
        original_init(self, model, load_device, offload_device, *args, **kwargs)
        if isinstance(model, qwen_type) and load_device.type == "xpu":
            self.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY,
                                      _guard_prefix_cache)

    for replacement, original in zip((select, put, init), originals):
        setattr(replacement, _MARKER, original)
    cache.select, cache.put, patcher.__init__ = select, put, init
    if copy_factory is not None:
        qwen.prefix_cached_attention = copy_factory
    return True, None
