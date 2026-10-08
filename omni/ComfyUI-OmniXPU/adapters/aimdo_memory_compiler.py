"""Private XPU graph backend and narrow wrappers for an unmodified ComfyUI.

The unsupported XPU lifecycle is owned here; CUDA always uses ComfyUI's original
functions. No public AIMDO record/capability API is replaced.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext
import functools
import inspect
import os
import sys
import threading
import weakref

from ..compiler_compat import preflight, SIGNATURES, validate_live_signature

_MARKER = "__omnixpu_aimdo_compiler_original__"
_RUNTIME = None


class _Graph:
    def __init__(self, runtime, native, device):
        self.runtime, self.native, self.device = runtime, native, device
        self._comfy_active = False
        self._comfy_xpu_diagnostic = True
        self._comfy_cuda_graph_modules = weakref.WeakSet()
        self.scope = None
        self.recording = True

    def __getattr__(self, name):
        return getattr(self.native, name)

    def iterate(self, name=None):
        result = self.native.iterate(name)
        window = getattr(self.runtime.local, "prefetch", None)
        if window is not None and window.graph is self:
            window.pause()
        return result


class _Pause:
    """Kitchen may share one context object between threads and nested calls."""
    def __init__(self, runtime, sync=False):
        self.runtime, self.sync = runtime, sync
        self.local = threading.local()

    def __enter__(self):
        graph = self.runtime.graph()
        if isinstance(graph, _Graph) and graph._comfy_active:
            context = self.runtime.owner.paused_graph_scope(graph, sync=self.sync)
        else:
            context = self.runtime.original_pause(self.sync)
        context.__enter__()
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = self.local.stack = []
        stack.append(context)
        return self

    def __exit__(self, *exc):
        return self.local.stack.pop().__exit__(*exc)


class _Prefetch:
    def __init__(self, runtime, graph):
        self.runtime, self.graph, self.context = runtime, graph, None

    def pause(self):
        if self.context is None:
            context = self.runtime.owner.paused_graph_scope(self.graph)
            context.__enter__()
            self.context = context

    def resume(self, exc=(None, None, None)):
        if self.context is not None:
            context, self.context = self.context, None
            context.__exit__(*exc)


class _GcProxy:
    """Only the prompt-worker module's gc reference is wrapped."""
    def __init__(self, runtime, original):
        self.runtime, self.original = runtime, original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def collect(self, *args, **kwargs):
        result = self.original.collect(*args, **kwargs)
        if getattr(self.runtime.local, "free_memory", False):
            self.runtime.local.after_gc = True
        return result


class Runtime:
    def __init__(self, management, prefetch, owner, torch, kitchen, int8, worker, queue_type):
        self.mm, self.mp, self.owner = management, prefetch, owner
        self.torch, self.ck, self.int8 = torch, kitchen, int8
        self.worker, self.queue_type = worker, queue_type
        self.local = threading.local()
        self.original_pause = prefetch.pause_malloc_graph
        self.originals = {}
        self.signatures = {}

    def bind(self, original, args, kwargs):
        signature = self.signatures.get(original)
        if signature is None:
            signature = self.signatures[original] = inspect.signature(original)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    def graph(self):
        return self.mp.MALLOC_GRAPHS.get(threading.get_ident())

    def enabled(self, original, device):
        if not self.mm.is_device_xpu(device):
            return original(device)
        if self.mp.args.disable_comfy_compiler or not self.mp.comfy.memory_management.aimdo_enabled:
            return False
        from comfy_aimdo import control
        capability = control.get_memory_compiler_capability()["native_owner_diagnostic"]
        return capability["active"] and capability["consumer_contract"] == "explicit_record_stream"

    def begin(self, device):
        if not self.mp.malloc_graph_enabled(device):
            return
        stream = self.mm.current_stream(device)
        graph = self.graph()
        if graph is None:
            graph = _Graph(self, self.owner.record_diagnostic(stream, self.mp.args.assert_graph_breaks), device)
            self.mp.MALLOC_GRAPHS[threading.get_ident()] = graph
        elif not isinstance(graph, _Graph) or graph.device != device or graph._comfy_active:
            raise RuntimeError("XPU compiler graph device/lifecycle mismatch")
        else:
            graph.native.push()
            graph.recording = True
        try:
            graph.scope = self.owner.compiler_scope(stream)
            graph.scope.__enter__()
        except BaseException:
            # A failed enter does not create an entered Python context.
            graph.scope = None
            self.cleanup()
            raise
        try:
            self.int8.set_allocation_context_factory(lambda: _Pause(self))
            self.ck.set_allocation_context(_Pause(self))
        except BaseException:
            self.cleanup()
            raise
        graph._comfy_active = True
        self.mp.MALLOC_GRAPH_USED = True

    def _exit_scope(self, graph):
        if graph.scope is not None:
            graph.scope.__exit__(None, None, None)
            graph.scope = None
        graph._comfy_active = False

    def end(self):
        graph = self.graph()
        if graph._comfy_active:
            try:
                broken = graph.native.pop()
            except BaseException:
                self.cleanup()
                raise
            graph.recording = False
            if broken:
                self.mp._malloc_graph_break()
            self._exit_scope(graph)

    def cleanup(self):
        graph = self.graph()
        if graph._comfy_cuda_graph_modules:
            raise RuntimeError("CUDA capture modules attached to an XPU-only compiler graph")
        if graph.recording:
            graph.native.abort()
            graph.recording = False
        self._exit_scope(graph)
        rogues = graph.native.rogue_count
        if not graph.native.close():
            raise RuntimeError("XPU compiler graph close was deferred")
        # Retain caller ownership until checked close succeeds. Count once.
        self.mp.MALLOC_GRAPHS.pop(threading.get_ident())
        self.mp.MALLOC_GRAPH_ROGUES += rogues

    @contextmanager
    def cast_context(self, stream, values):
        if stream is None or not self.owner.installed():
            yield
            return
        with ExitStack() as stack:
            if getattr(getattr(stream, "device", None), "type", None) == "xpu":
                stack.enter_context(_Pause(self))
            registered = set()
            for value in values:
                for tensor in self.storage_tensors(value):
                    if not self.mm.is_device_xpu(tensor.device):
                        continue
                    pointer = tensor.untyped_storage().data_ptr()
                    if pointer and pointer not in registered and self.owner.is_compiler_owner(pointer):
                        stack.enter_context(self.owner.consumer_scope(tensor, stream))
                        registered.add(pointer)
            yield

    def storage_tensors(self, value):
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from self.storage_tensors(item)
        elif isinstance(value, self.mm.comfy.quant_ops.QuantizedTensor):
            names, _ = value.__tensor_flatten__()
            for name in names:
                yield from self.storage_tensors(getattr(value, name))
        elif isinstance(value, self.torch.Tensor):
            yield value

    def prefetch(self, original, *args, **kwargs):
        bound = self.bind(original, args, kwargs)
        values = bound.arguments
        graph = self.graph()
        if (not isinstance(graph, _Graph) or not graph._comfy_active
                or values["queue"] is None):
            return original(*args, **kwargs)
        if graph.device != values["device"]:
            raise RuntimeError("prefetch device differs from compiler graph")
        previous = getattr(self.local, "prefetch", None)
        window = self.local.prefetch = _Prefetch(self, graph)
        core = values["core"]
        if core is not None:
            @functools.wraps(core)
            def active_core(*a, **kw):
                window.resume()
                return core(*a, **kw)
            values["core"] = active_core
        try:
            # No iterate is executed on this path. Admission verifies the
            # preceding XPU preamble has no tensor allocations.
            if values["malloc_scope"] is None:
                window.pause()
            return original(*bound.args, **bound.kwargs)
        finally:
            try:
                window.resume(sys.exc_info())
            finally:
                self.local.prefetch = previous

    def clear_caches(self):
        devices = [d for d in self.mm.get_all_torch_devices() if self.mm.is_device_xpu(d)]
        if not devices:
            return 0
        removed = sum(self.int8.clear_convrot_hadamard_cache(d.index)
                      + self.ck.clear_nvfp4_lut_cache(d.index) for d in devices)
        return removed + self.int8.release_onednn_int8_cache()

    def flags(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        bound = self.bind(original, args, kwargs)
        if bound.arguments["reset"]:
            self.local.free_memory = bool(result.get("free_memory", False))
            self.local.after_gc = False
        return result

    def empty_cache(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if getattr(self.local, "after_gc", False) and not getattr(self.local, "flushing", False):
            self.local.flushing = True
            try:
                if self.clear_caches():
                    original(*args, **kwargs)
                self.local.after_gc = False
                self.local.free_memory = False
            finally:
                self.local.flushing = False
        return result

    def install(self):
        if self.originals:
            return
        def wrap(module, name, dispatch):
            original = getattr(module, name)
            @functools.wraps(original)
            def replacement(*args, **kwargs):
                return dispatch(original, *args, **kwargs)
            setattr(replacement, _MARKER, original)
            replacement.__omnixpu_aimdo_compiler_runtime__ = self
            self.originals[(module, name)] = original
            setattr(module, name, replacement)

        wrap(self.mp, "malloc_graph_enabled", self.enabled)
        wrap(self.mp, "malloc_graph_begin", lambda original, device:
             self.begin(device) if self.mm.is_device_xpu(device) else original(device))
        for name, action in (("malloc_graph_end", self.end), ("cleanup_malloc_graph", self.cleanup)):
            wrap(self.mp, name, lambda original, action=action:
                 action() if isinstance(self.graph(), _Graph) else original())
        wrap(self.mp, "pause_malloc_graph", lambda original, sync=False: _Pause(self, sync))
        wrap(self.mp, "prefetch_queue_pop", self.prefetch)
        for name, stream_name, tensor_names in (
                ("get_cast_buffer", "offload_stream", ()),
                ("cast_to_gathered", "stream", ("tensors", "r", "r2")),
                ("cast_to", "stream", ("weight", "r"))):
            def cast(original, *args, stream_name=stream_name, tensor_names=tensor_names,
                     operation=name, **kwargs):
                bound = self.bind(original, args, kwargs)
                values = bound.arguments
                if operation == "cast_to":
                    weight = values["weight"]
                    same_device = values["device"] is None or weight.device == values["device"]
                    same_dtype = values["dtype"] is None or weight.dtype == values["dtype"]
                    if same_device and same_dtype and not values["copy"]:
                        return original(*args, **kwargs)
                elif operation == "get_cast_buffer":
                    buffers = getattr(self.mm, "STREAM_CAST_BUFFERS", None)
                    buffer = buffers.get(values[stream_name]) if buffers is not None else None
                    if buffer is not None and buffer.numel() >= values["size"]:
                        return original(*args, **kwargs)
                with self.cast_context(values[stream_name], [values[n] for n in tensor_names]):
                    return original(*args, **kwargs)
            wrap(self.mm, name, cast)
        wrap(self.queue_type, "get_flags", self.flags)
        wrap(self.mm, "soft_empty_cache", self.empty_cache)
        self.worker.gc = _GcProxy(self, self.worker.gc)


def apply():
    global _RUNTIME
    if os.environ.get("AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC", "0") != "1":
        return False, "private native-owner diagnostic disabled"
    if _RUNTIME is not None:
        return True, "already installed"
    from comfy_aimdo import native_owner
    if not native_owner.installed():
        raise SystemExit("[OmniXPU] compiler requested without the native-owner sidecar")
    try:
        root = preflight()
        import comfy.model_management as mm
        import comfy.model_prefetch as mp
        import comfy_kitchen as ck
        import execution
        from omni_xpu_kernel import int8
        import torch
        existing = getattr(mm.cast_to, "__omnixpu_aimdo_compiler_runtime__", None)
        if existing is not None:
            _RUNTIME = existing
            return True, "already installed"
        worker = next(m for name in ("__main__", "main")
                      if (m := sys.modules.get(name)) is not None
                      and getattr(m, "__file__", None)
                      and os.path.realpath(m.__file__) == os.path.join(root, "main.py"))
        for module, path in ((mm, "comfy/model_management.py"), (mp, "comfy/model_prefetch.py")):
            for name, names in SIGNATURES[path].items():
                function = getattr(module, name)
                validate_live_signature(function, names)
        for obj, name in ((ck, "set_allocation_context"), (ck, "clear_nvfp4_lut_cache"),
                          (int8, "set_allocation_context_factory"),
                          (int8, "clear_convrot_hadamard_cache"), (int8, "release_onednn_int8_cache")):
            if not callable(getattr(obj, name, None)):
                raise RuntimeError(f"persistent-cache interface missing: {name}")
        runtime = Runtime(mm, mp, native_owner, torch, ck, int8, worker, execution.PromptQueue)
        runtime.install()
        _RUNTIME = runtime
    except Exception as exc:
        raise SystemExit(f"[OmniXPU] caller changed after native takeover; restart with compiler disabled: {exc}") from exc
    return True, "private XPU memory compiler; ComfyUI source unchanged"
