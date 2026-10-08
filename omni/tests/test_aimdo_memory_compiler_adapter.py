"""Graph ownership and original-ComfyUI caller integration, without a device.

COMFYUI_CALLER_TEST_ROOT selects a pristine upstream source checkout for the
additional source-integration tests. Ordinary lifecycle tests are self-contained.
"""
import ast
from contextlib import contextmanager, nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import pytest

PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
SOURCE = os.environ.get("COMFYUI_CALLER_TEST_ROOT")


def load_adapter(monkeypatch):
    package = ModuleType("compiler_adapter_test")
    package.__path__ = [str(PLUGIN)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    adapters = ModuleType("compiler_adapter_test.adapters")
    adapters.__path__ = [str(PLUGIN / "adapters")]
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
    for name, path in (("compiler_compat", PLUGIN / "compiler_compat.py"),
                       ("adapters.aimdo_memory_compiler", PLUGIN / "adapters/aimdo_memory_compiler.py")):
        full = "compiler_adapter_test." + name
        spec = importlib.util.spec_from_file_location(full, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
    return module


def source_function(path, name, namespace, class_name=None):
    tree = ast.parse((Path(SOURCE) / path).read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def runtime(monkeypatch):
    adapter = load_adapter(monkeypatch)
    events = []
    owner = NS(active=False, paused=False, installed=lambda: True, is_compiler_owner=lambda p: p == 99)
    device = NS(type="xpu", index=0)
    stream = NS(device=device, sycl_queue=123)
    @contextmanager
    def scope(stream):
        events.append("scope-enter")
        if owner.active:
            raise RuntimeError("nested")
        owner.active = True
        try:
            yield
        finally:
            owner.active = False
            events.append("scope-exit")
    @contextmanager
    def pause(graph, sync=False):
        assert owner.active
        events.append("pause")
        previous, owner.paused = owner.paused, True
        try:
            yield
        finally:
            owner.paused = previous
            events.append("resume")
    @contextmanager
    def consumer(tensor, stream):
        events.append("consumer")
        yield
    owner.compiler_scope, owner.paused_graph_scope, owner.consumer_scope = scope, pause, consumer

    class Native:
        rogue_count = 3
        def __init__(self):
            self.fail_close = self.fail_pop = self.fail_abort = False
        def push(self):
            events.append("push")
        def pop(self):
            events.append("pop")
            if self.fail_pop:
                raise ValueError("pop-error")
            return False
        def iterate(self, name=None):
            assert owner.active and not owner.paused
            events.append("iterate")
            return False
        def abort(self):
            events.append("abort")
            if self.fail_abort:
                raise ValueError("abort-error")
        def close(self):
            assert not owner.active
            events.append("close")
            if self.fail_close:
                raise ValueError("close-error")
            return True
    def record(*args):
        events.append("record")
        return Native()
    owner.record_diagnostic = record
    class Tensor:
        def __init__(self, pointer=99):
            self.device, self.pointer = device, pointer
            self.dtype = "uint8"
        def untyped_storage(self):
            return NS(data_ptr=lambda: self.pointer)
    class QuantizedTensor:
        def __init__(self):
            self.weight, self.scale = Tensor(), Tensor()
        def __tensor_flatten__(self):
            return ("weight", "scale"), None
    def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
        events.append("copy")
        return weight
    def gathered(tensors, r, non_blocking=False, stream=None, r2=None):
        events.append("gather")
    def buffer(offload_stream, device, size, ref):
        events.append("buffer")
        assert owner.paused or not owner.active
        return Tensor(100)
    def empty(force=False):
        events.append("empty")
    mm = ModuleType("test_management")
    mm.cast_to, mm.cast_to_gathered, mm.get_cast_buffer, mm.soft_empty_cache = cast_to, gathered, buffer, empty
    mm.is_device_xpu = lambda d: d.type == "xpu"
    mm.current_stream = lambda d: stream
    mm.get_all_torch_devices = lambda: [device]
    mm.comfy = NS(quant_ops=NS(QuantizedTensor=QuantizedTensor))
    mp = ModuleType("test_prefetch")
    mp.MALLOC_GRAPHS, mp.MALLOC_GRAPH_ROGUES, mp.MALLOC_GRAPH_USED = {}, 0, False
    mp.args = NS(disable_comfy_compiler=False, assert_graph_breaks=False, disable_cuda_graphs=True)
    mp.comfy = NS(memory_management=NS(aimdo_enabled=True), model_management=mm)
    mp.malloc_graph_enabled = lambda device: False
    mp.malloc_graph_begin = lambda device: events.append("original-begin")
    mp.malloc_graph_end = lambda: events.append("original-end")
    mp.cleanup_malloc_graph = lambda: events.append("original-cleanup")
    mp.pause_malloc_graph = lambda sync=False: nullcontext()
    mp._malloc_graph_break = lambda: events.append("break")
    def prefetch(queue, device, module, dtype=None, core=None, enable_graph=False, generator=None, malloc_scope=None):
        graph = mp.MALLOC_GRAPHS[threading.get_ident()]
        if malloc_scope is not None:
            graph.iterate(malloc_scope)
        if queue is not None:
            queue.pop(0)
            events.append("housekeeping")
            assert owner.paused
        if core:
            core()
    mp.prefetch_queue_pop = prefetch
    def clear(index):
        events.append("cache-clear")
        return 1
    ck = NS(set_allocation_context=lambda context: events.append("kitchen-context"), clear_nvfp4_lut_cache=clear)
    int8 = NS(set_allocation_context_factory=lambda factory: events.append("int8-context"),
              clear_convrot_hadamard_cache=clear, release_onednn_int8_cache=lambda: 0)
    worker = NS(gc=NS(collect=lambda: events.append("gc")))
    class Queue:
        def __init__(self, flags=None):
            self.flags = flags or {}
        def get_flags(self, reset=True):
            flags = self.flags.copy()
            if reset:
                self.flags = {}
            return flags
    control = ModuleType("comfy_aimdo.control")
    control.get_memory_compiler_capability = lambda: {"native_owner_diagnostic": {
        "active": True, "consumer_contract": "explicit_record_stream"}}
    aimdo = ModuleType("comfy_aimdo")
    aimdo.control = control
    monkeypatch.setitem(sys.modules, "comfy_aimdo", aimdo)
    monkeypatch.setitem(sys.modules, "comfy_aimdo.control", control)
    r = adapter.Runtime(mm, mp, owner, NS(Tensor=Tensor), ck, int8, worker, Queue)
    r.events, r.device, r.stream, r.Tensor, r.QuantizedTensor = events, device, stream, Tensor, QuantizedTensor
    return r


def test_begin_end_replay_keeps_scope_order(runtime):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    r.mp.malloc_graph_end()
    r.mp.malloc_graph_begin(r.device)
    r.mp.malloc_graph_end()
    r.mp.cleanup_malloc_graph()
    assert r.events == ["record", "scope-enter", "int8-context", "kitchen-context", "pop", "scope-exit",
                        "push", "scope-enter", "int8-context", "kitchen-context", "pop", "scope-exit", "close"]
    assert r.mp.MALLOC_GRAPHS == {}
    assert r.mp.MALLOC_GRAPH_ROGUES == 3


@pytest.mark.parametrize("failure", ["close", "abort"])
def test_failed_cleanup_retains_graph_and_retry_counts_once(runtime, failure):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    graph = r.graph()
    setattr(graph.native, "fail_" + failure, True)
    with pytest.raises(ValueError, match=failure + "-error"):
        r.mp.cleanup_malloc_graph()
    assert r.graph() is graph
    assert r.mp.MALLOC_GRAPH_ROGUES == 0
    setattr(graph.native, "fail_" + failure, False)
    r.mp.cleanup_malloc_graph()
    assert r.graph() is None and r.mp.MALLOC_GRAPH_ROGUES == 3
    assert not r.owner.active


def test_pop_error_aborts_and_exits_scope(runtime):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    r.graph().native.fail_pop = True
    with pytest.raises(ValueError, match="pop-error"):
        r.mp.malloc_graph_end()
    assert not r.owner.active and r.graph() is None
    assert r.events[-3:] == ["abort", "scope-exit", "close"]


def test_failed_scope_enter_retains_no_entered_context(runtime):
    r = runtime
    @contextmanager
    def fail(stream):
        raise ValueError("scope-error")
        yield
    r.owner.compiler_scope = fail
    r.install()
    with pytest.raises(ValueError, match="scope-error"):
        r.mp.malloc_graph_begin(r.device)
    assert r.events == ["record", "abort", "close"]
    assert r.graph() is None


@pytest.mark.parametrize("scope", [None, "layer"])
def test_prefetch_preserves_iterate_and_core_allocation_windows(runtime, scope):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    r.events.clear()
    def core():
        assert r.owner.active and not r.owner.paused
        r.events.append("core")
    r.mp.prefetch_queue_pop([None], r.device, None, core=core, malloc_scope=scope)
    assert r.events == (["iterate"] if scope else []) + ["pause", "housekeeping", "resume", "core"]
    r.mp.cleanup_malloc_graph()


def test_prefetch_exception_restores_route(runtime):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    with pytest.raises(IndexError):
        r.mp.prefetch_queue_pop([], r.device, None, malloc_scope="layer")
    assert r.owner.active and not r.owner.paused
    r.mp.cleanup_malloc_graph()


def test_cast_registers_storage_once_before_enqueue(runtime):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    r.events.clear()
    r.mm.cast_to_gathered([r.Tensor(), r.QuantizedTensor()], r.Tensor(), stream=r.stream, r2=r.Tensor())
    assert r.events == ["pause", "consumer", "gather", "resume"]
    r.events.clear()
    r.mm.get_cast_buffer(r.stream, r.device, 512, None)
    assert r.events == ["pause", "buffer", "resume"]
    r.mp.cleanup_malloc_graph()


@pytest.mark.parametrize("unload", [False, True])
def test_free_request_clears_only_after_worker_gc(runtime, unload):
    r = runtime
    r.install()
    q = r.queue_type({"free_memory": True, "unload_models": unload})
    assert q.get_flags(reset=False)["free_memory"]
    r.mm.soft_empty_cache()
    assert r.events == ["empty"]
    q.get_flags()
    r.mm.soft_empty_cache()  # An earlier unload/pressure flush must not consume it.
    assert r.events == ["empty", "empty"]
    r.worker.gc.collect()
    r.mm.soft_empty_cache()
    assert r.events[-5:] == ["gc", "empty", "cache-clear", "cache-clear", "empty"]
    r.events.clear()
    r.worker.gc.collect()
    r.mm.soft_empty_cache()
    # No second cleanup in the same request. GC must not re-arm a consumed flag.
    assert r.events == ["gc", "empty"]


def test_cuda_lifecycle_and_compiler_off_delegate(runtime):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(NS(type="cuda"))
    r.mp.malloc_graph_end()
    r.mp.cleanup_malloc_graph()
    assert r.events == ["original-begin", "original-end", "original-cleanup"]
    r.mp.args.disable_comfy_compiler = True
    r.mp.malloc_graph_begin(r.device)
    assert r.graph() is None


def test_noop_cast_keeps_alias_without_registering_consumer(runtime):
    r = runtime
    r.install()
    r.mp.malloc_graph_begin(r.device)
    r.events.clear()
    tensor = r.Tensor()
    assert r.mm.cast_to(tensor, stream=r.stream) is tensor
    assert r.events == ["copy"]  # The original function, no added context.
    r.mp.cleanup_malloc_graph()


def test_install_is_idempotent_and_keeps_existing_wrapper(runtime):
    r = runtime
    original = r.mm.cast_to
    def external(*args, **kwargs):
        r.events.append("external")
        return original(*args, **kwargs)
    import functools
    r.mm.cast_to = functools.wraps(original)(external)
    r.install()
    installed = r.mm.cast_to
    r.install()
    assert r.mm.cast_to is installed
    r.mm.cast_to(r.Tensor())
    assert r.events == ["external", "copy"]


@pytest.mark.skipif(not SOURCE, reason="set COMFYUI_CALLER_TEST_ROOT for pristine source integration")
def test_pristine_source_admission_and_comments(runtime, tmp_path):
    import shutil
    compat = sys.modules["compiler_adapter_test.compiler_compat"]
    assert compat.preflight(SOURCE) == str(Path(SOURCE).resolve())
    for path in (*compat.SIGNATURES, "main.py", "execution.py"):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(SOURCE) / path, destination)
    with (tmp_path / "comfy/model_prefetch.py").open("a") as stream:
        stream.write("\n# Compatible upstream edit\ndef unrelated_api():\n    return 1\n")
    compat.preflight(tmp_path)
    path = tmp_path / "comfy/model_prefetch.py"
    path.write_text(path.read_text().replace("consumed = queue.pop(0)", "consumed = queue.pop(1)"))
    with pytest.raises(compat.CompatibilityError, match="allocation boundary changed"):
        compat.preflight(tmp_path)


@pytest.mark.skipif(not SOURCE, reason="set COMFYUI_CALLER_TEST_ROOT for pristine source integration")
@pytest.mark.parametrize("scope", [None, "layer"])
def test_unmodified_upstream_prefetch(runtime, scope):
    r = runtime
    r.mm.is_device_cuda = lambda d: False
    def cleanup(module, modules):
        assert r.owner.paused
        r.events.append("cleanup-prefetch")
    def pin(modules, device, dtype):
        assert r.owner.paused
        r.events.append("pin")
        return None, False
    namespace = {"MALLOC_GRAPHS": r.mp.MALLOC_GRAPHS, "threading": threading,
                 "args": r.mp.args, "comfy": r.mp.comfy, "_malloc_graph_break": r.mp._malloc_graph_break,
                 "cleanup_prefetched_modules": cleanup, "pin_modules": pin}
    r.mp.prefetch_queue_pop = source_function("comfy/model_prefetch.py", "prefetch_queue_pop", namespace)
    class Root:
        def modules(self):
            assert r.owner.paused
            r.events.append("traverse")
            yield NS(_v=True)
    r.install()
    r.mp.malloc_graph_begin(r.device)
    r.events.clear()
    def core():
        assert r.owner.active and not r.owner.paused
        r.events.append("core")
    r.mp.prefetch_queue_pop([(None, (None, [])), Root()], r.device, None,
                            core=core, malloc_scope=scope)
    assert r.events == (["iterate"] if scope else []) + ["pause", "cleanup-prefetch", "traverse", "pin", "resume", "core"]
    r.mp.cleanup_malloc_graph()


@pytest.mark.skipif(not SOURCE, reason="set COMFYUI_CALLER_TEST_ROOT for pristine source integration")
@pytest.mark.parametrize("flags,clears", [({"free_memory": True, "unload_models": False}, True),
    ({"free_memory": True}, True), ({"unload_models": True}, False), ({}, False)])
def test_unmodified_upstream_prompt_worker(runtime, flags, clears):
    r = runtime
    class StopWorker(BaseException):
        pass
    class Queue(r.queue_type):
        mutex = threading.RLock()
        reads = 0
        def get(self, timeout):
            self.reads += 1
            if self.reads > 1:
                raise StopWorker
            return None
    Queue.get_flags = source_function("execution.py", "get_flags", {}, "PromptQueue")
    r.queue_type = Queue
    executor = NS(reset=lambda: r.events.append("reset"))
    execution = NS(CacheType=NS(RAM_PRESSURE=1, CLASSIC=2, LRU=3, NONE=4),
                   PromptExecutor=lambda *a, **kw: executor)
    r.mm.unload_all_models = lambda: r.events.append("unload")
    r.worker = ModuleType("caller_worker_test")
    r.worker.gc = NS(collect=lambda: r.events.append("gc"))
    r.worker.args = NS(cache_classic=False, cache_none=True, cache_lru=0)
    r.worker.execution, r.worker.comfy = execution, r.mp.comfy
    r.worker.time = NS(perf_counter=lambda: 100)
    r.worker.hook_breaker_ac10a0 = NS(restore_functions=lambda: None)
    function = source_function("main.py", "prompt_worker", r.worker.__dict__)
    r.install()
    asset = NS(queue_output_scan=lambda: None, resume_background_scan=lambda: None)
    with pytest.raises(StopWorker):
        function(Queue(flags), NS(client_id=None), asset)
    assert ("cache-clear" in r.events) == clears
    if clears:
        assert r.events[-5:] == ["gc", "empty", "cache-clear", "cache-clear", "empty"]


@pytest.mark.parametrize('mode', ['clean', 'retired', 'wrong_abi'])
def test_build_separates_native_abi_from_comfyui_revision(tmp_path, mode):
    import subprocess
    docker = tmp_path / 'docker'
    log = tmp_path / 'arguments'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$DOCKER_TEST_ARGUMENTS"\n')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'],
        DOCKER_TEST_ARGUMENTS=str(log), AIMDO_XPU_BUILD_NATIVE_OWNER_DIAGNOSTIC='1',
        OMNI_TORCH_VERSION='2.14.0+xpu', OMNI_AIMDO_TORCH214_CALLER_PATCH='0',
        COMFYUI_COMMIT='0123456789abcdef0123456789abcdef01234567')
    if mode == 'retired':
        env['OMNI_AIMDO_TORCH214_CALLER_PATCH'] = '1'
    elif mode == 'wrong_abi':
        env['OMNI_TORCH_VERSION'] = '2.13.0+xpu'
    result = subprocess.run(['bash', str(PLUGIN.parent / 'build.sh')], env=env, capture_output=True, text=True)
    if mode == 'clean':
        assert result.returncode == 0, result.stderr
        assert 'AIMDO_XPU_BUILD_NATIVE_OWNER_DIAGNOSTIC=1' in log.read_text()
        assert 'COMFYUI_COMMIT=0123456789abcdef0123456789abcdef01234567' in log.read_text()
    else:
        assert result.returncode != 0 and not log.exists()
        assert ('source patch is retired' if mode == 'retired' else 'Torch 2.14 XPU ABI') in result.stderr
