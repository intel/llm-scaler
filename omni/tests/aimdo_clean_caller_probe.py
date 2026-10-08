"""Explicit device diagnostic against pristine ComfyUI, never auto-collected."""
from __future__ import annotations
import argparse
import ast
from contextlib import ExitStack
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", required=True, type=Path)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--affinity", required=True)
    parser.add_argument("--reference-caller-root", type=Path)
    args = parser.parse_args()
    assert os.environ.get("ZE_AFFINITY_MASK") == args.affinity
    assert os.environ.get("AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC") == "1"
    plugin = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
    root = args.comfyui_root.resolve()
    sys.path.insert(0, str(root))
    worker = types.ModuleType("main")
    worker.__file__, worker.gc = str(root / "main.py"), gc
    sys.modules["main"] = worker
    # Reproduce the official prestartup import and provider takeover order.
    import comfy_aimdo.control
    bootstrap = load("_clean_caller_bootstrap", plugin / "runtime_bootstrap.py")
    state = bootstrap.bootstrap(dynamic_vram_override=True)
    assert state["memory_compiler"]["status"] == "admitted", state
    import torch
    assert torch.xpu.device_count() == 1
    props = torch.xpu.get_device_properties(0)
    assert str(props.uuid) == args.expected_uuid and props.device_id == 0xE223
    from comfy_aimdo import control, native_owner
    print(json.dumps({"native_control": control.__file__, "native_library": str(control.lib._name),
                      "native_owner": native_owner.__file__}), flush=True)
    assert control.init_devices([0])
    import comfy.model_management as mm
    import comfy.model_prefetch as mp
    import comfy.memory_management as memory
    memory.aimdo_enabled = True
    package = types.ModuleType("clean_caller_adapter")
    package.__path__ = [str(plugin)]
    sys.modules[package.__name__] = package
    adapters = types.ModuleType(package.__name__ + ".adapters")
    adapters.__path__ = [str(plugin / "adapters")]
    sys.modules[adapters.__name__] = adapters
    adapter = load(adapters.__name__ + ".aimdo_memory_compiler", plugin / "adapters/aimdo_memory_compiler.py")
    if args.reference_caller_root is None:
        assert adapter.apply()[0]
        runtime = adapter._RUNTIME
    else:
        # Execute the retained reference in this fresh process only. No source
        # files are changed, and the reference is never installed as an adapter.
        reference = args.reference_caller_root
        path = reference / "comfy/model_prefetch.py"
        exec(compile(path.read_text(), str(path), "exec"), mp.__dict__)
        path = reference / "comfy/model_management.py"
        names = {"get_cast_buffer", "cast_to_gathered", "_cast_storage_tensors",
                 "_cast_stream_context", "cast_to"}
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        mm.__dict__.update(native_owner=native_owner, ExitStack=ExitStack)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), mm.__dict__)
        import threading
        runtime = types.SimpleNamespace(graph=lambda: mp.MALLOC_GRAPHS.get(threading.get_ident()))
    device = torch.device("xpu:0")
    side = torch.xpu.Stream(device=device)
    expected = torch.arange(4097, dtype=torch.int64).remainder(251).to(torch.uint8)
    receipts = []
    for cycle in range(2):
        mp.malloc_graph_begin(device)
        graph = runtime.graph()
        tensor = torch.empty((4097,), dtype=torch.uint8, device=device)
        tensor.copy_(expected)
        pointer = tensor.untyped_storage().data_ptr()
        assert native_owner.is_compiler_owner(pointer)
        side.wait_stream(torch.xpu.current_stream(device))
        copied = mm.cast_to(tensor, device=torch.device("cpu"), stream=side)
        side.synchronize()
        assert torch.equal(copied, expected)
        print(json.dumps({"stage": "after-cast", "proxy": native_owner.snapshot()}), flush=True)
        del tensor, copied
        persistent = []
        output = []
        original_pin = mp.pin_modules
        class Root:
            def modules(self):
                value = torch.empty((257,), dtype=torch.uint8, device=device)
                assert not native_owner.is_compiler_owner(value.untyped_storage().data_ptr())
                persistent.append(value)
                yield types.SimpleNamespace(_v=True)
        def pin(modules, device, dtype):
            value = torch.empty((263,), dtype=torch.uint8, device=device)
            assert not native_owner.is_compiler_owner(value.untyped_storage().data_ptr())
            persistent.append(value)
            return None, False
        def core():
            value = torch.empty((271,), dtype=torch.uint8, device=device)
            assert native_owner.is_compiler_owner(value.untyped_storage().data_ptr())
            value.fill_(17)
            output.append(value)
        try:
            mp.pin_modules = pin  # Synthetic pin body; original queue function runs.
            mp.prefetch_queue_pop([None, Root()], device, types.SimpleNamespace(),
                                  core=core, malloc_scope="prefetch-core")
        finally:
            mp.pin_modules = original_pin
        torch.xpu.synchronize(device)
        side.wait_stream(torch.xpu.current_stream(device))
        core_copy = mm.cast_to(output[0], device=torch.device("cpu"), stream=side)
        side.synchronize()
        assert torch.equal(core_copy, torch.full((271,), 17, dtype=torch.uint8))
        print(json.dumps({"stage": "after-core", "proxy": native_owner.snapshot()}), flush=True)
        receipts.append({"cycle": cycle, "payload_bytes": 4097, "payload_equal": True,
                         "cast_storage": pointer, "persistent_outside_graph": len(persistent),
                         "core_inside_graph": True})
        del core_copy
        output.clear()
        persistent.clear()
        gc.collect()
        # ComfyUI flushes its final named prefetch scope before ending the root.
        mp.prefetch_queue_pop(None, device, None, malloc_scope="prefetch-core")
        print(json.dumps({"stage": "before-end", "proxy": native_owner.snapshot()}), flush=True)
        mp.malloc_graph_end()
    mp.cleanup_malloc_graph()
    assert not mp.MALLOC_GRAPHS
    torch.xpu.synchronize(device)
    snapshot = native_owner.graph_ownership_snapshot()
    assert snapshot == {"live": 0, "deferred": 0}, snapshot
    proxy = native_owner.snapshot()
    assert proxy[12] == proxy[13] and proxy[15] == 0 and proxy[16] == 0, proxy
    if args.reference_caller_root is None:
        import threading
        queue = object.__new__(runtime.queue_type)
        queue.mutex = threading.RLock()
        queue.flags = {"free_memory": True, "unload_models": False}
        assert queue.get_flags() == {"free_memory": True, "unload_models": False}
        worker.gc.collect()
        assert runtime.local.after_gc
        mm.soft_empty_cache()
        assert not runtime.local.after_gc and not runtime.local.free_memory
    assert not control.get_memory_compiler_capability()["available"]
    print(json.dumps({"status": "passed", "device_uuid": str(props.uuid), "device_id": props.device_id,
                      "affinity": args.affinity, "receipts": receipts,
                      "graph_ownership": snapshot, "public_compiler_available": False,
                      "proxy": proxy, "reference_caller": args.reference_caller_root is not None,
                      "explicit_free_request": args.reference_caller_root is None,
                      "evidence_class": "functional_lifecycle",
                      "scope": "pristine caller; synthetic pin body; no workflow/image acceptance"}))


if __name__ == "__main__":
    main()
