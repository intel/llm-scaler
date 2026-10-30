"""ComfyUI 0.43 writable backing against the installed XPU provider."""
import gc
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def installed():
    root = Path(os.environ.get("COMFYUI_ROOT", "/llm/ComfyUI"))
    name = "_comfyui_omnixpu_runtime_bootstrap"
    bootstrap = sys.modules.get(name) or load(name, PLUGIN / "runtime_bootstrap.py")
    from comfy_aimdo import control  # The real startup imports the official entry first.
    state = bootstrap.get_state()
    if state["providers"].get("comfy_aimdo.xpu", {}).get("status") != "active":
        state = bootstrap.bootstrap(dynamic_vram_override=True)
    assert state["providers"]["comfy_aimdo.xpu"]["status"] == "active", state
    assert state["providers"]["comfy_kitchen.xpu"]["status"] == "active", state
    from comfy_aimdo import control
    assert control.init_devices([0])
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root))
        import comfy.ops as ops
        import comfy.memory_management as memory
        import comfy.model_management as management
        yield SimpleNamespace(ops=ops, memory=memory, management=management,
                              device=management.get_torch_device())
    gc.collect()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    control.deinit()


@pytest.fixture
def runtime(installed, monkeypatch):
    adapter = load("omnixpu_writable_weight_test", PLUGIN / "adapters/writable_weights.py")
    for owner, name in ((installed.memory, "write_tensor_file_slice"),
                        (installed.memory, "read_tensor_file_slice_into"),
                        (installed.ops.WritableWeight, "writeback")):
        function = getattr(owner, name)
        monkeypatch.setattr(owner, name, getattr(function, adapter._MARKER, function))
    installed.adapter = adapter
    return installed


def attach(memory, target, backing, offset):
    size = target.numel() * target.element_size()
    target.untyped_storage()._comfy_tensor_file_slice = memory.TensorFileSlice(
        backing, threading.Lock(), offset, size)


def test_installs_atomically_and_is_idempotent(runtime):
    assert runtime.adapter.apply() == (True, None)
    functions = runtime.memory.write_tensor_file_slice, runtime.ops.WritableWeight.writeback
    assert runtime.adapter.apply() == (True, "already patched")
    assert functions == (runtime.memory.write_tensor_file_slice, runtime.ops.WritableWeight.writeback)


def test_file_write_read_explicit_xpu_stream_and_bounds(runtime):
    assert runtime.adapter.apply()[0]
    size, offset = 1048579, 13
    target = torch.empty(size, dtype=torch.uint8)
    stream = torch.xpu.Stream()
    with tempfile.TemporaryFile() as backing:
        backing.write(bytes([177]) * (offset + size + 11))
        backing.flush()
        attach(runtime.memory, target, backing, offset)
        with torch.xpu.stream(stream):
            source = torch.arange(size, device=runtime.device, dtype=torch.int64).to(torch.uint8)
            runtime.memory.write_tensor_file_slice(target, source, stream=stream)
            restored = torch.empty_like(source)
            assert runtime.memory.read_tensor_file_slice_into(target, restored, stream=stream)
        stream.synchronize()
        assert torch.equal(restored, source)
        backing.seek(0)
        result = backing.read()
        assert result[:offset] == bytes([177]) * offset
        assert result[-11:] == bytes([177]) * 11
        assert result[offset:-11] == source.cpu().numpy().tobytes()
        with pytest.raises(ValueError, match="complete contiguous source"):
            runtime.memory.write_tensor_file_slice(target, source[:-1])


@pytest.mark.parametrize("pinned", [True, False])
def test_writable_weight_pinned_or_file_writeback(runtime, monkeypatch, pinned):
    assert runtime.adapter.apply()[0]
    layer = runtime.ops.WritableWeight((257,), torch.float32, device="cpu")
    layer._pin_state = {"fast_disk": False}
    layer._v_signature = None
    layer._pin_subset = "weights" if pinned else None
    layer._pins = {"weights": {"dirty": False}}
    host = torch.empty(2048, dtype=torch.uint8, pin_memory=True) if pinned else None
    monkeypatch.setattr(runtime.ops.comfy.pinned_memory, "get_pin", lambda *args: host)
    monkeypatch.setattr(runtime.ops.comfy.pinned_memory, "pin_memory", lambda *args, **kwargs: False)
    stream = torch.xpu.Stream()
    with tempfile.TemporaryFile() as backing:
        backing.write(bytes(1028))
        backing.flush()
        attach(runtime.memory, layer.weight, backing, 0)
        with torch.xpu.stream(stream):
            source = torch.empty(257, device=runtime.device).fill_(3.125)
            layer.writeback(source)
        if pinned:
            observed = runtime.memory.interpret_gathered_like([layer.weight], host)[0]
            assert layer._pins["weights"]["dirty"]
            assert torch.equal(observed, source.cpu())
        else:
            backing.seek(0)
            assert backing.read() == source.cpu().numpy().tobytes()


def test_unknown_contract_is_rejected_without_partial_patching(runtime, monkeypatch):
    def unknown(self, value):
        return value
    monkeypatch.setattr(runtime.ops.WritableWeight, "writeback", unknown)
    original = runtime.memory.write_tensor_file_slice
    ok, reason = runtime.adapter.apply()
    assert not ok and "unsupported" in reason
    assert runtime.memory.write_tensor_file_slice is original
