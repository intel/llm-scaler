"""Host-only QKV routing and weight-context lifetime tests; no Torch runtime."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"


class Tensor:
    def __init__(self):
        self.device = SimpleNamespace(type="xpu")
        self.dtype = "bf16"
        self.ndim = 3
        self.shape = (1, 17, 4096)
        self.requires_grad = False
        self.contiguous = True

    def is_contiguous(self):
        return self.contiguous


class Layer:
    def __init__(self, name, device):
        self.name = name
        self.calls = []
        self.weight = SimpleNamespace(
            shape=(4096, 4096), dtype="int8", _layout_cls="TensorWiseINT8Layout",
            _qdata=SimpleNamespace(device=device),
            _params=SimpleNamespace(convrot=True, convrot_groupsize=256,
                                    transposed=False, orig_dtype="bf16",
                                    scale=SimpleNamespace(device=device)),
        )
        self.quant_format = "int8_tensorwise"
        self.layout_type = "TensorWiseINT8Layout"
        self.bias = None
        self._forward_hooks = {}
        self._forward_pre_hooks = {}

    def forward(self, x):
        self.calls.append(x)
        return self.name + ":original"

    def __call__(self, x):
        return self.forward(x)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    events, gemms = [], []
    x = Tensor()
    layers = [Layer(name, x.device) for name in ("q", "k", "v")]
    torch = ModuleType("torch")
    torch.Tensor, torch.bfloat16 = Tensor, "bf16"
    torch.is_grad_enabled = lambda: False
    torch.compiler = SimpleNamespace(is_compiling=lambda: False)
    torch.xpu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(device_id=0xE223))
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    mm = ModuleType("comfy.model_management")
    mm.in_training = False
    ops = ModuleType("comfy.ops")
    active = []

    @contextmanager
    def cast(layer, **kwargs):
        assert not active, "projection weight lifetimes must not overlap"
        assert kwargs["offloadable"] and kwargs["want_requant"]
        active.append(layer.name)
        events.append("enter:" + layer.name)
        try:
            yield layer.weight, layer.bias
        finally:
            events.append("exit:" + layer.name)
            active.pop()

    ops.CastBiasWeightContext = cast
    ops.run_every_op = lambda: events.append("run_every_op")

    def rotate(value, group_size):
        assert value is x and group_size == 256
        events.append("rotate")
        return object()

    def quantize(value):
        events.append("quantize")
        return object(), object()

    def gemm(codes, scale, data, weight_scale, **kwargs):
        assert len(active) == 1
        gemms.append((codes, scale, active[0]))
        events.append("gemm:" + active[0])
        return active[0] + ":shared"

    native = SimpleNamespace(rotate_convrot=True, quantize_int8_rowwise_fused=True,
                             int8_linear_prequantized=True)
    int8 = SimpleNamespace(_get_native=lambda: native, rotate_convrot=rotate,
                           quantize_int8_rowwise=quantize,
                           int8_linear_prequantized=gemm)
    omni = ModuleType("omni_xpu_kernel")
    omni.int8 = int8
    for name, value in (("torch", torch), ("comfy", comfy),
                        ("comfy.model_management", mm), ("comfy.ops", ops),
                        ("omni_xpu_kernel", omni)):
        monkeypatch.setitem(sys.modules, name, value)
    comfy.model_management, comfy.ops = mm, ops
    adapter = load("omnixpu_qkv_host_test", PLUGIN / "adapters/qwen_image21_qkv.py")
    module = SimpleNamespace(**dict(zip(("to_q", "to_k", "to_v"), layers)))
    return SimpleNamespace(adapter=adapter, module=module, x=x, layers=layers,
                           torch=torch, mm=mm, int8=int8, events=events, gemms=gemms)


def test_preparation_is_shared_only_within_one_call(runtime):
    r = runtime
    for _ in range(2):
        assert r.adapter.project_qkv(r.module, r.x) == (
            "q:shared", "k:shared", "v:shared")
    one_call = ["rotate", "quantize"]
    for name in ("q", "k", "v"):
        one_call += ["run_every_op", "enter:" + name, "gemm:" + name, "exit:" + name]
    assert r.events == one_call * 2
    for offset in (0, 3):
        assert all(row[0] is r.gemms[offset][0] and row[1] is r.gemms[offset][1]
                   for row in r.gemms[offset:offset + 3])
    assert r.gemms[0][0] is not r.gemms[3][0]
    assert not any(layer.calls for layer in r.layers)


@pytest.mark.parametrize("case", ["cpu", "dtype", "strided", "grad", "training",
                                  "compiling", "missing_native"])
def test_unsupported_input_retains_original_projections(runtime, case):
    r = runtime
    if case == "cpu":
        r.x.device.type = "cpu"
    elif case == "dtype":
        r.x.dtype = "fp16"
    elif case == "strided":
        r.x.contiguous = False
    elif case == "grad":
        r.x.requires_grad = True
    elif case == "training":
        r.mm.in_training = True
    elif case == "compiling":
        r.torch.compiler.is_compiling = lambda: True
    else:
        r.int8._get_native = lambda: None
    assert r.adapter.project_qkv(r.module, r.x) == ("q:original", "k:original", "v:original")
    assert [layer.calls for layer in r.layers] == [[r.x]] * 3
    assert not r.events


def test_shared_preparation_does_not_query_a_physical_device_id(runtime):
    r = runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("shared QKV must select APIs, not a physical SKU")

    r.torch.xpu.get_device_properties = forbidden
    assert r.adapter.project_qkv(r.module, r.x) == ("q:shared", "k:shared", "v:shared")


@pytest.mark.parametrize("shape,widths", [
    ((1, 17, 256), (128, 256, 512)),
    ((2, 5, 1024), (1024, 512, 256)),
    ((1, 9, 6144), (6144, 4096, 2048)),
])
def test_dimensions_come_from_the_input_and_each_projection(runtime, shape, widths):
    r = runtime
    r.x.shape = shape
    for layer, width in zip(r.layers, widths):
        layer.weight.shape = (width, shape[-1])
    assert r.adapter.project_qkv(r.module, r.x) == ("q:shared", "k:shared", "v:shared")
    assert r.events.count("rotate") == r.events.count("quantize") == 1


@pytest.mark.parametrize("case", ["nondivisible", "empty_width", "weight_rank", "weight_k"])
def test_incompatible_dimensions_fall_back_before_preparation(runtime, case):
    r = runtime
    if case == "nondivisible":
        r.x.shape = (1, 17, 300)
    elif case == "empty_width":
        r.x.shape = (1, 17, 0)
    elif case == "weight_rank":
        r.layers[1].weight.shape = (4096,)
    else:
        r.layers[1].weight.shape = (4096, 1024)
    assert r.adapter.project_qkv(r.module, r.x) == ("q:original", "k:original", "v:original")
    assert not r.events


@pytest.mark.parametrize("name", ["rotate_convrot", "quantize_int8_rowwise_fused",
                                  "int8_linear_prequantized"])
def test_each_required_native_api_is_checked(runtime, name):
    r = runtime
    delattr(r.int8._get_native(), name)
    assert r.adapter.project_qkv(r.module, r.x) == ("q:original", "k:original", "v:original")
    assert not r.events


@pytest.mark.parametrize("case", ["hook", "weight_patch", "bias", "rotation", "custom_forward"])
def test_projection_customizations_fall_back_before_preparation(runtime, case):
    r = runtime
    layer = r.layers[1]
    if case == "hook":
        layer._forward_hooks[1] = object()
    elif case == "weight_patch":
        layer.weight_function = [object()]
    elif case == "bias":
        layer.bias = object()
    elif case == "rotation":
        layer.weight._params.convrot = False
    else:
        layer.forward = lambda x: "custom"
    expected = tuple(item(r.x) for item in r.layers)
    for item in r.layers:
        item.calls.clear()
    assert r.adapter.project_qkv(r.module, r.x) == expected
    assert not r.events


def test_cast_context_releases_weight_on_native_failure(runtime):
    r = runtime

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    r.int8.int8_linear_prequantized = fail
    with pytest.raises(RuntimeError, match="native failure"):
        r.adapter.project_qkv(r.module, r.x)
    assert r.events[-2:] == ["enter:q", "exit:q"]
    assert not any(event == "enter:k" for event in r.events)


def test_feature_switch_is_opt_in_and_obeys_master(monkeypatch):
    monkeypatch.delenv("OMNIXPU_QWEN_IMAGE21_QKV", raising=False)
    monkeypatch.setenv("OMNIXPU_ENABLE", "1")
    assert not load("qkv_default", PLUGIN / "config.py").config.qwen_image21_qkv
    monkeypatch.setenv("OMNIXPU_QWEN_IMAGE21_QKV", "1")
    assert load("qkv_enabled", PLUGIN / "config.py").config.qwen_image21_qkv
    monkeypatch.setenv("OMNIXPU_ENABLE", "0")
    assert not load("qkv_master_disabled", PLUGIN / "config.py").config.qwen_image21_qkv


class Attention:
    heads = 32

    def forward(self, x):
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1)
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)
        return q, k, v


def install_attention_module(monkeypatch, attention):
    for name in ("comfy.ldm", "comfy.ldm.qwen_image21", "comfy.ldm.qwen_image21.model"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    sys.modules["comfy.ldm.qwen_image21.model"].Attention = attention


def test_attention_patch_is_idempotent_and_preserves_projection_order(runtime, monkeypatch):
    class CompatibleAttention(Attention):
        pass

    r = runtime
    install_attention_module(monkeypatch, CompatibleAttention)
    calls = []

    class Projection:
        def __init__(self, name):
            self.name = name

        def view(self, *shape):
            return self.name, shape

    def project(module, x):
        calls.append((module, x))
        return tuple(Projection(name) for name in ("q", "k", "v"))

    monkeypatch.setattr(r.adapter, "project_qkv", project)
    assert r.adapter.apply() == (True, "")
    patched = CompatibleAttention.forward
    assert r.adapter.apply() == (True, "already applied")
    assert CompatibleAttention.forward is patched
    attention = CompatibleAttention()
    assert attention.forward(r.x) == tuple((name, (1, 17, 32, -1)) for name in ("q", "k", "v"))
    assert calls == [(attention, r.x)]
    assert Attention.forward is not patched


def test_incompatible_attention_source_is_not_patched(runtime, monkeypatch):
    class IncompatibleAttention:
        def forward(self, x):
            return x

    install_attention_module(monkeypatch, IncompatibleAttention)
    original = IncompatibleAttention.forward
    with pytest.raises(RuntimeError, match="unsupported Qwen Image 2.1 attention source"):
        runtime.adapter.apply()
    assert IncompatibleAttention.forward is original
