"""ComfyUI eligibility respects backend policy and explicit model precision."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).parents[1] / 'ComfyUI-OmniXPU/adapters/quantized_matmul.py'
spec = importlib.util.spec_from_file_location('quantized_matmul_adapter_test', PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.parametrize('device', ['xpu', 'cpu', 'cuda'])
@pytest.mark.parametrize('backend', ['xpu', 'eager', None])
def test_bridge_keeps_device_and_backend_policy(device, backend):
    upstream = {'int8_tensorwise', 'nvfp4', 'convrot_w4a4', 'asym_w4a8_int8'}
    calls = []
    def original(device=None):
        calls.append(device)
        return upstream
    def selected(name):
        if backend is None:
            raise RuntimeError('missing implementation')
        return backend
    registry = SimpleNamespace(is_available=lambda name: True, get_capable_backend=selected)
    bridged = adapter._wrap(original, lambda: torch.device(device), registry, RuntimeError)
    expected = upstream - {'int8_tensorwise'} if device == backend == 'xpu' else upstream
    assert bridged() == expected
    assert bridged(torch.device(device)) == expected
    assert calls == [None, torch.device(device)]
    assert 'int8_tensorwise' in upstream


@pytest.mark.parametrize('missing', adapter._INT8_OPERATIONS)
def test_partial_native_backend_keeps_emulation(missing):
    registry = SimpleNamespace(is_available=lambda name: True,
        get_capable_backend=lambda name: 'eager' if name == missing else 'xpu')
    bridged = adapter._wrap(lambda device=None: {'int8_tensorwise'},
        lambda: torch.device('xpu'), registry, RuntimeError)
    assert bridged() == {'int8_tensorwise'}


def test_disabled_native_backend_keeps_emulation():
    registry = SimpleNamespace(is_available=lambda name: False,
        get_capable_backend=lambda name: pytest.fail('disabled backend was consulted'))
    bridged = adapter._wrap(lambda device=None: {'int8_tensorwise'},
        lambda: torch.device('xpu'), registry, RuntimeError)
    assert bridged() == {'int8_tensorwise'}


@pytest.mark.parametrize('device', ['xpu', 'cpu', 'cuda'])
@pytest.mark.parametrize('backend', ['xpu', 'eager', None])
@pytest.mark.parametrize('available', [False, True])
def test_w4_bridge_requires_packed_companion_and_selected_backend(device, backend, available):
    upstream = {'asym_w4a8_int8', 'w6a8_int8', 'convrot_w4a4'}

    def selected(name):
        if name == 'w4a8_int8_linear':
            if backend is None:
                raise RuntimeError('missing W4 implementation')
            return backend
        return 'eager'

    registry = SimpleNamespace(is_available=lambda name: True, get_capable_backend=selected)
    bridged = adapter._wrap(lambda device=None: upstream, lambda: torch.device(device),
                            registry, RuntimeError, lambda: available)
    expected = upstream - {'asym_w4a8_int8'} if device == backend == 'xpu' and available else upstream
    assert bridged() == expected
    assert 'asym_w4a8_int8' in upstream


@pytest.fixture
def comfy_bridge(monkeypatch):
    ops = pytest.importorskip('comfy.ops')
    import comfy_kitchen as ck
    assert torch.xpu.is_available() and ck.list_backends()['xpu']['available']
    original = ops.get_disabled_quant_formats
    monkeypatch.setattr(ops, 'get_disabled_quant_formats', original)
    assert adapter.apply() == (True, '')
    assert adapter.apply() == (True, 'already patched')
    assert 'int8_tensorwise' not in ops.get_disabled_quant_formats(torch.device('xpu'))
    assert ops.get_disabled_quant_formats(torch.device('cpu')) == original(torch.device('cpu'))
    return ops


def loaded_linear(ops, *, convrot, full_precision, width=256, output=96, group=256):
    from comfy_kitchen.tensor import QuantizedTensor
    weight = torch.randn((output, width), dtype=torch.bfloat16, device='xpu')
    quantized = QuantizedTensor.from_float(weight, 'TensorWiseINT8Layout',
        per_channel=convrot, convrot=convrot, convrot_groupsize=group)
    klass = ops.mixed_precision_ops({}, torch.bfloat16,
        disabled=ops.get_disabled_quant_formats(torch.device('xpu')))
    model = klass.Linear(width, output, bias=False, device='xpu')
    configuration = {'format': 'int8_tensorwise', 'convrot': convrot,
        'convrot_groupsize': group, 'full_precision_matrix_mult': full_precision}
    # Only JSON metadata is decoded on CPU by ComfyUI's state-dict loader.
    # All weights, activations, references and numerical checks stay on XPU.
    encoded = torch.tensor(list(json.dumps(configuration).encode()), dtype=torch.uint8)
    model.load_state_dict({'weight': quantized._qdata,
        'weight_scale': quantized._params.scale, 'comfy_quant': encoded})
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('convrot', [False, True])
@pytest.mark.parametrize('full_precision', [False, True])
def test_actual_comfy_loader_and_native_dispatch(comfy_bridge, monkeypatch, convrot, full_precision):
    import comfy_kitchen as ck
    from omni_xpu_kernel import int8
    model = loaded_linear(comfy_bridge, convrot=convrot, full_precision=full_precision)
    assert model._full_precision_mm is full_precision
    assert model._full_precision_mm_config is full_precision
    x = torch.randn((37, 256), dtype=torch.bfloat16, device='xpu')
    weight = model.weight
    with torch.no_grad():
        if full_precision:
            expected = torch.nn.functional.linear(x, weight.dequantize())
        else:
            expected = ck.int8_linear(x, weight._qdata, weight._params.scale,
                out_dtype=x.dtype, convrot=convrot, convrot_groupsize=256)
    original = int8.int8_linear
    calls = []
    def observed(*args, **kwargs):
        calls.append(args[0].device.type)
        return original(*args, **kwargs)
    monkeypatch.setattr(int8, 'int8_linear', observed)
    with torch.no_grad():
        actual = model(x)
    assert calls == ([] if full_precision else ['xpu'])
    assert torch.equal(actual, expected)


def test_actual_weight_patch_preserves_comfy_full_precision_path(comfy_bridge, monkeypatch):
    from omni_xpu_kernel import int8
    model = loaded_linear(comfy_bridge, convrot=True, full_precision=False)
    model.weight_function = [lambda weight: weight * 1.25]
    x = torch.randn((37, 256), dtype=torch.bfloat16, device='xpu')
    monkeypatch.setattr(int8, 'int8_linear',
        lambda *args, **kwargs: pytest.fail('weight patch bypassed the ComfyUI cast path'))
    with torch.no_grad():
        expected = torch.nn.functional.linear(x, model.weight.dequantize() * 1.25)
        actual = model(x)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('full_precision', [False, True])
@pytest.mark.parametrize('rows', [1, 3, 9])
@torch.inference_mode()
def test_actual_comfy_w4_loader_and_packed_dispatch(comfy_bridge, monkeypatch, full_precision, rows):
    import comfy_kitchen as ck
    from omni_xpu_kernel import int8

    ops = comfy_bridge
    assert 'asym_w4a8_int8' not in ops.get_disabled_quant_formats(torch.device('xpu'))
    width, output, group, rotation = 256, 96, 16, 64
    klass = ops.mixed_precision_ops({}, torch.bfloat16,
        disabled=ops.get_disabled_quant_formats(torch.device('xpu')))
    model = klass.Linear(width, output, bias=False, device='xpu')
    config = {'format': 'asym_w4a8_int8', 'group_size': group,
              'convrot_groupsize': rotation, 'full_precision_matrix_mult': full_precision}
    model.load_state_dict({
        'weight': torch.randint(-128, 128, (output, width // 2), device='xpu', dtype=torch.int8),
        'weight_s_rel': torch.full((output, width // group), 4., device='xpu').to(torch.float8_e4m3fn),
        'weight_s_channel': torch.full((output,), .0025, device='xpu'),
        'weight_codebook': torch.linspace(-7, 7, 16, device='xpu'),
        'comfy_quant': torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8),
    })
    model.requires_grad_(False)
    assert model._full_precision_mm is full_precision
    assert model._full_precision_mm_config is full_precision
    x = torch.randn((1, rows, width), dtype=torch.bfloat16, device='xpu')
    weight = model.weight
    if full_precision:
        expected = torch.nn.functional.linear(x, weight.dequantize())
    else:
        params = weight._params
        with monkeypatch.context() as context:
            context.setattr(int8, 'supports_w4a8_linear_prequantized', lambda: False)
            expected = ck.w4a8_int8_linear(x, weight._qdata, params.scale, params.s_channel,
                codebook=params.codebook, group_size=group, convrot_groupsize=rotation,
                out_dtype=x.dtype)
    original = int8.w4a8_linear_prequantized
    calls = []

    def observed(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(int8, 'w4a8_linear_prequantized', observed)
    actual = model(x)
    assert calls == ([(1, rows, width)] if not full_precision and rows <= 8 else [])
    assert torch.equal(actual, expected)


@pytest.fixture
def shared_qkv(comfy_bridge):
    path = PATH.with_name("qwen_image21_qkv.py")
    spec = importlib.util.spec_from_file_location("qkv_integration_test", path)
    qkv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qkv)
    return comfy_bridge, qkv


def qkv_inputs(ops, width=256, group=256, weight_device="xpu"):
    layers = [loaded_linear(ops, convrot=True, full_precision=False,
                            width=width, output=output, group=group)
              for output in (96, 128, 64)]
    if weight_device != "xpu":
        for layer in layers:
            layer.to(weight_device)
    module = SimpleNamespace(**dict(zip(("to_q", "to_k", "to_v"), layers)))
    x = torch.randn((2, 5, width), dtype=torch.bfloat16, device="xpu")
    return module, x, layers


@pytest.mark.parametrize("width,group", [(128, 4), (128, 16), (128, 64),
                                       (1024, 64), (1024, 256), (1024, 1024)])
@pytest.mark.parametrize("weight_device", ["xpu", "cpu"])
@torch.inference_mode()
def test_shared_qkv_actual_comfy_projections_and_groups(shared_qkv, width, group, weight_device):
    ops, qkv = shared_qkv
    module, x, layers = qkv_inputs(ops, width, group, weight_device)
    expected = tuple(layer(x) for layer in layers)
    actual = qkv.project_qkv(module, x)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert qkv.get_stats()["shared_calls"] == 1


@pytest.mark.parametrize("kind", ["subclass", "class_patch", "helper_override"])
@torch.inference_mode()
def test_shared_qkv_preserves_real_class_overrides(shared_qkv, monkeypatch, kind):
    ops, qkv = shared_qkv
    module, x, layers = qkv_inputs(ops)
    layer = layers[1]
    name = "_forward" if kind == "helper_override" else "forward"
    original = getattr(type(layer), name)
    calls = []

    def custom(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs) + 1

    if kind == "class_patch":
        monkeypatch.setattr(type(layer), name, custom)
    else:
        layer.__class__ = type("CustomizedLinear", (type(layer),), {name: custom})
    expected = tuple(item(x) for item in layers)
    calls.clear()
    actual = qkv.project_qkv(module, x)
    assert calls
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert qkv.get_stats()["fallback:module_contract"] == 1
    assert not qkv.get_stats().get("shared_calls")


@pytest.mark.parametrize("policy", ["override", "disable", "priority", "constraint", "mixed_constraint"])
@torch.inference_mode()
def test_shared_qkv_respects_real_backend_registry(shared_qkv, monkeypatch, policy):
    from contextlib import nullcontext
    from comfy_kitchen.registry import registry
    from omni_xpu_kernel import int8
    ops, qkv = shared_qkv
    module, x, layers = qkv_inputs(ops)
    # Snapshot mutable registry configuration and restore it through monkeypatch.
    if policy == "disable":
        monkeypatch.setattr(registry, "_disabled", registry._disabled | {"xpu"})
    elif policy == "priority":
        monkeypatch.setattr(registry, "_priority", ["eager", "xpu"])
    elif policy in ("constraint", "mixed_constraint"):
        original = registry.validate_backend_for_call

        def validate(backend, name, kwargs):
            if (backend == "xpu" and name == "int8_linear"
                    and (policy == "constraint" or kwargs["weight"].shape[0] == 128)):
                from comfy_kitchen.constraints import ValidationResult
                return ValidationResult.fail("weight", "test constraint")
            return original(backend, name, kwargs)

        monkeypatch.setattr(registry, "validate_backend_for_call", validate)
    context = registry.use_backend("eager") if policy == "override" else nullcontext()
    with context:
        expected = tuple(layer(x) for layer in layers)
        native_calls = []
        original_native = int8.int8_linear_prequantized

        def observed(*args, **kwargs):
            assert policy == "mixed_constraint", "backend choice bypassed"
            native_calls.append(args[2].shape[0])
            return original_native(*args, **kwargs)

        monkeypatch.setattr(int8, "int8_linear_prequantized", observed)
        actual = qkv.project_qkv(module, x)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert bool(qkv.get_stats().get("shared_calls")) == (policy == "mixed_constraint")
    if policy == "mixed_constraint":
        assert native_calls == [96, 64]
        assert qkv.get_stats()["fallback:backend_projection"] == 1
    elif policy == "constraint":
        assert qkv.get_stats()["fallback:backend_projection"] == 3
    else:
        assert qkv.get_stats()["fallback:backend"] == 1
