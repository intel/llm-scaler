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


def loaded_linear(ops, *, convrot, full_precision):
    from comfy_kitchen.tensor import QuantizedTensor
    weight = torch.randn((96, 256), dtype=torch.bfloat16, device='xpu')
    quantized = QuantizedTensor.from_float(weight, 'TensorWiseINT8Layout',
        per_channel=convrot, convrot=convrot, convrot_groupsize=256)
    klass = ops.mixed_precision_ops({}, torch.bfloat16,
        disabled=ops.get_disabled_quant_formats(torch.device('xpu')))
    model = klass.Linear(256, 96, bias=False, device='xpu')
    configuration = {'format': 'int8_tensorwise', 'convrot': convrot,
        'convrot_groupsize': 256, 'full_precision_matrix_mult': full_precision}
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
