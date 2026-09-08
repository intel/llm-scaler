"""Metadata-only H3 adapter registration and call-routing contracts.

All fixture tensors live on the meta device. Native, norm, attention and MLP
results are opaque: these tests do not emulate CPU numerical references or
validate XPU correctness/performance. Cast/offload and physical-device behavior
must be checked separately in the real installed ComfyUI/XPU environment.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import linecache
import os
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"
_ADAPTERS = _PLUGIN / "adapters"


def _load_module(name: str, path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_adapter(monkeypatch):
    package_name = "omnixpu_h3_rms_modulation_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    adapters = types.ModuleType(f"{package_name}.adapters")
    adapters.__path__ = [str(_ADAPTERS)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py", monkeypatch)
    return _load_module(
        f"{adapters.__name__}.h3_rms_modulation",
        _ADAPTERS / "h3_rms_modulation.py",
        monkeypatch,
    )


def _metadata(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _opaque_output(value):
    assert value.is_meta, "routing stubs must not execute on real tensor data"
    return torch.empty_like(value)


def _packed_modulation(dtype=torch.bfloat16):
    packed = _metadata(3, 6 * 5376, dtype=dtype)
    return packed.chunk(6, dim=-1)


def _routing_h3_module(monkeypatch, state):
    module = types.ModuleType("comfy.ldm.minimax.model")
    module.__dict__.update(
        {"state": state, "torch": torch, "opaque_output": _opaque_output}
    )
    source = """
def _mod_scale_shift(h, shift, scale, segments):
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x, gate, other, segments):
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


class FakeNorm:
    def __init__(self, weight):
        assert weight.is_meta
        self.weight = weight
        self.bias = None
        self.normalized_shape = (weight.numel(),)
        self.eps = 1e-5

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        state["fallback_norm_calls"] += 1
        return opaque_output(x)


class DiTBlock:
    def __init__(self, weight, modulation):
        self.norm1 = FakeNorm(weight)
        self.norm2 = FakeNorm(weight)
        self.modulation = modulation

    def adaln_proj(self, _t_emb):
        state["projection_calls"] += 1
        return self.modulation

    def attn(self, h, rope_freqs=None, transformer_options={}):
        return opaque_output(h)

    def mlp(self, h):
        return opaque_output(h)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
"""
    # Source is retained only for the installed-source AST parity test. Runtime
    # registration must not depend on these fixture source bytes being present.
    filename = str(Path(__file__)) + ":h3-routing-fixture"
    monkeypatch.setitem(
        linecache.cache, filename,
        (len(source), None, source.splitlines(keepends=True), filename),
    )
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def _install_routing_stubs(monkeypatch, adapter, *, policy_supported):
    """Metadata interfaces only; policy_supported is an injected branch choice."""
    state = {
        "fallback_norm_calls": 0,
        "projection_calls": 0,
        "native_calls": 0,
    }
    h3_model = _routing_h3_module(monkeypatch, state)

    def native_fused(weight, value, scale, shift, segments, eps):
        state["native_calls"] += 1
        assert all(tensor.is_meta for tensor in (weight, value, scale, shift))
        assert weight.dtype == value.dtype == scale.dtype == shift.dtype
        assert eps == 1e-5
        # No RMS numerical emulation and no coupling to the eager helper.
        return _opaque_output(value)

    def policy_stub(value):
        assert value.is_meta, "this stub is not a physical-device classifier"
        return policy_supported

    candidate = SimpleNamespace(
        supports_rms_norm_segmented_modulation=lambda: True,
        rms_norm_segmented_modulation_supported=policy_stub,
        rms_norm_segmented_modulation=native_fused,
    )
    omni_package = types.ModuleType("omni_xpu_kernel")
    omni_package.norm = candidate

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.in_training = False
    ops = types.ModuleType("comfy.ops")
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    minimax = types.ModuleType("comfy.ldm.minimax")
    minimax.__path__ = []

    class CastBiasWeightContext:
        """Metadata cast only; no DynamicVRAM/residency/offload simulation."""

        def __init__(self, layer, value, *, offloadable):
            assert offloadable is True
            assert value.is_meta
            assert layer.bias is None, "this RMSNorm fixture has no bias"
            self.weight = layer.weight.to(device=value.device, dtype=value.dtype)

        def __enter__(self):
            return self.weight, None

        def __exit__(self, *_exc):
            return False

    ops.CastBiasWeightContext = CastBiasWeightContext
    ops.run_every_op = lambda: None
    comfy.model_management = model_management
    comfy.ops = ops
    comfy.ldm = ldm
    ldm.minimax = minimax
    minimax.model = h3_model

    for name, module in (
        ("omni_xpu_kernel", omni_package),
        ("comfy", comfy),
        ("comfy.model_management", model_management),
        ("comfy.ops", ops),
        ("comfy.ldm", ldm),
        ("comfy.ldm.minimax", minimax),
        ("comfy.ldm.minimax.model", h3_model),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return h3_model, state


def test_segment_contract_is_structural_not_sequence_specific(monkeypatch):
    adapter = _load_adapter(monkeypatch)

    assert adapter._segment_reason(
        [(0, 388, 0), (388, 802, 1), (802, 15787, 2)],
        15787,
        3,
    ) == ""
    assert adapter._segment_reason(
        [(0, 388, 0), (388, 802, 1), (802, 44929, 2)],
        44929,
        3,
    ) == ""
    assert adapter._segment_reason(
        [(0, 17, _metadata(dtype=torch.int64))], 17, 3
    ) == (
        "segment_value_type"
    )
    assert adapter._segment_reason([(1, 17, 0)], 17, 3) == (
        "segment_coverage"
    )
    assert adapter._segment_reason([(0, 17, 3)], 17, 3) == (
        "segment_modulation_row"
    )


@pytest.mark.parametrize("fallback", [False, True])
def test_compile_diagnostics_do_not_specialize_on_eager_counters(
    monkeypatch, fallback,
):
    adapter = _load_adapter(monkeypatch)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    class Layer:
        def __call__(self, value):
            return value.clone()

    def fused(_ops, _layer, value, _scale, _shift, _segments):
        return value.clone(), ""

    def modulate(value, _shift, _scale, _segments):
        return value.clone()

    monkeypatch.setattr(adapter, "_run_fused", fused)
    layer = Layer()

    def invoke(value):
        return adapter._rms_modulation(
            None, modulate, layer, value, value, value, [],
            "input_layout" if fallback else "",
        )[0]

    torch._dynamo.reset()
    try:
        compiled = torch.compile(invoke, backend=backend, fullgraph=True)
        value = _metadata(7, 5376)
        for _ in range(12):
            actual = compiled(value)
            assert actual.is_meta and actual.shape == value.shape
            # A separate eager call changes counters between compiled calls.
            # It must not invalidate a compiled graph's guards.
            invoke(value)
        assert len(graphs) == 1
        assert adapter.get_stats() == {
            "routed": 0 if fallback else 12,
            "fallback": 12 if fallback else 0,
            "reasons": {"input_layout": 12} if fallback else {},
        }
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("modulation_dtype", [torch.bfloat16, torch.float32])
def test_modulation_requires_the_real_packed_six_chunk_layout(
    monkeypatch, modulation_dtype
):
    adapter = _load_adapter(monkeypatch)
    x = _metadata(17, 5376, dtype=torch.bfloat16)
    segments = [(0, 3, 0), (3, 8, 1), (8, 17, 2)]
    values = _packed_modulation(modulation_dtype)

    assert values[0].stride() == (6 * 5376, 1)
    assert adapter._modulation_reason(values, x, segments) == ""
    contiguous = tuple(value.contiguous() for value in values)
    assert adapter._modulation_reason(contiguous, x, segments) == (
        "modulation_layout"
    )


def test_adapter_has_one_api_guarded_model_patch_and_no_lazy_bridge(
    monkeypatch,
):
    adapter = _load_adapter(monkeypatch)
    source = (_ADAPTERS / "h3_rms_modulation.py").read_text(encoding="utf-8")
    registry = (_PATCHES / "__init__.py").read_text(encoding="utf-8")

    assert "getsource" not in source
    assert "sha256" not in source
    assert "inspect.signature" in source
    assert "physical_bmg_sku" not in source
    assert "rms_norm_modulate_b580" not in source
    assert "rms_norm_segmented_modulation_supported" in source
    assert "ContextVar" not in source
    assert "_DeferredRmsNorm" not in source
    assert "layer_type.forward" not in source
    assert "h3_model._mod_scale_shift =" not in source
    assert source.count(".forward =") == 1
    assert registry.index('"norm_adapter"') < registry.index(
        '"h3_rms_modulation_adapter"'
    )


@pytest.mark.parametrize("policy_supported", [False, True])
@pytest.mark.parametrize("modulation_dtype", [torch.bfloat16, torch.float32])
def test_dispatch_uses_the_selected_interface_without_numerical_claims(
    monkeypatch, policy_supported, modulation_dtype
):
    adapter = _load_adapter(monkeypatch)
    h3_model, state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=policy_supported
    )
    weight = _metadata(5376, dtype=torch.bfloat16)
    modulation = _packed_modulation(modulation_dtype)
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    value = _metadata(7, 5376, dtype=torch.bfloat16)
    reference_block = h3_model.DiTBlock(weight, modulation)
    expected = reference_block.forward(
        value.clone(), None, segments, None
    )

    state.update(
        {"fallback_norm_calls": 0, "projection_calls": 0, "native_calls": 0}
    )
    original_modulate = h3_model._mod_scale_shift
    original_norm_forward = h3_model.FakeNorm.forward
    applied, reason = adapter.apply()
    assert (applied, reason) == (True, "")
    assert adapter.apply() == (True, "already patched")
    assert h3_model._mod_scale_shift is original_modulate
    assert h3_model.FakeNorm.forward is original_norm_forward

    block = h3_model.DiTBlock(weight, modulation)
    actual = block.forward(value.clone(), None, segments, None)

    assert actual.is_meta and expected.is_meta
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert state["projection_calls"] == 1
    assert state["native_calls"] == (2 if policy_supported else 0)
    assert state["fallback_norm_calls"] == (0 if policy_supported else 2)
    assert adapter.get_stats() == {
        "routed": 2 if policy_supported else 0,
        "fallback": 0,
        "reasons": {},
    }

    # The adapter never changes the norm class or its direct-call behavior.
    direct = block.norm1(value.clone())
    assert isinstance(direct, torch.Tensor)
    assert state["fallback_norm_calls"] == (1 if policy_supported else 3)


def test_unsupported_input_layout_uses_original_interface(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    h3_model, state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=True
    )
    weight = _metadata(5376, dtype=torch.bfloat16)
    modulation = _packed_modulation()
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    value = _metadata(5376, 7, dtype=torch.bfloat16).transpose(0, 1)
    assert not value.is_contiguous()

    reference_block = h3_model.DiTBlock(weight, modulation)
    expected = reference_block.forward(
        value.clone(), None, segments, None
    )
    state.update(
        {"fallback_norm_calls": 0, "projection_calls": 0, "native_calls": 0}
    )
    assert adapter.apply() == (True, "")

    block = h3_model.DiTBlock(weight, modulation)
    actual = block.forward(value.clone(), None, segments, None)

    assert actual.is_meta and expected.is_meta
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert state == {
        "fallback_norm_calls": 2,
        "projection_calls": 1,
        "native_calls": 0,
    }
    assert adapter.get_stats() == {
        "routed": 0,
        "fallback": 1,
        "reasons": {"input_layout": 1},
    }


@pytest.mark.parametrize("signature", [
    "self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, extra=None",
    "self, x, t_emb, mod_segments, rope_freqs, *, transformer_options={}",
    "self, x, t_emb, mod_segments, rope_freqs, transformer_options=None",
    "self, x, t_emb, mod_segments, rope_freqs=None, transformer_options={}",
    "self, x, t_emb, segments, rope_freqs, transformer_options={}",
    "self, *args, **kwargs",
])
def test_incompatible_forward_api_leaves_every_model_entry_point_untouched(
    monkeypatch, signature,
):
    adapter = _load_adapter(monkeypatch)
    h3_model, _state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=True
    )
    namespace = {}
    exec(f"def forward({signature}):\n    pass\n", namespace)
    monkeypatch.setattr(h3_model.DiTBlock, "forward", namespace["forward"])
    original_forward = h3_model.DiTBlock.forward
    original_modulate = h3_model._mod_scale_shift
    original_norm_forward = h3_model.FakeNorm.forward
    assert adapter.apply() == (
        False,
        "MiniMax H3 DiTBlock.forward API is unsupported",
    )
    assert h3_model.DiTBlock.forward is original_forward
    assert h3_model._mod_scale_shift is original_modulate
    assert h3_model.FakeNorm.forward is original_norm_forward


def test_registration_does_not_require_forward_source(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    model, _state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=True
    )
    original = model.DiTBlock.forward
    monkeypatch.delitem(linecache.cache, original.__code__.co_filename)

    def unavailable_source(*_args, **_kwargs):
        raise OSError("source file is unavailable")

    monkeypatch.setattr(inspect, "getsource", unavailable_source)
    assert adapter.apply() == (True, "")
    installed = model.DiTBlock.forward
    assert getattr(installed, adapter._PATCH_MARKER) is original
    assert inspect.signature(installed) == inspect.signature(original)
    assert adapter.apply() == (True, "already patched")
    assert model.DiTBlock.forward is installed


def test_unavailable_forward_signature_leaves_model_untouched(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    model, _state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=True
    )
    original = model.DiTBlock.forward

    def unavailable_signature(*_args, **_kwargs):
        raise ValueError("callable signature is unavailable")

    monkeypatch.setattr(inspect, "signature", unavailable_signature)
    assert adapter.apply() == (
        False, "MiniMax H3 DiTBlock.forward signature is unavailable",
    )
    assert model.DiTBlock.forward is original


@pytest.mark.parametrize("module_name, attribute, expected_reason", [
    ("omni_xpu_kernel", "supports_rms_norm_segmented_modulation",
     "native segmented RMS modulation is unavailable"),
    ("omni_xpu_kernel", "rms_norm_segmented_modulation_supported",
     "native segmented RMS modulation policy is unavailable"),
    ("omni_xpu_kernel", "rms_norm_segmented_modulation",
     "native segmented RMS modulation operation is unavailable"),
    ("comfy.ops", "CastBiasWeightContext",
     "ComfyUI cast/offload helpers are unavailable"),
    ("comfy.ops", "run_every_op",
     "ComfyUI cast/offload helpers are unavailable"),
    ("comfy.ldm.minimax.model", "_mod_scale_shift",
     "MiniMax H3 modulation helpers are unavailable"),
    ("comfy.ldm.minimax.model", "_mod_gate",
     "MiniMax H3 modulation helpers are unavailable"),
])
def test_missing_callable_leaves_model_entry_points_untouched(
    monkeypatch, module_name, attribute, expected_reason,
):
    adapter = _load_adapter(monkeypatch)
    model, _state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=True
    )
    module = sys.modules[module_name]
    if module_name == "omni_xpu_kernel":
        module = module.norm
    monkeypatch.setattr(module, attribute, None)
    entry_points = (
        model.DiTBlock.forward, model._mod_scale_shift,
        model._mod_gate, model.FakeNorm.forward,
    )

    assert adapter.apply() == (False, expected_reason)
    assert entry_points == (
        model.DiTBlock.forward, model._mod_scale_shift,
        model._mod_gate, model.FakeNorm.forward,
    )


@pytest.mark.parametrize("fallback", ["none", "modulation", "norm1", "norm2", "both"])
@pytest.mark.parametrize("modulation_dtype", [torch.bfloat16, torch.float32])
def test_local_fallback_preserves_single_projection_and_residual_order(
    monkeypatch, fallback, modulation_dtype,
):
    adapter = _load_adapter(monkeypatch)
    model, state = _install_routing_stubs(monkeypatch, adapter, policy_supported=True)
    modulation = _packed_modulation(modulation_dtype)
    if fallback == "modulation":
        modulation = tuple(value.contiguous() for value in modulation)
    block = model.DiTBlock(_metadata(5376), modulation)
    value = _metadata(7, 5376)
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    events, arguments, kernel_events = [], {}, []

    def trace(target, name, label):
        original = getattr(target, name)

        def traced(*args, **kwargs):
            events.append(label)
            arguments[label] = (args, kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, traced)

    for name in ("adaln_proj", "attn", "mlp"):
        trace(block, name, name)
    for name in ("norm1", "norm2"):
        trace(getattr(block, name), "forward", name)
    for name in ("_mod_scale_shift", "_mod_gate"):
        trace(model, name, name)
    trace(sys.modules["omni_xpu_kernel"].norm, "rms_norm_segmented_modulation", "native")

    ops = sys.modules["comfy.ops"]
    original_context = ops.CastBiasWeightContext

    class MetadataContext(original_context):
        # Inject only cast-result metadata. This checks adapter control flow,
        # not physical residency, DynamicVRAM or real weight-hook behavior.
        def __init__(self, layer, x, *, offloadable):
            super().__init__(layer, x, offloadable=offloadable)
            self.label = "norm1" if layer is block.norm1 else "norm2"

        def __enter__(self):
            events.append(f"cast:{self.label}")
            weight, bias = super().__enter__()
            if fallback in (self.label, "both"):
                weight = _metadata(2 * 5376)[::2]
            return weight, bias

        def __exit__(self, *exc):
            events.append(f"release:{self.label}")
            return super().__exit__(*exc)

    monkeypatch.setattr(ops, "CastBiasWeightContext", MetadataContext)
    monkeypatch.setattr(
        adapter, "log_debug_event",
        lambda *args, **kwargs: kernel_events.append((args, kwargs)),
    )
    assert adapter.apply() == (True, "")
    t_emb, rope, options = object(), object(), {"test_option": object()}
    output = block.forward(value, t_emb, segments, rope, transformer_options=options)

    expected = ["adaln_proj"]
    fused_calls = 0
    for name, operation in (("norm1", "attn"), ("norm2", "mlp")):
        if fallback != "modulation":
            expected.append(f"cast:{name}")
            if fallback not in (name, "both"):
                expected.append("native")
                fused_calls += 1
            expected.append(f"release:{name}")
        if fallback in ("modulation", name, "both"):
            expected.extend((name, "_mod_scale_shift"))
        expected.extend((operation, "_mod_gate"))
    assert events == expected
    assert output is value  # The original gate updates the residual in place.
    assert arguments["adaln_proj"][0] == (t_emb,)
    assert arguments["attn"][1]["rope_freqs"] is rope
    assert arguments["attn"][1]["transformer_options"] is options
    assert state == {
        "projection_calls": 1,
        "native_calls": fused_calls,
        "fallback_norm_calls": 2 - fused_calls,
    }
    reason = "modulation_layout" if fallback == "modulation" else "norm_weight_layout"
    assert adapter.get_stats() == {
        "routed": fused_calls,
        "fallback": 2 - fused_calls,
        "reasons": {reason: 2 - fused_calls} if fused_calls < 2 else {},
    }
    assert len(kernel_events) == bool(fused_calls)
    if fused_calls:
        assert kernel_events[0][1]["details"]["fused_calls"] == fused_calls


@pytest.mark.parametrize("hook_attribute", ["weight_function", "bias_function"])
@pytest.mark.parametrize("hooked_norm", ["norm1", "norm2"])
def test_hook_fallback_keeps_original_norm_cast_and_hook_once(
    monkeypatch, hook_attribute, hooked_norm,
):
    adapter = _load_adapter(monkeypatch)
    model, state = _install_routing_stubs(monkeypatch, adapter, policy_supported=True)
    block = model.DiTBlock(_metadata(5376), _packed_modulation())
    value = _metadata(7, 5376)
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    layer = getattr(block, hooked_norm)
    events = []

    def hook(weight):
        assert weight.is_meta
        events.append("hook")
        return weight

    setattr(layer, hook_attribute, [hook])
    if hook_attribute == "bias_function":
        layer.bias = _metadata(5376)
    ops = sys.modules["comfy.ops"]
    original_context = ops.CastBiasWeightContext

    class TracedContext(original_context):
        def __init__(self, selected, x, *, offloadable):
            self.selected = selected
            if selected is not layer:
                super().__init__(selected, x, offloadable=offloadable)
                return
            events.append("cast")
            self.weight = selected.weight.to(dtype=x.dtype)
            self.bias = selected.bias.to(dtype=x.dtype) if selected.bias is not None else None
            # Opaque metadata hooks follow their own weight/bias slot. Physical
            # casts, hook arithmetic and offload are checked by the XPU gate.
            if self.bias is not None:
                for function in getattr(selected, "bias_function", ()):
                    self.bias = function(self.bias)
            for function in getattr(selected, "weight_function", ()):
                self.weight = function(self.weight)

        def __enter__(self):
            if self.selected is layer:
                return self.weight, self.bias
            return super().__enter__()

        def __exit__(self, *args):
            if self.selected is layer:
                events.append("release")
            return super().__exit__(*args)

    original_forward = layer.forward

    def norm_forward(x):
        events.append("norm")
        with ops.CastBiasWeightContext(layer, x, offloadable=True):
            output = original_forward(x)
        return output

    monkeypatch.setattr(ops, "CastBiasWeightContext", TracedContext)
    monkeypatch.setattr(layer, "forward", norm_forward)
    original_modulate = model._mod_scale_shift

    def modulate(*args):
        events.append("modulation")
        return original_modulate(*args)

    monkeypatch.setattr(model, "_mod_scale_shift", modulate)
    assert adapter.apply() == (True, "")
    output = block.forward(value, None, segments, None)
    assert output is value
    assert events == ["norm", "cast", "hook", "release", "modulation"]
    assert state == {"projection_calls": 1, "native_calls": 1, "fallback_norm_calls": 1}
    assert adapter.get_stats() == {
        "routed": 1, "fallback": 1, "reasons": {"norm_" + hook_attribute: 1},
    }


@pytest.mark.parametrize("failure_call", [1, 2])
def test_native_error_propagates_without_replaying_model(monkeypatch, failure_call):
    adapter = _load_adapter(monkeypatch)
    model, state = _install_routing_stubs(monkeypatch, adapter, policy_supported=True)
    candidate = sys.modules["omni_xpu_kernel"].norm
    original = candidate.rms_norm_segmented_modulation
    calls = 0

    def native_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("native execution error")
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate, "rms_norm_segmented_modulation", native_error)
    assert adapter.apply() == (True, "")
    block = model.DiTBlock(_metadata(5376), _packed_modulation())
    with pytest.raises(RuntimeError, match="native execution error"):
        block.forward(
            _metadata(7, 5376), None,
            [(0, 2, 0), (2, 4, 1), (4, 7, 2)], None,
        )
    assert state["projection_calls"] == 1
    assert state["fallback_norm_calls"] == 0
    assert adapter.get_stats()["routed"] == failure_call - 1
    assert adapter.get_stats()["fallback"] == 0


def test_training_falls_back_before_native_routing(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    h3_model, state = _install_routing_stubs(
        monkeypatch, adapter, policy_supported=True
    )
    weight = _metadata(5376, dtype=torch.bfloat16)
    modulation = _packed_modulation()
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    value = _metadata(7, 5376, dtype=torch.bfloat16)
    assert adapter.apply() == (True, "")

    sys.modules["comfy.model_management"].in_training = True
    block = h3_model.DiTBlock(weight, modulation)
    block.forward(value.clone(), None, segments, None)
    assert state == {
        "fallback_norm_calls": 2,
        "projection_calls": 1,
        "native_calls": 0,
    }
    assert adapter.get_stats() == {
        "routed": 0,
        "fallback": 1,
        "reasons": {"training": 1},
    }


def test_adapter_keeps_runtime_weight_hook_and_structural_routing(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    source = (_ADAPTERS / "h3_rms_modulation.py").read_text(encoding="utf-8")

    assert adapter._MAX_SEGMENTS == 8
    assert "comfy.model_management.in_training" in source
    assert "CastBiasWeightContext" in source
    assert "run_every_op()" in source
    assert "sequence ==" not in source
    assert "sequence_length ==" not in source


def test_fixture_preserves_bias_free_norm_and_explicit_model_epsilon(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    model, _state = _install_routing_stubs(monkeypatch, adapter, policy_supported=True)
    block = model.DiTBlock(_metadata(5376), _packed_modulation())
    value = _metadata(7, 5376)
    ops = sys.modules["comfy.ops"]
    for layer in (block.norm1, block.norm2):
        assert layer.bias is None
        assert layer.eps == 1e-5
        layer.weight = _metadata(5376, dtype=torch.float32)
        with ops.CastBiasWeightContext(layer, value, offloadable=True) as (weight, bias):
            assert weight.is_meta and weight.dtype == value.dtype
            assert bias is None


def test_native_spy_does_not_compute_cpu_rms_or_call_eager_modulation(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    model, state = _install_routing_stubs(monkeypatch, adapter, policy_supported=True)

    def forbidden_reference(*_args, **_kwargs):
        raise AssertionError("a routing spy must not emulate native numerical behavior")

    monkeypatch.setattr(torch.nn.functional, "rms_norm", forbidden_reference)
    monkeypatch.setattr(model, "_mod_scale_shift", forbidden_reference)
    # Exercise the spy itself. Changing an eager helper here must not require
    # the production adapter to accept modified model semantics.
    shift, scale, *_rest = _packed_modulation()
    output = sys.modules["omni_xpu_kernel"].norm.rms_norm_segmented_modulation(
        _metadata(5376), _metadata(7, 5376), scale, shift,
        [(0, 2, 0), (2, 4, 1), (4, 7, 2)], eps=1e-5,
    )
    assert output.is_meta
    assert state == {
        "fallback_norm_calls": 0,
        "projection_calls": 0,
        "native_calls": 1,
    }


def test_fixture_call_boundaries_match_installed_comfy_source(monkeypatch):
    # Read source only: importing ComfyUI here would initialize a device/runtime
    # and would no longer be a portable metadata-only test.
    root = Path(os.environ.get("COMFYUI_ROOT", "/llm/ComfyUI"))
    source_path = root / "comfy" / "ldm" / "minimax" / "model.py"
    if not source_path.is_file():
        pytest.skip("installed ComfyUI source unavailable; fixture parity not checked")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level = {node.name: node for node in tree.body if hasattr(node, "name")}
    forward = next(
        node for node in top_level["DiTBlock"].body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    adapter = _load_adapter(monkeypatch)
    model, _state = _install_routing_stubs(monkeypatch, adapter, policy_supported=True)
    for real_node, fixture in (
        (forward, model.DiTBlock.forward),
        (top_level["_mod_scale_shift"], model._mod_scale_shift),
        (top_level["_mod_gate"], model._mod_gate),
    ):
        fixture_node = ast.parse(textwrap.dedent(inspect.getsource(fixture))).body[0]
        assert ast.dump(fixture_node) == ast.dump(real_node)
    constructor = next(
        node for node in top_level["MiniMaxH3Model"].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    defaults = dict(zip(
        [arg.arg for arg in constructor.args.args][-len(constructor.args.defaults):],
        constructor.args.defaults,
    ))
    assert ast.literal_eval(defaults["norm_eps"]) == 1e-5
