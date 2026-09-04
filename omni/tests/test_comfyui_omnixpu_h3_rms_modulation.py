"""Portable contracts for the H3 segmented RMS modulation adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"
_ADAPTERS = _PLUGIN / "adapters"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")
    return _load_module(
        f"{adapters.__name__}.h3_rms_modulation",
        _ADAPTERS / "h3_rms_modulation.py",
    )


def _packed_modulation():
    packed = torch.randn(3, 6 * 5376, dtype=torch.bfloat16)
    return packed.chunk(6, dim=-1)


def _fake_h3_module(state):
    module = types.ModuleType("comfy.ldm.minimax.model")
    module.__dict__.update({"state": state, "torch": torch})
    exec(
        """
def _mod_scale_shift(h, shift, scale, segments):
    for start, stop, row in segments:
        h[start:stop].mul_(1.0 + scale[row].to(h.dtype)).add_(
            shift[row].to(h.dtype)
        )
    return h


def _mod_gate(x, gate, other, segments):
    for start, stop, row in segments:
        x[start:stop].addcmul_(other[start:stop], gate[row].to(x.dtype))
    return x


class FakeNorm:
    def __init__(self, weight):
        self.weight = weight
        self.normalized_shape = (weight.numel(),)
        self.eps = 1e-6

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        state["fallback_norm_calls"] += 1
        return torch.nn.functional.rms_norm(
            x, self.normalized_shape, self.weight, self.eps
        )


class DiTBlock:
    def __init__(self, weight, modulation):
        self.norm1 = FakeNorm(weight)
        self.norm2 = FakeNorm(weight)
        self.modulation = modulation

    def adaln_proj(self, _t_emb):
        return self.modulation

    def attn(self, h, rope_freqs=None, transformer_options={}):
        return h * 0.25

    def mlp(self, h):
        return h * 0.5

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        state["forward_calls"] += 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
""",
        module.__dict__,
    )
    return module


def _install_fake_runtime(monkeypatch, adapter, *, policy_supported):
    state = {
        "fallback_norm_calls": 0,
        "forward_calls": 0,
        "native_calls": 0,
    }
    h3_model = _fake_h3_module(state)

    def native_fused(weight, value, scale, shift, segments, eps):
        state["native_calls"] += 1
        output = torch.nn.functional.rms_norm(
            value, (value.shape[-1],), weight, eps
        )
        return h3_model._mod_scale_shift(
            output, shift, scale, segments
        )

    candidate = SimpleNamespace(
        supports_rms_norm_segmented_modulation=lambda: True,
        rms_norm_segmented_modulation_supported=lambda _value: (
            policy_supported
        ),
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
        def __init__(self, layer, _value, *, offloadable):
            assert offloadable is True
            self.layer = layer

        def __enter__(self):
            return self.layer.weight, None

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

    adapter._EXPECTED_FORWARD_SHA256 = "fake-forward"
    adapter._EXPECTED_MODULATE_SHA256 = "fake-modulate"

    def fake_source_hash(function):
        if function is h3_model.DiTBlock.forward:
            return "fake-forward"
        if function is h3_model._mod_scale_shift:
            return "fake-modulate"
        raise AssertionError("unexpected source guard target")

    monkeypatch.setattr(adapter, "_source_sha256", fake_source_hash)
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
    assert adapter._segment_reason([(0, 17, torch.tensor(0))], 17, 3) == (
        "segment_value_type"
    )
    assert adapter._segment_reason([(1, 17, 0)], 17, 3) == (
        "segment_coverage"
    )
    assert adapter._segment_reason([(0, 17, 3)], 17, 3) == (
        "segment_modulation_row"
    )


def test_modulation_requires_the_real_packed_chunk_layout(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    x = torch.randn(17, 5376, dtype=torch.bfloat16)
    segments = [(0, 3, 0), (3, 8, 1), (8, 17, 2)]
    shift, scale, *_unused = _packed_modulation()

    assert shift.stride() == (6 * 5376, 1)
    assert adapter._modulation_reason(shift, scale, x, segments) == ""
    assert adapter._modulation_reason(
        shift.contiguous(), scale.contiguous(), x, segments
    ) == "modulation_layout"


def test_adapter_is_source_guarded_semantic_and_registered_after_generic_norm(
    monkeypatch,
):
    adapter = _load_adapter(monkeypatch)
    source = (_ADAPTERS / "h3_rms_modulation.py").read_text(encoding="utf-8")
    registry = (_PATCHES / "__init__.py").read_text(encoding="utf-8")

    assert len(adapter._EXPECTED_FORWARD_SHA256) == 64
    assert len(adapter._EXPECTED_MODULATE_SHA256) == 64
    assert "physical_bmg_sku" not in source
    assert "rms_norm_modulate_b580" not in source
    assert "rms_norm_segmented_modulation_supported" in source
    assert "self.adaln_proj(" not in source
    assert "self.attn(" not in source
    assert "self.mlp(" not in source
    assert registry.index('"norm_adapter"') < registry.index(
        '"h3_rms_modulation_adapter"'
    )


@pytest.mark.parametrize("policy_supported", [False, True])
def test_patched_forward_matches_original_for_policy_and_fallback(
    monkeypatch, policy_supported
):
    adapter = _load_adapter(monkeypatch)
    h3_model, state = _install_fake_runtime(
        monkeypatch, adapter, policy_supported=policy_supported
    )
    torch.manual_seed(1234)
    weight = torch.randn(5376, dtype=torch.bfloat16)
    modulation = _packed_modulation()
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    value = torch.randn(7, 5376, dtype=torch.bfloat16)
    reference_block = h3_model.DiTBlock(weight, modulation)
    expected = reference_block.forward(
        value.clone(), None, segments, None
    )

    state.update(
        {"fallback_norm_calls": 0, "forward_calls": 0, "native_calls": 0}
    )
    applied, reason = adapter.apply()
    assert (applied, reason) == (True, "")
    assert adapter.apply() == (True, "already patched")

    block = h3_model.DiTBlock(weight, modulation)
    actual = block.forward(value.clone(), None, segments, None)

    assert torch.equal(actual, expected)
    assert state["forward_calls"] == 1
    assert state["native_calls"] == (2 if policy_supported else 0)
    assert state["fallback_norm_calls"] == (0 if policy_supported else 2)
    assert adapter.get_stats() == {
        "routed": 2 if policy_supported else 0,
        "fallback": 0,
        "reasons": {},
    }

    # The class-level norm hook is transparent outside the H3 forward context.
    direct = block.norm1(value.clone())
    assert isinstance(direct, torch.Tensor)
    assert state["fallback_norm_calls"] == (1 if policy_supported else 3)


def test_patched_context_contract_fallback_matches_original(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    h3_model, state = _install_fake_runtime(
        monkeypatch, adapter, policy_supported=True
    )
    torch.manual_seed(4321)
    weight = torch.randn(5376, dtype=torch.bfloat16)
    modulation = _packed_modulation()
    segments = [(0, 2, 0), (2, 4, 1), (4, 7, 2)]
    value = torch.randn(5376, 7, dtype=torch.bfloat16).transpose(0, 1)
    assert not value.is_contiguous()

    reference_block = h3_model.DiTBlock(weight, modulation)
    expected = reference_block.forward(
        value.clone(), None, segments, None
    )
    state.update(
        {"fallback_norm_calls": 0, "forward_calls": 0, "native_calls": 0}
    )
    assert adapter.apply() == (True, "")

    block = h3_model.DiTBlock(weight, modulation)
    actual = block.forward(value.clone(), None, segments, None)

    assert torch.equal(actual, expected)
    assert state == {
        "fallback_norm_calls": 2,
        "forward_calls": 1,
        "native_calls": 0,
    }
    assert adapter.get_stats() == {
        "routed": 0,
        "fallback": 2,
        "reasons": {"input_layout": 2},
    }


def test_adapter_retains_training_weight_and_failure_boundaries(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    source = (_ADAPTERS / "h3_rms_modulation.py").read_text(encoding="utf-8")

    assert adapter._MAX_SEGMENTS == 8
    assert "comfy.model_management.in_training" in source
    assert "CastBiasWeightContext" in source
    assert "run_every_op()" in source
    assert "sequence ==" not in source
    assert "sequence_length ==" not in source
