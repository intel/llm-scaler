"""Portable control-flow tests for the PTL Z-Image RMS/gate route."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Candidate:
    def __init__(self):
        self.calls = []

    def _get_native(self):
        return types.SimpleNamespace(rms_norm_gate_residual=lambda: None)

    def rms_norm_gate_residual(
        self, weight, value, gate, residual, eps
    ):
        self.calls.append(
            (weight.shape, value.shape, gate.shape, residual.shape, eps)
        )
        return residual.clone()


def _load_patch(monkeypatch, *, target="ptl-h", torch_version="2.11.0+xpu"):
    monkeypatch.setattr(torch, "__version__", torch_version)
    package = types.ModuleType("omnixpu_zimage_rms_test")
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package.__name__}.patches")
    patches.__path__ = [str(_PATCHES)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")

    candidate = _Candidate()
    probe = types.ModuleType("ComfyUI-OmniXPU.probe")
    probe.norm = candidate
    omni = types.ModuleType("omni_xpu_kernel")
    omni.__xpu_target__ = target
    monkeypatch.setitem(sys.modules, "ComfyUI-OmniXPU.probe", probe)
    monkeypatch.setitem(sys.modules, "omni_xpu_kernel", omni)

    class Norm:
        def __init__(self):
            self.weight = torch.ones(3840, dtype=torch.bfloat16)
            self.bias = None
            self.eps = 1e-5
            self.comfy_cast_weights = True
            self.weight_function = []
            self.bias_function = []

        def __call__(self, value):
            return value

    class Modulation:
        def __init__(self):
            self.linear = types.SimpleNamespace(in_features=256)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.linear

        def __call__(self, value):
            return torch.zeros(
                (value.shape[0], 4 * 3840), dtype=torch.bfloat16
            )

    class JointTransformerBlock:
        def __init__(self):
            self.dim = 3840
            self.modulation = True
            self.adaLN_modulation = Modulation()
            self.attention_norm1 = Norm()
            self.attention_norm2 = Norm()
            self.ffn_norm1 = Norm()
            self.ffn_norm2 = Norm()
            self.attention = lambda value, *args, **kwargs: value
            self.feed_forward = lambda value: value
            self.original_calls = 0

        def forward(self, x, *args, **kwargs):
            self.original_calls += 1
            return x + 1

    cast_events = []
    comfy_ops = types.ModuleType("comfy.ops")

    def cast_bias_weight(norm, value, offloadable=True):
        cast_events.append(("cast", offloadable))
        return norm.weight, None, "stream"

    def uncast_bias_weight(norm, weight, bias, stream):
        cast_events.append(("uncast", stream))

    comfy_ops.cast_bias_weight = cast_bias_weight
    comfy_ops.uncast_bias_weight = uncast_bias_weight

    model = types.ModuleType("comfy.ldm.lumina.model")
    model.JointTransformerBlock = JointTransformerBlock
    model.modulate = lambda value, scale: value
    model.clamp_fp16 = lambda value: value
    model.apply_gate = lambda gate, value: gate * value
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    lumina = types.ModuleType("comfy.ldm.lumina")
    lumina.__path__ = []
    comfy.ops = comfy_ops
    comfy.ldm = ldm
    ldm.lumina = lumina
    lumina.model = model
    for name, module in (
        ("comfy", comfy),
        ("comfy.ops", comfy_ops),
        ("comfy.ldm", ldm),
        ("comfy.ldm.lumina", lumina),
        ("comfy.ldm.lumina.model", model),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    patch = _load_module(
        f"{patches.__name__}.patch_zimage_rms_gate",
        _PATCHES / "patch_zimage_rms_gate.py",
    )
    return patch, candidate, JointTransformerBlock, cast_events


def test_eligible_block_fuses_both_norm_gate_boundaries(monkeypatch):
    patch, candidate, Block, cast_events = _load_patch(monkeypatch)
    assert patch.apply() == (True, "")
    monkeypatch.setattr(
        patch, "_route_input", lambda *args, **kwargs: (True, "")
    )

    block = Block()
    x = torch.zeros((1, 64, 3840), dtype=torch.bfloat16)
    adaln = torch.zeros((1, 256), dtype=torch.bfloat16)
    output = block.forward(x, None, None, adaln_input=adaln)

    torch.testing.assert_close(output, x)
    assert block.original_calls == 0
    assert len(candidate.calls) == 2
    assert all(call[1:] == ((64, 3840), (3840,), (64, 3840), 1e-5)
               for call in candidate.calls)
    assert cast_events == [
        ("cast", True),
        ("uncast", "stream"),
        ("cast", True),
        ("uncast", "stream"),
    ]
    assert patch.get_stats() == {"routed": 2, "fallback": 0, "reasons": {}}


def test_cpu_block_keeps_original_forward(monkeypatch):
    patch, candidate, Block, _events = _load_patch(monkeypatch)
    assert patch.apply() == (True, "")
    block = Block()
    x = torch.zeros((1, 64, 3840), dtype=torch.bfloat16)
    adaln = torch.zeros((1, 256), dtype=torch.bfloat16)
    torch.testing.assert_close(
        block.forward(x, None, None, adaln_input=adaln), x + 1
    )
    assert block.original_calls == 1
    assert candidate.calls == []
    assert patch.get_stats()["reasons"] == {"device": 1}


def test_unvalidated_environment_skips_patch(monkeypatch):
    patch, _candidate, _Block, _events = _load_patch(
        monkeypatch, target="bmg"
    )
    assert patch.apply() == (False, "validated only for PTL-H")

    patch, _candidate, _Block, _events = _load_patch(
        monkeypatch, torch_version="2.12.0+xpu"
    )
    assert patch.apply() == (False, "validated only for Torch 2.11")


def test_environment_switch_is_nested_under_norm(monkeypatch):
    config_path = _PLUGIN / "config.py"
    monkeypatch.setenv("OMNIXPU_ENABLE", "1")
    monkeypatch.setenv("OMNIXPU_NORM", "1")
    monkeypatch.setenv("OMNIXPU_ZIMAGE_RMS_GATE", "0")
    config = _load_module("omnixpu_zimage_rms_config_off", config_path)
    assert config.config.norm
    assert not config.config.zimage_rms_gate

    monkeypatch.setenv("OMNIXPU_NORM", "0")
    monkeypatch.setenv("OMNIXPU_ZIMAGE_RMS_GATE", "1")
    config = _load_module("omnixpu_zimage_rms_config_parent_off", config_path)
    assert not config.config.norm
    assert not config.config.zimage_rms_gate
