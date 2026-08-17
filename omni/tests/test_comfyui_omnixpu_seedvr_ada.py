"""Portable tests for the guarded SeedVR2 Ada reshape patch."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_FIX = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "fixes"
    / "seedvr_ada.py"
)


def _load_fix():
    name = "omnixpu_seedvr_ada_test"
    spec = importlib.util.spec_from_file_location(name, _FIX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _ShapeProbe:
    shape = (2, 132)

    def __init__(self):
        self.reshape_args = None

    def reshape(self, *shape):
        self.reshape_args = shape
        return self


class _LegacyAda:
    dim = 11
    layers = ("attn", "mlp")

    def forward(self, emb):
        return emb.reshape(emb.shape[0], -1, len(self.layers), 3)


class _ExplicitAda:
    dim = 11
    layers = ("attn", "mlp")

    def forward(self, emb):
        return emb.reshape(emb.shape[0], self.dim, -1, 3)


class _UnknownAda:
    def forward(self, emb):
        return emb.reshape(emb.shape[0], -1, 6)


def test_rewrite_makes_the_feature_dimension_explicit():
    fix = _load_fix()
    patched, reason = fix._rewrite_forward(_LegacyAda.forward)

    assert reason is None
    probe = _ShapeProbe()
    patched(_LegacyAda(), probe)
    assert probe.reshape_args == (2, 11, -1, 3)


def test_rewrite_skips_an_upstream_fixed_implementation():
    fix = _load_fix()

    patched, reason = fix._rewrite_forward(_ExplicitAda.forward)

    assert patched is None
    assert reason == "upstream Ada reshape is already explicit"


def test_rewrite_rejects_an_unknown_implementation():
    fix = _load_fix()

    patched, reason = fix._rewrite_forward(_UnknownAda.forward)

    assert patched is None
    assert reason == "unsupported AdaSingle.forward reshape contract"


def test_apply_patches_the_comfyui_class_once(monkeypatch):
    fix = _load_fix()

    class AdaSingle:
        dim = 11
        layers = ("attn", "mlp")

        def forward(self, emb):
            return emb.reshape(emb.shape[0], -1, len(self.layers), 3)

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    seedvr = types.ModuleType("comfy.ldm.seedvr")
    seedvr.__path__ = []
    model = types.ModuleType("comfy.ldm.seedvr.model")
    model.AdaSingle = AdaSingle
    seedvr.model = model
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ldm", ldm)
    monkeypatch.setitem(sys.modules, "comfy.ldm.seedvr", seedvr)
    monkeypatch.setitem(sys.modules, "comfy.ldm.seedvr.model", model)

    assert fix.apply() == (True, None)
    probe = _ShapeProbe()
    AdaSingle().forward(probe)
    assert probe.reshape_args == (2, 11, -1, 3)
    assert fix.apply() == (
        False,
        "SeedVR2 Ada reshape patch is already applied",
    )
