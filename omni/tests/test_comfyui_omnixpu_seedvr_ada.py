"""Portable tests for the guarded SeedVR2 Ada reshape patch."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

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


class _Cache:
    def __init__(self):
        self.keys = []

    def __call__(self, key, function):
        self.keys.append(key)
        return function()


class _BoundedAda:
    dim = 4
    layers = ("attn", "mlp")
    attn_shift = torch.arange(4, dtype=torch.float32)
    attn_scale = torch.arange(4, dtype=torch.float32) / 10
    attn_gate = torch.arange(4, dtype=torch.float32) / 20

    def forward(
        self,
        hid,
        emb,
        layer,
        mode,
        cache=None,
        branch_tag="",
        hid_len=None,
    ):
        if cache is None:
            cache = _Cache()
        idx = self.layers.index(layer)
        emb = emb.reshape(emb.shape[0], -1, len(self.layers), 3)[:, :, idx, :]
        emb = emb.reshape(emb.shape[0], 1, emb.shape[1], emb.shape[2])

        if hid_len is not None:
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: torch.repeat_interleave(emb, hid_len, dim=0),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        if mode == "in":
            shiftB = comfy.ops.cast_to_input(shiftB, hid)
            scaleB = comfy.ops.cast_to_input(scaleB, hid)
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            if gateB is not None:
                gateB = comfy.ops.cast_to_input(gateB, hid)
                return hid.mul_(gateA + gateB)
            else:
                return hid.mul_(gateA)

        raise ValueError(f"Unknown AdaSingle mode: {mode}")


comfy = types.SimpleNamespace(
    ops=types.SimpleNamespace(cast_to_input=lambda value, _input: value)
)


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
    assert reason == "upstream Ada reshape and broadcast are already bounded"


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


def test_rewrite_removes_large_repeat_and_preserves_segmented_math(monkeypatch):
    fix = _load_fix()
    patched, reason = fix._rewrite_forward(_BoundedAda.forward)

    assert reason is None
    generator = torch.Generator(device="cpu").manual_seed(41)
    emb = torch.randn(2, 24, generator=generator)
    hid = torch.randn(5, 4, generator=generator)
    lengths = torch.tensor([2, 3])
    expanded = emb.reshape(2, 4, 2, 3)[:, :, 0, :]
    repeated = torch.repeat_interleave(expanded, lengths, dim=0)
    shift, scale, _ = repeated.unbind(-1)
    expected = hid.clone().mul_(scale + _BoundedAda.attn_scale).add_(
        shift + _BoundedAda.attn_shift
    )

    monkeypatch.setattr(
        torch,
        "repeat_interleave",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("conditioning must not be repeated")
        ),
    )
    cache = _Cache()
    actual = patched(
        _BoundedAda(),
        hid.clone(),
        emb,
        "attn",
        "in",
        cache=cache,
        branch_tag="vid",
        hid_len=lengths,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.keys == []


def test_rewrite_preserves_out_math_with_an_empty_batch_segment(monkeypatch):
    fix = _load_fix()
    patched, reason = fix._rewrite_forward(_BoundedAda.forward)

    assert reason is None
    generator = torch.Generator(device="cpu").manual_seed(59)
    emb = torch.randn(3, 24, generator=generator)
    hid = torch.randn(5, 4, generator=generator)
    lengths = torch.tensor([2, 0, 3])
    expanded = emb.reshape(3, 4, 2, 3)[:, :, 0, :]
    repeated = torch.repeat_interleave(expanded, lengths, dim=0)
    _, _, gate = repeated.unbind(-1)
    expected = hid.clone().mul_(gate + _BoundedAda.attn_gate)

    monkeypatch.setattr(
        torch,
        "repeat_interleave",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("conditioning must not be repeated")
        ),
    )
    cache = _Cache()
    actual = patched(
        _BoundedAda(),
        hid.clone(),
        emb,
        "attn",
        "out",
        cache=cache,
        branch_tag="vid",
        hid_len=lengths,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.keys == []


def test_rewrite_accepts_the_installed_comfyui_contract():
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = True
    seedvr = pytest.importorskip("comfy.ldm.seedvr.model")
    fix = _load_fix()

    patched, reason = fix._rewrite_forward(seedvr.AdaSingle.forward)

    assert reason is None
    assert patched is not None
