"""Portable tests for the guarded SeedVR2 cat-pad adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


_ADAPTER = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "adapters"
    / "seedvr_cat_pad.py"
)


def _load_adapter():
    name = "omnixpu_seedvr_cat_pad_test"
    spec = importlib.util.spec_from_file_location(name, _ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Tensor:
    is_xpu = True
    dtype = torch.float16
    requires_grad = False
    device = torch.device("xpu:0")

    def __init__(self, shape, stride, *, contiguous=False):
        self.shape = shape
        self._stride = stride
        self._contiguous = contiguous

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return self._contiguous

    def element_size(self):
        return 2


def _contract():
    shape = (1, 128, 4, 512, 512)
    input = _Tensor(
        shape,
        (134217728, 262144, 33554432, 512, 1),
    )
    prefix = _Tensor(
        (1, 128, 2, 512, 512),
        (67108864, 524288, 262144, 512, 1),
        contiguous=True,
    )
    return input, prefix


def test_route_requires_the_complete_materialization_contract():
    adapter = _load_adapter()
    adapter._layout = object()
    input, prefix = _contract()
    module = type("Conv", (), {"memory_limit": 2.0})()

    assert adapter._can_use(
        module, input, prefix, 3, (1, 1, 1, 1, 0, 0)
    )
    module.memory_limit = 0.25
    assert not adapter._can_use(
        module, input, prefix, 3, (1, 1, 1, 1, 0, 0)
    )
    module.memory_limit = 2.0
    assert not adapter._can_use(
        module, input, prefix, 4, (1, 1, 1, 1, 0, 0)
    )
    assert not adapter._can_use(
        module, input, prefix, 3, (0, 0, 0, 0, 0, 0)
    )


def test_route_rejects_a_different_input_layout():
    adapter = _load_adapter()
    adapter._layout = object()
    input, prefix = _contract()
    input._stride = (134217728, 1048576, 262144, 512, 1)
    module = type("Conv", (), {"memory_limit": 2.0})()

    assert not adapter._can_use(
        module, input, prefix, 3, (1, 1, 1, 1, 0, 0)
    )


def test_installed_comfyui_contract_is_accepted_once():
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = True
    seedvr_vae = pytest.importorskip("comfy.ldm.seedvr.vae")
    adapter = _load_adapter()
    conv = seedvr_vae.InflatedCausalConv3d
    original = conv.memory_limit_conv

    assert adapter._source_has_contract(original)
    try:
        assert adapter._patch(seedvr_vae) == (True, None)
        assert getattr(conv, adapter._PATCH_MARKER)
        assert adapter._patch(seedvr_vae) == (
            False,
            "SeedVR2 cat-pad adapter is already applied",
        )
    finally:
        conv.memory_limit_conv = original
        if hasattr(conv, adapter._PATCH_MARKER):
            delattr(conv, adapter._PATCH_MARKER)
