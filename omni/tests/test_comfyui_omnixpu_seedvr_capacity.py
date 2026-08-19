"""Installed-ComfyUI tests for bounded SeedVR2 activation scheduling."""

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
    / "seedvr_capacity.py"
)


def _load_adapter():
    name = "omnixpu_seedvr_capacity_test"
    spec = importlib.util.spec_from_file_location(name, _ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_seedvr():
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = True
    return pytest.importorskip("comfy.ldm.seedvr.model")


def test_chunk_rows_follow_the_live_byte_budget():
    adapter = _load_adapter()

    assert adapter._chunk_rows(1024, 1024, alignment=256) == 1
    assert adapter._chunk_rows(1024, 512 * 1024, alignment=256) == 512
    assert adapter._chunk_rows(1024, 511 * 1024, alignment=256) == 256


def test_chunked_forward_preallocates_one_output():
    adapter = _load_adapter()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, value):
            self.calls.append(value.shape[0])
            return value[:, :2] * 3

    module = Module()
    value = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    actual = adapter._chunked_forward(module, value, 4)

    assert module.calls == [4, 4, 2]
    torch.testing.assert_close(actual, value[:, :2] * 3, rtol=0, atol=0)


def test_bounded_swiglu_matches_the_installed_module():
    seedvr = _load_seedvr()
    comfy_ops = pytest.importorskip("comfy.ops")
    adapter = _load_adapter()
    module = seedvr.SwiGLUMLP(
        dim=12,
        expand_ratio=4,
        multiple_of=8,
        device="cpu",
        dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    generator = torch.Generator(device="cpu").manual_seed(53)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    value = torch.randn(17, 12, generator=generator)

    expected = module(value)
    actual = adapter._bounded_swiglu(module, value, 5)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "window_method",
    ["720pwin_by_size_bysize", "720pswin_by_size_bysize"],
)
@pytest.mark.parametrize("shared_weights", [False, True])
def test_bounded_attention_matches_the_installed_full_path(
    window_method, shared_weights, monkeypatch
):
    seedvr = _load_seedvr()
    comfy_ops = pytest.importorskip("comfy.ops")
    adapter = _load_adapter()
    heads = 2
    head_dim = 12
    dim = heads * head_dim
    attention = seedvr.NaSwinAttention(
        vid_dim=dim,
        txt_dim=dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=False,
        qk_norm=comfy_ops.disable_weight_init.RMSNorm,
        qk_norm_eps=1e-6,
        rope_type="mmrope3d",
        rope_dim=head_dim,
        shared_weights=shared_weights,
        window=(2, 2, 2),
        window_method=window_method,
        version=False,
        device="cpu",
        dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    generator = torch.Generator(device="cpu").manual_seed(47)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    vid = torch.randn(24, dim, generator=generator)
    txt = torch.randn(3, dim, generator=generator)
    vid_shape = torch.tensor([[2, 3, 4]], dtype=torch.long)
    txt_shape = torch.tensor([[3]], dtype=torch.long)

    calls = []
    optimized_var_attention = seedvr.optimized_var_attention

    def record(**kwargs):
        calls.append(kwargs["q"].shape[0])
        return optimized_var_attention(**kwargs)

    monkeypatch.setattr(seedvr, "optimized_var_attention", record)
    expected_vid, expected_txt = attention(
        vid.clone(),
        txt.clone(),
        vid_shape,
        txt_shape,
        seedvr.Cache(disable=True),
    )
    full_call_sizes = calls[:]

    calls.clear()
    qkv_row_bytes = 3 * heads * head_dim * vid.element_size()
    monkeypatch.setattr(
        adapter, "_ATTENTION_GROUP_BYTES", 12 * qkv_row_bytes
    )
    monkeypatch.setattr(adapter, "_LINEAR_GROUP_ROWS", 7)
    actual_vid, actual_txt = adapter._bounded_attention(
        seedvr,
        attention,
        vid.clone(),
        txt.clone(),
        vid_shape,
        txt_shape,
        seedvr.Cache(disable=True),
    )

    assert len(full_call_sizes) == 1
    assert len(calls) > 1
    assert max(calls) < full_call_sizes[0]
    torch.testing.assert_close(actual_vid, expected_vid, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_txt, expected_txt, rtol=1e-5, atol=1e-6)


def test_capacity_route_is_structural_not_model_named():
    adapter = _load_adapter()

    class Rope:
        mm = True

    class Attention:
        rope = Rope()
        heads = 20
        head_dim = 128

    management = type("Management", (), {"in_training": False})
    vid = torch.empty(1_000_000, 2560, device="meta", dtype=torch.bfloat16)
    txt = torch.empty(256, 2560, device="meta", dtype=torch.bfloat16)

    assert adapter._should_bound_attention(
        Attention(), vid, txt, torch.tensor([[32, 125, 250]]), management
    )
    assert not adapter._should_bound_attention(
        Attention(),
        vid[:1024],
        txt,
        torch.tensor([[4, 16, 16]]),
        management,
    )


def test_apply_accepts_the_installed_seedvr_contract_once():
    seedvr = _load_seedvr()
    adapter = _load_adapter()

    assert adapter.apply() == (True, None)
    assert getattr(seedvr.NaSwinAttention, adapter._PATCH_MARKER)
    assert getattr(seedvr.SwiGLUMLP, adapter._PATCH_MARKER)
    assert adapter.apply() == (
        False,
        "SeedVR2 capacity adapter is already applied",
    )
