"""Host-side contract tests for the packaged Sol-Attn CUTE API."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from omni_xpu_kernel import cute


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOL_CONFIG = PROJECT_ROOT / "omni_xpu_kernel/cute/sol_attn_config.h"
SOL_WRAPPER = PROJECT_ROOT / "omni_xpu_kernel/cute/sol_attn_torch.cpp"


def test_sol_attn_capability_requires_prepare_and_forward(monkeypatch):
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(prepare=lambda: None),
    )
    assert cute.supports_sol_attn() is False

    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare=lambda: None,
            forward_cute=lambda: None,
        ),
    )
    assert cute.supports_sol_attn() is True


def test_sol_attn_hides_native_prepare_contract(monkeypatch):
    calls = []
    result = object()

    def prepare(*args):
        calls.append(("prepare", args))
        return ("kc", "vm", "qc", "thresholds", "sinks")

    def forward(*args):
        calls.append(("forward", args))
        return result

    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(prepare=prepare, forward_cute=forward),
    )
    q = SimpleNamespace(shape=(1, 15787, 56, 128))
    k = object()
    v = object()

    actual = cute.sol_attn(
        q,
        k,
        v,
        tau=1.3,
        sink_blocks=(2, 4),
        sink_q=(5, 6),
    )

    assert actual is result
    prepare_args = calls[0][1]
    assert prepare_args[:3] == (q, k, v)
    assert prepare_args[3] == pytest.approx(128**-0.5)
    assert prepare_args[4:] == (1.3, 2, 4, 5, 6)
    assert calls[1][1] == (
        q,
        k,
        v,
        "kc",
        "vm",
        "qc",
        "thresholds",
        "sinks",
        pytest.approx(128**-0.5),
    )


def test_sol_attn_rejects_malformed_sink_ranges(monkeypatch):
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(prepare=lambda: None, forward_cute=lambda: None),
    )
    q = SimpleNamespace(shape=(1, 64, 1, 128))
    with pytest.raises(ValueError, match="must each contain two indices"):
        cute.sol_attn(q, object(), object(), sink_blocks=(0,))


def test_sol_attn_b580_tile_policy_requires_unforced_physical_device():
    config = SOL_CONFIG.read_text(encoding="utf-8")
    wrapper = SOL_WRAPPER.read_text(encoding="utf-8")
    compact = "".join(wrapper.split())

    assert "#define SOL_ATTN_Q_TILE 256" in config
    assert "#define SOL_ATTN_SUBGROUP_LAYOUT_Q 32" in config
    assert "#define SOL_ATTN_B580_Q_TILE 128" in config
    assert "#define SOL_ATTN_B580_SUBGROUP_LAYOUT_Q 16" in config
    assert "#define SOL_ATTN_B580_GRF_SIZE 256" in config
    assert (
        "constautoselection="
        "omni_xpu::device::get_bmg_selection_unwarned(queue);"
    ) in compact
    assert (
        "returnselection.physical_sku=="
        "omni_xpu::device::BmgSku::b580&&!selection.forced;"
    ) in compact
    assert "get_bmg_selection_unwarned(queue).physical_sku" not in wrapper
    assert wrapper.count("if (use_b580_tile_policy(q))") == 3
    assert "SolConfiguredTilePolicy" in wrapper
    assert "SolB580TilePolicy" in wrapper
