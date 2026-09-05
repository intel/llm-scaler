"""Host-side contract tests for the packaged Sol-Attn CUTE API."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from omni_xpu_kernel import cute


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOL_CONFIG = PROJECT_ROOT / "omni_xpu_kernel/cute/sol_attn_config.h"
SOL_MAINLOOP = PROJECT_ROOT / "omni_xpu_kernel/cute/sol_attn_mainloop.hpp"
SOL_PREPARE = PROJECT_ROOT / "omni_xpu_kernel/cute/sol_attn_prepare.cpp"
SOL_WRAPPER = PROJECT_ROOT / "omni_xpu_kernel/cute/sol_attn_torch.cpp"


@pytest.fixture(autouse=True)
def _stub_policy_warning_preparation(monkeypatch):
    monkeypatch.setattr(cute, "_prepare_bmg_policy_dispatch", lambda tensor: None)


def test_sol_attn_capability_requires_control_aware_ops(monkeypatch):
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
            prepare_with_controls=lambda: None,
            forward_cute_with_controls=lambda: None,
        ),
    )
    assert cute.supports_sol_attn() is True


def test_sol_attn_native_ops_preserve_legacy_caller_abi():
    prepare = "".join(SOL_PREPARE.read_text(encoding="utf-8").split())
    wrapper = "".join(SOL_WRAPPER.read_text(encoding="utf-8").split())

    assert (
        '"prepare(Tensorq,Tensork,Tensorv,floatscale,floattau,"'
        '"intsink_start,intsink_end,intsink_q_start,intsink_q_end)"'
        '"->(Tensor,Tensor,Tensor,Tensor,Tensor)"'
    ) in prepare
    assert (
        '"prepare_with_controls(Tensorq,Tensork,Tensorv,floatscale,"'
        '"floattau,intsink_start,intsink_end,intsink_q_start,"'
        '"intsink_q_end,inttopk_count,Tensorblock_len)"'
        '"->(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)"'
    ) in prepare
    assert (
        '"forward_cute(Tensorq,Tensork,Tensorv,Tensork_centroids,"'
        '"Tensorv_means,Tensorq_centroids,Tensorthresholds,"'
        '"Tensorkey_sinks,floatscale)->Tensor"'
    ) in wrapper
    assert "forward_cute_with_controls" in wrapper
    assert (
        '"forward_cute_serial_route_parent(Tensorq,Tensork,Tensorv,"'
        '"Tensork_centroids,Tensorv_means,Tensorq_centroids,"'
        '"Tensorthresholds,Tensorkey_sinks,floatscale)->Tensor"'
    ) in wrapper
    assert "forward_cute_serial_route_parent_with_controls" in wrapper


def test_sol_attn_hides_native_prepare_contract(monkeypatch):
    calls = []
    result = object()

    def prepare(*args):
        calls.append(("prepare", args))
        return ("kc", "vm", "qc", "thresholds", "sinks", "topk_routes")

    def forward(*args):
        calls.append(("forward", args))
        return result

    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare_with_controls=prepare,
            forward_cute_with_controls=forward,
        ),
    )
    q = torch.empty((1, 15787, 56, 128), device="meta")
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
    assert prepare_args[0] is q
    assert prepare_args[1] is k
    assert prepare_args[2] is v
    assert prepare_args[3] == pytest.approx(128**-0.5)
    assert prepare_args[4:-1] == (1.3, 2, 4, 5, 6, -1)
    assert prepare_args[-1].dtype == torch.int32
    assert prepare_args[-1].numel() == 0
    forward_args = calls[1][1]
    assert forward_args[0] is q
    assert forward_args[1] is k
    assert forward_args[2] is v
    assert forward_args[3:9] == (
        "kc", "vm", "qc", "thresholds", "sinks", "topk_routes"
    )
    assert forward_args[9].dtype == torch.float32
    assert forward_args[9].numel() == 0
    assert forward_args[10].dtype == torch.int32
    assert forward_args[10].numel() == 0
    assert forward_args[11:] == (
        pytest.approx(128**-0.5), True, False
    )


def test_sol_attn_prepares_core_owned_bmg_policy_warning(monkeypatch):
    prepared = []
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute, "_prepare_bmg_policy_dispatch", prepared.append
    )
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare_with_controls=lambda *args: (
                "kc", "vm", "qc", "thresholds", "sinks", "routes"
            ),
            forward_cute_with_controls=lambda *args: "out",
        ),
    )
    q = torch.empty((1, 64, 1, 128), device="meta")

    assert cute.sol_attn(q, object(), object()) == "out"
    assert prepared == [q]


def test_sol_attn_forwards_tail_bias_and_block_lengths(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare_with_controls=lambda *args: (
                "kc",
                "vm",
                "qc",
                "thresholds",
                "sinks",
                "topk_routes",
            ),
            forward_cute_with_controls=lambda *args: calls.append(args)
            or "out",
        ),
    )
    q = torch.empty((1, 64, 1, 128))

    assert cute.sol_attn(q, object(), object(), tail=False) == "out"
    assert calls[0][-2] is False

    with pytest.raises(ValueError, match="block_len"):
        cute.sol_attn(
            q,
            object(),
            object(),
            block_len=torch.ones(2, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="coarse_gate"):
        cute.sol_attn(
            q,
            object(),
            object(),
            coarse_gate=torch.ones(1),
        )

    calls.clear()
    bias = torch.tensor([True, False]).repeat(32)
    assert cute.sol_attn(q, object(), object(), key_bias=bias) == "out"
    forwarded_bias = calls[0][-5]
    assert forwarded_bias.shape == (1, 64)
    assert forwarded_bias.dtype == torch.float32
    assert forwarded_bias[0, 0] == 0
    assert forwarded_bias[0, 1] == float("-inf")

    calls.clear()
    lengths = torch.tensor([32], dtype=torch.int32)
    assert cute.sol_attn(q, object(), object(), block_len=lengths) == "out"
    assert torch.equal(calls[0][-4], lengths)

    calls.clear()
    assert cute.sol_attn(
        q,
        object(),
        object(),
        topk_ratio=0.25,
        tail=False,
    ) == "out"
    assert calls[0][-2:] == (False, True)


def test_sol_attn_topk_budget_matches_kitchen_rounding(monkeypatch):
    prepare_calls = []

    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare_with_controls=lambda *args: prepare_calls.append(args)
            or (
                "kc",
                "vm",
                "qc",
                "thresholds",
                "sinks",
                "topk_routes",
            ),
            forward_cute_with_controls=lambda *args: "out",
        ),
    )
    q = torch.empty((1, 64 * 10, 1, 128))

    cute.sol_attn(q, object(), object(), topk_ratio=0.25)
    assert prepare_calls[-1][-2] == 2

    cute.sol_attn(
        q,
        object(),
        object(),
        topk_ratio=0.5,
        sink_blocks=(0, 9),
    )
    assert prepare_calls[-1][-2] == 0

    with pytest.raises(ValueError, match="topk_ratio"):
        cute.sol_attn(q, object(), object(), topk_ratio=1.0)


def test_sol_attn_adds_coarse_gate_from_prepared_block_means(monkeypatch):
    q = torch.zeros((1, 64, 1, 128), dtype=torch.bfloat16)
    k_centroids = torch.ones((1, 1, 1, 128), dtype=torch.bfloat16)
    v_means = torch.full_like(k_centroids, 2)
    q_centroids = torch.ones((1, 1, 1, 128), dtype=torch.float32)
    thresholds = torch.zeros((1, 1, 1), dtype=torch.float32)
    sinks = torch.zeros((1, 1, 1), dtype=torch.uint8)
    routes = torch.empty(0, dtype=torch.uint8)

    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare_with_controls=lambda *args: (
                k_centroids,
                v_means,
                q_centroids,
                thresholds,
                sinks,
                routes,
            ),
            forward_cute_with_controls=lambda *args: torch.zeros_like(q),
        ),
    )
    gate = torch.full_like(q, 0.5)
    actual = cute.sol_attn(q, q, q, coarse_gate=gate)
    torch.testing.assert_close(actual, torch.ones_like(actual))


def test_sol_attn_rejects_malformed_sink_ranges(monkeypatch):
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        cute,
        "_sol_attn_ops",
        lambda: SimpleNamespace(
            prepare_with_controls=lambda: None,
            forward_cute_with_controls=lambda: None,
        ),
    )
    q = torch.empty((1, 64, 1, 128))
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


def test_sol_attn_default_contract_selects_compile_time_no_controls_mainloop():
    mainloop = "".join(SOL_MAINLOOP.read_text(encoding="utf-8").split())
    wrapper = "".join(SOL_WRAPPER.read_text(encoding="utf-8").split())

    assert "boolControlAware=true" in mainloop
    assert "ifconstexpr(ControlAware)" in mainloop
    assert "ifconstexpr(!ControlAware)" in mainloop
    assert (
        "constboolhas_controls=key_bias.numel()!=0||block_len.numel()!=0||"
        "!tail||route_inclusive;"
    ) in wrapper
    assert "ParentTag,true>(" in wrapper
    assert "ParentTag,false>(" in wrapper
    assert (
        "key_bias.numel()==0&&block_len.numel()==0&&tail&&!route_inclusive"
    ) in wrapper
