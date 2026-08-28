"""Host-side contract tests for the packaged Sol-Attn CUTE API."""

from types import SimpleNamespace

import pytest

from omni_xpu_kernel import cute


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
