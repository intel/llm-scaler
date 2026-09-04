from types import SimpleNamespace

import omni_xpu_kernel
import pytest


def test_native_capabilities_is_empty_when_extension_is_unavailable(monkeypatch):
    def unavailable():
        raise ImportError("native extension unavailable")

    monkeypatch.setattr(omni_xpu_kernel, "_load_extension", unavailable)
    assert omni_xpu_kernel.native_capabilities() == {}


def test_core_aot_target_comes_from_loaded_binary(monkeypatch):
    native = SimpleNamespace(__core_aot_target__="ptl-h")
    monkeypatch.setattr(omni_xpu_kernel, "_load_extension", lambda: native)

    assert omni_xpu_kernel.core_aot_target() == "ptl-h"


def test_core_aot_target_is_empty_for_legacy_or_unavailable_binary(monkeypatch):
    monkeypatch.setattr(
        omni_xpu_kernel, "_load_extension", lambda: SimpleNamespace()
    )
    assert omni_xpu_kernel.core_aot_target() == ""

    def unavailable():
        raise ImportError("native extension unavailable")

    monkeypatch.setattr(omni_xpu_kernel, "_load_extension", unavailable)
    assert omni_xpu_kernel.core_aot_target() == ""


def test_norm_h120_capability_comes_from_loaded_binary(monkeypatch):
    from omni_xpu_kernel import norm

    monkeypatch.setattr(
        norm, "_get_native", lambda: SimpleNamespace(__h120_fp16__=True)
    )
    assert norm.supports_h120_fp16() is True


def test_norm_h120_capability_is_false_for_legacy_binary(monkeypatch):
    from omni_xpu_kernel import norm

    monkeypatch.setattr(norm, "_get_native", lambda: SimpleNamespace())
    assert norm.supports_h120_fp16() is False


def test_segmented_rms_modulation_capability_policy_and_wrapper(monkeypatch):
    from omni_xpu_kernel import norm

    calls = []

    def run_native(*args):
        calls.append(args)
        return "output"

    native = SimpleNamespace(
        __rms_norm_segmented_modulation__=True,
        rms_norm_segmented_modulation_supported=lambda value: value == "input",
        rms_norm_segmented_modulation=run_native,
    )
    monkeypatch.setattr(norm, "_get_native", lambda: native)
    tensors = [object() for _ in range(4)]
    segments = [(0, 3, 2), (3, 7, 0)]

    assert norm.supports_rms_norm_segmented_modulation() is True
    assert norm.rms_norm_segmented_modulation_supported("input") is True
    assert norm.rms_norm_segmented_modulation_supported("other") is False
    assert norm.rms_norm_segmented_modulation(
        *tensors, segments, eps=1e-5
    ) == "output"
    assert calls == [
        (
            *tensors,
            [0, 3],
            [3, 7],
            [2, 0],
            1e-5,
        )
    ]


def test_segmented_rms_modulation_rejects_non_integer_segments(monkeypatch):
    from omni_xpu_kernel import norm

    monkeypatch.setattr(norm, "_get_native", lambda: SimpleNamespace())
    with pytest.raises(TypeError, match="Python integers"):
        norm.rms_norm_segmented_modulation(
            object(), object(), object(), object(), [(0, 7, True)]
        )

    assert norm.supports_rms_norm_segmented_modulation() is False
    assert norm.rms_norm_segmented_modulation_supported(object()) is False


def test_segmented_rms_modulation_policy_query_fails_closed(monkeypatch):
    from omni_xpu_kernel import norm

    def rejected(_input):
        raise RuntimeError("unsupported device policy")

    native = SimpleNamespace(
        __rms_norm_segmented_modulation__=True,
        rms_norm_segmented_modulation_supported=rejected,
    )
    monkeypatch.setattr(norm, "_get_native", lambda: native)

    assert norm.rms_norm_segmented_modulation_supported(object()) is False


def test_layout_cat_pad_capability_comes_from_loaded_binary(monkeypatch):
    from omni_xpu_kernel import layout

    monkeypatch.setattr(
        layout, "_get_native", lambda: SimpleNamespace(__cat_pad_bmg__=True)
    )
    assert layout.supports_cat_pad_bmg() is True


def test_layout_cat_pad_capability_is_false_for_legacy_binary(monkeypatch):
    from omni_xpu_kernel import layout

    monkeypatch.setattr(layout, "_get_native", lambda: SimpleNamespace())
    assert layout.supports_cat_pad_bmg() is False


def test_rotary_fast_capability_comes_from_loaded_binary(monkeypatch):
    from omni_xpu_kernel import rotary

    native = SimpleNamespace(kitchen_rope_fast_supported=lambda _x, _freqs: True)
    monkeypatch.setattr(rotary, "_get_native", lambda: native)

    assert rotary.supports_kitchen_rope_fast() is True
    assert rotary.kitchen_rope_fast_supported(object(), object()) is True


def test_rotary_fast_capability_is_safe_for_legacy_or_rejected_inputs(monkeypatch):
    from omni_xpu_kernel import rotary

    monkeypatch.setattr(rotary, "_get_native", lambda: SimpleNamespace())
    assert rotary.supports_kitchen_rope_fast() is False
    assert rotary.kitchen_rope_fast_supported(object(), object()) is False

    def rejected(_x, _freqs):
        raise RuntimeError("unsupported tensor contract")

    monkeypatch.setattr(
        rotary,
        "_get_native",
        lambda: SimpleNamespace(kitchen_rope_fast_supported=rejected),
    )
    assert rotary.kitchen_rope_fast_supported(object(), object()) is False


def test_ltx_rope_capability_comes_from_loaded_binary(monkeypatch):
    from omni_xpu_kernel import rotary

    native = SimpleNamespace(
        ltx_split_rope_direct_supported=lambda _input, _cos, _sin: True
    )
    monkeypatch.setattr(rotary, "_get_native", lambda: native)

    assert rotary.supports_ltx_split_rope_direct() is True
    assert rotary.ltx_split_rope_direct_supported(
        object(), object(), object()
    ) is True


def test_ltx_rope_capability_is_safe_for_legacy_or_rejected_inputs(
    monkeypatch,
):
    from omni_xpu_kernel import rotary

    monkeypatch.setattr(rotary, "_get_native", lambda: SimpleNamespace())
    assert rotary.supports_ltx_split_rope_direct() is False
    assert not rotary.ltx_split_rope_direct_supported(
        object(), object(), object()
    )

    def rejected(_input, _cos, _sin):
        raise RuntimeError("unsupported tensor contract")

    monkeypatch.setattr(
        rotary,
        "_get_native",
        lambda: SimpleNamespace(ltx_split_rope_direct_supported=rejected),
    )
    assert not rotary.ltx_split_rope_direct_supported(
        object(), object(), object()
    )
