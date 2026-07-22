from types import SimpleNamespace

import omni_xpu_kernel


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
