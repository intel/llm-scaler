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
