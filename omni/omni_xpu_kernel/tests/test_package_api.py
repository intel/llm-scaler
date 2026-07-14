import omni_xpu_kernel


def test_native_capabilities_is_empty_when_extension_is_unavailable(monkeypatch):
    def unavailable():
        raise ImportError("native extension unavailable")

    monkeypatch.setattr(omni_xpu_kernel, "_load_extension", unavailable)
    assert omni_xpu_kernel.native_capabilities() == {}
