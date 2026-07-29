from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORM_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "norm.cpp"
).read_text(encoding="utf-8")
SDP_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "sdp.cpp"
).read_text(encoding="utf-8")


def _selector_section(selector: str, next_section: str) -> str:
    return NORM_SOURCE.split(selector, 1)[1].split(next_section, 1)[0]


def _assert_common_ladder(section: str, kernel: str) -> None:
    assert "_WIN32" not in section
    for limit, gs in ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16)):
        assert f"if (nb <= {limit})" in section
        assert f"{kernel}<IT, {gs}," in section
    assert f"return {kernel}<IT, 32, BS>;" in section


def test_norm_dispatch_ladders_are_platform_independent():
    _assert_common_ladder(
        _selector_section(
            "select_rms_kernel(int nb)",
            "// LayerNorm dispatch",
        ),
        "rms_norm_kernel",
    )
    _assert_common_ladder(
        _selector_section(
            "select_ln_kernel(int nb)",
            "// Fused add rms norm dispatch",
        ),
        "layer_norm_kernel",
    )
    _assert_common_ladder(
        _selector_section(
            "select_fused_kernel(int nb)",
            "// ============================================================================\n"
            "// Public C++ API",
        ),
        "fused_add_rms_norm_kernel",
    )


def test_windows_sdp_loader_resolves_every_exported_kernel():
    loader = SDP_SOURCE.split("std::call_once(load_once", 1)[1]
    platform_loader = loader.split("#ifdef _WIN32", 1)[1]
    windows_loader, linux_loader = platform_loader.split("#else", 1)
    linux_loader = linux_loader.split("#endif", 1)[0]
    for symbol in (
        "sdp_fp16",
        "sdp_bf16io",
        "sdp_fp16_fast",
        "sdp_fp16_hd64",
        "sdp_bf16io_hd64",
    ):
        assert (
            f'GetProcAddress(library.handle, "{symbol}")'
            in windows_loader
        )
        assert f'dlsym(library.handle, "{symbol}")' in linux_loader
