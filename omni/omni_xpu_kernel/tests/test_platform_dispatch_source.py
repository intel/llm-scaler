from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORM_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "norm.cpp"
).read_text(encoding="utf-8")
SDP_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "sdp.cpp"
).read_text(encoding="utf-8")
CUTE_FMHA_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "cute" / "cute_fmha_torch.cpp"
).read_text(encoding="utf-8")
BMG_POLICY_SOURCE = (
    PROJECT_ROOT
    / "omni_xpu_kernel"
    / "csrc"
    / "generated"
    / "bmg_kernel_policy_generated.h"
).read_text(encoding="utf-8")
BINDINGS_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "bindings.cpp"
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


def test_segmented_rms_modulation_uses_native_capability_without_sku_whitelist():
    policy_query = NORM_SOURCE.split(
        "bool rms_norm_segmented_modulation_supported", 1
    )[1].split("torch::Tensor rms_norm_segmented_modulation", 1)[0]

    assert "return input.is_xpu();" in policy_query
    assert "get_bmg_selection" not in policy_query
    assert "BmgKernelProfile" not in policy_query
    assert "physical_sku" not in policy_query
    assert ".forced" not in policy_query
    for legacy_name in (
        "rms_norm_modulate_b580",
        "supports_rms_norm_modulate_b580",
        "__rms_norm_modulate_b580__",
    ):
        assert legacy_name not in NORM_SOURCE
        assert legacy_name not in BINDINGS_SOURCE


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


def test_h3_vae_s1797_keeps_explicit_b580_b60_and_b70_kv_policies():
    b580_policy = BMG_POLICY_SOURCE.split("struct B580KernelPolicy", 1)[1]
    b580_policy = b580_policy.split("struct B60KernelPolicy", 1)[0]
    b60_policy = BMG_POLICY_SOURCE.split("struct B60KernelPolicy", 1)[1]
    b60_policy = b60_policy.split("struct B70KernelPolicy", 1)[0]
    b70_policy = BMG_POLICY_SOURCE.split("struct B70KernelPolicy", 1)[1]
    b70_policy = b70_policy.split("struct GenericBmgKernelPolicy", 1)[0]
    generic_policy = BMG_POLICY_SOURCE.split(
        "struct GenericBmgKernelPolicy", 1
    )[1].split("struct B580AdalnCandidatePolicy", 1)[0]
    assert "h3_vae_d64_s1797_kv_tile = 32" in b580_policy
    assert "h3_vae_d64_s1797_kv_tile = 64" in b60_policy
    assert "h3_vae_d64_s1797_kv_tile = 32" in b70_policy
    assert "h3_vae_d64_s1797_kv_tile = 32" in generic_policy
    assert 'policy["h3_vae_d64_s1797_kv_tile"]' in BINDINGS_SOURCE


def test_h3_vae_s1797_queries_b580_geometry_and_b60_inside_exact_shape():
    template_section = CUTE_FMHA_SOURCE.split("struct D128TileKernel", 1)[0]
    assert "int KvTileOverride = 0" in template_section
    assert (
        "KvTileOverride > 0 ? KvTileOverride : PlatformConfig::KV_TILE"
        in CUTE_FMHA_SOURCE
    )

    h3_section = CUTE_FMHA_SOURCE.split(
        "at::Tensor sdp_minimax_h3_vae_d64", 1
    )[1].split("at::Tensor sdp_bhld_d120", 1)[0]
    shape_guard = h3_section.index("if (L == 1797)")
    device_query = h3_section.index("get_bmg_selection_unwarned(queue)")
    b580_candidate = h3_section.index(
        "B580PolicyCandidate::\n            h3_vae_d64_s1797_kv_tile"
    )
    physical_b580 = h3_section.index(
        "selection.physical_sku == omni_xpu::device::BmgSku::b580"
    )
    b580_geometry = h3_section.index("B580KernelPolicy::")
    b60_profile = h3_section.index("BmgKernelProfile::b60")
    candidate = h3_section.index("h3_vae_d64_s1797_kv_tile")
    fallback = h3_section.index(
        "run_d128_tile<cutlass::half_t, 0, 0, 0, 0, 0, 64>("
    )
    assert (
        shape_guard
        < device_query
        < b580_candidate
        < physical_b580
        < b580_geometry
        < b60_profile
        < fallback
    )
    assert candidate < fallback
    assert "!selection.forced" in h3_section
