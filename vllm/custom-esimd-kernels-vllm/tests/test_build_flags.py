import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import esimd_build_extention as build_extension  # noqa: E402


def test_sycl_dlink_uses_common_compile_target(monkeypatch):
    monkeypatch.setenv("TORCH_XPU_ARCH_LIST", "bmg-g21")

    flags = build_extension._get_sycl_dlink_flags(
        build_extension._COMMON_SYCL_FLAGS)

    assert "-fsycl-targets=spir64_gen,spir64" in flags
    assert '-Xs "-device bmg-g21"' in flags


def test_sycl_dlink_uses_extension_target_override(monkeypatch):
    monkeypatch.setenv("TORCH_XPU_ARCH_LIST", "bmg-g21")
    compile_flags = [
        *build_extension._COMMON_SYCL_FLAGS,
        "-fsycl-targets=spir64_gen",
    ]

    flags = build_extension._get_sycl_dlink_flags(compile_flags)

    assert "-fsycl-targets=spir64_gen" in flags
    assert "-fsycl-targets=spir64_gen,spir64" not in flags
    assert flags.count("-fsycl-targets=spir64_gen") == 1
