"""Standalone build for prefill FMHA development tasks.

Usage:
  # Build a specific task kernel:
  KERNEL=test_dpas_hd256 TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  KERNEL=flash_attn_minimal TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace

  # With doubleGRF:
  KERNEL=flash_attn_minimal USE_DOUBLE_GRF=1 TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace

  # Default (no KERNEL env): builds flash_attn_minimal
"""
import os
import sys
from pathlib import Path

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from esimd_build_extention import BuildExtension

import torch
from torch.utils.cpp_extension import SyclExtension

torch_include = str(Path(torch.__file__).parent / "include")

kernel_name = os.environ.get("KERNEL", "flash_attn_minimal")
use_double_grf = os.environ.get("USE_DOUBLE_GRF", "0") == "1"

sycl_flags = [
    "-ffast-math",
    "-fsycl-device-code-split=per_kernel",
    f"-I{torch_include}",
]

if use_double_grf:
    sycl_flags += ["-fsycl-targets=spir64_gen", "-Xs", "-device bmg-g21 -options -doubleGRF"]

print(f">>> Building: {kernel_name}")
print(f">>> doubleGRF: {'ON' if use_double_grf else 'OFF'}")

setup(
    name=f"prefill-attn-{kernel_name}",
    version="0.0.1",
    ext_modules=[
        SyclExtension(
            name=kernel_name,
            sources=[f"{kernel_name}.sycl"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "sycl": sycl_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
