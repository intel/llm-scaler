"""Standalone setup for building ONLY the moe_int4_prefill_ops extension.

Avoids triggering re-build of other extensions (some of which fail on the
current toolchain because they AOT-compile for XeLPG iGPUs that do not
support DPAS2 intrinsics).

Usage:
    cd /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm
    python /tmp/setup_moe_prefill.py build_ext --inplace
"""
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

# Run from the package root so relative source paths resolve.
root = Path(__file__).parent.resolve()
pkg_root = Path("/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm").resolve()

# When invoked from the package root (recommended) we use cwd; otherwise fall
# back to the hard-coded absolute path.
if (Path.cwd() / "setup.py").exists():
    root = Path.cwd().resolve()
else:
    root = pkg_root

import torch
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm.moe_int4_prefill_ops",
        sources=[
            "csrc/moe_prefill/moe_prefill_int4.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_prefill",
            root / "csrc" / "xpu" / "esimd_kernels",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]

setup(
    name="moe_int4_prefill_ops-build",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
