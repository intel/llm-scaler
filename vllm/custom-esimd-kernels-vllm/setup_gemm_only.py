"""Build only the ESIMD GEMM/GEMV extension for fast kernel iteration."""

from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension

from esimd_build_extention import BuildExtension


root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")

setup(
    name="custom-esimd-kernels-vllm-gemm-only",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        SyclExtension(
            name="custom_esimd_kernels_vllm.custom_esimd_kernels_gemm",
            sources=[
                "csrc/xpu/esimd_kernel_gemm.sycl",
                "csrc/xpu/torch_extension_gemm.cc",
            ],
            include_dirs=[root / "include", root / "csrc"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "sycl": [
                    "-ffast-math",
                    "-fsycl-device-code-split=per_kernel",
                    f"-I{torch_include}",
                ],
            },
            extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
            py_limited_api=False,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
