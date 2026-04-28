from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension

os.environ.setdefault("TORCH_XPU_ARCH_LIST", "bmg")

from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch

torch_include = str(Path(torch.__file__).parent / "include")


setup(
    name="custom-esimd-kernels-vllm-moe-int4-only",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        SyclExtension(
            name="custom_esimd_kernels_vllm.moe_int4_ops",
            sources=[
                "csrc/moe_batch/moe_int4.sycl",
            ],
            include_dirs=[
                root / "csrc" / "moe_batch",
                root / "csrc" / "xpu" / "esimd_kernels",
                root / "csrc",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "sycl": [
                    "-fsycl",
                    "-ffast-math",
                    "-fsycl-device-code-split=per_kernel",
                    "-fsycl-targets=spir64_gen",
                    "-Xs",
                    "-device bmg",
                    f"-I{torch_include}",
                ],
            },
            extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
            py_limited_api=False,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)