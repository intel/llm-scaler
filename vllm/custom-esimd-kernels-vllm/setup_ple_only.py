"""Build the PLE-only dispatcher DSO without unrelated DPAS translation units.

The resulting shared object is loaded with ``torch.ops.load_library``.  It is
not imported as a Python module; the operators register in the
``custom_esimd_kernels_vllm`` dispatcher namespace via TORCH_LIBRARY_FRAGMENT.
This build intentionally installs no ``custom_esimd_kernels_vllm`` Python
package, so it cannot overwrite or trigger the production package loader.
Run standalone tests in a fresh process with the resulting DSO path.
"""

from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension

from esimd_build_extention import BuildExtension


ROOT = Path(__file__).resolve().parent
TORCH_INCLUDE = str(Path(torch.__file__).parent / "include")


setup(
    name="custom-esimd-kernels-vllm-ple-only",
    version="0.1.0",
    # This distribution is a DSO artifact only.  Do not install the
    # production Python package alongside it.
    packages=[],
    py_modules=[],
    ext_modules=[
        SyclExtension(
            name="ple_ops",
            sources=[
                "csrc/xpu/esimd_kernel_ple.sycl",
                "csrc/xpu/torch_extension_ple.cc",
            ],
            include_dirs=[ROOT / "include", ROOT / "csrc"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "sycl": [
                    "-ffast-math",
                    "-fsycl-device-code-split=per_kernel",
                    f"-I{TORCH_INCLUDE}",
                ],
            },
            extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
            py_limited_api=False,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
