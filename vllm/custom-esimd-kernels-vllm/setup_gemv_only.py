"""Build isolated ESIMD artifacts for fast kernel iteration.

This is not a production package build.  The resulting top-level DSOs are
loaded with ``torch.ops.load_library`` from a fresh process and never installed
under the production ``custom_esimd_kernels_vllm`` package.  The main artifact
intentionally excludes PLE, whose production owner is the full ``setup.py``
build.
"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension

# The full PyTorch architecture list can exceed icpx's device-link source
# location limit for this large build-only artifact.  Keep BMG as the
# reproducible default for the target cards, while allowing an explicit
# TORCH_XPU_ARCH_LIST override for another device family.
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "bmg-g31")

from esimd_build_extention import BuildExtension
import torch

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / 'include')

ext_modules = [
    SyclExtension(
        name='custom_esimd_kernels_gemv_only',
        sources=[
            'csrc/xpu/esimd_kernel.sycl',
            'csrc/xpu/torch_extension.cc',
        ],
        include_dirs=[
            root / 'include',
            root / 'csrc',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'sycl': ['-ffast-math', '-fsycl-device-code-split=per_kernel',
                     f'-I{torch_include}'],
        },
        extra_link_args=['-Wl,-rpath,$ORIGIN/torch/lib'],
        py_limited_api=False,
    ),
    SyclExtension(
        name='custom_esimd_kernels_gemm_only',
        sources=[
            'csrc/xpu/esimd_kernel_gemm.sycl',
            'csrc/xpu/torch_extension_gemm.cc',
        ],
        include_dirs=[
            root / 'include',
            root / 'csrc',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'sycl': ['-ffast-math', '-fsycl-device-code-split=per_kernel',
                     f'-I{torch_include}'],
        },
        extra_link_args=['-Wl,-rpath,$ORIGIN/torch/lib'],
        py_limited_api=False,
    ),
    SyclExtension(
        name='custom_esimd_kernels_moe_only',
        sources=[
            'csrc/xpu/esimd_kernel_moe.sycl',
            'csrc/xpu/torch_extension_moe.cc',
        ],
        include_dirs=[
            root / 'include',
            root / 'csrc',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'sycl': ['-ffast-math', '-fsycl-device-code-split=per_kernel',
                     f'-I{torch_include}'],
        },
        extra_link_args=['-Wl,-rpath,$ORIGIN/torch/lib'],
        py_limited_api=False,
    ),
]

setup(
    name='custom-esimd-kernels-vllm-gemv-only',
    version='0.1.0',
    # Build artifacts only; never install the production Python package.
    packages=[],
    py_modules=[],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
