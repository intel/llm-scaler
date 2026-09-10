"""Standalone setup script that only builds the Qwen3.8 QSA extension.

The target architecture intentionally comes from ``TORCH_XPU_ARCH_LIST`` so
the focused build uses the same architecture selection as the full wheel:

    TORCH_XPU_ARCH_LIST=bmg-g31 python setup_qsa_only.py build_ext --inplace
"""
from pathlib import Path

from setuptools import find_packages, setup

from esimd_build_extention import BuildExtension
from qsa_build import make_qsa_extension


root = Path(__file__).parent.resolve()

import torch


torch_include = str(Path(torch.__file__).parent / "include")


setup(
    name="custom-esimd-kernels-vllm-qsa-only",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[make_qsa_extension(root, torch_include)],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
