"""Build only the changed Eagle module with the original 2025.3.2 compiler.

Put that compiler in PATH, then run `python build_eagle_only.py build_ext
--build-lib <output> --build-temp <temporary>`. No package or runtime is
installed by this script; its output is not a complete replacement wheel.
"""
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from esimd_build_extention import BuildExtension

setup(name='imawg4-eagle-experiment', version='0.0.1',
      ext_modules=[SyclExtension(
          name='custom_esimd_kernels_sglang.eagle_ops',
          sources=[str(root / 'csrc/eagle/eagle.sycl')],
          include_dirs=[str(root / 'csrc/eagle')],
          library_dirs=[str(Path(sys.prefix) / 'lib')],
          extra_compile_args={
              'cxx': ['-O3', '-std=c++20'],
              'sycl': ['-ffast-math', '-fsycl-device-code-split=per_kernel',
                       f"-I{Path(torch.__file__).parent / 'include'}"],
          },
          extra_link_args=['-Wl,-rpath,$ORIGIN/../../torch/lib'],
      )], cmdclass={'build_ext': BuildExtension})
