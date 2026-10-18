"""Build only the opt-in Q8 consistency companion with the baseline toolchain."""
import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from esimd_build_extention import BuildExtension

setup(name="custom-esimd-cache-gemv", version="0.1.0",
      ext_modules=[SyclExtension(
          name="custom_esimd_kernels_sglang.cache_gemv_ops",
          sources=[str(root / "csrc/xpu/cache_gemv.sycl"), str(root / "csrc/xpu/cache_moe.sycl")],
          include_dirs=[str(root / "csrc/xpu")],
          library_dirs=[str(Path(sys.prefix) / "lib")],
          extra_compile_args={"cxx": ["-O3", "-std=c++20"],
                              "sycl": ["-O3", "-std=c++20", "-ffast-math", "-fsycl-device-code-split=per_kernel"]},
          extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
      )], cmdclass={"build_ext": BuildExtension})
