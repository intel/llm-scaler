import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension
import torch

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")

# DPAS/XMX prefill SDPA (fp16, HD=256). AOT for the actual GPU (JIT DPAS is
# unreliable) with -doubleGRF. Device overridable via OMNI_XPU_DEVICE.
_DEV = os.environ.get("OMNI_XPU_DEVICE", "bmg")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_sglang.custom_esimd_kernels_prefill_dpas",
        sources=[
            "csrc/xpu/esimd_kernel_prefill_dpas.sycl",
            "csrc/xpu/torch_extension_prefill_dpas.cc",
        ],
        include_dirs=[root / "include", root / "csrc"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen",
                     "-Xs", f"-device {_DEV} -options -doubleGRF",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]

setup(
    name="prefill_dpas_rebuild",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
