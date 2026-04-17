"""Build only the fused paged attention module."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension
import torch

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm.page_attn_fused_ops",
        sources=["csrc/eagle/page_attn_fused.sycl"],
        include_dirs=[root / "csrc" / "eagle"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]

setup(
    name="custom-esimd-kernels-vllm-fused",
    version="0.1.0",
    packages=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
