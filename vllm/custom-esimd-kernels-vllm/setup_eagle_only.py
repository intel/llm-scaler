"""Build only the eagle_ops extension (incremental rebuild for fp8 KV work)."""
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension
import torch

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm.eagle_ops",
        sources=[
            "csrc/eagle/eagle.sycl",
        ],
        include_dirs=[
            root / "csrc" / "eagle",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
]

setup(
    name="custom_esimd_kernels_vllm_eagle_only",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
