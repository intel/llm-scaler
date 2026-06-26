"""Setup for custom_esimd_kernels (BMG ESIMD kernels for SGLang).

Build with:
    TORCH_XPU_ARCH_LIST=bmg CXX=icpx python3 setup.py build_ext --inplace -j 1

Modules built:
    custom_esimd_kernels_gemm        -- FP8 GEMM (M>=2) + INT4 GEMM (DPAS)
    custom_esimd_kernels_moe_batch   -- FP8 MoE (silu, e4m3 routed)
    custom_esimd_kernels_attn        -- sglang flat-NHD decode SDPA (split-K)
"""
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    # FP8 GEMM (M>=2) + INT4 GEMM
    SyclExtension(
        name="custom_esimd_kernels.custom_esimd_kernels_gemm",
        sources=[
            "csrc/xpu/esimd_kernel_gemm.sycl",
            "csrc/xpu/torch_extension_gemm.cc",
        ],
        include_dirs=[
            str(root / "include"),
            str(root / "csrc"),
            str(root / "csrc/xpu"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
    # MoE Batch (FP8 e4m3 + e5m2): builds blocks moe_router_forward,
    # moe_up_forward, moe_down_forward, moe_topk, moe_silu_mul, moe_accumulate
    # plus full-fused moe_forward_full / moe_forward_full_gelu_tanh.
    SyclExtension(
        name="custom_esimd_kernels.custom_esimd_kernels_moe_batch",
        sources=[
            "csrc/moe_batch/moe.sycl",
        ],
        include_dirs=[
            str(root / "csrc/moe_batch"),
            str(root / "csrc/xpu"),
            str(root / "csrc/xpu/esimd_kernels"),
            str(root / "csrc"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
    # Decode SDPA for sglang's flat NHD KV-cache layout (head_dim=256, GQA).
    # Targets Qwen3.5-MoE on BMG; one work-item per (batch, q_head), online
    # softmax, fp32 accumulate.
    SyclExtension(
        name="custom_esimd_kernels.custom_esimd_kernels_attn",
        sources=[
            "csrc/eagle/sglang_attn.sycl",
        ],
        include_dirs=[
            str(root / "csrc/eagle"),
            str(root / "csrc"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
]

setup(
    name="custom_esimd_kernels",
    version="0.2.0",
    packages=find_packages(where="python_v2"),
    package_dir={"": "python_v2"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.0.0",
    ],
)
