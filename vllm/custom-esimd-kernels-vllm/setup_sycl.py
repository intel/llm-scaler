import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import SyclExtension

# The full PyTorch architecture list can exceed icpx's device-link source
# location limit for this large build-only artifact.  Keep BMG as the
# reproducible default for the target cards, while allowing an explicit
# TORCH_XPU_ARCH_LIST override for another device family.
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "bmg-g31")

from esimd_build_extention import BuildExtension
from qsa_build import make_qsa_extension

root = Path(__file__).parent.resolve()

import torch
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_sycl_only",
        sources=[
            "csrc/xpu/esimd_kernel.sycl",
            "csrc/xpu/torch_extension.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
]

### for lgrf esimd kernels (GDN conv fused — separate module, doubleGRF)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_lgrf_sycl_only",
        sources=[
            "csrc/xpu/esimd_kernel_lgrf.sycl",
            "csrc/xpu/torch_extension_lgrf.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
)
### for lgrf esimd kernels

### MoE auxiliary kernels — no DPAS, standard compilation
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_moe_sycl_only",
        sources=[
            "csrc/xpu/esimd_kernel_moe.sycl",
            "csrc/xpu/torch_extension_moe.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
)
### MoE auxiliary kernels

### FP8 GEMM (M>1) — uses DPAS, compile with JIT only (no AOT to avoid device mismatch)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_gemm_sycl_only",
        sources=[
            "csrc/xpu/esimd_kernel_gemm.sycl",
            "csrc/xpu/torch_extension_gemm.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
)
### FP8 GEMM kernels

### TopK V2 — vectorized softmax+topk for 512 experts (AOT for BMG)
ext_modules.append(
    SyclExtension(
        name="esimd_topk_v2_sycl_only",
        sources=[
            "csrc/xpu/esimd_kernel_topk_v2.sycl",
            "csrc/xpu/torch_extension_topk_v2.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
)
### TopK V2 kernels

### Eagle kernels (GDN + Page Attention) — from custom-esimd-kernels-vllm-eagle
ext_modules.append(
    SyclExtension(
        name="eagle_ops_sycl_only",
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
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
)
### Eagle kernels

### MoE Batch kernels (Router, TopK, Up/Down, Accumulate) — from custom-esimd-kernels-vllm-moe-batch-test
ext_modules.append(
    SyclExtension(
        name="moe_ops_sycl_only",
        sources=[
            "csrc/moe_batch/moe.sycl",
        ],
        include_dirs=[],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/torch/lib"],
        py_limited_api=False,
    )
)
### MoE Batch kernels

### Qwen3.8 TP8-rank sparse paged attention — FP16 packed-cache ABI
ext_modules.append(
    make_qsa_extension(
        root,
        torch_include,
        extension_name="qsa_ops_sycl_only",
        rpath="$ORIGIN/torch/lib",
    )
)
### Qwen3.8 QSA kernel

setup(
    name="custom-esimd-kernels-vllm-sycl-only",
    version="0.1.0",
    # This is a build-only artifact set.  It must not install the production
    # Python package or extensions under its canonical module names.
    packages=[],
    py_modules=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
