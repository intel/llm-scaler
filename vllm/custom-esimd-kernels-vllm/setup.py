import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels",
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
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]

### for lgrf esimd kernels (GDN conv fused — separate module, doubleGRF)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_lgrf",
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
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### for lgrf esimd kernels

### MoE auxiliary kernels — no DPAS, standard compilation
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_moe",
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
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE auxiliary kernels

### FP8 GEMM (M>1) — uses DPAS, compile with JIT only (no AOT to avoid device mismatch)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_gemm",
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
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### FP8 GEMM kernels

### TopK V2 — vectorized softmax+topk for 512 experts (AOT for BMG)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.esimd_topk_v2",
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
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### TopK V2 kernels

### Eagle kernels (GDN + Page Attention) — from custom-esimd-kernels-vllm-eagle
ext_modules.append(
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
    )
)
### Eagle kernels

### MoE Batch kernels (Router, TopK, Up/Down, Accumulate) — FP8
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.moe_ops",
        sources=[
            "csrc/moe_batch/moe.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_batch",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE Batch kernels (FP8)

### MoE INT4 Batch kernels (Router, TopK, Up/Down, Finalize) — INT4
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.moe_int4_ops",
        sources=[
            "csrc/moe_batch/moe_int4.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_batch",
            root / "csrc" / "xpu" / "esimd_kernels",  # for moe_ops.h (TopK V2)
            root / "csrc",  # for relative includes
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE INT4 Batch kernels

### MoE INT4 Prefill kernels (DPAS-based, for large-M prefill) — AOT BMG only
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.moe_int4_prefill_ops",
        sources=[
            "csrc/moe_prefill/moe_prefill_int4.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_prefill",
            root / "csrc" / "xpu" / "esimd_kernels",  # for moe_ops.h (TopK V2)
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE INT4 Prefill kernels

setup(
    name="custom-esimd-kernels-vllm",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
