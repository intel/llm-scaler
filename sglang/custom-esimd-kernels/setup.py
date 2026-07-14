import sys
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_sglang.custom_esimd_kernels",
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
        name="custom_esimd_kernels_sglang.custom_esimd_kernels_lgrf",
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
        name="custom_esimd_kernels_sglang.custom_esimd_kernels_moe",
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
# [skip-ptl-fp8-gemm] ext_modules.append(
# [skip-ptl-fp8-gemm]     SyclExtension(
# [skip-ptl-fp8-gemm]         name="custom_esimd_kernels_sglang.custom_esimd_kernels_gemm",
# [skip-ptl-fp8-gemm]         sources=[
# [skip-ptl-fp8-gemm]             "csrc/xpu/esimd_kernel_gemm.sycl",
# [skip-ptl-fp8-gemm]             "csrc/xpu/torch_extension_gemm.cc",
# [skip-ptl-fp8-gemm]         ],
# [skip-ptl-fp8-gemm]         include_dirs=[
# [skip-ptl-fp8-gemm]             root / "include",
# [skip-ptl-fp8-gemm]             root / "csrc",
# [skip-ptl-fp8-gemm]         ],
# [skip-ptl-fp8-gemm]         extra_compile_args={
# [skip-ptl-fp8-gemm]             "cxx": ["-O3", "-std=c++17"],
# [skip-ptl-fp8-gemm]             "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-fp8-gemm]                      f"-I{torch_include}"],
# [skip-ptl-fp8-gemm]         },
# [skip-ptl-fp8-gemm]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-fp8-gemm]         py_limited_api=False,
# [skip-ptl-fp8-gemm]     )
# [skip-ptl-fp8-gemm] )
### FP8 GEMM kernels

### TopK V2 — vectorized softmax+topk for 512 experts (AOT for BMG)
# [skip-ptl-topk-v2] ext_modules.append(
# [skip-ptl-topk-v2]     SyclExtension(
# [skip-ptl-topk-v2]         name="custom_esimd_kernels_sglang.esimd_topk_v2",
# [skip-ptl-topk-v2]         sources=[
# [skip-ptl-topk-v2]             "csrc/xpu/esimd_kernel_topk_v2.sycl",
# [skip-ptl-topk-v2]             "csrc/xpu/torch_extension_topk_v2.cc",
# [skip-ptl-topk-v2]         ],
# [skip-ptl-topk-v2]         include_dirs=[
# [skip-ptl-topk-v2]             root / "include",
# [skip-ptl-topk-v2]             root / "csrc",
# [skip-ptl-topk-v2]         ],
# [skip-ptl-topk-v2]         extra_compile_args={
# [skip-ptl-topk-v2]             "cxx": ["-O3", "-std=c++17"],
# [skip-ptl-topk-v2]             "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-topk-v2]                      "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
# [skip-ptl-topk-v2]                      f"-I{torch_include}"],
# [skip-ptl-topk-v2]         },
# [skip-ptl-topk-v2]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-topk-v2]         py_limited_api=False,
# [skip-ptl-topk-v2]     )
# [skip-ptl-topk-v2] )
### TopK V2 kernels

### Eagle kernels (GDN + Page Attention) — from custom-esimd-kernels-vllm-eagle
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.eagle_ops",
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

### Grouped GGUF MoE GGEMV (Q4_K up + Q5_K/Q6_K down, doubleGRF DPAS).
# Used by sglang gguf.py for GGUF MoE prefill + MTP-verify (small-M N=16 occupancy
# tile). DPAS REQUIRES AOT for the actual GPU — JIT'ing DPAS on PTL is unreliable —
# so this ext is AOT to OMNI_XPU_DEVICE (default ptl-u Xe3) with -doubleGRF, unlike
# the JIT eagle_ops above. Ported from cc_workspace POC moe_q4k_prefill_poc.
_MOE_GROUPED_DEV = os.environ.get("OMNI_XPU_DEVICE", "ptl-u")
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.moe_grouped_gguf_xpu",
        sources=[
            "csrc/moe_grouped/moe_grouped_entry.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_grouped",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen",
                     "-Xs", f"-device {_MOE_GROUPED_DEV} -options -doubleGRF",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### Grouped GGUF MoE GGEMV

### Prefill DPAS SDPA (fp16, HD=256) — doubleGRF DPAS/XMX kernel.
# PTL-proven fp16 prefill SDPA that replaces the NaN-ing cutlass-sycl flash
# fp16 prefill path on Xe. DPAS/XMX kernel -> AOT for the actual GPU (JIT DPAS
# is unreliable) with -doubleGRF (heavy per-thread register state). Op namespace
# stays custom_esimd_kernels_vllm (matches the ported op + xpu_backend call).
_PREFILL_DPAS_DEV = os.environ.get("OMNI_XPU_DEVICE", "bmg")
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.custom_esimd_kernels_prefill_dpas",
        sources=[
            "csrc/xpu/esimd_kernel_prefill_dpas.sycl",
            "csrc/xpu/torch_extension_prefill_dpas.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen",
                     "-Xs", f"-device {_PREFILL_DPAS_DEV} -options -doubleGRF",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### Prefill DPAS SDPA

### MoE Batch kernels (Router, TopK, Up/Down, Accumulate) — FP8
# [skip-ptl-moe-batch] ext_modules.append(
# [skip-ptl-moe-batch]     SyclExtension(
# [skip-ptl-moe-batch]         name="custom_esimd_kernels_sglang.moe_ops",
# [skip-ptl-moe-batch]         sources=[
# [skip-ptl-moe-batch]             "csrc/moe_batch/moe.sycl",
# [skip-ptl-moe-batch]         ],
# [skip-ptl-moe-batch]         include_dirs=[
# [skip-ptl-moe-batch]             root / "csrc" / "moe_batch",
# [skip-ptl-moe-batch]         ],
# [skip-ptl-moe-batch]         extra_compile_args={
# [skip-ptl-moe-batch]             "cxx": ["-O3", "-std=c++20"],
# [skip-ptl-moe-batch]             "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-moe-batch]                      f"-I{torch_include}"],
# [skip-ptl-moe-batch]         },
# [skip-ptl-moe-batch]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-moe-batch]         py_limited_api=False,
# [skip-ptl-moe-batch]     )
# [skip-ptl-moe-batch] )
### MoE Batch kernels (FP8)

### MoE INT4 Batch kernels (Router, TopK, Up/Down, Finalize) — INT4
# [skip-ptl-moe-int4] ext_modules.append(
# [skip-ptl-moe-int4]     SyclExtension(
# [skip-ptl-moe-int4]         name="custom_esimd_kernels_sglang.moe_int4_ops",
# [skip-ptl-moe-int4]         sources=[
# [skip-ptl-moe-int4]             "csrc/moe_batch/moe_int4.sycl",
# [skip-ptl-moe-int4]         ],
# [skip-ptl-moe-int4]         include_dirs=[
# [skip-ptl-moe-int4]             root / "csrc" / "moe_batch",
# [skip-ptl-moe-int4]             root / "csrc" / "xpu" / "esimd_kernels",  # for moe_ops.h (TopK V2)
# [skip-ptl-moe-int4]             root / "csrc",  # for relative includes
# [skip-ptl-moe-int4]         ],
# [skip-ptl-moe-int4]         extra_compile_args={
# [skip-ptl-moe-int4]             "cxx": ["-O3", "-std=c++20"],
# [skip-ptl-moe-int4]             "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-moe-int4]                      f"-I{torch_include}"],
# [skip-ptl-moe-int4]         },
# [skip-ptl-moe-int4]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-moe-int4]         py_limited_api=False,
# [skip-ptl-moe-int4]     )
# [skip-ptl-moe-int4] )
### MoE INT4 Batch kernels

### MoE INT4 Prefill kernels (DPAS-based, for large-M prefill) — AOT BMG only
# [skip-ptl-moe-int4-prefill] ext_modules.append(
# [skip-ptl-moe-int4-prefill]     SyclExtension(
# [skip-ptl-moe-int4-prefill]         name="custom_esimd_kernels_sglang.moe_int4_prefill_ops",
# [skip-ptl-moe-int4-prefill]         sources=[
# [skip-ptl-moe-int4-prefill]             "csrc/moe_prefill/moe_prefill_int4.sycl",
# [skip-ptl-moe-int4-prefill]         ],
# [skip-ptl-moe-int4-prefill]         include_dirs=[
# [skip-ptl-moe-int4-prefill]             root / "csrc" / "moe_prefill",
# [skip-ptl-moe-int4-prefill]             root / "csrc" / "xpu" / "esimd_kernels",  # for moe_ops.h (TopK V2)
# [skip-ptl-moe-int4-prefill]             root / "csrc",
# [skip-ptl-moe-int4-prefill]         ],
# [skip-ptl-moe-int4-prefill]         extra_compile_args={
# [skip-ptl-moe-int4-prefill]             "cxx": ["-O3", "-std=c++20"],
# [skip-ptl-moe-int4-prefill]             "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-moe-int4-prefill]                      "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
# [skip-ptl-moe-int4-prefill]                      f"-I{torch_include}"],
# [skip-ptl-moe-int4-prefill]         },
# [skip-ptl-moe-int4-prefill]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-moe-int4-prefill]         py_limited_api=False,
# [skip-ptl-moe-int4-prefill]     )
# [skip-ptl-moe-int4-prefill] )
### MoE INT4 Prefill kernels

# ============================================================================
# Merged from the former custom-esimd-kernels (v2) package.
# These four extensions were previously shipped as a *separate* python package
# (`custom_esimd_kernels`) but are the ones actually built + used on the BMG
# fp8 image, so they are now folded into this single package. Each keeps its
# original TORCH_LIBRARY namespace (declared in the C++ sources):
#   custom_esimd_kernels_gemm      -> torch.ops.custom_esimd_kernels   (FP8/INT4 GEMM, M>=2)
#   moe_ops                        -> torch.ops.moe_ops                (FP8 MoE batch, silu routed)
#   moe_fp8_prefill_ops            -> torch.ops.moe_fp8_prefill_ops    (FP8 M-tiled DPAS MoE prefill)
#   custom_esimd_kernels_attn      -> torch.ops.sgl_esimd_attn + pybind (decode SDPA, flat NHD)
# NOTE: the gemm ext's v2 decls (esimd_gemm_int4_pgrp / *_bmg) have been merged
# into the shared include/kernel_ops.h, so all extensions now use one header.
# ============================================================================

### FP8 GEMM (M>=2) + INT4 GEMM (DPAS) — from v2
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.custom_esimd_kernels_gemm",
        sources=[
            "csrc/xpu/esimd_kernel_gemm.sycl",
            "csrc/xpu/torch_extension_gemm.cc",
        ],
        include_dirs=[
            root / "include",   # shared header (now incl. esimd_gemm_int4_pgrp, *_bmg)
            root / "csrc",
            root / "csrc" / "xpu",
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
### FP8/INT4 GEMM (v2)

### MoE Batch (FP8 e4m3+e5m2): moe_forward_full_silu_routed + building blocks — from v2
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.moe_ops",
        sources=[
            "csrc/moe_batch/moe.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_batch",
            root / "csrc" / "xpu",
            root / "csrc" / "xpu" / "esimd_kernels",
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
### MoE Batch (v2)

### FP8 M-tiled DPAS MoE prefill (e4m3+e5m2) — from v2
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.moe_fp8_prefill_ops",
        sources=[
            "csrc/moe_prefill/moe_prefill_fp8.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_prefill",
            root / "csrc" / "xpu",
            root / "csrc" / "xpu" / "esimd_kernels",
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
### FP8 MoE prefill (v2)

### Decode SDPA for sglang flat NHD KV-cache (head_dim=256, GQA) — from v2
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_sglang.custom_esimd_kernels_attn",
        sources=[
            "csrc/eagle/sglang_attn.sycl",
        ],
        include_dirs=[
            root / "csrc" / "eagle",
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
### Decode SDPA (v2)

setup(
    name="custom-esimd-kernels-sglang",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
