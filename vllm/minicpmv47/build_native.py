# SPDX-License-Identifier: Apache-2.0
"""Rebuild fusion, TP2 collective, and attention overlays against the base ABI."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path


def run(command, cwd=None, env=None):
    print(shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def configure(source: Path, build: Path, attention: bool):
    recipe = Path(__file__).resolve().parent
    env = os.environ.copy()
    if not attention:
        # Matches the JIT SPIR-V fusion overlay; attention keeps Xe2 AOT.
        env["VLLM_XPU_AOT_DEVICES"] = ""
        env["VLLM_XPU_XE2_AOT_DEVICES"] = ""
    switches = {
        "VLLM_XPU_ENABLE_XE2": True,
        "VLLM_XPU_ENABLE_XE_DEFAULT": False,
        "VLLM_XPU_ENABLE_XE3P": False,
        "VLLM_XPU_ENABLE_ONEDNN": False,
        "BASIC_KERNELS_ENABLED": False,
        "FA2_KERNELS_ENABLED": attention,
        "MOE_KERNELS_ENABLED": not attention,
        "XPU_SPECIFIC_KERNELS_ENABLED": not attention,
        "GDN_KERNELS_ENABLED": False,
        "MQA_LOGITS_KERNELS_ENABLED": False,
        "MHC_KERNELS_ENABLED": False,
        "XPUMEM_ALLOCATOR_ENABLED": False,
    }
    command = [
        "cmake",
        "-S",
        source,
        "-B",
        build,
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', 'icpx')}",
        f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
        f"-DFETCHCONTENT_BASE_DIR={build.parent / 'deps'}",
        f"-DVLLM_CHUNK_PREFILL_CONFIG={recipe / 'chunk-prefill.conf'}",
        f"-DVLLM_PAGED_DECODE_CONFIG={recipe / 'paged-decode.conf'}",
    ]
    if attention:
        # Give the rebuilt SYCL kernels distinct IDs from the stock binaries.
        command.append("-DCMAKE_CXX_FLAGS=-Dvllm_xpu_xe2=minicpmv47_xe2")
    command += [
        f"-D{name}={'ON' if enabled else 'OFF'}" for name, enabled in switches.items()
    ]
    run(command, env=env)


def commands(build: Path):
    return json.loads((build / "compile_commands.json").read_text())


def command_for(rows, source: Path):
    return next(row for row in rows if Path(row["file"]).resolve() == source.resolve())


def compile_object(row, source: Path, output: Path, build: Path, extra=()):
    output.parent.mkdir(parents=True, exist_ok=True)
    arguments = shlex.split(row["command"])
    arguments[arguments.index("-o") + 1] = str(output)
    arguments[arguments.index("-c") + 1] = str(source)
    arguments[1:1] = list(extra)
    run(arguments, cwd=build)


def fusion_bindings(source: Path) -> str:
    bindings = (source / "csrc/xpu/torch_bindings.cpp").read_text()
    pieces = []
    start = bindings.index("  // One-shot P2P custom all-reduce")
    pieces.append(bindings[start : bindings.index("\n#ifdef", start)])
    for name in ("fp8_gemm_w8a16_pair", "moe_shared_fused_decode_interface"):
        start = bindings.index(f'  xpu_ops.def(\n      "{name}')
        pieces.append(bindings[start : bindings.index("\n#endif", start)])
    start = bindings.index('  xpu_ops.def(\n      "qkv_split_norm_rope')
    pieces.append(bindings[start : bindings.index("\n}", start)])
    router = (source / "csrc/moe/torch_bindings.cpp").read_text()
    start = router.index('  m.def(\n      "router_gemv_topk_softmax')
    router = router[start : router.index("  // Apply topk sigmoid", start)]
    return (
        """#include <torch/library.h>
#include "xpu/ops.h"
#include "xpu/gemv/fp8_gemv_interface.h"
#include "xpu/grouped_gemm/moe_shared_fused_interface.h"
#include "moe/moe_ops.h"
TORCH_LIBRARY_FRAGMENT(_xpu_C, xpu_ops) {
"""
        + "\n".join(pieces)
        + "\n}\nTORCH_LIBRARY_FRAGMENT(_moe_C, m) {\n"
        + router
        + """
}
torch::Tensor minicpm_fp8_dispatch(
    const torch::Tensor& a, const torch::Tensor& b,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias) {
  if (auto out = vllm::fp8_gemv::try_fp8_gemv_w8a16(a, b, scale, bias))
    return *out;
  // Call the stock oneDNN symbol, avoiding dispatcher recursion.
  return ::fp8_gemm_w8a16(a, b, scale, bias);
}
TORCH_LIBRARY_IMPL(_xpu_C, XPU, m) {
  m.impl("fp8_gemm_w8a16", &minicpm_fp8_dispatch);
}
"""
    )


def build_fusions(source: Path, build: Path, output: Path):
    rows = commands(build)
    binding_source = build / "minicpm_fusion_bindings.cpp"
    binding_source.write_text(fusion_bindings(source))
    template = command_for(rows, source / "csrc/moe/moe_router_decode.cpp")
    objects = []
    selected = [
        "csrc/moe/moe_router_decode.cpp",
        "csrc/xpu/custom_ar/custom_ar.cpp",
        "csrc/xpu/gemv/xe_2/fp8_gemv_xe2.cpp",
        "csrc/xpu/gemv/xe_2/norm_fp8_gemv_xe2.cpp",
        "csrc/xpu/grouped_gemm/xe_2/moe_shared_fused_decode_xe2.cpp",
        "csrc/xpu/grouped_gemm/moe_shared_fused_interface.cpp",
        "csrc/xpu/sycl/qkv_split_norm_rope.cpp",
    ]
    for relative in selected:
        path = source / relative
        target = build / "overlay_objects" / (path.stem + ".o")
        compile_object(command_for(rows, path), path, target, build)
        objects.append(target)
    target = build / "overlay_objects/minicpm_fusion_bindings.o"
    compile_object(template, binding_source, target, build, ["-I" + str(source)])
    objects.append(target)
    torchlib = Path(distribution("torch").locate_file("torch/lib"))
    kernelroot = Path(distribution("vllm-xpu-kernels").locate_file("vllm_xpu_kernels"))
    stock = next(kernelroot.glob("_xpu_C*.so"))
    compiler = shlex.split(template["command"])[0]
    run(
        [
            compiler,
            "-shared",
            "-fsycl",
            "-fsycl-max-parallel-link-jobs=16",
            "-flink-huge-device-code",
            "-Xspirv-translator",
            "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
            "-o",
            output / "bf16-performance.so",
            *objects,
            stock,
            "-L" + str(torchlib),
            "-Wl,-rpath," + str(torchlib) + ":" + str(kernelroot),
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_xpu",
            "-lc10",
            "-lc10_xpu",
            "-lze_loader",
        ]
    )


def link_command(build: Path, target: str):
    lines = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", target], text=True
    ).splitlines()
    arguments = shlex.split(lines[-1])
    start = next(i for i, arg in enumerate(arguments) if Path(arg).name == "icpx")
    end = arguments.index("&&", start) if "&&" in arguments[start:] else len(arguments)
    return arguments[start:end]


def build_attention(source: Path, build: Path, output: Path, jobs: int):
    run(["cmake", "--build", build, "--target", "attn_kernels_xe_2", "-j", str(jobs)])
    for row in commands(build):
        path = Path(row["file"])
        if (
            path.parent == source / "csrc/flash_attn"
            or path == source / "csrc/xpu/attn/attn_interface.cpp"
        ):
            arguments = shlex.split(row["command"])
            arguments = [
                arg
                for arg in arguments
                if not arg.startswith("-DTORCH_EXTENSION_NAME=")
            ]
            arguments.insert(1, "-DTORCH_EXTENSION_NAME=_minicpm_dspark_fa2")
            target = build / arguments[arguments.index("-o") + 1]
            target.parent.mkdir(parents=True, exist_ok=True)
            run(arguments, cwd=build)
    native_name = "libminicpm_dspark_attn_kernels.so"
    arguments = link_command(build, "attn_kernels_xe_2")
    arguments[arguments.index("-o") + 1] = native_name
    arguments = [
        "-Wl,-soname," + native_name if arg.startswith("-Wl,-soname,") else arg
        for arg in arguments
    ]
    arguments.insert(1, "-Wl,-Bsymbolic")
    run(arguments, cwd=build)
    torchlib = Path(distribution("torch").locate_file("torch/lib"))
    arguments = link_command(build, "_vllm_fa2_C")
    arguments[arguments.index("-o") + 1] = str(output / "_minicpm_dspark_fa2.abi3.so")
    arguments = [
        native_name if arg == "libattn_kernels_xe_2.so" else arg for arg in arguments
    ]
    arguments = [
        "-Wl,-rpath,$ORIGIN:" + str(torchlib) if arg.startswith("-Wl,-rpath,") else arg
        for arg in arguments
    ]
    arguments.insert(1, "-Wl,-Bsymbolic")
    run(arguments, cwd=build)
    shutil.copy2(build / native_name, output / native_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    build = arguments.build.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure(source, build / "fusions", attention=False)
    build_fusions(source, build / "fusions", output)
    configure(source, build / "attention", attention=True)
    build_attention(source, build / "attention", output, arguments.jobs)
