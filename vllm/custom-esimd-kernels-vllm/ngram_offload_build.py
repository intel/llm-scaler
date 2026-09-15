"""PLE pinned-host lookup 的独立构建定义，完整 wheel 与 build-only 共用。"""

from torch.utils.cpp_extension import SyclExtension


def make_ngram_offload_extension():
    return SyclExtension(
        name="custom_esimd_kernels_vllm.ngram_offload_ops",
        sources=["csrc/ngram/ngram_host_lookup.sycl"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-O3", "-fsycl-device-code-split=per_kernel"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
