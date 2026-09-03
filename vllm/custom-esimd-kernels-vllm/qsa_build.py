"""Shared build definition for the fixed-contract Qwen3.8 QSA extension."""

from pathlib import Path

from torch.utils.cpp_extension import SyclExtension

QSA_EXTENSION_NAME = "custom_esimd_kernels_vllm.qsa_ops"
QSA_ABI_VERSION = 2
QSA_ROW_STORE_ABI_VERSION = 3
QSA_ROW_STORE_RECEIPT_ABI_VERSION = 4
QSA_ROW_STORE_PARALLEL_ABI_VERSION = 1
QSA_PARALLEL_ROWS_PER_WG = 1
QSA_PARALLEL_LANES = 16
QSA_INDEXER_POSTPROCESS_ABI_VERSION = 1
QSA_GROUP_COMPRESSION_ABI_VERSION = 1
QSA_SELECTION_PAGE_SIZES = (64, 128)
QSA_ATTENTION_PAGE_SIZES = (256, 512)
QSA_DEFINES = (
    "QSA_PARALLEL_ROWS_PER_WG=1",
    "QSA_PARALLEL_LANES=16",
    "QSA_NATIVE_ACTIVATION_FP16=1",
    "QSA_NATIVE_EXACT_PACKED_CACHE=1",
    "QSA_NATIVE_KV_TILE=4",
    "QSA_NATIVE_SUBGROUP=32",
    "QSA_NATIVE_TOKEN_LOADER=1",
    "QSA_NATIVE_TRIM_VALID_WIDTH=1",
    "QSA_NATIVE_PACKED_KV=1",
    "QSA_NATIVE_PIPELINED_LOADER=0",
    "QSA_NATIVE_STAGE_VALIDITY=0",
    "QSA_NATIVE_BOUNDED_SCAN=0",
    "QSA_NATIVE_SINGLE_EXP=0",
    "QSA_NATIVE_BLOCK_SOFTMAX=0",
    "QSA_NATIVE_FAST_EXP=1",
)


def make_qsa_extension(
    root: Path,
    torch_include: str,
    *,
    extension_name: str = QSA_EXTENSION_NAME,
    rpath: str = "$ORIGIN/../../torch/lib",
) -> SyclExtension:
    """Return a FP16 packed-cache QSA extension definition.

    Production keeps the canonical package name and package-relative RPATH.
    Isolated build-only variants can opt into a top-level name and matching
    RPATH without changing the production setup.
    """

    return SyclExtension(
        name=extension_name,
        sources=[
            "csrc/qsa/qsa_sparse_attention.sycl",
            "csrc/qsa/qsa_select_paged_tokens.sycl",
            "csrc/qsa/qsa_store_cache_rows.sycl",
            "csrc/qsa/qsa_indexer_norm_rope.sycl",
            "csrc/qsa/qsa_group_compression.sycl",
        ],
        include_dirs=[root / "csrc" / "qsa"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": [
                "-O3",
                "-fsycl-device-code-split=per_kernel",
                f"-I{torch_include}",
                *(f"-D{definition}" for definition in QSA_DEFINES),
            ],
        },
        extra_link_args=[f"-Wl,-rpath,{rpath}"],
        py_limited_api=False,
    )
