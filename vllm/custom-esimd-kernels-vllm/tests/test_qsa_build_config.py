import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa_build import (
    QSA_ABI_VERSION,
    QSA_ATTENTION_PAGE_SIZES,
    QSA_DEFINES,
    QSA_EXTENSION_NAME,
    QSA_FUSION_ABI_VERSION,
    QSA_FUSION_PAGE_SIZE,
    QSA_FUSION_WORKGROUP,
    QSA_GROUP_COMPRESSION_ABI_VERSION,
    QSA_INDEXER_POSTPROCESS_ABI_VERSION,
    QSA_PARALLEL_LANES,
    QSA_PARALLEL_ROWS_PER_WG,
    QSA_ROW_STORE_ABI_VERSION,
    QSA_ROW_STORE_PARALLEL_ABI_VERSION,
    QSA_ROW_STORE_RECEIPT_ABI_VERSION,
    QSA_SELECTION_PAGE_SIZES,
    make_qsa_extension,
)


def test_qsa_extension_name_is_stable():
    assert QSA_EXTENSION_NAME == "custom_esimd_kernels_vllm.qsa_ops"


def test_qsa_build_exposes_page_specialization_contract():
    assert QSA_ABI_VERSION == 2
    assert QSA_ROW_STORE_ABI_VERSION == 3
    assert QSA_ROW_STORE_RECEIPT_ABI_VERSION == 4
    assert QSA_ROW_STORE_PARALLEL_ABI_VERSION == 1
    assert QSA_PARALLEL_ROWS_PER_WG == 1
    assert QSA_PARALLEL_LANES == 16
    assert QSA_FUSION_ABI_VERSION == 1
    assert QSA_FUSION_PAGE_SIZE == 64
    assert QSA_FUSION_WORKGROUP == 1024
    assert QSA_INDEXER_POSTPROCESS_ABI_VERSION == 1
    assert QSA_GROUP_COMPRESSION_ABI_VERSION == 1
    assert QSA_SELECTION_PAGE_SIZES == (64, 128)
    assert QSA_ATTENTION_PAGE_SIZES == (256, 512)


def test_qsa_build_matches_validated_fp16_packed_contract():
    assert set(QSA_DEFINES) == {
        "QSA_PARALLEL_ROWS_PER_WG=1",
        "QSA_PARALLEL_LANES=16",
        "FUSED_WG_SIZE=1024",
        "FUSED_PAGE_CACHE=0",
        "FUSED_PAGE_CACHE_SLOTS=64",
        "FUSED_Q_LOCAL_FP32=0",
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
    }


def test_qsa_build_contains_attention_and_selection_sources():
    root = Path(__file__).resolve().parents[1]
    extension = make_qsa_extension(root, "/torch/include")

    assert extension.sources == [
        "csrc/qsa/qsa_sparse_attention.sycl",
        "csrc/qsa/qsa_select_paged_tokens.sycl",
        "csrc/qsa/qsa_store_cache_rows.sycl",
        "csrc/qsa/qsa_indexer_norm_rope.sycl",
        "csrc/qsa/qsa_q_norm_rope_select.sycl",
        "csrc/qsa/qsa_group_compression.sycl",
    ]
    module_source = (root / "csrc/qsa/qsa_sparse_attention.sycl").read_text()
    for required in (
        "qsa_store_cache_rows_r_aware_v1",
        "qsa_q_norm_rope_select_v1",
        "qsa_fusion_abi_version",
        "qsa_row_store_parallel_abi_version",
        "qsa_row_store_parallel_requires_unique_slots",
    ):
        assert required in module_source


def test_qsa_row_store_source_has_fixed_allocation_free_contract():
    root = Path(__file__).resolve().parents[1]
    source = (root / "csrc/qsa/qsa_store_cache_rows.sycl").read_text()

    assert "sycl::range<1>(1)" in source
    assert "store_cache_rows_v4" in source
    assert "receipt_ptr" in source
    assert "kReceiptUnwritten" in source
    assert "receipt_ptr[row * 2] = kReceiptUnwritten" in source
    assert "row_store_receipt_async_completion_required" in (
        root / "csrc/qsa/qsa_sparse_attention.sycl"
    ).read_text()
    assert "slot < 0 || slot >= capacity" in source
    assert "store_cache_rows_r_aware_v1" in source
    assert "QsaStoreCacheRowsParallelKernel" in source
    assert "unique_slots_proven" in source
    assert "sycl::range<1>(global)" in source
    for forbidden in (
        "std::vector",
        "at::empty",
        "at::zeros",
        ".wait(",
        ".wait_and_throw(",
        "synchronize(",
        "nonzero",
    ):
        assert forbidden not in source


def test_qsa_selection_workgroup_is_512():
    root = Path(__file__).resolve().parents[1]
    source = (root / "csrc/qsa/qsa_select_paged_tokens.sycl").read_text()
    assert "constexpr int64_t kWorkgroup = 512;" in source


def test_qsa_group_compression_source_has_fixed_decode_contract():
    root = Path(__file__).resolve().parents[1]
    source = (root / "csrc/qsa/qsa_group_compression.sycl").read_text()

    for required in (
        "group_compress_v1",
        "historical_ring_proven",
        "kCompressionRatio = 4",
        "kHeadDim = 128",
        "group compression",
        "caller-owned",
        "rope position cache",
        "packed_key_position_views_disjoint",
        "at::kBFloat16",
        "max_uint / key_element_size",
    ):
        assert required in source
    for forbidden in (
        "at::empty",
        "at::zeros",
        "std::vector",
        ".wait(",
        ".wait_and_throw(",
        "synchronize(",
        "index_copy",
        "store_cache",
    ):
        assert forbidden not in source
