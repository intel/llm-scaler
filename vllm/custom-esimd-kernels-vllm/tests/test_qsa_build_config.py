import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa_build import (
    QSA_ABI_VERSION,
    QSA_ATTENTION_PAGE_SIZES,
    QSA_DEFINES,
    QSA_EXTENSION_NAME,
    QSA_ROW_STORE_ABI_VERSION,
    QSA_SELECTION_PAGE_SIZES,
    make_qsa_extension,
)


def test_qsa_extension_name_is_stable():
    assert QSA_EXTENSION_NAME == "custom_esimd_kernels_vllm.qsa_ops"


def test_qsa_build_exposes_page_specialization_contract():
    assert QSA_ABI_VERSION == 2
    assert QSA_ROW_STORE_ABI_VERSION == 3
    assert QSA_SELECTION_PAGE_SIZES == (64, 128)
    assert QSA_ATTENTION_PAGE_SIZES == (256, 512)


def test_qsa_build_matches_validated_fp16_packed_contract():
    assert set(QSA_DEFINES) == {
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
    ]


def test_qsa_row_store_source_has_fixed_allocation_free_contract():
    root = Path(__file__).resolve().parents[1]
    source = (root / "csrc/qsa/qsa_store_cache_rows.sycl").read_text()

    assert "sycl::range<1>(1)" in source
    assert "slot < 0 || slot >= capacity" in source
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
