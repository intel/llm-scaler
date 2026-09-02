from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa_build import (  # noqa: E402
    QSA_DEFINES,
    QSA_EXTENSION_NAME,
    make_qsa_extension,
)


def test_qsa_extension_name_is_stable():
    assert QSA_EXTENSION_NAME == "custom_esimd_kernels_vllm.qsa_ops"


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
    ]
