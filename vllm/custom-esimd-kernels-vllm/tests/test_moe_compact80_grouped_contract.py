from pathlib import Path

SOURCE = (
    Path(__file__).parents[1]
    / "csrc"
    / "moe_batch"
    / "moe_int4.sycl"
)
GROUPED_NAME = (
    "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1"
)


def _source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _grouped_body(source: str) -> str:
    start = source.index("torch::Tensor " + GROUPED_NAME + "(")
    end = source.index("\ntorch::Tensor moe_tiny_fp16_shared_up(", start)
    return source[start:end]


def test_grouped_compact80_abi_is_additive_and_registered() -> None:
    source = _source_text()
    # declaration, schema string, and the two registration references
    assert source.count(GROUPED_NAME) == 4
    assert "MoeCompact80GroupedBuffers" in source
    assert "s_moe_compact80_grouped_buffers_by_stream" in source
    assert "tiled variant keeps two tokens" in source


def test_grouped_compact80_preflight_precedes_cache_and_submits() -> None:
    body = _grouped_body(_source_text())
    checks = body.index("check_moe_asymmetric_v1_tensor(")
    alias_check = body.index("moe_asymmetric_v1_tensors_overlap")
    cache = body.index("ensure_moe_compact80_grouped_buffers")
    topk = body.index("moe_topk_v2_host")
    assert checks < alias_check < cache < topk
    assert "compact80 grouped out v1 requires 2..8 tokens" in body
    assert "output must not alias any input tensor" in body


def test_grouped_compact80_reuses_expert_rows_and_preserves_old_m1_symbol() -> None:
    source = _source_text()
    assert "leader" in source
    assert "flat_routes = TOKEN_TILE * top_k" in source
    assert "moe compact80 grouped tiled up" in source

    m1_start = source.index(
        "torch::Tensor moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1("
    )
    hostchain_start = source.index(
        "torch::Tensor moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1(",
        m1_start,
    )
    assert GROUPED_NAME not in source[m1_start:hostchain_start]
