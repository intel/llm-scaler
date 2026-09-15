"""Correctness checks for the fixed-contract Qwen3.8 QSA XPU extension."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

HEADS = 3
Q6_HEADS = 6
HEAD_DIM = 256
INDEX_WIDTH = 2051
PACKED_STRIDES = {
    256: (131072, 512, 256, 1),
    512: (262144, 512, 256, 1),
}


def _xpu_available() -> bool:
    try:
        return torch.xpu.is_available() and torch.xpu.device_count() > 0
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA validation requires an XPU"
)


def _load_qsa_extension():
    if dso := os.environ.get("QSA_TEST_DSO"):
        spec = importlib.util.spec_from_file_location("qsa_ops", dso)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module("custom_esimd_kernels_vllm.qsa_ops")
    except ImportError:
        # Focused in-place builds may not contain the package's other modules.
        package_dir = (
            Path(__file__).resolve().parents[1]
            / "python"
            / "custom_esimd_kernels_vllm"
        )
        candidates = sorted(package_dir.glob("qsa_ops*.so"))
        if len(candidates) != 1:
            raise
        spec = importlib.util.spec_from_file_location("qsa_ops", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSA extension: {candidates[0]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="module")
def qsa_ops():
    return _load_qsa_extension()


def _canonicalize_singleton_dim_strides(tensor: torch.Tensor) -> torch.Tensor:
    strides = list(tensor.stride())
    previous_stride = 1
    changed = False
    for dim in range(tensor.dim() - 1, -1, -1):
        if tensor.shape[dim] == 1 and strides[dim] != previous_stride:
            strides[dim] = previous_stride
            changed = True
        previous_stride = strides[dim] * tensor.shape[dim]
    return tensor.as_strided(tensor.shape, strides) if changed else tensor


def _make_inputs(case: str, rows: int, page_size: int, heads: int = HEADS):
    pages_per_request = (INDEX_WIDTH + page_size - 1) // page_size
    pages = rows * pages_per_request
    packed_kv = torch.randn(
        pages,
        1,
        page_size,
        2 * HEAD_DIM,
        dtype=torch.float16,
        device="xpu",
    )
    k_cache, v_cache = packed_kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    k_cache = _canonicalize_singleton_dim_strides(k_cache)
    v_cache = _canonicalize_singleton_dim_strides(v_cache)
    q = 0.1 * torch.randn(
        rows, heads, HEAD_DIM, dtype=torch.float16, device="xpu"
    )
    logical_indices = torch.full(
        (rows, INDEX_WIDTH), -1, dtype=torch.int32, device="xpu"
    )
    block_table = torch.arange(
        pages, dtype=torch.int32, device="xpu"
    ).view(rows, pages_per_request)
    token_to_req = torch.arange(rows, dtype=torch.int32, device="xpu")

    if case == "empty":
        pass
    elif case == "holes_duplicates_pages":
        values = torch.tensor(
            [0, 511, -1, 512, 513, 512, -1, 1023, 1024, 1535],
            dtype=torch.int32,
            device="xpu",
        )
        for row in range(rows):
            logical_indices[row, : values.numel()] = torch.roll(values, row)
    elif case == "valid_width_32":
        logical_indices[:, :32] = torch.arange(
            32, dtype=torch.int32, device="xpu"
        )
    elif case == "full_width_2051":
        logical_indices[:] = torch.arange(
            INDEX_WIDTH, dtype=torch.int32, device="xpu"
        )
    else:
        raise ValueError(case)

    assert tuple(k_cache.stride()) == PACKED_STRIDES[page_size]
    assert tuple(v_cache.stride()) == PACKED_STRIDES[page_size]
    assert v_cache.storage_offset() == HEAD_DIM
    return (
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        page_size,
        torch.empty_like(q),
        packed_kv,
    )


def _reference(
    q, k_cache, v_cache, logical_indices, block_table, token_to_req, page_size
):
    output = torch.zeros_like(q)
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if logical.numel() == 0:
            continue
        request = int(token_to_req[row].item())
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets, 0]
        values = v_cache[pages, offsets, 0]
        scores = torch.einsum("hd,kd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * (HEAD_DIM**-0.5), dim=-1)
        output[row] = torch.einsum(
            "hk,kd->hd", probabilities, values.float()
        ).to(q.dtype)
    return output


def test_qsa_module_contract(qsa_ops):
    assert qsa_ops.qsa_abi_version == 2
    assert qsa_ops.qsa_token_split_candidate_abi_version == 3
    assert qsa_ops.qsa_token_split_candidate_max_rows == 128
    assert qsa_ops.qsa_token_split_candidate_page_size == 256
    assert qsa_ops.qsa_token_split_candidate_partial_count == 43
    assert qsa_ops.qsa_token_split_candidate_batch_opt_max_rows == 6
    assert qsa_ops.qsa_token_split_candidate_fused_abi_version == 1
    assert qsa_ops.qsa_token_split_candidate_fused_single_launch == 1
    assert qsa_ops.qsa_q6_abi_version == 1
    assert qsa_ops.qsa_q6_query_heads == Q6_HEADS
    assert qsa_ops.qsa_q6_kv_heads == 1
    assert qsa_ops.qsa_q6_token_split_v3 == 1
    assert qsa_ops.qsa_q6_page512 == 1
    assert callable(qsa_ops.sparse_paged_attention_q6_v1)
    assert callable(qsa_ops.sparse_attention_token_split_candidate_q6_v1)
    assert qsa_ops.qsa_row_store_abi_version == 3
    assert callable(qsa_ops.qsa_store_cache_rows_v3)
    assert qsa_ops.row_store_predicated_bounds == 1
    assert qsa_ops.row_store_duplicate_slots_last_write == 1
    assert qsa_ops.row_store_strided_rows == 1
    assert qsa_ops.row_store_strided_position_width == 1
    assert qsa_ops.row_store_storage_offsets == 1
    assert qsa_ops.row_store_inner_contiguous == 1
    assert qsa_ops.row_store_no_workspace == 1
    assert qsa_ops.row_store_fp16_width == 128
    assert qsa_ops.row_store_bf16_width == 128
    assert qsa_ops.row_store_int64_width == 3
    assert qsa_ops.selection_page_sizes == (64, 128)
    assert qsa_ops.attention_page_sizes == (256, 512)
    assert qsa_ops.activation_dtype == "float16"
    assert qsa_ops.exact_packed_cache == 1
    assert qsa_ops.kv_tile == 4
    assert qsa_ops.subgroup == 32
    assert qsa_ops.token_loader == 1
    assert qsa_ops.trim_valid_width == 1
    assert qsa_ops.packed_kv == 1
    assert qsa_ops.fast_exp == 1
    assert qsa_ops.selection_output_width == INDEX_WIDTH


@pytest.mark.parametrize("page_size", [256, 512])
@pytest.mark.parametrize(
    "operation", ["sparse_paged_attention_v2", "sparse_paged_attention_v3"]
)
@pytest.mark.parametrize(
    "case",
    ["empty", "holes_duplicates_pages", "valid_width_32", "full_width_2051"],
)
@pytest.mark.parametrize("rows", [1, 2])
def test_qsa_matches_reference_and_preserves_inputs(
    qsa_ops, case, rows, page_size, operation
):
    torch.manual_seed(20260828)
    args = list(_make_inputs(case, rows, page_size))
    packed_kv = args.pop()
    q, _, _, logical_indices, block_table, token_to_req, page_size, out = args
    snapshots = [
        q.clone(),
        packed_kv.clone(),
        logical_indices.clone(),
        block_table.clone(),
        token_to_req.clone(),
    ]
    expected = _reference(*args[:6], page_size)

    returned = getattr(qsa_ops, operation)(*args)
    torch.xpu.synchronize()

    assert q.dtype == torch.float16
    assert args[1].dtype == torch.float16
    assert args[2].dtype == torch.float16
    assert out.dtype == torch.float16
    assert tuple(args[1].stride()) == PACKED_STRIDES[page_size]
    assert tuple(args[2].stride()) == PACKED_STRIDES[page_size]
    assert args[2].storage_offset() == HEAD_DIM
    assert returned.data_ptr() == out.data_ptr()
    assert torch.isfinite(returned).all()
    assert torch.allclose(
        returned.float(), expected.float(), atol=2e-2, rtol=2e-2
    )
    for actual, snapshot in zip(
        [q, packed_kv, logical_indices, block_table, token_to_req], snapshots
    ):
        assert torch.equal(actual, snapshot)


def test_qsa_v3_v2_abi_reuses_scratch_across_rows_and_streams(qsa_ops):
    """Grow/reuse scratch with queued consumers, without per-call waits."""
    cases = [
        _make_inputs("holes_duplicates_pages", rows, 256)
        for rows in (1, 4, 2, 128, 1)
    ]
    expected = [_reference(*args[:7]) for args in cases]
    streams = [torch.xpu.current_stream(), torch.xpu.Stream()]
    streams[1].wait_stream(streams[0])
    results = []
    for stream in streams:
        with torch.xpu.stream(stream):
            for _ in range(12):
                for index, args in enumerate(cases):
                    out = torch.empty_like(args[0])
                    returned = qsa_ops.sparse_paged_attention_v3(*args[:7], out)
                    assert returned.data_ptr() == out.data_ptr()
                    results.append((index, out.clone()))
    torch.xpu.synchronize()
    for index, out in results:
        torch.testing.assert_close(out, expected[index], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("rows", [0, 129])
def test_qsa_v3_v2_abi_falls_back_outside_row_range(qsa_ops, rows):
    args = _make_inputs("valid_width_32", rows, 256)
    expected = qsa_ops.sparse_paged_attention_v2(*args[:8]).clone()
    out = torch.empty_like(args[0])
    returned = qsa_ops.sparse_paged_attention_v3(*args[:7], out)
    torch.xpu.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["v_storage", "output_alias", "indices_dtype"])
def test_qsa_v3_v2_abi_rejects_before_output_write(qsa_ops, invalid):
    args = list(_make_inputs("valid_width_32", 1, 256)[:8])
    if invalid == "v_storage":
        args[2] = _make_inputs("empty", 1, 256)[2]
    elif invalid == "output_alias":
        args[7] = args[0]
    else:
        args[3] = args[3].long()
    args[7].fill_(7)
    with pytest.raises(RuntimeError):
        qsa_ops.sparse_paged_attention_v3(*args)
    torch.xpu.synchronize()
    assert torch.all(args[7] == 7)


@pytest.mark.parametrize("case", ["holes_duplicates_pages", "full_width_2051"])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_qsa_token_split_candidate_matches_reference_for_m_rows(
    qsa_ops, case, rows
):
    """The v3 ABI must cover M=1 and M>1 with the same caller contract."""
    torch.manual_seed(20260906 + rows)
    args = list(_make_inputs(case, rows, 256))
    packed_kv = args.pop()
    q, k_cache, v_cache, logical, block_table, token_to_req, page_size, out = args
    partials = torch.empty(
        rows, 3, qsa_ops.qsa_token_split_candidate_partial_count, 258,
        dtype=torch.float32,
        device="xpu",
    )
    expected = _reference(
        q, k_cache, v_cache, logical, block_table, token_to_req, page_size
    )

    returned = qsa_ops.sparse_attention_token_split_candidate_v3(
        q,
        packed_kv,
        logical,
        block_table,
        token_to_req,
        page_size,
        out,
        partials,
    )
    # No host wait is allowed between phase 0 and phase 1 or between calls;
    # synchronize only at the observation boundary.
    assert returned.data_ptr() == out.data_ptr()
    torch.xpu.synchronize()
    assert torch.isfinite(returned).all()
    assert torch.allclose(
        returned.float(), expected.float(), atol=2e-2, rtol=2e-2
    )


def test_qsa_token_split_candidate_async_workspace_reuse(qsa_ops):
    """Back-to-back M>1 submissions remain correct on one in-order stream."""
    rows = 4
    args = list(_make_inputs("full_width_2051", rows, 256))
    packed_kv = args.pop()
    _, k_cache, v_cache, logical, block_table, token_to_req, page_size, _ = args
    partials = torch.empty(
        rows, 3, qsa_ops.qsa_token_split_candidate_partial_count, 258,
        dtype=torch.float32,
        device="xpu",
    )
    outputs = []
    expected = []
    for seed in (20260907, 20260908, 20260909):
        torch.manual_seed(seed)
        q = torch.randn(
            rows, HEADS, HEAD_DIM, dtype=torch.float16, device="xpu"
        )
        out = torch.empty_like(q)
        returned = qsa_ops.sparse_attention_token_split_candidate_v3(
            q,
            packed_kv,
            logical,
            block_table,
            token_to_req,
            page_size,
            out,
            partials,
        )
        assert returned.data_ptr() == out.data_ptr()
        outputs.append(out)
        expected.append(
            _reference(q, k_cache, v_cache, logical, block_table, token_to_req, page_size)
        )

    torch.xpu.synchronize()
    for actual, target in zip(outputs, expected):
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual.float(), target.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("page_size", [256, 512])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_qsa_q6_native_paths_match_independent_reference(qsa_ops, rows, page_size):
    torch.manual_seed(20260910 + rows + page_size)
    args = list(_make_inputs("holes_duplicates_pages", rows, page_size, Q6_HEADS))
    packed_kv = args.pop()
    q, k_cache, v_cache, logical, block_table, token_to_req, _, out = args
    expected = _reference(
        q, k_cache, v_cache, logical, block_table, token_to_req, page_size
    )

    sparse = qsa_ops.sparse_paged_attention_q6_v1(*args)
    partials = torch.full(
        (rows, Q6_HEADS, 43, 258),
        float("nan"),
        dtype=torch.float32,
        device="xpu",
    )
    split_out = torch.empty_like(q)
    split = qsa_ops.sparse_attention_token_split_candidate_q6_v1(
        q,
        packed_kv,
        logical,
        block_table,
        token_to_req,
        page_size,
        split_out,
        partials,
    )
    torch.xpu.synchronize()

    assert sparse.data_ptr() == out.data_ptr()
    assert split.data_ptr() == split_out.data_ptr()
    torch.testing.assert_close(sparse, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(split, expected, atol=2e-2, rtol=2e-2)
    active_partials = 43 if 2 <= rows <= 6 else 33
    assert torch.isfinite(partials[:, :, :active_partials]).all()
    assert torch.isnan(partials[:, :, active_partials:]).all()


@pytest.mark.parametrize("page_size", [256, 512])
def test_qsa_q6_tp4_rank_mapping_matches_global_24_head_reference(
    qsa_ops, page_size
):
    """TP4 ranks 0/1 share KV0 and ranks 2/3 share KV1."""
    torch.manual_seed(20260911 + page_size)
    rows = 2
    global_q = torch.randn(
        rows, 24, HEAD_DIM, dtype=torch.float16, device="xpu"
    )
    kv_inputs = [
        _make_inputs("holes_duplicates_pages", rows, page_size, Q6_HEADS)
        for _ in range(2)
    ]
    actual_rank_outputs = []
    expected_rank_outputs = []
    for rank in range(4):
        source = kv_inputs[rank // 2]
        local_q = global_q[:, rank * Q6_HEADS : (rank + 1) * Q6_HEADS].contiguous()
        out = torch.empty_like(local_q)
        actual_rank_outputs.append(
            qsa_ops.sparse_paged_attention_q6_v1(
                local_q, *source[1:6], page_size, out
            )
        )
        expected_rank_outputs.append(
            _reference(local_q, *source[1:6], page_size)
        )
    torch.xpu.synchronize()

    actual = torch.cat(actual_rank_outputs, dim=1)
    expected = torch.cat(expected_rank_outputs, dim=1)
    assert actual.shape == global_q.shape
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_qsa_q3_exports_reject_q6_without_relaxing_the_legacy_abi(qsa_ops):
    args = list(_make_inputs("valid_width_32", 1, 256, Q6_HEADS))
    with pytest.raises(RuntimeError, match=r"\[R,3,256\]"):
        qsa_ops.sparse_paged_attention_v2(*args[:8])
    partials = torch.empty((1, Q6_HEADS, 43, 258), dtype=torch.float32, device="xpu")
    with pytest.raises(RuntimeError, match=r"\[rows,3,43,258\]"):
        qsa_ops.sparse_attention_token_split_candidate_v3(
            args[0], args[8], *args[3:7], args[7], partials
        )


def test_qsa_rejects_bf16_query(qsa_ops):
    args = list(_make_inputs("valid_width_32", 1, 256)[:-1])
    args[0] = args[0].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="q must be float16"):
        qsa_ops.sparse_paged_attention_v2(*args)


def test_qsa_rejects_nonpacked_cache(qsa_ops):
    args = list(_make_inputs("valid_width_32", 1, 256)[:-1])
    args[1] = args[1].contiguous()
    args[2] = args[2].contiguous()
    with pytest.raises(RuntimeError, match="must have exact packed strides"):
        qsa_ops.sparse_paged_attention_v2(*args)
