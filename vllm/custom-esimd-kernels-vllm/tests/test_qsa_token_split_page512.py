"""QSA v3 page512 GPU 回归；由 main 在正式 E2E 结束后手动运行。"""

import pytest
import torch

from test_qsa_sparse_attention_xpu import (
    HEAD_DIM,
    HEADS,
    INDEX_WIDTH,
    _load_qsa_extension,
    _make_inputs,
    _reference,
    _xpu_available,
)


pytestmark = pytest.mark.skipif(
    not _xpu_available(), reason="QSA page512 验证需要 XPU"
)


@pytest.fixture(scope="module")
def qsa_ops():
    ops = _load_qsa_extension()
    # 旧 DSO 必须明确失败，不能把未覆盖 page512 的运行报告成通过。
    assert getattr(ops, "qsa_token_split_candidate_page512", 0) == 1
    return ops


def _inputs(rows, page_size, case="boundaries_holes_duplicates"):
    # 至多三个请求，共享物理 KV，覆盖多行属于同一请求的 MTP 形态。
    requests = min(rows, 3)
    args = list(_make_inputs("empty", requests, page_size))
    args[0] = 0.4 * torch.randn(
        rows, HEADS, HEAD_DIM, dtype=torch.float16, device="xpu"
    )
    args[3] = torch.full(
        (rows, INDEX_WIDTH), -1, dtype=torch.int32, device="xpu"
    )
    # 不改变物理 KV 内容；反转并轮转 page table，防止恒等映射掩盖寻址错误。
    table = args[4]
    table.copy_(torch.roll(table.flatten().flip(0), 2).view_as(table))
    args[5] = torch.tensor(
        [((2, 0, 2, 1)[row % 4] % requests) for row in range(rows)],
        dtype=torch.int32,
        device="xpu",
    )
    args[7] = torch.empty_like(args[0])
    logical = args[3]
    if case == "full_width":
        logical.copy_(torch.arange(INDEX_WIDTH, dtype=torch.int32, device="xpu"))
    elif case == "boundaries_holes_duplicates":
        # 同时跨 255/256、511/512 和更远页；在两种 partial 分界及尾部放 token。
        positions = torch.tensor(
            [0, 1, 2, 3, 4, 5, 46, 47, 48, 49, 62, 63, 64, 65,
             127, 128, 2047, 2048, 2049, 2050],
            device="xpu",
        )
        values = torch.tensor(
            [0, 254, 255, 256, 257, 510, 511, 512, 513, 512,
             255, 256, 511, 512, 1023, 1024, 1535, 1536, 2047, 2048],
            dtype=torch.int32,
            device="xpu",
        )
        for row in range(rows):
            logical[row, positions] = torch.roll(values, row)
        if rows > 1:
            logical[-1].fill_(-1)  # 同批次混合有效行和全 holes 行。
    elif case != "all_holes":
        raise ValueError(case)
    assert args[8].shape == (args[8].shape[0], 1, page_size, 512)
    assert args[8].stride(0) == page_size * 512
    return args


def _partials(rows):
    # NaN 初值能检测遗漏的写入；前后哨兵检测 caller-owned workspace 越界。
    storage = torch.full(
        (rows + 2, HEADS, 43, 258), float("nan"),
        dtype=torch.float32, device="xpu",
    )
    return storage, storage[1:-1]


def _launch(ops, args, partials):
    return ops.sparse_attention_token_split_candidate_v3(
        args[0], args[8], args[3], args[4], args[5], args[6], args[7], partials
    )


def _v2_reference(ops, args):
    # 必须调用原 sparse_paged_attention_v2；不能用 v3 或 token-split 自己作 golden。
    out = torch.empty_like(args[0])
    return ops.sparse_paged_attention_v2(*args[:7], out)


def test_page512_capability_preserves_legacy_abi(qsa_ops):
    assert qsa_ops.qsa_token_split_candidate_page512 == 1
    assert qsa_ops.qsa_token_split_candidate_page_size == 256
    assert qsa_ops.qsa_token_split_candidate_abi_version == 3
    assert qsa_ops.qsa_token_split_candidate_max_rows == 128
    assert qsa_ops.qsa_token_split_candidate_partial_count == 43
    assert qsa_ops.qsa_token_split_candidate_batch_opt_max_rows == 6
    assert qsa_ops.qsa_token_split_candidate_fused_abi_version == 1
    assert qsa_ops.qsa_token_split_candidate_fused_single_launch == 1


@pytest.mark.parametrize("page_size", [256, 512])
@pytest.mark.parametrize("rows", [*range(1, 9), 16, 64, 128])
@pytest.mark.parametrize(
    "case", ["boundaries_holes_duplicates", "full_width", "all_holes"]
)
def test_page512_matches_v2_and_math_reference(qsa_ops, page_size, rows, case):
    torch.manual_seed(20260907 + rows)
    args = _inputs(rows, page_size, case)
    storage, partials = _partials(rows)
    snapshots = [args[i].clone() for i in (0, 3, 4, 5, 8)]
    expected_v2 = _v2_reference(qsa_ops, args)
    expected_math = _reference(*args[:7])
    returned = _launch(qsa_ops, args, partials)
    assert returned.data_ptr() == args[7].data_ptr()
    torch.xpu.synchronize()

    assert torch.isfinite(returned).all()
    torch.testing.assert_close(expected_v2, expected_math, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(returned, expected_v2, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(returned, expected_math, atol=2e-3, rtol=2e-3)
    for index, snapshot in zip((0, 3, 4, 5, 8), snapshots):
        assert torch.equal(args[index], snapshot)
    assert torch.isnan(storage[0]).all()
    assert torch.isnan(storage[-1]).all()
    active_partials = 43 if 2 <= rows <= 6 else 33
    assert torch.isfinite(partials[:, :, :active_partials]).all()
    assert torch.isnan(partials[:, :, active_partials:]).all()
    if case == "all_holes":
        assert torch.count_nonzero(returned) == 0
        assert torch.count_nonzero(partials[:, :, :active_partials, 1:]) == 0
    elif case == "boundaries_holes_duplicates" and rows > 1:
        assert torch.count_nonzero(returned[-1]) == 0


def test_page512_current_stream_producer_consumer_and_workspace_reuse(qsa_ops):
    torch.manual_seed(20260908)
    # 跨 M1..8 和两种 page size，phase1/workspace 复用不能读上次调用残留。
    cases = [_inputs(rows, size) for size in (512, 256) for rows in range(1, 9)]
    expected = [_v2_reference(qsa_ops, args) for args in cases]
    initial_stream = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    results = []
    owned_buffers = []
    for stream in streams:
        stream.wait_stream(initial_stream)
        with torch.xpu.stream(stream):
            storage, workspace = _partials(8)
            query = torch.empty_like(cases[7][0])
            output = torch.empty_like(query)
            owned_buffers.append((storage, workspace, query, output))
            for _ in range(2):
                for index, args in enumerate(cases):
                    rows = args[0].shape[0]
                    queued_args = list(args)
                    queued_args[0] = query[:rows]
                    queued_args[7] = output[:rows]
                    # producer、两次 kernel submit、consumer 间没有 host 等待。
                    query[:rows].copy_(args[0])
                    workspace.fill_(float("nan"))
                    returned = _launch(qsa_ops, queued_args, workspace[:rows])
                    assert returned.data_ptr() == output.data_ptr()
                    results.append((index, returned.clone()))
    torch.xpu.synchronize()
    for index, actual in results:
        torch.testing.assert_close(actual, expected[index], atol=2e-3, rtol=2e-3)
    for storage, _, _, _ in owned_buffers:
        assert torch.isnan(storage[0]).all()
        assert torch.isnan(storage[-1]).all()


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize(
    "layout",
    ["only_first", "first_each", "middle_each", "last_each",
     "tail_first", "tail_middle", "tail_last", "duplicate_islands"],
)
def test_page512_empty_partial_extreme_slot_layouts(qsa_ops, rows, layout):
    torch.manual_seed(20260909 + rows)
    args = _inputs(rows, 512, "all_holes")
    logical = args[3]
    logical.fill_(-7)  # 所有负 logical 都是 hole，不局限于 -1。
    tokens_per_partial = 48 if 2 <= rows <= 6 else 64
    starts = list(range(0, INDEX_WIDTH, tokens_per_partial))
    positions = {
        "only_first": [0],
        "first_each": starts,
        "middle_each": [
            start + min(tokens_per_partial, INDEX_WIDTH - start) // 2
            for start in starts
        ],
        "last_each": [
            min(start + tokens_per_partial, INDEX_WIDTH) - 1 for start in starts
        ],
        "tail_first": [starts[-1]],
        "tail_middle": [starts[-1] + (INDEX_WIDTH - starts[-1]) // 2],
        "tail_last": [INDEX_WIDTH - 1],
        "duplicate_islands": [
            1, tokens_per_partial // 2, tokens_per_partial + 1, INDEX_WIDTH - 2
        ],
    }[layout]
    for index, slot in enumerate(positions):
        # logical=0 必须判为有效；重复 token 横跨 partial，末 slot 可仍是 hole。
        logical[:, slot] = (0, 511, 0, 512)[index % 4]
    expected_v2 = _v2_reference(qsa_ops, args)
    expected_math = _reference(*args[:7])
    storage, partials = _partials(rows)
    returned = _launch(qsa_ops, args, partials)
    torch.xpu.synchronize()

    torch.testing.assert_close(returned, expected_v2, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(returned, expected_math, atol=2e-3, rtol=2e-3)
    occupied = {slot // tokens_per_partial for slot in positions}
    for partial_id in range(len(starts)):
        state = partials[:, :, partial_id]
        assert torch.isfinite(state).all()
        if partial_id in occupied:
            assert torch.all(state[:, :, 1] > 0)
        else:
            assert torch.all(state[:, :, 0] == -1.0e30)
            assert torch.count_nonzero(state[:, :, 1:]) == 0
    assert torch.isnan(partials[:, :, len(starts):]).all()
    assert torch.isnan(storage[0]).all()
    assert torch.isnan(storage[-1]).all()


@pytest.mark.parametrize("rows", [1, 5, 8])
def test_page512_empty_partial_overwrites_reused_state(qsa_ops, rows):
    args = _inputs(rows, 512, "all_holes")
    expected_empty = _v2_reference(qsa_ops, args)
    # 最后一行之后的非负 guard 不属于 logical tensor；复用非空 workspace
    # 后必须完整覆盖空 partial。保留原 v3 的有限 Q/K/V 输入契约。
    logical_storage = torch.zeros(
        rows * INDEX_WIDTH + 64, dtype=torch.int32, device="xpu"
    )
    args[3] = logical_storage[:rows * INDEX_WIDTH].view(rows, INDEX_WIDTH)
    storage, partials = _partials(rows)
    initial_stream = torch.xpu.current_stream()
    stream = torch.xpu.Stream()
    stream.wait_stream(initial_stream)
    with torch.xpu.stream(stream):
        # 所有 slot=0，先写满真实非空状态，再复用同一 workspace 为全 holes。
        _launch(qsa_ops, args, partials)
        nonempty_state = partials.clone()
        args[3].fill_(-7)
        returned = _launch(qsa_ops, args, partials)
        consumed = returned.clone()
    torch.xpu.synchronize()

    active = 43 if 2 <= rows <= 6 else 33
    assert torch.all(nonempty_state[:, :, :active, 1] > 0)
    torch.testing.assert_close(consumed, expected_empty, atol=0, rtol=0)
    assert torch.all(partials[:, :, :active, 0] == -1.0e30)
    assert torch.count_nonzero(partials[:, :, :active, 1:]) == 0
    assert torch.isnan(partials[:, :, active:]).all()
    assert torch.isnan(storage[0]).all()
    assert torch.isnan(storage[-1]).all()
    assert torch.count_nonzero(logical_storage[rows * INDEX_WIDTH:]) == 0


def test_fused_v4_still_rejects_page512(qsa_ops):
    args = _inputs(1, 512)
    _, partials = _partials(1)
    args[7].fill_(7)
    counters = torch.zeros(1, 3, dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="page_size=256"):
        qsa_ops.sparse_attention_token_split_candidate_v4(
            args[0], args[8], args[3], args[4], args[5], 512,
            args[7], partials, counters, 1,
        )
    torch.xpu.synchronize()
    assert torch.all(args[7] == 7)


@pytest.mark.parametrize(
    "invalid",
    ["page_size", "page_shape", "page_stride", "partials", "dtype", "rows"],
)
def test_page512_contract_rejects_before_write(qsa_ops, invalid):
    args = _inputs(129 if invalid == "rows" else 2, 512)
    storage, partials = _partials(args[0].shape[0])
    args[7].fill_(7)
    if invalid == "page_size":
        args[6] = 128
    elif invalid == "page_shape":
        args[8] = args[8][:, :, :256, :]
    elif invalid == "page_stride":
        args[8] = torch.empty_strided(
            args[8].shape, (512 * 512 + 512, 512 * 512, 512, 1),
            dtype=torch.float16, device="xpu",
        )
    elif invalid == "partials":
        partials = partials[:, :, :42, :].contiguous()
    elif invalid == "dtype":
        args[3] = args[3].long()
    with pytest.raises(RuntimeError):
        _launch(qsa_ops, args, partials)
    torch.xpu.synchronize()
    assert torch.all(args[7] == 7)
    assert torch.isnan(storage).all()
