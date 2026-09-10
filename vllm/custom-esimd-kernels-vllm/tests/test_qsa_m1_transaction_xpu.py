"""M1 side-cache transaction: exact bytes, all-or-nothing preflight, async."""

import pytest
import torch
from test_qsa_sparse_attention_xpu import _load_qsa_extension

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


@pytest.fixture(scope="module")
def ops():
    return _load_qsa_extension()


def case(dtype, slot=3, count=3, stride=1, page_size=128, ring_size=8):
    # Raw keys and position sidecar share one padded allocation, as in vLLM.
    packed = torch.zeros((2, ring_size, 280), dtype=torch.uint8, device="xpu")
    raw = packed[..., :256].view(dtype).unsqueeze(2)
    positions = packed[..., 256:].view(torch.int64).unsqueeze(2)
    compressed = torch.zeros((2, page_size, 1, 128), dtype=dtype, device="xpu")
    slots = torch.tensor([slot], dtype=torch.int64, device="xpu")
    key = torch.randint(-32768, 32767, (1, 128), dtype=torch.int16, device="xpu").view(
        dtype
    )
    position = torch.arange(3 * stride, dtype=torch.int64, device="xpu")[::stride].view(
        1, 1, 3
    )
    stores = ((raw, slots, key), (positions, slots, position))
    if count == 3:
        stores = ((compressed, slots, key.view(1, 1, 128)),) + stores
    return stores, (packed, compressed)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("slot", [-2, -1, 0, 3, 15, 16, 256])
@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("ring_size", [4, 8])
def test_exact_legacy_bytes(
    ops, dtype, slot, count, stride, fused, page_size, ring_size
):
    stores, backing = case(dtype, slot, count, stride, page_size, ring_size)
    # Save full backing incl untouched padding; NaNs must also copy bitwise.
    for cache, mapping, rows in stores:
        ops.qsa_store_cache_rows_v3(cache, mapping, rows)
    golden = [t.clone() for t in backing]
    for t in backing:
        t.zero_()
    assert ops.try_store_m1_transaction_v1(stores, fused) is True
    torch.xpu.synchronize()
    for actual, expected in zip(backing, golden):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("fused", [False, True])
def test_bad_last_store_does_not_mutate_first(ops, fused):
    stores, backing = case(torch.float16)
    cache, slots, _rows = stores[-1]
    bad = stores[:-1] + ((cache, slots, cache[0, 0]),)
    with pytest.raises(RuntimeError, match="alias"):
        ops.try_store_m1_transaction_v1(bad, fused)
    torch.xpu.synchronize()
    assert all(torch.count_nonzero(t) == 0 for t in backing)


@pytest.mark.parametrize("rows", [0, 2, 5, 8])
def test_unsupported_batch_is_side_effect_free(ops, rows):
    cache = torch.zeros((1, 8, 1, 128), dtype=torch.float16, device="xpu")
    source = torch.ones((rows, 128), dtype=cache.dtype, device="xpu")
    slots = torch.zeros(rows, dtype=torch.int64, device="xpu")
    assert ops.try_store_m1_transaction_v1(((cache, slots, source),), True) is False
    assert torch.count_nonzero(cache) == 0


@pytest.mark.parametrize("fallback", ["slot_int32", "convert_dtype", "strided_key"])
def test_valid_fallback_cases_decline_without_writes(ops, fallback):
    stores, backing = case(torch.float16)
    cache, slots, rows = stores[0]
    if fallback == "slot_int32":
        slots = slots.to(torch.int32)
    elif fallback == "convert_dtype":
        rows = rows.float()
    else:
        rows = torch.ones((1, 256), dtype=cache.dtype, device="xpu")[:, ::2]
    assert (
        ops.try_store_m1_transaction_v1(((cache, slots, rows),) + stores[1:], True)
        is False
    )
    torch.xpu.synchronize()
    assert all(torch.count_nonzero(t) == 0 for t in backing)


def test_cross_store_dependency_preserves_order(ops):
    a = torch.zeros((1, 8, 1, 128), dtype=torch.float16, device="xpu")
    b = torch.zeros_like(a)
    source = torch.ones((1, 128), dtype=a.dtype, device="xpu")
    slots = torch.zeros(1, dtype=torch.int64, device="xpu")
    stores = ((a, slots, source), (b, slots, a[0, 0]))
    assert ops.try_store_m1_transaction_v1(stores, True)
    assert torch.equal(b[0, 0], source)


@pytest.mark.parametrize("fused", [False, True])
def test_independent_storage_alias_is_rejected_before_writes(ops, fused):
    stores, backing = case(torch.float16)
    cache, slots, _rows = stores[0]
    alias = torch.from_dlpack(cache[0, 0])
    assert alias.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    with pytest.raises(RuntimeError, match="alias"):
        ops.try_store_m1_transaction_v1(stores[:-1] + ((cache, slots, alias),), fused)
    torch.xpu.synchronize()
    assert all(torch.count_nonzero(t) == 0 for t in backing)


def test_current_stream_order_no_shared_scratch(ops):
    parent = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    keepalive = []
    for index, stream in enumerate(streams):
        stream.wait_stream(parent)
        with torch.xpu.stream(stream):
            stores, backing = case(torch.float16)
            for value in range(1, 25):
                for cache, slots, rows in stores:
                    rows.fill_(value + index * 100)
                assert ops.try_store_m1_transaction_v1(stores, True)
            copies = [cache.clone() for cache, _, _ in stores]
            event = torch.xpu.Event()
            event.record()
            keepalive.append((stores, backing, copies, event, index))
    for stores, backing, copies, event, index in keepalive:
        event.synchronize()
        for copy in copies:
            assert torch.all(copy[0, 3] == 24 + index * 100)


@pytest.mark.parametrize("offset", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [129, 130, 132, 140])
def test_element_aligned_unaligned_cache_and_input(ops, offset, stride):
    backing = torch.zeros(8 * stride + offset, dtype=torch.float16, device="xpu")
    cache = backing[offset:].as_strided((1, 8, 1, 128), (8 * stride, stride, 128, 1))
    source = torch.arange(128 + offset, dtype=torch.float16, device="xpu")[
        offset:
    ].view(1, 128)
    slots = torch.tensor([2], dtype=torch.int64, device="xpu")
    assert ops.try_store_m1_transaction_v1(((cache, slots, source),), True)
    assert torch.equal(cache[0, 2, 0], source[0])
