"""Native fixed-cache decode: independent FP32 reference and device lifecycle."""
import pytest
import torch
from omni_xpu_kernel import kitchen

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")


def reference(q, k, v, lengths, causal=True):
    b, h, s, d = q.shape
    groups = h // k.shape[1]
    lengths = lengths.long().clamp(0, k.shape[2])
    valid = torch.arange(k.shape[2], device=q.device)[None, :] < lengths[:, None]
    k = k.float().masked_fill(~valid[:, None, :, None], 0).repeat_interleave(groups, 1)
    v = v.float().masked_fill(~valid[:, None, :, None], 0).repeat_interleave(groups, 1)
    scores = q.float() @ k.transpose(-1, -2) * d**-0.5
    limit = lengths[:, None, None, None]
    if causal:
        limit = limit - s + 1 + torch.arange(s, device=q.device)[None, None, :, None]
    scores.masked_fill_(torch.arange(k.shape[2], device=q.device) >= limit, -torch.inf)
    lse = scores.logsumexp(-1)
    p = torch.exp(scores - lse.masked_fill(torch.isneginf(lse), 0)[..., None])
    out = (p @ v).transpose(1, 2).reshape(b, s, h*d).to(q.dtype)
    return out, lse


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("seq", [1, 4, 8])
@pytest.mark.parametrize("ratio", [1, 2, 5])
def test_native_decode_staircase_strides_and_empty_rows(dim, seq, ratio):
    torch.manual_seed(151)
    q = torch.randn(2, 2*ratio, seq, dim, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(2, 513, 2, dim, device="xpu", dtype=q.dtype).transpose(1, 2)
    v = torch.randn_like(k)
    for values in ([-3, 529], [seq-1, 259]):
        length = torch.tensor(values, device="xpu", dtype=torch.int64)
        actual, lse = kitchen.decode_attention(q, k, v, length)
        expected, el = reference(q, k, v, length)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(lse, el, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seq", [1, 4, 8])
def test_native_step_merge_and_output_alias(seq):
    torch.manual_seed(157)
    q = torch.randn(2, 6, seq, 256, device="xpu", dtype=torch.bfloat16)
    pk = torch.randn(2, 2, 257, 256, device="xpu", dtype=q.dtype)
    pv = torch.randn_like(pk)
    k = torch.randn(2, 2, seq, 256, device="xpu", dtype=q.dtype)
    v = torch.randn_like(k)
    lens = torch.tensor([0, 257], device="xpu", dtype=torch.int32)
    out, lse = kitchen.decode_attention(q, pk, pv, lens, False)
    expected = []
    for b, n in enumerate([0, 257]):
        kk = torch.cat([pk[b:b+1, :, :n], k[b:b+1]], 2)
        vv = torch.cat([pv[b:b+1, :, :n], v[b:b+1]], 2)
        expected.append(reference(q[b:b+1], kk, vv, lens[b:b+1]*0+n+seq)[0])
    assert kitchen.decode_step_merge(out, lse, q, k, v, out) is None
    torch.testing.assert_close(out, torch.cat(expected), rtol=0.01, atol=0.003)


def test_native_decode_stream_and_graph_device_lengths():
    q = torch.randn(1, 4, 4, 256, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 513, 256, device="xpu", dtype=q.dtype)
    v = torch.randn_like(k)
    lens = torch.tensor([513], device="xpu", dtype=torch.int32)
    torch.xpu.synchronize()
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        actual, _ = kitchen.decode_attention(q, k, v, lens)
    stream.synchronize()
    torch.testing.assert_close(actual, reference(q, k, v, lens)[0], rtol=0.01, atol=0.002)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        actual, lse = kitchen.decode_attention(q, k, v, lens)
    for n in (0, 1, 257, 513):
        lens.fill_(n); graph.replay(); torch.xpu.synchronize()
        expected, el = reference(q, k, v, lens)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(lse, el, rtol=1e-5, atol=1e-5)


def test_decode_rejects_invalid_metadata_and_output_overlap():
    q = torch.zeros(1, 2, 2, 256, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 2, 256, device="xpu", dtype=q.dtype)
    length = torch.tensor([2], device="xpu", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="lengths"):
        kitchen.decode_attention(q, k, k, length.cpu())
    with pytest.raises(RuntimeError, match="BF16"):
        kitchen.decode_attention(q.float(), k, k, length)
    with pytest.raises(RuntimeError, match="memory location|overlap"):
        kitchen.decode_step_merge(q.transpose(1, 2).reshape(1,2,512).contiguous(),
            torch.zeros(1,2,2,device="xpu"), q,k,k,q.view(1,2,512))


@pytest.mark.parametrize("dim", [128, 256])
def test_native_decode_long_prefix(dim):
    """Match upstream's 32768-slot case, including the capped split count."""
    torch.manual_seed(181)
    q = torch.randn(1, 24, 1, dim, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 32768, dim, device="xpu", dtype=q.dtype)
    v = torch.randn_like(k)
    lengths = torch.tensor([32768], device="xpu", dtype=torch.int32)
    actual, lse = kitchen.decode_attention(q, k, v, lengths)
    expected, expected_lse = reference(q, k, v, lengths)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("second_capacity", [4096, 512])
def test_native_decode_independent_stream_graphs(second_capacity):
    """Port upstream's graph scratch-isolation check to two XPU streams.

    Both replays wait on the same event. Scheduling may serialize execution,
    so this establishes independent graph/stream correctness, not overlap.
    """
    torch.manual_seed(191)
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    states, graphs, outputs = [], [], []
    for stream, capacity in zip(streams, [4096, second_capacity], strict=True):
        q = torch.randn(1, 24, 1, 256, device="xpu", dtype=torch.bfloat16)
        k = torch.randn(1, 4, capacity, 256, device="xpu", dtype=q.dtype)
        v = torch.randn_like(k)
        lengths = torch.tensor([capacity - 128], device="xpu", dtype=torch.int32)
        state = (q, k, v, lengths)
        states.append(state)
        stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(stream):
            kitchen.decode_attention(*state)
        stream.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            outputs.append(kitchen.decode_attention(*state))
        graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.xpu.synchronize()
    for _ in range(40):
        for state in states:
            state[0].add_(0.0078125)
        expected = [kitchen.decode_attention(*state) for state in states]
        for out, lse in outputs:
            out.fill_(torch.nan)
            lse.fill_(torch.nan)
        ready = torch.xpu.Event()
        ready.record()
        for stream, graph in zip(streams, graphs, strict=True):
            stream.wait_event(ready)
            with torch.xpu.stream(stream):
                graph.replay()
        torch.xpu.synchronize()
        for actual, reference_output in zip(outputs, expected, strict=True):
            for got, want in zip(actual, reference_output, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
