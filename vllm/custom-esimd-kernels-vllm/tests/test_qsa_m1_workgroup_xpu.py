"""M1 workgroup backend: 32K scattered KV and asynchronous caller ownership."""
import pytest
import torch

from test_qsa_sparse_attention_xpu import _load_qsa_extension, _xpu_available
from test_qsa_token_split_boundaries_xpu import _reference_cpu


pytestmark = pytest.mark.skipif(not _xpu_available(), reason="requires XPU")


@pytest.mark.parametrize("heads", [3, 6])
@pytest.mark.parametrize("page_size", [256, 512])
@pytest.mark.parametrize("query_scale", [0.2, 3.0])
def test_scattered_32k_m1_streams_match_independent_reference(
    heads, page_size, query_scale
):
    ops = _load_qsa_extension()
    generator = torch.Generator().manual_seed(16092026)
    kv_cpu = torch.randn(32768 // page_size, 1, page_size, 512,
                         generator=generator).half()
    q_cpu = (torch.randn(1, heads, 256, generator=generator) * query_scale).half()
    groups = torch.randperm(8191, generator=generator)[:512]
    ids = torch.cat(((groups[:, None] * 4 + torch.arange(4)).flatten(),
                     torch.tensor([32765, 32766, 32767]))).int()[None]
    ids[:, 23:45] = -7
    ids[:, 46] = ids[:, 45]  # multiplicity, not a set
    table = torch.randperm(32768 // page_size, generator=generator).int()[None]
    request = torch.zeros(1, dtype=torch.int32)
    expected = _reference_cpu(q_cpu, kv_cpu, ids, table, request)
    q, kv, logical, blocks, req = [x.to("xpu") for x in
                                  (q_cpu, kv_cpu, ids, table, request)]
    parent = torch.xpu.current_stream()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    results, owned = [], []
    launch = (ops.sparse_attention_token_split_candidate_q6_v1 if heads == 6
              else ops.sparse_attention_token_split_candidate_v3)
    for stream in streams:
        stream.wait_stream(parent)
        with torch.xpu.stream(stream):
            storage = torch.full((3, heads, 43, 258), float("nan"),
                                 dtype=torch.float32, device="xpu")
            scratch = storage[1:2]
            output = torch.empty_like(q)
            owned.append((storage, scratch, output))
            for _ in range(3):
                launch(q, kv, logical, blocks, req, page_size, output, scratch)
                results.append(output.clone())
    for stream in streams:
        parent.wait_stream(stream)
    for actual in results:
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)
    for storage, _, _ in owned:
        assert torch.isnan(storage[0]).all()
        assert torch.isnan(storage[2]).all()
        assert torch.isnan(storage[1, :, 33:]).all()
