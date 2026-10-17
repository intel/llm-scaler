"""The same GDN input must survive page-aligned launch/checkpoint boundaries.

IMAWG4_EAGLE_LIBRARY optionally selects an isolated built extension, without
installing or upgrading any package. Tests compare both persisted state dtypes
and ensure that merely requesting snapshots cannot change the recurrence.
"""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def native_extension():
    if not torch.xpu.is_available():
        pytest.skip("Intel XPU required")
    library = os.environ.get("IMAWG4_EAGLE_LIBRARY")
    if library:
        torch.ops.load_library(library)
    else:
        import custom_esimd_kernels_sglang  # noqa: F401


@pytest.mark.parametrize("tokens", [129, 264, 513, 1025])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_cache_page_roundtrip(tokens, state_dtype):
    generator = torch.Generator().manual_seed(42)

    def randn(shape, dtype=torch.float16):
        return (torch.randn(shape, generator=generator) * 0.05).to(dtype).to("xpu")

    inputs = [randn((1, tokens, 1, 128)), randn((1, tokens, 1, 128)),
              randn((1, tokens, 2, 128)),
              -torch.rand((1, tokens, 2), generator=generator).to("xpu"),
              torch.rand((1, tokens, 2), generator=generator).to("xpu")]
    initial = randn((1, 2, 128, 128), state_dtype)

    def run(xs, state, snapshots=0):
        seq = torch.tensor([0, xs[0].shape[1]], dtype=torch.int32, device="xpu")
        return torch.ops.eagle_ops.chunk_gated_delta_rule_extend_consistent(
            *xs, state, seq, 128 ** -.5, snapshots, 64)

    full, _, checkpoints = run(inputs, initial.clone(), 64)
    without_snapshots, _, _ = run(inputs, initial.clone())
    torch.testing.assert_close(full, without_snapshots, rtol=0, atol=0)

    state = initial.clone()
    pieces = []
    for start in range(0, tokens, 64):
        out, state, _ = run([x[:, start:start + 64].contiguous() for x in inputs], state)
        pieces.append(out)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=0, atol=0)

    for prefix in (64, 128):
        resumed, _, _ = run([x[:, prefix:].contiguous() for x in inputs],
                            checkpoints[:, prefix // 64].clone().contiguous())
        torch.testing.assert_close(full[:, prefix:], resumed, rtol=0, atol=0)


def test_legacy_api_ignores_consistency_environment(monkeypatch):
    gen = torch.Generator().manual_seed(7)
    q = (torch.randn((1, 129, 1, 128), generator=gen) * .05).half().to("xpu")
    v = q.repeat(1, 1, 2, 1).contiguous()
    g = torch.full((1, 129, 2), -.1, device="xpu")
    beta = torch.full_like(g, .5)
    initial = torch.zeros((1, 2, 128, 128), dtype=torch.float16, device="xpu")
    seq = torch.tensor([0, 129], dtype=torch.int32, device="xpu")
    args = (q, q, v, g, beta)
    legacy = torch.ops.eagle_ops.chunk_gated_delta_rule_extend
    consistent = torch.ops.eagle_ops.chunk_gated_delta_rule_extend_consistent
    expected = legacy(*args, initial.clone(), seq, 128 ** -.5)[0]
    monkeypatch.setenv("SGLANG_XPU_GGUF_KERNEL_CONSISTENCY", "1")
    actual = legacy(*args, initial.clone(), seq, 128 ** -.5)[0]
    explicit_off = consistent(*args, initial.clone(), seq, 128 ** -.5, 0, 0)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit_off, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="must be 0.*or 64"):
        consistent(*args, initial.clone(), seq, 128 ** -.5, 0, 32)
    with pytest.raises(RuntimeError, match="snapshot interval"):
        consistent(*args, initial.clone(), seq, 128 ** -.5, 32, 64)
