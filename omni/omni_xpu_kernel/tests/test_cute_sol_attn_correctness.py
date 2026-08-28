"""Correctness for the packaged BMG Sol-Attn CUTE entry point."""

import math

import pytest
import torch


def has_bmg_sol_attn():
    try:
        import omni_xpu_kernel
        from omni_xpu_kernel import cute

        return (
            torch.xpu.is_available()
            and omni_xpu_kernel.__xpu_target__ == "bmg"
            and cute is not None
            and cute.supports_sol_attn()
        )
    except Exception:
        return False


def summaries(tensor, operation):
    batch, tokens, heads, dim = tensor.shape
    blocks = (tokens + 63) // 64
    values = []
    for block in range(blocks):
        part = tensor[:, block * 64 : min(tokens, (block + 1) * 64)].float()
        reduced = part.mean(dim=1) if operation == "mean" else part.sum(dim=1)
        values.append(reduced)
    return torch.stack(values, dim=2).to(tensor.dtype).reshape(
        batch, heads, blocks, dim
    )


def reference(q, k, v, scale, tau, sink_blocks, sink_q):
    batch, tokens, heads, dim = q.shape
    blocks = (tokens + 63) // 64
    q_centroids = summaries(q, "mean").float()
    k_centroids = summaries(k, "mean")
    v_sums = summaries(v, "sum")
    k_float = k_centroids.float()
    k_mean = k_float.mean(dim=2)
    k_variance = (
        k_float.square().mean(dim=2) - k_mean.square()
    ).clamp_min(0)
    raw_mean = (q_centroids * k_mean.unsqueeze(2)).sum(dim=-1)
    raw_variance = (
        q_centroids.square() * k_variance.unsqueeze(2)
    ).sum(dim=-1)
    log2_scale = scale * math.log2(math.e)
    thresholds = raw_mean * log2_scale + tau * torch.sqrt(
        raw_variance * log2_scale * log2_scale + 1.0e-6
    )
    route_scores = torch.einsum(
        "bhqd,bhkd->bhqk", q_centroids, k_centroids.float()
    ) * log2_scale
    routes = route_scores > thresholds.unsqueeze(-1)
    block_ids = torch.arange(blocks, device=q.device)
    routes |= (
        (block_ids[:, None] - block_ids[None, :]).abs()[None, None] <= 1
    )
    routes[:, :, :, sink_blocks[0] : sink_blocks[1]] = True
    routes[:, :, sink_q[0] : sink_q[1], :] = True

    output = torch.empty_like(q)
    for batch_index in range(batch):
        for head in range(heads):
            for query_token in range(tokens):
                query_block = query_token // 64
                query = q[batch_index, query_token, head].float()
                scores = []
                values = []
                for key_block in range(blocks):
                    start = key_block * 64
                    stop = min(tokens, start + 64)
                    if routes[
                        batch_index, head, query_block, key_block
                    ]:
                        scores.append(
                            k[batch_index, start:stop, head].float() @ query
                        )
                        values.append(v[batch_index, start:stop, head].float())
                    else:
                        scores.append(
                            query
                            @ k_centroids[
                                batch_index, head, key_block
                            ].float()
                        )
                        values.append(
                            v_sums[batch_index, head, key_block]
                            .float()
                            .unsqueeze(0)
                            / (stop - start)
                        )
                scaled = [item * scale for item in scores]
                maximum = max(item.max() for item in scaled)
                numerator = torch.zeros(dim, device=q.device)
                denominator = torch.zeros((), device=q.device)
                for key_block, (score, value) in enumerate(
                    zip(scaled, values)
                ):
                    probability = torch.exp(score - maximum)
                    if routes[
                        batch_index, head, query_block, key_block
                    ]:
                        numerator += probability @ value
                        denominator += probability.sum()
                    else:
                        start = key_block * 64
                        length = min(tokens, start + 64) - start
                        numerator += probability * value[0] * length
                        denominator += probability * length
                output[batch_index, query_token, head] = (
                    numerator / denominator
                ).to(q.dtype)
    return output


@pytest.mark.skipif(
    not has_bmg_sol_attn(), reason="BMG packaged Sol-Attn unavailable"
)
@pytest.mark.parametrize(
    "tokens,heads,tau,sink_blocks,sink_q,seed",
    [
        (31, 1, 1.0, (0, 0), (0, 0), 1),
        (65, 2, 1.3, (0, 0), (0, 0), 2),
        (257, 1, 100.0, (0, 0), (0, 0), 3),
        (193, 1, 100.0, (0, 1), (0, 1), 4),
    ],
)
def test_sol_attn_matches_algorithm_reference(
    tokens, heads, tau, sink_blocks, sink_q, seed
):
    from omni_xpu_kernel import cute

    generator = torch.Generator(device="xpu").manual_seed(seed)
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    k = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    v = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    scale = 128**-0.5
    actual = cute.sol_attn(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
    )
    expected = reference(q, k, v, scale, tau, sink_blocks, sink_q)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    not has_bmg_sol_attn(), reason="BMG packaged Sol-Attn unavailable"
)
def test_sol_attn_accepts_h3_qkv_stride_and_zero_adversarial():
    from omni_xpu_kernel import cute

    tokens = 129
    packed = torch.zeros(
        (1, tokens, 3, 2, 128), device="xpu", dtype=torch.bfloat16
    )
    q, k, v = (packed[:, :, index, :, :] for index in range(3))
    actual = cute.sol_attn(q, k, v, tau=1.3)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    assert actual.stride() == (tokens * 2 * 128, 2 * 128, 128, 1)
