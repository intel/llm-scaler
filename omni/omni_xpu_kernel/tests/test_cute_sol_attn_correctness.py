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


def reference(
    q,
    k,
    v,
    scale,
    tau,
    sink_blocks,
    sink_q,
    tail=True,
    topk_ratio=0.0,
    key_bias=None,
):
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
    if topk_ratio:
        ranked = route_scores.clone()
        ranked[..., sink_blocks[0] : sink_blocks[1]] = float("-inf")
        sink_count = max(
            0,
            min(sink_blocks[1], blocks) - min(sink_blocks[0], blocks),
        )
        selectable = blocks - sink_count
        topk_count = max(
            0,
            min(
                selectable - 1,
                max(1, round(topk_ratio * selectable)),
            ),
        )
        if topk_count:
            row_threshold = ranked.topk(topk_count, dim=-1).values[..., -1:]
            routes = ranked >= row_threshold
        else:
            routes = torch.zeros_like(ranked, dtype=torch.bool)
    else:
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
                if key_bias is not None:
                    for key_block in range(blocks):
                        if routes[
                            batch_index, head, query_block, key_block
                        ]:
                            start = key_block * 64
                            stop = min(tokens, start + 64)
                            scaled[key_block] += key_bias[
                                batch_index, start:stop
                            ].float()
                maximum = max(
                    item.max()
                    for key_block, item in enumerate(scaled)
                    if tail
                    or routes[
                        batch_index, head, query_block, key_block
                    ]
                )
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
                    elif tail:
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


@pytest.mark.skipif(
    not has_bmg_sol_attn(), reason="BMG packaged Sol-Attn unavailable"
)
def test_sol_attn_tail_false_matches_routed_only_reference():
    from omni_xpu_kernel import cute

    generator = torch.Generator(device="xpu").manual_seed(5)
    shape = (1, 257, 1, 128)
    q = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    k = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    v = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    scale = 128**-0.5
    actual = cute.sol_attn(q, k, v, scale=scale, tau=100.0, tail=False)
    expected = reference(
        q, k, v, scale, 100.0, (0, 0), (0, 0), tail=False
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    not has_bmg_sol_attn(), reason="BMG packaged Sol-Attn unavailable"
)
@pytest.mark.parametrize(
    "tokens,topk_ratio,sink_blocks,seed",
    [
        (320, 0.4, (0, 0), 6),
        (320, 0.5, (0, 4), 7),
        (512, 0.25, (0, 0), 8),
    ],
)
def test_sol_attn_topk_no_tail_matches_reference(
    tokens, topk_ratio, sink_blocks, seed
):
    from omni_xpu_kernel import cute

    generator = torch.Generator(device="xpu").manual_seed(seed)
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    k = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    v = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    scale = 128**-0.5
    actual = cute.sol_attn(
        q,
        k,
        v,
        scale=scale,
        topk_ratio=topk_ratio,
        tail=False,
        sink_blocks=sink_blocks,
    )
    expected = reference(
        q,
        k,
        v,
        scale,
        1.0,
        sink_blocks,
        (0, 0),
        tail=False,
        topk_ratio=topk_ratio,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    not has_bmg_sol_attn(), reason="BMG packaged Sol-Attn unavailable"
)
@pytest.mark.parametrize("bias_kind", ["float", "bool"])
def test_sol_attn_key_bias_matches_reference_at_zero_qk_scale(bias_kind):
    from omni_xpu_kernel import cute

    generator = torch.Generator(device="xpu").manual_seed(11)
    shape = (1, 65, 1, 128)
    q = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    k = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    v = torch.randn(shape, generator=generator, device="xpu").bfloat16()
    if bias_kind == "bool":
        key_bias = torch.ones(65, device="xpu", dtype=torch.bool)
        key_bias[::3] = False
        reference_bias = torch.where(key_bias, 0.0, float("-inf")).view(1, -1)
    else:
        key_bias = torch.linspace(-2.0, 2.0, 65, device="xpu")
        reference_bias = key_bias.view(1, -1)
    actual = cute.sol_attn(
        q,
        k,
        v,
        scale=0.0,
        sink_q=(0, 2),
        key_bias=key_bias,
        tail=False,
    )
    expected = reference(
        q,
        k,
        v,
        0.0,
        1.0,
        (0, 0),
        (0, 2),
        tail=False,
        key_bias=reference_bias,
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    assert torch.isfinite(actual).all()
