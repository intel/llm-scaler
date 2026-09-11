"""XPU regression for histogram/remainder rounding at a real H3 bin boundary."""
import pytest
import torch

from test_cute_sol_token_correctness import histogram_reference, ops


def boundary_inputs(tokens, layout):
    groups = ((tokens + 63) // 64 + 1) // 2
    q = torch.zeros((1, 2, groups, 128), device="xpu", dtype=torch.int8)
    # Exact INT8 dot 15213, reduced from H3 head 15 / group 36 / key 6723.
    q[..., 0] = 127
    q[..., 1] = 100
    if layout == "packed":
        packed = torch.zeros((1, tokens, 3, 2, 128), device="xpu", dtype=torch.int8)
        k, v = packed[:, :, 1], packed[:, :, 2]
    else:
        k, v = (torch.zeros((1, 2, tokens, 128), device="xpu", dtype=torch.int8)
                .permute(0, 2, 1, 3) for _ in range(2))
    k[..., 0] = 119
    k[..., 1] = 1
    qs = torch.full(q.shape[:3], 0.054441437125205994, device="xpu")
    refs = torch.full_like(qs, 2.931443214416504)
    ks = torch.full((1, 2, tokens), 0.04669388383626938, device="xpu")
    common = torch.ones((1, 2, groups, (tokens + 63) // 64),
                        device="xpu", dtype=torch.uint8)
    return q, qs, refs, k, ks, v, common


@pytest.mark.parametrize("layout", ["bhtd", "packed"])
def test_token_bin_boundary_uses_same_fp32_rounding(layout):
    q, qs, refs, k, ks, v, common = boundary_inputs(64, layout)
    ks[..., 1:] = 0
    hist, bins, valid, _ = histogram_reference(q, qs, refs, k, ks, common, 128**-0.5)
    assert bool((bins[..., 0] == 40).all().item())
    assert bool((valid.sum(-1) == 1).all().item())
    actual = ops().token_histogram(q, qs, refs, k, ks, common, 128**-0.5)
    torch.testing.assert_close(actual, hist, rtol=0, atol=0)
    cutoff = torch.full(q.shape[:3], 40, device="xpu", dtype=torch.int32)
    for tail in (False, True):
        _, counts, _ = ops().token_remainder(
            q, qs, refs, k, ks, v, common, cutoff, 128**-0.5, 64, tail)
        torch.testing.assert_close(counts, torch.ones_like(counts), rtol=0, atol=0)


@pytest.mark.parametrize("budget", [64, 128, 192, 256])
@pytest.mark.parametrize("layout", ["bhtd", "packed"])
def test_token_boundary_cannot_overflow_whole_bin_budget(budget, layout):
    q, qs, refs, k, ks, v, common = boundary_inputs(budget + 1, layout)
    ks[..., :budget] *= 2
    hist = ops().token_histogram(q, qs, refs, k, ks, common, 128**-0.5)
    cutoff = ops().token_bin_cutoff(hist, budget)
    axis = torch.arange(128, device="xpu")
    admitted = torch.where(axis >= cutoff.unsqueeze(-1), hist, 0).sum(-1).to(torch.int32)
    assert bool((admitted == budget).all().item())
    for tail in (False, True):
        indices, counts, _ = ops().token_remainder(
            q, qs, refs, k, ks, v, common, cutoff, 128**-0.5, budget, tail)
        # Do not submit a malformed carrier to a selected-token consumer.
        torch.testing.assert_close(counts, admitted, rtol=0, atol=0)
        assert bool((counts <= budget).all().item())
        wanted = torch.arange(budget, device="xpu", dtype=torch.int32).expand_as(indices)
        torch.testing.assert_close(ops().sort_token_indices(indices, counts), wanted, rtol=0, atol=0)
