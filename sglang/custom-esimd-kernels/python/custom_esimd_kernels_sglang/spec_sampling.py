"""XPU implementations of the speculative-sampling ops that sgl-kernel only
provides for CUDA.

sglang's ``eagle_info_v2`` imports three symbols from ``sgl_kernel`` behind an
``if is_cuda() or is_musa():`` guard::

    top_k_renorm_prob
    top_p_renorm_prob
    tree_speculative_sampling_target_only

On XPU the guard is false, so the names are simply undefined and any request
with ``temperature > 0`` under MTP raises ``NameError`` deep inside the verify
step -- which takes down the whole server, not just the request. That limits
speculative decoding to greedy decoding on XPU.

This module supplies the three ops. ``tree_speculative_sampling_target_only``
is split across two implementations:

  * the sequential per-request tree walk runs in SYCL
    (``spec_sampling_ops.tree_spec_sampling_walk``), because it is irregular
    pointer chasing that torch cannot express efficiently;
  * the final draw from the residual distribution
    ``relu(target_probs - draft_probs)`` is done here in torch, because it is a
    plain reduction + prefix scan that torch already expresses exactly.

The CUDA kernel fuses both phases only to reuse its block's shared memory.

Reference: sgl-kernel/csrc/speculative/speculative_sampling.cuh
           :: TreeSpeculativeSamplingTargetOnly
"""

import os

import torch

# Escape hatch for reproducing the ordering bug the barriers below work around;
# see the comment at the call site. Leave this alone in normal use.
_WALK_BARRIER = os.environ.get("SPEC_WALK_SYNC", "1") == "1"


def _walk_barrier():
    if _WALK_BARRIER:
        torch.xpu.synchronize()


__all__ = [
    "top_k_renorm_prob",
    "top_p_renorm_prob",
    "tree_speculative_sampling_target_only",
]


def _as_row_tensor(value, n, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.full((n,), value, device=device, dtype=dtype)


def _renorm(probs, keep):
    out = probs * keep
    return out / out.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(out.dtype).tiny)


def top_k_renorm_prob(probs: torch.Tensor, top_k) -> torch.Tensor:
    """Zero every entry outside each row's top-k, then renormalise.

    Matches flashinfer's ``top_k_renorm_probs``: rank-based, so exact ties at
    the k-th position are resolved by the sort order rather than by keeping
    extra entries.
    """
    n, d = probs.shape
    k = _as_row_tensor(top_k, n, probs.device, torch.long).clamp_(min=1, max=d)
    _, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    rank = torch.arange(d, device=probs.device).unsqueeze(0)
    keep_sorted = rank < k.unsqueeze(-1)
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    return _renorm(probs, keep)


def top_p_renorm_prob(probs: torch.Tensor, top_p) -> torch.Tensor:
    """Keep the shortest high-probability prefix reaching mass ``top_p``, renormalise.

    An entry is kept when the mass strictly above it is still below ``top_p``,
    so the element that crosses the threshold is included and at least one
    entry always survives.
    """
    n, d = probs.shape
    p = _as_row_tensor(top_p, n, probs.device, probs.dtype)
    sorted_vals, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    exclusive_cum = sorted_vals.cumsum(dim=-1) - sorted_vals
    keep_sorted = exclusive_cum < p.unsqueeze(-1)
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    return _renorm(probs, keep)


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,          # mutable, int32 [tot_num_draft_tokens]
    accept_index: torch.Tensor,      # mutable, int32 [bs, num_spec_step]
    accept_token_num: torch.Tensor,  # mutable, int32 [bs]
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,       # mutable, used as scratch (see module doc)
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    # `deterministic` selects a scan algorithm in the CUDA kernel. The torch
    # reduction used here is already run-to-run deterministic, so the flag has
    # nothing to switch and is accepted only for signature compatibility.
    from custom_esimd_kernels_sglang.spec_sampling_ops import tree_spec_sampling_walk

    bs, num_draft_tokens, d = target_probs.shape
    num_spec_step = accept_index.shape[1]
    device = target_probs.device

    # Row of target_probs the residual draw must come from, and the slot in
    # `predicts` that receives it. Both are byproducts of the walk.
    prob_row = torch.empty(bs, dtype=torch.int32, device=device)
    last_idx = torch.empty(bs, dtype=torch.int32, device=device)

    # The walk is a bare `queue.submit` on the current XPU stream, and its two
    # index outputs are consumed immediately afterwards by torch indexing ops.
    # Without the barriers below that hand-off is not ordered in practice:
    # `index_select` reads `prob_row` while it still holds whatever
    # `torch.empty` returned, and an out-of-range value there trips
    #   torch-xpu-ops .../sycl/Indexing.h:622 "index out of bounds"
    # which aborts the rank outright. Under TP the surviving rank then blocks
    # forever in the next collective, so it surfaces as a hang rather than a
    # crash. Measured on BFCL multi_turn_base: 528 assertion hits and a dead
    # rank without the barriers, zero across a full 200-case run with them.
    #
    # Why the hand-off is unordered is NOT yet understood -- every other kernel
    # in this package submits the same way and documents (moe_grouped_entry.sycl
    # NOTE #102) that same-stream ordering makes waiting unnecessary. The
    # difference here may be that this is the only SYCL->torch boundary rather
    # than SYCL->SYCL. Until that is pinned down, do not "optimise" these away,
    # and do not assume the kernel-side submit is safe to change.
    _walk_barrier()
    tree_spec_sampling_walk(
        predicts,
        accept_index,
        accept_token_num,
        draft_probs,
        prob_row,
        last_idx,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        target_probs,
        float(threshold_single),
        float(threshold_acc),
    )
    _walk_barrier()

    rows = prob_row.long()
    q = target_probs.reshape(-1, d).index_select(0, rows)
    p = draft_probs.reshape(-1, d).index_select(0, rows)

    # When every draft token was accepted the extra token is a bonus drawn from
    # the target distribution itself; there is no draft mass to subtract.
    bonus = (accept_token_num == num_spec_step - 1).reshape(-1, 1)
    residual = torch.where(bonus, q, (q - p).clamp_min(0))

    u = uniform_samples_for_final_sampling.to(residual.dtype) * residual.sum(dim=-1)
    crossed = residual.cumsum(dim=-1) > u.unsqueeze(-1)

    # Lowest index whose inclusive prefix sum exceeds u. Selected with an
    # explicit min over indices rather than argmax, whose tie-breaking among
    # equal maxima is not guaranteed.
    idx = torch.arange(d, device=device, dtype=torch.int32).unsqueeze(0)
    first_crossing = torch.where(crossed, idx, d).min(dim=-1).values

    # u can land past the total mass when it is very close to 1 and the
    # probabilities do not quite sum to it. The CUDA kernel then falls back to
    # the last index carrying any mass.
    positive = residual > 0
    last_positive = torch.where(positive, idx, -1).max(dim=-1).values
    fallback = torch.where(last_positive >= 0, last_positive, d - 1)

    sampled = torch.where(first_crossing < d, first_crossing, fallback)
    predicts.index_copy_(0, last_idx.long(), sampled.to(torch.int32))
