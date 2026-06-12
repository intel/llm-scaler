import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

# Patch torch.median / torch.nanmedian on XPU.
#
# Why: on Intel XPU the dim-reduction median/nanmedian kernel is pathologically
# slow -- far slower than a full sort on the same device, and well behind the
# corresponding CUDA path -- independent of reduction length, dim, memory
# layout, value distribution and dtype (including integer types). The global
# median path (no dim argument) is healthy, so we only intervene on the
# dim-reduction path and leave everything else to the original implementation.
#
# Strategy (all paths produce bit-exact VALUES vs torch.median):
#   - small N  -> odd-even (Batcher) min/max compare-exchange network
#                 (lowest overhead for small reduction lengths)
#   - large N  -> torch.sort along dim, take the lower-median element
#                 (on XPU sort outperforms kthvalue)
# Indices: torch.median's index tie-break is implementation-defined; our fast
# paths return a VALID index (a position whose value equals the median value)
# via equality-argmax. Set OMNIXPU_MEDIAN_STRICT_INDICES=1 to force kthvalue,
# which reproduces torch.median's exact indices.

_orig_median = None
_orig_nanmedian = None

SMALL_N_MAX = 16  # measured crossover: minmax network <= sort up to ~N16

# dtypes where the XPU kernel is slow and our fast paths are valid
_FAST_DTYPES = (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8,
)

_net_cache = {}


def _oddeven_network(n):
    if n in _net_cache:
        return _net_cache[n]
    pairs = []

    def merge(lo, n2, r):
        step = r * 2
        if step < n2:
            merge(lo, n2, step)
            merge(lo + r, n2, step)
            for i in range(lo + r, lo + n2 - r, step):
                pairs.append((i, i + r))
        else:
            pairs.append((lo, lo + r))

    def sort(lo, n2):
        if n2 > 1:
            m = n2 // 2
            sort(lo, m)
            sort(lo + m, m)
            merge(lo, n2, 1)

    p = 1
    while p < n:
        p *= 2
    sort(0, p)
    net = [(a, b) for a, b in pairs if a < n and b < n]
    _net_cache[n] = net
    return net


def _minmax_lower_median(moved):
    # moved: reduction dim already at position 0, shape [N, ...]
    n = moved.shape[0]
    cols = [moved[i] for i in range(n)]
    for a, b in _oddeven_network(n):
        lo = torch.minimum(cols[a], cols[b])
        hi = torch.maximum(cols[a], cols[b])
        cols[a], cols[b] = lo, hi
    return cols[(n - 1) // 2]


def _fast_dim_median(input, dim, keepdim, strict_indices):
    moved = input.movedim(dim, 0)
    n = moved.shape[0]
    if strict_indices:
        # kthvalue reproduces torch.median exactly (values + indices)
        k = (n - 1) // 2 + 1
        v, i = torch.kthvalue(moved, k, dim=0)
    else:
        if n <= SMALL_N_MAX:
            v = _minmax_lower_median(moved)
        else:
            s, _ = torch.sort(moved, dim=0)
            v = s[(n - 1) // 2]
        # a valid median index: first position along dim holding the value
        i = (moved == v.unsqueeze(0)).to(torch.uint8).argmax(dim=0)
    if keepdim:
        v = v.unsqueeze(dim)
        i = i.unsqueeze(dim)
    return v, i


def _should_handle(input, dim):
    return (
        dim is not None
        and isinstance(input, torch.Tensor)
        and input.device.type == "xpu"
        and input.dtype in _FAST_DTYPES
        and input.dim() > 0
    )


def apply():
    global _orig_median, _orig_nanmedian
    import os

    _orig_median = torch.median
    _orig_nanmedian = torch.nanmedian
    strict = os.environ.get("OMNIXPU_MEDIAN_STRICT_INDICES", "0") != "0"

    def _patched_median(input, dim=None, keepdim=False, *, out=None):
        if out is not None or not _should_handle(input, dim):
            if dim is None:
                return _orig_median(input)
            return _orig_median(input, dim, keepdim, out=out) if out is not None \
                else _orig_median(input, dim, keepdim)
        # preserve NaN propagation: torch.median returns NaN if any NaN in slice
        if input.is_floating_point() and torch.isnan(input).any():
            return _orig_median(input, dim, keepdim)
        try:
            v, i = _fast_dim_median(input, dim, keepdim, strict)
            return torch.return_types.median((v, i))
        except Exception as e:  # never break the graph
            log.warning("[OmniXPU] median fast path failed (%s); fallback", e)
            return _orig_median(input, dim, keepdim)

    def _patched_nanmedian(input, dim=None, keepdim=False, *, out=None):
        if out is not None or not _should_handle(input, dim):
            if dim is None:
                return _orig_nanmedian(input)
            return _orig_nanmedian(input, dim, keepdim)
        # nanmedian ignores NaNs; if NaNs present, semantics differ -> fallback
        if input.is_floating_point() and torch.isnan(input).any():
            return _orig_nanmedian(input, dim, keepdim)
        try:
            v, i = _fast_dim_median(input, dim, keepdim, strict)
            return torch.return_types.nanmedian((v, i))
        except Exception as e:
            log.warning("[OmniXPU] nanmedian fast path failed (%s); fallback", e)
            return _orig_nanmedian(input, dim, keepdim)

    torch.median = _patched_median
    torch.nanmedian = _patched_nanmedian
    return True, None
