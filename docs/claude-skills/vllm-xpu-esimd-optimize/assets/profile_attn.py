"""PROFILE_ATTN harness — Phase 4 attention timing breakdown.

Patch this snippet into FlashAttentionImpl.forward (or TritonAttentionImpl
.forward, whichever your model dispatches to) inside the editable vllm-xpu
checkout. PROFILE_ATTN=0 (default) is a zero-overhead pass-through; set
PROFILE_ATTN=1 to bucket every decode call (max_query_len==1) by
(gqa_ratio, is_sliding) and dump cumulative (count, total_ms) every 600
calls to stderr.

Use to confirm whether the attention kernel itself is the bottleneck
before writing a replacement. Typical gemma4-26B numbers:
  GQA=2 sliding ~ 0.10 ms / call × 25 layers
  GQA=8 full    ~ 0.105 ms / call × 5 layers
  total ~ 3.0 ms / token  (already near-optimal)

Apply pattern:
1. Locate `def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, ...)` in
   `vllm/v1/attention/backends/flash_attn.py` (or triton_attn.py).
2. Rename it to `_inner_forward`.
3. Add a new `forward` matching the original signature that calls
   `_inner_forward` with optional timing wrap.

Snippet (paste right before the renamed `_inner_forward`):
"""

# ----- begin patch fragment -----
def forward(
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
    output_scale=None,
    output_block_scale=None,
):
    import os
    if os.environ.get("PROFILE_ATTN", "0") == "1" \
            and attn_metadata is not None \
            and attn_metadata.max_query_len == 1:
        import time as _time
        import sys as _sys
        import torch
        _nq = query.shape[1] if query.dim() >= 2 else 0
        _nkv = kv_cache.shape[3] if kv_cache.dim() >= 4 else 0
        _gqa = _nq // _nkv if _nkv else 0
        _sw = self.sliding_window if hasattr(self, "sliding_window") else None
        _is_sliding = (_sw is not None and _sw[0] != -1)
        torch.xpu.synchronize()
        _t0 = _time.perf_counter()
        _ret = self._inner_forward(layer, query, key, value, kv_cache,
                                    attn_metadata, output,
                                    output_scale, output_block_scale)
        torch.xpu.synchronize()
        _dt = (_time.perf_counter() - _t0) * 1000.0
        global _ATTN_STATS_GLOBAL
        try:
            _ATTN_STATS_GLOBAL
        except NameError:
            _ATTN_STATS_GLOBAL = {}
        _stats = _ATTN_STATS_GLOBAL
        _key = (_gqa, _is_sliding)
        _e = _stats.setdefault(_key, [0, 0.0])
        _e[0] += 1
        _e[1] += _dt
        _stats["_call_n"] = _stats.get("_call_n", 0) + 1
        if _stats["_call_n"] % 600 == 0:
            _to_print = {k: v for k, v in _stats.items() if k != "_call_n"}
            print(f"[ATTN STATS @{_stats['_call_n']}] {_to_print}",
                  flush=True, file=_sys.stderr)
        return _ret
    return self._inner_forward(layer, query, key, value, kv_cache,
                                attn_metadata, output,
                                output_scale, output_block_scale)
# ----- end patch fragment -----
