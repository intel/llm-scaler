"""Bound SeedVR2 window-attention materialization without changing its graph.

ComfyUI's ``NaSwinAttention`` projects QKV for every video token, partitions
it into windows, and concatenates each window with the text tokens for one
varlen attention call. Above a byte threshold this adapter runs those same
upstream operations over consecutive groups of whole windows, so only one
group's QKV is live at a time. The Linear modules, QK norms, RoPE, attention
backend, and output projection remain ComfyUI's own. MLP chunking is owned by
ComfyUI's ``NaMMSRTransformerBlock._chunked_mlp``.
"""

from __future__ import annotations

import functools
import inspect
import logging

import torch


log = logging.getLogger("ComfyUI-OmniXPU")


_ATTENTION_THRESHOLD_BYTES = 4 * 1024**3
_ATTENTION_GROUP_BYTES = 256 * 1024**2
_PATCH_MARKER = "_omnixpu_seedvr_capacity_patched"
_CACHE_PREFIX = "omnixpu_capacity."
_attention_logged = False

_MODULE_SYMBOLS = (
    "AttentionTensorContainer",
    "NaMMRotaryEmbedding3d",
    "_apply_rope1_partial",
    "optimized_var_attention",
    "repeat_concat_idx",
    "window",
)
_ATTENTION_SOURCE_CONTRACT = (
    "vid_qkv, txt_qkv = self.proj_qkv(vid, txt)",
    "window_slices = self.window_op((t, h, w), self.window)",
    "lambda: window_idx(vid_shape, make_window)",
    "vid_q, txt_q = self.norm_q(vid_q, txt_q)",
    "vid_k, txt_k = self.norm_k(vid_k, txt_k)",
    "vid_q, vid_k, window_shape, txt_q, txt_k, txt_shape, txt_shape_win, cache_win",
    "repeat_concat_idx(vid_len_win, txt_len, window_count)",
    "preferred_attention=self.comfy_attention",
    "vid_out, txt_out = unconcat_win(out)",
    "vid_out, txt_out = self.proj_out(vid_out, txt_out)",
)
_ROPE_SOURCE_CONTRACT = (
    "self.get_freqs(vid_shape, txt_shape, vid_txt_shape)",
    "vid_q = _apply_rope1_partial(vid_q, vid_freqs)",
    "txt_q = _apply_rope1_partial(txt_q, txt_freqs)",
)


def _source_has(function, snippets):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return all(snippet in source for snippet in snippets)


def _branch(module, name):
    """Return the vid or txt module of a ComfyUI SeedVR2 ``MMModule``."""
    return module.all if module.shared_weights else getattr(module, name)


def _window_plan(seedvr_model, vid_shape, make_window):
    """Token order, shapes, and lengths of the windows ``window_idx`` builds."""
    token_index = torch.arange(
        vid_shape.prod(-1).sum(), device=vid_shape.device
    ).unsqueeze(-1)
    order, window_shape, _, window_lengths, _ = seedvr_model.window(
        token_index, vid_shape, make_window
    )
    return order.squeeze(-1), window_shape, window_lengths


def _window_groups(window_lengths, row_bytes, group_bytes):
    """Split windows into consecutive groups holding at most ``group_bytes`` of QKV."""
    max_tokens = max(1, group_bytes // row_bytes)
    groups = []
    first_window = first_token = tokens = 0
    for index, length in enumerate(window_lengths):
        if tokens and tokens + length > max_tokens:
            groups.append(
                (first_window, index, first_token, first_token + tokens)
            )
            first_window, first_token, tokens = (
                index,
                first_token + tokens,
                0,
            )
        tokens += length
    groups.append(
        (first_window, len(window_lengths), first_token, first_token + tokens)
    )
    return groups


def _qkv_bytes(self, vid):
    return vid.shape[0] * 3 * self.heads * self.head_dim * vid.element_size()


def _should_bound_attention(self, vid, txt, vid_shape, model_management):
    return bool(
        getattr(self.rope, "mm", False)
        and not getattr(self, "version_7b", False)
        and vid_shape.shape[0] == 1
        and not model_management.in_training
        and not vid.requires_grad
        and not txt.requires_grad
        and _qkv_bytes(self, vid) > _ATTENTION_THRESHOLD_BYTES
    )


def _bounded_attention(seedvr_model, self, vid, txt, vid_shape, txt_shape, cache):
    cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

    def make_window(value):
        t, h, w, _ = value.shape
        window_slices = self.window_op((t, h, w), self.window)
        return [value[st, sh, sw] for st, sh, sw in window_slices]

    order, window_shape, window_lengths = cache_win(
        _CACHE_PREFIX + "windows",
        lambda: _window_plan(seedvr_model, vid_shape, make_window),
    )
    row_bytes = 3 * self.heads * self.head_dim * vid.element_size()
    groups = cache_win(
        _CACHE_PREFIX + f"groups_{row_bytes}_{_ATTENTION_GROUP_BYTES}",
        lambda: _window_groups(
            window_lengths, row_bytes, _ATTENTION_GROUP_BYTES
        ),
    )

    def group_rope(first, last):
        # mmrope frequencies depend only on each window's own shape and the
        # text length of its sample, so a window subset yields its own rows.
        return self.rope.get_freqs(
            window_shape[first:last],
            txt_shape,
            txt_shape.repeat(last - first, 1),
        )

    def group_concat(first, last):
        return seedvr_model.repeat_concat_idx(
            window_shape[first:last].prod(-1),
            txt_shape.prod(-1).to(window_shape.device),
            torch.tensor([last - first], device=window_shape.device),
        )

    vid_qkv_module = _branch(self.proj_qkv, "vid")
    vid_norm_q = _branch(self.norm_q, "vid")
    vid_norm_k = _branch(self.norm_k, "vid")

    txt = txt.to(device=vid.device, dtype=vid.dtype)
    txt_qkv = _branch(self.proj_qkv, "txt")(txt).reshape(
        txt.shape[0], 3, self.heads, self.head_dim
    )
    txt_q, txt_k, txt_v = txt_qkv.unbind(1)
    del txt_qkv
    txt_q = _branch(self.norm_q, "txt")(txt_q)
    txt_k = _branch(self.norm_k, "txt")(txt_k)

    vid_out = vid.new_empty((vid.shape[0], self.heads * self.head_dim))
    txt_sum = None
    for group_index, (first, last, token_start, token_end) in enumerate(groups):
        vid_freqs, txt_freqs = cache_win(
            _CACHE_PREFIX + f"rope_{first}_{last}",
            functools.partial(group_rope, first, last),
        )
        if group_index == 0:
            txt_freqs = txt_freqs.to(device=txt_q.device)
            txt_q = seedvr_model._apply_rope1_partial(txt_q, txt_freqs)
            txt_k = seedvr_model._apply_rope1_partial(txt_k, txt_freqs)

        group_order = order[token_start:token_end]
        vid_qkv = vid_qkv_module(torch.index_select(vid, 0, group_order))
        vid_qkv = vid_qkv.reshape(
            vid_qkv.shape[0], 3, self.heads, self.head_dim
        )
        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        del vid_qkv
        vid_q = vid_norm_q(vid_q)
        vid_k = vid_norm_k(vid_k)
        vid_freqs = vid_freqs.to(device=vid_q.device)
        vid_q = seedvr_model._apply_rope1_partial(vid_q, vid_freqs)
        vid_k = seedvr_model._apply_rope1_partial(vid_k, vid_freqs)

        concat, unconcat, cu_seqlens = cache_win(
            _CACHE_PREFIX + f"concat_{first}_{last}",
            functools.partial(group_concat, first, last),
        )
        container = seedvr_model.AttentionTensorContainer
        q = container(concat(vid_q, txt_q))
        del vid_q
        k = container(concat(vid_k, txt_k))
        del vid_k
        v = container(concat(vid_v, txt_v))
        del vid_v
        out = seedvr_model.optimized_var_attention(
            q=q,
            k=k,
            v=v,
            preferred_attention=self.comfy_attention,
            heads=self.heads,
            skip_reshape=True,
            skip_output_reshape=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        del q, k, v
        group_vid, group_txt = unconcat(out)
        del out
        vid_out.index_copy_(0, group_order, group_vid.flatten(1, 2))
        # ComfyUI averages the text output over every window; weight each
        # group's mean by its window count to keep that average.
        weighted = group_txt.float().mul_(last - first)
        txt_sum = weighted if txt_sum is None else txt_sum.add_(weighted)
        del group_vid, group_txt

    txt_out = txt_sum.div_(len(window_lengths)).to(vid.dtype).flatten(1, 2)
    return self.proj_out(vid_out, txt_out)


def _patch_attention(seedvr_model):
    attention = getattr(seedvr_model, "NaSwinAttention", None)
    if attention is None:
        return False, "NaSwinAttention is unavailable"
    if getattr(attention, _PATCH_MARKER, False):
        return False, "SeedVR2 capacity adapter is already applied"
    missing = [
        name for name in _MODULE_SYMBOLS if not hasattr(seedvr_model, name)
    ]
    if missing:
        return False, "SeedVR2 model lacks " + ", ".join(missing)
    if not _source_has(attention.forward, _ATTENTION_SOURCE_CONTRACT):
        return False, "unsupported NaSwinAttention.forward contract"
    if not _source_has(
        seedvr_model.NaMMRotaryEmbedding3d.forward, _ROPE_SOURCE_CONTRACT
    ):
        return False, "unsupported NaMMRotaryEmbedding3d.forward contract"

    original = attention.forward

    @functools.wraps(original)
    def forward(self, vid, txt, vid_shape, txt_shape, cache):
        global _attention_logged
        if _should_bound_attention(
            self,
            vid,
            txt,
            vid_shape,
            seedvr_model.comfy.model_management,
        ):
            if not _attention_logged:
                log.info(
                    "[OmniXPU] SeedVR2 bounded attention: "
                    "tokens=%d heads=%d head_dim=%d qkv_bytes=%d",
                    vid.shape[0],
                    self.heads,
                    self.head_dim,
                    _qkv_bytes(self, vid),
                )
                _attention_logged = True
            return _bounded_attention(
                seedvr_model,
                self,
                vid,
                txt,
                vid_shape,
                txt_shape,
                cache,
            )
        return original(self, vid, txt, vid_shape, txt_shape, cache)

    attention.forward = forward
    setattr(attention, _PATCH_MARKER, True)
    return True, None


def apply():
    try:
        from comfy.ldm.seedvr import model as seedvr_model
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy.ldm.seedvr", "comfy.ldm.seedvr.model"}:
            return False, "ComfyUI SeedVR2 model is not available"
        raise

    return _patch_attention(seedvr_model)
