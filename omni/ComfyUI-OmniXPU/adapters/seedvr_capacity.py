"""Bound large SeedVR2 activation lifetimes without changing its graph.

The adapter keeps the selected attention backend and quantized Linear modules
opaque.  It only changes when independent rows and windows are materialized.
"""

from __future__ import annotations

import functools
import inspect
import logging

import torch


log = logging.getLogger("ComfyUI-OmniXPU")


_ATTENTION_THRESHOLD_BYTES = 4 * 1024**3
_ATTENTION_GROUP_BYTES = 256 * 1024**2
_MLP_THRESHOLD_BYTES = 4 * 1024**3
_MLP_GROUP_BYTES = 384 * 1024**2
_LINEAR_GROUP_ROWS = 65536
_PATCH_MARKER = "_omnixpu_seedvr_capacity_patched"
_attention_logged = False
_swiglu_logged = False

_ATTENTION_SOURCE_CONTRACT = (
    "vid_qkv, txt_qkv = self.proj_qkv(vid, txt)",
    "window_idx(vid_shape, make_window)",
    "optimized_var_attention(",
    "self.proj_out(vid_out, txt_out)",
)
_SWIGLU_SOURCE_CONTRACT = (
    "self.proj_out(",
    "F.silu(self.proj_in_gate(x))",
    "self.proj_in(x)",
)


def _source_has(function, snippets):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return all(snippet in source for snippet in snippets)


def _chunk_rows(row_bytes, target_bytes, alignment=256):
    rows = max(1, target_bytes // max(1, row_bytes))
    if rows >= alignment:
        rows = max(alignment, rows // alignment * alignment)
    return rows


def _chunked_forward(module, value, chunk_rows):
    value_2d = value.reshape(-1, value.shape[-1])
    output = None
    offset = 0
    for chunk in value_2d.split(chunk_rows, dim=0):
        chunk_output = module(chunk)
        if output is None:
            output = chunk_output.new_empty(
                (value_2d.shape[0], *chunk_output.shape[1:])
            )
        output[offset : offset + chunk_output.shape[0]] = chunk_output
        offset += chunk_output.shape[0]
    return output.reshape(*value.shape[:-1], *output.shape[1:])


def _bounded_swiglu(module, value, chunk_rows):
    return _chunked_forward(
        lambda chunk: module.proj_out(
            torch.nn.functional.silu(module.proj_in_gate(chunk))
            * module.proj_in(chunk)
        ),
        value,
        chunk_rows,
    )


def _window_indices(seedvr_model, hid_shape, window_fn):
    hid_idx = torch.arange(
        hid_shape.prod(-1).sum(), device=hid_shape.device
    ).unsqueeze(-1)
    target_idx, target_shape, target_windows, target_lengths, window_counts = (
        seedvr_model.window(hid_idx, hid_shape, window_fn)
    )
    return (
        target_idx.squeeze(-1),
        target_shape,
        target_windows,
        target_lengths,
        window_counts,
    )


def _should_bound_attention(self, vid, txt, vid_shape, model_management):
    qkv_bytes = (
        vid.shape[0]
        * 3
        * self.heads
        * self.head_dim
        * vid.element_size()
    )
    return bool(
        self.rope is not None
        and self.rope.mm
        and vid_shape.shape[0] == 1
        and not model_management.in_training
        and not vid.requires_grad
        and not txt.requires_grad
        and qkv_bytes > _ATTENTION_THRESHOLD_BYTES
    )


def _bounded_attention(seedvr_model, self, vid, txt, vid_shape, txt_shape, cache):
    cache_win = cache.namespace(
        f"{self.window_method}_{self.window}_sd3"
    )

    def make_window(value):
        temporal, height, width, _ = value.shape
        slices = self.window_op(
            (temporal, height, width), self.window
        )
        return [value[st, sh, sw] for st, sh, sw in slices]

    window_order, window_shape, _, vid_lengths, window_counts = cache_win(
        "win_indices",
        lambda: _window_indices(seedvr_model, vid_shape, make_window),
    )
    window_count = window_counts[0]

    qkv_row_bytes = 3 * self.heads * self.head_dim * vid.element_size()
    max_group_tokens = max(1, _ATTENTION_GROUP_BYTES // qkv_row_bytes)
    groups = []
    first_window = first_token = group_tokens = 0
    for next_window, window_tokens in enumerate(vid_lengths, 1):
        if group_tokens and group_tokens + window_tokens > max_group_tokens:
            groups.append(
                (
                    first_window,
                    next_window - 1,
                    first_token,
                    first_token + group_tokens,
                )
            )
            first_window = next_window - 1
            first_token += group_tokens
            group_tokens = 0
        group_tokens += window_tokens
    groups.append(
        (first_window, window_count, first_token, first_token + group_tokens)
    )

    vid_qkv_module = (
        self.proj_qkv.vid
        if not self.proj_qkv.shared_weights
        else self.proj_qkv.all
    )
    txt_qkv_module = (
        self.proj_qkv.txt
        if not self.proj_qkv.shared_weights
        else self.proj_qkv.all
    )
    vid_norm_q = (
        self.norm_q.vid if not self.norm_q.shared_weights else self.norm_q.all
    )
    txt_norm_q = (
        self.norm_q.txt if not self.norm_q.shared_weights else self.norm_q.all
    )
    vid_norm_k = (
        self.norm_k.vid if not self.norm_k.shared_weights else self.norm_k.all
    )
    txt_norm_k = (
        self.norm_k.txt if not self.norm_k.shared_weights else self.norm_k.all
    )

    txt = txt.to(device=vid.device, dtype=vid.dtype)
    txt_qkv = txt_qkv_module(txt).reshape(
        txt.shape[0], 3, self.heads, self.head_dim
    )
    txt_q, txt_k, txt_v = txt_qkv.unbind(1)
    txt_q = txt_norm_q(txt_q)
    txt_k = txt_norm_k(txt_k)

    _, txt_freqs = self.rope.get_freqs(window_shape[:1], txt_shape)
    txt_freqs = txt_freqs.to(device=txt_q.device)
    txt_q = seedvr_model._apply_rope1_partial(
        txt_q.transpose(0, 1), txt_freqs
    ).transpose(0, 1)
    txt_k = seedvr_model._apply_rope1_partial(
        txt_k.transpose(0, 1), txt_freqs
    ).transpose(0, 1)

    txt_windows = txt.new_empty(
        (txt.shape[0], window_count, self.heads, self.head_dim)
    )
    for window_start, window_end, token_start, token_end in groups:
        group_order = window_order[token_start:token_end]
        vid_group = torch.index_select(vid, 0, group_order)
        vid_qkv = vid_qkv_module(vid_group).reshape(
            vid_group.shape[0], 3, self.heads, self.head_dim
        )
        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        vid_q = vid_norm_q(vid_q)
        vid_k = vid_norm_k(vid_k)

        group_window_shape = window_shape[window_start:window_end]
        group_txt_shape = txt_shape.repeat(window_end - window_start, 1)
        vid_freqs, _ = self.rope.get_freqs(
            group_window_shape, group_txt_shape
        )
        vid_freqs = vid_freqs.to(device=vid_q.device)
        vid_q = seedvr_model._apply_rope1_partial(
            vid_q.transpose(0, 1), vid_freqs
        ).transpose(0, 1)
        vid_k = seedvr_model._apply_rope1_partial(
            vid_k.transpose(0, 1), vid_freqs
        ).transpose(0, 1)

        group_vid_lengths = vid_lengths[window_start:window_end]
        group_vid_lengths_tensor = group_window_shape.prod(-1)
        group_window_count = window_end - window_start
        txt_length = txt.shape[0]
        all_lengths = [
            length + txt_length for length in group_vid_lengths
        ]
        q = seedvr_model.repeat_concat(
            vid_q,
            txt_q,
            group_vid_lengths_tensor,
            txt_shape.prod(-1),
            [group_window_count],
        )
        k = seedvr_model.repeat_concat(
            vid_k,
            txt_k,
            group_vid_lengths_tensor,
            txt_shape.prod(-1),
            [group_window_count],
        )
        v = seedvr_model.repeat_concat(
            vid_v,
            txt_v,
            group_vid_lengths_tensor,
            txt_shape.prod(-1),
            [group_window_count],
        )
        out = seedvr_model.optimized_var_attention(
            q=q,
            k=k,
            v=v,
            heads=self.heads,
            skip_reshape=True,
            skip_output_reshape=True,
            cu_seqlens_q=seedvr_model.cumulative_lengths(all_lengths),
            cu_seqlens_k=seedvr_model.cumulative_lengths(all_lengths),
        )

        vid_parts = []
        for relative_window, (window_out, vid_length) in enumerate(
            zip(out.split(all_lengths), group_vid_lengths)
        ):
            vid_parts.append(window_out[:vid_length])
            txt_windows[:, window_start + relative_window] = window_out[
                vid_length:
            ]
        vid.index_copy_(
            0, group_order, torch.cat(vid_parts).flatten(1, 2)
        )

    txt_out = txt_windows.mean(1).flatten(1, 2)
    vid_proj_out = (
        self.proj_out.vid
        if not self.proj_out.shared_weights
        else self.proj_out.all
    )
    vid_out = _chunked_forward(
        vid_proj_out, vid, _LINEAR_GROUP_ROWS
    )
    if not self.proj_out.vid_only:
        txt_proj_out = (
            self.proj_out.txt
            if not self.proj_out.shared_weights
            else self.proj_out.all
        )
        txt_out = txt_out.to(device=vid_out.device, dtype=vid_out.dtype)
        txt_out = txt_proj_out(txt_out)
    return vid_out, txt_out


def _patch_attention(seedvr_model):
    attention = getattr(seedvr_model, "NaSwinAttention", None)
    if attention is None:
        return False, "NaSwinAttention is unavailable"
    if getattr(attention, _PATCH_MARKER, False):
        return False, "NaSwinAttention is already patched"
    if not _source_has(attention.forward, _ATTENTION_SOURCE_CONTRACT):
        return False, "unsupported NaSwinAttention.forward contract"

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
                qkv_bytes = (
                    vid.shape[0]
                    * 3
                    * self.heads
                    * self.head_dim
                    * vid.element_size()
                )
                log.info(
                    "[OmniXPU] SeedVR2 bounded attention: "
                    "tokens=%d heads=%d head_dim=%d qkv_bytes=%d",
                    vid.shape[0],
                    self.heads,
                    self.head_dim,
                    qkv_bytes,
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


def _patch_swiglu(seedvr_model):
    swiglu = getattr(seedvr_model, "SwiGLUMLP", None)
    if swiglu is None:
        return False, "SwiGLUMLP is unavailable"
    if getattr(swiglu, _PATCH_MARKER, False):
        return False, "SwiGLUMLP is already patched"
    if not _source_has(swiglu.forward, _SWIGLU_SOURCE_CONTRACT):
        return False, "unsupported SwiGLUMLP.forward contract"

    original = swiglu.forward

    @functools.wraps(original)
    def forward(self, value):
        global _swiglu_logged
        hidden = int(self.proj_in_gate.weight.shape[0])
        rows = value.numel() // value.shape[-1]
        intermediate_bytes = rows * hidden * value.element_size()
        if (
            seedvr_model.comfy.model_management.in_training
            or value.requires_grad
            or intermediate_bytes <= _MLP_THRESHOLD_BYTES
        ):
            return original(self, value)

        row_bytes = (3 * hidden + value.shape[-1]) * value.element_size()
        chunk_rows = _chunk_rows(row_bytes, _MLP_GROUP_BYTES)
        if not _swiglu_logged:
            log.info(
                "[OmniXPU] SeedVR2 bounded SwiGLU: "
                "rows=%d hidden=%d chunk_rows=%d intermediate_bytes=%d",
                rows,
                hidden,
                chunk_rows,
                intermediate_bytes,
            )
            _swiglu_logged = True
        return _bounded_swiglu(self, value, chunk_rows)

    swiglu.forward = forward
    setattr(swiglu, _PATCH_MARKER, True)
    return True, None


def apply():
    try:
        from comfy.ldm.seedvr import model as seedvr_model
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy.ldm.seedvr", "comfy.ldm.seedvr.model"}:
            return False, "ComfyUI SeedVR2 model is not available"
        raise

    attention = getattr(seedvr_model, "NaSwinAttention", None)
    swiglu = getattr(seedvr_model, "SwiGLUMLP", None)
    if attention is None or swiglu is None:
        return False, "SeedVR2 attention or SwiGLU class is unavailable"
    if getattr(attention, _PATCH_MARKER, False) or getattr(
        swiglu, _PATCH_MARKER, False
    ):
        return False, "SeedVR2 capacity adapter is already applied"
    if not _source_has(attention.forward, _ATTENTION_SOURCE_CONTRACT):
        return False, "unsupported NaSwinAttention.forward contract"
    if not _source_has(swiglu.forward, _SWIGLU_SOURCE_CONTRACT):
        return False, "unsupported SwiGLUMLP.forward contract"

    attention_ok, attention_reason = _patch_attention(seedvr_model)
    swiglu_ok, swiglu_reason = _patch_swiglu(seedvr_model)
    if not attention_ok or not swiglu_ok:
        raise RuntimeError(
            "SeedVR2 capacity patch changed after preflight: "
            f"attention={attention_reason}, swiglu={swiglu_reason}"
        )
    return True, None
