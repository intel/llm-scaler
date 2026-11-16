"""Qwen3.5 vision attention: CuTe FP32 first, bounded FP32 fallback."""
import functools
import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")
_state = {"installed": False, "cute": 0, "sub_quad": 0}


def status():
    return dict(_state)


def install():
    from comfy.ldm.modules import attention
    try:
        from comfy.text_encoders import qwen35
    except ImportError:
        _state["reason"] = "Qwen3.5 vision module unavailable"
        return
    from omni_xpu_kernel import cute

    cls = getattr(qwen35, "Qwen35VisionAttention", None)
    if cls is None:
        _state["reason"] = "Qwen3.5 vision class unavailable"
        return
    original = cls.forward
    if getattr(original, "_omnixpu_vision_attention", False):
        return
    stock = (attention.attention_pytorch, attention.attention_basic)
    capable = getattr(cute, "supports_fp32_d72_bhld", lambda: False)()
    _state.update(installed=True, kernel_available=bool(capable))

    def dispatch(upstream, q, k, v, heads, **kwargs):
        # Leave custom attention policies and non-FP32 callers in control.
        if upstream not in stock or q.device.type != "xpu" or q.dtype != torch.float32:
            return upstream(q, k, v, heads, **kwargs)
        native = (
            capable
            and kwargs.get("skip_reshape", False)
            and not set(kwargs).difference({"skip_reshape", "skip_output_reshape", "attn_precision"})
            and q.ndim == k.ndim == v.ndim == 4
            and k.device == v.device == q.device
            and k.dtype == v.dtype == q.dtype
            and q.shape[1] == heads
            and q.shape[0] == k.shape[0] == v.shape[0]
            and q.shape[1] == k.shape[1] == v.shape[1]
            and q.shape[-1] == k.shape[-1] == v.shape[-1] == 72
            and k.shape[2] == v.shape[2]
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        )
        route = "cute" if native else "sub_quad"
        _state[route] += 1
        if _state[route] == 1:
            log.info("[OmniXPU] Qwen3.5 FP32 vision attention: %s, shape=%s, CUTE available=%s",
                     route, tuple(q.shape), capable)
        if native:
            # Native errors propagate; they must not silently disable a fast path.
            out = cute.sdp_bhld_fp32_d72(q, k, v)
            if kwargs.get("skip_output_reshape", False):
                return out
            return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return attention.attention_sub_quad(q, k, v, heads, **kwargs)

    @functools.wraps(original)
    def forward(self, x, cu_seqlens, position_embeddings, optimized_attention=None):
        chosen = optimized_attention
        if chosen in stock:
            chosen = functools.partial(dispatch, chosen)
        return original(self, x, cu_seqlens, position_embeddings, optimized_attention=chosen)

    forward._omnixpu_vision_attention = True
    cls.forward = forward
