"""Keep Qwen3.5 GDN prefill L2 normalization intermediates in FP32."""

from types import FunctionType

import torch


_MARKER = "_omnixpu_gdn_precision_original"


def _l2norm_fp32(x, eps):
    # The mixed-input reduction casts in the kernel, without materializing x.float().
    denominator = torch.linalg.vector_norm(
        x, ord=2, dim=-1, keepdim=True, dtype=torch.float32,
    ).clamp_min(eps)
    return x / denominator


class _Functional:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def normalize(self, input, p=2.0, dim=1, eps=1e-12, out=None):
        if (input.is_xpu and input.dtype in (torch.float16, torch.bfloat16)
                and p == 2 and dim == -1 and out is None):
            return _l2norm_fp32(input, eps)
        return self.original.normalize(input, p=p, dim=dim, eps=eps, out=out)


def apply():
    try:
        from comfy.text_encoders import qwen35
    except ImportError:
        return False, "Qwen3.5 text encoder is unavailable"
    original = qwen35.torch_chunk_gated_delta_rule
    if hasattr(original, _MARKER):
        return True, None
    if (not isinstance(original, FunctionType)
            or "F" not in original.__globals__
            or "normalize" not in original.__code__.co_names):
        return False, "GDN chunk no longer exposes functional normalization"

    # Reuse the upstream algorithm and its initial/final dtype boundary. Only
    # this function sees the precision adapter; global F.normalize is untouched.
    namespace = dict(original.__globals__)
    namespace["F"] = _Functional(namespace["F"])
    patched = FunctionType(
        original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__,
    )
    patched.__kwdefaults__ = original.__kwdefaults__
    patched.__annotations__ = dict(original.__annotations__)
    patched.__qualname__ = original.__qualname__
    patched.__module__ = original.__module__
    patched.__doc__ = original.__doc__
    patched.__dict__.update(original.__dict__)
    setattr(patched, _MARKER, original)
    qwen35.torch_chunk_gated_delta_rule = patched
    return True, None
