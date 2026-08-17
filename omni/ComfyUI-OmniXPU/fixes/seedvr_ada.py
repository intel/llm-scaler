"""Bound SeedVR2 Ada modulation and apply its explicit embedding reshape.

The patch is intentionally source guarded.  It changes only the known
upstream expressions and declines to patch an unknown implementation.
"""

import functools
import inspect
import textwrap

_LEGACY_RESHAPE = (
    "emb.reshape(emb.shape[0], -1, len(self.layers), 3)"
)
_EXPLICIT_RESHAPE = (
    "emb.reshape(emb.shape[0], self.dim, -1, 3)"
)
_LEGACY_REPEAT = """    if hid_len is not None:
        emb = cache(
            f"emb_repeat_{idx}_{branch_tag}",
            lambda: torch.repeat_interleave(emb, hid_len, dim=0),
        )

"""
_LEGACY_IN = """    if mode == "in":
        shiftB = comfy.ops.cast_to_input(shiftB, hid)
        scaleB = comfy.ops.cast_to_input(scaleB, hid)
        return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
"""
_BOUNDED_IN = """    if mode == "in":
        shiftB = comfy.ops.cast_to_input(shiftB, hid)
        scaleB = comfy.ops.cast_to_input(scaleB, hid)
        if hid_len is not None and emb.shape[0] > 1:
            offset = 0
            for length, shift, scale in zip(hid_len.tolist(), shiftA, scaleA):
                hid[offset:offset + length].mul_(scale + scaleB).add_(shift + shiftB)
                offset += length
            return hid
        return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
"""
_LEGACY_OUT = """    if mode == "out":
        if gateB is not None:
            gateB = comfy.ops.cast_to_input(gateB, hid)
            return hid.mul_(gateA + gateB)
        else:
            return hid.mul_(gateA)
"""
_BOUNDED_OUT = """    if mode == "out":
        if gateB is not None:
            gateB = comfy.ops.cast_to_input(gateB, hid)
        if hid_len is not None and emb.shape[0] > 1:
            offset = 0
            for length, gate in zip(hid_len.tolist(), gateA):
                hid[offset:offset + length].mul_(gate if gateB is None else gate + gateB)
                offset += length
            return hid
        return hid.mul_(gateA if gateB is None else gateA + gateB)
"""
_PATCH_MARKER = "_omnixpu_seedvr_ada_reshape_patched"


def _rewrite_forward(forward):
    """Return a guarded copy of ``forward`` with the explicit Ada layout."""
    if forward.__code__.co_freevars:
        return None, "AdaSingle.forward has unsupported closure state"

    try:
        source = textwrap.dedent(inspect.getsource(forward))
    except (OSError, TypeError) as exc:
        return None, f"AdaSingle.forward source unavailable: {exc}"

    reshape_changed = source.count(_LEGACY_RESHAPE) == 1
    if reshape_changed:
        source = source.replace(_LEGACY_RESHAPE, _EXPLICIT_RESHAPE)
    elif source.count(_EXPLICIT_RESHAPE) != 1:
        return None, "unsupported AdaSingle.forward reshape contract"

    if _LEGACY_REPEAT in source:
        if source.count(_LEGACY_IN) != 1 or source.count(_LEGACY_OUT) != 1:
            return None, "unsupported AdaSingle.forward modulation contract"
        source = source.replace(_LEGACY_REPEAT, "")
        source = source.replace(_LEGACY_IN, _BOUNDED_IN)
        source = source.replace(_LEGACY_OUT, _BOUNDED_OUT)
    elif "torch.repeat_interleave(emb, hid_len, dim=0)" in source:
        return None, "unsupported AdaSingle.forward repeat contract"
    elif _EXPLICIT_RESHAPE in source and not reshape_changed:
        return None, "upstream Ada reshape and broadcast are already bounded"

    namespace = {}
    filename = inspect.getsourcefile(forward) or "<seedvr_ada.py>"
    exec(  # noqa: S102 - exact, source-guarded upstream function rewrite
        compile(source, f"{filename}:OmniXPU-Ada-reshape", "exec"),
        forward.__globals__,
        namespace,
    )
    patched = namespace.get(forward.__name__)
    if patched is None:
        return None, "rewritten AdaSingle.forward was not defined"
    functools.update_wrapper(patched, forward)
    return patched, None


def apply():
    try:
        from comfy.ldm.seedvr import model as seedvr_model
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy.ldm.seedvr", "comfy.ldm.seedvr.model"}:
            return False, "ComfyUI SeedVR2 model is not available"
        raise

    ada_single = getattr(seedvr_model, "AdaSingle", None)
    if ada_single is None or not hasattr(ada_single, "forward"):
        return False, "ComfyUI SeedVR2 AdaSingle is not available"
    if getattr(ada_single, _PATCH_MARKER, False):
        return False, "SeedVR2 Ada reshape patch is already applied"

    patched, reason = _rewrite_forward(ada_single.forward)
    if patched is None:
        return False, reason

    ada_single.forward = patched
    setattr(ada_single, _PATCH_MARKER, True)
    return True, None
