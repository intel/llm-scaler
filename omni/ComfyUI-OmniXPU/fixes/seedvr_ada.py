"""Apply the explicit SeedVR2 Ada embedding reshape at runtime.

The patch is intentionally source guarded.  It changes only the known
upstream reshape expression and declines to patch an unknown implementation.
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
_PATCH_MARKER = "_omnixpu_seedvr_ada_reshape_patched"


def _rewrite_forward(forward):
    """Return a guarded copy of ``forward`` with the explicit Ada layout."""
    if forward.__code__.co_freevars:
        return None, "AdaSingle.forward has unsupported closure state"

    try:
        source = textwrap.dedent(inspect.getsource(forward))
    except (OSError, TypeError) as exc:
        return None, f"AdaSingle.forward source unavailable: {exc}"

    if _EXPLICIT_RESHAPE in source:
        return None, "upstream Ada reshape is already explicit"
    if source.count(_LEGACY_RESHAPE) != 1:
        return None, "unsupported AdaSingle.forward reshape contract"

    source = source.replace(_LEGACY_RESHAPE, _EXPLICIT_RESHAPE)
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
