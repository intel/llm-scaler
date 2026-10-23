"""Refuse host-side KV/prefix caches on builds that cannot unmap host memory.

``QwenImage21Cache`` with ``device="cpu"`` (or ``"auto"`` falling back to host
under VRAM pressure) keeps the step-independent K/V in a ``PoseBranchCache``
that can live on the host.  ``adapters/qwen_image21_cache.py`` moves that host
storage to pinned memory so the runtime owns it; this adapter covers the case
where the host pages are never released at all.

Releasing the host side goes through ``comfy.model_management.unpin_memory``,
which calls ``torch.cuda.cudart().cudaHostUnregister``.  ``MAX_PINNED_MEMORY``
is only assigned for CUDA/ROCm, so on Intel XPU ``pin_memory``/``unpin_memory``
are no-ops and the driver keeps the host pages mapped in the process' GPU
address space: WDDM "Shared GPU memory" retains the whole cache until ComfyUI
is restarted.  Measured on Arc A770 16GB, Qwen-Image 2.1 edit, 1024x1024,
native int8 ConvRot UNETLoader, 12 steps, per-process
``\\GPU Process Memory(pid_*)\\Shared Usage``:

| host storage | peak during run | after the run |
|---|---|---|
| ComfyUI default (pageable) | 1073 MB | 1026 MB, never returns |
| pinned (as ``qwen_image21_cache.py`` stores it) | 2066 MB | 2050 MB, never returns |
| device side (``device="gpu"``) | 18 MB | 2 MB |

It survives ``POST /free {"unload_models": true, "free_memory": true}`` and
``torch.xpu.empty_cache``, and it reproduces with comfy-aimdo disabled and with
this whole custom node disabled, so the pageable case is ComfyUI's own
behaviour (upstream Comfy-Org/ComfyUI#16463).

The recompute path is ComfyUI's documented fallback when the cache cannot be
allocated and it retains nothing, so this adapter makes such builds take it:

  - ``device="cpu"``: the host store is refused, so it behaves like ``off``;
  - ``device="auto"``: only a decision that lands on host is dropped - the
    device-side cache is still used while there is spare VRAM;
  - ``device="gpu"``: untouched.

The guard is only installed when the compute device is XPU, so CUDA/ROCm runs
are untouched even with ``--disable-pinned-memory`` (which also leaves
``MAX_PINNED_MEMORY`` at ``-1``).  Anything unexpected inside the guard falls
through to the upstream call.

Env ``OMNIXPU_HOST_KV_CACHE``: ``1`` (default) guard on, ``0`` upstream
behaviour (host cache, retained mapping).
"""

import logging
import os

log = logging.getLogger("ComfyUI-OmniXPU")


def host_cache_releasable() -> bool:
    """True when host pages used for a KV cache can be unregistered later."""

    try:
        import comfy.model_management as mm
    except Exception:  # pragma: no cover - depends on host ComfyUI
        return True
    return getattr(mm, "MAX_PINNED_MEMORY", -1) > 0


def apply():
    if os.environ.get("OMNIXPU_HOST_KV_CACHE", "1") == "0":
        return False, "disabled by env"

    try:
        import comfy.model_management as mm
    except Exception as exc:  # pragma: no cover - depends on host ComfyUI
        return False, f"comfy.model_management unavailable: {exc}"
    # The retained mapping is an XPU-only observation.  Gate on the device so a
    # CUDA/ROCm run never changes behaviour, not even with
    # --disable-pinned-memory (which also leaves MAX_PINNED_MEMORY at -1).
    if getattr(mm.get_torch_device(), "type", None) != "xpu":
        return False, "requires XPU"
    if host_cache_releasable():
        return False, "host memory is registerable; upstream host cache kept"

    try:
        import comfy.ldm.qwen_image21.model as qwen_model
    except Exception as exc:  # pragma: no cover - depends on host ComfyUI
        return False, f"qwen_image21 model unavailable: {exc}"

    cls = getattr(qwen_model, "QwenImage21Transformer2DModel", None)
    if cls is None or not hasattr(cls, "select_prefix_cache"):
        return False, "QwenImage21Transformer2DModel.select_prefix_cache unavailable"
    if getattr(cls.select_prefix_cache, "_omnixpu_host_guard", False):
        return False, "already guarded"

    original = cls.select_prefix_cache

    def select_prefix_cache(self, key, cache_bytes, device, options):
        try:
            if options.get("device") == "cpu" and not host_cache_releasable():
                return None, False
        except Exception:  # pragma: no cover - never break the forward pass
            return original(self, key, cache_bytes, device, options)
        cache, filled = original(self, key, cache_bytes, device, options)
        try:
            if (cache is not None and not host_cache_releasable()
                    and getattr(cache, "store_device", None) is not None
                    and cache.store_device.type == "cpu"):
                # `auto` fell back to host: drop it before any K/V is stored.
                cache.free()
                if self.prefix_cache is cache:
                    self.prefix_cache = None
                return None, False
        except Exception:  # pragma: no cover - keep upstream behaviour
            return cache, filled
        return cache, filled

    select_prefix_cache._omnixpu_host_guard = True
    cls.select_prefix_cache = select_prefix_cache
    return (
        True,
        "host KV cache refused where host memory cannot be unregistered "
        "(Comfy-Org/ComfyUI#16463); device-side cache unchanged",
    )
