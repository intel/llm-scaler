import logging
import importlib.util
import os
import sys

log = logging.getLogger("ComfyUI-OmniXPU")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Resolve package directory (works regardless of how we're loaded)
_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = "ComfyUI-OmniXPU"


def _load(rel_path, mod_name):
    """Load a .py file as a named module."""
    fpath = os.path.join(_DIR, *rel_path.split("/"))
    if os.path.isdir(fpath):
        fpath = os.path.join(fpath, "__init__.py")
    elif not fpath.endswith(".py"):
        fpath += ".py"
    search = [os.path.dirname(fpath)] if os.path.basename(fpath) == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(mod_name, fpath, submodule_search_locations=search)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        log.info("[OmniXPU] No Intel XPU detected, skipping all patches")
    else:
        # Load probe and run capability check
        probe = _load("probe.py", f"{_PKG}.probe")
        probe.probe()

        # Load config and patches
        _load("config.py", f"{_PKG}.config")
        patches = _load("patches", f"{_PKG}.patches")

        config_mod = sys.modules[f"{_PKG}.config"]
        patches.apply_all_patches(config_mod.config)

        # Load diagnostics node
        _load("nodes/__init__.py", f"{_PKG}.nodes")
        diag = _load("nodes/diagnostics.py", f"{_PKG}.nodes.diagnostics")
        NODE_CLASS_MAPPINGS["OmniXPUStatus"] = diag.OmniXPUStatus
        NODE_DISPLAY_NAME_MAPPINGS["OmniXPUStatus"] = "OmniXPU Status"

except Exception as e:
    import traceback
    log.error("[OmniXPU] Initialization failed: %s\n%s", e, traceback.format_exc())

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
