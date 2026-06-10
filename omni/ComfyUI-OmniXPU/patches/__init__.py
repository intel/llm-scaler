import logging

log = logging.getLogger("ComfyUI-OmniXPU")

# Patch registry: ordered list of (name, status, reason)
_registry = []


def _record(name, status, reason=""):
    _registry.append({"name": name, "status": status, "reason": reason})
    if status == "applied":
        log.info("[OmniXPU] %s: applied", name)
    elif status == "skipped":
        log.info("[OmniXPU] %s: skipped (%s)", name, reason)
    elif status == "failed":
        log.warning("[OmniXPU] %s: FAILED (%s)", name, reason)


def get_status():
    return list(_registry)


def apply_all_patches(cfg):
    import importlib, os, sys

    _patches_dir = os.path.dirname(os.path.abspath(__file__))
    _pkg_name = os.path.basename(os.path.dirname(_patches_dir))

    def _load_patch(name):
        fpath = os.path.join(_patches_dir, name + ".py")
        mod_name = f"{_pkg_name}.patches.{name}"
        spec = importlib.util.spec_from_file_location(mod_name, fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.apply

    apply_interpolate = _load_patch("patch_interpolate")
    apply_fp8_fix = _load_patch("patch_fp8_fix")
    apply_norm = _load_patch("patch_norm")
    apply_rope = _load_patch("patch_rope")
    apply_fp8_gemm = _load_patch("patch_fp8_gemm")
    apply_attention = _load_patch("patch_attention")

    _apply_one("interpolate_fix", cfg.interpolate_fix, apply_interpolate)
    _apply_one("fp8_neg_zero_fix", cfg.fp8_neg_zero_fix, apply_fp8_fix)
    _apply_one("norm", cfg.norm, apply_norm)
    _apply_one("rope", cfg.rope, apply_rope)
    _apply_one("fp8_gemm", cfg.fp8_gemm, apply_fp8_gemm)
    _apply_one("attention", cfg.attention, apply_attention)


def _apply_one(name, enabled, apply_fn):
    if not enabled:
        _record(name, "skipped", "disabled by env")
        return
    try:
        ok, reason = apply_fn()
        if ok:
            _record(name, "applied")
        else:
            _record(name, "skipped", reason)
    except Exception as e:
        _record(name, "failed", str(e))
