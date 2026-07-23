import logging
from dataclasses import dataclass
from pathlib import Path

from .debug import debug_enabled, verbose_debug_enabled

log = logging.getLogger("ComfyUI-OmniXPU")

# Runtime component status for the diagnostics node.
_registry = []


@dataclass(frozen=True)
class Component:
    name: str
    flag: str
    kind: str
    owner: str
    module: str


# Generic XPU operators are intentionally absent: comfy_kitchen owns their
# registration, capability checks, dispatch, and eager fallback.
COMPONENTS = (
    Component(
        "attention_adapter",
        "attention",
        "adapter",
        "ComfyUI-OmniXPU",
        "adapters/attention.py",
    ),
    Component(
        "norm_adapter",
        "norm",
        "adapter",
        "ComfyUI-OmniXPU",
        "adapters/norm.py",
    ),
    Component(
        "fp8_model_adapter",
        "fp8_gemm",
        "adapter",
        "ComfyUI-OmniXPU",
        "adapters/fp8_gemm.py",
    ),
    Component(
        "int8_ffn_adapter",
        "int8_ffn",
        "adapter",
        "ComfyUI-OmniXPU",
        "adapters/int8_ffn.py",
    ),
    Component(
        "legacy_interpolate_fix",
        "interpolate_fix",
        "legacy_fix",
        "upstream_pending",
        "fixes/legacy_interpolate.py",
    ),
    Component(
        "legacy_median_fix",
        "median_fix",
        "legacy_fix",
        "upstream_pending",
        "fixes/legacy_median.py",
    ),
)


def _record(component, status, reason=""):
    _registry.append(
        {
            "name": component.name,
            "kind": component.kind,
            "owner": component.owner,
            "module": component.module,
            "status": status,
            "reason": reason,
        }
    )
    if status == "applied":
        log.info("[OmniXPU] %s: applied", component.name)
    elif status == "skipped":
        log.info("[OmniXPU] %s: skipped (%s)", component.name, reason)
    elif status == "failed":
        log.warning("[OmniXPU] %s: FAILED (%s)", component.name, reason)


def get_status():
    return list(_registry)


def get_components():
    return [
        {
            "name": component.name,
            "flag": component.flag,
            "kind": component.kind,
            "owner": component.owner,
            "module": component.module,
        }
        for component in COMPONENTS
    ]


def apply_all_patches(cfg):
    import importlib
    import sys

    _registry.clear()
    if verbose_debug_enabled():
        log.info("[OmniXPU] verbose debug tracing enabled (dispatch + kernel)")
    elif debug_enabled():
        log.info("[OmniXPU] debug tracing enabled (kernel only)")

    plugin_dir = Path(__file__).resolve().parent.parent
    pkg_name = __name__.rsplit(".", 1)[0]

    def _load_component(component):
        relative = Path(component.module)
        fpath = plugin_dir / relative
        mod_name = f"{pkg_name}.{'.'.join(relative.with_suffix('').parts)}"
        spec = importlib.util.spec_from_file_location(mod_name, fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.apply

    for component in COMPONENTS:
        if not getattr(cfg, component.flag):
            _record(component, "skipped", "disabled by env")
            continue
        try:
            apply_fn = _load_component(component)
            ok, reason = apply_fn()
            if ok:
                _record(component, "applied")
            else:
                _record(component, "skipped", reason or "")
        except Exception as exc:
            _record(component, "failed", str(exc))
