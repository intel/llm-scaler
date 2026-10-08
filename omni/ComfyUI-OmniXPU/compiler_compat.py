"""Device-free admission for the private AIMDO caller adapter.

Admission checks the interfaces and ordering needed by the runtime wrappers.
Source fingerprints and the repository revision are not compatibility IDs.
This module must not import Torch or any ComfyUI execution module.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


class CompatibilityError(RuntimeError):
    pass


SIGNATURES = {
    "comfy/model_management.py": {
        "get_cast_buffer": ("offload_stream", "device", "size", "ref"),
        "cast_to_gathered": ("tensors", "r", "non_blocking", "stream", "r2"),
        "cast_to": ("weight", "dtype", "device", "non_blocking", "copy", "stream", "r"),
        "soft_empty_cache": ("force",),
    },
    "comfy/model_prefetch.py": {
        "malloc_graph_enabled": ("device",), "malloc_graph_begin": ("device",),
        "malloc_graph_end": (), "cleanup_malloc_graph": (), "pause_malloc_graph": ("sync",),
        "prefetch_queue_pop": ("queue", "device", "module", "dtype", "core", "enable_graph", "generator", "malloc_scope"),
    },
}


def _function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise CompatibilityError(f"expected one {name} function")
    return matches[0]


def _signature(node, names):
    args = node.args
    positional = [a.arg for a in (*args.posonlyargs, *args.args)]
    parameters = positional + [a.arg for a in args.kwonlyargs]
    required = positional[:len(positional) - len(args.defaults)] + [
        a.arg for a, default in zip(args.kwonlyargs, args.kw_defaults) if default is None]
    if not set(names) <= set(parameters) or not set(required) <= set(names):
        raise CompatibilityError(f"unsupported signature: {node.name}")


def _aliases(node):
    """Resolve simple local aliases; local names are not an interface contract."""
    aliases = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Name):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = child.value.id
    return aliases


def _name(node, aliases):
    if not isinstance(node, ast.Name):
        return None
    name, seen = node.id, set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _calls(node):
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)),
                  key=lambda n: (n.lineno, n.col_offset))


def _method(call, method):
    return isinstance(call.func, ast.Attribute) and call.func.attr == method


def _prefetch_contract(function):
    aliases = _aliases(function)
    calls = _calls(function)
    graphs = {target.id for node in ast.walk(function) if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Call) and _method(node.value, "get")
              and _name(node.value.func.value, aliases) == "MALLOC_GRAPHS"
              for target in node.targets if isinstance(target, ast.Name)}
    iterations = [c for c in calls if _method(c, "iterate")
                  and _name(c.func.value, aliases) in graphs]
    pops = [c for c in calls if _method(c, "pop") and _name(c.func.value, aliases) == "queue"]
    if not graphs or not iterations or len(pops) != 1:
        raise CompatibilityError("prefetch registry/iterate/queue interface changed")
    pop = pops[0]
    if len(pop.args) != 1 or not isinstance(pop.args[0], ast.Constant) or pop.args[0].value != 0:
        raise CompatibilityError("prefetch allocation boundary changed: expected queue.pop(0)")
    housekeeping = [c for c in calls if _name(c.func, aliases) in
                    {"pin_modules", "cleanup_prefetched_modules"}]
    if not housekeeping or max(c.lineno for c in iterations) >= min(pop.lineno, *(c.lineno for c in housekeeping)):
        raise CompatibilityError("prefetch housekeeping must follow graph iterate")
    cores = [c for c in calls if _name(c.func, aliases) == "core"]
    if not any(c.lineno > max(pop.lineno, *(c.lineno for c in housekeeping)) for c in cores):
        raise CompatibilityError("prefetch core must follow housekeeping")


def _request_contract(tree):
    worker = _function(tree, "prompt_worker")
    calls = _calls(worker)
    flags = [c for c in calls if _method(c, "get_flags")]
    resets = [c for c in calls if _method(c, "reset")]
    collects = [c for c in calls if _method(c, "collect") and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "gc"]
    flushes = [c for c in calls if _method(c, "soft_empty_cache")]
    if len(flags) != 1 or not resets or len(collects) != 1 or not flushes:
        raise CompatibilityError("prompt-worker flag/reset/GC/flush interface changed")
    flag, collect = flags[0], collects[0]
    if ((flag.args and isinstance(flag.args[0], ast.Constant) and not flag.args[0].value)
            or any(k.arg == "reset" and isinstance(k.value, ast.Constant) and not k.value.value for k in flag.keywords)):
        raise CompatibilityError("prompt-worker must consume queue flags")
    if not flag.lineno < min(c.lineno for c in resets) < collect.lineno < max(c.lineno for c in flushes):
        raise CompatibilityError("free-memory cleanup order changed")


def validate_live_signature(function, names):
    import inspect
    parameters = inspect.signature(function).parameters
    required = {name for name, p in parameters.items() if p.default is p.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    if not set(names) <= parameters.keys() or not required <= set(names):
        raise CompatibilityError(f"live caller signature changed: {function.__name__}")


def find_root():
    for name in ("__main__", "main"):
        path = getattr(sys.modules.get(name), "__file__", None)
        if path and Path(path).name == "main.py":
            root = Path(path).resolve().parent
            if (root / "comfy/model_prefetch.py").is_file():
                return root
    root = Path(__file__).resolve().parents[2]
    if (root / "main.py").is_file() and (root / "comfy/model_prefetch.py").is_file():
        return root
    raise CompatibilityError("ComfyUI source root unavailable")


def preflight(root=None):
    root = Path(root) if root is not None else find_root()
    trees = {}
    try:
        for path in (*SIGNATURES, "main.py", "execution.py"):
            trees[path] = ast.parse((root / path).read_text(encoding="utf-8"))
        for path, functions in SIGNATURES.items():
            for name, names in functions.items():
                _signature(_function(trees[path], name), names)
        prefetch = _function(trees["comfy/model_prefetch.py"], "prefetch_queue_pop")
        _prefetch_contract(prefetch)
        _request_contract(trees["main.py"])
        queue = next(n for n in trees["execution.py"].body
                     if isinstance(n, ast.ClassDef) and n.name == "PromptQueue")
        flag_method = _function(queue, "get_flags")
        _signature(flag_method, ("self", "reset"))
        positional = [a.arg for a in (*flag_method.args.posonlyargs, *flag_method.args.args)]
        defaults = dict(zip(positional[-len(flag_method.args.defaults):], flag_method.args.defaults))
        defaults.update((a.arg, d) for a, d in zip(flag_method.args.kwonlyargs, flag_method.args.kw_defaults))
        reset = defaults.get("reset")
        if not isinstance(reset, ast.Constant) or reset.value is not True:
            raise CompatibilityError("PromptQueue.get_flags must default to consuming flags")
    except (OSError, SyntaxError, StopIteration) as exc:
        raise CompatibilityError(f"ComfyUI caller source unavailable: {exc}") from exc
    return str(root.resolve())


def preflight_dependencies():
    """Inspect kernel Python API without importing Torch or loading its DSO."""
    spec = importlib.util.find_spec("omni_xpu_kernel")
    if spec is None or not spec.origin:
        raise CompatibilityError("OmniXPU kernel package unavailable")
    path = Path(spec.origin).parent / "int8/__init__.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in ("set_allocation_context_factory", "clear_convrot_hadamard_cache",
                     "release_onednn_int8_cache"):
            _function(tree, name)
    except (OSError, SyntaxError) as exc:
        raise CompatibilityError(f"kernel cache API unavailable: {exc}") from exc
