"""Device-free admission for the private AIMDO caller adapter.

Only allocation-sensitive control flow is fingerprinted. Comments, locations,
unrelated ComfyUI changes and the repository revision are not compatibility IDs.
This module must not import Torch or any ComfyUI execution module.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys


class CompatibilityError(RuntimeError):
    pass


_PREFETCH_FLOW = "0c79a754c654fcda4de53888ac85b372ec396be871410c2ba3628bae37df1396"
_GC_FLOW = "ba7450cfb3a3edf6aa4bb69fc79d07bae3e3e3840e5ec610f74dea08d81090d8"
_REQUEST_FLOW = "fc8418a5b601d2741e8c347d246d55ba4570f8274089fd370f5a907eccdfa729"
_QUEUE_FLAGS = "3eda23d7a1a71b4311eeddee0ff514384bcc39ee3140ddb8bbce300c0827b36c"
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


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise CompatibilityError(f"expected one {name} function")
    return matches[0]


def _signature(node, names):
    args = node.args
    if (tuple(a.arg for a in (*args.posonlyargs, *args.args)) != names
            or args.kwonlyargs or args.vararg or args.kwarg):
        raise CompatibilityError(f"unsupported signature: {node.name}")


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
        if _digest(prefetch) != _PREFETCH_FLOW:
            raise CompatibilityError("prefetch allocation boundary changed")
        worker = _function(trees["main.py"], "prompt_worker")
        blocks = [n for n in ast.walk(worker) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == "need_gc"]
        gc_blocks = [n for n in blocks if any(isinstance(c, ast.Call)
                     and ast.unparse(c.func) == "gc.collect" for c in ast.walk(n))]
        if len(gc_blocks) != 1 or _digest(gc_blocks[0]) != _GC_FLOW:
            raise CompatibilityError("prompt-worker GC/cache boundary changed")
        loops = [n for n in worker.body if isinstance(n, ast.While)]
        if len(loops) != 1:
            raise CompatibilityError("prompt-worker queue loop changed")
        starts = [i for i, n in enumerate(loops[0].body) if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "flags"]
        if (len(starts) != 1 or _digest(ast.Module(body=loops[0].body[starts[0]:], type_ignores=[]))
                != _REQUEST_FLOW):
            raise CompatibilityError("free-memory request semantics changed")
        calls = [ast.unparse(n.func) for stmt in loops[0].body
                 for n in ast.walk(stmt) if isinstance(n, ast.Call)]
        for required in ("q.get_flags", "e.reset", "gc.collect",
                         "comfy.model_management.soft_empty_cache"):
            if calls.count(required) != 1:
                raise CompatibilityError(f"prompt-worker request boundary changed: {required}")
        if not (calls.index("q.get_flags") < calls.index("e.reset") < calls.index("gc.collect")
                < calls.index("comfy.model_management.soft_empty_cache")):
            raise CompatibilityError("free-memory cleanup order changed")
        queue = next(n for n in trees["execution.py"].body
                     if isinstance(n, ast.ClassDef) and n.name == "PromptQueue")
        _signature(_function(queue, "get_flags"), ("self", "reset"))
        if _digest(_function(queue, "get_flags")) != _QUEUE_FLAGS:
            raise CompatibilityError("queue flag ownership/reset semantics changed")
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
