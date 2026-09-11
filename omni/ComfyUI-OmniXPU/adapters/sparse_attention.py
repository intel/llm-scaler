"""Let upstream block-sparse eligibility recognize Kitchen's native XPU API."""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from functools import update_wrapper
from pathlib import Path

_MARKER = "__omnixpu_sparse_eligibility_original__"


def _extend_device_guard(function, tensor_name):
    """Change one explicit CUDA exclusion while retaining the upstream body."""
    if function.__closure__:
        raise ValueError("upstream eligibility unexpectedly captures closure state")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("upstream eligibility source is not a function")
    definition = tree.body[0]
    if definition.name != function.__name__ or definition.decorator_list:
        raise ValueError("upstream eligibility definition is unsupported")
    guards = []
    expected = ast.dump(ast.parse(f"{tensor_name}.device.type", mode="eval").body)
    for node in ast.walk(definition):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.NotEq)
                and ast.dump(node.left) == expected
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "cuda"):
            guards.append(node)
    if len(guards) != 1:
        raise ValueError("expected exactly one explicit CUDA device exclusion")
    guards[0].ops = [ast.NotIn()]
    guards[0].comparators = [ast.Tuple(
        elts=[ast.Constant("cuda"), ast.Constant("xpu")], ctx=ast.Load())]
    for node in ast.walk(definition):
        if isinstance(node, ast.Constant) and node.value == "not on CUDA":
            node.value = "not on CUDA or XPU"
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(function), "exec"),
         function.__globals__, namespace)
    adapted = update_wrapper(namespace[function.__name__], function)
    setattr(adapted, _MARKER, function)
    return adapted


def _upstream_module():
    # ComfyUI's file loader registers this built-in as nodes_sparse_attention,
    # whereas a normal package import creates a separate module object.
    # Follow the registered node class so its closures use the adapted guards.
    nodes = sys.modules.get("nodes")
    node = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("BlockSparseAttention")
    if node is not None:
        module = sys.modules.get(node.__module__)
        if module is None or getattr(module, "BlockSparseAttention", None) is not node:
            raise ValueError("registered sparse node has no owning module")
        expected = Path(nodes.__file__).resolve().parent / "comfy_extras/nodes_sparse_attention.py"
        if Path(module.__file__).resolve() != expected:
            raise ValueError("registered sparse node is not the upstream built-in")
        return module
    from comfy_extras import nodes_sparse_attention
    return nodes_sparse_attention


def apply():
    try:
        import comfy_kitchen as ck
        if not ck.sol_attn_is_available():
            return False, "complete native XPU Sol API is unavailable"
        upstream = _upstream_module()
        functions = (upstream._ineligible, upstream.h3_eligible)
        if all(hasattr(f, _MARKER) for f in functions):
            return True, "already patched"
        if any(hasattr(f, _MARKER) for f in functions):
            return False, "upstream sparse eligibility is partially patched"
        # Build both replacements before changing either module binding.
        generic = _extend_device_guard(functions[0], "q")
        h3 = _extend_device_guard(functions[1], "x")
    except (ImportError, AttributeError, OSError, TypeError, ValueError, SyntaxError) as exc:
        return False, f"upstream sparse eligibility is unsupported: {exc}"
    upstream._ineligible, upstream.h3_eligible = generic, h3
    return True, ""


__all__ = ["apply"]
