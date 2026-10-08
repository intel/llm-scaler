"""RED/GREEN test for the _is_uniform_decode shape-alias fix (vllm PR #53059).

Extracts the function from the stock and patched gpu_model_runner.py files
and runs the PR's regression cases against each. The aliased-prefill case
MUST fail on stock (bug present) and pass on patched (fix works).
"""
import ast
import textwrap
import types


def extract_fn(path, name):
    src = open(path).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise LookupError(name)


STOCK = extract_fn("/home/dom/fixes/qwen38-vllm/gpu_model_runner.py.stock",
                   "_is_uniform_decode")
PATCHED = extract_fn("/home/dom/fixes/qwen38-vllm/gpu_model_runner.py",
                     "_is_uniform_decode")


def make_self(computed, prompt):
    """Minimal stand-in for `self` with the input_batch arrays."""
    import numpy as np
    return types.SimpleNamespace(input_batch=types.SimpleNamespace(
        num_computed_tokens_cpu=__import__("numpy").array(computed),
        num_prompt_tokens=__import__("numpy").array(prompt),
    ))


def run(fn_src, self_obj, **kw):
    ns = {}
    exec("import numpy\n" + textwrap.dedent(fn_src), ns)
    if "self" in ns["_is_uniform_decode"].__code__.co_varnames:
        return ns["_is_uniform_decode"](self_obj, **kw)
    return ns["_is_uniform_decode"](**kw)


CASES = [
    # (name, computed, prompt, kwargs, stock_expected, patched_expected)
    ("genuine decode 16x1", [10] * 16, [8] * 16,
     dict(max_num_scheduled_tokens=1, uniform_decode_query_len=1, num_tokens=16, num_reqs=16),
     True, True),
    ("shape mismatch 2x1", [10] * 16, [8] * 16,
     dict(max_num_scheduled_tokens=2, uniform_decode_query_len=1, num_tokens=16, num_reqs=16),
     False, False),
    ("total mismatch", [10] * 16, [8] * 16,
     dict(max_num_scheduled_tokens=1, uniform_decode_query_len=1, num_tokens=8, num_reqs=16),
     False, False),
    ("genuine spec decode 5-q", [10] * 7, [8] * 7,
     dict(max_num_scheduled_tokens=5, uniform_decode_query_len=5, num_tokens=30, num_reqs=6),
     True, True),
    ("spec shape mismatch", [10] * 7, [8] * 7,
     dict(max_num_scheduled_tokens=5, uniform_decode_query_len=4, num_tokens=30, num_reqs=6),
     False, False),
    ("spec total mismatch", [10] * 7, [8] * 7,
     dict(max_num_scheduled_tokens=5, uniform_decode_query_len=5, num_tokens=36, num_reqs=6),
     False, False),
    # --- THE BUG CASES: aliased prefill shapes ---
    ("ALIASED 3-token prefill (k=2)", [0], [3],
     dict(max_num_scheduled_tokens=3, uniform_decode_query_len=3, num_tokens=3, num_reqs=1),
     True, False),
    ("ALIASED 2-token prefill (k=1, OURS)", [0], [2],
     dict(max_num_scheduled_tokens=2, uniform_decode_query_len=2, num_tokens=2, num_reqs=1),
     True, False),
    ("ALIASED chunked-prefill last chunk", [5], [8],
     dict(max_num_scheduled_tokens=3, uniform_decode_query_len=3, num_tokens=3, num_reqs=1),
     True, False),
    ("ALIASED mixed decode+prefill", [5, 0], [3, 3],
     dict(max_num_scheduled_tokens=3, uniform_decode_query_len=3, num_tokens=6, num_reqs=2),
     True, False),
    ("ALIASED 1-token prompt no-spec", [0], [1],
     dict(max_num_scheduled_tokens=1, uniform_decode_query_len=1, num_tokens=1, num_reqs=1),
     True, False),
    ("same shape, past prompt (decode)", [5], [3],
     dict(max_num_scheduled_tokens=3, uniform_decode_query_len=3, num_tokens=3, num_reqs=1),
     True, True),
    ("force=True (capture path)", [0], [3],
     dict(max_num_scheduled_tokens=3, uniform_decode_query_len=3, num_tokens=3, num_reqs=1,
          force_uniform_decode=True),
     True, True),
    ("force=False", [10] * 4, [8] * 4,
     dict(max_num_scheduled_tokens=1, uniform_decode_query_len=1, num_tokens=4, num_reqs=4,
          force_uniform_decode=False),
     False, False),
]

red_fails, green_fails = [], []
for name, comp, prom, kw, stock_exp, patched_exp in CASES:
    s = run(STOCK, make_self(comp, prom), **kw)
    p = run(PATCHED, make_self(comp, prom), **kw)
    ok_s = (s == stock_exp)
    ok_p = (p == patched_exp)
    tag = ""
    if not ok_s:
        red_fails.append(name); tag += " <<RED-FAIL(stock wrong)"
    if not ok_p:
        green_fails.append(name); tag += " <<GREEN-FAIL(patch wrong)"
    print(f"{name:38} stock={s!r:5}(want {stock_exp!r:5}) patched={p!r:5}(want {patched_exp!r:5}){tag}")

print()
print(f"RED  (stock must get bug cases WRONG): {'PASS' if not red_fails else 'FAIL ' + str(red_fails)}")
print(f"GREEN (patched must get all right):    {'PASS' if not green_fails else 'FAIL ' + str(green_fails)}")
