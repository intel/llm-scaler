"""Static and CPU-only contracts for custom ESIMD DSO ownership."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_PRODUCTION = ROOT / "setup.py"
SETUP_PLE_ONLY = ROOT / "setup_ple_only.py"
SETUP_GEMV_ONLY = ROOT / "setup_gemv_only.py"
SETUP_SYCL = ROOT / "setup_sycl.py"
ESIMD_UTILS = (
    ROOT.parents[2]
    / "applications.ai.gpu.llm-scaler-vllm"
    / "vllm/model_executor/layers/esimd_utils.py"
)


def _load_esimd_utils():
    spec = importlib.util.spec_from_file_location(
        "esimd_utils_owner_contract", ESIMD_UTILS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_call(path: Path) -> ast.Call:
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(calls) == 1
    return calls[0]


def _keyword(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"missing setup keyword: {name}")


def _extension_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "SyclExtension":
            continue
        value = _keyword(node, "name")
        assert isinstance(value, ast.Constant) and isinstance(value.value, str)
        names.append(value.value)
    return names


def _integer_literal(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _integer_literal(node.operand)
        return -operand if operand is not None else None
    return None


def _call_argument(
    call: ast.Call, position: int, keyword_name: str
) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    if len(call.args) > position:
        return call.args[position]
    return None


def _assigned_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


class _LocalBindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []

    def bind_arguments(self, arguments: ast.arguments) -> None:
        self.names.extend(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        )
        if arguments.vararg is not None:
            self.names.append(arguments.vararg.arg)
        if arguments.kwarg is not None:
            self.names.append(arguments.kwarg.arg)

    def _visit_definition_expressions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(
                default
                for default in node.args.kw_defaults
                if default is not None
            ),
        ):
            self.visit(expression)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.append(node.name)
        self._visit_definition_expressions(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.append(node.name)
        self._visit_definition_expressions(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        for expression in (
            *node.decorator_list,
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        ):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (
            *node.args.defaults,
            *(
                default
                for default in node.args.kw_defaults
                if default is not None
            ),
        ):
            self.visit(default)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.append(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.extend(
            alias.asname or alias.name.partition(".")[0]
            for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.extend(alias.asname or alias.name for alias in node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.append(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.names.append(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.names.append(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.names.append(node.rest)
        self.generic_visit(node)


def _function_local_bindings(function: ast.FunctionDef) -> list[str]:
    collector = _LocalBindingCollector()
    collector.bind_arguments(function.args)
    for statement in function.body:
        collector.visit(statement)
    return collector.names


def _assert_unique_local_binding(
    function: ast.FunctionDef, name: str
) -> None:
    assert _function_local_bindings(function).count(name) == 1


def _insert_statement_after(
    root: ast.AST, target: ast.stmt, statement: ast.stmt
) -> None:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, child in enumerate(value):
                if child is target:
                    value.insert(index + 1, statement)
                    return
    raise AssertionError("target statement is not attached to the AST")


def _returned_assignment_value(function: ast.FunctionDef) -> ast.AST:
    returned_values = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and node.value is not None
        and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        )
    ]
    assert len(returned_values) == 1
    returned_value = returned_values[0]
    assert isinstance(returned_value, ast.Name)

    _assert_unique_local_binding(function, returned_value.id)
    assignments = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assigned_name(node) == returned_value.id
        and node.value is not None
    ]
    assert len(assignments) == 1
    return assignments[0]


def _short_conv_prepare_contracts(
    tree: ast.Module,
) -> dict[str, tuple[ast.FunctionDef, ast.Tuple]]:
    resolver_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_short_conv_op"
    ]
    assert len(resolver_functions) == 1
    resolver_function = resolver_functions[0]
    assert resolver_function.args.args
    resolver_receiver = resolver_function.args.args[0].arg
    _assert_unique_local_binding(resolver_function, resolver_receiver)
    resolver_call = _returned_assignment_value(resolver_function)
    assert (
        isinstance(resolver_call, ast.Call)
        and isinstance(resolver_call.func, ast.Attribute)
        and isinstance(resolver_call.func.value, ast.Name)
        and resolver_call.func.value.id == resolver_receiver
        and resolver_call.func.attr == "_production_ple_op"
    )
    resolved_name = _call_argument(resolver_call, 0, "name")
    assert isinstance(resolved_name, ast.Name) and resolved_name.id == "name"
    _assert_unique_local_binding(resolver_function, resolved_name.id)

    contracts: dict[str, tuple[ast.FunctionDef, ast.Tuple]] = {}
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        for assignment in (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ):
            value = assignment.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and function.args.args
                and value.func.value.id == function.args.args[0].arg
                and value.func.attr == "_resolve_short_conv_op"
            ):
                continue
            op_name = _call_argument(value, 2, "name")
            if not (
                isinstance(op_name, ast.Constant)
                and isinstance(op_name.value, str)
                and op_name.value.startswith("ple_short_conv_")
            ):
                continue
            operation_name = _assigned_name(assignment)
            assert operation_name is not None
            caller_receiver = function.args.args[0].arg
            _assert_unique_local_binding(function, caller_receiver)
            _assert_unique_local_binding(function, operation_name)

            prepared_args: list[ast.Tuple] = []
            for call in ast.walk(function):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_PreparedShortConvCall"
                ):
                    continue
                saved_operation = _call_argument(call, 2, "operation")
                saved_args = _call_argument(call, 3, "args")
                if (
                    isinstance(saved_operation, ast.Name)
                    and saved_operation.id == operation_name
                    and isinstance(saved_args, ast.Tuple)
                ):
                    prepared_args.append(saved_args)
            assert len(prepared_args) == 1
            assert op_name.value not in contracts
            contracts[op_name.value] = (function, prepared_args[0])
    return contracts


def _assert_short_conv_submit_uses_prepared_fields(tree: ast.Module) -> None:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_prepared_short_conv_call"
    ]
    assert len(functions) == 1
    function = functions[0]
    assert function.args.args
    prepared_name = function.args.args[0].arg
    _assert_unique_local_binding(function, prepared_name)
    operation_names = [
        _assigned_name(node)
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == prepared_name
        and node.value.attr == "operation"
    ]
    assert len(operation_names) == 1 and operation_names[0] is not None
    operation_name = operation_names[0]
    _assert_unique_local_binding(function, operation_name)

    submit_calls = []
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        uses_saved_operation = (
            isinstance(call.func, ast.Name) and call.func.id == operation_name
        )
        uses_saved_args = any(
            isinstance(argument, ast.Starred)
            and isinstance(argument.value, ast.Attribute)
            and isinstance(argument.value.value, ast.Name)
            and argument.value.value.id == prepared_name
            and argument.value.attr == "args"
            for argument in call.args
        )
        if uses_saved_operation and uses_saved_args:
            submit_calls.append(call)
    assert len(submit_calls) == 1

def _legacy_null_sentinel(tree: ast.Module) -> int:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_native_state_indices"
    ]
    assert len(functions) == 1
    returns = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Return)
    ]
    assert len(returns) == 1
    mapping = returns[0].value
    assert isinstance(mapping, ast.Call)
    assert isinstance(mapping.func, ast.Attribute) and mapping.func.attr == "where"
    assert len(mapping.args) == 3

    condition, replacement, passthrough = mapping.args
    assert isinstance(condition, ast.Compare) and len(condition.ops) == 1
    assert isinstance(condition.ops[0], ast.Eq) and len(condition.comparators) == 1
    compared_names = {
        node.id
        for node in (condition.left, condition.comparators[0])
        if isinstance(node, ast.Name)
    }
    assert compared_names == {"indices", "NULL_BLOCK_ID"}
    assert isinstance(replacement, ast.Call) and len(replacement.args) >= 2
    assert (
        isinstance(replacement.func, ast.Attribute)
        and replacement.func.attr == "full_like"
    )
    assert isinstance(replacement.args[0], ast.Name)
    assert replacement.args[0].id == "indices"
    sentinel = _integer_literal(replacement.args[1])
    assert sentinel is not None and sentinel < 0
    assert isinstance(passthrough, ast.Name) and passthrough.id == "indices"
    return sentinel


def _is_direct_null_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name) and node.id == "NULL_BLOCK_ID"
    ) or _integer_literal(node) == 0


def _module_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = _assigned_name(node)
        if name is not None and node.value is not None:
            assignments[name] = node.value
    return assignments


def _static_string_values(
    node: ast.AST,
    assignments: dict[str, ast.AST],
    resolving: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        assert node.id in assignments
        assert node.id not in resolving
        return _static_string_values(
            assignments[node.id], assignments, (*resolving, node.id)
        )
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values: list[str] = []
        for element in node.elts:
            value = element.value if isinstance(element, ast.Starred) else element
            values.extend(_static_string_values(value, assignments, resolving))
        return tuple(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (
            *_static_string_values(node.left, assignments, resolving),
            *_static_string_values(node.right, assignments, resolving),
        )
    raise AssertionError(
        f"loader schema declaration is not statically resolvable: {ast.dump(node)}"
    )


def _is_schema_short_name(node: ast.AST, schema_name: str) -> bool:
    if not (
        isinstance(node, ast.Subscript)
        and _integer_literal(node.slice) == 2
        and isinstance(node.value, ast.Call)
    ):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == schema_name
        and call.func.attr == "rpartition"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "::"
        and not call.keywords
    )


def _ple_loader_schema_contract(
    tree: ast.Module,
) -> tuple[str, dict[str, str], set[str]]:
    assignments = _module_assignments(tree)
    standalone_schemas = set(
        _static_string_values(
            assignments["_PLE_STANDALONE_SCHEMAS"], assignments
        )
    )

    op_schema_node = assignments["_PLE_OP_SCHEMAS"]
    if isinstance(op_schema_node, ast.Dict):
        op_schemas: dict[str, str] = {}
        for key, value in zip(
            op_schema_node.keys, op_schema_node.values, strict=True
        ):
            assert key is not None
            key_values = _static_string_values(key, assignments)
            schema_values = _static_string_values(value, assignments)
            assert len(key_values) == len(schema_values) == 1
            op_schemas[key_values[0]] = schema_values[0]
    else:
        assert isinstance(op_schema_node, ast.DictComp)
        assert len(op_schema_node.generators) == 1
        generator = op_schema_node.generators[0]
        assert (
            isinstance(generator.target, ast.Name)
            and not generator.ifs
            and not generator.is_async
        )
        schema_name = generator.target.id
        assert (
            isinstance(op_schema_node.value, ast.Name)
            and op_schema_node.value.id == schema_name
        )
        assert _is_schema_short_name(op_schema_node.key, schema_name)
        schemas = _static_string_values(generator.iter, assignments)
        op_schemas = {schema.rpartition("::")[2]: schema for schema in schemas}

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_qwen38_ple_op"
    ]
    assert len(functions) == 1
    function = functions[0]
    assert function.args.args
    op_name = function.args.args[0].arg
    _assert_unique_local_binding(function, op_name)
    supported_checks = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == op_name
        and len(node.ops) == len(node.comparators) == 1
        and isinstance(node.ops[0], ast.NotIn)
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "_PLE_OP_SCHEMAS"
    ]
    assert len(supported_checks) == 1
    returned_getattr = _returned_assignment_value(function)
    assert (
        isinstance(returned_getattr, ast.Call)
        and isinstance(returned_getattr.func, ast.Name)
        and returned_getattr.func.id == "getattr"
        and len(returned_getattr.args) >= 2
        and isinstance(returned_getattr.args[0], ast.Attribute)
        and isinstance(returned_getattr.args[0].value, ast.Attribute)
        and isinstance(returned_getattr.args[0].value.value, ast.Name)
        and returned_getattr.args[0].value.value.id == "torch"
        and returned_getattr.args[0].value.attr == "ops"
        and isinstance(returned_getattr.args[1], ast.Name)
        and returned_getattr.args[1].id == op_name
    )
    return returned_getattr.args[0].attr, op_schemas, standalone_schemas


def _assert_ple_loader_supports_ops(
    tree: ast.Module, operation_names: set[str]
) -> None:
    namespace, loader_op_schemas, standalone_schemas = (
        _ple_loader_schema_contract(tree)
    )
    used_op_schemas = {
        name: f"{namespace}::{name}" for name in operation_names
    }
    assert used_op_schemas.items() <= loader_op_schemas.items()
    assert set(used_op_schemas.values()) <= standalone_schemas


def test_production_main_dso_is_the_single_ple_owner() -> None:
    source = SETUP_PRODUCTION.read_text()
    registration = (ROOT / "csrc/xpu/torch_extension_ple.cc").read_text()

    assert source.count("csrc/xpu/esimd_kernel_ple.sycl") == 1
    assert source.count("csrc/xpu/torch_extension_ple.cc") == 1
    assert "TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m)" in registration
    assert "TORCH_LIBRARY(custom_esimd_kernels_vllm, m)" not in registration
    assert set(re.findall(r'm\.def\("([^("]+)\(', registration)) == {
        "ple_ngram_ids",
        "ple_embedding_gather",
        "ple_grouped_norm",
        "hc_grouped_norm_v1",
        "hc_gate_mix_v1",
        "hc_combine_v1",
        "hc_combine_norm_v1",
        "ple_score_gate",
        "ple_gated_value",
        "ple_residual_add",
        "ple_short_conv_decode",
        "ple_short_conv_decode_trusted",
        "ple_short_conv_prefill",
        "ple_short_conv_prefill_trusted",
        "ple_short_conv_spec",
        "ple_short_conv_spec_trusted",
    }


def test_ple_only_is_dso_artifact_without_production_package() -> None:
    source = SETUP_PLE_ONLY.read_text()
    call = _setup_call(SETUP_PLE_ONLY)
    packages = _keyword(call, "packages")

    assert "find_packages" not in source
    assert isinstance(packages, ast.List) and not packages.elts
    assert isinstance(_keyword(call, "py_modules"), ast.List)
    assert not any(
        isinstance(keyword.value, ast.Dict) and keyword.arg == "package_dir"
        for keyword in call.keywords
    )
    assert _extension_names(SETUP_PLE_ONLY) == ["ple_ops"]
    assert "$ORIGIN/torch/lib" in source
    assert "$ORIGIN/../../torch/lib" not in source
    assert "esimd_kernel_ple.sycl" in source
    assert "torch_extension_ple.cc" in source


def test_build_only_variants_cannot_install_or_build_canonical_package() -> None:
    for path in (SETUP_GEMV_ONLY, SETUP_SYCL):
        source = path.read_text()
        call = _setup_call(path)
        packages = _keyword(call, "packages")
        assert "find_packages" not in source
        assert isinstance(packages, ast.List) and not packages.elts
        assert isinstance(_keyword(call, "py_modules"), ast.List)
        assert not any(
            isinstance(keyword.value, ast.Dict) and keyword.arg == "package_dir"
            for keyword in call.keywords
        )
        names = _extension_names(path)
        assert names
        assert all("custom_esimd_kernels_vllm." not in name for name in names)
        assert all(name.endswith("_only") for name in names)
        assert "$ORIGIN/torch/lib" in source
        assert "$ORIGIN/../../torch/lib" not in source
        assert 'os.environ.setdefault("TORCH_XPU_ARCH_LIST", "bmg-g31")' in source
        if path == SETUP_SYCL:
            assert 'extension_name="qsa_ops_sycl_only"' in source
            assert "custom_esimd_kernels_vllm.qsa_ops" not in source
        assert "esimd_kernel_ple.sycl" not in source
        assert "torch_extension_ple.cc" not in source


def test_main_prefix_does_not_match_gemm_variant(tmp_path: Path) -> None:
    main = tmp_path / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
    gemm = tmp_path / "custom_esimd_kernels_gemm.cpython-312-x86_64-linux-gnu.so"
    main.touch()
    gemm.touch()

    module = _load_esimd_utils()
    assert module._extension_candidates(tmp_path, "custom_esimd_kernels") == [main]
    assert module._extension_candidates(
        tmp_path, "custom_esimd_kernels_gemm"
    ) == [gemm]


def test_ambiguous_exact_prefix_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "custom_esimd_kernels.cpython-311-x86_64-linux-gnu.so"
    second = tmp_path / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
    first.touch()
    second.touch()

    module = _load_esimd_utils()
    with pytest.raises(RuntimeError, match="ambiguous"):
        module._require_single_extension(tmp_path, "custom_esimd_kernels")


def test_loader_rejects_standalone_ple_and_unknown_prefix() -> None:
    module = _load_esimd_utils()

    with pytest.raises(ImportError, match="standalone-only"):
        module.load_esimd_library("ple_ops")
    with pytest.raises(ValueError, match="unsupported"):
        module.load_esimd_library("not_a_custom_extension")


def test_default_loader_loads_canonical_main_before_gemm(tmp_path: Path, monkeypatch) -> None:
    main = tmp_path / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
    gemm = tmp_path / "custom_esimd_kernels_gemm.cpython-312-x86_64-linux-gnu.so"
    main.touch()
    gemm.touch()

    module = _load_esimd_utils()
    loaded: list[Path] = []
    registered: set[str] = set()

    def load_library(path: str) -> None:
        candidate = Path(path)
        loaded.append(candidate)
        if candidate == main:
            registered.add(module._MAIN_SCHEMA)

    monkeypatch.setattr(module, "_find_esimd_package_dir", lambda: tmp_path)
    monkeypatch.setattr(
        module, "_dispatcher_has_schema", registered.__contains__
    )
    monkeypatch.setattr(module.torch.ops, "load_library", load_library)
    module._ESIMD_LOADED = False

    module._load_esimd_extensions()

    assert loaded == [main, gemm]


def test_ple_standalone_loader_is_disabled_without_explicit_path(monkeypatch) -> None:
    module = _load_esimd_utils()
    monkeypatch.delenv("VLLM_XPU_QWEN38_PLE_STANDALONE_DSO", raising=False)
    assert module.load_qwen38_ple_standalone_library() is False


def test_ple_standalone_loader_rejects_relative_or_missing_path(tmp_path: Path) -> None:
    module = _load_esimd_utils()
    with pytest.raises(ImportError, match="absolute"):
        module.load_qwen38_ple_standalone_library("ple_ops.so")
    with pytest.raises(ImportError, match="does not exist"):
        module.load_qwen38_ple_standalone_library(str(tmp_path / "missing.so"))


def test_ple_standalone_loader_requires_all_short_conv_schemas(
    tmp_path: Path, monkeypatch
) -> None:
    dso = tmp_path / "ple_ops.so"
    dso.touch()
    module = _load_esimd_utils()
    loaded: list[Path] = []
    monkeypatch.setattr(module, "_dispatcher_has_schema", lambda schema: False)
    monkeypatch.setattr(
        module.torch.ops, "load_library", lambda path: loaded.append(Path(path))
    )
    with pytest.raises(RuntimeError, match="missing schemas"):
        module.load_qwen38_ple_standalone_library(dso)
    assert loaded == [dso]


def test_ple_standalone_loader_refuses_existing_dispatcher_owner(
    tmp_path: Path, monkeypatch
) -> None:
    dso = tmp_path / "ple_ops.so"
    dso.touch()
    module = _load_esimd_utils()
    monkeypatch.setattr(
        module, "_dispatcher_has_schema", lambda schema: schema == module._PLE_SCHEMA
    )
    with pytest.raises(RuntimeError, match="already owned"):
        module.load_qwen38_ple_standalone_library(dso)


def test_production_short_conv_dependency_contract_is_coherent() -> None:
    ple_layer_source = (
        ROOT.parents[2]
        / "applications.ai.gpu.llm-scaler-vllm"
        / "vllm/models/qwen3_8_flash_next/xpu/ple_layer.py"
    ).read_text()
    assert "VLLM_XPU_QWEN38_PLE_STANDALONE_DSO" not in ple_layer_source
    assert "get_qwen38_ple_op" in ple_layer_source
    assert "VLLM_XPU_ENABLE_QWEN38_PLE_NATIVE" in ple_layer_source

    null_id_tree = ast.parse(
        (
            ROOT.parents[2]
            / "applications.ai.gpu.llm-scaler-vllm"
            / "vllm/v1/attention/backends/utils.py"
        ).read_text()
    )
    null_id_values = [
        node.value
        for node in null_id_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and target.id == "NULL_BLOCK_ID"
    ]
    assert len(null_id_values) == 1
    assert _integer_literal(null_id_values[0]) == 0

    ple_layer_tree = ast.parse(ple_layer_source)
    null_id_imports = [
        alias
        for node in ple_layer_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.v1.attention.backends.utils"
        for alias in node.names
        if alias.name == "NULL_BLOCK_ID" and alias.asname is None
    ]
    assert len(null_id_imports) == 1

    contracts = _short_conv_prepare_contracts(ple_layer_tree)
    _assert_short_conv_submit_uses_prepared_fields(ple_layer_tree)
    legacy_ops = {
        "ple_short_conv_decode",
        "ple_short_conv_prefill",
        "ple_short_conv_spec",
    }
    trusted_ops = {f"{name}_trusted" for name in legacy_ops}
    requested_ops = set(contracts)
    # The llm-scaler ABI commit precedes its dependent vLLM caller commit.  The
    # caller must select one complete family and obey that family's null ABI.
    assert requested_ops in (legacy_ops, trusted_ops)

    loader_source = ESIMD_UTILS.read_text()
    loader_tree = ast.parse(loader_source)
    _assert_ple_loader_supports_ops(loader_tree, requested_ops)

    caller_override = ast.parse(ple_layer_source)
    resolved_assignments = [
        node
        for node in ast.walk(caller_override)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_resolve_short_conv_op"
        and isinstance(_call_argument(node.value, 2, "name"), ast.Constant)
    ]
    assert len(resolved_assignments) == 3
    overwritten_operation = _assigned_name(resolved_assignments[0])
    assert overwritten_operation is not None
    _insert_statement_after(
        caller_override,
        resolved_assignments[0],
        ast.Assign(
            targets=[ast.Name(id=overwritten_operation, ctx=ast.Store())],
            value=ast.parse("lambda *args: None", mode="eval").body,
        ),
    )
    with pytest.raises(AssertionError):
        _short_conv_prepare_contracts(caller_override)

    caller_nested_definition = ast.parse(ple_layer_source)
    nested_resolved_assignments = [
        node
        for node in ast.walk(caller_nested_definition)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_resolve_short_conv_op"
        and isinstance(_call_argument(node.value, 2, "name"), ast.Constant)
    ]
    assert len(nested_resolved_assignments) == 3
    nested_operation_name = _assigned_name(nested_resolved_assignments[0])
    assert nested_operation_name is not None
    nested_operation = ast.parse(
        "def operation(*args):\n    return None\n"
    ).body[0]
    assert isinstance(nested_operation, ast.FunctionDef)
    nested_operation.name = nested_operation_name
    _insert_statement_after(
        caller_nested_definition,
        nested_resolved_assignments[0],
        nested_operation,
    )
    with pytest.raises(AssertionError):
        _short_conv_prepare_contracts(caller_nested_definition)

    executor_override = ast.parse(ple_layer_source)
    executor_functions = [
        node
        for node in ast.walk(executor_override)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_prepared_short_conv_call"
    ]
    assert len(executor_functions) == 1
    executor_function = executor_functions[0]
    operation_definitions = [
        node
        for node in ast.walk(executor_function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "operation"
    ]
    assert len(operation_definitions) == 1
    overwritten_operation = _assigned_name(operation_definitions[0])
    assert overwritten_operation is not None
    _insert_statement_after(
        executor_override,
        operation_definitions[0],
        ast.Assign(
            targets=[ast.Name(id=overwritten_operation, ctx=ast.Store())],
            value=ast.parse("lambda *args: None", mode="eval").body,
        ),
    )
    with pytest.raises(AssertionError):
        _assert_short_conv_submit_uses_prepared_fields(executor_override)

    executor_nested_definition = ast.parse(ple_layer_source)
    nested_executor_functions = [
        node
        for node in ast.walk(executor_nested_definition)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_prepared_short_conv_call"
    ]
    assert len(nested_executor_functions) == 1
    nested_executor = nested_executor_functions[0]
    nested_operation_definitions = [
        node
        for node in ast.walk(nested_executor)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "operation"
    ]
    assert len(nested_operation_definitions) == 1
    nested_operation_name = _assigned_name(nested_operation_definitions[0])
    assert nested_operation_name is not None
    nested_operation = ast.parse(
        "def operation(*args):\n    return None\n"
    ).body[0]
    assert isinstance(nested_operation, ast.FunctionDef)
    nested_operation.name = nested_operation_name
    _insert_statement_after(
        executor_nested_definition,
        nested_operation_definitions[0],
        nested_operation,
    )
    with pytest.raises(AssertionError):
        _assert_short_conv_submit_uses_prepared_fields(
            executor_nested_definition
        )

    resolver_self_rebinding = ast.parse(ple_layer_source)
    self_rebinding_resolvers = [
        node
        for node in ast.walk(resolver_self_rebinding)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_short_conv_op"
    ]
    assert len(self_rebinding_resolvers) == 1
    self_rebinding_resolver = self_rebinding_resolvers[0]
    self_name = self_rebinding_resolver.args.args[0].arg
    self_rebinding_resolver.body.insert(
        0,
        ast.Assign(
            targets=[ast.Name(id=self_name, ctx=ast.Store())],
            value=ast.Name(id="unrelated_loader", ctx=ast.Load()),
        ),
    )
    with pytest.raises(AssertionError):
        _short_conv_prepare_contracts(resolver_self_rebinding)

    resolver_receiver_mutation = ast.parse(ple_layer_source)
    receiver_resolvers = [
        node
        for node in ast.walk(resolver_receiver_mutation)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_short_conv_op"
    ]
    assert len(receiver_resolvers) == 1
    receiver_call = _returned_assignment_value(receiver_resolvers[0])
    assert isinstance(receiver_call, ast.Call)
    assert isinstance(receiver_call.func, ast.Attribute)
    receiver_call.func.value = ast.Name(
        id="unrelated_loader", ctx=ast.Load()
    )
    with pytest.raises(AssertionError):
        _short_conv_prepare_contracts(resolver_receiver_mutation)

    resolver_mutation = ast.parse(ple_layer_source)
    mutated_resolvers = [
        node
        for node in ast.walk(resolver_mutation)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_short_conv_op"
    ]
    assert len(mutated_resolvers) == 1
    successful_returns = [
        node
        for node in ast.walk(mutated_resolvers[0])
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "operation"
    ]
    assert len(successful_returns) == 1
    successful_returns[0].value = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="_unrelated_dispatch",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )
    with pytest.raises(AssertionError):
        _short_conv_prepare_contracts(resolver_mutation)

    namespace_mutation = ast.parse(loader_source)
    loader_functions = [
        node
        for node in ast.walk(namespace_mutation)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_qwen38_ple_op"
    ]
    assert len(loader_functions) == 1
    loader_function = loader_functions[0]
    returned_getattr = _returned_assignment_value(loader_function)
    assert isinstance(returned_getattr, ast.Call)
    assert isinstance(returned_getattr.args[0], ast.Attribute)
    canonical_namespace = returned_getattr.args[0].attr
    op_name = loader_function.args.args[0].arg
    alias_name = "_requested_operation_name"
    loader_function.body[1:1] = [
        ast.Assign(
            targets=[ast.Name(id=alias_name, ctx=ast.Store())],
            value=ast.Name(id=op_name, ctx=ast.Load()),
        ),
        ast.Assign(
            targets=[ast.Name(id="_unused_operation", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="getattr", ctx=ast.Load()),
                args=[
                    ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="torch", ctx=ast.Load()),
                            attr="ops",
                            ctx=ast.Load(),
                        ),
                        attr=canonical_namespace,
                        ctx=ast.Load(),
                    ),
                    ast.Name(id=op_name, ctx=ast.Load()),
                ],
                keywords=[],
            ),
        ),
    ]
    returned_getattr.args[0].attr = "unrelated_dispatcher"
    returned_getattr.args[1] = ast.Name(id=alias_name, ctx=ast.Load())
    with pytest.raises(AssertionError):
        _assert_ple_loader_supports_ops(namespace_mutation, requested_ops)

    loader_name_rebinding = ast.parse(loader_source)
    name_rebinding_functions = [
        node
        for node in ast.walk(loader_name_rebinding)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_qwen38_ple_op"
    ]
    assert len(name_rebinding_functions) == 1
    name_rebinding_function = name_rebinding_functions[0]
    rebound_name = name_rebinding_function.args.args[0].arg
    name_rebinding_function.body.insert(
        0,
        ast.Assign(
            targets=[ast.Name(id=rebound_name, ctx=ast.Store())],
            value=ast.Constant(value="ple_ngram_ids"),
        ),
    )
    with pytest.raises(AssertionError):
        _assert_ple_loader_supports_ops(
            loader_name_rebinding, requested_ops
        )

    fixed_name_mutation = ast.parse(loader_source)
    fixed_name_functions = [
        node
        for node in ast.walk(fixed_name_mutation)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_qwen38_ple_op"
    ]
    assert len(fixed_name_functions) == 1
    fixed_name_getattr = _returned_assignment_value(fixed_name_functions[0])
    assert isinstance(fixed_name_getattr, ast.Call)
    fixed_name_getattr.args[1] = ast.Constant(value="ple_ngram_ids")
    with pytest.raises(AssertionError):
        _assert_ple_loader_supports_ops(fixed_name_mutation, requested_ops)

    undeclared_trusted = ast.parse(loader_source)
    for node in ast.walk(undeclared_trusted):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.rpartition("::")[2] in trusted_ops
        ):
            node.value = f"{node.value}_not_declared"
    with pytest.raises(AssertionError):
        _assert_ple_loader_supports_ops(undeclared_trusted, trusted_ops)

    sentinel = (
        _legacy_null_sentinel(ple_layer_tree)
        if requested_ops == legacy_ops
        else None
    )
    state_index_positions = {
        "ple_short_conv_decode": 3,
        "ple_short_conv_prefill": 4,
        "ple_short_conv_spec": 4,
    }
    for op_name, (function, prepared_args) in contracts.items():
        base_op_name = op_name.removesuffix("_trusted")
        state_indices = prepared_args.elts[state_index_positions[base_op_name]]
        conversions = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and _assigned_name(node)
            == (state_indices.id if isinstance(state_indices, ast.Name) else None)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_native_state_indices"
        ]
        if requested_ops == trusted_ops:
            inline_conversions = [
                node
                for node in ast.walk(state_indices)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_native_state_indices"
            ]
            assert _is_direct_null_zero(prepared_args.elts[-1])
            assert not conversions and not inline_conversions
        else:
            assert sentinel is not None
            assert _integer_literal(prepared_args.elts[-1]) == sentinel
            assert isinstance(state_indices, ast.Name)
            assert len(conversions) == 1


def test_projection_schemas_declare_caller_owned_outputs() -> None:
    gemv_source = (ROOT / "csrc/xpu/torch_extension.cc").read_text()
    gemm_source = (ROOT / "csrc/xpu/torch_extension_gemm.cc").read_text()

    assert (
        'esimd_gemv_fp16(Tensor input, Tensor weight, '
        'Tensor(a!) output) -> Tensor(a!)' in gemv_source
    )
    assert (
        'esimd_hc_down_fp16_out(Tensor input, Tensor weight, '
        'Tensor(a!) output) -> ()' in gemv_source
    )
    assert 'esimd_gemv_int4(Tensor input, Tensor weight, Tensor weight_scale, ' in gemv_source
    assert 'Tensor(a!) output) -> Tensor(a!)' in gemv_source
    assert 'esimd_gemm_int4_pgrp(Tensor input, Tensor weight, Tensor weight_scale, ' in gemm_source
    assert 'Tensor(a!) output) -> Tensor(a!)' in gemm_source
