"""Public tensor-API compiler contracts on the installed native XPU backend.

The finite inventory is deliberate: adding an API requires a fixture and an
explicit mutation/alias/availability decision. It is not a claim about every
shape, device, autograd mode or native symbol.
"""

import ast
import importlib
import json
from pathlib import Path

import pytest
import torch

import omni_xpu_kernel as omni
from omni_xpu_kernel import cute, fp8, gguf, int8, layout, linear, norm, rotary, sdp, svdq


APIS = {
    "norm": ("rms_norm", "layer_norm", "group_norm_bmg", "group_norm_seedvr_bmg",
             "rms_norm_segmented_modulation", "rms_norm_gate_residual", "fused_add_rms_norm",
             "fused_rms_norm_linear", "fused_adaln", "fused_rms_adaln"),
    "int8": ("quantize_int8_tensorwise", "quantize_int8_rowwise", "dequantize_int8_simple",
             "dequantize_int8_simple_dtype", "mm_int8", "int8_linear", "int8_linear_prequantized",
             "int8_linear_shared_input", "fused_silu_mul", "fused_silu_mul_quantize_rowwise",
             "fused_swiglu_quantize_rowwise", "fused_gelu_tanh_quantize_rowwise",
             "rotate_convrot", "quantize_int8_convrot_weight", "dequantize_int8_convrot_weight"),
    "fp8": ("quantize_per_tensor", "dequantize_per_tensor", "stochastic_rounding"),
    "gguf": ("dequantize_q4_0", "dequantize_q4_0_comfyui", "dequantize_q4_1", "dequantize_q8_0",
             "dequantize_q4_k", "dequantize_q6_k", "dequantize_batch"),
    "svdq": ("dequantize_w4", "dequantize_u4", "unpack_int4", "quantize_act_int4", "quantize_act_uint4",
             "onednn_int4_gemm", "onednn_int4_gemm_preconverted", "onednn_int4_gemm_add_to_output",
             "fused_convert_add", "fused_smooth_convert", "fused_smooth_mul_convert", "prepare_onednn_weights"),
    "rotary": ("rotary_emb", "apply_kitchen_rope1", "apply_kitchen_rope1_", "apply_kitchen_rope",
               "apply_kitchen_rope_", "apply_kitchen_rope_split_half1", "apply_kitchen_rope_split_half1_",
               "apply_kitchen_rope_split_half", "apply_kitchen_rope_split_half_", "rms_kitchen_rope1",
               "rms_kitchen_rope1_", "rms_kitchen_rope", "rms_kitchen_rope_", "rms_kitchen_rope_split_half1",
               "rms_kitchen_rope_split_half1_", "rms_kitchen_rope_split_half", "rms_kitchen_rope_split_half_",
               "apply_ltx_split_rope_direct"),
    "cute": ("sdp", "sdp_bhld_d120", "sdp_bhld_d128", "sdp_wan22_cross", "sdp_minimax_h3_vae_d64", "sol_attn"),
    "sdp": ("sdp",),
    "linear": ("onednn_w8a16_fp8", "try_onednn_w8a16_fp8"),
    "layout": ("cat_pad_bmg",),
}
API_NAMES = tuple(f"{module}.{name}" for module, names in APIS.items() for name in names)


def _rand(shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="xpu", dtype=dtype) * 0.25


def _packed(rows=32, width=128):
    return torch.randint(0, 256, (rows, width // 2), device="xpu", dtype=torch.uint8)


def _gguf_bytes(format, blocks=3):
    # Finite nonzero scales and arbitrary packed payload, including QK scale
    # codes. Building byte representations on CPU is fixture construction;
    # every native/compiler execution remains on XPU in the admitted container.
    size = {"q4_0": 18, "q4_1": 20, "q8_0": 34, "q4_k": 144, "q6_k": 210}[format]
    data = torch.arange(blocks * size, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(blocks, size)
    scale = torch.tensor([0.125], dtype=torch.float16).view(torch.uint8)
    if format == "q6_k":
        data[:, -2:] = scale
    else:
        data[:, :2] = scale
        if format in ("q4_1", "q4_k"):
            data[:, 2:4] = scale
    return data.flatten().to("xpu")


def case(api, *, dtype=torch.bfloat16, rows=3):
    module, name = api.split(".")
    function = getattr(importlib.import_module("omni_xpu_kernel." + module), name)
    if module == "gguf":
        if name == "dequantize_batch":
            formats = ["q4_0", "q4_1", "q8_0", "q4_k", "q6_k"]
            return function, ([_gguf_bytes(f, rows) for f in formats], formats, dtype), {}
        format = name.removeprefix("dequantize_").removesuffix("_comfyui")
        return function, (_gguf_bytes(format, rows), dtype), {}
    if module == "fp8":
        x = _rand((rows, 65), dtype)
        scale = torch.tensor(0.125, device="xpu")
        if name == "dequantize_per_tensor":
            return function, (x.to(torch.float8_e4m3fn), scale, dtype), {}
        if name == "stochastic_rounding":
            rng = torch.randint(0, 256, x.shape, device="xpu", dtype=torch.uint8)
            return function, (x, rng), {}
        return function, (x, scale), {}
    if module == "linear":
        x = _rand((rows, 2048), torch.float16)
        weight = _rand((1024, 2048), torch.float32).to(torch.float8_e4m3fn)
        return function, (x, weight, torch.full((1024,), 0.125, device="xpu")), {}
    if module == "svdq":
        x = _rand((rows, 128), dtype)
        packed = _packed(32)
        scales = torch.full((2, 32), 0.125, device="xpu", dtype=dtype)
        if name.startswith("dequantize"):
            return function, (packed, scales, dtype), {}
        if name == "unpack_int4":
            return function, (_packed(rows), True), {}
        if name.startswith("quantize_act"):
            return function, (x.abs() if name.endswith("uint4") else x, 64), {}
        if name == "prepare_onednn_weights":
            return function, (packed, scales), {}
        if name == "onednn_int4_gemm":
            return function, (x, packed, scales), {}
        if name.startswith("onednn_int4_gemm"):
            prepared, scale_half = svdq.prepare_onednn_weights(packed, scales)
            args = (x.to(torch.float16), prepared, scale_half)
            if name.endswith("add_to_output"):
                args += (_rand((rows, 32), torch.bfloat16),)
            return function, args, {}
        if name == "fused_convert_add":
            return function, (_rand((rows, 32)), _rand((rows + 2, 48), torch.float16), _rand((rows, 32))), {}
        factor = torch.full((128,), 1.25, device="xpu", dtype=torch.bfloat16 if name == "fused_smooth_convert" else torch.float16)
        return function, (x.to(torch.bfloat16), factor), {}
    if module == "norm":
        x = _rand((rows, 128), dtype)
        weight = _rand((128,), dtype)
        if name == "rms_norm":return function, (weight, x), {}
        if name == "layer_norm":return function, (x, weight, _rand((128,), dtype)), {}
        if name == "fused_add_rms_norm":return function, (x, _rand(x.shape, dtype), weight), {}
        if name == "fused_rms_norm_linear":return function, (x, weight, _rand((32, 128), dtype)), {}
        if name in ("fused_adaln", "fused_rms_adaln"):
            return function, (x, _rand(x.shape, dtype), _rand(x.shape, dtype)), {}
        if name == "rms_norm_gate_residual":
            x = _rand((64, 3840))
            return function, (_rand((3840,)), x, _rand((3840,)), _rand(x.shape)), {}
        if name == "rms_norm_segmented_modulation":
            x = _rand((17, 5376))
            packed = _rand((3, 6 * 5376))
            shift, scale, *_ = packed.chunk(6, dim=-1)
            return function, (_rand((5376,)), x, scale, shift, [(0, 3, 0), (3, 8, 1), (8, 17, 2)]), {}
        if name == "group_norm_bmg":x = _rand((1, 512, 128, 128))
        else:x = _rand((1, 512, 2, 128, 128), torch.float16).transpose(1, 2).reshape(2, 512, 128, 128)
        return function, (x, 32, _rand((512,), x.dtype), _rand((512,), x.dtype)), {}
    if module == "layout":
        x = _rand((1, 2, 512, 128, 128), torch.float16).transpose(1, 2)
        return function, (_rand((1, 512, 2, 128, 128), torch.float16), x), {}
    if module == "rotary":
        if name == "rotary_emb":
            return function, (_rand((rows, 128), dtype), _rand((rows, 64), torch.float32), _rand((rows, 64), torch.float32), rows, 1), {}
        if name == "apply_ltx_split_rope_direct":
            x = _rand((1, rows, 3 * 128), dtype)
            cos = _rand((1, rows, 3, 64), dtype).transpose(1, 2)
            return function, (x, cos, torch.randn_like(cos)), {}
        q = _rand((1, 3, rows, 128), dtype)
        k = _rand(q.shape, dtype)
        freqs = _rand((1, 1, rows, 64, 2, 2), torch.float32)
        single = "1" in name
        args = (q, freqs) if single else (q, k, freqs)
        if name.startswith("rms_"):
            args += (_rand((128,), dtype),)
        return function, args, {}
    if module in ("cute", "sdp"):
        if name == "sdp_wan22_cross":
            return function, (_rand((1, 75600, 40, 128), torch.float16), _rand((1, 512, 40, 128), torch.float16), _rand((1, 512, 40, 128), torch.float16)), {}
        if name == "sdp_minimax_h3_vae_d64":
            q = _rand((1, 37, 32, 64), torch.float16).transpose(1, 2)
            k = torch.randn_like(q)
            v = _rand((1, 37, 32, 192), torch.float16)[..., 128:].transpose(1, 2)
            return function, (q, k, v), {}
        if name.startswith("sdp_bhld"):
            dim = int(name[-3:])
            q = _rand((1, 32, 4, dim), dtype).transpose(1, 2)
            return function, (q, torch.randn_like(q), torch.randn_like(q)), {}
        q = _rand((1, 64, 4, 128), dtype)
        return function, (q, torch.randn_like(q), torch.randn_like(q)), {}
    if module == "int8":
        x = _rand((rows, 128), dtype)
        y = _rand(x.shape, dtype)
        weight = torch.randint(-127, 128, (32, 128), device="xpu", dtype=torch.int8)
        scale = torch.full((32, 1), 0.125, device="xpu")
        if name in ("quantize_int8_tensorwise", "quantize_int8_rowwise", "fused_gelu_tanh_quantize_rowwise"):
            return function, (x,), {}
        if name in ("fused_silu_mul", "fused_silu_mul_quantize_rowwise"):
            return function, (x, y), {}
        if name == "fused_swiglu_quantize_rowwise":return function, (torch.cat((x, y), -1),), {}
        if name in ("dequantize_int8_simple", "dequantize_int8_simple_dtype"):
            return function, (weight, scale), {}
        if name == "mm_int8":return function, (weight[:rows], weight.T), {}
        if name == "int8_linear":return function, (x, weight, scale), {"bias": _rand((32,), dtype), "out_dtype": dtype}
        if name == "int8_linear_prequantized":
            q, s = int8.quantize_int8_rowwise(x)
            return function, (q, s, weight, scale), {"out_dtype": dtype}
        if name == "int8_linear_shared_input":return function, (x, weight, scale, weight.clone(), scale.clone()), {"out_dtype": dtype}
        if name in ("rotate_convrot", "quantize_int8_convrot_weight"):return function, (x, 64), {}
        if name == "dequantize_int8_convrot_weight":return function, (weight, scale, 64), {}
    raise AssertionError(f"Missing valid fixture for {api}")


def clone_tree(value, memo=None):
    # Retain arbitrary views, holes, offsets, shared storage and repeated input
    # identity. A plain clone silently removes packed-QKV aliasing and strides.
    if memo is None:memo = ({}, {})
    objects, storage = memo
    if isinstance(value, torch.Tensor):
        if id(value) in objects:return objects[id(value)]
        key = (value.untyped_storage()._cdata, value.dtype, value.device)
        if key not in storage:
            backing = torch.empty(0, device=value.device, dtype=value.dtype).set_(
                value.untyped_storage(), 0, (value.untyped_storage().nbytes() // value.element_size(),), (1,))
            storage[key] = backing.clone()
        result = storage[key].as_strided(value.shape, value.stride(), value.storage_offset())
        objects[id(value)] = result
        return result
    if isinstance(value, tuple):return tuple(clone_tree(x, memo) for x in value)
    if isinstance(value, list):return [clone_tree(x, memo) for x in value]
    if isinstance(value, dict):return {k:clone_tree(v, memo) for k,v in value.items()}
    return value


def same_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.stride() == expected.stride()
        assert actual.device == expected.device
        # Includes NaN payloads, signed zero and float8 encodings.
        assert torch.equal(actual.contiguous().reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:same_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a,b in zip(actual, expected):same_tree(a,b)
    else:assert actual == expected


def _tensors(tree):
    if isinstance(tree, torch.Tensor):return [tree]
    if isinstance(tree, dict):return sum((_tensors(x) for x in tree.values()), [])
    if isinstance(tree, (tuple, list)):return sum((_tensors(x) for x in tree), [])
    return []


def alias_signature(outputs, inputs):
    tensors = _tensors((outputs, inputs))
    return [(a is b, torch._C._is_alias_of(a,b)) for i,a in enumerate(tensors) for b in tensors[i+1:]]


def test_public_tensor_inventory_has_no_unclassified_api():
    root = Path(__file__).resolve().parents[1] / "omni_xpu_kernel"
    actual = set()
    for module in APIS:
        for f in ast.parse((root/module/"__init__.py").read_text()).body:
            if not isinstance(f, ast.FunctionDef) or f.name.startswith("_"):continue
            if f.name.startswith("supports_") or f.name.endswith("_supported") or f.name == "is_available" or "_cache_" in f.name:continue
            actual.add(module + "." + f.name)
    assert actual == set(API_NAMES)
    assert len(actual) == 75


@pytest.mark.parametrize("api", API_NAMES)
def test_public_api_compile_output_state_and_alias_contract(api, record_property):
    assert torch.xpu.is_available(), "installed XPU environment required"
    torch._dynamo.reset()
    torch.manual_seed(20260907)
    function, args, kwargs = case(api)
    record_property("api", api)
    if api == "norm.rms_norm_gate_residual" and omni.core_aot_target() != "ptl-h":
        record_property("coverage", "native_target_unavailable_error_preserved")
        with pytest.raises(RuntimeError, match="PTL-H"):
            function(*args, **kwargs)
        with pytest.raises(RuntimeError, match="PTL-H"):
            torch.compile(function, fullgraph=True)(*args, **kwargs)
        return
    fullgraph = api != "linear.try_onednn_w8a16_fp8"
    record_property("coverage", "fullgraph" if fullgraph else "runtime_optional_eager_boundary")
    expected_args, expected_kwargs = clone_tree((args, kwargs))
    actual_args, actual_kwargs = clone_tree((args, kwargs))
    with torch.inference_mode():
        expected = function(*expected_args, **expected_kwargs)
        actual = torch.compile(function, backend="inductor", fullgraph=fullgraph)(*actual_args, **actual_kwargs)
        same_tree(actual, expected)
        same_tree((actual_args, actual_kwargs), (expected_args, expected_kwargs))
        assert alias_signature(actual, (actual_args, actual_kwargs)) == alias_signature(expected, (expected_args, expected_kwargs))


@pytest.mark.parametrize("api", [
    "fp8.quantize_per_tensor", "fp8.dequantize_per_tensor", "fp8.stochastic_rounding",
    "gguf.dequantize_q4_0_comfyui", "svdq.unpack_int4", "svdq.quantize_act_int4",
    "norm.fused_add_rms_norm", "norm.fused_adaln", "rotary.rotary_emb",
    "int8.quantize_int8_tensorwise", "int8.mm_int8",
])
def test_dynamic_rows_reuse_one_full_graph(api, record_property):
    from torch._dynamo.testing import CompileCounterWithBackend
    torch._dynamo.reset()
    function, _, _ = case(api)
    class RecordedCounter(CompileCounterWithBackend):
        def __init__(self):
            super().__init__("inductor")
            self.attempts = []
        def __call__(self, graph, inputs):
            attempt = {"graph": graph.code}
            self.attempts.append(attempt)
            try:
                return super().__call__(graph, inputs)
            except BaseException as error:
                attempt["exception"] = repr(error)
                raise
    counter = RecordedCounter()
    counts = []
    compiled = torch.compile(function, backend=counter, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for rows in (3, 5, 7):
            _, args, kwargs = case(api, rows=rows)
            ea, ek = clone_tree((args, kwargs))
            ca, ck = clone_tree((args, kwargs))
            expected = function(*ea, **ek)
            actual = compiled(*ca, **ck)
            counts.append(counter.frame_count)
            same_tree(actual, expected)
            same_tree((ca, ck), (ea, ek))
    record_property("compile_attempts_by_rows", json.dumps(counts))
    record_property("startup_retries", json.dumps([a.get("exception") for a in counter.attempts]))
    # Backend analysis can retry during the first call. The dynamic contract
    # is one cached full graph and no further compilation as row counts change.
    assert counts == [counts[0]] * 3, (counts, counter.attempts)
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    assert len(_debug_get_cache_entry_list(function)) == 1


def operator_cases():
    """Every registered boundary has a real native fixture, including privates."""
    result = {}
    root = Path(__file__).resolve().parents[1] / "omni_xpu_kernel"
    for module in APIS:
        for function in ast.parse((root/module/"__init__.py").read_text()).body:
            if not isinstance(function, ast.FunctionDef):continue
            for decorator in function.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "compile_op" and not function.name.startswith("_")):
                    result[ast.literal_eval(decorator.args[0])] = module + "." + function.name
    result.update({
        "quantize_int8_tensorwise": "int8.quantize_int8_tensorwise",
        "quantize_int8_tensorwise_scaled": "int8.quantize_int8_tensorwise",
        "rms_norm_segmented_modulation": "norm.rms_norm_segmented_modulation",
        "cute_sol_attn": "cute.sol_attn",
        "rotary_rms_rope1": "rotary.rms_kitchen_rope1",
        "rotary_rms_rope1_": "rotary.rms_kitchen_rope1_",
        "rotary_rms_rope": "rotary.rms_kitchen_rope",
        "rotary_rms_rope_": "rotary.rms_kitchen_rope_",
    })
    return result


OPERATOR_CASES = operator_cases()


def test_every_dispatcher_boundary_has_a_native_fixture():
    from omni_xpu_kernel._compile_ops import _OPERATORS
    assert set(OPERATOR_CASES) == set(_OPERATORS)


@pytest.mark.parametrize("name", sorted(OPERATOR_CASES))
def test_dispatcher_schema_fake_and_aot_contract(name, record_property, monkeypatch):
    api = OPERATOR_CASES[name]
    _, args, kwargs = case(api)
    record_property("operator", name)
    if name == "rms_norm_gate_residual" and omni.core_aot_target() != "ptl-h":
        pytest.skip("native PTL-H operation unavailable on admitted B70; error covered separately")
    if name == "quantize_int8_tensorwise_scaled":
        args += (torch.tensor(0.125, device="xpu"),)
    elif name == "rms_norm_segmented_modulation":
        args = args[:-1] + tuple(list(values) for values in zip(*args[-1]))
    elif name.startswith("rotary_rms_rope"):
        if "rope1" not in name:args += (args[-1],)
        args += (1e-6, False, 0)
    if name == "cute_sdp_minimax_h3_vae_d64":
        # opcheck's AOT helper clones each tensor, which densifies the required
        # interleaved V view before even calling the eager operator. This op
        # is non-mutating, so run the same AOT checker directly on its real views.
        from torch.testing._internal.optests.aot_autograd import aot_autograd_check
        operator = getattr(torch.ops.omni_xpu, name).default
        torch.library.opcheck(operator, args, kwargs,
                              test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))
        aot_autograd_check(operator, args, kwargs, dynamic=True, check_gradients=False)
    elif name in ("fp8_dequantize_per_tensor", "linear_onednn_w8a16_fp8"):
        # SchemaCheckMode uses allclose to detect input mutation. Torch XPU
        # cannot multiply float8 for that check; compare the raw bytes instead.
        # Only the checker changes, never operator execution or input dtype.
        original_allclose = torch.allclose
        def float8_byte_equal(a, b, *pos, **kw):
            if a.dtype == b.dtype == torch.float8_e4m3fn:
                return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            return original_allclose(a, b, *pos, **kw)
        with monkeypatch.context() as context:
            context.setattr(torch, "allclose", float8_byte_equal)
            torch.library.opcheck(getattr(torch.ops.omni_xpu, name).default, args, kwargs,
                                  test_utils=("test_schema",))
        torch.library.opcheck(getattr(torch.ops.omni_xpu, name).default, args, kwargs,
                              test_utils=("test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"))
    else:
        torch.library.opcheck(getattr(torch.ops.omni_xpu, name).default, args, kwargs)


def check_variant(function, args, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    torch._dynamo.reset()
    ea, ek = clone_tree((args, kwargs))
    ca, ck = clone_tree((args, kwargs))
    with torch.inference_mode():
        expected = function(*ea, **ek)
        actual = torch.compile(function, fullgraph=True)(*ca, **ck)
        same_tree(actual, expected)
        same_tree((ca, ck), (ea, ek))
        assert alias_signature(actual, (ca, ck)) == alias_signature(expected, (ea, ek))
    return actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [0, 177])
@pytest.mark.parametrize("scale_kind", ["absmax", "xpu_alias", "cpu", "converted"])
def test_tensorwise_seed_scale_alias_and_strides(dtype, seed, scale_kind):
    x = _rand((128, 7), dtype).T[1:6]
    scale = {"absmax": None, "xpu_alias": torch.tensor(0.015, device="xpu"),
             "cpu": torch.tensor(0.015), "converted": torch.tensor(0.015, device="xpu", dtype=dtype)}[scale_kind]
    rng = torch.xpu.get_rng_state().clone()
    output = check_variant(int8.quantize_int8_tensorwise, (x, scale, seed))
    assert torch.equal(torch.xpu.get_rng_state(), rng)
    if seed:
        repeat = int8.quantize_int8_tensorwise(x, scale, seed)
        same_tree(output, repeat)
        other = int8.quantize_int8_tensorwise(x, scale, seed + 1)
        assert not torch.equal(output[0], other[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [0, 177])
def test_rowwise_seed_and_strides(dtype, seed):
    x = _rand((128, 7), dtype).T[1:6]
    rng = torch.xpu.get_rng_state().clone()
    check_variant(int8.quantize_int8_rowwise, (x, seed))
    assert torch.equal(torch.xpu.get_rng_state(), rng)
    torch.library.opcheck(torch.ops.omni_xpu.quantize_int8_rowwise.default, (x, seed))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["dequantize_w4", "dequantize_u4", "quantize_act_int4", "quantize_act_uint4"])
def test_svdq_quantization_dtype_contract(name, dtype):
    function, args, kwargs = case("svdq." + name, dtype=dtype)
    check_variant(function, args, kwargs)
    torch.library.opcheck(getattr(torch.ops.omni_xpu, "svdq_" + name).default, args, kwargs)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("rms", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_rotary_packed_qkv_alias_and_partial_rotation(split, rms, inplace):
    packed = _rand((1, 5, 3, 3, 128))
    q, k, _ = packed.unbind(2)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    name = ("rms" if rms else "apply") + "_kitchen_rope" + ("_split_half" if split else "") + ("_" if inplace else "")
    width = 32 if rms and split else 64
    args = (q, k, _rand((1, 1, 5, width, 2, 2), torch.float32))
    kwargs = {}
    if rms:
        args += (_rand((128,)),)
        if split:kwargs["rot_dim"] = 64
    check_variant(getattr(rotary, name), args, kwargs)


@pytest.mark.parametrize("name", APIS["cute"])
def test_cute_first_call_can_be_compiled_in_fresh_process(name):
    import os
    import subprocess
    import sys
    program = """
import sys, torch
sys.path.insert(0, sys.argv[1])
from test_torch_compile_api import case, same_tree
function, args, kwargs = case('cute.' + sys.argv[2])
with torch.inference_mode():
    actual = torch.compile(function, fullgraph=True)(*args, **kwargs)
    expected = function(*args, **kwargs)
    same_tree(actual, expected)
torch.xpu.synchronize()
"""
    process = subprocess.run([sys.executable, "-c", program, str(Path(__file__).parent), name],
                             text=True, capture_output=True, env=dict(os.environ), timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("api", ["fp8.quantize_per_tensor", "fp8.dequantize_per_tensor", "fp8.stochastic_rounding"])
def test_fp8_dtypes_and_offset_inputs(api, dtype):
    function, args, kwargs = case(api, dtype=dtype, rows=7)
    args = (args[0][1:6],) + args[1:]
    if api.endswith("stochastic_rounding"):
        args = (args[0], args[1][1:6])
    check_variant(function, args, kwargs)


def test_fp8_rounding_rng_is_an_explicit_runtime_tensor():
    x = torch.linspace(-2, 2, 1024, device="xpu", dtype=torch.float32)
    rng0 = torch.zeros(x.shape, device="xpu", dtype=torch.uint8)
    rng1 = torch.full(x.shape, 255, device="xpu", dtype=torch.uint8)
    compiled = torch.compile(fp8.stochastic_rounding, fullgraph=True)
    state = torch.xpu.get_rng_state().clone()
    with torch.inference_mode():
        a, b = compiled(x, rng0), compiled(x, rng1)
        same_tree(a, fp8.stochastic_rounding(x, rng0))
        same_tree(b, fp8.stochastic_rounding(x, rng1))
        assert not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert torch.equal(state, torch.xpu.get_rng_state())


@pytest.mark.parametrize("mode", ["float_bias", "bool_bias", "masked", "topk", "sinks", "tail_off", "block_len", "coarse_gate"])
def test_sol_attention_runtime_controls(mode):
    q = _rand((1, 257, 1, 128))
    args = (q, torch.randn_like(q), torch.randn_like(q))
    options = {
        "float_bias": {"key_bias": torch.linspace(-2, 2, 257, device="xpu"), "scale": 0.0},
        "bool_bias": {"key_bias": torch.arange(257, device="xpu") % 3 != 0},
        "masked": {"key_bias": torch.zeros(257, device="xpu", dtype=torch.bool), "tail": False},
        "topk": {"topk_ratio": 0.4, "tail": False},
        "sinks": {"sink_blocks": (0, 1), "sink_q": [0, 1], "tau": 100.0},
        "tail_off": {"tail": False, "tau": 100.0},
        "block_len": {"block_len": torch.tensor([64, 64, 64, 64, 1], device="xpu", dtype=torch.int32)},
        "coarse_gate": {"coarse_gate": torch.full_like(q, 0.5)},
    }
    check_variant(cute.sol_attn, args, options[mode])


@pytest.mark.parametrize("compile_optional", [False, True])
def test_fp8_optional_fault_injected_failure_and_negative_cache(compile_optional, monkeypatch):
    # Primitive availability changes with oneDNN/device versions. Inject the
    # documented native failure protocol to test the optional Python boundary;
    # this is not a claim that this shape fails on the admitted B70.
    from test_fp8_negative_cache import _UnsupportedNative
    native = _UnsupportedNative()
    monkeypatch.setattr(linear, "_get_native", lambda: native)
    linear.fp8_cache_clear()
    _, args, _ = case("linear.onednn_w8a16_fp8")
    x = args[0]
    def caller(x, weight, scale):
        result = linear.try_onednn_w8a16_fp8(x, weight, scale)
        return x[:, :1] + 1 if result is None else result
    function = torch.compile(caller, fullgraph=False) if compile_optional else caller
    with torch.inference_mode():
        first = function(*args)
        stats = linear.fp8_failure_cache_stats()
        assert stats["failures"] == 1 and stats["size"] == 1
        second = function(*args)
        after = linear.fp8_failure_cache_stats()
        assert after["failures"] == 1 and after["negative_hits"] == stats["negative_hits"] + 1
        same_tree(first, second)
        same_tree(first, x[:, :1] + 1)
    linear.fp8_cache_clear()


@pytest.mark.parametrize("api", ["norm.fused_add_rms_norm", "rotary.rms_kitchen_rope_", "svdq.fused_convert_add"])
def test_mutable_inputs_can_share_storage(api):
    function, args, kwargs = case(api)
    if api == "norm.fused_add_rms_norm":
        args = (args[0], args[0], args[2])
    elif api.startswith("rotary"):
        args = (args[0], args[0], *args[2:])
    else:
        args = (args[0], args[1], args[0])
    check_variant(function, args, kwargs)


def test_ordered_fp8_preserves_unused_calls_and_native_cache_counts():
    _, args, _ = case("linear.onednn_w8a16_fp8")
    def program(x, weight, scales):
        linear.onednn_w8a16_fp8(x, weight, scales)
        linear.onednn_w8a16_fp8(x, weight, scales)
        return linear.onednn_w8a16_fp8(x, weight, scales)
    with torch.inference_mode():
        linear.fp8_cache_clear()
        expected = program(*args)
        eager_stats = linear.fp8_cache_stats()
        linear.fp8_cache_clear()
        compiled = torch.compile(program, fullgraph=True)
        actual = compiled(*args)
        assert linear.fp8_cache_stats() == eager_stats
        assert eager_stats["misses"] == 1 and eager_stats["hits"] == 2
        same_tree(actual, expected)
    linear.fp8_cache_clear()


def test_ordered_sdp_preserves_unused_and_duplicate_native_calls(monkeypatch):
    from types import SimpleNamespace
    native = sdp._get_native()
    calls = []
    def observed(q, k, v):
        calls.append(v.data_ptr())
        return native.sdp(q, k, v)
    monkeypatch.setattr(sdp, "_get_native", lambda: SimpleNamespace(sdp=observed))
    _, args, _ = case("sdp.sdp")
    q, k, v = args
    other = v * 2
    def program(q, k, v, other):
        sdp.sdp(q, k, v)
        sdp.sdp(q, k, other)
        return sdp.sdp(q, k, v)
    with torch.inference_mode():
        expected = program(q, k, v, other)
        eager_calls = list(calls)
        calls.clear()
        actual = torch.compile(program, fullgraph=True)(q, k, v, other)
        assert calls == eager_calls == [v.data_ptr(), other.data_ptr(), v.data_ptr()]
        same_tree(actual, expected)
