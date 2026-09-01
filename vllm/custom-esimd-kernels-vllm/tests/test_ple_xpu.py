"""Real-XPU smoke/correctness tests for the standalone PLE DSO.

Run with one physical card exposed, for example::

    ZE_AFFINITY_MASK=0 \
      PLE_FIXTURE_ROOT=/tmp/qwen38-ple-fixtures-v2-20260831 \
      PLE_DSO=/tmp/libple_standalone.so pytest -q tests/test_ple_xpu.py

The test loads only the standalone PLE registration DSO.  It does not start a
vLLM server and does not use a TP process group.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

FIXTURE_ROOT = Path(
    os.environ.get(
        "PLE_FIXTURE_ROOT",
        "/llm/models/test/qwen38_ngram_ko/kernel_optimizer/workflow/pilot_cases/ple/fixtures",
    )
)
DSO = os.environ.get("PLE_DSO", "/tmp/libple_standalone.so")
GEMV_DSO = os.environ.get("PLE_GEMV_DSO", "")
GEMM_DSO = os.environ.get("PLE_GEMM_DSO", "")
REQUIRED_FIXTURE_CASES = frozenset(
    {
        "ple_embedding_local_assembly",
        "ple_gated_value_norm",
        "ple_grouped_norm_key_query",
        "ple_ngram_ids_decode",
        "ple_ngram_ids_prefill_eos",
        "ple_projection_int4_canonical_gemm",
        "ple_projection_int4_canonical_gemv",
        "ple_projection_int4_key_value",
        "ple_residual_add",
        "ple_score_gate",
        "ple_short_conv_decode",
        "ple_short_conv_decode_float16_ds_padded_offset",
        "ple_short_conv_decode_float16_sd_padded_offset",
        "ple_short_conv_decode_float32_ds_padded_offset",
        "ple_short_conv_decode_float32_sd_padded_offset",
        "ple_short_conv_mixed_permutation",
        "ple_short_conv_mixed_permutation_float16_ds",
        "ple_short_conv_mixed_permutation_float16_sd",
        "ple_short_conv_mixed_permutation_float32_ds",
        "ple_short_conv_prefill",
        "ple_short_conv_prefill_float16_ds_padded_offset",
        "ple_short_conv_prefill_float16_sd_padded_offset",
        "ple_short_conv_prefill_float32_ds_padded_offset",
        "ple_short_conv_prefill_float32_sd_padded_offset",
        "ple_short_conv_spec",
        "ple_short_conv_spec_float16_ds_padded_offset",
        "ple_short_conv_spec_float16_sd_padded_offset",
        "ple_short_conv_spec_float32_ds_padded_offset",
        "ple_short_conv_spec_float32_sd_padded_offset",
        "ple_staged_decode_fp16",
        "ple_staged_full_decode_fp16",
        "ple_staged_prefill_fp16",
        "ple_staged_spec_fp16",
    }
)
DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _require_fixture_root() -> None:
    missing = sorted(
        case
        for case in REQUIRED_FIXTURE_CASES
        if not (FIXTURE_ROOT / case / "manifest.json").is_file()
    )
    if missing:
        preview = ", ".join(missing[:3])
        if len(missing) > 3:
            preview += f", ... ({len(missing)} total)"
        pytest.skip(
            f"PLE fixture root is incomplete: {FIXTURE_ROOT}; missing {preview}. "
            "Set PLE_FIXTURE_ROOT to the generated qwen38.ple.fixture.v2 root."
        )


def _require_xpu() -> torch.device:
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if not Path(DSO).exists():
        pytest.skip(f"PLE_DSO does not exist: {DSO}")
    return torch.device("xpu:0")


def _load_case(name: str, device: torch.device):
    if sys.byteorder != "little":
        pytest.fail("PLE fixtures require a little-endian host")
    manifest_path = FIXTURE_ROOT / name / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(
            f"PLE fixture is missing: {manifest_path}. Set PLE_FIXTURE_ROOT to "
            "the generated qwen38.ple.fixture.v2 root."
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["schema"] == "qwen38.ple.fixture.v2"
    declared_manifest_hash = manifest.pop("manifest_sha256")
    unsigned_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(unsigned_manifest).hexdigest() == declared_manifest_hash

    entries_by_kind = (
        ("input", manifest["input_bins"]),
        ("output", manifest["output_bins"]),
    )
    payload_root = hashlib.sha256()
    for kind, entries in entries_by_kind:
        for entry in entries:
            payload = (manifest_path.parent / entry["path"]).read_bytes()
            assert entry["endianness"] == "little"
            assert len(payload) == entry["bytes"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            payload_root.update(kind.encode("ascii"))
            payload_root.update(b"\\0")
            payload_root.update(entry["name"].encode("utf-8"))
            payload_root.update(b"\\0")
            payload_root.update(entry["sha256"].encode("ascii"))
            payload_root.update(b"\\n")
    assert payload_root.hexdigest() == manifest["payload_root_sha256"]

    def load(entry: dict) -> torch.Tensor:
        dtype = DTYPES[entry["dtype"]]
        shape = tuple(int(dimension) for dimension in entry["shape"])
        stride = tuple(int(value) for value in entry["stride"])
        storage_offset = int(entry["storage_offset"])
        storage_numel = int(entry["storage_numel"])
        assert len(shape) == len(stride)
        assert storage_offset >= 0 and storage_numel >= 0
        max_index = storage_offset
        if all(dimension > 0 for dimension in shape):
            for dimension, step in zip(shape, stride):
                assert step >= 0
                max_index += (dimension - 1) * step
            assert max_index < storage_numel
        host_storage = torch.from_file(
            str(manifest_path.parent / entry["path"]),
            shared=False,
            size=storage_numel,
            dtype=dtype,
        )
        logical_payload = host_storage.as_strided(
            shape, stride, storage_offset
        ).contiguous().numpy().tobytes(order="C")
        assert hashlib.sha256(logical_payload).hexdigest() == entry["logical_sha256"]
        device_storage = torch.empty((storage_numel,), dtype=dtype, device=device)
        device_storage.copy_(host_storage)
        return device_storage.as_strided(shape, stride, storage_offset)

    inputs = {entry["name"]: load(entry) for entry in manifest["input_bins"]}
    outputs = {entry["name"]: load(entry) for entry in manifest["output_bins"]}
    return inputs, outputs, manifest["inputs"]


def _storage_flat(tensor: torch.Tensor) -> torch.Tensor:
    element_size = tensor.element_size()
    storage_nbytes = tensor.untyped_storage().nbytes()
    assert element_size > 0 and storage_nbytes % element_size == 0
    return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        tensor.untyped_storage(),
        0,
        (storage_nbytes // element_size,),
        (1,),
    )


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.cpu()
    expected = expected.cpu()
    if expected.dtype in (torch.int32, torch.int64, torch.uint8, torch.bool):
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)


def _load_ops_wrapper():
    wrapper_path = Path(__file__).resolve().parents[1] / (
        "python/custom_esimd_kernels_vllm/ops.py"
    )
    spec = importlib.util.spec_from_file_location("ple_ops_wrapper", wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def device() -> torch.device:
    device = _require_xpu()
    torch.ops.load_library(DSO)
    for env_name, path in (("PLE_GEMV_DSO", GEMV_DSO), ("PLE_GEMM_DSO", GEMM_DSO)):
        if path:
            if not Path(path).exists():
                pytest.skip(f"{env_name} does not exist: {path}")
            torch.ops.load_library(path)
    return device


def test_frozen_fixture_manifest_is_complete() -> None:
    _require_fixture_root()


def test_arithmetic_primitives_match_frozen_golden(device: torch.device) -> None:
    ids, golden, meta = _load_case("ple_ngram_ids_decode", device)
    ngram_out = torch.empty_like(golden["ngram_ids"])
    torch.ops.custom_esimd_kernels_vllm.ple_ngram_ids(
        ids["input_ids"], ids["query_start_loc"], ids["ngram_context"],
        ids["layer_multipliers"], ids["ngram_heads_vocab_sizes"],
        ids["ngram_heads_offsets"], ngram_out,
        meta["eos_token_id"], meta["heads_per_ngram"],
    )
    torch.xpu.synchronize()
    _assert_equal(ngram_out, golden["ngram_ids"])

    ids, golden, _ = _load_case("ple_embedding_local_assembly", device)
    local_out = torch.empty_like(golden["local_partial"])
    torch.ops.custom_esimd_kernels_vllm.ple_embedding_gather(
        ids["ngram_ids"], ids["local_weight"], ids["local_vocab_start"],
        ids["local_num_rows"], local_out,
    )
    torch.xpu.synchronize()
    _assert_equal(local_out, golden["local_partial"])

    ids, golden, meta = _load_case("ple_grouped_norm_key_query", device)
    norm_out = torch.empty_like(golden["output"])
    torch.ops.custom_esimd_kernels_vllm.ple_grouped_norm(
        ids["input"], ids["weight"], norm_out, meta["eps"], meta["group_size"]
    )
    torch.xpu.synchronize()
    _assert_equal(norm_out, golden["output"])

    ids, golden, meta = _load_case("ple_score_gate", device)
    gate_out = torch.empty_like(golden["gate"])
    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        ids["key_norm"], ids["query_norm"], gate_out, meta["hidden_size"]
    )
    torch.xpu.synchronize()
    _assert_equal(gate_out, golden["gate"])

    ids, golden, meta = _load_case("ple_gated_value_norm", device)
    gated_out = torch.empty_like(golden["gated_value"])
    torch.ops.custom_esimd_kernels_vllm.ple_gated_value(
        ids["gate"], ids["value"], gated_out, 2
    )
    torch.xpu.synchronize()
    _assert_equal(gated_out, golden["gated_value"])

    ids, golden, _ = _load_case("ple_residual_add", device)
    residual_out = torch.empty_like(golden["output"])
    torch.ops.custom_esimd_kernels_vllm.ple_residual_add(
        ids["gated_value_flat"], ids["conv_output"], residual_out
    )
    torch.xpu.synchronize()
    _assert_equal(residual_out, golden["output"])


def _score_gate_reference(
    key: torch.Tensor,
    query: torch.Tensor,
    hidden_size: int = 2560,
) -> torch.Tensor:
    key_groups = key.float().reshape(-1, hidden_size)
    query_groups = query.float().reshape(-1, hidden_size)
    score = (key_groups * query_groups).sum(-1) / (hidden_size**0.5)
    signed_root = torch.where(
        score == 0.0,
        torch.zeros_like(score),
        torch.sign(score) * torch.sqrt(torch.clamp(score.abs(), min=1.0e-6)),
    )
    return torch.sigmoid(signed_root).to(key.dtype)


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [((1, 10240), (1, 4)), ((1, 4, 2560), (1, 4, 1))],
)
def test_ple_score_gate_target_fast_path_matches_reference(
    device: torch.device,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(46)
    key = torch.randn(input_shape, dtype=torch.float16, device=device) * 0.25
    query = torch.randn(input_shape, dtype=torch.float16, device=device) * 0.25
    output = torch.empty(output_shape, dtype=torch.float16, device=device)
    output_pointer = output.data_ptr()
    expected = _score_gate_reference(key, query).reshape(output_shape)

    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        key, query, output, 2560
    )
    torch.xpu.synchronize()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


def test_ple_score_gate_target_fast_path_preserves_exact_zero(
    device: torch.device,
) -> None:
    key = torch.zeros((1, 10240), dtype=torch.float16, device=device)
    query = torch.zeros_like(key)
    key[0, 2560:2562] = 1.0
    query[0, 2560:2562] = torch.tensor(
        [1.0, -1.0], dtype=torch.float16, device=device
    )
    output = torch.empty((1, 4), dtype=torch.float16, device=device)

    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        key, query, output, 2560
    )
    torch.xpu.synchronize()

    assert torch.equal(output, torch.full_like(output, 0.5))


def test_ple_score_gate_target_fast_path_preserves_signed_clamp(
    device: torch.device,
) -> None:
    key = torch.zeros((1, 10240), dtype=torch.float16, device=device)
    query = torch.zeros_like(key)
    key[0, 0] = 1.0e-3
    query[0, 0] = 1.0e-3
    key[0, 2560] = 1.0e-3
    query[0, 2560] = -1.0e-3
    output = torch.empty((1, 4), dtype=torch.float16, device=device)
    expected = _score_gate_reference(key, query).reshape_as(output)

    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        key, query, output, 2560
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(output, expected, rtol=0.0, atol=5.0e-4)
    assert output[0, 0].cpu() > 0.5
    assert output[0, 1].cpu() < 0.5
    assert torch.equal(output[0, 2:].cpu(), torch.full((2,), 0.5))


def test_ple_score_gate_target_fast_path_matches_generic_nan_semantics(
    device: torch.device,
) -> None:
    key = torch.zeros((1, 10240), dtype=torch.float16, device=device)
    query = torch.ones_like(key)
    key[0, 0] = float("nan")
    fast_output = torch.empty((1, 4), dtype=torch.float16, device=device)

    odd_backing = torch.empty(
        (10241,), dtype=torch.float16, device=device
    )
    odd_key = odd_backing[1:].reshape_as(key)
    assert odd_key.is_contiguous()
    assert odd_key.data_ptr() % 4 == 2
    odd_key.copy_(key)
    generic_output = torch.empty_like(fast_output)

    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        key, query, fast_output, 2560
    )
    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        odd_key, query, generic_output, 2560
    )
    torch.xpu.synchronize()

    assert torch.isfinite(fast_output).all().cpu()
    assert torch.equal(fast_output, generic_output)
    assert fast_output[0, 0].cpu() > 0.5
    assert torch.equal(
        fast_output[0, 1:].cpu(), torch.full((3,), 0.5)
    )


def test_ple_score_gate_generic_rows_match_reference(
    device: torch.device,
) -> None:
    torch.manual_seed(47)
    key = torch.randn((2, 10240), dtype=torch.float16, device=device) * 0.25
    query = torch.randn((2, 10240), dtype=torch.float16, device=device) * 0.25
    output = torch.empty((2, 4), dtype=torch.float16, device=device)
    expected = _score_gate_reference(key, query).reshape_as(output)

    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        key, query, output, 2560
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("odd_tensor", ["key", "query", "output"])
def test_ple_score_gate_target_shape_preserves_odd_offset_views(
    device: torch.device,
    odd_tensor: str,
) -> None:
    def odd_offset(shape: tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for dimension in shape:
            numel *= dimension
        backing = torch.empty(
            (numel + 1,), dtype=torch.float16, device=device
        )
        view = backing[1:].reshape(shape)
        assert view.is_contiguous()
        assert view.data_ptr() % 4 == 2
        return view

    torch.manual_seed(48)
    key = (
        odd_offset((1, 10240))
        if odd_tensor == "key"
        else torch.empty((1, 10240), dtype=torch.float16, device=device)
    )
    query = (
        odd_offset((1, 10240))
        if odd_tensor == "query"
        else torch.empty((1, 10240), dtype=torch.float16, device=device)
    )
    key.normal_(mean=0.0, std=0.25)
    query.normal_(mean=0.0, std=0.25)
    output = (
        odd_offset((1, 4))
        if odd_tensor == "output"
        else torch.empty((1, 4), dtype=torch.float16, device=device)
    )
    expected = _score_gate_reference(key, query).reshape_as(output)

    torch.ops.custom_esimd_kernels_vllm.ple_score_gate(
        key, query, output, 2560
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [1, 2])
def test_ple_grouped_norm_target_fast_path_preserves_generic_rows(
    device: torch.device,
    rows: int,
) -> None:
    torch.manual_seed(40 + rows)
    input_tensor = (
        torch.randn((rows, 10240), dtype=torch.float16, device=device) * 0.25
    )
    weight = (
        torch.randn((10240,), dtype=torch.float16, device=device) * 0.05
    )
    output = torch.empty_like(input_tensor)
    output_pointer = output.data_ptr()
    eps = 1.0e-5

    grouped = input_tensor.float().reshape(rows, 4, 2560)
    variance = grouped.square().mean(-1, keepdim=True)
    expected = (
        grouped * torch.rsqrt(variance + eps)
    ).reshape_as(input_tensor) * (1.0 + weight.float())
    expected = expected.to(input_tensor.dtype)

    torch.ops.custom_esimd_kernels_vllm.ple_grouped_norm(
        input_tensor,
        weight,
        output,
        eps,
        2560,
    )
    torch.xpu.synchronize()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("odd_tensor", ["input", "weight", "output"])
def test_ple_grouped_norm_target_shape_preserves_odd_offset_views(
    device: torch.device,
    odd_tensor: str,
) -> None:
    def odd_offset(shape: tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for dimension in shape:
            numel *= dimension
        backing = torch.empty(
            (numel + 1,),
            dtype=torch.float16,
            device=device,
        )
        view = backing[1:].reshape(shape)
        assert view.is_contiguous()
        assert view.data_ptr() % 4 == 2
        return view

    torch.manual_seed(45)
    input_tensor = (
        odd_offset((1, 10240))
        if odd_tensor == "input"
        else torch.empty((1, 10240), dtype=torch.float16, device=device)
    )
    weight = (
        odd_offset((10240,))
        if odd_tensor == "weight"
        else torch.empty((10240,), dtype=torch.float16, device=device)
    )
    output = (
        odd_offset((1, 10240))
        if odd_tensor == "output"
        else torch.empty((1, 10240), dtype=torch.float16, device=device)
    )
    input_tensor.normal_(mean=0.0, std=0.25)
    weight.normal_(mean=0.0, std=0.05)
    output_pointer = output.data_ptr()
    eps = 1.0e-5

    grouped = input_tensor.float().reshape(1, 4, 2560)
    variance = grouped.square().mean(-1, keepdim=True)
    expected = (
        grouped * torch.rsqrt(variance + eps)
    ).reshape_as(input_tensor) * (1.0 + weight.float())
    expected = expected.to(input_tensor.dtype)

    torch.ops.custom_esimd_kernels_vllm.ple_grouped_norm(
        input_tensor,
        weight,
        output,
        eps,
        2560,
    )
    torch.xpu.synchronize()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_hc_grouped_norm_v1_matches_target_reference(
    device: torch.device,
) -> None:
    torch.manual_seed(41)
    input_tensor = (
        torch.randn((1, 10240), dtype=torch.float16, device=device) * 0.25
    )
    weight = (
        torch.randn((10240,), dtype=torch.float16, device=device) * 0.05
    )
    output = torch.empty_like(input_tensor)
    output_pointer = output.data_ptr()
    eps = 1e-6

    grouped = input_tensor.float().reshape(1, 4, 2560)
    variance = grouped.square().mean(-1, keepdim=True)
    expected = (
        grouped * torch.rsqrt(variance + eps)
    ).reshape_as(input_tensor) * (1.0 + weight.float())
    expected = expected.to(input_tensor.dtype)

    torch.ops.custom_esimd_kernels_vllm.hc_grouped_norm_v1(
        input_tensor, weight, output, eps
    )
    torch.xpu.synchronize()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_hc_grouped_norm_v1_rejects_wrong_shape_and_alias(
    device: torch.device,
) -> None:
    input_tensor = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    )
    weight = torch.randn((10240,), dtype=torch.float16, device=device)

    with pytest.raises(RuntimeError, match="expects input/output"):
        torch.ops.custom_esimd_kernels_vllm.hc_grouped_norm_v1(
            input_tensor.repeat(2, 1),
            weight,
            torch.empty((2, 10240), dtype=torch.float16, device=device),
            1e-6,
        )

    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_grouped_norm_v1(
            input_tensor, weight, input_tensor, 1e-6
        )


def test_hc_gate_mix_v1_matches_target_reference(
    device: torch.device,
) -> None:
    torch.manual_seed(43)
    input_tensor = (
        torch.randn((1, 10240), dtype=torch.float16, device=device) * 0.5
    )
    gate = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    )
    output = torch.empty(
        (1, 2560), dtype=torch.float16, device=device
    )
    output_pointer = output.data_ptr()

    expected = (
        torch.sigmoid(gate.float()) * input_tensor.float()
    ).reshape(1, 4, 2560).mean(1).to(input_tensor.dtype)

    torch.ops.custom_esimd_kernels_vllm.hc_gate_mix_v1(
        input_tensor, gate, output
    )
    torch.xpu.synchronize()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_hc_gate_mix_v1_rejects_wrong_shape_and_alias(
    device: torch.device,
) -> None:
    input_tensor = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    )
    gate = torch.randn((1, 10240), dtype=torch.float16, device=device)
    output = torch.empty((1, 2560), dtype=torch.float16, device=device)

    with pytest.raises(RuntimeError, match="expects input/gate"):
        torch.ops.custom_esimd_kernels_vllm.hc_gate_mix_v1(
            input_tensor.repeat(2, 1), gate.repeat(2, 1), output
        )

    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_gate_mix_v1(
            input_tensor, gate, input_tensor[:, :2560]
        )

    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_gate_mix_v1(
            input_tensor, gate, gate[:, :2560]
        )


def test_hc_combine_v1_matches_target_reference(
    device: torch.device,
) -> None:
    torch.manual_seed(47)
    hidden_states = (
        torch.randn((1, 10240), dtype=torch.float16, device=device)
        * 0.5
    )
    block_output = (
        torch.randn((1, 2560), dtype=torch.float16, device=device)
        * 0.5
    )
    injection = torch.randn(
        (1, 4), dtype=torch.float16, device=device
    )
    output = torch.empty_like(hidden_states)
    output_pointer = output.data_ptr()

    injection_weight = 2.0 * torch.sigmoid(injection.float() / 4)
    expected = (
        hidden_states.float().reshape(1, 4, 2560)
        + block_output.float().unsqueeze(1)
        * injection_weight.unsqueeze(-1)
    ).reshape_as(hidden_states).to(hidden_states.dtype)

    torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(
        hidden_states, block_output, injection, output
    )
    torch.xpu.synchronize()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_hc_combine_v1_accepts_offset_injection_view(
    device: torch.device,
) -> None:
    hidden_states = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    )
    block_output = torch.randn(
        (1, 2560), dtype=torch.float16, device=device
    )
    merged = torch.randn(
        (1, 336), dtype=torch.float16, device=device
    )
    injection = merged.split((320, 4, 12), dim=-1)[1]
    assert injection.storage_offset() == 320
    assert injection.is_contiguous()
    output = torch.empty_like(hidden_states)
    expected = (
        hidden_states.float().reshape(1, 4, 2560)
        + block_output.float().unsqueeze(1)
        * (2.0 * torch.sigmoid(injection.float() / 4)).unsqueeze(-1)
    ).reshape_as(hidden_states).to(hidden_states.dtype)

    torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(
        hidden_states, block_output, injection, output
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_hc_combine_v1_rejects_wrong_shape_and_alias(
    device: torch.device,
) -> None:
    hidden_states = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    )
    block_output = torch.randn(
        (1, 2560), dtype=torch.float16, device=device
    )
    injection = torch.randn((1, 4), dtype=torch.float16, device=device)
    output = torch.empty_like(hidden_states)

    with pytest.raises(RuntimeError, match="expects hidden/output"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(
            hidden_states.repeat(2, 1),
            block_output.repeat(2, 1),
            injection.repeat(2, 1),
            output.repeat(2, 1),
        )

    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(
            hidden_states, block_output, injection, hidden_states
        )

    block_backing = torch.empty(
        (1, 10240), dtype=torch.float16, device=device
    )
    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(
            hidden_states, block_backing[:, :2560], injection, block_backing
        )

    injection_backing = torch.empty(
        (1, 10240), dtype=torch.float16, device=device
    )
    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(
            hidden_states,
            block_output,
            injection_backing[:, :4],
            injection_backing,
        )

    odd_hidden = torch.empty(
        (1, 10241), dtype=torch.float16, device=device
    )[:, 1:]
    odd_block = torch.empty(
        (1, 2561), dtype=torch.float16, device=device
    )[:, 1:]
    odd_injection = torch.empty(
        (1, 5), dtype=torch.float16, device=device
    )[:, 1:]
    odd_output = torch.empty(
        (1, 10241), dtype=torch.float16, device=device
    )[:, 1:]
    assert all(
        tensor.is_contiguous() and tensor.storage_offset() == 1
        for tensor in (odd_hidden, odd_block, odd_injection, odd_output)
    )
    for tensors in (
        (odd_hidden, block_output, injection, output),
        (hidden_states, odd_block, injection, output),
        (hidden_states, block_output, odd_injection, output),
        (hidden_states, block_output, injection, odd_output),
    ):
        with pytest.raises(RuntimeError, match="4-byte aligned"):
            torch.ops.custom_esimd_kernels_vllm.hc_combine_v1(*tensors)


def _hc_combine_norm_reference(
    hidden_states: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = (
        hidden_states.float().reshape(1, 4, 2560)
        + block_output.float().unsqueeze(1)
        * (2.0 * torch.sigmoid(injection.float() / 4)).unsqueeze(-1)
    ).reshape_as(hidden_states).to(hidden_states.dtype)
    grouped = combined.float().reshape(1, 4, 2560)
    inverse = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    normed = (
        grouped * inverse * (1.0 + weight.float().reshape(1, 4, 2560))
    ).reshape_as(hidden_states).to(hidden_states.dtype)
    return combined, normed


def test_hc_combine_norm_v1_matches_sequential_reference(
    device: torch.device,
) -> None:
    torch.manual_seed(89)
    eps = 1.0e-6
    hidden_states = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    ) * 0.25
    block_output = torch.randn(
        (1, 2560), dtype=torch.float16, device=device
    ) * 0.25
    merged = torch.randn((1, 336), dtype=torch.float16, device=device)
    injection = merged.split((320, 4, 12), dim=-1)[1]
    weight = torch.randn((10240,), dtype=torch.float16, device=device) * 0.05
    combined_output = torch.empty_like(hidden_states)
    normed_output = torch.empty_like(hidden_states)
    combined_pointer = combined_output.data_ptr()
    normed_pointer = normed_output.data_ptr()
    expected_combined, expected_normed = _hc_combine_norm_reference(
        hidden_states, block_output, injection, weight, eps
    )

    torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
        hidden_states,
        block_output,
        injection,
        weight,
        combined_output,
        normed_output,
        eps,
    )
    torch.xpu.synchronize()

    assert injection.storage_offset() == 320
    assert combined_output.data_ptr() == combined_pointer
    assert normed_output.data_ptr() == normed_pointer
    torch.testing.assert_close(
        combined_output, expected_combined, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        normed_output, expected_normed, rtol=2e-2, atol=2e-2
    )


def test_hc_combine_norm_v1_rejects_contract_violations(
    device: torch.device,
) -> None:
    hidden_states = torch.randn(
        (1, 10240), dtype=torch.float16, device=device
    )
    block_output = torch.randn(
        (1, 2560), dtype=torch.float16, device=device
    )
    injection = torch.randn((1, 4), dtype=torch.float16, device=device)
    weight = torch.randn((10240,), dtype=torch.float16, device=device)
    combined_output = torch.empty_like(hidden_states)
    normed_output = torch.empty_like(hidden_states)

    with pytest.raises(RuntimeError, match="expects hidden/combined/normed"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
            hidden_states.repeat(2, 1),
            block_output,
            injection,
            weight,
            combined_output,
            normed_output,
            1.0e-6,
        )
    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
            hidden_states,
            block_output,
            injection,
            weight,
            hidden_states,
            normed_output,
            1.0e-6,
        )
    shared_outputs = torch.empty(
        (1, 20480), dtype=torch.float16, device=device
    )
    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
            hidden_states,
            block_output,
            injection,
            weight,
            shared_outputs[:, :10240],
            shared_outputs[:, 10240:],
            1.0e-6,
        )
    weight_backing = torch.empty(
        (1, 10240), dtype=torch.float16, device=device
    )
    with pytest.raises(RuntimeError, match="must not share storage"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
            hidden_states,
            block_output,
            injection,
            weight_backing.reshape(-1),
            combined_output,
            weight_backing,
            1.0e-6,
        )
    odd_weight = torch.empty(
        (10241,), dtype=torch.float16, device=device
    )[1:]
    with pytest.raises(RuntimeError, match="4-byte aligned"):
        torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
            hidden_states,
            block_output,
            injection,
            odd_weight,
            combined_output,
            normed_output,
            1.0e-6,
        )
    for invalid_eps in (0.0, 1.0e-300):
        with pytest.raises(RuntimeError, match="finite and positive"):
            torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
                hidden_states,
                block_output,
                injection,
                weight,
                combined_output,
                normed_output,
                invalid_eps,
            )


def test_canonical_int4_projection_dispatch_and_golden(
    device: torch.device,
) -> None:
    if not GEMV_DSO or not GEMM_DSO:
        pytest.skip(
            "BMG-compatible GEMV and GEMM DSOs are required for canonical projection"
        )
    if not torch._C._jit_get_schemas_for_operator(
        "custom_esimd_kernels_vllm::esimd_gemv_int4"
    ) or not torch._C._jit_get_schemas_for_operator(
        "custom_esimd_kernels_vllm::esimd_gemm_int4_pgrp"
    ):
        pytest.skip("canonical projection schemas are unavailable")
    wrapper = _load_ops_wrapper()
    calls: list[str] = []
    real_gemv = wrapper.esimd_gemv_int4
    real_gemm = wrapper.esimd_gemm_int4_pgrp

    def gemv_spy(*args, **kwargs):
        calls.append("gemv")
        return real_gemv(*args, **kwargs)

    def gemm_spy(*args, **kwargs):
        calls.append("gemm")
        return real_gemm(*args, **kwargs)

    wrapper.esimd_gemv_int4 = gemv_spy
    wrapper.esimd_gemm_int4_pgrp = gemm_spy

    for case_name, expected_dispatch in (
        ("ple_projection_int4_canonical_gemv", "gemv"),
        ("ple_projection_int4_canonical_gemm", "gemm"),
    ):
        inputs, golden, meta = _load_case(case_name, device)
        before = {name: tensor.clone() for name, tensor in inputs.items()}
        for projection_name in ("key", "value"):
            output = torch.empty_like(golden[f"{projection_name}_output"])
            wrapper.ple_projection_int4(
                inputs["input"],
                inputs[f"{projection_name}_weight_esimd"],
                inputs[f"{projection_name}_scale_esimd"],
                output,
            )
            torch.xpu.synchronize()
            _assert_equal(output, golden[f"{projection_name}_output"])
        assert meta["dispatch"] == expected_dispatch
        assert calls[-2:] == [expected_dispatch, expected_dispatch]
        for name, tensor in inputs.items():
            assert torch.equal(tensor, before[name]), name


def _int4_reference(
    input_tensor: torch.Tensor,
    weight_esimd: torch.Tensor,
    scale_esimd: torch.Tensor,
) -> torch.Tensor:
    packed = weight_esimd.cpu().to(torch.int16)
    low = (packed & 0xF).float() - 8.0
    high = ((packed >> 4) & 0xF).float() - 8.0
    unpacked = torch.empty(
        (weight_esimd.size(0), weight_esimd.size(1) * 2), dtype=torch.float32
    )
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    dequantized = unpacked * scale_esimd.cpu().float().repeat_interleave(
        128, dim=1
    )
    return (input_tensor.cpu().float() @ dequantized.transpose(0, 1)).to(
        torch.float16
    )


def test_staged_full_k0_to_k10_matches_all_frozen_intermediates(
    device: torch.device,
) -> None:
    if not GEMV_DSO:
        pytest.skip("GEMV DSO is required for the FP16 staged projections")
    inputs, golden, meta = _load_case("ple_staged_full_decode_fp16", device)
    wrapper = _load_ops_wrapper()
    before = {
        name: tensor.clone()
        for name, tensor in inputs.items()
        if name != "conv_state"
    }
    assembled = torch.empty_like(golden["assembled_embedding"])
    result = wrapper.ple_staged_full(
        inputs["input_ids"],
        inputs["query_start_loc"],
        inputs["ngram_context"],
        inputs["layer_multipliers"],
        inputs["ngram_heads_vocab_sizes"],
        inputs["ngram_heads_offsets"],
        inputs["local_weight"],
        inputs["local_vocab_start"],
        inputs["local_num_rows"],
        inputs["rank_local_partials"],
        assembled,
        inputs["hidden_states"],
        inputs["key_weight"],
        None,
        inputs["value_weight"],
        None,
        inputs["norm_key_weight"],
        inputs["norm_query_weight"],
        inputs["norm_conv_weight"],
        inputs["conv_state"],
        inputs["conv_weights"],
        inputs["state_indices"],
        inputs["has_initial_state"],
        meta["mode"],
        meta["eps"],
        meta["group_size"],
        meta["dilation"],
        meta["eos_token_id"],
        meta["heads_per_ngram"],
        None,
        None,
        0,
        True,
        -1,
        meta["projection_kind"],
    )
    torch.xpu.synchronize()
    assert result["assembled_embedding"] is assembled
    for name, expected in golden.items():
        try:
            _assert_equal(result[name], expected)
        except AssertionError as exc:
            raise AssertionError(f"first divergent staged output: {name}") from exc
    for name, tensor in before.items():
        assert torch.equal(inputs[name], tensor), name


def test_int4_gemm_m_boundaries_are_bounded_and_correct(
    device: torch.device,
) -> None:
    if not GEMV_DSO or not GEMM_DSO:
        pytest.skip("BMG-compatible GEMV and GEMM DSOs are required")
    if not torch._C._jit_get_schemas_for_operator(
        "custom_esimd_kernels_vllm::esimd_gemv_int4"
    ) or not torch._C._jit_get_schemas_for_operator(
        "custom_esimd_kernels_vllm::esimd_gemm_int4_pgrp"
    ):
        pytest.skip("INT4 projection schemas are unavailable")

    rows, n, k = 129, 16, 128
    input_cpu = torch.arange(rows * k, dtype=torch.float32).reshape(rows, k)
    input_cpu = ((input_cpu.remainder(37) - 18.0) / 7.0).to(torch.float16)
    weight_cpu = torch.arange(n * (k // 2), dtype=torch.int64)
    weight_cpu = weight_cpu.remainder(256).to(torch.uint8).reshape(n, k // 2)
    scale_cpu = torch.linspace(0.25, 1.25, n, dtype=torch.float32).reshape(n, 1)
    scale_cpu = scale_cpu.to(torch.float16)
    input_tensor = input_cpu.to(device)
    weight_esimd = weight_cpu.to(device)
    scale_esimd = scale_cpu.to(device)

    native_input = input_tensor[:64].contiguous()
    native_output = torch.full((64, n), float("nan"), dtype=torch.float16, device=device)
    torch.ops.custom_esimd_kernels_vllm.esimd_gemm_int4_pgrp(
        native_input, weight_esimd, scale_esimd, native_output
    )
    torch.xpu.synchronize()
    _assert_equal(
        native_output, _int4_reference(native_input, weight_esimd, scale_esimd)
    )

    for m in (65, 128, 129):
        bad_output = torch.full((m, n), 17.0, dtype=torch.float16, device=device)
        with pytest.raises(RuntimeError, match=r"M must be in \[2, 64\]"):
            torch.ops.custom_esimd_kernels_vllm.esimd_gemm_int4_pgrp(
                input_tensor[:m].contiguous(),
                weight_esimd,
                scale_esimd,
                bad_output,
            )
        assert torch.equal(bad_output, torch.full_like(bad_output, 17.0))

    wrapper = _load_ops_wrapper()
    for m in (64, 65, 128, 129):
        wrapper_output = torch.empty((m, n), dtype=torch.float16, device=device)
        wrapper.ple_projection_int4(
            input_tensor[:m].contiguous(),
            weight_esimd,
            scale_esimd,
            wrapper_output,
        )
        torch.xpu.synchronize()
        _assert_equal(
            wrapper_output,
            _int4_reference(input_tensor[:m], weight_esimd, scale_esimd),
        )


@pytest.mark.parametrize(
    "op_name", ("ple_short_conv_decode", "ple_short_conv_decode_trusted")
)
def test_short_conv_decode_matches_output_and_final_state(
    device: torch.device, op_name: str
) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_decode", device)
    state_before = inputs["conv_state"].clone()
    input_before = inputs["input"].clone()
    output = torch.empty_like(golden["output"])
    operation = getattr(torch.ops.custom_esimd_kernels_vllm, op_name)
    operation(
        inputs["input"], inputs["conv_state"], inputs["conv_weights"],
        inputs["state_indices"], inputs["has_initial_state"], output,
        meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])
    assert torch.equal(inputs["input"], input_before)
    assert inputs["conv_state"].shape == state_before.shape


@pytest.mark.parametrize(
    "op_name", ("ple_short_conv_prefill", "ple_short_conv_prefill_trusted")
)
def test_short_conv_prefill_matches_output_and_final_state(
    device: torch.device, op_name: str
) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_prefill", device)
    output = torch.empty_like(golden["output"])
    operation = getattr(torch.ops.custom_esimd_kernels_vllm, op_name)
    operation(
        inputs["input"], inputs["query_start_loc"], inputs["conv_state"],
        inputs["conv_weights"], inputs["state_indices"],
        inputs["has_initial_state"], output, meta["dilation"],
        meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])


@pytest.mark.parametrize(
    "op_name", ("ple_short_conv_spec", "ple_short_conv_spec_trusted")
)
def test_short_conv_spec_matches_output_and_final_state(
    device: torch.device, op_name: str
) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_spec", device)
    output = torch.empty_like(golden["output"])
    operation = getattr(torch.ops.custom_esimd_kernels_vllm, op_name)
    operation(
        inputs["input"], inputs["query_start_loc"], inputs["conv_state"],
        inputs["conv_weights"], inputs["state_indices"],
        inputs["num_accepted_tokens"], output, meta["num_spec_tokens"],
        meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])


@pytest.mark.parametrize(
    "case_name",
    (
        "ple_short_conv_mixed_permutation",
        "ple_short_conv_mixed_permutation_float16_sd",
        "ple_short_conv_mixed_permutation_float16_ds",
        "ple_short_conv_mixed_permutation_float32_ds",
    ),
)
def test_short_conv_mixed_three_way_matches_output_and_final_state(
    device: torch.device, case_name: str
) -> None:
    inputs, golden, meta = _load_case(case_name, device)
    wrapper = _load_ops_wrapper()
    before = {
        name: tensor.clone()
        for name, tensor in inputs.items()
        if name not in ("conv_state",)
    }
    output = torch.full_like(golden["output"], float("nan"))
    result = wrapper.ple_short_conv_mixed_three_way(
        inputs["input"],
        inputs["conv_state"],
        inputs["conv_weights"],
        inputs["spec_token_indices"],
        inputs["decode_token_indices"],
        inputs["prefill_token_indices"],
        inputs["spec_query_start_loc"],
        inputs["spec_state_indices"],
        inputs["num_accepted_tokens"],
        inputs["decode_state_indices"],
        inputs["decode_has_initial_state"],
        inputs["prefill_query_start_loc"],
        inputs["prefill_state_indices"],
        inputs["prefill_has_initial_state"],
        output,
        meta["num_spec_tokens"],
        meta["dilation"],
        meta["state_dim_first"],
        meta["null_block_id"],
    )
    torch.xpu.synchronize()
    assert result is output
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])
    _assert_equal(
        _storage_flat(inputs["conv_state"]),
        golden["final_conv_state_storage"],
    )
    for name, tensor in before.items():
        assert torch.equal(inputs[name], tensor), name


def test_spec_supports_independent_offset_and_accepted_dtypes(
    device: torch.device,
) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_spec", device)
    for offset_dtype in (torch.int32, torch.int64):
        for accepted_dtype in (torch.int32, torch.int64):
            state = inputs["conv_state"].clone()
            output = torch.empty_like(golden["output"])
            query_start_loc = inputs["query_start_loc"].to(offset_dtype)
            accepted = inputs["num_accepted_tokens"].to(accepted_dtype)
            torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec(
                inputs["input"], query_start_loc, state,
                inputs["conv_weights"], inputs["state_indices"], accepted,
                output, meta["num_spec_tokens"], meta["dilation"],
                meta["state_dim_first"], meta["null_block_id"],
            )
            torch.xpu.synchronize()
            _assert_equal(output, golden["output"])
            _assert_equal(state, golden["final_conv_state"])


def test_trusted_decode_accepts_reserved_null_slot_zero(
    device: torch.device,
) -> None:
    inputs, _, meta = _load_case("ple_short_conv_decode", device)
    reserved = torch.full_like(inputs["conv_state"][:1], 17.0)
    initial_state = torch.cat((reserved, inputs["conv_state"]), dim=0)
    trusted_state = initial_state.clone()
    legacy_state = initial_state.clone()
    source_indices = inputs["state_indices"].to(torch.int64)
    trusted_indices = torch.where(
        source_indices == meta["null_block_id"],
        torch.zeros_like(source_indices),
        source_indices + 1,
    )
    legacy_indices = torch.where(
        trusted_indices == 0,
        torch.full_like(trusted_indices, meta["null_block_id"]),
        trusted_indices,
    )
    trusted_output = torch.empty_like(inputs["input"])
    legacy_output = torch.empty_like(inputs["input"])

    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_decode_trusted(
        inputs["input"],
        trusted_state,
        inputs["conv_weights"],
        trusted_indices,
        inputs["has_initial_state"],
        trusted_output,
        meta["dilation"],
        meta["state_dim_first"],
        0,
    )
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_decode(
        inputs["input"],
        legacy_state,
        inputs["conv_weights"],
        legacy_indices,
        inputs["has_initial_state"],
        legacy_output,
        meta["dilation"],
        meta["state_dim_first"],
        -1,
    )
    torch.xpu.synchronize()

    _assert_equal(trusted_output, legacy_output)
    _assert_equal(trusted_state, legacy_state)
    null_rows = trusted_indices == 0
    assert torch.count_nonzero(trusted_output[null_rows]) == 0
    assert torch.equal(trusted_state[0], reserved[0])

    with pytest.raises(RuntimeError, match="must not identify a real state slot"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_decode(
            inputs["input"],
            initial_state.clone(),
            inputs["conv_weights"],
            trusted_indices,
            inputs["has_initial_state"],
            torch.empty_like(inputs["input"]),
            meta["dilation"],
            meta["state_dim_first"],
            0,
        )


def _reserved_zero_state_and_indices(inputs: dict, null_block_id: int):
    reserved = torch.full_like(inputs["conv_state"][:1], 17.0)
    initial_state = torch.cat((reserved, inputs["conv_state"]), dim=0)
    source_indices = inputs["state_indices"].to(torch.int64)
    trusted_indices = torch.where(
        source_indices == null_block_id,
        torch.zeros_like(source_indices),
        source_indices + 1,
    )
    legacy_indices = torch.where(
        trusted_indices == 0,
        torch.full_like(trusted_indices, null_block_id),
        trusted_indices,
    )
    return reserved, initial_state, trusted_indices, legacy_indices


def _force_nonempty_request_to_reserved_zero(
    query_start_loc: torch.Tensor,
    trusted_indices: torch.Tensor,
    legacy_indices: torch.Tensor,
    legacy_null_block_id: int,
) -> None:
    starts = query_start_loc.cpu().tolist()
    request = next(
        request
        for request in range(len(starts) - 1)
        if starts[request] < starts[request + 1]
    )
    trusted_indices[request] = 0
    legacy_indices[request] = legacy_null_block_id


def _null_token_rows(
    query_start_loc: torch.Tensor, trusted_indices: torch.Tensor
) -> list[int]:
    starts = query_start_loc.cpu().tolist()
    slots = trusted_indices.cpu().tolist()
    rows = [
        row
        for request, slot in enumerate(slots)
        if slot == 0
        for row in range(starts[request], starts[request + 1])
    ]
    assert rows
    return rows


def _short_conv_args(
    case_name: str,
    inputs: dict,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    meta: dict,
    null_block_id: int,
) -> tuple:
    if case_name == "ple_short_conv_decode":
        return (
            inputs["input"],
            state,
            inputs["conv_weights"],
            state_indices,
            inputs["has_initial_state"],
            output,
            meta["dilation"],
            meta["state_dim_first"],
            null_block_id,
        )
    prefix = (
        inputs["input"],
        inputs["query_start_loc"],
        state,
        inputs["conv_weights"],
        state_indices,
    )
    if case_name == "ple_short_conv_prefill":
        return (
            *prefix,
            inputs["has_initial_state"],
            output,
            meta["dilation"],
            meta["state_dim_first"],
            null_block_id,
        )
    assert case_name == "ple_short_conv_spec"
    return (
        *prefix,
        inputs["num_accepted_tokens"],
        output,
        meta["num_spec_tokens"],
        meta["dilation"],
        meta["state_dim_first"],
        null_block_id,
    )


@pytest.mark.parametrize(
    "case_name",
    (
        "ple_short_conv_decode",
        "ple_short_conv_prefill",
        "ple_short_conv_spec",
    ),
)
def test_legacy_short_conv_rejects_null_block_id_zero(
    device: torch.device, case_name: str
) -> None:
    inputs, golden, meta = _load_case(case_name, device)
    state = inputs["conv_state"].clone()
    state_before = state.clone()
    assert state.size(0) > 0
    operation = getattr(torch.ops.custom_esimd_kernels_vllm, case_name)
    arguments = _short_conv_args(
        case_name,
        inputs,
        state,
        inputs["state_indices"],
        torch.empty_like(golden["output"]),
        meta,
        0,
    )

    with pytest.raises(RuntimeError, match="null_block_id"):
        operation(*arguments)
    assert torch.equal(state, state_before)


@pytest.mark.parametrize(
    "case_name",
    (
        "ple_short_conv_decode",
        "ple_short_conv_prefill",
        "ple_short_conv_spec",
    ),
)
def test_trusted_short_conv_rejects_nonreserved_state_slot_one(
    device: torch.device, case_name: str
) -> None:
    inputs, golden, meta = _load_case(case_name, device)
    reserved, state, trusted_indices, _ = _reserved_zero_state_and_indices(
        inputs, meta["null_block_id"]
    )
    state_before = state.clone()
    assert state.size(0) > 1
    assert torch.equal(state[0], reserved[0])
    assert torch.equal(state[1], inputs["conv_state"][0])
    operation = getattr(
        torch.ops.custom_esimd_kernels_vllm, f"{case_name}_trusted"
    )
    arguments = _short_conv_args(
        case_name,
        inputs,
        state,
        trusted_indices,
        torch.empty_like(golden["output"]),
        meta,
        1,
    )

    with pytest.raises(RuntimeError, match="null_block_id"):
        operation(*arguments)
    assert torch.equal(state, state_before)


def test_trusted_prefill_accepts_reserved_null_slot_zero(
    device: torch.device,
) -> None:
    inputs, _, meta = _load_case("ple_short_conv_prefill", device)
    reserved, initial_state, trusted_indices, legacy_indices = (
        _reserved_zero_state_and_indices(inputs, meta["null_block_id"])
    )
    _force_nonempty_request_to_reserved_zero(
        inputs["query_start_loc"],
        trusted_indices,
        legacy_indices,
        meta["null_block_id"],
    )
    trusted_state = initial_state.clone()
    legacy_state = initial_state.clone()
    trusted_output = torch.empty_like(inputs["input"])
    legacy_output = torch.empty_like(inputs["input"])

    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_prefill_trusted(
        inputs["input"],
        inputs["query_start_loc"],
        trusted_state,
        inputs["conv_weights"],
        trusted_indices,
        inputs["has_initial_state"],
        trusted_output,
        meta["dilation"],
        meta["state_dim_first"],
        0,
    )
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_prefill(
        inputs["input"],
        inputs["query_start_loc"],
        legacy_state,
        inputs["conv_weights"],
        legacy_indices,
        inputs["has_initial_state"],
        legacy_output,
        meta["dilation"],
        meta["state_dim_first"],
        meta["null_block_id"],
    )
    torch.xpu.synchronize()

    _assert_equal(trusted_output, legacy_output)
    _assert_equal(trusted_state, legacy_state)
    null_rows = _null_token_rows(inputs["query_start_loc"], trusted_indices)
    assert torch.count_nonzero(trusted_output[null_rows]) == 0
    assert torch.equal(trusted_state[0], reserved[0])


def test_trusted_spec_accepts_reserved_null_slot_zero(
    device: torch.device,
) -> None:
    inputs, _, meta = _load_case("ple_short_conv_spec", device)
    reserved, initial_state, trusted_indices, legacy_indices = (
        _reserved_zero_state_and_indices(inputs, meta["null_block_id"])
    )
    _force_nonempty_request_to_reserved_zero(
        inputs["query_start_loc"],
        trusted_indices,
        legacy_indices,
        meta["null_block_id"],
    )
    trusted_state = initial_state.clone()
    legacy_state = initial_state.clone()
    trusted_output = torch.empty_like(inputs["input"])
    legacy_output = torch.empty_like(inputs["input"])

    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec_trusted(
        inputs["input"],
        inputs["query_start_loc"],
        trusted_state,
        inputs["conv_weights"],
        trusted_indices,
        inputs["num_accepted_tokens"],
        trusted_output,
        meta["num_spec_tokens"],
        meta["dilation"],
        meta["state_dim_first"],
        0,
    )
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec(
        inputs["input"],
        inputs["query_start_loc"],
        legacy_state,
        inputs["conv_weights"],
        legacy_indices,
        inputs["num_accepted_tokens"],
        legacy_output,
        meta["num_spec_tokens"],
        meta["dilation"],
        meta["state_dim_first"],
        meta["null_block_id"],
    )
    torch.xpu.synchronize()

    _assert_equal(trusted_output, legacy_output)
    _assert_equal(trusted_state, legacy_state)
    null_rows = _null_token_rows(inputs["query_start_loc"], trusted_indices)
    assert torch.count_nonzero(trusted_output[null_rows]) == 0
    assert torch.equal(trusted_state[0], reserved[0])


def test_trusted_short_conv_rejects_structure_alias_and_capacity(
    device: torch.device,
) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_decode", device)
    operation = torch.ops.custom_esimd_kernels_vllm.ple_short_conv_decode_trusted
    state_before = inputs["conv_state"].clone()

    with pytest.raises(RuntimeError, match="one row per input token"):
        operation(
            inputs["input"],
            inputs["conv_state"],
            inputs["conv_weights"],
            inputs["state_indices"][:-1],
            inputs["has_initial_state"][:-1],
            torch.empty_like(golden["output"]),
            meta["dilation"],
            meta["state_dim_first"],
            meta["null_block_id"],
        )
    assert torch.equal(inputs["conv_state"], state_before)

    with pytest.raises(RuntimeError, match="must not share storage"):
        operation(
            inputs["input"],
            inputs["conv_state"],
            inputs["conv_weights"],
            inputs["state_indices"],
            inputs["has_initial_state"],
            inputs["input"],
            meta["dilation"],
            meta["state_dim_first"],
            meta["null_block_id"],
        )
    assert torch.equal(inputs["conv_state"], state_before)

    required_state = (inputs["conv_weights"].size(1) - 1) * meta["dilation"]
    if meta["state_dim_first"]:
        undersized_state = inputs["conv_state"][..., : required_state - 1].clone()
    else:
        undersized_state = inputs["conv_state"][:, : required_state - 1, :].clone()
    undersized_before = undersized_state.clone()
    with pytest.raises(RuntimeError, match="shape/layout is incompatible"):
        operation(
            inputs["input"],
            undersized_state,
            inputs["conv_weights"],
            inputs["state_indices"],
            inputs["has_initial_state"],
            torch.empty_like(golden["output"]),
            meta["dilation"],
            meta["state_dim_first"],
            meta["null_block_id"],
        )
    assert torch.equal(undersized_state, undersized_before)


def test_trusted_prefill_and_spec_reject_metadata_structure(
    device: torch.device,
) -> None:
    prefill, prefill_golden, prefill_meta = _load_case(
        "ple_short_conv_prefill", device
    )
    with pytest.raises(RuntimeError, match=r"exactly requests \+ 1"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_prefill_trusted(
            prefill["input"],
            prefill["query_start_loc"][:-1],
            prefill["conv_state"],
            prefill["conv_weights"],
            prefill["state_indices"],
            prefill["has_initial_state"],
            torch.empty_like(prefill_golden["output"]),
            prefill_meta["dilation"],
            prefill_meta["state_dim_first"],
            prefill_meta["null_block_id"],
        )

    spec, spec_golden, spec_meta = _load_case("ple_short_conv_spec", device)
    state_before = spec["conv_state"].clone()
    with pytest.raises(RuntimeError, match="one entry per request"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec_trusted(
            spec["input"],
            spec["query_start_loc"],
            spec["conv_state"],
            spec["conv_weights"],
            spec["state_indices"],
            spec["num_accepted_tokens"][:-1],
            torch.empty_like(spec_golden["output"]),
            spec_meta["num_spec_tokens"],
            spec_meta["dilation"],
            spec_meta["state_dim_first"],
            spec_meta["null_block_id"],
        )
    assert torch.equal(spec["conv_state"], state_before)


def test_state_metadata_rejects_duplicate_and_malformed_requests(
    device: torch.device,
) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_decode", device)
    duplicate = inputs["state_indices"].clone()
    duplicate[1] = duplicate[0]
    with pytest.raises(RuntimeError, match="duplicate valid state slot"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_decode(
            inputs["input"], inputs["conv_state"].clone(),
            inputs["conv_weights"], duplicate, inputs["has_initial_state"],
            torch.empty_like(golden["output"]), meta["dilation"],
            meta["state_dim_first"], meta["null_block_id"],
        )

    prefill, prefill_golden, prefill_meta = _load_case(
        "ple_short_conv_prefill", device
    )
    malformed = prefill["query_start_loc"].clone()
    malformed[-1] -= 1
    with pytest.raises(RuntimeError, match="last entry"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_prefill(
            prefill["input"], malformed, prefill["conv_state"].clone(),
            prefill["conv_weights"], prefill["state_indices"],
            prefill["has_initial_state"], torch.empty_like(prefill_golden["output"]),
            prefill_meta["dilation"], prefill_meta["state_dim_first"],
            prefill_meta["null_block_id"],
        )


def test_spec_rejects_accepted_count_out_of_range(device: torch.device) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_spec", device)
    accepted = inputs["num_accepted_tokens"].clone()
    accepted[0] = meta["num_spec_tokens"] + 2
    with pytest.raises(RuntimeError, match="num_accepted_tokens"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec(
            inputs["input"], inputs["query_start_loc"],
            inputs["conv_state"].clone(), inputs["conv_weights"],
            inputs["state_indices"], accepted,
            torch.empty_like(golden["output"]), meta["num_spec_tokens"],
            meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
        )


def test_spec_rejects_zero_accepted_count(device: torch.device) -> None:
    inputs, golden, meta = _load_case("ple_short_conv_spec", device)
    accepted = inputs["num_accepted_tokens"].clone()
    accepted[0] = 0
    before_state = inputs["conv_state"].clone()
    with pytest.raises(RuntimeError, match=r"num_accepted_tokens.*\[1,"):
        torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec(
            inputs["input"], inputs["query_start_loc"],
            inputs["conv_state"], inputs["conv_weights"],
            inputs["state_indices"], accepted,
            torch.empty_like(golden["output"]), meta["num_spec_tokens"],
            meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
        )
    assert torch.equal(inputs["conv_state"], before_state)


def test_spec_valid_empty_request_preserves_state(device: torch.device) -> None:
    inputs, _, meta = _load_case("ple_short_conv_spec", device)
    input_tensor = inputs["input"][:1].contiguous()
    state = inputs["conv_state"].clone()
    before_state = state.clone()
    query_start_loc = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    state_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accepted = torch.ones((2,), dtype=torch.int32, device=device)
    output = torch.empty_like(input_tensor)

    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec(
        input_tensor, query_start_loc, state, inputs["conv_weights"],
        state_indices, accepted, output, meta["num_spec_tokens"],
        meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    assert torch.equal(state[0], before_state[0])
    assert not torch.equal(state[1], before_state[1])
    assert torch.count_nonzero(output) > 0


@pytest.mark.parametrize(
    "case_name",
    (
        "ple_short_conv_decode_float16_sd_padded_offset",
        "ple_short_conv_decode_float16_ds_padded_offset",
        "ple_short_conv_decode_float32_sd_padded_offset",
        "ple_short_conv_decode_float32_ds_padded_offset",
    ),
)
def test_decode_preserves_padded_state_backing(
    device: torch.device, case_name: str
) -> None:
    inputs, golden, meta = _load_case(case_name, device)
    output = torch.empty_like(golden["output"])
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_decode(
        inputs["input"], inputs["conv_state"], inputs["conv_weights"],
        inputs["state_indices"], inputs["has_initial_state"], output,
        meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])
    _assert_equal(
        _storage_flat(inputs["conv_state"]), golden["final_conv_state_storage"]
    )


@pytest.mark.parametrize(
    "case_name",
    (
        "ple_short_conv_prefill_float16_sd_padded_offset",
        "ple_short_conv_prefill_float16_ds_padded_offset",
        "ple_short_conv_prefill_float32_sd_padded_offset",
        "ple_short_conv_prefill_float32_ds_padded_offset",
    ),
)
def test_prefill_preserves_padded_state_backing(
    device: torch.device, case_name: str
) -> None:
    inputs, golden, meta = _load_case(case_name, device)
    output = torch.empty_like(golden["output"])
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_prefill(
        inputs["input"], inputs["query_start_loc"], inputs["conv_state"],
        inputs["conv_weights"], inputs["state_indices"],
        inputs["has_initial_state"], output, meta["dilation"],
        meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])
    _assert_equal(
        _storage_flat(inputs["conv_state"]), golden["final_conv_state_storage"]
    )


@pytest.mark.parametrize(
    "case_name",
    (
        "ple_short_conv_spec_float16_sd_padded_offset",
        "ple_short_conv_spec_float16_ds_padded_offset",
        "ple_short_conv_spec_float32_sd_padded_offset",
        "ple_short_conv_spec_float32_ds_padded_offset",
    ),
)
def test_spec_preserves_padded_state_backing(
    device: torch.device, case_name: str
) -> None:
    inputs, golden, meta = _load_case(case_name, device)
    output = torch.empty_like(golden["output"])
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_spec(
        inputs["input"], inputs["query_start_loc"], inputs["conv_state"],
        inputs["conv_weights"], inputs["state_indices"],
        inputs["num_accepted_tokens"], output, meta["num_spec_tokens"],
        meta["dilation"], meta["state_dim_first"], meta["null_block_id"],
    )
    torch.xpu.synchronize()
    _assert_equal(output, golden["output"])
    _assert_equal(inputs["conv_state"], golden["final_conv_state"])
    _assert_equal(
        _storage_flat(inputs["conv_state"]), golden["final_conv_state_storage"]
    )


def test_empty_prefill_clears_caller_owned_output(device: torch.device) -> None:
    width, state_len = 2, 9
    input_tensor = torch.empty((0, width), dtype=torch.float16, device=device)
    query_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    state = torch.zeros((1, width, state_len), dtype=torch.float32, device=device)
    weights = torch.ones((width, 4), dtype=torch.float16, device=device)
    indices = torch.empty((0,), dtype=torch.int32, device=device)
    initial = torch.empty((0,), dtype=torch.bool, device=device)
    output = torch.full_like(input_tensor, float("nan"))
    torch.ops.custom_esimd_kernels_vllm.ple_short_conv_prefill(
        input_tensor, query_start_loc, state, weights, indices, initial, output,
        3, True, -1,
    )
    torch.xpu.synchronize()
    assert torch.equal(output, torch.zeros_like(output))


def test_ngram_rejects_zero_vocab_size(device: torch.device) -> None:
    inputs, golden, meta = _load_case("ple_ngram_ids_decode", device)
    vocab = inputs["ngram_heads_vocab_sizes"].clone()
    vocab[0] = 0
    with pytest.raises(RuntimeError, match="vocab sizes"):
        torch.ops.custom_esimd_kernels_vllm.ple_ngram_ids(
            inputs["input_ids"], inputs["query_start_loc"],
            inputs["ngram_context"], inputs["layer_multipliers"], vocab,
            inputs["ngram_heads_offsets"], torch.empty_like(golden["ngram_ids"]),
            meta["eos_token_id"], meta["heads_per_ngram"],
        )


def test_embedding_metadata_rejects_negative_oversized_and_overflow_values(
    device: torch.device,
) -> None:
    inputs, golden, _ = _load_case("ple_embedding_local_assembly", device)
    cases = (
        ("negative", "local_vocab_start", -1, r"non-negative"),
        ("oversized", "local_num_rows", inputs["local_weight"].size(0) + 1,
         r"must not exceed"),
        ("overflow", "local_vocab_start", 2**63 - 1, r"overflows int64"),
    )
    for _, field, value, message in cases:
        start = inputs["local_vocab_start"].clone()
        rows = inputs["local_num_rows"].clone()
        (start if field == "local_vocab_start" else rows)[0] = value
        output = torch.full_like(golden["local_partial"], 19.0)
        with pytest.raises(RuntimeError, match=message):
            torch.ops.custom_esimd_kernels_vllm.ple_embedding_gather(
                inputs["ngram_ids"], inputs["local_weight"], start, rows, output
            )
        assert torch.equal(output, torch.full_like(output, 19.0))


def test_primitive_alias_guards_reject_storage_views(device: torch.device) -> None:
    ops = torch.ops.custom_esimd_kernels_vllm

    ngram_storage = torch.zeros((8,), dtype=torch.int64, device=device)
    ngram_input = ngram_storage[:2]
    ngram_input.copy_(torch.tensor([1, 2], dtype=torch.int64, device=device))
    ngram_output = ngram_storage[:4].reshape(2, 2)
    with pytest.raises(RuntimeError, match="must not alias N-gram"):
        ops.ple_ngram_ids(
            ngram_input,
            torch.tensor([0, 2], dtype=torch.int64, device=device),
            torch.tensor([[99]], dtype=torch.int64, device=device),
            torch.tensor([3, 5], dtype=torch.int64, device=device),
            torch.tensor([17, 19], dtype=torch.int64, device=device),
            torch.tensor([0, 17], dtype=torch.int64, device=device),
            ngram_output,
            99,
            2,
        )

    raw_storage = torch.empty((32,), dtype=torch.uint8, device=device)
    embedding_ids = raw_storage.view(torch.int64)[:2].reshape(1, 2)
    embedding_ids.fill_(0)
    embedding_output = raw_storage.view(torch.float16)[:4].reshape(1, 4)
    with pytest.raises(RuntimeError, match="local_partial must not share storage"):
        ops.ple_embedding_gather(
            embedding_ids,
            torch.ones((1, 2), dtype=torch.float16, device=device),
            torch.tensor([0], dtype=torch.int64, device=device),
            torch.tensor([1], dtype=torch.int64, device=device),
            embedding_output,
        )

    norm_storage = torch.empty((9,), dtype=torch.float16, device=device)
    norm_input = norm_storage[:8].reshape(2, 4)
    norm_output = norm_storage[1:9].reshape(2, 4)
    with pytest.raises(RuntimeError, match="must not share storage"):
        ops.ple_grouped_norm(
            norm_input,
            torch.ones((4,), dtype=torch.float16, device=device),
            norm_output,
            1.0e-5,
            4,
        )

    key = torch.zeros((1, 2, 4), dtype=torch.float16, device=device)
    query = torch.zeros_like(key)
    with pytest.raises(RuntimeError, match="must not share storage"):
        ops.ple_score_gate(key, query, key.reshape(-1)[:2], 4)

    gated_storage = torch.empty((8,), dtype=torch.float16, device=device)
    gate = gated_storage[:2].reshape(1, 2, 1)
    gated_output = gated_storage.reshape(1, 2, 4)
    with pytest.raises(RuntimeError, match="must not share storage"):
        ops.ple_gated_value(
            gate,
            torch.ones((1, 4), dtype=torch.float16, device=device),
            gated_output,
            2,
        )

    residual_input = torch.zeros((1, 4), dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError, match="must not share storage"):
        ops.ple_residual_add(
            residual_input,
            torch.zeros_like(residual_input),
            residual_input,
        )

    decode_inputs, _, decode_meta = _load_case(
        "ple_short_conv_decode", device
    )
    with pytest.raises(RuntimeError, match="must not share storage"):
        ops.ple_short_conv_decode(
            decode_inputs["input"],
            decode_inputs["conv_state"],
            decode_inputs["conv_weights"],
            decode_inputs["state_indices"],
            decode_inputs["has_initial_state"],
            decode_inputs["input"],
            decode_meta["dilation"],
            decode_meta["state_dim_first"],
            decode_meta["null_block_id"],
        )
