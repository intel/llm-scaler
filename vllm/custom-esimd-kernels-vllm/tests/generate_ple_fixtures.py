"""Generate deterministic raw-bin PLE fixtures and a self-describing manifest.

The generator depends only on ``ple_reference``.  Candidate kernels are never
used to produce the golden files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ple_reference import (  # noqa: E402
    embedding_assemble,
    embedding_local,
    grouped_norm,
    ngram_ids,
    projection_int4,
    score_gate,
    short_conv_decode,
    short_conv_mixed_three_way,
    short_conv_prefill,
    short_conv_spec,
    staged_ple,
    staged_ple_full,
)


def _dtype_name(value: torch.Tensor | torch.dtype) -> str:
    dtype = value if isinstance(value, torch.dtype) else value.dtype
    names = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported fixture dtype: {dtype}") from exc


def _storage_numel(tensor: torch.Tensor) -> int:
    element_size = tensor.element_size()
    storage_nbytes = tensor.untyped_storage().nbytes()
    if element_size <= 0 or storage_nbytes % element_size:
        raise ValueError(f"storage size is not aligned for {tensor.dtype}")
    return storage_nbytes // element_size


def _storage_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Return a typed 1-D view over the complete backing storage."""
    return torch.empty(0, dtype=tensor.dtype).set_(
        tensor.untyped_storage(),
        0,
        (_storage_numel(tensor),),
        (1,),
    )


def _write_tensor(
    root: Path,
    tensor: torch.Tensor,
    name: str,
    semantic: str,
) -> dict[str, Any]:
    if sys.byteorder != "little":
        raise RuntimeError("PLE fixtures require a little-endian host")
    tensor = tensor.detach().cpu()
    storage_numel = _storage_numel(tensor)
    payload = bytes(tensor.untyped_storage())
    logical_payload = tensor.contiguous().numpy().tobytes(order="C")
    if len(payload) != storage_numel * tensor.element_size():
        raise ValueError(f"unexpected backing storage size for {name}")
    relative = Path("buffers") / f"{name}.bin"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "name": name,
        "path": relative.as_posix(),
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor),
        "endianness": "little",
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "storage_numel": storage_numel,
        "bytes": len(payload),
        "logical_bytes": len(logical_payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "logical_sha256": hashlib.sha256(logical_payload).hexdigest(),
        "semantic": semantic,
    }


def _payload_root_sha256(
    input_bins: list[dict[str, Any]], output_bins: list[dict[str, Any]]
) -> str:
    digest = hashlib.sha256()
    for kind, entries in (("input", input_bins), ("output", output_bins)):
        for entry in entries:
            digest.update(kind.encode("ascii"))
            digest.update(b"\\0")
            digest.update(entry["name"].encode("utf-8"))
            digest.update(b"\\0")
            digest.update(entry["sha256"].encode("ascii"))
            digest.update(b"\\n")
    return digest.hexdigest()


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    encoded = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_case(
    root: Path,
    name: str,
    inputs: dict[str, tuple[torch.Tensor, str]],
    outputs: dict[str, tuple[torch.Tensor, str]],
    metadata: dict[str, Any],
) -> None:
    case_root = root / name
    input_bins = [
        _write_tensor(case_root, value, key, semantic)
        for key, (value, semantic) in inputs.items()
    ]
    output_bins = [
        _write_tensor(case_root, value, key, semantic)
        for key, (value, semantic) in outputs.items()
    ]
    manifest = {
        "schema": "qwen38.ple.fixture.v2",
        "case": name,
        "generator": "tests/generate_ple_fixtures.py",
        "oracle": "tests/ple_reference.py",
        "inputs": metadata,
        "input_bins": input_bins,
        "output_bins": output_bins,
        "payload_root_sha256": _payload_root_sha256(input_bins, output_bins),
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    (case_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _make_padded_state(
    *,
    slots: int,
    channels: int,
    capacity: int,
    state_dtype: torch.dtype,
    state_dim_first: bool,
    padding: int,
    storage_offset: int,
    sentinel: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a logical state view over padded, sentinel-filled storage."""
    physical_capacity = capacity + padding
    storage_numel = (
        storage_offset + slots * channels * physical_capacity + 3
    )
    backing = torch.full(
        (storage_numel,), sentinel, dtype=state_dtype
    )
    if state_dim_first:
        shape = (slots, channels, capacity)
        stride = (
            channels * physical_capacity,
            physical_capacity,
            1,
        )
    else:
        shape = (slots, capacity, channels)
        stride = (
            physical_capacity * channels,
            channels,
            1,
        )
    state = backing.as_strided(shape, stride, storage_offset)
    values = torch.arange(state.numel(), dtype=torch.float32).reshape(shape)
    state.copy_(values.to(state_dtype))
    return state, backing


def _state_output_storage(
    initial_state: torch.Tensor,
    reference_state: torch.Tensor,
) -> torch.Tensor:
    """Apply reference logical values to an initial physical backing copy."""
    expected_backing = _storage_flat(initial_state).clone()
    expected_view = expected_backing.as_strided(
        initial_state.shape,
        initial_state.stride(),
        initial_state.storage_offset(),
    )
    expected_view.copy_(reference_state)
    return expected_backing


def generate(output_root: Path) -> None:
    torch.manual_seed(20260829)
    output_root.mkdir(parents=True, exist_ok=True)

    input_ids = torch.tensor([7, 8, 99, 11, 13], dtype=torch.int64)
    starts = torch.tensor([0, 3, 5], dtype=torch.int64)
    context = torch.tensor([[5, 6], [99, 99]], dtype=torch.int64)
    multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)
    vocab = torch.tensor([17, 19, 23, 29], dtype=torch.int64)
    offsets = torch.tensor([0, 17, 36, 59], dtype=torch.int64)
    ids = ngram_ids(
        input_ids, starts, context, multipliers, vocab, offsets,
        eos_token_id=99, heads_per_ngram=2,
    )
    _write_case(
        output_root,
        "ple_ngram_ids_prefill_eos",
        {
            "input_ids": (input_ids, "packed token IDs"),
            "query_start_loc": (starts, "request token offsets"),
            "ngram_context": (context, "request-local history"),
            "layer_multipliers": (multipliers, "signed int64 hash multipliers"),
            "ngram_heads_vocab_sizes": (vocab, "per-head positive modulus"),
            "ngram_heads_offsets": (offsets, "per-head global ID offset"),
        },
        {"ngram_ids": (ids, "EOS-aware global N-gram IDs")},
        {"eos_token_id": 99, "heads_per_ngram": 2},
    )

    embed_ids = ids[:2]
    weight = torch.arange(6 * 4, dtype=torch.float16).reshape(6, 4)
    local = embedding_local(embed_ids, weight, 2, 3)
    stacked = torch.stack(
        [local + float(rank) for rank in range(8)], dim=0
    )
    assembled = embedding_assemble(stacked)
    _write_case(
        output_root,
        "ple_embedding_local_assembly",
        {
            "ngram_ids": (embed_ids, "global N-gram IDs"),
            "local_weight": (weight, "runtime local embedding shard"),
            "local_vocab_start": (torch.tensor([2], dtype=torch.int64), "shard start"),
            "local_num_rows": (torch.tensor([3], dtype=torch.int64), "shard rows"),
            "rank_local_partials": (stacked, "deterministic TP partials"),
        },
        {
            "local_partial": (local, "one-rank local partial before reduction"),
            "assembled_embedding": (assembled, "sum across explicit TP partials"),
        },
        {"tp_contract_width": 8, "tp_world_size": 8,
         "assembly": "sum_of_eight_explicit_local_partials"},
    )

    projection_input = torch.arange(128, dtype=torch.float16).reshape(1, 128) / 17
    packed = torch.randint(0, 256, (16, 64), dtype=torch.uint8)
    scales = (torch.arange(16, dtype=torch.float16).reshape(16, 1) + 1) / 8
    projection_output = projection_int4(projection_input, packed, scales)
    _write_case(
        output_root,
        "ple_projection_int4_key_value",
        {
            "input": (projection_input, "FP16 activation"),
            "weight_esimd": (packed, "offset-binary Q4_0 packed weight"),
            "scale_esimd": (scales, "FP16 per-128-K scale"),
        },
        {"output": (projection_output, "FP32-accumulated FP16 projection")},
        {"N": 16, "K": 128, "group_size": 128, "nibble_order": "low_even_high_odd"},
    )

    # Canonical Qwen3.8 PLE projections use replicated row-major weights with
    # K=2560 and separate key/value output widths.  Keep M=1 and M>1 in
    # distinct cases so the XPU harness exercises both GEMV and GEMM dispatch.
    canonical_k = 2560
    canonical_specs = (
        ("ple_projection_int4_canonical_gemv", 1),
        ("ple_projection_int4_canonical_gemm", 4),
    )
    canonical_input = torch.randn(4, canonical_k, dtype=torch.float16) * 0.05
    canonical_packed = {
        "key": torch.randint(0, 256, (10240, canonical_k // 2), dtype=torch.uint8),
        "value": torch.randint(0, 256, (2560, canonical_k // 2), dtype=torch.uint8),
    }
    canonical_scales = {
        name: (torch.rand(weight.size(0), canonical_k // 128, dtype=torch.float32) * 0.25 + 0.01).to(torch.float16)
        for name, weight in canonical_packed.items()
    }
    for case_name, rows in canonical_specs:
        projection_input = canonical_input[:rows].contiguous()
        outputs = {
            name: projection_int4(projection_input, weight, canonical_scales[name])
            for name, weight in canonical_packed.items()
        }
        _write_case(
            output_root,
            case_name,
            {
                "input": (projection_input, "FP16 activation"),
                "key_weight_esimd": (canonical_packed["key"], "replicated key offset-binary Q4_0 packed weight"),
                "key_scale_esimd": (canonical_scales["key"], "key FP16 per-128-K scale"),
                "value_weight_esimd": (canonical_packed["value"], "replicated value offset-binary Q4_0 packed weight"),
                "value_scale_esimd": (canonical_scales["value"], "value FP16 per-128-K scale"),
            },
            {
                "key_output": (outputs["key"], "FP32-accumulated FP16 key projection"),
                "value_output": (outputs["value"], "FP32-accumulated FP16 value projection"),
            },
            {
                "M": rows,
                "N_key": 10240,
                "N_value": 2560,
                "K": canonical_k,
                "group_size": 128,
                "nibble_order": "low_even_high_odd",
                "weight_layout": "replicated_row_major_packed_q4_0",
                "dispatch": "gemv" if rows == 1 else "gemm",
            },
        )

    width, state_len = 3, 9
    x = torch.randn(3, width, dtype=torch.float16)
    state = torch.randn(3, width, state_len, dtype=torch.float32)
    weights = torch.randn(width, 4, dtype=torch.float16)
    indices = torch.tensor([0, -1, 2], dtype=torch.int32)
    initial = torch.tensor([True, True, False])
    decode_out, decode_state = short_conv_decode(
        x, state, weights, indices, initial, dilation=3
    )
    _write_case(
        output_root,
        "ple_short_conv_decode",
        {
            "input": (x, "packed decode activation"),
            "conv_state": (state, "FP32 caller-owned initial state"),
            "conv_weights": (weights, "depthwise dilated weights"),
            "state_indices": (indices, "int32 slot mapping with NULL"),
            "has_initial_state": (initial, "history enable mask"),
        },
        {
            "output": (decode_out, "decode output; NULL row is zero"),
            "final_conv_state": (decode_state, "reference updated state"),
            "final_conv_state_storage": (
                _state_output_storage(state, decode_state),
                "complete expected physical state backing",
            ),
        },
        {"dilation": 3, "state_dim_first": True, "null_block_id": -1},
    )

    prefill_x = torch.randn(4, 2, dtype=torch.float16)
    prefill_state = torch.randn(3, 2, state_len, dtype=torch.float32)
    prefill_weights = torch.randn(2, 4, dtype=torch.float16)
    prefill_starts = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
    prefill_indices = torch.tensor([0, -1, 2], dtype=torch.int32)
    prefill_initial = torch.tensor([True, True, False])
    prefill_out, prefill_final = short_conv_prefill(
        prefill_x, prefill_starts, prefill_state, prefill_weights,
        prefill_indices, prefill_initial, dilation=3,
    )
    _write_case(
        output_root,
        "ple_short_conv_prefill",
        {
            "input": (prefill_x, "ragged packed prefill activation"),
            "query_start_loc": (prefill_starts, "int32 ragged offsets"),
            "conv_state": (prefill_state, "FP32 initial state"),
            "conv_weights": (prefill_weights, "depthwise dilated weights"),
            "state_indices": (prefill_indices, "request slots"),
            "has_initial_state": (prefill_initial, "history enable mask"),
        },
        {
            "output": (prefill_out, "unpacked output in original token order"),
            "final_conv_state": (prefill_final, "per-request final state"),
            "final_conv_state_storage": (
                _state_output_storage(prefill_state, prefill_final),
                "complete expected physical state backing",
            ),
        },
        {"dilation": 3, "state_dim_first": True, "null_block_id": -1},
    )

    spec_x = torch.randn(4, 2, dtype=torch.float16)
    spec_state = torch.randn(2, 2, state_len + 3, dtype=torch.float32)
    spec_weights = torch.randn(2, 4, dtype=torch.float16)
    spec_starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    spec_indices = torch.tensor([0, 1], dtype=torch.int32)
    accepted = torch.tensor([2, 4], dtype=torch.int32)
    spec_out, spec_final = short_conv_spec(
        spec_x, spec_starts, spec_state, spec_weights, spec_indices, accepted,
        dilation=3, num_spec_tokens=3,
    )
    _write_case(
        output_root,
        "ple_short_conv_spec",
        {
            "input": (spec_x, "speculative candidate tokens"),
            "query_start_loc": (spec_starts, "int32 spec ragged offsets"),
            "conv_state": (spec_state, "rollback-capable extended state"),
            "conv_weights": (spec_weights, "depthwise dilated weights"),
            "state_indices": (spec_indices, "spec request slots"),
            "num_accepted_tokens": (accepted, "accepted prefix lengths"),
        },
        {
            "output": (spec_out, "spec output after rollback"),
            "final_conv_state": (spec_final, "candidate extended state"),
            "final_conv_state_storage": (
                _state_output_storage(spec_state, spec_final),
                "complete expected physical state backing",
            ),
        },
        {"dilation": 3, "num_spec_tokens": 3, "state_dim_first": True, "null_block_id": -1},
    )

    def write_decode_state_variant(
        name: str,
        state_dtype: torch.dtype,
        state_dim_first: bool,
    ) -> None:
        variant_x = torch.tensor(
            [[0.5, -1.0], [1.0, 2.0], [-2.0, 0.25]],
            dtype=torch.float16,
        )
        variant_state, _ = _make_padded_state(
            slots=2,
            channels=2,
            capacity=state_len,
            state_dtype=state_dtype,
            state_dim_first=state_dim_first,
            padding=3,
            storage_offset=2,
            sentinel=-777.0,
        )
        variant_weights = torch.tensor(
            [[0.25, -0.5, 0.75, -1.0]] * 2,
            dtype=torch.float16,
        )
        variant_indices = torch.tensor([0, -1, 1], dtype=torch.int32)
        variant_initial = torch.tensor([True, False, True])
        variant_output, variant_final = short_conv_decode(
            variant_x, variant_state, variant_weights, variant_indices,
            variant_initial, dilation=3, state_dim_first=state_dim_first,
        )
        _write_case(
            output_root,
            name,
            {
                "input": (variant_x, "padded-offset decode activation"),
                "conv_state": (variant_state, f"{state_dtype} padded {state_dim_first} state"),
                "conv_weights": (variant_weights, "depthwise dilated weights"),
                "state_indices": (variant_indices, "int32 slot mapping with NULL"),
                "has_initial_state": (variant_initial, "history enable mask"),
            },
            {
                "output": (variant_output, "decode output; NULL row is zero"),
                "final_conv_state": (variant_final, "reference updated state"),
                "final_conv_state_storage": (
                    _state_output_storage(variant_state, variant_final),
                    "complete expected physical state backing",
                ),
            },
            {
                "dilation": 3,
                "state_dim_first": state_dim_first,
                "null_block_id": -1,
                "state_dtype": _dtype_name(variant_state),
                "physical_padding": 3,
                "storage_offset": 2,
                "padding_sentinel": -777.0,
            },
        )

    def write_prefill_state_variant(
        name: str,
        state_dtype: torch.dtype,
        state_dim_first: bool,
    ) -> None:
        variant_x = torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [4.0, -2.0]],
            dtype=torch.float16,
        )
        variant_state, _ = _make_padded_state(
            slots=2,
            channels=2,
            capacity=state_len,
            state_dtype=state_dtype,
            state_dim_first=state_dim_first,
            padding=4,
            storage_offset=3,
            sentinel=-888.0,
        )
        variant_weights = torch.ones((2, 4), dtype=torch.float16)
        variant_starts = torch.tensor([0, 1, 1, 3], dtype=torch.int32)
        variant_indices = torch.tensor([0, -1, 1], dtype=torch.int32)
        variant_initial = torch.tensor([True, True, False])
        variant_output, variant_final = short_conv_prefill(
            variant_x, variant_starts, variant_state, variant_weights,
            variant_indices, variant_initial, dilation=3,
            state_dim_first=state_dim_first,
        )
        _write_case(
            output_root,
            name,
            {
                "input": (variant_x, "padded-offset prefill activation"),
                "query_start_loc": (variant_starts, "int32 ragged offsets"),
                "conv_state": (variant_state, f"{state_dtype} padded {state_dim_first} state"),
                "conv_weights": (variant_weights, "depthwise dilated weights"),
                "state_indices": (variant_indices, "request slots with NULL"),
                "has_initial_state": (variant_initial, "history enable mask"),
            },
            {
                "output": (variant_output, "prefill output in packed token order"),
                "final_conv_state": (variant_final, "reference updated state"),
                "final_conv_state_storage": (
                    _state_output_storage(variant_state, variant_final),
                    "complete expected physical state backing",
                ),
            },
            {
                "dilation": 3,
                "state_dim_first": state_dim_first,
                "null_block_id": -1,
                "state_dtype": _dtype_name(variant_state),
                "physical_padding": 4,
                "storage_offset": 3,
                "padding_sentinel": -888.0,
            },
        )

    def write_spec_state_variant(
        name: str,
        state_dtype: torch.dtype,
        state_dim_first: bool,
    ) -> None:
        variant_x = torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            dtype=torch.float16,
        )
        variant_state, _ = _make_padded_state(
            slots=2,
            channels=2,
            capacity=state_len + 3,
            state_dtype=state_dtype,
            state_dim_first=state_dim_first,
            padding=2,
            storage_offset=4,
            sentinel=-999.0,
        )
        variant_weights = torch.ones((2, 4), dtype=torch.float16)
        variant_starts = torch.tensor([0, 1, 3], dtype=torch.int32)
        variant_indices = torch.tensor([0, 1], dtype=torch.int32)
        variant_accepted = torch.tensor([1, 3], dtype=torch.int32)
        variant_output, variant_final = short_conv_spec(
            variant_x, variant_starts, variant_state, variant_weights,
            variant_indices, variant_accepted, dilation=3,
            num_spec_tokens=3, state_dim_first=state_dim_first,
        )
        _write_case(
            output_root,
            name,
            {
                "input": (variant_x, "padded-offset speculative activation"),
                "query_start_loc": (variant_starts, "int32 spec ragged offsets"),
                "conv_state": (variant_state, f"{state_dtype} padded {state_dim_first} state"),
                "conv_weights": (variant_weights, "depthwise dilated weights"),
                "state_indices": (variant_indices, "spec request slots"),
                "num_accepted_tokens": (variant_accepted, "accepted prefix lengths"),
            },
            {
                "output": (variant_output, "spec output after rollback"),
                "final_conv_state": (variant_final, "candidate extended state"),
                "final_conv_state_storage": (
                    _state_output_storage(variant_state, variant_final),
                    "complete expected physical state backing",
                ),
            },
            {
                "dilation": 3,
                "num_spec_tokens": 3,
                "state_dim_first": state_dim_first,
                "null_block_id": -1,
                "state_dtype": _dtype_name(variant_state),
                "physical_padding": 2,
                "storage_offset": 4,
                "padding_sentinel": -999.0,
            },
        )

    for variant_dtype in (torch.float16, torch.float32):
        for state_dim_first in (True, False):
            layout = "sd" if state_dim_first else "ds"
            write_decode_state_variant(
                f"ple_short_conv_decode_{_dtype_name(variant_dtype)}_"
                f"{layout}_padded_offset",
                variant_dtype,
                state_dim_first=state_dim_first,
            )
            write_prefill_state_variant(
                f"ple_short_conv_prefill_{_dtype_name(variant_dtype)}_"
                f"{layout}_padded_offset",
                variant_dtype,
                state_dim_first=state_dim_first,
            )
            write_spec_state_variant(
                f"ple_short_conv_spec_{_dtype_name(variant_dtype)}_"
                f"{layout}_padded_offset",
                variant_dtype,
                state_dim_first=state_dim_first,
            )

    def write_mixed_state_variant(
        name: str,
        state_dtype: torch.dtype,
        state_dim_first: bool,
    ) -> None:
        mixed_x = torch.tensor(
            [
                [0.5, -1.0],
                [1.0, 2.0],
                [-2.0, 0.25],
                [3.0, -4.0],
                [1.5, 2.5],
                [-0.75, 0.125],
            ],
            dtype=torch.float16,
        )
        num_spec_tokens = 3
        mixed_capacity = state_len + num_spec_tokens
        mixed_state, _ = _make_padded_state(
            slots=5,
            channels=2,
            capacity=mixed_capacity,
            state_dtype=state_dtype,
            state_dim_first=state_dim_first,
            padding=2,
            storage_offset=4,
            sentinel=-1234.0,
        )
        mixed_weights = torch.tensor(
            [[0.25, -0.5, 0.75, -1.0]] * 2,
            dtype=torch.float16,
        )
        spec_tokens = torch.tensor([5, 1], dtype=torch.int32)
        decode_tokens = torch.tensor([4, 0], dtype=torch.int32)
        prefill_tokens = torch.tensor([3, 2], dtype=torch.int32)
        spec_starts = torch.tensor([0, 2], dtype=torch.int32)
        spec_states = torch.tensor([0], dtype=torch.int32)
        accepted = torch.tensor([2], dtype=torch.int32)
        decode_states = torch.tensor([1, -1], dtype=torch.int32)
        decode_initial = torch.tensor([True, False])
        prefill_starts = torch.tensor([0, 0, 2], dtype=torch.int32)
        prefill_states = torch.tensor([3, 4], dtype=torch.int32)
        prefill_initial = torch.tensor([True, False])
        mixed_output, mixed_final = short_conv_mixed_three_way(
            mixed_x,
            mixed_state,
            mixed_weights,
            spec_tokens,
            decode_tokens,
            prefill_tokens,
            spec_starts,
            spec_states,
            accepted,
            decode_states,
            decode_initial,
            prefill_starts,
            prefill_states,
            prefill_initial,
            dilation=3,
            num_spec_tokens=num_spec_tokens,
            state_dim_first=state_dim_first,
        )
        _write_case(
            output_root,
            name,
            {
                "input": (mixed_x, "packed mixed activation"),
                "conv_state": (
                    mixed_state,
                    f"{state_dtype} padded mixed state",
                ),
                "conv_weights": (mixed_weights, "depthwise dilated weights"),
                "spec_token_indices": (
                    spec_tokens,
                    "stable speculative token permutation",
                ),
                "decode_token_indices": (
                    decode_tokens,
                    "stable decode token permutation",
                ),
                "prefill_token_indices": (
                    prefill_tokens,
                    "stable prefill token permutation",
                ),
                "spec_query_start_loc": (
                    spec_starts,
                    "spec request offsets",
                ),
                "spec_state_indices": (spec_states, "spec state slots"),
                "num_accepted_tokens": (accepted, "accepted spec prefix"),
                "decode_state_indices": (
                    decode_states,
                    "decode token state slots with NULL",
                ),
                "decode_has_initial_state": (
                    decode_initial,
                    "decode history enable mask",
                ),
                "prefill_query_start_loc": (
                    prefill_starts,
                    "prefill request offsets with empty request",
                ),
                "prefill_state_indices": (
                    prefill_states,
                    "prefill request state slots",
                ),
                "prefill_has_initial_state": (
                    prefill_initial,
                    "prefill history enable mask",
                ),
            },
            {
                "output": (mixed_output, "mixed output in original token order"),
                "final_conv_state": (mixed_final, "mixed final state"),
                "final_conv_state_storage": (
                    _state_output_storage(mixed_state, mixed_final),
                    "complete expected physical state backing",
                ),
            },
            {
                "dilation": 3,
                "num_spec_tokens": num_spec_tokens,
                "state_dim_first": state_dim_first,
                "null_block_id": -1,
                "state_dtype": _dtype_name(mixed_state),
                "physical_padding": 2,
                "storage_offset": 4,
                "padding_sentinel": -1234.0,
                "token_order": "spec_decode_prefill_stable_permutation",
                "execution_order": "spec_decode_prefill",
            },
        )

    for variant_dtype in (torch.float16, torch.float32):
        for state_dim_first in (True, False):
            layout = "sd" if state_dim_first else "ds"
            name = (
                "ple_short_conv_mixed_permutation"
                if variant_dtype == torch.float32 and state_dim_first
                else f"ple_short_conv_mixed_permutation_"
                f"{_dtype_name(variant_dtype)}_{layout}"
            )
            write_mixed_state_variant(name, variant_dtype, state_dim_first)

    tokens, groups, hidden = 2, 2, 8
    embedding = torch.randn(tokens, hidden, dtype=torch.float16)
    hidden_states = torch.randn(tokens, groups * hidden, dtype=torch.float16)
    key_weight = torch.randn(groups * hidden, hidden, dtype=torch.float16)
    value_weight = torch.randn(hidden, hidden, dtype=torch.float16)
    norm = torch.randn(groups * hidden, dtype=torch.float16) / 10
    staged_state = torch.randn(2, groups * hidden, 3, dtype=torch.float32)
    staged_weights = torch.randn(groups * hidden, 2, dtype=torch.float16)
    staged_indices = torch.tensor([0, 1], dtype=torch.int64)
    staged_initial = torch.tensor([True, False])
    staged = staged_ple(
        embedding=embedding,
        hidden_states=hidden_states,
        key_weight=key_weight,
        value_weight=value_weight,
        norm_key_weight=norm,
        norm_query_weight=norm,
        norm_conv_weight=norm,
        conv_state=staged_state,
        conv_weights=staged_weights,
        state_indices=staged_indices,
        has_initial_state=staged_initial,
        mode="decode",
        eps=1e-5,
        group_size=hidden,
        dilation=3,
    )
    staged_inputs = {
        "embedding": (embedding, "assembled PLE embedding"),
        "hidden_states": (hidden_states, "HC query input"),
        "key_weight": (key_weight, "FP16 row-major key projection"),
        "value_weight": (value_weight, "FP16 row-major value projection"),
        "norm_key_weight": (norm, "grouped norm scale"),
        "norm_query_weight": (norm, "grouped norm scale"),
        "norm_conv_weight": (norm, "grouped norm scale"),
        "conv_state": (staged_state, "caller-owned state"),
        "conv_weights": (staged_weights, "depthwise conv weights"),
        "state_indices": (staged_indices, "state slots"),
        "has_initial_state": (staged_initial, "history enable mask"),
    }
    staged_outputs = {
        key: (value, f"staged {key} golden") for key, value in staged.items()
    }
    _write_case(
        output_root,
        "ple_staged_decode_fp16",
        staged_inputs,
        staged_outputs,
        {"mode": "decode", "eps": 1e-5, "group_size": hidden, "dilation": 3},
    )

    # Complete K0-K10 standalone case.  The second explicit partial is a
    # deterministic stand-in for the caller-owned TP assembly boundary; this
    # fixture never starts a process group or claims an eight-rank execution.
    full_input_ids = torch.tensor([7, 8], dtype=torch.int64)
    full_query_start_loc = torch.tensor([0, 2], dtype=torch.int64)
    full_context = torch.tensor([[99, 99]], dtype=torch.int64)
    full_local_weight = (
        torch.arange(128 * 2, dtype=torch.float32).reshape(128, 2) / 17
    ).to(torch.float16)
    full_local_start = torch.tensor([0], dtype=torch.int64)
    full_local_rows = torch.tensor([128], dtype=torch.int64)
    full_rank_partials = torch.full((2, 2, 8), 0.25, dtype=torch.float16)
    full_rank_partials[0].zero_()
    full_hidden_states = torch.randn(2, 16, dtype=torch.float16)
    full_key_weight = torch.randn(16, 8, dtype=torch.float16) * 0.1
    full_value_weight = torch.randn(8, 8, dtype=torch.float16) * 0.1
    full_norm = torch.randn(16, dtype=torch.float16) / 10
    full_state = torch.randn(2, 16, 3, dtype=torch.float32)
    full_conv_weights = torch.randn(16, 2, dtype=torch.float16)
    full_state_indices = torch.tensor([0, 1], dtype=torch.int32)
    full_initial = torch.tensor([True, False])
    full_staged = staged_ple_full(
        input_ids=full_input_ids,
        query_start_loc=full_query_start_loc,
        ngram_context=full_context,
        layer_multipliers=multipliers,
        ngram_heads_vocab_sizes=vocab,
        ngram_heads_offsets=offsets,
        local_weight=full_local_weight,
        local_vocab_start=full_local_start,
        local_num_rows=full_local_rows,
        rank_local_partials=full_rank_partials,
        hidden_states=full_hidden_states,
        key_weight=full_key_weight,
        value_weight=full_value_weight,
        norm_key_weight=full_norm,
        norm_query_weight=full_norm,
        norm_conv_weight=full_norm,
        conv_state=full_state,
        conv_weights=full_conv_weights,
        state_indices=full_state_indices,
        has_initial_state=full_initial,
        mode="decode",
        eps=1e-5,
        group_size=8,
        dilation=3,
        eos_token_id=99,
        heads_per_ngram=2,
    )
    _write_case(
        output_root,
        "ple_staged_full_decode_fp16",
        {
            "input_ids": (full_input_ids, "K0 packed token IDs"),
            "query_start_loc": (full_query_start_loc, "K0 request offsets"),
            "ngram_context": (full_context, "K0 request-local EOS history"),
            "layer_multipliers": (multipliers, "K0 signed int64 hash multipliers"),
            "ngram_heads_vocab_sizes": (vocab, "K0 per-head modulus"),
            "ngram_heads_offsets": (offsets, "K0 per-head ID offsets"),
            "local_weight": (full_local_weight, "K1 local embedding shard"),
            "local_vocab_start": (full_local_start, "K1 shard start"),
            "local_num_rows": (full_local_rows, "K1 shard row count"),
            "rank_local_partials": (
                full_rank_partials,
                "K2 explicit caller-owned local partials",
            ),
            "hidden_states": (full_hidden_states, "HC query input"),
            "key_weight": (full_key_weight, "FP16 row-major key projection"),
            "value_weight": (full_value_weight, "FP16 row-major value projection"),
            "norm_key_weight": (full_norm, "key grouped norm scale"),
            "norm_query_weight": (full_norm, "query grouped norm scale"),
            "norm_conv_weight": (full_norm, "conv grouped norm scale"),
            "conv_state": (full_state, "caller-owned decode state"),
            "conv_weights": (full_conv_weights, "depthwise dilated weights"),
            "state_indices": (full_state_indices, "decode state slots"),
            "has_initial_state": (full_initial, "decode history mask"),
        },
        {
            key: (value, f"full staged {key} golden")
            for key, value in full_staged.items()
        },
        {
            "mode": "decode",
            "eps": 1e-5,
            "group_size": 8,
            "dilation": 3,
            "eos_token_id": 99,
            "heads_per_ngram": 2,
            "projection_kind": "fp16",
            "tp_contract_width": 2,
            "assembly": "explicit_sum_no_communicator",
        },
    )

    # Standalone arithmetic cases keep each primitive independently testable.
    decode_ids = torch.tensor([31], dtype=torch.int64)
    decode_starts = torch.tensor([0, 1], dtype=torch.int64)
    decode_context = torch.tensor([[99, 99]], dtype=torch.int64)
    decode_ngram = ngram_ids(
        decode_ids, decode_starts, decode_context, multipliers, vocab, offsets,
        eos_token_id=99, heads_per_ngram=2,
    )
    _write_case(
        output_root,
        "ple_ngram_ids_decode",
        {
            "input_ids": (decode_ids, "single packed decode token"),
            "query_start_loc": (decode_starts, "int32 decode offsets"),
            "ngram_context": (decode_context, "EOS-padded history"),
            "layer_multipliers": (multipliers, "signed int64 hash multipliers"),
            "ngram_heads_vocab_sizes": (vocab, "per-head modulus"),
            "ngram_heads_offsets": (offsets, "per-head ID offset"),
        },
        {"ngram_ids": (decode_ngram, "single-token EOS-aware IDs")},
        {"eos_token_id": 99, "heads_per_ngram": 2},
    )

    arith_values = torch.randn(2, groups * hidden, dtype=torch.float16)
    arith_weight = torch.randn(groups * hidden, dtype=torch.float16) / 10
    arith_norm = grouped_norm(arith_values, arith_weight, 1e-5, hidden)
    _write_case(
        output_root,
        "ple_grouped_norm_key_query",
        {
            "input": (arith_values, "key or query flattened activation"),
            "weight": (arith_weight, "1-plus grouped norm scale"),
        },
        {"output": (arith_norm, "FP32-variance grouped norm result")},
        {"eps": 1e-5, "group_size": hidden, "accum_dtype": "float32"},
    )

    key = torch.randn(2, groups, hidden, dtype=torch.float16)
    query = torch.randn(2, groups, hidden, dtype=torch.float16)
    gate = score_gate(key, query, hidden)
    _write_case(
        output_root,
        "ple_score_gate",
        {
            "key_norm": (key, "normalized key [T,C,H]"),
            "query_norm": (query, "normalized query [T,C,H]"),
        },
        {"gate": (gate, "signed-square-root sigmoid gate")},
        {"hidden_size": hidden, "accum_dtype": "float32"},
    )

    value = torch.randn(2, hidden, dtype=torch.float16)
    gated = gate * value.unsqueeze(-2)
    gated_norm = grouped_norm(
        gated.flatten(-2), arith_weight, 1e-5, hidden
    )
    _write_case(
        output_root,
        "ple_gated_value_norm",
        {
            "gate": (gate, "C-way gate broadcast"),
            "value": (value, "projected value [T,H]"),
            "norm_weight": (arith_weight, "grouped conv-input scale"),
        },
        {
            "gated_value": (gated, "broadcast gated value [T,C,H]"),
            "normalized": (gated_norm, "flattened grouped-normalized value"),
        },
        {"eps": 1e-5, "group_size": hidden},
    )

    residual = gated.flatten(-2)
    conv = torch.randn_like(residual)
    residual_output = residual + conv
    _write_case(
        output_root,
        "ple_residual_add",
        {
            "gated_value_flat": (residual, "flattened gated value"),
            "conv_output": (conv, "short-conv output"),
        },
        {"output": (residual_output, "caller-owned residual sum")},
        {"accum_dtype": "float16"},
    )

    # The two additional staged branches exercise ragged offsets and rollback
    # independently of ordinary decode; their intermediate outputs remain
    # explicit in the manifest just like the decode case above.
    staged_prefill_state = torch.randn(2, groups * hidden, 3, dtype=torch.float32)
    staged_prefill_starts = torch.tensor([0, 2, 2], dtype=torch.int32)
    staged_prefill_indices = torch.tensor([0, -1], dtype=torch.int32)
    staged_prefill_initial = torch.tensor([True, True])
    staged_prefill_embedding = torch.randn(2, hidden, dtype=torch.float16)
    staged_prefill_hidden = torch.randn(2, groups * hidden, dtype=torch.float16)
    staged_prefill = staged_ple(
        embedding=staged_prefill_embedding,
        hidden_states=staged_prefill_hidden,
        key_weight=key_weight,
        value_weight=value_weight,
        norm_key_weight=norm,
        norm_query_weight=norm,
        norm_conv_weight=norm,
        conv_state=staged_prefill_state,
        conv_weights=staged_weights,
        state_indices=staged_prefill_indices,
        has_initial_state=staged_prefill_initial,
        mode="prefill",
        eps=1e-5,
        group_size=hidden,
        dilation=3,
        query_start_loc=staged_prefill_starts,
    )
    _write_case(
        output_root,
        "ple_staged_prefill_fp16",
        {
            "embedding": (staged_prefill_embedding, "assembled PLE embedding"),
            "hidden_states": (staged_prefill_hidden, "HC query input"),
            "key_weight": (key_weight, "FP16 key projection"),
            "value_weight": (value_weight, "FP16 value projection"),
            "norm_key_weight": (norm, "grouped norm scale"),
            "norm_query_weight": (norm, "grouped norm scale"),
            "norm_conv_weight": (norm, "grouped norm scale"),
            "conv_state": (staged_prefill_state, "caller-owned state"),
            "conv_weights": (staged_weights, "depthwise conv weights"),
            "state_indices": (staged_prefill_indices, "request slots with NULL"),
            "has_initial_state": (staged_prefill_initial, "history mask"),
            "query_start_loc": (staged_prefill_starts, "ragged token offsets"),
        },
        {key: (value, f"staged {key} golden") for key, value in staged_prefill.items()},
        {"mode": "prefill", "eps": 1e-5, "group_size": hidden, "dilation": 3},
    )

    staged_spec_state = torch.randn(2, groups * hidden, 6, dtype=torch.float32)
    staged_spec_x = torch.randn(4, hidden, dtype=torch.float16)
    staged_spec_starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    staged_spec_indices = torch.tensor([0, 1], dtype=torch.int32)
    staged_accepted = torch.tensor([2, 3], dtype=torch.int32)
    staged_spec_embedding = torch.randn(4, hidden, dtype=torch.float16)
    staged_spec_hidden = torch.randn(4, groups * hidden, dtype=torch.float16)
    staged_spec = staged_ple(
        embedding=staged_spec_embedding,
        hidden_states=staged_spec_hidden,
        key_weight=key_weight,
        value_weight=value_weight,
        norm_key_weight=norm,
        norm_query_weight=norm,
        norm_conv_weight=norm,
        conv_state=staged_spec_state,
        conv_weights=staged_weights,
        state_indices=staged_spec_indices,
        has_initial_state=None,
        mode="spec",
        eps=1e-5,
        group_size=hidden,
        dilation=3,
        query_start_loc=staged_spec_starts,
        num_accepted_tokens=staged_accepted,
        num_spec_tokens=3,
    )
    _write_case(
        output_root,
        "ple_staged_spec_fp16",
        {
            "embedding": (staged_spec_embedding, "assembled PLE embedding"),
            "hidden_states": (staged_spec_hidden, "HC query input"),
            "key_weight": (key_weight, "FP16 key projection"),
            "value_weight": (value_weight, "FP16 value projection"),
            "norm_key_weight": (norm, "grouped norm scale"),
            "norm_query_weight": (norm, "grouped norm scale"),
            "norm_conv_weight": (norm, "grouped norm scale"),
            "conv_state": (staged_spec_state, "rollback-capable state"),
            "conv_weights": (staged_weights, "depthwise conv weights"),
            "state_indices": (staged_spec_indices, "spec request slots"),
            "query_start_loc": (staged_spec_starts, "spec ragged offsets"),
            "num_accepted_tokens": (staged_accepted, "accepted prefix lengths"),
        },
        {key: (value, f"staged {key} golden") for key, value in staged_spec.items()},
        {"mode": "spec", "eps": 1e-5, "group_size": hidden, "dilation": 3,
         "num_spec_tokens": 3},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    generate(args.output_root)
