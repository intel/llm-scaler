"""CPU tests for PLE wrapper validation and mixed transaction semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


REFERENCE_PATH = Path(__file__).resolve().parent / "ple_reference.py"
_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "ple_reference_contract_test", REFERENCE_PATH
)
assert _REFERENCE_SPEC is not None and _REFERENCE_SPEC.loader is not None
reference = importlib.util.module_from_spec(_REFERENCE_SPEC)
_REFERENCE_SPEC.loader.exec_module(reference)


OPS_PATH = Path(__file__).resolve().parents[1] / (
    "python/custom_esimd_kernels_vllm/ops.py"
)
_SPEC = importlib.util.spec_from_file_location("ple_ops_contract_test", OPS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops)


def _valid_mixed_case(*, mode: str = "decode") -> dict[str, torch.Tensor | int | str]:
    input_tensor = torch.arange(3 * 2, dtype=torch.float16).reshape(3, 2)
    state = torch.arange(3 * 2 * 6, dtype=torch.float32).reshape(3, 2, 6)
    weights = torch.ones((2, 2), dtype=torch.float16)
    if mode == "decode":
        non_query = torch.tensor([0, 2], dtype=torch.int32)
        non_state = torch.tensor([1, -1], dtype=torch.int32)
        non_initial = torch.tensor([True, False], dtype=torch.bool)
    else:
        non_query = torch.tensor([0, 1, 2], dtype=torch.int32)
        non_state = torch.tensor([1, -1], dtype=torch.int32)
        non_initial = torch.tensor([True, True], dtype=torch.bool)
    return {
        "input": input_tensor,
        "conv_state": state,
        "conv_weights": weights,
        "spec_token_indices": torch.tensor([2], dtype=torch.int32),
        "non_spec_token_indices": torch.tensor([0, 1], dtype=torch.int32),
        "spec_query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "spec_state_indices": torch.tensor([0], dtype=torch.int32),
        "num_accepted_tokens": torch.tensor([1], dtype=torch.int32),
        "non_spec_mode": mode,
        "non_spec_query_start_loc": non_query,
        "non_spec_state_indices": non_state,
        "non_spec_has_initial_state": non_initial,
        "output": torch.full_like(input_tensor, -7),
        "num_spec_tokens": 2,
        "dilation": 3,
        "state_dim_first": True,
        "null_block_id": -1,
    }


def _call_mixed(case: dict[str, object]) -> torch.Tensor:
    return ops.ple_short_conv_mixed(**case)


def _valid_three_way_case() -> dict[str, object]:
    input_tensor = torch.arange(5 * 2, dtype=torch.float16).reshape(5, 2)
    state = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)
    return {
        "input": input_tensor,
        "conv_state": state,
        "conv_weights": torch.ones((2, 2), dtype=torch.float16),
        "spec_token_indices": torch.tensor([4], dtype=torch.int32),
        "decode_token_indices": torch.tensor([1, 3], dtype=torch.int32),
        "prefill_token_indices": torch.tensor([2, 0], dtype=torch.int32),
        "spec_query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "spec_state_indices": torch.tensor([0], dtype=torch.int32),
        "num_accepted_tokens": torch.tensor([1], dtype=torch.int32),
        "decode_state_indices": torch.tensor([1, 2], dtype=torch.int32),
        "decode_has_initial_state": torch.tensor([True, False]),
        "prefill_query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "prefill_state_indices": torch.tensor([3], dtype=torch.int32),
        "prefill_has_initial_state": torch.tensor([True]),
        "output": torch.full_like(input_tensor, -7),
        "num_spec_tokens": 2,
        "dilation": 3,
        "state_dim_first": True,
        "null_block_id": -1,
    }


def test_three_way_mixed_executes_in_order_and_restores_packed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_three_way_case()
    calls: list[str] = []

    def fake_spec(*args: object) -> torch.Tensor:
        calls.append("spec")
        state = args[2]
        output = args[6]
        assert isinstance(state, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        state.add_(1)
        output.fill_(11)
        return output

    def fake_decode(*args: object) -> torch.Tensor:
        calls.append("decode")
        state = args[1]
        output = args[5]
        assert isinstance(state, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        state.add_(10)
        output.fill_(22)
        return output

    def fake_prefill(*args: object) -> torch.Tensor:
        calls.append("prefill")
        state = args[2]
        output = args[6]
        assert isinstance(state, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        state.add_(100)
        output.fill_(33)
        return output

    monkeypatch.setattr(ops, "ple_short_conv_spec", fake_spec)
    monkeypatch.setattr(ops, "ple_short_conv_decode", fake_decode)
    monkeypatch.setattr(ops, "ple_short_conv_prefill", fake_prefill)

    before_state = case["conv_state"].clone()
    result = ops.ple_short_conv_mixed_three_way(**case)

    assert calls == ["spec", "decode", "prefill"]
    expected_output = torch.tensor(
        [[33, 33], [22, 22], [33, 33], [22, 22], [11, 11]],
        dtype=torch.float16,
    )
    assert torch.equal(result, expected_output)
    assert result is case["output"]
    assert torch.equal(case["conv_state"], before_state + 111)


def test_three_way_mixed_rejects_cross_branch_state_duplicate_before_launch() -> None:
    case = _valid_three_way_case()
    case["prefill_state_indices"] = torch.tensor([0], dtype=torch.int32)
    before_state = case["conv_state"].clone()
    before_output = case["output"].clone()

    with pytest.raises(ValueError, match="duplicate valid"):
        ops.ple_short_conv_mixed_three_way(**case)
    assert torch.equal(case["conv_state"], before_state)
    assert torch.equal(case["output"], before_output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "decode_token_indices",
            torch.tensor([1, 1], dtype=torch.int32),
            "permutation",
        ),
        (
            "num_accepted_tokens",
            torch.tensor([0], dtype=torch.int32),
            "supported range",
        ),
        (
            "prefill_query_start_loc",
            torch.tensor([0, 1, 2], dtype=torch.int32),
            "one entry per request",
        ),
        (
            "decode_has_initial_state",
            torch.tensor([True], dtype=torch.bool),
            "token-indexed",
        ),
    ],
)
def test_three_way_mixed_rejects_malformed_branch_metadata(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    case = _valid_three_way_case()
    case[field] = value
    before_state = case["conv_state"].clone()
    before_output = case["output"].clone()

    with pytest.raises(ValueError, match=message):
        ops.ple_short_conv_mixed_three_way(**case)
    assert torch.equal(case["conv_state"], before_state)
    assert torch.equal(case["output"], before_output)


def test_three_way_mixed_empty_spec_branch_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_three_way_case()
    case.update(
        {
            "spec_token_indices": torch.empty(0, dtype=torch.int32),
            "spec_query_start_loc": torch.tensor([0], dtype=torch.int32),
            "spec_state_indices": torch.empty(0, dtype=torch.int32),
            "num_accepted_tokens": torch.empty(0, dtype=torch.int32),
            "decode_token_indices": torch.tensor([1, 3], dtype=torch.int32),
            "prefill_token_indices": torch.tensor([4, 2, 0], dtype=torch.int32),
            "decode_state_indices": torch.tensor([0, 1], dtype=torch.int32),
            "decode_has_initial_state": torch.tensor([True, True]),
            "prefill_query_start_loc": torch.tensor([0, 1, 3], dtype=torch.int32),
            "prefill_state_indices": torch.tensor([2, 3], dtype=torch.int32),
            "prefill_has_initial_state": torch.tensor([True, True]),
        }
    )
    calls: list[str] = []

    def fail_spec(*args: object) -> torch.Tensor:
        raise AssertionError("empty spec branch must not launch")

    def fake_decode(*args: object) -> torch.Tensor:
        calls.append("decode")
        output = args[5]
        assert isinstance(output, torch.Tensor)
        output.fill_(22)
        return output

    def fake_prefill(*args: object) -> torch.Tensor:
        calls.append("prefill")
        output = args[6]
        assert isinstance(output, torch.Tensor)
        output.fill_(33)
        return output

    monkeypatch.setattr(ops, "ple_short_conv_spec", fail_spec)
    monkeypatch.setattr(ops, "ple_short_conv_decode", fake_decode)
    monkeypatch.setattr(ops, "ple_short_conv_prefill", fake_prefill)

    result = ops.ple_short_conv_mixed_three_way(**case)
    assert calls == ["decode", "prefill"]
    assert torch.equal(
        result,
        torch.tensor(
            [[33, 33], [22, 22], [33, 33], [22, 22], [33, 33]],
            dtype=torch.float16,
        ),
    )


def test_three_way_mixed_does_not_commit_after_third_branch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_three_way_case()
    before_state = case["conv_state"].clone()
    before_output = case["output"].clone()
    calls: list[str] = []

    def fake_spec(*args: object) -> torch.Tensor:
        calls.append("spec")
        state = args[2]
        output = args[6]
        assert isinstance(state, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        state.add_(1)
        output.fill_(11)
        return output

    def fake_decode(*args: object) -> torch.Tensor:
        calls.append("decode")
        state = args[1]
        output = args[5]
        assert isinstance(state, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        state.add_(10)
        output.fill_(22)
        return output

    def failing_prefill(*args: object) -> torch.Tensor:
        calls.append("prefill")
        state = args[2]
        assert isinstance(state, torch.Tensor)
        state.add_(100)
        raise RuntimeError("simulated third branch failure")

    monkeypatch.setattr(ops, "ple_short_conv_spec", fake_spec)
    monkeypatch.setattr(ops, "ple_short_conv_decode", fake_decode)
    monkeypatch.setattr(ops, "ple_short_conv_prefill", failing_prefill)

    with pytest.raises(RuntimeError, match="simulated third"):
        ops.ple_short_conv_mixed_three_way(**case)
    assert calls == ["spec", "decode", "prefill"]
    assert torch.equal(case["conv_state"], before_state)
    assert torch.equal(case["output"], before_output)


def test_reference_three_way_matches_explicit_branch_sequence() -> None:
    case = _valid_three_way_case()
    mixed_output, mixed_state = reference.short_conv_mixed_three_way(
        case["input"],
        case["conv_state"],
        case["conv_weights"],
        case["spec_token_indices"],
        case["decode_token_indices"],
        case["prefill_token_indices"],
        case["spec_query_start_loc"],
        case["spec_state_indices"],
        case["num_accepted_tokens"],
        case["decode_state_indices"],
        case["decode_has_initial_state"],
        case["prefill_query_start_loc"],
        case["prefill_state_indices"],
        case["prefill_has_initial_state"],
        dilation=case["dilation"],
        num_spec_tokens=case["num_spec_tokens"],
        null_block_id=case["null_block_id"],
        state_dim_first=case["state_dim_first"],
    )
    spec_x = case["input"].index_select(0, torch.tensor([4]))
    decode_x = case["input"].index_select(0, torch.tensor([1, 3]))
    prefill_x = case["input"].index_select(0, torch.tensor([2, 0]))
    spec_output, state = reference.short_conv_spec(
        spec_x,
        case["spec_query_start_loc"],
        case["conv_state"],
        case["conv_weights"],
        case["spec_state_indices"],
        case["num_accepted_tokens"],
        dilation=case["dilation"],
        num_spec_tokens=case["num_spec_tokens"],
    )
    decode_output, state = reference.short_conv_decode(
        decode_x,
        state,
        case["conv_weights"],
        case["decode_state_indices"],
        case["decode_has_initial_state"],
        dilation=case["dilation"],
    )
    prefill_output, expected_state = reference.short_conv_prefill(
        prefill_x,
        case["prefill_query_start_loc"],
        state,
        case["conv_weights"],
        case["prefill_state_indices"],
        case["prefill_has_initial_state"],
        dilation=case["dilation"],
    )
    expected_output = torch.empty_like(case["input"])
    expected_output.index_copy_(0, torch.tensor([4]), spec_output)
    expected_output.index_copy_(0, torch.tensor([1, 3]), decode_output)
    expected_output.index_copy_(0, torch.tensor([2, 0]), prefill_output)
    assert torch.equal(mixed_output, expected_output)
    assert torch.equal(mixed_state, expected_state)


def test_projection_int4_chunks_rows_above_native_gemm_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tensor = torch.zeros((65, 4), dtype=torch.float16)
    weight = torch.zeros((8, 2), dtype=torch.uint8)
    scales = torch.ones((8, 1), dtype=torch.float16)
    output = torch.full((65, 8), -7.0, dtype=torch.float16)
    calls: list[tuple[str, int, int]] = []

    def fake_validate(
        input: torch.Tensor,
        weight_esimd: torch.Tensor,
        scale_esimd: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[int, int, int]:
        return input.size(0), weight_esimd.size(0), input.size(1)

    def fake_gemm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(("gemm", input.size(0), output.size(0)))
        output.fill_(2)
        return output

    def fake_gemv(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(("gemv", input.size(0), output.size(0)))
        output.fill_(1)
        return output

    monkeypatch.setattr(ops, "_validate_ple_projection_int4", fake_validate)
    monkeypatch.setattr(ops, "esimd_gemm_int4_pgrp", fake_gemm)
    monkeypatch.setattr(ops, "esimd_gemv_int4", fake_gemv)

    result = ops.ple_projection_int4(input_tensor, weight, scales, output)

    assert result is output
    assert calls == [("gemm", 64, 64), ("gemv", 1, 1)]
    assert torch.equal(output[:64], torch.full((64, 8), 2.0, dtype=output.dtype))
    assert torch.equal(output[64:], torch.ones((1, 8), dtype=output.dtype))


def test_mixed_preflight_rejects_duplicate_state_before_launch() -> None:
    case = _valid_mixed_case()
    case["non_spec_state_indices"] = torch.tensor([0, -1], dtype=torch.int32)
    before_state = case["conv_state"].clone()
    before_output = case["output"].clone()
    with pytest.raises(ValueError, match="duplicate valid"):
        _call_mixed(case)
    assert torch.equal(case["conv_state"], before_state)
    assert torch.equal(case["output"], before_output)


def test_mixed_preflight_rejects_zero_accepted_count() -> None:
    case = _valid_mixed_case()
    case["num_accepted_tokens"] = torch.tensor([0], dtype=torch.int32)
    before_state = case["conv_state"].clone()
    before_output = case["output"].clone()
    with pytest.raises(ValueError, match="supported range"):
        _call_mixed(case)
    assert torch.equal(case["conv_state"], before_state)
    assert torch.equal(case["output"], before_output)


def test_mixed_state_transaction_does_not_commit_after_second_branch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _valid_mixed_case()
    before_state = case["conv_state"].clone()
    before_output = case["output"].clone()
    calls: list[str] = []

    def fake_spec(
        input: torch.Tensor,
        query_start_loc: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        output: torch.Tensor,
        num_spec_tokens: int,
        dilation: int,
        state_dim_first: bool,
        null_block_id: int,
    ) -> torch.Tensor:
        calls.append("spec")
        conv_state.add_(10)
        output.fill_(1)
        return output

    def failing_decode(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append("decode")
        state = args[1]
        assert isinstance(state, torch.Tensor)
        state.add_(20)
        raise RuntimeError("simulated second branch failure")

    monkeypatch.setattr(ops, "ple_short_conv_spec", fake_spec)
    monkeypatch.setattr(ops, "ple_short_conv_decode", failing_decode)
    with pytest.raises(RuntimeError, match="simulated"):
        _call_mixed(case)
    assert calls == ["spec", "decode"]
    assert torch.equal(case["conv_state"], before_state)
    assert torch.equal(case["output"], before_output)


def test_mixed_preflight_rejects_output_alias_with_metadata() -> None:
    case = _valid_mixed_case()
    case["output"] = case["input"]
    with pytest.raises(ValueError, match="output"):
        _call_mixed(case)


def test_mixed_prefill_requires_request_indexed_offsets_and_mask() -> None:
    case = _valid_mixed_case(mode="prefill")
    case["non_spec_has_initial_state"] = torch.tensor([True], dtype=torch.bool)
    with pytest.raises(ValueError, match="initial mask"):
        _call_mixed(case)
