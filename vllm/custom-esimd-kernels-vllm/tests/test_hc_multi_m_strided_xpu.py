"""Real-XPU contract tests for the additive strided HC combine+norm ABI.

The production injection layout is the middle slice of the merged HC gate
projection: ``[M, 336][:, 320:324]``.  It has shape ``[M, 4]`` and stride
``(336, 1)`` for M > 1, so these tests must not materialize it contiguously
before invoking the strided operator.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


STRIDED_SCHEMA = (
    "custom_esimd_kernels_vllm::hc_combine_norm_multi_m_strided_v1"
)
OLD_MULTI_SCHEMA = "custom_esimd_kernels_vllm::hc_combine_norm_multi_m_v1"
OLD_M4_SCHEMA = "custom_esimd_kernels_vllm::hc_combine_norm_m4_v1"

HC_COUNT = 4
HC_HIDDEN = 2560
HC_WIDTH = HC_COUNT * HC_HIDDEN
MERGED_WIDTH = 336
INJECTION_START = 320
EPS = 1.0e-6


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")

    configured = os.environ.get("HC_MULTI_M_STRIDED_DSO") or os.environ.get(
        "HC_CHAIN_DSO"
    )
    if configured:
        dso = Path(configured)
        if not dso.is_file():
            pytest.fail(f"configured HC DSO does not exist: {dso}")
        torch.ops.load_library(str(dso))
    else:
        try:
            import custom_esimd_kernels_vllm  # noqa: F401
        except ImportError as exc:
            pytest.skip(
                "canonical custom_esimd_kernels_vllm package is unavailable: "
                f"{exc}"
            )

    for schema in (STRIDED_SCHEMA, OLD_MULTI_SCHEMA, OLD_M4_SCHEMA):
        if not torch._C._jit_get_schemas_for_operator(schema):
            pytest.fail(f"canonical main DSO did not register {schema}")
    return torch.device("xpu:0")


def _random_fp16(
    device: torch.device,
    shape: tuple[int, ...],
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(device=device, dtype=torch.float16).contiguous()


def _make_case(device: torch.device, rows: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = _random_fp16(device, (rows, HC_WIDTH), generator, 0.20)
    block = _random_fp16(device, (rows, HC_HIDDEN), generator, 0.20)
    merged = _random_fp16(device, (rows, MERGED_WIDTH), generator, 0.50)
    injection = merged[:, INJECTION_START : INJECTION_START + HC_COUNT]
    weight = _random_fp16(device, (HC_WIDTH,), generator, 0.03)

    assert injection.shape == (rows, HC_COUNT)
    assert injection.stride() == (MERGED_WIDTH, 1)
    assert injection.storage_offset() == INJECTION_START
    assert not injection.is_contiguous()
    return hidden, block, merged, injection, weight


def _reference(
    hidden: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hidden.shape[0]
    injection_scale = 2.0 * torch.sigmoid(injection.float() * 0.25)
    combined = (
        hidden.float().view(rows, HC_COUNT, HC_HIDDEN)
        + block.float().unsqueeze(1) * injection_scale.unsqueeze(-1)
    ).to(torch.float16)
    inverse_rms = torch.rsqrt(
        combined.float().square().mean(dim=-1, keepdim=True) + EPS
    )
    normed = (
        combined.float()
        * inverse_rms
        * (1.0 + weight.float().view(1, HC_COUNT, HC_HIDDEN))
    ).to(torch.float16)
    return combined.view_as(hidden), normed.view_as(hidden)


def _outputs(
    device: torch.device, rows: int, first: float = 0.125, second: float = -0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((rows, HC_WIDTH), first, dtype=torch.float16, device=device),
        torch.full((rows, HC_WIDTH), second, dtype=torch.float16, device=device),
    )


def _run_strided(
    hidden: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
    combined: torch.Tensor,
    normed: torch.Tensor,
) -> None:
    result = torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_multi_m_strided_v1(
        hidden, block, injection, weight, combined, normed, EPS
    )
    assert result is None


@pytest.mark.parametrize("rows", range(2, 9))
def test_real_projection_slice_matches_torch_and_old_contiguous_abi(
    device: torch.device, rows: int
) -> None:
    hidden, block, merged, injection, weight = _make_case(
        device, rows, 6100 + rows
    )
    actual = _outputs(device, rows)
    old = _outputs(device, rows)
    expected = _reference(hidden, block, injection, weight)
    input_snapshots = tuple(
        tensor.cpu().clone() for tensor in (hidden, block, merged, weight)
    )
    output_pointers = tuple(tensor.data_ptr() for tensor in actual)

    _run_strided(hidden, block, injection, weight, *actual)
    contiguous_injection = injection.contiguous()
    assert (
        torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_multi_m_v1(
            hidden, block, contiguous_injection, weight, *old, EPS
        )
        is None
    )
    torch.xpu.synchronize()

    assert tuple(tensor.data_ptr() for tensor in actual) == output_pointers
    for value, snapshot in zip((hidden, block, merged, weight), input_snapshots):
        assert torch.equal(value.cpu(), snapshot)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0.0, atol=2.0e-3)
    for value, contiguous_value in zip(actual, old):
        torch.testing.assert_close(value, contiguous_value, rtol=0.0, atol=0.0)


def test_aligned_offset_and_larger_row_stride(device: torch.device) -> None:
    rows = 5
    hidden, block, _, _, weight = _make_case(device, rows, 6205)
    generator = torch.Generator(device="cpu").manual_seed(7205)
    backing = _random_fp16(device, (rows * 337 + 2,), generator, 0.50)
    merged = backing[2:].view(rows, 337)
    injection = merged[:, 320:324]
    assert injection.stride() == (337, 1)
    assert injection.storage_offset() == 322
    actual = _outputs(device, rows)
    expected = _reference(hidden, block, injection, weight)

    _run_strided(hidden, block, injection, weight, *actual)
    torch.xpu.synchronize()

    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0.0, atol=2.0e-3)


def test_current_and_nondefault_streams_are_independent(device: torch.device) -> None:
    rows = 5
    hidden, block, _, injection, weight = _make_case(device, rows, 6305)
    current_outputs = _outputs(device, rows)
    other_outputs = _outputs(device, rows, first=0.375, second=-0.50)
    expected = _reference(hidden, block, injection, weight)
    torch.xpu.synchronize()

    current_stream = torch.xpu.current_stream(device=device)
    other_stream = torch.xpu.Stream(device=device)
    with torch.xpu.stream(current_stream):
        _run_strided(hidden, block, injection, weight, *current_outputs)
    with torch.xpu.stream(other_stream):
        _run_strided(hidden, block, injection, weight, *other_outputs)
    torch.xpu.synchronize()

    for outputs in (current_outputs, other_outputs):
        for value, reference in zip(outputs, expected):
            torch.testing.assert_close(value, reference, rtol=0.0, atol=2.0e-3)
    for current, other in zip(current_outputs, other_outputs):
        torch.testing.assert_close(current, other, rtol=0.0, atol=0.0)


def test_legacy_abis_keep_rejecting_real_strided_layout(
    device: torch.device,
) -> None:
    rows = 4
    hidden, block, _, injection, weight = _make_case(device, rows, 6404)
    ops = torch.ops.custom_esimd_kernels_vllm

    for operation in (
        ops.hc_combine_norm_m4_v1,
        ops.hc_combine_norm_multi_m_v1,
    ):
        outputs = _outputs(device, rows)
        snapshots = tuple(tensor.cpu().clone() for tensor in outputs)
        with pytest.raises(RuntimeError, match="injection must be contiguous"):
            operation(hidden, block, injection, weight, *outputs, EPS)
        torch.xpu.synchronize()
        for value, snapshot in zip(outputs, snapshots):
            assert torch.equal(value.cpu(), snapshot)


def _assert_rejected_without_writes(
    hidden: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
    match: str,
) -> None:
    outputs = _outputs(hidden.device, hidden.shape[0])
    snapshots = tuple(tensor.cpu().clone() for tensor in outputs)
    with pytest.raises(RuntimeError, match=match):
        _run_strided(hidden, block, injection, weight, *outputs)
    torch.xpu.synchronize()
    for value, snapshot in zip(outputs, snapshots):
        assert torch.equal(value.cpu(), snapshot)


@pytest.mark.parametrize("rows", [1, 9])
def test_rejects_unsupported_rows_without_writes(
    device: torch.device, rows: int
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(6500 + rows)
    hidden = _random_fp16(device, (rows, HC_WIDTH), generator, 0.20)
    block = _random_fp16(device, (rows, HC_HIDDEN), generator, 0.20)
    merged = _random_fp16(device, (rows, MERGED_WIDTH), generator, 0.50)
    injection = merged[:, INJECTION_START : INJECTION_START + HC_COUNT]
    weight = _random_fp16(device, (HC_WIDTH,), generator, 0.03)
    _assert_rejected_without_writes(
        hidden, block, injection, weight, "expects hidden/combined/normed"
    )


def test_rejects_invalid_injection_strides_and_alignment_without_writes(
    device: torch.device,
) -> None:
    rows = 3
    hidden, block, _, _, weight = _make_case(device, rows, 6603)

    inner_stride_backing = torch.empty(
        (rows, 8), dtype=torch.float16, device=device
    )
    inner_stride_two = inner_stride_backing[:, ::2]
    assert inner_stride_two.shape == (rows, HC_COUNT)
    assert inner_stride_two.stride() == (8, 2)

    overlap_backing = torch.empty(
        (rows * 3 + 1,), dtype=torch.float16, device=device
    )
    overlapping_rows = overlap_backing.as_strided((rows, HC_COUNT), (3, 1))

    odd_offset_backing = torch.empty(
        (rows, 337), dtype=torch.float16, device=device
    )
    odd_offset = odd_offset_backing[:, 321:325]
    assert odd_offset.stride() == (337, 1)
    assert odd_offset.storage_offset() == 321

    for injection, match in (
        (inner_stride_two, "inner stride 1"),
        (overlapping_rows, "non-overlapping row stride >= 4"),
        (odd_offset, "4-byte aligned"),
    ):
        _assert_rejected_without_writes(hidden, block, injection, weight, match)


def test_other_tensors_keep_old_contiguous_and_alias_contracts(
    device: torch.device,
) -> None:
    rows = 3
    hidden, block, _, injection, weight = _make_case(device, rows, 6703)
    hidden_backing = torch.empty(
        (rows, HC_WIDTH + 2), dtype=torch.float16, device=device
    )
    strided_hidden = hidden_backing[:, :HC_WIDTH]
    assert not strided_hidden.is_contiguous()
    _assert_rejected_without_writes(
        strided_hidden, block, injection, weight, "hidden_states must be contiguous"
    )

    bf16_injection = injection.to(torch.bfloat16)
    _assert_rejected_without_writes(
        hidden, block, bf16_injection, weight, "injection must have dtype float16"
    )

    combined = torch.full_like(hidden, 0.125)
    normed = combined
    snapshot = combined.cpu().clone()
    with pytest.raises(RuntimeError, match="must not share storage"):
        _run_strided(hidden, block, injection, weight, combined, normed)
    torch.xpu.synchronize()
    assert torch.equal(combined.cpu(), snapshot)
