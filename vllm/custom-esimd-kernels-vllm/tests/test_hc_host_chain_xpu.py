"""Independent XPU tests for the HC M=1 host-batched chain.

The test intentionally imports only the canonical ESIMD package, or loads the
canonical main DSO named by ``HC_CHAIN_DSO``.  It does not import vLLM.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

CHAIN_SCHEMA = "custom_esimd_kernels_vllm::hc_combine_mix_m1_v1"
EPS = 1.0e-6
ROWS = 1
HC_WIDTH = 10240
HC_HIDDEN = 2560
HC_COUNT = 4
DOWN_WIDTH = 336
HC_RANK = 320


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    return torch.device("xpu:0")


@pytest.fixture(scope="module", autouse=True)
def _load_canonical_main(device: torch.device) -> None:
    configured = os.environ.get("HC_CHAIN_DSO")
    if configured:
        dso = Path(configured)
        if not dso.is_file():
            pytest.fail(f"HC_CHAIN_DSO does not exist: {dso}")
        torch.ops.load_library(str(dso))
    else:
        try:
            import custom_esimd_kernels_vllm  # noqa: F401
        except ImportError as exc:
            pytest.skip(
                f"canonical custom_esimd_kernels_vllm package is unavailable: {exc}"
            )

    if not torch._C._jit_get_schemas_for_operator(CHAIN_SCHEMA):
        pytest.fail(f"canonical main DSO did not register {CHAIN_SCHEMA}")


def _chain_op():
    return torch.ops.custom_esimd_kernels_vllm.hc_combine_mix_m1_v1


def _random_fp16(
    device: torch.device,
    shape: tuple[int, ...],
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    cpu_value = torch.randn(shape, generator=generator, dtype=torch.float32)
    return (cpu_value * scale).to(device=device, dtype=torch.float16).contiguous()


def _make_inputs(device: torch.device, *, rows: int = ROWS, seed: int = 3801):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (
        _random_fp16(device, (rows, HC_WIDTH), generator, 0.25),
        _random_fp16(device, (rows, HC_HIDDEN), generator, 0.25),
        _random_fp16(device, (rows, HC_COUNT), generator, 0.5),
        _random_fp16(device, (HC_WIDTH,), generator, 0.05),
        _random_fp16(device, (DOWN_WIDTH, HC_WIDTH), generator, 0.01),
        _random_fp16(device, (HC_WIDTH, HC_RANK), generator, 0.01),
    )


def _make_outputs(device: torch.device, *, rows: int = ROWS):
    shapes = (
        (rows, HC_WIDTH),
        (rows, HC_WIDTH),
        (rows, DOWN_WIDTH),
        (rows, HC_WIDTH),
        (rows, HC_HIDDEN),
    )
    sentinels = (0.125, -0.25, 0.375, -0.5, 0.625)
    return tuple(
        torch.full(shape, value, dtype=torch.float16, device=device)
        for shape, value in zip(shapes, sentinels)
    )


def _chain_args(inputs, outputs, eps: float = EPS):
    return (*inputs, *outputs, eps)


def _run_chain(inputs, outputs, eps: float = EPS):
    return _chain_op()(*_chain_args(inputs, outputs, eps))


def _run_four_existing_ops(inputs, outputs, eps: float = EPS) -> None:
    hidden, block, injection, norm_weight, down_weight, up_weight = inputs
    combined, normed, down, gate, mixed = outputs
    ops = torch.ops.custom_esimd_kernels_vllm

    assert (
        ops.hc_combine_norm_v1(
            hidden,
            block,
            injection,
            norm_weight,
            combined,
            normed,
            eps,
        )
        is None
    )
    assert ops.esimd_hc_down_fp16_out(normed, down_weight, down) is None
    assert (
        ops.esimd_gemv_fp16(down[:, :HC_RANK], up_weight, gate).data_ptr()
        == gate.data_ptr()
    )
    assert ops.hc_gate_mix_v1(normed, gate, mixed) is None


def _assert_exact(actual, expected) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        assert torch.equal(actual_tensor.cpu(), expected_tensor.cpu())


def _snapshot_outputs(outputs):
    torch.xpu.synchronize()
    return tuple(output.cpu().clone() for output in outputs)


def _assert_outputs_unchanged(outputs, snapshots) -> None:
    torch.xpu.synchronize()
    for output, snapshot in zip(outputs, snapshots):
        assert torch.equal(output.cpu(), snapshot)


def _assert_rejected_without_writes(inputs, outputs, eps: float = EPS) -> None:
    snapshots = _snapshot_outputs(outputs)
    with pytest.raises(RuntimeError):
        _run_chain(inputs, outputs, eps)
    _assert_outputs_unchanged(outputs, snapshots)


def test_chain_matches_the_four_existing_ops_bitwise(device: torch.device) -> None:
    inputs = _make_inputs(device)
    chain_outputs = _make_outputs(device)
    reference_outputs = _make_outputs(device)

    assert _run_chain(inputs, chain_outputs) is None
    _run_four_existing_ops(inputs, reference_outputs)
    torch.xpu.synchronize()

    _assert_exact(chain_outputs, reference_outputs)
    assert chain_outputs[2][:, 320:324].shape == (1, 4)


def test_rejects_bad_dtype_before_writing_outputs(device: torch.device) -> None:
    inputs = list(_make_inputs(device))
    inputs[5] = inputs[5].float()
    _assert_rejected_without_writes(tuple(inputs), _make_outputs(device))


def test_rejects_m_gt_1_before_writing_outputs(device: torch.device) -> None:
    inputs = _make_inputs(device, rows=2, seed=3802)
    outputs = _make_outputs(device, rows=2)
    _assert_rejected_without_writes(inputs, outputs)


def test_rejects_late_output_before_writing_outputs(device: torch.device) -> None:
    inputs = _make_inputs(device, seed=3803)
    outputs = list(_make_outputs(device))
    outputs[4] = torch.full(
        (1, HC_HIDDEN - 1),
        0.875,
        dtype=torch.float16,
        device=device,
    )
    _assert_rejected_without_writes(inputs, tuple(outputs))


def test_rejects_output_input_and_output_output_aliases_without_writes(
    device: torch.device,
) -> None:
    inputs = _make_inputs(device, seed=3804)

    output_input_alias = list(_make_outputs(device))
    output_input_alias[0] = inputs[0]
    _assert_rejected_without_writes(inputs, tuple(output_input_alias))

    shared = torch.full(
        (1, 2 * HC_WIDTH),
        0.9375,
        dtype=torch.float16,
        device=device,
    )
    output_output_alias = list(_make_outputs(device))
    output_output_alias[0] = shared[:, :HC_WIDTH]
    output_output_alias[1] = shared[:, HC_WIDTH:]
    _assert_rejected_without_writes(inputs, tuple(output_output_alias))


@pytest.mark.parametrize(
    "bad_eps",
    (0.0, -1.0e-6, 1.0e-300, float("nan"), float("inf")),
)
def test_rejects_invalid_eps_before_writing_outputs(
    device: torch.device,
    bad_eps: float,
) -> None:
    inputs = _make_inputs(device, seed=3805)
    _assert_rejected_without_writes(inputs, _make_outputs(device), bad_eps)


def _run_repeated_on_stream(
    inputs,
    device: torch.device,
    stream: torch.xpu.Stream,
) -> None:
    with torch.xpu.stream(stream):
        actual = _make_outputs(device)
        expected = _make_outputs(device)
        for _ in range(8):
            assert _run_chain(inputs, actual) is None
        _run_four_existing_ops(inputs, expected)

    stream.synchronize()
    _assert_exact(actual, expected)


def test_repeated_submissions_on_current_and_nondefault_stream(
    device: torch.device,
) -> None:
    inputs = _make_inputs(device, seed=3806)
    torch.xpu.synchronize()

    current_stream = torch.xpu.current_stream(device=device)
    nondefault_stream = torch.xpu.Stream(device=device)
    _run_repeated_on_stream(inputs, device, current_stream)
    _run_repeated_on_stream(inputs, device, nondefault_stream)
