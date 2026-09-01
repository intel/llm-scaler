import os
from pathlib import Path

import pytest
import torch

NUM_EXPERTS = 512
TOP_K = 10
TOKEN_COUNTS = (1, 4, 64)


def _dso_path() -> Path:
    configured = os.environ.get("MOE_INT4_DSO")
    if configured:
        return Path(configured)
    package_dir = Path(__file__).parents[1] / "python" / "custom_esimd_kernels_vllm"
    matches = tuple(package_dir.glob("moe_int4_ops*.so"))
    if len(matches) != 1:
        raise RuntimeError(
            "set MOE_INT4_DSO or build exactly one focused moe_int4_ops DSO"
        )
    return matches[0]


@pytest.fixture(scope="module", autouse=True)
def _load_focused_dso() -> None:
    dso = _dso_path()
    if not dso.is_file():
        raise RuntimeError(f"focused DSO does not exist: {dso}")
    torch.ops.load_library(str(dso))


def _unique_logits(rows: int) -> torch.Tensor:
    levels = torch.arange(NUM_EXPERTS, dtype=torch.float32) / 32.0 - 8.0
    result = []
    for row in range(rows):
        generator = torch.Generator().manual_seed(20260901 + row)
        result.append(levels[torch.randperm(NUM_EXPERTS, generator=generator)])
    return torch.stack(result).half()


def _edge_logits(case: str, rows: int) -> torch.Tensor:
    if case == "all_tie":
        return torch.zeros(rows, NUM_EXPERTS, dtype=torch.float16)
    if case == "partial_tie":
        logits = torch.full((rows, NUM_EXPERTS), -8.0, dtype=torch.float16)
        tied = (0, 7, 63, 64, 65, 127, 128, 191, 255, 256, 383, 511)
        logits[:, tied] = 4.0
        return logits
    if case == "finite_extremes":
        logits = torch.full((rows, NUM_EXPERTS), -80.0, dtype=torch.float16)
        logits[:, 511] = 80.0
        logits[:, 257] = 40.0
        logits[:, 128] = 20.0
        logits[:, 64] = 0.0
        return logits
    raise AssertionError(f"unknown case: {case}")


def _reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    indices = torch.argsort(
        probabilities, dim=-1, descending=True, stable=True
    )[:, :TOP_K]
    weights = probabilities.gather(1, indices)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).half()
    return weights, indices.to(torch.int32)


def _assert_matches_reference(logits: torch.Tensor) -> None:
    reference_weight, reference_idx = _reference(logits)
    native_weight, native_idx = torch.ops.moe_int4_ops.moe_topk_int4(
        logits.to("xpu"), TOP_K, NUM_EXPERTS, True
    )
    native_weight = native_weight.cpu()
    native_idx = native_idx.cpu()

    assert native_weight.shape == reference_weight.shape
    assert native_idx.shape == reference_idx.shape
    assert native_weight.dtype == torch.float16
    assert native_idx.dtype == torch.int32
    assert torch.all((native_idx >= 0) & (native_idx < NUM_EXPERTS))
    for row in range(logits.shape[0]):
        assert torch.unique(native_idx[row]).numel() == TOP_K
    torch.testing.assert_close(
        native_idx.sort(dim=-1).values,
        reference_idx.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )

    native_by_expert = torch.zeros(
        logits.shape[0], NUM_EXPERTS, dtype=torch.float32
    )
    reference_by_expert = torch.zeros_like(native_by_expert)
    native_by_expert.scatter_(1, native_idx.long(), native_weight.float())
    reference_by_expert.scatter_(
        1, reference_idx.long(), reference_weight.float()
    )

    torch.testing.assert_close(
        native_by_expert != 0,
        reference_by_expert != 0,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        native_by_expert,
        reference_by_expert,
        rtol=2e-3,
        atol=2e-4,
    )
    torch.testing.assert_close(
        native_weight.float().sum(dim=-1),
        torch.ones(logits.shape[0]),
        rtol=0,
        atol=5e-4,
    )


@pytest.mark.parametrize("rows", TOKEN_COUNTS)
def test_e512_k10_unique_logits(rows: int) -> None:
    _assert_matches_reference(_unique_logits(rows))


@pytest.mark.parametrize("rows", TOKEN_COUNTS)
@pytest.mark.parametrize("case", ("all_tie", "partial_tie", "finite_extremes"))
def test_e512_k10_edge_logits(case: str, rows: int) -> None:
    _assert_matches_reference(_edge_logits(case, rows))
