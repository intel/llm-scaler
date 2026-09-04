"""Correctness checks for the fused QSA Q norm, RoPE and selection ABI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="QSA fusion validation requires an XPU"
)


@pytest.fixture(scope="module")
def qsa_ops():
    package_dir = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "custom_esimd_kernels_vllm"
    )
    candidates = sorted(package_dir.glob("qsa_ops*.so"))
    if len(candidates) != 1:
        pytest.skip("focused QSA DSO is not built")
    spec = importlib.util.spec_from_file_location("qsa_ops", candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load QSA extension: {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_case(rows: int, mrope: bool):
    device = torch.device("xpu")
    generator = torch.Generator(device="cpu").manual_seed(7000 + rows)
    projected = torch.randn(
        rows, 4, 128, generator=generator, dtype=torch.float16
    ).to(device)
    weight = (torch.randn(128, generator=generator) * 0.2).to(
        device=device, dtype=torch.float16
    )
    query_positions = torch.full((rows,), 9214, dtype=torch.int64)
    if mrope:
        positions = torch.stack(
            (
                query_positions,
                query_positions + 1,
                query_positions + 2,
            )
        )
    else:
        positions = query_positions
    positions = positions.to(device)
    cache_cpu = torch.randn(
        36, 64, 1, 128, generator=generator, dtype=torch.float16
    )
    cache = cache_cpu.to(device)
    page_table = torch.arange(36, dtype=torch.int32).view(1, -1).to(device)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    sequence_lengths = torch.full(
        (1,), 9216, dtype=torch.int32, device=device
    )
    cos_sin_cpu = torch.empty((10000, 64), dtype=torch.float16)
    positions_cpu = torch.arange(10000, dtype=torch.float32).view(-1, 1)
    pairs = torch.arange(32, dtype=torch.float32).view(1, -1)
    angles = (positions_cpu + 1.0) * (pairs + 1.0) * 0.0007
    cos_sin_cpu[:, :32] = torch.cos(angles).to(torch.float16)
    cos_sin_cpu[:, 32:] = torch.sin(angles).to(torch.float16)
    return (
        projected,
        weight,
        positions,
        cos_sin_cpu.to(device),
        cache,
        page_table,
        token_to_req,
        query_positions.to(device),
        sequence_lengths,
        mrope,
    )


@pytest.mark.parametrize("rows", [1, 16, 32])
@pytest.mark.parametrize("mrope", [False, True])
def test_qsa_fusion_matches_two_step_chain(qsa_ops, rows: int, mrope: bool):
    (
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        mrope,
    ) = _make_case(rows, mrope)
    chain_q = torch.empty_like(projected)
    chain_out = torch.empty((rows, 2051), dtype=torch.int32, device="xpu")
    qsa_ops.qsa_indexer_norm_rope_v1(
        projected,
        chain_q,
        weight,
        positions,
        cos_sin_cache,
        mrope,
        True,
    )
    qsa_ops.qsa_select_paged_tokens_v2(
        chain_q,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        2048,
        4,
        64,
        chain_out,
    )

    fused_q = torch.empty_like(projected)
    fused_out = torch.empty_like(chain_out)
    returned = qsa_ops.qsa_q_norm_rope_select_v1(
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        fused_q,
        fused_out,
        mrope,
        True,
    )
    torch.xpu.synchronize()

    assert returned.data_ptr() == fused_out.data_ptr()
    for fused_row, chain_row in zip(fused_out, chain_out):
        fused_tokens = torch.sort(fused_row[fused_row >= 0]).values
        chain_tokens = torch.sort(chain_row[chain_row >= 0]).values
        assert torch.equal(fused_tokens, chain_tokens)
    assert torch.isfinite(fused_q).all()
    assert torch.isfinite(chain_q).all()
    assert (fused_q.float() - chain_q.float()).abs().max() <= 2.5e-4

    selection_only_out = torch.empty_like(chain_out)
    selection_only = qsa_ops.qsa_q_norm_rope_select_v1(
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        torch.empty(0, dtype=torch.float16, device="xpu"),
        selection_only_out,
        mrope,
        True,
    )
    torch.xpu.synchronize()
    assert selection_only.data_ptr() == selection_only_out.data_ptr()
    assert torch.equal(selection_only_out, fused_out)


def test_qsa_fusion_rejects_unproven_positions(qsa_ops):
    (
        projected,
        weight,
        positions,
        cos_sin_cache,
        compressed_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        mrope,
    ) = _make_case(1, False)
    with pytest.raises(RuntimeError, match="trusted position-bound proof"):
        qsa_ops.qsa_q_norm_rope_select_v1(
            projected,
            weight,
            positions,
            cos_sin_cache,
            compressed_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            torch.empty_like(projected),
            torch.empty((1, 2051), dtype=torch.int32, device="xpu"),
            mrope,
            False,
        )
