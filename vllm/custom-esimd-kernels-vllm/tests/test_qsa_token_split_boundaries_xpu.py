"""QCR001/002: masked non-finites and opt-in fixed-DSO large addresses.

The >4 GiB tests MUST NOT run against the old DSO. They require both an
explicit QSA_TEST_DSO and its independently verified post-fix SHA256 in
QSA_TEST_LARGE_ADDRESS_FIXED_SHA256. Ordinary runs skip those five cases.
Collection and the source/address contract test do not initialize XPU.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch


HD = 256
WIDTH = 2051
CONFIGS = [
    pytest.param(256, 3, False, id="page256-q3-v3"),
    pytest.param(512, 3, False, id="page512-q3-v3"),
    pytest.param(256, 6, False, id="page256-q6-v3"),
    pytest.param(512, 6, False, id="page512-q6-v3"),
    pytest.param(256, 3, True, id="page256-q3-v4"),
]


@pytest.fixture(scope="module")
def qsa_ops():
    if not torch.xpu.is_available():
        pytest.skip("QSA kernel boundaries require XPU")
    # Reuse the existing explicit-DSO/default-package loader, but import it
    # only at GPU-test execution so CPU collection never probes a device.
    from test_qsa_sparse_attention_xpu import _load_qsa_extension

    return _load_qsa_extension()


def _launch(ops, q, packed, logical, table, requests, out, partials, fused):
    page_size = packed.shape[2]
    args = (q, packed, logical, table, requests, page_size, out, partials)
    if fused:
        ready = torch.zeros((q.shape[0], 3), dtype=torch.int32, device=q.device)
        returned = ops.sparse_attention_token_split_candidate_v4(*args, ready, 0)
    elif q.shape[1] == 6:
        returned = ops.sparse_attention_token_split_candidate_q6_v1(*args)
    else:
        returned = ops.sparse_attention_token_split_candidate_v3(*args)
    assert returned.data_ptr() == out.data_ptr()
    # Observe only after the whole producer/merge invocation has been queued.
    torch.xpu.synchronize()
    return out.cpu()


def _reference_cpu(q, packed, logical, table, requests):
    """Independent FP32 softmax over selected tokens, never safe-row data."""
    expected = torch.zeros_like(q)
    page_size = packed.shape[2]
    for row in range(q.shape[0]):
        selected = logical[row][logical[row] >= 0].long()
        if selected.numel() == 0:
            continue
        pages = table[int(requests[row]), selected // page_size].long()
        kv = packed[pages, 0, selected % page_size].float()
        logits = q[row].float() @ kv[:, :HD].T / 16.0
        expected[row] = (logits.softmax(-1) @ kv[:, HD:]).to(q.dtype)
    return expected


@pytest.mark.parametrize("page_size,heads,fused", CONFIGS)
@pytest.mark.parametrize("rows", [1, 2, 7])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")],
                         ids=["nan", "posinf", "neginf"])
@pytest.mark.parametrize("case", ["all_holes", "mixed_rows", "tail_only"])
def test_unselected_nonfinite_values_are_masked(
    qsa_ops, page_size, heads, fused, rows, bad, case
):
    generator = torch.Generator().manual_seed(20260913)
    q_cpu = (torch.randn(rows, heads, HD, generator=generator) * 0.2).half()
    packed_cpu = (torch.randn(2, 1, page_size, 2 * HD,
                              generator=generator) * 0.2).half()
    # Every request's hole/padded-slot safe address is token zero. It is not
    # selected by any live slot; no finite-data assumption may leak from it.
    packed_cpu[:, 0, 0, HD:] = bad
    table_cpu = torch.tensor([[0], [1]], dtype=torch.int32)
    requests_cpu = torch.arange(rows, dtype=torch.int32) % 2
    logical_cpu = torch.full((rows, WIDTH), -7, dtype=torch.int32)
    if case == "tail_only":
        logical_cpu[:, -1] = 4
    elif case == "mixed_rows":
        for row in range(rows):
            # Include both all-hole and live rows in the SAME M>1 launch.
            if rows > 1 and row == rows - 1:
                continue
            logical_cpu[row, [0, 47, 48, 63, 64, 2048, 2050]] = torch.tensor(
                [1, 2, 1, 3, 4, 2, 4], dtype=torch.int32
            )
    expected = _reference_cpu(
        q_cpu, packed_cpu, logical_cpu, table_cpu, requests_cpu
    )
    q, packed, logical, table, requests = [
        value.to("xpu") for value in
        (q_cpu, packed_cpu, logical_cpu, table_cpu, requests_cpu)
    ]
    storage = torch.full((rows + 2, heads, 43, 258), float("nan"),
                         dtype=torch.float32, device="xpu")
    partials = storage[1:-1]
    output_storage = torch.full((rows + 2, heads, HD), 123.0,
                                dtype=torch.float16, device="xpu")
    out = output_storage[1:-1]
    actual = _launch(qsa_ops, q, packed, logical, table, requests,
                     out, partials, fused)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    empty_rows = (logical_cpu < 0).all(dim=1)
    assert torch.count_nonzero(actual[empty_rows]) == 0
    assert torch.isnan(storage[0]).all() and torch.isnan(storage[-1]).all()
    active = 43 if 2 <= rows <= 6 else 33
    assert torch.isfinite(partials[:, :, :active]).all()
    assert torch.isnan(partials[:, :, active:]).all()
    assert (output_storage[0] == 123).all() and (output_storage[-1] == 123).all()
    torch.testing.assert_close(packed.cpu(), packed_cpu, equal_nan=True,
                               atol=0, rtol=0)


@pytest.mark.parametrize("page_size,heads,fused", CONFIGS)
@pytest.mark.parametrize("rows", [1, 2, 7])
def test_selected_nan_values_still_propagate(qsa_ops, page_size, heads, fused, rows):
    q = torch.zeros((rows, heads, HD), dtype=torch.float16, device="xpu")
    packed = torch.zeros((1, 1, page_size, 2 * HD), dtype=q.dtype, device=q.device)
    packed[0, 0, 1, HD:] = float("nan")
    logical = torch.full((rows, WIDTH), -1, dtype=torch.int32, device=q.device)
    logical[:, 0] = 1
    table = torch.zeros((1, 1), dtype=torch.int32, device=q.device)
    requests = torch.zeros(rows, dtype=torch.int32, device=q.device)
    partials = torch.empty((rows, heads, 43, 258), dtype=torch.float32,
                           device=q.device)
    actual = _launch(qsa_ops, q, packed, logical, table, requests,
                     torch.empty_like(q), partials, fused)
    assert torch.isnan(actual).all(), "Do not sanitize a valid selected V"


@pytest.mark.parametrize("page_size", [256, 512])
def test_page_address_source_contract(page_size):
    source = (Path(__file__).resolve().parents[1] / "csrc/qsa/"
              "qsa_token_split_attention.sycl").read_text()
    # Check all three consumers: phase0, optional prefetch, optional v4.
    assert source.count("static_cast<int64_t>(safe_physical_page) *") == 3
    assert "packed_kv + safe_physical_page *" not in source
    first = (1 << 31) // (page_size * 512)
    assert (first - 1) * page_size * 512 < (1 << 31)
    assert first * page_size * 512 == (1 << 31)
    assert first * page_size * 512 * 2 == (1 << 32)


@pytest.mark.parametrize("page_size,heads,fused", CONFIGS)
def test_large_physical_page_fixed_dso_only(qsa_ops, page_size, heads, fused):
    fixed_hash = os.environ.get("QSA_TEST_LARGE_ADDRESS_FIXED_SHA256")
    dso_path = os.environ.get("QSA_TEST_DSO")
    if not fixed_hash or not dso_path:
        pytest.skip("Dangerous on old DSO: requires explicit verified fixed DSO + SHA256")
    # Never infer safety from a patched source tree or the default package.
    loaded_path = Path(qsa_ops.__file__).resolve()
    assert loaded_path == Path(dso_path).resolve()
    with loaded_path.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual_hash == fixed_hash.lower(), "Fixed-DSO approval hash mismatch"

    high_page = (1 << 31) // (page_size * 512)
    try:
        packed = torch.empty((high_page + 1, 1, page_size, 2 * HD),
                             dtype=torch.float16, device="xpu")
    except torch.OutOfMemoryError:
        pytest.skip("Needs a contiguous >4 GiB packed KV allocation on the selected GPU")
    # Only addressed pages need initialization; never touch the huge allocation
    # with a full-tensor random fill, clone, or CPU copy.
    packed[0].zero_()
    packed[0, 0, :, HD:] = -9.0
    packed[high_page - 1].zero_()
    packed[high_page - 1, 0, :, HD:] = 1.0
    packed[high_page].zero_()
    packed[high_page, 0, :, HD:] = 2.0
    q = torch.zeros((1, heads, HD), dtype=packed.dtype, device=packed.device)
    table = torch.tensor([[high_page - 1, high_page]], dtype=torch.int32,
                         device=q.device)
    requests = torch.zeros(1, dtype=torch.int32, device=q.device)
    logical = torch.full((1, WIDTH), -1, dtype=torch.int32, device=q.device)
    logical[0, [0, 64, 2050]] = torch.tensor(
        [0, page_size, 2 * page_size - 1], dtype=torch.int32, device=q.device
    )
    partials = torch.empty((1, heads, 43, 258), dtype=torch.float32, device=q.device)
    actual = _launch(qsa_ops, q, packed, logical, table, requests,
                     torch.empty_like(q), partials, fused)
    # K=Q=0 => uniform weights over V={1,2,2}; independent exact FP32 reference.
    expected = torch.full((1, heads, HD), 5.0 / 3.0, dtype=torch.float16)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
