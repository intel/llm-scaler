"""Exercise each production owner of the shared TopK V2 template."""

import importlib

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load_owners():
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    # PYTHONPATH can select an isolated, complete candidate package.
    for name in ("moe_ops", "moe_int4_ops", "moe_int4_prefill_ops", "esimd_topk_v2"):
        importlib.import_module(f"custom_esimd_kernels_vllm.{name}")


SHAPES = [(512, 10), (512, 8), (256, 10), (256, 8), (128, 8), (128, 10)]
CASES = (
    [(owner, e, k) for owner in ("fp8", "standalone") for e, k in SHAPES]
    + [("int4", e, k) for e, k in [(512, 10), (256, 8), (128, 8), (128, 10)]]
    + [("prefill", e, k) for e, k in [(256, 8), (128, 8), (128, 10)]]
)


def run_topk(owner, x, e, k):
    if owner == "int4":
        return torch.ops.moe_int4_ops.moe_topk_int4(x, k, e, True)
    if owner == "fp8":
        indices, weights = torch.ops.moe_ops.moe_topk(x, k, True)
        return weights, indices
    if owner == "prefill":
        return torch.ops.moe_int4_prefill_ops.moe_topk_softmax(x, k, e)
    weights = torch.empty((x.shape[0], k), dtype=torch.float16, device=x.device)
    indices = torch.empty((x.shape[0], k), dtype=torch.int32, device=x.device)
    torch.ops.esimd_topk_v2.topk_v2(x, weights, indices, x.shape[0], e, k)
    return weights, indices


@pytest.mark.parametrize("owner,e,k", CASES)
@pytest.mark.parametrize("rows", [1, 4, 64])
@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), -float("inf")])
def test_shared_topk_finite_and_nonfinite_rows(owner, e, k, rows, bad_value):
    generator = torch.Generator().manual_seed(901 + e + k)
    x = torch.stack(
        [(torch.randperm(e, generator=generator).float() / 32 - 8) for _ in range(rows)]
    ).half()
    if bad_value is not None:
        x[::2] = bad_value
    weights, indices = run_topk(owner, x.to("xpu"), e, k)
    weights, indices = weights.cpu(), indices.cpu()
    for row in range(rows):
        if bad_value is not None and row % 2 == 0:
            assert indices[row].tolist() == list(range(k))
            assert torch.isnan(weights[row]).all()
        else:
            expected_ids = torch.argsort(x[row].float(), descending=True)[:k]
            expected_weights = torch.softmax(x[row, expected_ids].float(), -1).half()
            torch.testing.assert_close(indices[row], expected_ids.int(), rtol=0, atol=0)
            torch.testing.assert_close(
                weights[row], expected_weights, rtol=2e-3, atol=2e-4
            )
