"""Metadata-only MoE helpers live in the main DSO, not moe_int4_ops.

Use MOE_METADATA_DSO for build-only testing without importing vLLM.
"""

import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load_main():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if dso := os.environ.get("MOE_METADATA_DSO"):
        torch.ops.load_library(dso)
    else:
        import custom_esimd_kernels_vllm  # noqa: F401
    if not torch._C._jit_get_schemas_for_operator(
        "custom_esimd_kernels_vllm::qwen38_moe_weight_contract_v1"
    ):
        pytest.skip("canonical main DSO lacks optional MoE metadata helpers")


def test_weight_contract_count_and_device():
    op = torch.ops.custom_esimd_kernels_vllm.qwen38_moe_weight_contract_v1
    with pytest.raises(RuntimeError, match="seven weights"):
        op([], torch.device("xpu:0"))
    cpu = torch.empty(1)
    assert op([cpu] * 7, torch.device("xpu:0")) == 1
    assert op([cpu] * 7, torch.device("cpu")) == 2
    assert op([torch.empty(1, dtype=torch.uint8)] * 7, torch.device("cpu")) == 3


@pytest.mark.parametrize("device", ["cpu", "xpu:0"])
def test_overlap_checks_current_byte_ranges_without_writes(device):
    op = torch.ops.custom_esimd_kernels_vllm.qwen38_moe_output_overlaps_v1
    base = torch.arange(64, dtype=torch.float16, device=device)
    output = base[8:16]
    assert op(output, [base[12:20]])
    # Unlike ATen storage aliasing, the caller allows disjoint storage views.
    assert not op(output, [base[16:24]])
    assert not op(output, [base[:8]])
    assert not op(output, [base[:0]])
    assert not op(output, [])
    assert op(output, [base.view(torch.uint8)[17:20]])
    output.set_(base.untyped_storage(), 32, (8,))
    assert not op(output, [base[12:20]])
    assert op(output, [base[36:44]])
    torch.testing.assert_close(
        base.cpu(), torch.arange(64, dtype=torch.float16), rtol=0, atol=0
    )
