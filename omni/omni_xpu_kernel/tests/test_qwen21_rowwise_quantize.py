"""Compare the B70 K12288 SLM route with the unchanged generic row path."""

import pytest
import torch

import omni_xpu_kernel
from omni_xpu_kernel import int8


def _b70_native_or_skip():
    if not torch.xpu.is_available() or omni_xpu_kernel.__xpu_target__ != "bmg":
        pytest.skip("requires installed BMG XPU native extension")
    if torch.xpu.get_device_properties(0).device_id != 0xE223:
        pytest.skip("requires Arc Pro B70")
    native = int8._get_native()
    if native is None or not hasattr(native, "quantize_int8_rowwise_fused"):
        pytest.fail("B70 native rowwise quantization is unavailable")
    return native


@pytest.mark.parametrize("rows", (4032, 4096, 12188))
@pytest.mark.parametrize("distribution", ("normal", "zeros", "alternating_extreme"))
def test_k12288_matches_generic_subrows_byte_for_byte(rows, distribution):
    native = _b70_native_or_skip()
    columns = 12288
    if distribution == "normal":
        generator = torch.Generator(device="xpu").manual_seed(20260926 + rows)
        value = torch.randn((rows, columns), dtype=torch.bfloat16,
                            device="xpu", generator=generator)
    elif distribution == "zeros":
        value = torch.zeros((rows, columns), dtype=torch.bfloat16, device="xpu")
    else:
        pattern = torch.tensor(
            [-32768.0, -16.0, -1.0, -0.0, 0.0, 1.0, 16.0, 32768.0],
            dtype=torch.bfloat16, device="xpu")
        value = pattern.repeat(rows * (columns // pattern.numel())).reshape(rows, columns)

    candidate_q, candidate_scale = native.quantize_int8_rowwise_fused(value)
    assert candidate_q.shape == value.shape
    assert candidate_scale.shape == (rows, 1)
    assert candidate_q.dtype == torch.int8
    assert candidate_scale.dtype == torch.float32

    # Every control call has M < 4032, so it executes the pre-existing generic
    # route on the same row data. Rowwise outputs must be independent of the
    # other rows in the full candidate call.
    for start in range(0, rows, 4031):
        stop = min(start + 4031, rows)
        reference_q, reference_scale = native.quantize_int8_rowwise_fused(
            value[start:stop].contiguous())
        assert torch.equal(candidate_q[start:stop], reference_q)
        assert torch.equal(
            candidate_scale[start:stop].view(torch.int32),
            reference_scale.view(torch.int32))
