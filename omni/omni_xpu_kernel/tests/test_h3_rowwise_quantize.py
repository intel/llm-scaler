"""Focused coverage for MiniMax H3 BMG rowwise quantization routes."""

from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INT8_SOURCE = (
    PROJECT_ROOT / "omni_xpu_kernel" / "csrc" / "int8_quantize_esimd.cpp"
).read_text(encoding="utf-8")


def _bmg_native_or_skip():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("native H3 rowwise quantization requires XPU")

    import omni_xpu_kernel
    from omni_xpu_kernel import int8

    if omni_xpu_kernel.__xpu_target__ != "bmg":
        pytest.skip("H3 rowwise specialization is a BMG route")
    native = int8._get_native()
    if native is None or not hasattr(native, "quantize_int8_rowwise_fused"):
        pytest.skip("native extension lacks fused rowwise quantization")
    return native


def test_h3_dispatch_is_sequence_structural():
    assert "RowwiseQuantizeH3HiddenBMGConfig" in INT8_SOURCE
    assert "RowwiseQuantizeH3FFNDownBMGConfig" in INT8_SOURCE
    assert "static constexpr int MinimumRows = 4096;" in INT8_SOURCE
    assert (
        "M >= RowwiseQuantizeH3HiddenBMGConfig::MinimumRows"
        in INT8_SOURCE
    )
    assert (
        "M >= RowwiseQuantizeH3FFNDownBMGConfig::MinimumRows"
        in INT8_SOURCE
    )
    assert "M == 34842" not in INT8_SOURCE
    assert "M == 44929" not in INT8_SOURCE


@pytest.mark.parametrize("columns", [5376, 7168])
@pytest.mark.parametrize("input_kind", ["random", "alternating_extremes"])
def test_h3_threshold_shapes_match_rowwise_reference(columns, input_kind):
    native = _bmg_native_or_skip()
    rows = 4096
    if input_kind == "random":
        torch.xpu.manual_seed_all(20260805 + columns)
        value = torch.randn(
            rows, columns, device="xpu", dtype=torch.bfloat16
        )
    else:
        value = torch.empty(
            rows, columns, device="xpu", dtype=torch.bfloat16
        )
        value[:, 0::2] = 32768.0
        value[:, 1::2] = -32768.0

    actual_q, actual_scale = native.quantize_int8_rowwise_fused(value)
    expected_scale = (
        value.float().abs().amax(dim=-1, keepdim=True) / 127.0
    ).clamp(min=1e-30)
    expected_q = (
        torch.round(value.float() / expected_scale)
        .clamp(-128, 127)
        .to(torch.int8)
    )

    torch.testing.assert_close(
        actual_scale, expected_scale, rtol=1e-6, atol=1e-8
    )
    max_quant_diff = (
        actual_q.to(torch.int16) - expected_q.to(torch.int16)
    ).abs().max()
    assert max_quant_diff.item() <= 1
