import custom_esimd_kernels_vllm  # noqa: F401
import pytest
import torch


@pytest.mark.parametrize(
    ("batch", "width", "output_width"),
    [
        (1, 16, 2816),
        (3, 48, 2816),
        (8, 63, 2816),
        (1, 65, 2816),
        (3, 67, 2816),
        (8, 129, 2816),
        (1, 528, 2816),
        (3, 528, 2816),
        (2, 65, 8),
        (2, 528, 8),
        (3, 48, 8),
        (2, 128, 8),
    ],
)
def test_fp8_gemm_handles_short_and_gemma4_tp4_k_tails(
    batch, width, output_width
):
    torch.manual_seed(716)
    device = "xpu"
    x = (torch.randn(batch, width, device=device) * 0.2).half()
    weight = (torch.randn(output_width, width, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)
    actual = torch.empty(batch, output_width, dtype=torch.float16, device=device)

    torch.ops.custom_esimd_kernels_vllm.esimd_gemm_fp8_pert(
        x, weight, scale, actual
    )
    expected = x.float() @ weight.float().T

    torch.testing.assert_close(actual.float(), expected, atol=0.06, rtol=0.03)
