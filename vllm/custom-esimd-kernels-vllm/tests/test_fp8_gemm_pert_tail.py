import custom_esimd_kernels_vllm  # noqa: F401
import pytest
import torch


@pytest.mark.parametrize("batch", [1, 3])
def test_fp8_gemm_handles_gemma4_tp4_down_projection(batch):
    torch.manual_seed(716)
    device = "xpu"
    x = (torch.randn(batch, 528, device=device) * 0.2).half()
    weight = (torch.randn(2816, 528, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)
    actual = torch.empty(batch, 2816, dtype=torch.float16, device=device)

    torch.ops.custom_esimd_kernels_vllm.esimd_gemm_fp8_pert(
        x, weight, scale, actual
    )
    expected = x.float() @ weight.float().T

    torch.testing.assert_close(actual.float(), expected, atol=0.06, rtol=0.03)
