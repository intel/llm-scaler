"""Explicit batch-independent Q8 API must equal the released M=1 API exactly."""
import os
import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load():
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    import custom_esimd_kernels_sglang  # noqa: F401
    path = os.environ.get("IMAWG4_CACHE_GEMV_LIBRARY")
    if path:
        torch.ops.load_library(path)
    else:
        import custom_esimd_kernels_sglang.cache_gemv_ops  # noqa: F401


@pytest.mark.parametrize("shape", [(32, 2048), (32, 2112), (256, 2048), (2048, 2048), (513, 768)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 16])
def test_equal_to_single_row(shape, rows):
    N, K = shape
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(rows, K, generator=gen).half().to("xpu")
    q = torch.randint(-127, 128, (N, K), generator=gen, dtype=torch.int8).to("xpu")
    s = (torch.rand(N, K // 32, generator=gen) * .01).half().to("xpu")
    expected = torch.empty(rows, N, dtype=torch.float16, device="xpu")
    for row in range(rows):
        torch.ops.custom_esimd_kernels_sglang.esimd_gemv_q8_0(x[row:row+1], q, s, expected[row:row+1])
    actual = torch.empty_like(expected)
    torch.ops.custom_esimd_kernels_sglang.esimd_gemv_q8_0_consistent(x, q, s, actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
