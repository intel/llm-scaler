"""Fixed fused-MoE geometry must match the original single-row API bitwise."""
import os
import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load():
    if not torch.xpu.is_available():
        pytest.skip("requires Intel XPU")
    import custom_esimd_kernels_sglang  # noqa: F401
    path = os.environ.get("IMAWG4_CACHE_GEMV_LIBRARY")
    if path:
        torch.ops.load_library(path)
    else:
        import custom_esimd_kernels_sglang.cache_gemv_ops  # noqa: F401


def fixture_args(rows, hidden, intermediate, shared):
    generator = torch.Generator().manual_seed(73)
    experts, top_k = 4, 2

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).half().to("xpu")

    def byte(shape):
        return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8).to("xpu")

    def quant(shape):
        return torch.randint(-127, 128, shape, generator=generator, dtype=torch.int8).to("xpu")

    return [rand((rows, hidden)), rand((rows, hidden)), rand((hidden,), .1) + 1,
            1e-6, rand((experts, hidden), .01),
            byte((experts, intermediate, hidden // 2)), rand((experts, intermediate, hidden // 32), .001),
            rand((experts, intermediate, hidden // 32), .01),
            byte((experts, intermediate, hidden // 2)), rand((experts, intermediate, hidden // 32), .001),
            rand((experts, intermediate, hidden // 32), .01),
            byte((experts, hidden, intermediate // 2)), byte((experts, hidden, intermediate // 8)),
            rand((experts, hidden, intermediate // 32), .001), rand((experts, hidden, intermediate // 32), .01),
            quant((2 * shared, hidden)), rand((2 * shared, hidden // 32), .001),
            quant((hidden, shared)), rand((hidden, shared // 32), .001), rand((hidden,), .01),
            experts, top_k, intermediate, shared, False, True]


@pytest.mark.parametrize("hidden,intermediate,shared", [(1024, 128, 128), (2048, 256, 256), (4096, 256, 256)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 16])
def test_fused_moe_matches_released_single_row(hidden, intermediate, shared, rows):
    args = fixture_args(rows, hidden, intermediate, shared)
    expected = []
    for row in range(rows):
        single = [args[0][row:row+1], args[1][row:row+1], *args[2:]]
        expected.append(torch.ops.custom_esimd_kernels_sglang.esimd_moe_forward_full_gguf_norm(*single))
    actual = torch.ops.custom_esimd_kernels_sglang.esimd_moe_forward_full_gguf_norm_consistent(*args)
    for index in range(2):
        torch.testing.assert_close(actual[index], torch.cat([row[index] for row in expected]), rtol=0, atol=0)
