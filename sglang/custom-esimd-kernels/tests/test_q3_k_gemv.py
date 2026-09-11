import pytest
import torch

from custom_esimd_kernels_sglang import esimd_gemv_q3_k, esimd_gemv_q3_k_m


def _case(m: int, n: int, k: int, output_slice: bool = False):
    assert k % 256 == 0
    row = torch.arange(n, dtype=torch.int64).view(n, 1)
    col = torch.arange(k, dtype=torch.int64).view(1, k)
    low = ((row + col) % 4).to(torch.uint8)
    subtract = (((row * 3 + col) % 5) == 0).to(torch.uint8)
    low4 = low.reshape(n, -1, 4)
    ql = (
        low4[:, :, 0]
        | (low4[:, :, 1] << 2)
        | (low4[:, :, 2] << 4)
        | (low4[:, :, 3] << 6)
    ).contiguous()
    sub8 = subtract.reshape(n, -1, 8)
    qh = (
        sub8 << torch.arange(8, dtype=torch.uint8).view(1, 1, 8)
    ).sum(dim=2).to(torch.uint8).contiguous()

    generator = torch.Generator().manual_seed(20260910 + m + n + k)
    scale = (
        torch.rand((n, k // 16), generator=generator) * 0.008 - 0.004
    ).to(torch.float16)
    x = torch.randn((m, k), generator=generator, dtype=torch.float16) * 0.25
    dense = (
        (low.float() - 4.0 * subtract.float()).view(n, k // 16, 16)
        * scale.float().unsqueeze(-1)
    ).view(n, k)
    expected = (x.float() @ dense.t()).to(torch.float16)

    x_xpu = x.to("xpu")
    ql_xpu = ql.to("xpu")
    qh_xpu = qh.to("xpu")
    scale_xpu = scale.to("xpu")
    if output_slice:
        storage = torch.full(
            (m, n + 11), -123.0, dtype=torch.float16, device="xpu"
        )
        output = storage[:, 3 : 3 + n]
        assert not output.is_contiguous()
    else:
        storage = None
        output = torch.empty((m, n), dtype=torch.float16, device="xpu")

    if m == 1:
        esimd_gemv_q3_k(x_xpu, ql_xpu, qh_xpu, scale_xpu, output)
    else:
        esimd_gemv_q3_k_m(x_xpu, ql_xpu, qh_xpu, scale_xpu, output)
    torch.xpu.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    if storage is not None:
        host_storage = storage.cpu()
        assert torch.all(host_storage[:, :3] == -123)
        assert torch.all(host_storage[:, 3 + n :] == -123)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("k", [256, 512, 5120])
def test_q3_k_gemv_fixed_tiles(m, k):
    _case(m, n=37, k=k)


@pytest.mark.parametrize("m", [3, 17])
def test_q3_k_gemv_arbitrary_m(m):
    _case(m, n=19, k=512)


def test_q3_k_gemv_noncontiguous_output_slice():
    _case(m=4, n=33, k=512, output_slice=True)
