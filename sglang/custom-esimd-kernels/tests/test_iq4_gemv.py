import pytest
import torch

from custom_esimd_kernels_sglang import esimd_gemv_iq4, esimd_gemv_iq4_m


IQ4_LUT = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10,
     1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.float32,
)


def _case(m: int, n: int, k: int, output_slice: bool = False):
    assert k % 32 == 0
    row = torch.arange(n, dtype=torch.int64).view(n, 1)
    col = torch.arange(k, dtype=torch.int64).view(1, k)
    indices = ((row + col) % 16).to(torch.uint8)
    packed = (indices[:, 0::2] | (indices[:, 1::2] << 4)).contiguous()

    generator = torch.Generator().manual_seed(20260910 + m + n + k)
    scale = (
        torch.rand((n, k // 32), generator=generator, dtype=torch.float32)
        * 0.004
        - 0.002
    ).to(torch.float16)
    x = torch.randn((m, k), generator=generator, dtype=torch.float16) * 0.25
    dense = IQ4_LUT[indices.long()] * scale.float().repeat_interleave(32, dim=1)
    expected = (x.float() @ dense.t()).to(torch.float16)

    x_xpu = x.to("xpu")
    packed_xpu = packed.to("xpu")
    scale_xpu = scale.to("xpu")
    if output_slice:
        storage = torch.full((m, n + 11), -123.0, dtype=torch.float16, device="xpu")
        output = storage[:, 3 : 3 + n]
        assert not output.is_contiguous()
    else:
        storage = None
        output = torch.empty((m, n), dtype=torch.float16, device="xpu")

    if m == 1:
        esimd_gemv_iq4(x_xpu, packed_xpu, scale_xpu, output)
    else:
        esimd_gemv_iq4_m(x_xpu, packed_xpu, scale_xpu, output)
    torch.xpu.synchronize()
    actual = output.cpu()
    diff = (actual.float() - expected.float()).abs()
    assert diff.max().item() <= 0.01
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    if storage is not None:
        host_storage = storage.cpu()
        assert torch.all(host_storage[:, :3] == -123)
        assert torch.all(host_storage[:, 3 + n :] == -123)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("k", [256, 512, 3072])
def test_iq4_gemv_fixed_tiles(m, k):
    _case(m, n=37, k=k)


@pytest.mark.parametrize("m", [1, 3, 17])
def test_iq4_gemv_arbitrary_m_and_split_k(m):
    _case(m, n=19, k=96)


def test_iq4_gemv_noncontiguous_output_slice():
    _case(m=4, n=33, k=512, output_slice=True)


def test_iq4_gemv_qwen38_ssm_shape():
    _case(m=1, n=5120, k=3072)
