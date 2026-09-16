"""TP4 M1 widened loads and unchanged fused2 fallbacks, with async outputs."""
import pytest
import torch
from custom_esimd_kernels_vllm import esimd_gemv_int4_fused2


@pytest.mark.parametrize("n0,n1,k,scale_offsets", [
    (4096, 24, 2560, (0, 0)), (2048, 12, 2560, (0, 0)),
    (3072, 16, 2560, (0, 0)), (4096, 24, 1536, (0, 0)),
    (4096, 24, 2560, (1, 0)), (4096, 24, 2560, (0, 1)),
    (4096, 24, 2560, (1, 1)),
])
def test_signed_scales_golden_and_independent_streams(n0, n1, k, scale_offsets):
    generator = torch.Generator().manual_seed(3916)
    x_cpu = (torch.randn(1, k, generator=generator) * 0.25).half()
    matrices, references = [], []
    for n, scale_offset in zip((n0, n1), scale_offsets):
        packed = torch.randint(256, (n, k // 2), generator=generator, dtype=torch.uint8)
        scales = (torch.rand(n, k // 128, generator=generator) * 0.04 - 0.02).half()
        scales[:, ::3] = 0
        q = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).float() - 8
        w = q * scales.float().repeat_interleave(128, dim=1)
        references.append((x_cpu.float() @ w.T).half())
        backing = torch.empty(scales.numel() + scale_offset,
                              dtype=torch.float16, device="xpu")
        device_scales = backing[scale_offset:].view_as(scales)
        device_scales.copy_(scales)
        assert device_scales.is_contiguous()
        assert device_scales.data_ptr() % 4 == scale_offset * 2
        matrices.append((packed.to("xpu"), device_scales))
    x = x_cpu.to("xpu")
    ready = torch.xpu.Event()
    ready.record()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_event(ready)
        with torch.xpu.stream(stream):
            for _ in range(8):
                a = torch.empty(1, n0, device="xpu", dtype=torch.float16)
                b = torch.empty(1, n1, device="xpu", dtype=torch.float16)
                esimd_gemv_int4_fused2(x, *matrices[0], a, *matrices[1], b)
                outputs.append((a + 0, b + 0))
    for stream in streams:
        stream.synchronize()
    baseline = tuple(t.cpu() for t in outputs[0])
    for pair in outputs:
        for actual, expected, first in zip(pair, references, baseline):
            actual = actual.cpu()
            assert torch.isfinite(actual).all()
            assert torch.equal(actual, first)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.003)
