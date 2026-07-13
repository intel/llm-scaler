"""Correctness test for the large-M block-scaled MoE W8A16 GEMM."""

import time

import torch

import custom_esimd_kernels_vllm as esimd
from test_moe_forward_full_fp8_block import dequant, quantize_block


def main() -> None:
    torch.manual_seed(20260713)
    device = "xpu"
    experts, n, k = 4, 256, 256
    counts = torch.tensor([33, 40, 27, 36], dtype=torch.int32, device=device)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0)
    total = int(counts.sum().cpu())

    x = (torch.randn(total, k, device=device) * 0.2).half()
    weight, scale = quantize_block(
        torch.randn(experts, n, k, device=device) * 0.15
    )
    out = torch.empty(total, n, dtype=torch.float16, device=device)

    esimd.esimd_moe_gemm_fp8_blockscale(
        x, weight, scale, out, offsets, n, k, experts
    )
    torch.xpu.synchronize()

    ref = torch.empty(total, n, dtype=torch.float32, device=device)
    dw = dequant(weight, scale)
    start = 0
    for expert, count in enumerate(counts.cpu().tolist()):
        ref[start : start + count] = x[start : start + count].float() @ dw[expert].t()
        start += count

    out_f = out.float()
    mean_rel = ((out_f - ref).abs().mean() / ref.abs().mean()).item()
    cosine = torch.nn.functional.cosine_similarity(
        out_f.flatten(), ref.flatten(), dim=0
    ).item()
    assert mean_rel < 3e-3, mean_rel
    assert cosine > 0.9999, cosine

    for _ in range(10):
        esimd.esimd_moe_gemm_fp8_blockscale(
            x, weight, scale, out, offsets, n, k, experts
        )
    torch.xpu.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        esimd.esimd_moe_gemm_fp8_blockscale(
            x, weight, scale, out, offsets, n, k, experts
        )
    torch.xpu.synchronize()
    mean_ms = (time.perf_counter() - begin) * 10.0
    print(f"mean_rel={mean_rel:.6e} cosine={cosine:.8f} mean_ms={mean_ms:.4f}")
    print("RESULT: ALL OK")


if __name__ == "__main__":
    main()
