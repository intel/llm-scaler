"""Correctness tests for large-M block-scaled MoE W8A16 GEMM."""

import time

import pytest
import torch

import custom_esimd_kernels_vllm as esimd


FP8_MAX = 448.0
BLOCK = 128
requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU is required"
)


def quantize_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    leading = weight.shape[:-2]
    n, k = weight.shape[-2:]
    flat = weight.reshape(-1, n, k)
    q = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (flat.shape[0], (n + BLOCK - 1) // BLOCK, (k + BLOCK - 1) // BLOCK),
        dtype=torch.float32,
        device=weight.device,
    )
    for e in range(flat.shape[0]):
        for nb in range(scale.shape[1]):
            for kb in range(scale.shape[2]):
                block = flat[
                    e,
                    nb * BLOCK : (nb + 1) * BLOCK,
                    kb * BLOCK : (kb + 1) * BLOCK,
                ]
                s = block.abs().max().clamp_min(1e-12) / FP8_MAX
                scale[e, nb, kb] = s
                q[
                    e,
                    nb * BLOCK : (nb + 1) * BLOCK,
                    kb * BLOCK : (kb + 1) * BLOCK,
                ] = (block / s).to(torch.float8_e4m3fn)
    return q.reshape(*leading, n, k), scale.reshape(
        *leading, scale.shape[-2], scale.shape[-1]
    )


def dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = q.shape[-2:]
    expanded = scale.repeat_interleave(BLOCK, -2).repeat_interleave(BLOCK, -1)
    return q.float() * expanded[..., :n, :k]


def check_all_e4m3_encodings() -> None:
    """Exercise every raw E4M3 code through dense, decode, and prefill paths."""
    raw = (
        torch.arange(256, dtype=torch.uint8)
        .view(1, 256, 1)
        .expand(1, 256, 128)
        .contiguous()
    )
    reference = raw[0, :, 0].view(torch.float8_e4m3fn).float() * 128
    finite = torch.isfinite(reference)
    weight = raw.to("xpu")
    scale = torch.ones((1, 2, 1), dtype=torch.float32, device="xpu")

    x = torch.ones((1, 128), dtype=torch.float16, device="xpu")
    dense_out = torch.empty((1, 256), dtype=torch.float16, device="xpu")
    esimd.esimd_gemm_fp8_blockscale(x, weight[0], scale[0], dense_out, 128, 128)

    outputs = [("dense", dense_out)]
    for total, label in ((1, "moe_decode"), (128, "moe_prefill")):
        moe_x = torch.ones((total, 128), dtype=torch.float16, device="xpu")
        moe_out = torch.empty((total, 256), dtype=torch.float16, device="xpu")
        offsets = torch.tensor([0, total], dtype=torch.int32, device="xpu")
        esimd.esimd_moe_gemm_fp8_blockscale(
            moe_x, weight, scale, moe_out, offsets, 256, 128, 1, 128, 128
        )
        outputs.append((label, moe_out))

    torch.xpu.synchronize()
    for label, output in outputs:
        actual = output[0].cpu().float()
        mismatches = torch.nonzero(finite & (actual != reference)).flatten()
        assert mismatches.numel() == 0, (
            label,
            [(int(i), actual[i].item(), reference[i].item()) for i in mismatches],
        )
        assert actual[0x80].item() == 0.0, (label, actual[0x80].item())
    print("all 254 finite E4M3 encodings: dense/decode/prefill exact")


def run_prefill_case(*, benchmark: bool) -> None:
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

    message = f"mean_rel={mean_rel:.6e} cosine={cosine:.8f}"
    if benchmark:
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
        message += f" mean_ms={mean_ms:.4f}"
    print(message)
    print("RESULT: ALL OK")


@requires_xpu
def test_all_e4m3_encodings() -> None:
    check_all_e4m3_encodings()


@requires_xpu
def test_fp8_blockscale_moe_prefill() -> None:
    run_prefill_case(benchmark=False)


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise SystemExit("XPU is required")
    check_all_e4m3_encodings()
    run_prefill_case(benchmark=True)
