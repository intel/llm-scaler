"""Correctness test for the batch-1 FP8-block fused MoE decode op."""

import torch

import custom_esimd_kernels_vllm as esimd


FP8_MAX = 448.0
BLOCK = 128


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


def reference(
    x,
    logits,
    w13,
    s13,
    shared_w13,
    shared_s13,
    w2,
    s2,
    shared_w2,
    shared_s2,
    shared_gate,
    top_k,
):
    scores = torch.softmax(logits.float(), dim=-1)
    top_weights, top_ids = torch.topk(scores, top_k, dim=-1)
    top_weights /= top_weights.sum(dim=-1, keepdim=True)
    x32 = x.float()
    out = torch.zeros_like(x32)
    dw13 = dequant(w13, s13)
    dw2 = dequant(w2, s2)
    for route in range(top_k):
        eid = int(top_ids[0, route])
        gate, up = (x32 @ dw13[eid].t()).chunk(2, dim=-1)
        inter = torch.nn.functional.silu(gate) * up
        out += top_weights[0, route] * (inter @ dw2[eid].t())
    shared_gate_v = torch.sigmoid(x32 @ shared_gate.float().t())
    gate, up = (x32 @ dequant(shared_w13, shared_s13).t()).chunk(2, dim=-1)
    shared_inter = torch.nn.functional.silu(gate) * up
    out += shared_gate_v * (shared_inter @ dequant(shared_w2, shared_s2).t())
    return out


def main():
    device = "xpu"
    experts, hidden, intermediate, top_k = 16, 256, 128, 4
    for seed in range(5):
        torch.manual_seed(seed)
        x = (torch.randn(1, hidden, device=device) * 0.2).half()
        logits = (torch.randn(1, experts, device=device) * 0.4).half()
        w13, s13 = quantize_block(
            torch.randn(experts, 2 * intermediate, hidden, device=device) * 0.15
        )
        w2, s2 = quantize_block(
            torch.randn(experts, hidden, intermediate, device=device) * 0.15
        )
        shared_w13, shared_s13 = quantize_block(
            torch.randn(2 * intermediate, hidden, device=device) * 0.15
        )
        shared_w2, shared_s2 = quantize_block(
            torch.randn(hidden, intermediate, device=device) * 0.15
        )
        shared_gate = (torch.randn(1, hidden, device=device) * 0.1).half()

        expected = reference(
            x,
            logits,
            w13,
            s13,
            shared_w13,
            shared_s13,
            w2,
            s2,
            shared_w2,
            shared_s2,
            shared_gate,
            top_k,
        )
        actual_half = esimd.moe_forward_full_fp8_block(
            x,
            logits,
            w13,
            s13,
            shared_w13,
            shared_s13,
            w2,
            s2,
            shared_w2,
            shared_s2,
            shared_gate,
            top_k,
            1,
            experts,
        )
        actual = actual_half.float()
        torch.xpu.synchronize()
        mean_rel = (actual - expected).abs().mean() / expected.abs().mean()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
        print(f"seed={seed} mean_rel={mean_rel.item():.6e} cos={cosine.item():.8f}")
        assert mean_rel < 3e-3
        assert cosine > 0.9999

        if seed == 0:
            chained_expected = reference(
                actual_half,
                logits,
                w13,
                s13,
                shared_w13,
                shared_s13,
                w2,
                s2,
                shared_w2,
                shared_s2,
                shared_gate,
                top_k,
            )
            chained_actual = esimd.moe_forward_full_fp8_block(
                actual_half,
                logits,
                w13,
                s13,
                shared_w13,
                shared_s13,
                w2,
                s2,
                shared_w2,
                shared_s2,
                shared_gate,
                top_k,
                1,
                experts,
            ).float()
            torch.xpu.synchronize()
            chained_mean_rel = (
                chained_actual - chained_expected
            ).abs().mean() / chained_expected.abs().mean()
            chained_cosine = torch.nn.functional.cosine_similarity(
                chained_actual.flatten(), chained_expected.flatten(), dim=0
            )
            print(
                f"chained mean_rel={chained_mean_rel.item():.6e} "
                f"cos={chained_cosine.item():.8f}"
            )
            assert chained_mean_rel < 3e-3
            assert chained_cosine > 0.9999
    print("ALL_PASS")


if __name__ == "__main__":
    main()
