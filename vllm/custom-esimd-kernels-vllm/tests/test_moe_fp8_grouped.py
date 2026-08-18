import math

import custom_esimd_kernels_vllm
import custom_esimd_kernels_vllm.moe_int4_prefill_ops  # noqa: F401
import pytest
import torch

DEV = torch.device("xpu")


def gelu_tanh(t):
    return 0.5 * t * (
        1.0 + torch.tanh(math.sqrt(2 / math.pi) * (t + 0.044715 * t**3))
    )


def ref_moe(x, weights, indices, w13, w2, gu_s, dn_s, top_k, inter):
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(top_k):
            eid = indices[t, k].item()
            g = (w13[eid].to(torch.float16) * float(gu_s[eid])) @ x[t]
            gate, up = g.split(inter)
            mid = gelu_tanh(gate.float()).to(torch.float16) * up
            out[t] += (
                (w2[eid].to(torch.float16) * float(dn_s[eid])) @ mid
            ) * weights[t, k]
    return out


@pytest.mark.parametrize("native_layout", [False, True])
def test_grouped_fp8_matches_reference(native_layout):
    num_experts, top_k, hidden, intermediate, tokens = 8, 2, 256, 64, 8
    scale = 0.3
    torch.manual_seed(42)
    x = torch.randn(tokens, hidden, dtype=torch.float16, device=DEV) * 0.05
    logits = torch.randn(tokens, num_experts, dtype=torch.float16, device=DEV)
    probs = torch.softmax(logits.float(), -1)
    routing_weights, routing_indices = torch.topk(probs, top_k, -1)
    routing_weights = (
        routing_weights / routing_weights.sum(-1, keepdim=True)
    ).to(torch.float16)
    routing_indices = routing_indices.to(torch.int32)
    w13 = (
        (torch.randn(num_experts, 2 * intermediate, hidden, device=DEV) * 0.05)
        .clamp(-scale, scale)
        .div(scale)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        (torch.randn(num_experts, hidden, intermediate, device=DEV) * 0.05)
        .clamp(-scale, scale)
        .div(scale)
        .to(torch.float8_e4m3fn)
    )
    gu_s = torch.full(
        (num_experts,), scale, dtype=torch.float32, device=DEV
    )
    dn_s = torch.full(
        (num_experts,), scale, dtype=torch.float32, device=DEV
    )
    reference = ref_moe(
        x,
        routing_weights,
        routing_indices,
        w13,
        w2,
        gu_s,
        dn_s,
        top_k,
        intermediate,
    )
    go = torch.ops.moe_int4_prefill_ops.moe_prefill_gather_forward_v2(
        routing_indices.contiguous(), num_experts
    )
    expert_offsets, expert_tokens = go[0], go[1]
    routing_weights = routing_weights.reshape(-1).contiguous()
    if native_layout:
        op = torch.ops.moe_ops.moe_forward_full_fp8_grouped_native
        w13_arg = w13.transpose(1, 2).contiguous().view(torch.uint8)
        w2_arg = w2.transpose(1, 2).contiguous().view(torch.uint8)
    else:
        op = torch.ops.moe_ops.moe_forward_full_fp8_grouped
        w13_arg = w13.view(torch.uint8)
        w2_arg = w2.view(torch.uint8)

    output = op(
        x.contiguous(),
        w13_arg,
        gu_s,
        w2_arg,
        dn_s,
        routing_weights,
        expert_offsets,
        expert_tokens,
        top_k,
        num_experts,
    )
    torch.xpu.synchronize()

    assert not torch.isnan(output).any()
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


def test_grouped_fp8_rejects_unaligned_hidden_size():
    x = torch.empty((1, 24), dtype=torch.float16, device=DEV)
    gate_up = torch.empty((1, 32, 24), dtype=torch.uint8, device=DEV)
    scale = torch.ones((1,), dtype=torch.float32, device=DEV)
    offsets = torch.zeros((1,), dtype=torch.int32, device=DEV)
    tokens = torch.zeros((1,), dtype=torch.int32, device=DEV)

    with pytest.raises(RuntimeError, match="must be multiples of 16"):
        torch.ops.moe_ops.moe_up_fp8_grouped(
            x, gate_up, scale, offsets, tokens, 1, 1
        )
