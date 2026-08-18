import math

import custom_esimd_kernels_vllm  # noqa: F401
import pytest
import torch


def _gelu_tanh(value):
    return 0.5 * value * (
        1.0
        + torch.tanh(
            math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)
        )
    )


def _make_case(hidden, intermediate, seed):
    num_experts, top_k = 8, 2
    scale = 0.3
    torch.manual_seed(seed)
    device = torch.device("xpu")
    x = torch.randn((1, hidden), dtype=torch.float16, device=device) * 0.05
    logits = torch.randn(
        (1, num_experts), dtype=torch.float16, device=device
    )
    gate_up = (
        (
            torch.randn(
                (num_experts, 2 * intermediate, hidden), device=device
            )
            * 0.05
        )
        .clamp(-scale, scale)
        .div(scale)
        .to(torch.float8_e4m3fn)
    )
    down = (
        (
            torch.randn(
                (num_experts, hidden, intermediate), device=device
            )
            * 0.05
        )
        .clamp(-scale, scale)
        .div(scale)
        .to(torch.float8_e4m3fn)
    )
    gate_up_scale = torch.full(
        (num_experts,), scale, dtype=torch.float32, device=device
    )
    down_scale = torch.full(
        (num_experts,), scale, dtype=torch.float32, device=device
    )
    expert_scale = torch.ones(
        (num_experts,), dtype=torch.float32, device=device
    )

    probabilities = torch.softmax(logits.float(), dim=-1)
    routing_weights, routing_indices = torch.topk(
        probabilities, top_k, dim=-1
    )
    routing_weights = (
        routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    ).to(torch.float16)
    reference = torch.zeros_like(x)
    for route in range(top_k):
        expert = routing_indices[0, route].item()
        projected = (
            gate_up[expert].to(torch.float16) * gate_up_scale[expert]
        ) @ x[0]
        gate, up = projected.split(intermediate)
        activated = _gelu_tanh(gate.float()).to(torch.float16) * up
        reference[0] += (
            (down[expert].to(torch.float16) * down_scale[expert]) @ activated
        ) * routing_weights[0, route]

    args = (
        x,
        logits,
        gate_up,
        gate_up_scale,
        down,
        down_scale,
        expert_scale,
        top_k,
        num_experts,
    )
    return args, reference


@pytest.mark.parametrize(
    ("hidden", "intermediate"),
    [(256, 192), (288, 64), (2816, 704)],
)
def test_canonical_decode_handles_32_element_tail_chunks(
    hidden, intermediate
):
    args, reference = _make_case(hidden, intermediate, seed=hidden)
    output = torch.ops.moe_ops.moe_forward_full_gelu_tanh_decode(*args)
    torch.xpu.synchronize()
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


def test_decode_scratch_cache_reallocates_for_new_shape():
    first_args, first_reference = _make_case(256, 192, seed=1)
    second_args, second_reference = _make_case(288, 64, seed=2)

    first_output = (
        torch.ops.moe_ops.moe_forward_full_gelu_tanh_decode(*first_args).clone()
    )
    second_output = (
        torch.ops.moe_ops.moe_forward_full_gelu_tanh_decode(*second_args).clone()
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(
        first_output, first_reference, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        second_output, second_reference, atol=3e-2, rtol=3e-2
    )


def test_decode_scratch_cache_isolated_per_stream():
    first_args, first_reference = _make_case(256, 192, seed=3)
    second_args, second_reference = _make_case(256, 192, seed=4)
    first_stream = torch.xpu.Stream()
    second_stream = torch.xpu.Stream()

    with torch.xpu.stream(first_stream):
        first_output = (
            torch.ops.moe_ops.moe_forward_full_gelu_tanh_decode(
                *first_args
            ).clone()
        )
    with torch.xpu.stream(second_stream):
        second_output = (
            torch.ops.moe_ops.moe_forward_full_gelu_tanh_decode(
                *second_args
            ).clone()
        )
    torch.xpu.synchronize()

    torch.testing.assert_close(
        first_output, first_reference, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        second_output, second_reference, atol=3e-2, rtol=3e-2
    )
