import pytest
import torch
import torch.nn.functional as F

import custom_esimd_kernels_sglang


NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2816
INTERMEDIATE_SIZE_TP2 = 352


@torch.no_grad()
@pytest.mark.parametrize("with_start", [False, True])
@pytest.mark.parametrize("req_index_dtype", [torch.int32, torch.int64])
def test_xpu_create_kv_indices(with_start, req_index_dtype):
    device = torch.device("xpu")
    req_to_token = torch.arange(
        4 * 32, dtype=torch.int32, device=device
    ).view(4, 32)
    req_pool_indices = torch.tensor(
        [2, 0, 3], dtype=req_index_dtype, device=device
    )
    lengths = torch.tensor(
        [5, 3, 7], dtype=torch.int32, device=device
    )
    starts = (
        torch.tensor([4, 2, 1], dtype=torch.int32, device=device)
        if with_start
        else None
    )
    kv_indptr = torch.tensor(
        [0, 5, 8, 15], dtype=torch.int32, device=device
    )
    output = torch.empty(15, dtype=torch.int32, device=device)
    custom_esimd_kernels_sglang.xpu_create_kv_indices(
        req_to_token,
        req_pool_indices,
        lengths,
        kv_indptr,
        starts,
        output,
        7,
    )
    starts_cpu = [4, 2, 1] if with_start else [0, 0, 0]
    reference = torch.cat(
        [
            req_to_token[req, start : start + length]
            for req, start, length in zip(
                [2, 0, 3], starts_cpu, [5, 3, 7]
            )
        ]
    )
    torch.testing.assert_close(output, reference)


def _reference(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    num_tokens = x.shape[0]
    x_routes = x.repeat_interleave(TOP_K, dim=0)
    route_ids = topk_ids.reshape(-1)
    route_weights = topk_weights.reshape(-1)
    partials = torch.empty(
        num_tokens * TOP_K,
        HIDDEN_SIZE,
        dtype=torch.float16,
        device=x.device,
    )

    for expert_id in torch.unique(route_ids).cpu().tolist():
        mask = route_ids == expert_id
        gate_up = F.linear(
            x_routes[mask],
            w13[expert_id].to(torch.float16) * w13_scale[expert_id],
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.gelu(gate, approximate="tanh") * up
        partials[mask] = F.linear(
            intermediate,
            w2[expert_id].to(torch.float16) * w2_scale[expert_id],
        ) * route_weights[mask, None]

    return partials.view(num_tokens, TOP_K, HIDDEN_SIZE).float().sum(1).half()


def _inputs(fp8_dtype: torch.dtype, num_tokens: int):
    device = torch.device("xpu")
    torch.manual_seed(42)
    x = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        dtype=torch.float16,
        device=device,
    ) * 0.05
    topk_ids = (
        torch.arange(num_tokens * TOP_K, device=device, dtype=torch.int32)
        .view(num_tokens, TOP_K)
        .remainder_(16)
    )
    topk_weights = torch.rand(
        num_tokens,
        TOP_K,
        dtype=torch.float16,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    w13 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE_SIZE_TP2,
            HIDDEN_SIZE,
            dtype=torch.float16,
            device=device,
        )
        .mul_(0.5)
        .to(fp8_dtype)
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE_TP2,
            dtype=torch.float16,
            device=device,
        )
        .mul_(0.5)
        .to(fp8_dtype)
    )
    w13_scale = torch.full(
        (NUM_EXPERTS,), 0.02, dtype=torch.float32, device=device
    )
    w2_scale = torch.full(
        (NUM_EXPERTS,), 0.02, dtype=torch.float32, device=device
    )
    return x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale


@torch.no_grad()
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_gemma4_moe_fp8_real_tp2_shape(fp8_dtype):
    for num_tokens in (1, 8, 32):
        args = _inputs(fp8_dtype, num_tokens)
        reference = _reference(*args)
        x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale = args

        if num_tokens <= 8:
            output = (
                custom_esimd_kernels_sglang.moe_forward_full_gelu_tanh_routed(
                    x,
                    topk_weights,
                    topk_ids,
                    w13,
                    w13_scale,
                    w2,
                    w2_scale,
                    TOP_K,
                    NUM_EXPERTS,
                )
            )
        else:
            gate_up_scale = (
                w13_scale[:, None].expand(NUM_EXPERTS, 2).contiguous()
            )
            output = (
                custom_esimd_kernels_sglang.moe_prefill_full_fp8_gelu_tanh(
                    x,
                    topk_weights,
                    topk_ids,
                    w13,
                    gate_up_scale,
                    w2,
                    w2_scale,
                    TOP_K,
                    NUM_EXPERTS,
                )
            )

        torch.testing.assert_close(output, reference, rtol=0.04, atol=0.01)


@torch.no_grad()
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_gemma4_moe_fp8_fused_decode_from_logits(fp8_dtype):
    x, _, _, w13, w13_scale, w2, w2_scale = _inputs(fp8_dtype, 1)
    torch.manual_seed(17)
    logits = torch.randn(
        1, NUM_EXPERTS, dtype=torch.float16, device=x.device
    )
    per_expert_scale = (
        torch.rand(NUM_EXPERTS, dtype=torch.float32, device=x.device) + 0.5
    )
    topk_logits, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_weights = F.softmax(topk_logits.float(), dim=-1)
    topk_weights *= per_expert_scale[topk_ids]
    reference = _reference(
        x,
        topk_weights.half(),
        topk_ids.int(),
        w13,
        w13_scale,
        w2,
        w2_scale,
    )

    output = custom_esimd_kernels_sglang.moe_forward_full_gelu_tanh_decode(
        x,
        logits,
        w13,
        w13_scale,
        w2,
        w2_scale,
        per_expert_scale,
        TOP_K,
        NUM_EXPERTS,
    )
    torch.testing.assert_close(output, reference, rtol=0.04, atol=0.01)


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_gemma4_production_topk(num_tokens):
    device = torch.device("xpu")
    torch.manual_seed(19)
    logits = torch.randn(
        num_tokens,
        NUM_EXPERTS,
        dtype=torch.float16,
        device=device,
    )
    topk_ids, topk_weights = (
        custom_esimd_kernels_sglang.moe_batch_topk(
            logits, TOP_K, True
        )
    )
    reference_logits, reference_ids = torch.topk(
        logits, TOP_K, dim=-1
    )
    reference_weights = F.softmax(
        reference_logits.float(), dim=-1
    )
    torch.testing.assert_close(topk_ids, reference_ids.int())
    torch.testing.assert_close(
        topk_weights.float(),
        reference_weights,
        rtol=0.002,
        atol=0.0002,
    )
    expert_scale = torch.rand(
        NUM_EXPERTS, dtype=torch.float32, device=device
    )
    scaled = topk_weights * expert_scale[
        topk_ids.to(torch.long)
    ].to(torch.float16)
    reference_scaled = reference_weights * expert_scale[
        reference_ids
    ]
    torch.testing.assert_close(
        scaled.float(),
        reference_scaled,
        rtol=0.003,
        atol=0.0003,
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_gemma4_moe_fp8_gelu_tanh_saturates(fp8_dtype):
    device = torch.device("xpu")
    scale = 0.5
    w13 = torch.zeros(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE_TP2,
        HIDDEN_SIZE,
        dtype=fp8_dtype,
        device=device,
    )
    w2 = torch.zeros(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE_TP2,
        dtype=fp8_dtype,
        device=device,
    )
    w13[0, INTERMEDIATE_SIZE_TP2, 0] = 2.0
    w2[0, :, 0] = 2.0
    w13_scale = torch.full(
        (NUM_EXPERTS,), scale, dtype=torch.float32, device=device
    )
    w2_scale = torch.full(
        (NUM_EXPERTS,), scale, dtype=torch.float32, device=device
    )

    for gate_value in (24.0, 160.0):
        w13[0, 0, 0] = gate_value / scale
        for num_tokens in (1, 32):
            x = torch.zeros(
                num_tokens, HIDDEN_SIZE, dtype=torch.float16, device=device
            )
            x[:, 0] = 1.0
            topk_ids = torch.zeros(
                num_tokens, 1, dtype=torch.int32, device=device
            )
            topk_weights = torch.ones(
                num_tokens, 1, dtype=torch.float16, device=device
            )
            if num_tokens == 1:
                output = (
                    custom_esimd_kernels_sglang.moe_forward_full_gelu_tanh_routed(
                        x,
                        topk_weights,
                        topk_ids,
                        w13,
                        w13_scale,
                        w2,
                        w2_scale,
                        1,
                        NUM_EXPERTS,
                    )
                )
            else:
                output = (
                    custom_esimd_kernels_sglang.moe_prefill_full_fp8_gelu_tanh(
                        x,
                        topk_weights,
                        topk_ids,
                        w13,
                        w13_scale[:, None].expand(NUM_EXPERTS, 2).contiguous(),
                        w2,
                        w2_scale,
                        1,
                        NUM_EXPERTS,
                    )
                )

            expected_value = F.gelu(
                torch.tensor(gate_value, dtype=torch.float32, device=device),
                approximate="tanh",
            )
            expected = torch.full_like(output, expected_value)
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@torch.no_grad()
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_gemma4_moe_fp8_fused_decode_gelu_tanh_saturates(fp8_dtype):
    device = torch.device("xpu")
    scale = 0.5
    x = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float16, device=device)
    x[:, 0] = 1.0
    logits = torch.full(
        (1, NUM_EXPERTS), -10.0, dtype=torch.float16, device=device
    )
    logits[:, 0] = 10.0
    w13 = torch.zeros(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE_TP2,
        HIDDEN_SIZE,
        dtype=fp8_dtype,
        device=device,
    )
    w2 = torch.zeros(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE_TP2,
        dtype=fp8_dtype,
        device=device,
    )
    w13[0, INTERMEDIATE_SIZE_TP2, 0] = 2.0
    w2[0, :, 0] = 2.0
    w13_scale = torch.full(
        (NUM_EXPERTS,), scale, dtype=torch.float32, device=device
    )
    w2_scale = torch.full_like(w13_scale, scale)
    per_expert_scale = torch.ones_like(w13_scale)

    for gate_value in (24.0, 160.0):
        w13[0, 0, 0] = gate_value / scale
        output = (
            custom_esimd_kernels_sglang.moe_forward_full_gelu_tanh_decode(
                x,
                logits,
                w13,
                w13_scale,
                w2,
                w2_scale,
                per_expert_scale,
                1,
                NUM_EXPERTS,
            )
        )
        expected_value = F.gelu(
            torch.tensor(gate_value, dtype=torch.float32, device=device),
            approximate="tanh",
        )
        expected = torch.full_like(output, expected_value)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@torch.no_grad()
def test_gemma4_router_norm_gemv_norm_fp16():
    device = torch.device("xpu")
    eps = 1e-6
    torch.manual_seed(42)
    residual = torch.randn(
        1, HIDDEN_SIZE, dtype=torch.float16, device=device
    )
    scale_with_root = torch.randn(
        HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.02
    proj_weight = torch.randn(
        NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.02
    pre_ff_weight = torch.randn(
        HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.02
    router_logits = torch.empty(
        1, NUM_EXPERTS, dtype=torch.float16, device=device
    )
    moe_input = torch.empty_like(residual)

    custom_esimd_kernels_sglang.esimd_norm_gemv_norm_fp16(
        residual,
        scale_with_root,
        proj_weight,
        pre_ff_weight,
        router_logits,
        moe_input,
        eps,
    )

    normalized = residual.float() * torch.rsqrt(
        residual.float().square().mean(dim=-1, keepdim=True) + eps
    )
    reference_input = normalized * pre_ff_weight.float()
    reference_logits = (normalized * scale_with_root.float()) @ proj_weight.float().T
    torch.testing.assert_close(
        moe_input.float(), reference_input, rtol=0.01, atol=0.002
    )
    torch.testing.assert_close(
        router_logits.float(), reference_logits, rtol=0.02, atol=0.005
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_gemma4_dense_norm_gemv_gelu_fusion(fp8_dtype):
    device = torch.device("xpu")
    dense_intermediate = 1056
    eps = 1e-6
    torch.manual_seed(23)
    attention_output = torch.randn(
        1, HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.05
    residual_input = torch.randn_like(attention_output) * 0.05
    post_weight = torch.randn(
        HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.05
    pre_weight = torch.randn_like(post_weight) * 0.05
    gate_up_weight = (
        torch.randn(
            2 * dense_intermediate,
            HIDDEN_SIZE,
            dtype=torch.float16,
            device=device,
        )
        .mul_(0.5)
        .to(fp8_dtype)
    )
    gate_up_scale = torch.tensor(
        [0.02], dtype=torch.float32, device=device
    )
    residual_output = torch.empty_like(residual_input)
    activation_output = torch.empty(
        1, dense_intermediate, dtype=torch.float16, device=device
    )

    custom_esimd_kernels_sglang.esimd_norm_add_norm_gemv_gelu_fp8(
        attention_output,
        residual_input,
        post_weight,
        pre_weight,
        gate_up_weight,
        gate_up_scale,
        residual_output,
        activation_output,
        eps,
        eps,
    )

    attention_norm = attention_output.float() * torch.rsqrt(
        attention_output.float().square().mean(dim=-1, keepdim=True) + eps
    )
    reference_residual = (
        attention_norm * post_weight.float() + residual_input.float()
    )
    pre_norm = reference_residual * torch.rsqrt(
        reference_residual.square().mean(dim=-1, keepdim=True) + eps
    )
    pre_norm *= pre_weight.float()
    gate_up = F.linear(
        pre_norm,
        gate_up_weight.float() * gate_up_scale,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    reference_activation = F.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(
        residual_output.float(),
        reference_residual,
        rtol=0.02,
        atol=0.003,
    )
    torch.testing.assert_close(
        activation_output.float(),
        reference_activation,
        rtol=0.05,
        atol=0.01,
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_gemma4_input_norm_qkv_gemv_fusion(fp8_dtype):
    device = torch.device("xpu")
    qkv_size = 4096
    eps = 1e-6
    torch.manual_seed(31)
    input = torch.randn(
        1, HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.05
    norm_weight = torch.randn(
        HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.05
    qkv_weight = (
        torch.randn(
            qkv_size,
            HIDDEN_SIZE,
            dtype=torch.float16,
            device=device,
        )
        .mul_(0.5)
        .to(fp8_dtype)
    )
    qkv_scale = torch.tensor(
        [0.02], dtype=torch.float32, device=device
    )
    output = torch.empty(
        1, qkv_size, dtype=torch.float16, device=device
    )

    custom_esimd_kernels_sglang.esimd_rmsnorm_gemv_fp8(
        input,
        norm_weight,
        qkv_weight,
        qkv_scale,
        output,
        eps,
    )

    normalized = input.float() * torch.rsqrt(
        input.float().square().mean(dim=-1, keepdim=True) + eps
    )
    normalized = (normalized * norm_weight.float()).half().float()
    reference = F.linear(
        normalized,
        qkv_weight.float() * qkv_scale,
    )
    torch.testing.assert_close(
        output.float(), reference, rtol=0.04, atol=0.01
    )


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_gemma4_dual_rmsnorm_residual_scalar(num_tokens):
    device = torch.device("xpu")
    eps = 1e-6
    scalar = 0.75
    torch.manual_seed(41)
    x1 = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.05
    x2 = torch.randn_like(x1) * 0.05
    residual = torch.randn_like(x1) * 0.05
    weight1 = torch.randn(
        HIDDEN_SIZE, dtype=torch.float16, device=device
    ) * 0.05
    weight2 = torch.randn_like(weight1) * 0.05
    weight3 = torch.randn_like(weight1) * 0.05
    output = torch.empty_like(x1)

    custom_esimd_kernels_sglang.esimd_dual_rmsnorm_residual_scalar(
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        output,
        eps,
        eps,
        eps,
        scalar,
    )

    norm1 = x1.float() * torch.rsqrt(
        x1.float().square().mean(dim=-1, keepdim=True) + eps
    ) * weight1.float()
    norm2 = x2.float() * torch.rsqrt(
        x2.float().square().mean(dim=-1, keepdim=True) + eps
    ) * weight2.float()
    combined = norm1 + norm2
    reference = combined * torch.rsqrt(
        combined.square().mean(dim=-1, keepdim=True) + eps
    ) * weight3.float()
    reference = (reference + residual.float()) * scalar
    torch.testing.assert_close(
        output.float(), reference, rtol=0.03, atol=0.005
    )
