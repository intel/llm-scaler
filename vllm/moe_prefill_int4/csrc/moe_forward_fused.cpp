#include "utils.h"
#include "grouped_gemm_interface.h"
#include <torch/torch.h>
#include <c10/xpu/XPUStream.h>

static c10::OperatorHandle find_op(const char* name) {
    return c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
}

torch::Tensor moe_forward_fused_cutlass(
    torch::Tensor x,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor w13,
    torch::Tensor w13_scales,
    torch::Tensor w2,
    torch::Tensor w2_scales,
    int64_t num_experts,
    int64_t top_k) {

    int64_t M = x.size(0);
    int64_t H = x.size(1);
    int64_t two_I = w13.size(1);
    int64_t I = two_I / 2;
    int64_t total = M * top_k;

    auto opts = x.options();
    auto efto = torch::zeros({num_experts + 1}, opts.dtype(torch::kInt64));
    auto remapped = torch::empty({total, H}, opts);
    auto u2p = torch::empty({M, top_k}, opts.dtype(torch::kInt32));
    auto gemm1_out = torch::empty({total, two_I}, opts);
    auto act_out = torch::empty({total, I}, opts);
    auto gemm2_out = torch::empty({total, H}, opts);
    auto output = torch::empty({M, H}, opts);
    auto& topk_ids_i64 = topk_ids; // already int64 from Python

    static auto remap_op = find_op("_moe_C::remap_hidden_states");
    remap_op.typed<void(
        torch::Tensor&, const c10::optional<torch::Tensor>&,
        torch::Tensor&, const c10::optional<torch::Tensor>&,
        const c10::optional<torch::Tensor>&,
        torch::Tensor&, torch::Tensor&, torch::Tensor&,
        int64_t, int64_t)>().call(
        x, c10::nullopt, remapped, c10::nullopt, c10::nullopt,
        efto, u2p, topk_ids_i64, num_experts, num_experts);

    cutlass_grouped_gemm_interface(
        remapped, w13, w13_scales, c10::nullopt,
        gemm1_out, efto, two_I, H, num_experts,
        true, false, false);

    static auto silu_op = find_op("_C::silu_and_mul");
    silu_op.typed<void(torch::Tensor&, torch::Tensor&)>().call(act_out, gemm1_out);

    cutlass_grouped_gemm_interface(
        act_out, w2, w2_scales, c10::nullopt,
        gemm2_out, efto, H, I, num_experts,
        true, false, false);

    static auto gather_op = find_op("_moe_C::moe_gather");
    gather_op.typed<void(
        torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
        const torch::Tensor&, const torch::Tensor&, int64_t)>().call(
        output, gemm2_out, topk_weights, u2p, efto, num_experts);

    return output;
}

// Full fused: topk + routed + shared expert, single Python dispatch
torch::Tensor moe_forward_full_fused_cutlass(
    torch::Tensor x,               // [M, H] fp16
    torch::Tensor logits,           // [M, E] fp16 (router logits)
    torch::Tensor w13,              // [E, 2*I, K/2] uint8
    torch::Tensor w13_scales,       // [E, 2*I, K/GS] fp16
    torch::Tensor w2,               // [E, H, I/2] uint8
    torch::Tensor w2_scales,        // [E, H, I/GS] fp16
    torch::Tensor shared_gate_up_weight,   // shared expert gate_up weight
    torch::Tensor shared_gate_up_scale,    // shared expert gate_up scale (or empty)
    torch::Tensor shared_down_weight,      // shared expert down weight
    torch::Tensor shared_down_scale,       // shared expert down scale (or empty)
    torch::Tensor shared_expert_gate_weight, // [1, H] fp16
    int64_t num_experts,
    int64_t top_k) {

    int64_t M = x.size(0);
    int64_t H = x.size(1);
    int64_t two_I = w13.size(1);
    int64_t I = two_I / 2;
    int64_t total = M * top_k;

    auto opts = x.options();

    // ── TopK via ipex ──
    auto topk_weights = torch::empty({M, top_k}, opts.dtype(torch::kFloat32));
    auto topk_ids = torch::empty({M, top_k}, opts.dtype(torch::kInt32));
    auto topk_expert_idx = torch::empty({M, top_k}, opts.dtype(torch::kInt32));

    static auto topk_op = c10::Dispatcher::singleton().findSchemaOrThrow("torch_ipex::topk_softmax", "moe");
    topk_op.typed<void(torch::Tensor&, torch::Tensor&, torch::Tensor&,
                        torch::Tensor&, bool)>().call(
        topk_weights, topk_ids, topk_expert_idx, logits, true);

    auto topk_ids_i64 = topk_ids.to(torch::kInt64);

    // ── Routed MoE ──
    auto efto = torch::zeros({num_experts + 1}, opts.dtype(torch::kInt64));
    auto remapped = torch::empty({total, H}, opts);
    auto u2p = torch::empty({M, top_k}, opts.dtype(torch::kInt32));
    auto gemm1_out = torch::empty({total, two_I}, opts);
    auto act_out = torch::empty({total, I}, opts);
    auto gemm2_out = torch::empty({total, H}, opts);
    auto output = torch::empty({M, H}, opts);

    static auto remap_op = find_op("_moe_C::remap_hidden_states");
    remap_op.typed<void(
        torch::Tensor&, const c10::optional<torch::Tensor>&,
        torch::Tensor&, const c10::optional<torch::Tensor>&,
        const c10::optional<torch::Tensor>&,
        torch::Tensor&, torch::Tensor&, torch::Tensor&,
        int64_t, int64_t)>().call(
        x, c10::nullopt, remapped, c10::nullopt, c10::nullopt,
        efto, u2p, topk_ids_i64, num_experts, num_experts);

    cutlass_grouped_gemm_interface(
        remapped, w13, w13_scales, c10::nullopt,
        gemm1_out, efto, two_I, H, num_experts,
        true, false, false);

    static auto silu_op = find_op("_C::silu_and_mul");
    silu_op.typed<void(torch::Tensor&, torch::Tensor&)>().call(act_out, gemm1_out);

    cutlass_grouped_gemm_interface(
        act_out, w2, w2_scales, c10::nullopt,
        gemm2_out, efto, H, I, num_experts,
        true, false, false);

    static auto gather_op = find_op("_moe_C::moe_gather");
    gather_op.typed<void(
        torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
        const torch::Tensor&, const torch::Tensor&, int64_t)>().call(
        output, gemm2_out, topk_weights, u2p, efto, num_experts);

    // ── Shared expert (FP16 matmul path) ──
    // gate_up = x @ shared_gate_up_weight.T
    auto shared_gu = torch::mm(x, shared_gate_up_weight.t());
    int64_t shared_I = shared_gu.size(1) / 2;
    auto shared_act = torch::empty({M, shared_I}, opts);
    silu_op.typed<void(torch::Tensor&, torch::Tensor&)>().call(shared_act, shared_gu);
    // down = shared_act @ shared_down_weight.T
    auto shared_down = torch::mm(shared_act, shared_down_weight.t());
    // gate = sigmoid(x @ shared_expert_gate_weight.T)
    auto gate_val = torch::sigmoid(torch::mm(x, shared_expert_gate_weight.t()));
    // output += shared_down * gate
    output.add_(shared_down * gate_val);

    return output;
}
