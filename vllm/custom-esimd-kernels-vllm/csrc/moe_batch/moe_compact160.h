// SPDX-License-Identifier: Apache-2.0
// TP4: physical I=160, original group128 scales (128+32), FP16 shared expert.
// Included by moe_int4.sycl after the common kernels and validation helpers.
#pragma once

static int64_t qwen38_moe_compact160_weight_contract_v1(
    at::TensorList tensors, c10::Device device) {
    TORCH_CHECK(tensors.size() == 7, "compact160 preflight expects seven weights");
    static constexpr int64_t shapes[7][3] = {
        {512,320,1280}, {512,320,20}, {512,2560,80}, {512,2560,2},
        {320,2560,0}, {2560,160,0}, {1,2560,0}};
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (t.device() != device) return 1;
        if (t.scalar_type() != ((i == 0 || i == 2) ? at::kByte : at::kHalf)) return 2;
        const int dims = i < 4 ? 3 : 2;
        if (t.dim() != dims) return 3;
        for (int j = 0; j < dims; ++j) if (t.size(j) != shapes[i][j]) return 3;
        if (!t.is_contiguous() || reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16) return 4;
    }
    return 0;
}

struct MoeCompact160Buffers {
    torch::Tensor ids, weights, routed, shared, gates, logits;
};

static MoeCompact160Buffers& compact160_buffers(
    const c10::xpu::XPUStream& stream, const torch::Device& device) {
    // Separate from TP8: no resize of a workspace that an earlier launch owns.
    static thread_local std::unordered_map<c10::xpu::XPUStream, MoeCompact160Buffers> cache;
    auto found = cache.find(stream);
    if (found != cache.end()) return found->second;
    const auto half = torch::TensorOptions().device(device).dtype(torch::kHalf);
    MoeCompact160Buffers value{
        torch::empty({32,10}, half.dtype(torch::kInt32)),
        torch::empty({32,10}, half), torch::empty({32*10,160}, half),
        torch::empty({32,160}, half), torch::empty({32}, half.dtype(torch::kFloat)),
        torch::empty({1,512}, half)};
    return cache.emplace(stream, std::move(value)).first->second;
}

template <bool Router>
static torch::Tensor compact160_forward(
    torch::Tensor x, torch::Tensor logits_or_router, torch::Tensor router_scale,
    torch::Tensor w13, torch::Tensor s13, torch::Tensor w2, torch::Tensor s2,
    torch::Tensor shared_up, torch::Tensor shared_down, torch::Tensor shared_gate,
    torch::Tensor output, int64_t top_k, int64_t num_shared, int64_t num_experts) {
    TORCH_CHECK(top_k == 10 && num_shared == 1 && num_experts == 512,
                "compact160 requires top_k=10, one shared expert, E=512");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 32,
                "compact160 requires M=1..32");
    const int m = x.size(0);
    const auto device = x.device();
    check_moe_asymmetric_v1_tensor(x, "x", torch::kHalf, {m,2560}, device);
    check_moe_asymmetric_v1_tensor(output, "output", torch::kHalf, {m,2560}, device);
    TORCH_CHECK(qwen38_moe_compact160_weight_contract_v1(
        {w13,s13,w2,s2,shared_up,shared_down,shared_gate}, device) == 0,
        "compact160 weight contract failed");
    if constexpr (Router) {
        TORCH_CHECK(m == 1, "compact160 router hostchain requires M=1");
        check_moe_asymmetric_v1_tensor(logits_or_router, "router", torch::kByte, {512,1280}, device);
        check_moe_asymmetric_v1_tensor(router_scale, "router_scale", torch::kHalf, {512,20}, device);
        TORCH_CHECK(!router_scale.is_neg() && !router_scale.is_conj() &&
                    !moe_asymmetric_v1_tensors_overlap(output, router_scale),
                    "compact160 invalid router scale or output alias");
    } else {
        check_moe_asymmetric_v1_tensor(logits_or_router, "logits", torch::kHalf, {m,512}, device);
    }
    for (const auto& t : {x, logits_or_router, w13, s13, w2, s2, shared_up, shared_down, shared_gate}) {
        TORCH_CHECK(!t.is_neg() && !t.is_conj(), "compact160 rejects lazy neg/conj tensors");
        TORCH_CHECK(!moe_asymmetric_v1_tensors_overlap(output, t),
                    "compact160 output must not alias inputs");
    }
    TORCH_CHECK(!output.is_neg() && !output.is_conj(), "compact160 rejects lazy output");
    const auto stream = c10::xpu::getCurrentXPUStream(device.index());
    auto& queue = stream.queue();
    TORCH_CHECK(queue.is_in_order() && queue.get_context() == c10::xpu::get_device_context() &&
                queue.get_device() == c10::xpu::get_raw_device(device.index()),
                "compact160 requires current in-order XPU queue/device/context");
    // All checks and workspace allocation precede the first submit.
    auto& b = compact160_buffers(stream, device);
    const fp16* logits = reinterpret_cast<const fp16*>(logits_or_router.data_ptr());
    if constexpr (Router) {
        GEMV_int4_host(
            reinterpret_cast<uint8_t*>(x.data_ptr()),
            reinterpret_cast<uint8_t*>(logits_or_router.data_ptr()),
            reinterpret_cast<uint8_t*>(router_scale.data_ptr()),
            reinterpret_cast<uint8_t*>(b.logits.data_ptr()), 512, 2560, queue);
        logits = reinterpret_cast<const fp16*>(b.logits.data_ptr());
    }
    moe_topk_v2_host<512,10>(logits, reinterpret_cast<fp16*>(b.weights.data_ptr()),
                            b.ids.data_ptr<int32_t>(), m, queue);
    if (m == 1) {
        moe_tiny_m_up_cutlass_int4_with_shared_fp16_kernel<int32_t>(
            reinterpret_cast<const fp16*>(x.data_ptr()), w13.data_ptr<uint8_t>(),
            reinterpret_cast<const fp16*>(s13.data_ptr()), b.ids.data_ptr<int32_t>(),
            reinterpret_cast<const fp16*>(shared_up.data_ptr()),
            reinterpret_cast<const fp16*>(shared_gate.data_ptr()),
            reinterpret_cast<fp16*>(b.routed.data_ptr()), reinterpret_cast<fp16*>(b.shared.data_ptr()),
            b.gates.data_ptr<float>(), m, 10, 2560, 160, 160, 1, device);
    } else {
        moe_ws_up_cutlass_int4_with_shared_fp16_kernel<int32_t>(
            reinterpret_cast<const fp16*>(x.data_ptr()), w13.data_ptr<uint8_t>(),
            reinterpret_cast<const fp16*>(s13.data_ptr()), b.ids.data_ptr<int32_t>(),
            reinterpret_cast<const fp16*>(shared_up.data_ptr()),
            reinterpret_cast<const fp16*>(shared_gate.data_ptr()),
            reinterpret_cast<fp16*>(b.routed.data_ptr()), reinterpret_cast<fp16*>(b.shared.data_ptr()),
            b.gates.data_ptr<float>(), m, 10, 2560, 160, 160, 1, device);
    }
    moe_ws_down_cutlass_int4_with_shared_fp16_kernel<int32_t,4,false,true>(
        reinterpret_cast<const fp16*>(b.routed.data_ptr()), w2.data_ptr<uint8_t>(),
        reinterpret_cast<const fp16*>(s2.data_ptr()),
        reinterpret_cast<const fp16*>(b.weights.data_ptr()), b.ids.data_ptr<int32_t>(),
        reinterpret_cast<const fp16*>(b.shared.data_ptr()), b.gates.data_ptr<float>(),
        reinterpret_cast<const fp16*>(shared_down.data_ptr()),
        reinterpret_cast<fp16*>(output.data_ptr()), m, 10, 2560, 160, 160, 1, device);
    return output;
}

static torch::Tensor moe_forward_compact160_out_v1(
    torch::Tensor x, torch::Tensor logits, torch::Tensor w13, torch::Tensor s13,
    torch::Tensor w2, torch::Tensor s2, torch::Tensor shared_up,
    torch::Tensor shared_down, torch::Tensor shared_gate, torch::Tensor output,
    int64_t top_k, int64_t num_shared, int64_t num_experts) {
    return compact160_forward<false>(x, logits, {}, w13, s13, w2, s2,
        shared_up, shared_down, shared_gate, output, top_k, num_shared, num_experts);
}

static torch::Tensor moe_forward_compact160_router_out_v1(
    torch::Tensor x, torch::Tensor router, torch::Tensor router_scale,
    torch::Tensor w13, torch::Tensor s13, torch::Tensor w2, torch::Tensor s2,
    torch::Tensor shared_up, torch::Tensor shared_down, torch::Tensor shared_gate,
    torch::Tensor output, int64_t top_k, int64_t num_shared, int64_t num_experts) {
    return compact160_forward<true>(x, router, router_scale, w13, s13, w2, s2,
        shared_up, shared_down, shared_gate, output, top_k, num_shared, num_experts);
}
