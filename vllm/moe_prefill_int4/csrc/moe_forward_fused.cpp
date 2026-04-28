// Fused MoE decode — optimized for minimal per-call overhead.
// Key optimizations:
//   1. Thread-local buffer pool (no torch::empty per call)
//   2. Pre-cached typed dispatcher handles (no schema lookup per call)
//   3. torch::mm_out for shared expert (no output allocation)
//   4. topk_ids int32→int64 copy via pre-allocated buffer

#include "utils.h"
#include "grouped_gemm_interface.h"
#include <torch/torch.h>
#include <c10/xpu/XPUStream.h>

// ── Buffer pool: allocate once, reuse every call ──
struct MoEBuffers {
    int64_t M_cap = 0, total_cap = 0;
    int64_t E = 0, H = 0, two_I = 0, I = 0, top_k = 0;
    int64_t shared_two_I = 0, shared_I = 0;

    torch::Tensor efto, remapped, u2p;
    torch::Tensor gemm1_out, act_out, gemm2_out, output;
    torch::Tensor topk_w, topk_i, topk_ei, topk_i64;
    // Shared expert buffers
    torch::Tensor shared_gu_out, shared_act, shared_down, gate_val;
    // Atomic buffer for CUTLASS
    torch::Tensor atomic_buf;

    void ensure(int64_t M, int64_t total, int64_t E_, int64_t H_, int64_t two_I_,
                int64_t I_, int64_t top_k_, int64_t shared_two_I_, torch::TensorOptions opts) {
        if (M <= M_cap && total <= total_cap && E_ == E) {
            efto.zero_();
            return;
        }
        M_cap = std::max(M, M_cap);
        total_cap = std::max(total, total_cap);
        E = E_; H = H_; two_I = two_I_; I = I_; top_k = top_k_;
        shared_two_I = shared_two_I_;
        shared_I = shared_two_I_ / 2;
        auto M_a = M_cap; auto tot_a = total_cap;

        efto = torch::zeros({E + 1}, opts.dtype(torch::kInt64));
        remapped = torch::empty({tot_a, H}, opts);
        u2p = torch::empty({M_a, top_k}, opts.dtype(torch::kInt32));
        gemm1_out = torch::empty({tot_a, two_I}, opts);
        act_out = torch::empty({tot_a, I}, opts);
        gemm2_out = torch::empty({tot_a, H}, opts);
        output = torch::empty({M_a, H}, opts);
        topk_w = torch::empty({M_a, top_k}, opts.dtype(torch::kFloat32));
        topk_i = torch::empty({M_a, top_k}, opts.dtype(torch::kInt32));
        topk_ei = torch::empty({M_a, top_k}, opts.dtype(torch::kInt32));
        topk_i64 = torch::empty({M_a, top_k}, opts.dtype(torch::kInt64));
        if (shared_two_I > 0) {
            shared_gu_out = torch::empty({M_a, shared_two_I}, opts);
            shared_act = torch::empty({M_a, shared_I}, opts);
            shared_down = torch::empty({M_a, H}, opts);
            gate_val = torch::empty({M_a, 1}, opts);
        }
    }
};

static thread_local MoEBuffers g_bufs;

// ── Pre-cached typed dispatcher handles ──
struct DispatchCache {
    bool inited = false;

    using RemapFn = void(torch::Tensor&, const c10::optional<torch::Tensor>&,
        torch::Tensor&, const c10::optional<torch::Tensor>&,
        const c10::optional<torch::Tensor>&,
        torch::Tensor&, torch::Tensor&, torch::Tensor&,
        int64_t, int64_t);
    using SiluFn = void(torch::Tensor&, torch::Tensor&);
    using GatherFn = void(torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
        const torch::Tensor&, const torch::Tensor&, int64_t);
    using TopkFn = void(torch::Tensor&, torch::Tensor&, torch::Tensor&,
        torch::Tensor&, bool);

    c10::optional<c10::TypedOperatorHandle<RemapFn>> remap;
    c10::optional<c10::TypedOperatorHandle<SiluFn>> silu;
    c10::optional<c10::TypedOperatorHandle<GatherFn>> gather;
    c10::optional<c10::TypedOperatorHandle<TopkFn>> topk;

    void init() {
        if (inited) return;
        auto& d = c10::Dispatcher::singleton();
        remap = d.findSchemaOrThrow("_moe_C::remap_hidden_states", "").typed<RemapFn>();
        silu = d.findSchemaOrThrow("_C::silu_and_mul", "").typed<SiluFn>();
        gather = d.findSchemaOrThrow("_moe_C::moe_gather", "").typed<GatherFn>();
        topk = d.findSchemaOrThrow("torch_ipex::topk_softmax", "moe").typed<TopkFn>();
        inited = true;
    }
};

static DispatchCache g_disp;

// ── Routed-only fused op ──
torch::Tensor moe_forward_fused_cutlass(
    torch::Tensor x, torch::Tensor topk_weights, torch::Tensor topk_ids,
    torch::Tensor w13, torch::Tensor w13_scales,
    torch::Tensor w2, torch::Tensor w2_scales,
    int64_t num_experts, int64_t top_k) {

    g_disp.init();
    int64_t M = x.size(0), H = x.size(1);
    int64_t two_I = w13.size(1), I = two_I / 2, total = M * top_k;
    auto opts = x.options();

    g_bufs.ensure(M, total, num_experts, H, two_I, I, top_k, 0, opts);
    auto& B = g_bufs;
    B.efto.zero_();
    auto& topk_ids_i64 = topk_ids; // already int64

    g_disp.remap->call(x, c10::nullopt, B.remapped, c10::nullopt, c10::nullopt,
        B.efto, B.u2p, topk_ids_i64, num_experts, num_experts);

    cutlass_grouped_gemm_interface(B.remapped, w13, w13_scales, c10::nullopt,
        B.gemm1_out, B.efto, two_I, H, num_experts, true, false, false);

    g_disp.silu->call(B.act_out, B.gemm1_out);

    cutlass_grouped_gemm_interface(B.act_out, w2, w2_scales, c10::nullopt,
        B.gemm2_out, B.efto, H, I, num_experts, true, false, false);

    g_disp.gather->call(B.output, B.gemm2_out, topk_weights, B.u2p, B.efto, num_experts);

    return B.output.slice(0, 0, M).clone();
}

// ── Full fused: topk + routed + shared expert ──
torch::Tensor moe_forward_full_fused_cutlass(
    torch::Tensor x, torch::Tensor logits,
    torch::Tensor w13, torch::Tensor w13_scales,
    torch::Tensor w2, torch::Tensor w2_scales,
    torch::Tensor shared_gu_w,      // [2*I_shared, H] fp16
    torch::Tensor shared_gu_scale,  // unused
    torch::Tensor shared_d_w,       // [H, I_shared] fp16
    torch::Tensor shared_d_scale,   // unused
    torch::Tensor shared_gate_w,    // [1, H] fp16
    int64_t num_experts, int64_t top_k) {

    g_disp.init();
    int64_t M = x.size(0), H = x.size(1);
    int64_t two_I = w13.size(1), I = two_I / 2, total = M * top_k;
    int64_t s_two_I = (shared_gu_w.numel() > 0) ? shared_gu_w.size(0) : 0;
    auto opts = x.options();

    g_bufs.ensure(M, total, num_experts, H, two_I, I, top_k, s_two_I, opts);
    auto& B = g_bufs;
    B.efto.zero_();

    // ── TopK (outputs to pre-allocated buffers) ──
    auto tw = B.topk_w.slice(0, 0, M);
    auto ti = B.topk_i.slice(0, 0, M);
    auto tei = B.topk_ei.slice(0, 0, M);
    g_disp.topk->call(tw, ti, tei, logits, true);

    // int32 → int64 copy (device-to-device, no sync)
    auto ti64 = B.topk_i64.slice(0, 0, M);
    ti64.copy_(ti);

    // ── Remap ──
    auto remapped = B.remapped.slice(0, 0, total);
    auto u2p = B.u2p.slice(0, 0, M);
    g_disp.remap->call(x, c10::nullopt, remapped, c10::nullopt, c10::nullopt,
        B.efto, u2p, ti64, num_experts, num_experts);

    // ── CUTLASS GEMM W13 ──
    auto gemm1 = B.gemm1_out.slice(0, 0, total);
    cutlass_grouped_gemm_interface(remapped, w13, w13_scales, c10::nullopt,
        gemm1, B.efto, two_I, H, num_experts, true, false, false);

    // ── SiLU + Mul ──
    auto act = B.act_out.slice(0, 0, total);
    g_disp.silu->call(act, gemm1);

    // ── CUTLASS GEMM W2 ──
    auto gemm2 = B.gemm2_out.slice(0, 0, total);
    cutlass_grouped_gemm_interface(act, w2, w2_scales, c10::nullopt,
        gemm2, B.efto, H, I, num_experts, true, false, false);

    // ── Gather ──
    auto out = B.output.slice(0, 0, M);
    g_disp.gather->call(out, gemm2, tw, u2p, B.efto, num_experts);

    // ── Shared expert (mm_out to pre-allocated buffers) ──
    if (shared_gu_w.numel() > 0) {
        auto sgu_t = shared_gu_w.t();
        auto sd_t = shared_d_w.t();
        auto sg_t = shared_gate_w.t();

        auto sgu_out = B.shared_gu_out.slice(0, 0, M);
        auto s_act = B.shared_act.slice(0, 0, M);
        auto s_down = B.shared_down.slice(0, 0, M);
        auto gv = B.gate_val.slice(0, 0, M);

        torch::mm_out(sgu_out, x, sgu_t);
        g_disp.silu->call(s_act, sgu_out);
        torch::mm_out(s_down, s_act, sd_t);
        torch::mm_out(gv, x, sg_t);
        gv.sigmoid_();
        out.add_(s_down * gv);
    }

    return out.clone();
}
