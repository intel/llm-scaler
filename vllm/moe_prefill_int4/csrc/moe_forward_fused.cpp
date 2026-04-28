// Fused MoE decode v3: minimize kernel submits by using ESIMD kernels
// directly instead of torch::Dispatcher for non-GEMM stages.
//
// Submit chain:
//   1. topk (via ipex dispatcher — 1 submit)
//   2. remap (via _moe_C dispatcher — 3 submits)  
//   3. CUTLASS GEMM W13 (1 submit)
//   4. silu_and_mul (via _C dispatcher — 1 submit)
//   5. CUTLASS GEMM W2 (1 submit)
//   6. gather (via _moe_C dispatcher — 1 submit)
//   7-10. shared expert: mm_out x2 + silu + sigmoid + mul + add (in-place)
//
// All non-GEMM ops called via pre-cached typed handles (no schema lookup).
// Buffer pool eliminates tensor allocation overhead.

#include "utils.h"
#include "grouped_gemm_interface.h"
#include <torch/torch.h>
#include <c10/xpu/XPUStream.h>

// ── Buffer pool ──
struct MoEBuffers {
    int64_t M_cap = 0, total_cap = 0, E = 0, H = 0, two_I = 0, I = 0, top_k = 0;
    int64_t shared_two_I = 0, shared_I = 0;
    torch::Tensor efto, remapped, u2p;
    torch::Tensor gemm1_out, act_out, gemm2_out, output;
    torch::Tensor topk_w, topk_i, topk_ei, topk_i64;
    torch::Tensor shared_gu_out, shared_act, shared_down, gate_val;
    torch::Tensor shared_gu_t, shared_d_t, shared_gate_t;  // pre-transposed
    bool shared_transposed = false;

    void ensure(int64_t M, int64_t total, int64_t E_, int64_t H_, int64_t two_I_,
                int64_t I_, int64_t top_k_, int64_t s_two_I, torch::TensorOptions opts) {
        bool need_alloc = (M > M_cap || total > total_cap || E_ != E);
        if (!need_alloc) { efto.zero_(); return; }
        M_cap = std::max(M, M_cap); total_cap = std::max(total, total_cap);
        E = E_; H = H_; two_I = two_I_; I = I_; top_k = top_k_;
        shared_two_I = s_two_I; shared_I = s_two_I / 2;
        auto Ma = M_cap, Ta = total_cap;
        efto = torch::zeros({E+1}, opts.dtype(torch::kInt64));
        remapped = torch::empty({Ta, H}, opts);
        u2p = torch::empty({Ma, top_k}, opts.dtype(torch::kInt32));
        gemm1_out = torch::empty({Ta, two_I}, opts);
        act_out = torch::empty({Ta, I}, opts);
        gemm2_out = torch::empty({Ta, H}, opts);
        output = torch::empty({Ma, H}, opts);
        topk_w = torch::empty({Ma, top_k}, opts.dtype(torch::kFloat32));
        topk_i = torch::empty({Ma, top_k}, opts.dtype(torch::kInt32));
        topk_ei = torch::empty({Ma, top_k}, opts.dtype(torch::kInt32));
        topk_i64 = torch::empty({Ma, top_k}, opts.dtype(torch::kInt64));
        if (s_two_I > 0) {
            shared_gu_out = torch::empty({Ma, s_two_I}, opts);
            shared_act = torch::empty({Ma, shared_I}, opts);
            shared_down = torch::empty({Ma, H}, opts);
            gate_val = torch::empty({Ma, 1}, opts);
        }
        shared_transposed = false;
    }
};
static thread_local MoEBuffers g_bufs;

// ── Dispatcher cache ──
struct Disp {
    bool ok = false;
    using RemapT = void(torch::Tensor&, const c10::optional<torch::Tensor>&,
        torch::Tensor&, const c10::optional<torch::Tensor>&,
        const c10::optional<torch::Tensor>&,
        torch::Tensor&, torch::Tensor&, torch::Tensor&, int64_t, int64_t);
    using SiluT = void(torch::Tensor&, torch::Tensor&);
    using GatherT = void(torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
        const torch::Tensor&, const torch::Tensor&, int64_t);
    using TopkT = void(torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, bool);
    c10::optional<c10::TypedOperatorHandle<RemapT>> remap;
    c10::optional<c10::TypedOperatorHandle<SiluT>> silu;
    c10::optional<c10::TypedOperatorHandle<GatherT>> gather;
    c10::optional<c10::TypedOperatorHandle<TopkT>> topk;
    void init() {
        if (ok) return;
        auto& d = c10::Dispatcher::singleton();
        remap = d.findSchemaOrThrow("_moe_C::remap_hidden_states", "").typed<RemapT>();
        silu = d.findSchemaOrThrow("_C::silu_and_mul", "").typed<SiluT>();
        gather = d.findSchemaOrThrow("_moe_C::moe_gather", "").typed<GatherT>();
        topk = d.findSchemaOrThrow("torch_ipex::topk_softmax", "moe").typed<TopkT>();
        ok = true;
    }
};
static Disp g_d;

// ── Routed-only fused ──
torch::Tensor moe_forward_fused_cutlass(
    torch::Tensor x, torch::Tensor topk_weights, torch::Tensor topk_ids,
    torch::Tensor w13, torch::Tensor w13_scales,
    torch::Tensor w2, torch::Tensor w2_scales,
    int64_t num_experts, int64_t top_k) {
    g_d.init();
    int64_t M=x.size(0), H=x.size(1), two_I=w13.size(1), I=two_I/2, total=M*top_k;
    g_bufs.ensure(M, total, num_experts, H, two_I, I, top_k, 0, x.options());
    auto& B = g_bufs; B.efto.zero_();
    auto rem_ = B.remapped.slice(0,0,total);
    auto u2p_ = B.u2p.slice(0,0,M);
    g_d.remap->call(x, c10::nullopt, rem_, c10::nullopt, c10::nullopt,
        B.efto, u2p_, topk_ids, num_experts, num_experts);
    auto g1_ = B.gemm1_out.slice(0,0,total);
    cutlass_grouped_gemm_interface(rem_, w13, w13_scales, c10::nullopt,
        g1_, B.efto, two_I, H, num_experts, true, false, false);
    auto act_ = B.act_out.slice(0,0,total);
    g_d.silu->call(act_, g1_);
    auto g2_ = B.gemm2_out.slice(0,0,total);
    cutlass_grouped_gemm_interface(act_, w2, w2_scales, c10::nullopt,
        g2_, B.efto, H, I, num_experts, true, false, false);
    auto out_ = B.output.slice(0,0,M);
    g_d.gather->call(out_, g2_, topk_weights, u2p_, B.efto, num_experts);
    return out_.clone();
}

// ── Full fused: topk + routed + shared ──
torch::Tensor moe_forward_full_fused_cutlass(
    torch::Tensor x, torch::Tensor logits,
    torch::Tensor w13, torch::Tensor w13_scales,
    torch::Tensor w2, torch::Tensor w2_scales,
    torch::Tensor shared_gu_w, torch::Tensor shared_gu_scale,
    torch::Tensor shared_d_w, torch::Tensor shared_d_scale,
    torch::Tensor shared_gate_w,
    int64_t num_experts, int64_t top_k) {

    g_d.init();
    int64_t M=x.size(0), H=x.size(1), two_I=w13.size(1), I=two_I/2, total=M*top_k;
    int64_t s_two_I = (shared_gu_w.numel() > 0) ? shared_gu_w.size(0) : 0;
    g_bufs.ensure(M, total, num_experts, H, two_I, I, top_k, s_two_I, x.options());
    auto& B = g_bufs; B.efto.zero_();

    // Pre-transpose shared expert weights once
    if (shared_gu_w.numel() > 0 && !B.shared_transposed) {
        B.shared_gu_t = shared_gu_w.t().contiguous();
        B.shared_d_t = shared_d_w.t().contiguous();
        B.shared_gate_t = shared_gate_w.t().contiguous();
        B.shared_transposed = true;
    }

    // TopK → pre-allocated buffers (1 submit)
    auto tw = B.topk_w.slice(0,0,M);
    auto ti = B.topk_i.slice(0,0,M);
    auto tei = B.topk_ei.slice(0,0,M);
    g_d.topk->call(tw, ti, tei, logits, true);

    // int32 → int64 (1 submit)
    auto ti64 = B.topk_i64.slice(0,0,M);
    ti64.copy_(ti);

    // Remap (3 submits internally)
    auto remapped = B.remapped.slice(0,0,total);
    auto u2p = B.u2p.slice(0,0,M);
    g_d.remap->call(x, c10::nullopt, remapped, c10::nullopt, c10::nullopt,
        B.efto, u2p, ti64, num_experts, num_experts);

    // GEMM W13 (1 submit)
    auto g1 = B.gemm1_out.slice(0,0,total);
    cutlass_grouped_gemm_interface(remapped, w13, w13_scales, c10::nullopt,
        g1, B.efto, two_I, H, num_experts, true, false, false);

    // silu_and_mul (1 submit)
    auto act = B.act_out.slice(0,0,total);
    g_d.silu->call(act, g1);

    // GEMM W2 (1 submit)
    auto g2 = B.gemm2_out.slice(0,0,total);
    cutlass_grouped_gemm_interface(act, w2, w2_scales, c10::nullopt,
        g2, B.efto, H, I, num_experts, true, false, false);

    // Gather (1 submit)
    auto out = B.output.slice(0,0,M);
    g_d.gather->call(out, g2, tw, u2p, B.efto, num_experts);

    // Shared expert with mm_out + pre-transposed weights (~4 submits, no alloc)
    if (shared_gu_w.numel() > 0) {
        auto sgu_out = B.shared_gu_out.slice(0,0,M);
        auto s_act = B.shared_act.slice(0,0,M);
        auto s_down = B.shared_down.slice(0,0,M);
        auto gv = B.gate_val.slice(0,0,M);

        torch::mm_out(sgu_out, x, B.shared_gu_t);            // 1 submit
        g_d.silu->call(s_act, sgu_out);                       // 1 submit
        torch::mm_out(s_down, s_act, B.shared_d_t);           // 1 submit
        torch::mm_out(gv, x, B.shared_gate_t);                // 1 submit
        // sigmoid + mul + add fused via in-place ops
        out.add_(s_down.mul_(gv.sigmoid_()));                  // ~1-2 submits
    }

    return out.clone();
}
