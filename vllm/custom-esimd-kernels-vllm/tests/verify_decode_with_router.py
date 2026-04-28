"""Verify decode including router: old ESIMD vs new CUTLASS.

OLD path: moe_forward_full_int4 (internal router + routed + shared, all fused)
  - Internal router uses the SAME w13/w2 K-major marlin weights

NEW path: external router (IPEX or ESIMD) + topk_softmax + xpu_fused_moe + shared
  - Router is separate, then topk_softmax, then fused_moe

The OLD path does its own internal topk inside the kernel. To match it exactly,
the NEW path must use the SAME topk results. This test checks whether the
topk_softmax from IPEX produces the same routing as the internal one.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=0 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/verify_decode_with_router.py"
"""
import sys, pathlib, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe, implement_zp
from custom_esimd_kernels_vllm import moe_int4_ops
from custom_esimd_kernels_vllm.ops import moe_forward_full_int4

sys.path.insert(0, str(pathlib.Path("/llm/models/test/llm-scaler-vllm-xpu")))
from vllm.model_executor.layers.quantization.sym_int4 import ggml_quantize_tensor

DEVICE = "xpu"
DTYPE = torch.float16
GS = 128
_MARLIN_SHUFFLED_IDX = (0, 4, 1, 5, 2, 6, 3, 7)

H, I, E, TK = 3072, 1024, 256, 8
TWO_I = 2 * I


def ggml_to_ipex_kmajor_marlin(qw_nk, sc_nk):
    qw_t = qw_nk.permute(0, 2, 1).contiguous()
    sc_t = sc_nk.permute(0, 2, 1).contiguous()
    qw = qw_t.to(torch.int64) & 0xFFFFFFFF
    shuffled = torch.zeros_like(qw)
    for p in range(8):
        o = _MARLIN_SHUFFLED_IDX[p]
        shuffled |= ((qw >> (o * 4)) & 0xF) << (p * 4)
    return (shuffled & 0xFFFFFFFF).to(torch.int32).contiguous(), sc_t


def ggml_to_cutlass_nmajor(qw_nk):
    E, N, Kp8 = qw_nk.shape
    u8 = qw_nk.view(torch.uint8).reshape(E, N, Kp8 * 4).contiguous()
    out = torch.empty_like(u8)
    for i in range(E):
        out[i] = implement_zp(u8[i])
    out = out.contiguous()
    out.xpu_fused_moe = True
    return out


def quantize_experts(W_fp16, GS):
    E, N, K = W_fp16.shape
    W_q = torch.empty(E, N, K // 8, dtype=torch.int32)
    W_s = torch.empty(E, N, K // GS, dtype=torch.float16)
    for e in range(E):
        w_e = W_fp16[e].float().contiguous()
        q_buf = torch.zeros(N, K // 8, dtype=torch.int32)
        s_buf = torch.zeros(N, K // GS, dtype=torch.float16)
        q, s = ggml_quantize_tensor(w_e, q_buf, s_buf, N, K, block_size=GS, transpose=False)
        W_q[e] = q; W_s[e] = s
    return W_q, W_s


def compare(name, a, b):
    diff = (a.float() - b.float())
    max_abs = diff.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
    status = "PASS" if cos > 0.9999 else "FAIL"
    print(f"  [{status}] {name}: max_abs={max_abs:.4e}  cos={cos:.6f}")
    return cos > 0.9999


# ═══ Quantize weights ═══
torch.manual_seed(42)
print("Quantizing...")
W13_fp16 = (torch.randn(E, TWO_I, H, dtype=torch.float32) * 0.02).to(torch.float16)
W13_q_ggml, W13_s_ggml = quantize_experts(W13_fp16, GS)
W2_fp16 = (torch.randn(E, H, I, dtype=torch.float32) * 0.02).to(torch.float16)
W2_q_ggml, W2_s_ggml = quantize_experts(W2_fp16, GS)

W13_q_ggml_xpu = W13_q_ggml.to(DEVICE); W13_s_ggml_xpu = W13_s_ggml.to(DEVICE)
W2_q_ggml_xpu = W2_q_ggml.to(DEVICE); W2_s_ggml_xpu = W2_s_ggml.to(DEVICE)

W13_q_ipex, W13_s_ipex = ggml_to_ipex_kmajor_marlin(W13_q_ggml_xpu, W13_s_ggml_xpu)
W2_q_ipex, W2_s_ipex = ggml_to_ipex_kmajor_marlin(W2_q_ggml_xpu, W2_s_ggml_xpu)

W13_q_cut = ggml_to_cutlass_nmajor(W13_q_ggml_xpu)
W13_s_cut = W13_s_ggml_xpu
W2_q_cut = ggml_to_cutlass_nmajor(W2_q_ggml_xpu)
W2_s_cut = W2_s_ggml_xpu

# Shared expert (FP16)
sgu_w = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
sd_w = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
sg_w = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
sgus_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)
sds_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)

# Router weights (GGML N-major for ESIMD moe_router_forward_int4)
print("Quantizing router...")
gate_fp16 = torch.randn(E, H, dtype=torch.float32) * 0.01
gate_fp16 = gate_fp16.to(torch.float16)
gate_q_buf = torch.zeros(E, H // 8, dtype=torch.int32)
gate_s_buf = torch.zeros(E, H // GS, dtype=torch.float16)
gate_q, gate_s = ggml_quantize_tensor(
    gate_fp16.float().contiguous(), gate_q_buf, gate_s_buf, E, H,
    block_size=GS, transpose=False)
# For ESIMD router: [E, H/8] int32 GGML N-major, [E, H/GS] fp16
gate_q_xpu = gate_q.to(DEVICE)
gate_s_xpu = gate_s.to(DEVICE)
# view as uint8 for GGML: [E, H/2]
gate_q_u8 = gate_q_xpu.view(torch.uint8).reshape(E, H // 8 * 4)


def shared_expert_forward(x):
    gate_up = x @ sgu_w.t()
    gate = gate_up[:, :I]; up = gate_up[:, I:]
    act = torch.nn.functional.silu(gate) * up
    down = act @ sd_w.t()
    gate_val = torch.sigmoid(x @ sg_w.t())
    return down * gate_val


print("\n═══ Test 1: Same logits → same output (sanity check) ═══")
for batch in [1, 4, 8]:
    torch.manual_seed(200 + batch)
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1
    logits = torch.randn(batch, E, dtype=DTYPE, device=DEVICE)

    out_old = moe_forward_full_int4(
        x, logits, W13_q_ipex, W13_s_ipex, sgu_w, sgus_empty,
        W2_q_ipex, W2_s_ipex, sd_w, sds_empty, sg_w, TK, 1, E, False)

    tw = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw, ti, tei, logits, True)

    routed = xpu_fused_moe(
        hidden_states=x, w13=W13_q_cut, w13_scales=W13_s_cut, w13_bias=None,
        w2=W2_q_cut, w2_scales=W2_s_cut, w2_bias=None,
        topk_weights=tw, topk_ids=ti, n_experts_per_token=TK,
        activation="silu", num_experts=E, is_int4=True)
    out_new = routed + shared_expert_forward(x)
    compare(f"batch={batch} same logits", out_old, out_new)


print("\n═══ Test 2: Router logits comparison ═══")
print("Compare ESIMD moe_router_forward_int4 vs FP16 reference router")
for batch in [1, 4, 8]:
    torch.manual_seed(300 + batch)
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1

    # ESIMD router (GGML layout)
    logits_esimd = moe_int4_ops.moe_router_forward_int4(
        x, gate_q_u8, gate_s_xpu, True)

    # FP16 reference router
    logits_fp16 = x @ gate_fp16.to(DEVICE).t()

    compare(f"batch={batch} router logits (ESIMD vs FP16)", logits_esimd, logits_fp16)

    # Check if they produce the same topk
    tw1 = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti1 = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei1 = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw1, ti1, tei1, logits_esimd, True)

    tw2 = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti2 = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei2 = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw2, ti2, tei2, logits_fp16, True)

    ids_match = (ti1.sort(dim=1).values == ti2.sort(dim=1).values).all().item()
    print(f"  TopK IDs match: {ids_match}")


print("\n═══ Test 3: Full pipeline with ESIMD router (old path end-to-end) ═══")
print("OLD: moe_forward_full_int4 (internal router from logits)")
print("NEW: ESIMD router → topk_softmax → xpu_fused_moe → shared")
print("Both use same x, router produces logits internally vs externally")
for batch in [1, 4, 8]:
    torch.manual_seed(400 + batch)
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1

    # Compute logits with ESIMD router (same as old path uses internally)
    logits_esimd = moe_int4_ops.moe_router_forward_int4(
        x, gate_q_u8, gate_s_xpu, True)

    # OLD: feed logits to moe_forward_full_int4
    out_old = moe_forward_full_int4(
        x, logits_esimd, W13_q_ipex, W13_s_ipex, sgu_w, sgus_empty,
        W2_q_ipex, W2_s_ipex, sd_w, sds_empty, sg_w, TK, 1, E, False)

    # NEW: same logits → topk → xpu_fused_moe → shared
    tw = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw, ti, tei, logits_esimd, True)

    routed = xpu_fused_moe(
        hidden_states=x, w13=W13_q_cut, w13_scales=W13_s_cut, w13_bias=None,
        w2=W2_q_cut, w2_scales=W2_s_cut, w2_bias=None,
        topk_weights=tw, topk_ids=ti, n_experts_per_token=TK,
        activation="silu", num_experts=E, is_int4=True)
    out_new = routed + shared_expert_forward(x)

    compare(f"batch={batch} ESIMD-router e2e", out_old, out_new)


print("\n═══ Test 4: Check what self.gate(x) produces in serving ═══")
print("In actual serving, self.gate is a quantized ReplicatedLinear (IPEX WoQ).")
print("The old path used moe_router_forward_int4 with GGML-layout weights.")
print("If self.gate(x) produces different logits, the routing changes → output changes.")
print("This would explain 'first token correct, subsequent wrong' if prefill")
print("modifies some state that affects decode routing.\n")

# Simulate: what if router logits are slightly different?
torch.manual_seed(500)
x = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.1
logits_a = moe_int4_ops.moe_router_forward_int4(x, gate_q_u8, gate_s_xpu, True)

# Perturb logits slightly (simulating different quantization)
logits_b = logits_a + torch.randn_like(logits_a) * 0.001

tw_a = torch.empty(1, TK, dtype=torch.float32, device=DEVICE)
ti_a = torch.empty(1, TK, dtype=torch.int32, device=DEVICE)
tei_a = torch.empty(1, TK, dtype=torch.int32, device=DEVICE)
torch.ops.torch_ipex.topk_softmax(tw_a, ti_a, tei_a, logits_a, True)

tw_b = torch.empty(1, TK, dtype=torch.float32, device=DEVICE)
ti_b = torch.empty(1, TK, dtype=torch.int32, device=DEVICE)
tei_b = torch.empty(1, TK, dtype=torch.int32, device=DEVICE)
torch.ops.torch_ipex.topk_softmax(tw_b, ti_b, tei_b, logits_b, True)

ids_match = (ti_a.sort(dim=1).values == ti_b.sort(dim=1).values).all().item()
print(f"  Tiny perturbation (0.001 std) → TopK IDs match: {ids_match}")
print(f"  TopK IDs old: {ti_a[0].tolist()}")
print(f"  TopK IDs new: {ti_b[0].tolist()}")

print("\nDone.")
