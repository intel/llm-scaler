"""Verify decode accuracy: old ESIMD path vs new CUTLASS path.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=0 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/verify_decode_accuracy.py"
"""
import sys, pathlib, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe, implement_zp
from custom_esimd_kernels_vllm.ops import moe_forward_full_int4

sys.path.insert(0, str(pathlib.Path("/llm/models/test/llm-scaler-vllm-xpu")))
from vllm.model_executor.layers.quantization.sym_int4 import ggml_quantize_tensor

DEVICE = "xpu"
DTYPE = torch.float16
GS = 128
_MARLIN_SHUFFLED_IDX = (0, 4, 1, 5, 2, 6, 3, 7)


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


def compare(name, a, b):
    diff = (a.float() - b.float())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
    status = "PASS" if cos > 0.9999 else "FAIL"
    print(f"  [{status}] {name}: max_abs={max_abs:.4e}  mean={mean_abs:.4e}  cos={cos:.6f}")
    return cos > 0.9999


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


H, I, E, TK = 3072, 1024, 256, 8
TWO_I = 2 * I

torch.manual_seed(42)
print("Quantizing W13...")
W13_fp16 = (torch.randn(E, TWO_I, H, dtype=torch.float32) * 0.02).to(torch.float16)
W13_q_ggml, W13_s_ggml = quantize_experts(W13_fp16, GS)

print("Quantizing W2...")
W2_fp16 = (torch.randn(E, H, I, dtype=torch.float32) * 0.02).to(torch.float16)
W2_q_ggml, W2_s_ggml = quantize_experts(W2_fp16, GS)

print("Converting formats...")
W13_q_ggml_xpu = W13_q_ggml.to(DEVICE); W13_s_ggml_xpu = W13_s_ggml.to(DEVICE)
W2_q_ggml_xpu = W2_q_ggml.to(DEVICE); W2_s_ggml_xpu = W2_s_ggml.to(DEVICE)

W13_q_ipex, W13_s_ipex = ggml_to_ipex_kmajor_marlin(W13_q_ggml_xpu, W13_s_ggml_xpu)
W2_q_ipex, W2_s_ipex = ggml_to_ipex_kmajor_marlin(W2_q_ggml_xpu, W2_s_ggml_xpu)

W13_q_cut = ggml_to_cutlass_nmajor(W13_q_ggml_xpu)
W13_s_cut = W13_s_ggml_xpu
W2_q_cut = ggml_to_cutlass_nmajor(W2_q_ggml_xpu)
W2_s_cut = W2_s_ggml_xpu

sgu_w = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
sd_w = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
sg_w = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
sgus_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)
sds_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)


def shared_expert_forward(x):
    gate_up = x @ sgu_w.t()
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    act = torch.nn.functional.silu(gate) * up
    down = act @ sd_w.t()
    gate_val = torch.sigmoid(x @ sg_w.t())
    return down * gate_val


all_pass = True
for batch in [1, 2, 4, 8, 16, 32]:
    print(f"\n{'='*60}")
    print(f"Batch = {batch}")
    print(f"{'='*60}")
    M = batch
    torch.manual_seed(100 + batch)
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE) * 0.1
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

    # OLD: moe_forward_full_int4 (fused routed + shared)
    out_old = moe_forward_full_int4(
        x, logits,
        W13_q_ipex, W13_s_ipex,
        sgu_w, sgus_empty,
        W2_q_ipex, W2_s_ipex,
        sd_w, sds_empty,
        sg_w, TK, 1, E, False)

    # NEW: topk → xpu_fused_moe → shared_expert
    tw = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
    ti = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    tei = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw, ti, tei, logits, True)

    routed = xpu_fused_moe(
        hidden_states=x,
        w13=W13_q_cut, w13_scales=W13_s_cut, w13_bias=None,
        w2=W2_q_cut, w2_scales=W2_s_cut, w2_bias=None,
        topk_weights=tw, topk_ids=ti,
        n_experts_per_token=TK, activation="silu",
        num_experts=E, is_int4=True)

    shared = shared_expert_forward(x)
    out_new = routed + shared

    ok = compare("OLD vs NEW (full)", out_old, out_new)
    all_pass = all_pass and ok

print(f"\n{'='*60}")
print(f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
