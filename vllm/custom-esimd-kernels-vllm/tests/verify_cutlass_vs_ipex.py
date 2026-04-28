"""Verify CUTLASS N-major uint8 GEMM matches IPEX K-major marlin GEMM.

Uses the SAME quantized weights (from ggml_quantize_tensor), converts to
both IPEX K-major marlin and CUTLASS N-major uint8, then compares outputs.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=0 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/verify_cutlass_vs_ipex.py"
"""
import sys, pathlib, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import implement_zp

sys.path.insert(0, str(pathlib.Path("/llm/models/test/llm-scaler-vllm-xpu")))
from vllm.model_executor.layers.quantization.sym_int4 import ggml_quantize_tensor

DEVICE = "xpu"
DTYPE = torch.float16
_MARLIN_SHUFFLED_IDX = (0, 4, 1, 5, 2, 6, 3, 7)
GS = 128


def ggml_to_ipex_kmajor_marlin(qweight_nk, scales_nk):
    """[E, N, K/8] int32 → [E, K/8, N] int32 marlin-shuffled + [E, K/GS, N]"""
    qw_t = qweight_nk.permute(0, 2, 1).contiguous()
    sc_t = scales_nk.permute(0, 2, 1).contiguous()
    qw = qw_t.to(torch.int64) & 0xFFFFFFFF
    shuffled = torch.zeros_like(qw)
    for new_pos in range(8):
        old_pos = _MARLIN_SHUFFLED_IDX[new_pos]
        nibble = (qw >> (old_pos * 4)) & 0xF
        shuffled |= nibble << (new_pos * 4)
    return (shuffled & 0xFFFFFFFF).to(torch.int32).contiguous(), sc_t


def ggml_to_cutlass_nmajor(qweight_nk):
    """[E, N, K/8] int32 → [E, N, K/2] uint8 + implement_zp"""
    E, N, Kp8 = qweight_nk.shape
    u8 = qweight_nk.view(torch.uint8).reshape(E, N, Kp8 * 4).contiguous()
    out = torch.empty_like(u8)
    for i in range(E):
        out[i] = implement_zp(u8[i])
    out = out.contiguous()
    out.xpu_fused_moe = True
    return out


def test_gemm_accuracy(E, N, K, M, label):
    print(f"\n{'='*60}")
    print(f"Test: {label}  E={E} N={N} K={K} GS={GS} M={M}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    W_fp16 = (torch.randn(E, N, K, dtype=torch.float32) * 0.5).to(torch.float16)

    # ── Quantize all experts with ggml (single source of truth) ──
    W_q_ggml = torch.empty(E, N, K // 8, dtype=torch.int32)
    W_s_ggml = torch.empty(E, N, K // GS, dtype=torch.float16)
    for e in range(E):
        w_e = W_fp16[e].float().contiguous()
        q_buf = torch.zeros(N, K // 8, dtype=torch.int32)
        s_buf = torch.zeros(N, K // GS, dtype=torch.float16)
        q, s = ggml_quantize_tensor(w_e, q_buf, s_buf, N, K, block_size=GS, transpose=False)
        W_q_ggml[e] = q
        W_s_ggml[e] = s

    W_q_ggml_xpu = W_q_ggml.to(DEVICE)
    W_s_ggml_xpu = W_s_ggml.to(DEVICE)

    # ── IPEX K-major marlin (from same ggml) ──
    W_q_ipex, W_s_ipex = ggml_to_ipex_kmajor_marlin(W_q_ggml_xpu, W_s_ggml_xpu)

    # ── CUTLASS N-major uint8 (from same ggml) ──
    W_q_cut = ggml_to_cutlass_nmajor(W_q_ggml_xpu)
    W_s_cut = W_s_ggml_xpu  # N-major [E, N, K/GS] stays as-is

    # ── Routing: equal tokens per expert ──
    TK = 8
    total = M * TK
    tokens_per_expert = total // E
    remainder = total % E

    rows = torch.full((E,), tokens_per_expert, dtype=torch.int32, device=DEVICE)
    if remainder > 0:
        rows[:remainder] += 1

    efto = torch.zeros(E + 1, dtype=torch.int64, device=DEVICE)
    efto[1:] = torch.cumsum(rows.to(torch.int64), dim=0)
    actual_total = int(efto[E].item())

    # ── Input ──
    torch.manual_seed(123)
    x = torch.randn(actual_total, K, dtype=DTYPE, device=DEVICE) * 0.1

    # ── IPEX GEMM ──
    out_ipex = torch.empty(actual_total, N, dtype=DTYPE, device=DEVICE)
    torch.ops.torch_ipex.group_mm_int4_out_marlin(
        out_ipex, x, W_q_ipex, W_s_ipex, None, rows, None, GS)

    # ── CUTLASS GEMM ──
    out_cut = torch.empty(actual_total, N, dtype=DTYPE, device=DEVICE)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=x, ptr_B=W_q_cut, ptr_scales=W_s_cut, ptr_bias=None,
        ptr_D=out_cut, expert_first_token_offset=efto,
        N=N, K=K, num_experts=E,
        is_B_int4=True, is_B_mxfp4=False)

    # ── FP16 reference ──
    W_fp16_dev = W_fp16.to(DEVICE)
    out_ref = torch.zeros(actual_total, N, dtype=DTYPE, device=DEVICE)
    offset = 0
    for e in range(E):
        r = int(rows[e].item())
        if r > 0:
            out_ref[offset:offset + r] = x[offset:offset + r] @ W_fp16_dev[e].t()
            offset += r

    # ── Compare ──
    def compare(name, a, b):
        diff = (a.float() - b.float())
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
        print(f"  {name}: max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}  cosine={cos:.6f}")
        return cos

    cos_ipex_ref = compare("IPEX vs FP16 ref", out_ipex, out_ref)
    cos_cut_ref = compare("CUTLASS vs FP16 ref", out_cut, out_ref)
    cos_ipex_cut = compare("IPEX vs CUTLASS", out_ipex, out_cut)

    ok = cos_ipex_cut > 0.999
    print(f"  PASS: {'YES' if ok else 'NO'} (IPEX≈CUTLASS cosine > 0.999)")
    return ok


# ── Run tests ──
results = []
results.append(test_gemm_accuracy(E=4, N=64, K=128, M=8, label="tiny"))
results.append(test_gemm_accuracy(E=8, N=2048, K=3072, M=64, label="122B W13"))
results.append(test_gemm_accuracy(E=8, N=3072, K=1024, M=64, label="122B W2"))

print(f"\n{'='*60}")
print(f"Summary: {sum(results)}/{len(results)} tests passed")
if all(results):
    print("ALL PASS")
else:
    print("SOME FAILED")
