import torch, sys, pathlib
import intel_extension_for_pytorch
torch.ops.load_library("/llm/models/test/llm-scaler/vllm/moe_prefill_int4/build/libmoe_prefill_gemm_int4.so")
sys.path.insert(0, str(pathlib.Path("/llm/models/test/llm-scaler/vllm/moe_prefill_int4/tests")))
from test_gemm_marlin_format import marlin_to_nmajor
sys.path.insert(0, str(pathlib.Path("/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests")))
from test_moe_prefill_int4 import quantize_experts_ipex, GROUP_SIZE
from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

DEVICE = "xpu"
H, I, E, TK, GS = 3072, 1024, 64, 8, 128
TWO_I = 2*I; M = 512; total = M*TK

torch.manual_seed(0)
W13 = (torch.randn(E, TWO_I, H, dtype=torch.float32) * 0.02).to(torch.float16)
W13_q, W13_s, _ = quantize_experts_ipex(W13, GS)
W13_q = W13_q.to(DEVICE); W13_s = W13_s.to(DEVICE)

# Proven correct: convert to N-major
W13_nm, W13_s_nm = marlin_to_nmajor(W13_q, W13_s)

# Routing
torch.manual_seed(42)
tw = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
ti = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
tei = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
logits = torch.randn(M, E, dtype=torch.float16, device=DEVICE)
torch.ops.torch_ipex.topk_softmax(tw, ti, tei, logits, True)
off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
efto = torch.zeros(E+1, dtype=torch.int64, device=DEVICE)
efto[:E] = off.to(torch.int64); efto[E] = total
x_perm = ops.moe_prefill_scatter_x_forward(
    torch.randn(M, H, dtype=torch.float16, device=DEVICE)*0.1, tok, TK)

# IPEX reference
out_ipex = torch.empty(total, TWO_I, dtype=torch.float16, device=DEVICE)
torch.ops.torch_ipex.group_mm_int4_out_marlin(
    out_ipex, x_perm, W13_q, W13_s, None, rows, None, GS)

# N-major (proven bit-exact with IPEX)
out_nm = torch.empty(total, TWO_I, dtype=torch.float16, device=DEVICE)
torch.ops.moe_prefill_gemm.grouped_gemm_int4(
    x_perm, W13_nm, W13_s_nm, None, out_nm, efto, TWO_I, H, E, True, False, False)

# K-major with in-place unshuffle/reshuffle
# Note: W13_q gets modified in-place by unshuffle then restored by reshuffle
W13_q_clone = W13_q.clone()  # save original to verify reshuffle restores it
out_km = torch.empty(total, TWO_I, dtype=torch.float16, device=DEVICE)
torch.ops.moe_prefill_gemm.grouped_gemm_int4(
    x_perm, W13_q, W13_s, None, out_km, efto, TWO_I, H, E, True, False, True)

# Check if reshuffle restored the original weights
restore_ok = (W13_q == W13_q_clone).all().item()
print(f"Weight restored after reshuffle: {restore_ok}")

d_nm_ipex = (out_ipex.float()-out_nm.float()).abs().max().item()
d_km_ipex = (out_ipex.float()-out_km.float()).abs().max().item()
d_nm_km = (out_nm.float()-out_km.float()).abs().max().item()
print(f"IPEX vs N-major: {d_nm_ipex:.4e}  (should be 0)")
print(f"IPEX vs K-major: {d_km_ipex:.4e}")
print(f"N-major vs K-major: {d_nm_km:.4e}")
