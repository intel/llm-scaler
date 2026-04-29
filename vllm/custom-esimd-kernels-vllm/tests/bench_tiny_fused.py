"""Benchmark tiny fused kernel vs IPEX vs C++ fused — 32-layer serving pattern."""
import torch, time, json, glob
import intel_extension_for_pytorch as ipex
from custom_esimd_kernels_vllm import moe_int4_ops
from custom_esimd_kernels_vllm.ops import (
    moe_forward_full_int4,
    moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared,
)
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
torch.serialization.add_safe_globals([getattr])

D = "xpu"; T = torch.float16; GS = 128
prefix = sorted(glob.glob("/llm/models/test/moe_dump/*_meta.json"))[0].replace("_meta.json", "")
meta = json.load(open(f"{prefix}_meta.json"))
def L(f):
    t = torch.load(f, map_location=D, weights_only=False)
    return t.data.contiguous() if hasattr(t, 'data') else t.contiguous()

w13 = L(f"{prefix}_w13.pt"); w13s = L(f"{prefix}_w13s.pt")
w2 = L(f"{prefix}_w2.pt"); w2s = L(f"{prefix}_w2s.pt")
gate_w = L(f"{prefix}_gate_w.pt"); gate_s = L(f"{prefix}_gate_s.pt")
shared_gu = L(f"{prefix}_shared_gu.pt"); shared_d = L(f"{prefix}_shared_d.pt")
shared_gate = L(f"{prefix}_shared_gate.pt")
E = meta["E"]; TK = meta["TK"]; H = meta["H"]
two_I = w13.size(1); I = two_I // 2
empty = torch.empty(0, dtype=T, device=D)

# IPEX weights (random, same shape)
W13i = torch.randint(-2**30, 2**30, (E, H//8, two_I), dtype=torch.int32, device=D)
W13si = (torch.rand(E, H//GS, two_I, device=D) * 0.04 + 0.002).to(T)
W2i = torch.randint(-2**30, 2**30, (E, I//8, H), dtype=torch.int32, device=D)
W2si = (torch.rand(E, I//GS, H, device=D) * 0.04 + 0.002).to(T)

dummy_w = torch.randn(H, H, dtype=T, device=D) * 0.001
N_LAYERS = 32; WARMUP = 10; ITERS = 50

def bench_step(moe_fn, M):
    x = torch.randn(M, H, dtype=T, device=D) * 0.1
    for _ in range(WARMUP):
        for _ in range(N_LAYERS):
            moe_fn(x)
            torch.mm(x, dummy_w)
        torch.xpu.synchronize()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for _ in range(N_LAYERS):
            moe_fn(x)
            torch.mm(x, dummy_w)
        torch.xpu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3

print(f"Real weights: E={E} H={H} I={I} TK={TK}")
print(f"32-layer step (ms) with dummy attention between layers:")
print(f"| M | IPEX(ms) | Tiny(ms) | C++fused(ms) | Tiny/IPEX |")
print(f"|--:|---------:|---------:|-------------:|----------:|")

for M in [1, 8, 64]:
    logits_buf = torch.randn(M, E, dtype=T, device=D)

    def _ipex(x):
        return moe_forward_full_int4(x, logits_buf, W13i, W13si, shared_gu, empty,
            W2i, W2si, shared_d, empty, shared_gate, TK, 1, E, False)

    def _tiny(x):
        lg = moe_int4_ops.moe_router_forward_int4(x, gate_w, gate_s, True)
        probs = torch.softmax(lg.float(), dim=-1)
        tw, ti = torch.topk(probs, TK, dim=-1)
        tw = (tw / tw.sum(dim=-1, keepdim=True)).to(T)
        ti = ti.to(torch.int32)
        return moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
            x, w13, w13s, w2, w2s, tw, ti, shared_gu, shared_d, shared_gate, 1)

    def _cpp(x):
        lg = moe_int4_ops.moe_router_forward_int4(x, gate_w, gate_s, True)
        return torch.ops.moe_prefill_gemm.moe_forward_full_fused_cutlass(
            x, lg, w13, w13s, w2, w2s, shared_gu, empty, shared_d, empty, shared_gate, E, TK)

    u_ipex = bench_step(_ipex, M)
    u_tiny = bench_step(_tiny, M)
    u_cpp = bench_step(_cpp, M)
    print(f"| {M:2d} | {u_ipex:8.1f} | {u_tiny:8.1f} | {u_cpp:12.1f} | {u_tiny/u_ipex:.3f}x |")

print("\nDone.")
