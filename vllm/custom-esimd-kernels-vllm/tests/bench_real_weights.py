"""Benchmark with real dumped weights at various M values."""
import torch, time, json, glob, intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from custom_esimd_kernels_vllm import moe_int4_ops
from custom_esimd_kernels_vllm.ops import moe_forward_full_int4
torch.ops.load_library("/llm/models/test/llm-scaler/vllm/moe_prefill_int4/build/libmoe_prefill_gemm_int4.so")

D = "xpu"; T = torch.float16
dump_dir = "/llm/models/test/moe_dump"
metas = sorted(glob.glob(f"{dump_dir}/*_meta.json"))
prefix = metas[0].replace("_meta.json", "")
meta = json.load(open(f"{prefix}_meta.json"))

def L(f):
    t = torch.load(f, map_location=D, weights_only=False)
    if hasattr(t, 'data'):
        t = t.data
    return t.contiguous()

w13 = L(f"{prefix}_w13.pt")
w13s = L(f"{prefix}_w13s.pt")
w2 = L(f"{prefix}_w2.pt")
w2s = L(f"{prefix}_w2s.pt")
gate_w = L(f"{prefix}_gate_w.pt")
gate_s = L(f"{prefix}_gate_s.pt")
shared_gu = L(f"{prefix}_shared_gu.pt")
shared_d = L(f"{prefix}_shared_d.pt")
shared_gate = L(f"{prefix}_shared_gate.pt")

E = meta["E"]; TK = meta["TK"]; H = meta["H"]
empty = torch.empty(0, dtype=T, device=D)

print(f"Real weights from {prefix}")
print(f"w13={list(w13.shape)} {w13.dtype}")
print(f"w2={list(w2.shape)} {w2.dtype}")
print(f"shared_gu={list(shared_gu.shape)} {shared_gu.dtype}")
print(f"shared_d={list(shared_d.shape)} {shared_d.dtype}")
print(f"gate_w={list(gate_w.shape)} {gate_w.dtype}")

# Check if we also need IPEX format weights for old path comparison
# w13 is CUTLASS N-major uint8, need K-major int32 for IPEX
has_ipex = (w13.dtype == torch.int32 and w13.dim() == 3 and w13.size(1) == H // 8)
print(f"\nIPEX format available: {has_ipex}")

def b(fn):
    for _ in range(30):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / 200 * 1e6

print(f"\n== NEW (CUTLASS fused) ==")
print(f"| M  | router(us) | fused(us) | total(us) | 32L(ms) |")
print(f"|---:|-----------:|----------:|----------:|--------:|")

for M in [1, 2, 4, 8, 16, 32, 64]:
    x = torch.randn(M, H, dtype=T, device=D) * 0.1
    u_r = b(lambda: moe_int4_ops.moe_router_forward_int4(x, gate_w, gate_s, True))
    lg = moe_int4_ops.moe_router_forward_int4(x, gate_w, gate_s, True)
    u_f = b(lambda: torch.ops.moe_prefill_gemm.moe_forward_full_fused_cutlass(
        x, lg, w13, w13s, w2, w2s, shared_gu, empty, shared_d, empty, shared_gate, E, TK))
    print(f"| {M:2d} | {u_r:10.0f} | {u_f:9.0f} | {u_r+u_f:9.0f} | {(u_r+u_f)*32/1000:7.1f} |")
    del x
    torch.xpu.empty_cache()

# If IPEX weights available, compare
if has_ipex:
    print(f"\n== OLD (IPEX fused ESIMD) ==")
    print(f"| M  | total(us) | 32L(ms) |")
    print(f"|---:|----------:|--------:|")
    for M in [1, 2, 4, 8, 16, 32, 64]:
        x = torch.randn(M, H, dtype=T, device=D) * 0.1
        logits = torch.randn(M, E, dtype=T, device=D)
        u = b(lambda: moe_forward_full_int4(
            x, logits, w13, w13s, shared_gu, empty, w2, w2s, shared_d, empty, shared_gate, TK, 1, E, False))
        print(f"| {M:2d} | {u:9.0f} | {u*32/1000:7.1f} |")
        del x, logits
        torch.xpu.empty_cache()

print("\nDone.")
