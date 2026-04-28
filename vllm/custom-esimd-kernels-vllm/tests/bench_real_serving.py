"""Benchmark that matches real serving execution pattern.

Key insight: serving doesn't run MoE in a tight loop. Between each MoE layer
there's attention + norm + all_reduce + Python dispatch. This breaks GPU queue
pipelining, so each MoE layer effectively starts from a "cold" queue.

To simulate this, we sync after EACH layer call, not after all layers.
This matches how profiling shows ~330us/layer instead of benchmark's ~134us/layer.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=7 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/bench_real_serving.py"
"""
import torch, time, json, glob, gc
import intel_extension_for_pytorch as ipex
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
    if hasattr(t, 'data'): t = t.data
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

# Also make IPEX K-major weights for old path comparison
# w13 is [256, 512, 1536] uint8 = [E, 2*I, K/2]
two_I = w13.size(1)  # 512
K_w13 = w13.size(2) * 2  # 3072
I = two_I // 2  # 256
w13_ipex = torch.randint(-2**30, 2**30, (E, K_w13 // 8, two_I), dtype=torch.int32, device=D)
w13s_ipex = (torch.rand(E, K_w13 // 128, two_I, device=D) * 0.04 + 0.002).to(T)
K_w2 = w2.size(2) * 2  # 256
N_w2 = w2.size(1)  # 3072
w2_ipex = torch.randint(-2**30, 2**30, (E, K_w2 // 8, N_w2), dtype=torch.int32, device=D)
w2s_ipex = (torch.rand(E, K_w2 // 128, N_w2, device=D) * 0.04 + 0.002).to(T)

print(f"Real weights: w13={list(w13.shape)}, w2={list(w2.shape)}")
print(f"  E={E}, TK={TK}, H={H}, I={I}, 2*I={two_I}")
print()

WARMUP = 10
ITERS = 50
N_LAYERS = 32

# Simulate a dummy "other work" between MoE layers (attention, norm, etc.)
# This is a small matmul that breaks GPU queue pipelining
dummy_w = torch.randn(H, H, dtype=T, device=D) * 0.001

def bench_serving_pattern(moe_fn, M, label):
    """Run N_LAYERS MoE calls with dummy work between each.
    Sync once per full decode step (not per layer)."""
    x = torch.randn(M, H, dtype=T, device=D) * 0.1

    # Warmup
    for _ in range(WARMUP):
        for _ in range(N_LAYERS):
            moe_fn(x)
            torch.mm(x, dummy_w)  # simulate attention/norm between layers
        torch.xpu.synchronize()

    # Measure: sync per decode step (realistic)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for _ in range(N_LAYERS):
            moe_fn(x)
            torch.mm(x, dummy_w)
        torch.xpu.synchronize()
    total_step = (time.perf_counter() - t0) / ITERS * 1e3  # ms per step

    # Measure: sync per layer (matches profiling)
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    for _ in range(ITERS):
        for _ in range(N_LAYERS):
            moe_fn(x)
            torch.mm(x, dummy_w)
            torch.xpu.synchronize()
    total_sync = (time.perf_counter() - t1) / ITERS * 1e3

    # Measure: MoE only, sync per layer (isolate MoE cost)
    torch.xpu.synchronize()
    t2 = time.perf_counter()
    for _ in range(ITERS):
        for _ in range(N_LAYERS):
            moe_fn(x)
            torch.xpu.synchronize()
    moe_only_sync = (time.perf_counter() - t2) / ITERS * 1e3

    # Measure: tight loop (benchmark style)
    torch.xpu.synchronize()
    t3 = time.perf_counter()
    for _ in range(ITERS * N_LAYERS):
        moe_fn(x)
    torch.xpu.synchronize()
    tight = (time.perf_counter() - t3) / (ITERS * N_LAYERS) * 1e6  # us per layer

    del x; torch.xpu.empty_cache()
    return total_step, total_sync, moe_only_sync, tight

print("=" * 80)
print(f"| M | method | step(ms) | sync/L(ms) | moe_sync(ms) | tight(us/L) |")
print(f"|--:|--------|--------:|-----------:|-------------:|------------:|")

for M in [1, 8, 64]:
    # NEW: CUTLASS fused
    def new_moe(x):
        lg = moe_int4_ops.moe_router_forward_int4(x, gate_w, gate_s, True)
        return torch.ops.moe_prefill_gemm.moe_forward_full_fused_cutlass(
            x, lg, w13, w13s, w2, w2s, shared_gu, empty, shared_d, empty, shared_gate, E, TK)

    s1, s2, s3, s4 = bench_serving_pattern(new_moe, M, "NEW")
    print(f"| {M:2d} | NEW    | {s1:7.1f} | {s2:10.1f} | {s3:12.1f} | {s4:11.0f} |")

    # OLD: IPEX fused
    logits_buf = torch.randn(M, E, dtype=T, device=D)
    def old_moe(x):
        return moe_forward_full_int4(
            x, logits_buf, w13_ipex, w13s_ipex, shared_gu, empty,
            w2_ipex, w2s_ipex, shared_d, empty, shared_gate, TK, 1, E, False)

    s1, s2, s3, s4 = bench_serving_pattern(old_moe, M, "OLD")
    print(f"| {M:2d} | OLD    | {s1:7.1f} | {s2:10.1f} | {s3:12.1f} | {s4:11.0f} |")

    del logits_buf; torch.xpu.empty_cache(); gc.collect()

print()
print("step    = 32 layers + dummy work, sync per step (closest to real serving)")
print("sync/L  = 32 layers + dummy work, sync per layer")
print("moe_sync= 32 layers MoE only, sync per layer (matches profiling)")
print("tight   = tight loop per layer (standard benchmark)")
