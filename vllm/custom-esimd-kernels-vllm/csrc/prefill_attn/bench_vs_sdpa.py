"""Benchmark: ESIMD Task 3 (SLM) vs torch SDPA."""
import ctypes, glob, os, torch, time
import torch.nn.functional as F

so3 = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn_slm*.so"))
ctypes.CDLL(so3[0])
fwd3 = torch.ops.flash_attn_slm.forward
device = torch.device("xpu:0")


def bench_sdpa(q_t, k_t, v_t, N=10):
    q = q_t.unsqueeze(0).unsqueeze(0)
    k = k_t.unsqueeze(0).unsqueeze(0)
    v = v_t.unsqueeze(0).unsqueeze(0)
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N


def bench_esimd(q_t, k_t, v_t, N=10):
    for _ in range(3):
        fwd3(q_t, k_t, v_t)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fwd3(q_t, k_t, v_t)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N


print(f"Device: {torch.xpu.get_device_name(0)}")
print(f"{'Q':>5} {'KV':>6} {'SDPA':>10} {'ESIMD':>10} {'ESIMD/SDPA':>12}")
print("-" * 48)

cases = [(32, 256), (32, 1024), (128, 1024), (256, 1024),
         (256, 2048), (512, 2048), (512, 4096)]

for ql, kvl in cases:
    q_t = torch.randn(ql, 256, device=device, dtype=torch.float16)
    k_t = torch.randn(kvl, 256, device=device, dtype=torch.float16)
    v_t = torch.randn(kvl, 256, device=device, dtype=torch.float16)
    N = 5 if kvl >= 4096 else 10
    t_sdpa = bench_sdpa(q_t, k_t, v_t, N)
    t_esimd = bench_esimd(q_t, k_t, v_t, N)
    ratio = t_esimd / t_sdpa
    print(f"{ql:5d} {kvl:6d} {t_sdpa*1000:8.2f}ms {t_esimd*1000:8.2f}ms {ratio:10.1f}x")
