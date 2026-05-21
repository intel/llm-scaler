"""Task 4: correctness + benchmark vs SDPA."""
import ctypes, glob, os, torch, time
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
so = glob.glob(os.path.join(script_dir, "flash_attn_vec*.so"))
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_vec.forward
device = torch.device("xpu:0")


def ref(Q, K, V):
    q = Q.unsqueeze(0).unsqueeze(0).float()
    k = K.unsqueeze(0).unsqueeze(0).float()
    v = V.unsqueeze(0).unsqueeze(0).float()
    return F.scaled_dot_product_attention(q, k, v, is_causal=False).squeeze(0).squeeze(0).half()


print(f"Device: {torch.xpu.get_device_name(0)}")

# === Correctness ===
print("\n=== Correctness ===")
cases = [(32, 32), (32, 128), (64, 256), (128, 512), (256, 1024), (512, 2048), (31, 100), (33, 65)]
all_pass = True
for ql, kvl in cases:
    torch.manual_seed(ql * 100 + kvl)
    Q = torch.randn(ql, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V)
    expected = ref(Q, K, V)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok:
        all_pass = False
    print(f"  Q={ql:3d} KV={kvl:4d}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  Overall: {'ALL PASS' if all_pass else 'FAILED'}")

# === Performance vs SDPA ===
print("\n=== Performance: Task4 vs SDPA ===")
bench = [(32, 256), (32, 1024), (128, 1024), (256, 1024), (256, 2048), (512, 4096)]
print(f"{'Q':>5} {'KV':>6} {'SDPA':>10} {'Task4':>10} {'Ratio':>8}")
print("-" * 45)

for ql, kvl in bench:
    Q = torch.randn(ql, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, 256, device=device, dtype=torch.float16)
    N = 5 if kvl >= 4096 else 10

    # SDPA
    q_b = Q.unsqueeze(0).unsqueeze(0)
    k_b = K.unsqueeze(0).unsqueeze(0)
    v_b = V.unsqueeze(0).unsqueeze(0)
    for _ in range(3):
        F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=False)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=False)
    torch.xpu.synchronize()
    t_sdpa = (time.perf_counter() - t0) / N

    # Task 4
    for _ in range(3):
        fwd(Q, K, V)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fwd(Q, K, V)
    torch.xpu.synchronize()
    t_esimd = (time.perf_counter() - t0) / N

    ratio = t_esimd / t_sdpa
    print(f"{ql:5d} {kvl:6d} {t_sdpa*1000:8.2f}ms {t_esimd*1000:8.2f}ms {ratio:7.1f}x")
