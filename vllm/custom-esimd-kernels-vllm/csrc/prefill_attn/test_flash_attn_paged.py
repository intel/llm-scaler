"""Task 8: Paged KV Cache correctness test.

Tests:
  1. Contiguous mode (block_table empty) — should match flash_attn_opt
  2. Paged mode with sequential page allocation — should match contiguous
  3. Paged mode with shuffled pages — should still match

Usage:
  USE_DOUBLE_GRF=1 KERNEL=flash_attn_paged TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  ZE_AFFINITY_MASK=4 python3 test_flash_attn_paged.py
"""
import ctypes, glob, os, sys, torch, time
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
so = glob.glob(os.path.join(script_dir, "flash_attn_paged*.so"))
if not so:
    print("Build first")
    sys.exit(1)
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_paged.forward
device = torch.device("xpu:0")

PAGE_SIZE = 64


def reference_causal(Q, K, V, q_start_pos=0):
    qh, kvh = Q.size(1), K.size(1)
    ql, kvl = Q.size(0), K.size(0)
    r = qh // kvh
    K_e = K.repeat_interleave(r, dim=1)
    V_e = V.repeat_interleave(r, dim=1)
    q = Q.permute(1, 0, 2).unsqueeze(0).float()
    k = K_e.permute(1, 0, 2).unsqueeze(0).float()
    v = V_e.permute(1, 0, 2).unsqueeze(0).float()
    if q_start_pos == 0 and ql == kvl:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(0).permute(1, 0, 2).half()
    mask = torch.zeros(ql, kvl, device=device, dtype=torch.float32)
    for i in range(ql):
        valid = min(q_start_pos + i + 1, kvl)
        mask[i, valid:] = float('-inf')
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1, 0, 2).half()


def make_paged_kv(K_contig, V_contig, page_size, shuffle=False):
    """Convert contiguous [kv_len, num_kv_heads, HD] to paged layout."""
    kv_len = K_contig.size(0)
    num_kv_heads = K_contig.size(1)
    num_pages = (kv_len + page_size - 1) // page_size

    # Allocate extra blocks to test non-sequential physical allocation
    num_physical_blocks = num_pages * 2 if shuffle else num_pages
    K_paged = torch.zeros(num_physical_blocks, page_size, num_kv_heads, 256,
                          device=device, dtype=torch.float16)
    V_paged = torch.zeros(num_physical_blocks, page_size, num_kv_heads, 256,
                          device=device, dtype=torch.float16)

    # Build block table
    if shuffle:
        perm = torch.randperm(num_physical_blocks)[:num_pages]
    else:
        perm = torch.arange(num_pages)
    block_table = perm.to(torch.int32).to(device)

    # Fill pages
    for logical_block in range(num_pages):
        physical_block = perm[logical_block].item()
        start = logical_block * page_size
        end = min(start + page_size, kv_len)
        length = end - start
        K_paged[physical_block, :length] = K_contig[start:end]
        V_paged[physical_block, :length] = V_contig[start:end]

    return K_paged, V_paged, block_table


print(f"Device: {torch.xpu.get_device_name(0)}")

# === Test 1: Contiguous mode ===
print("\n=== Test 1: Contiguous Mode (empty block_table) ===")
pass1 = True
cases = [(64, 64, 12, 2, 0), (128, 1024, 12, 2, 896), (256, 2048, 12, 2, 1792)]
for ql, kvl, qh, kvh, qsp in cases:
    torch.manual_seed(ql + kvl + qsp)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    empty_bt = torch.empty(0, dtype=torch.int32, device=device)

    actual = fwd(Q, K, V, empty_bt, kvl, qsp, PAGE_SIZE)
    expected = reference_causal(Q, K, V, qsp)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass1 = False
    print(f"  Q={ql:3d} KV={kvl:4d} qsp={qsp:4d}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass1 else 'FAILED'}")

# === Test 2: Paged mode (sequential pages) ===
print("\n=== Test 2: Paged Mode (sequential allocation) ===")
pass2 = True
cases = [(64, 64, 12, 2, 0), (128, 1024, 12, 2, 896), (256, 2048, 12, 2, 1792)]
for ql, kvl, qh, kvh, qsp in cases:
    torch.manual_seed(ql + kvl + qsp)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K_contig = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V_contig = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)

    K_paged, V_paged, block_table = make_paged_kv(K_contig, V_contig, PAGE_SIZE, shuffle=False)
    actual = fwd(Q, K_paged, V_paged, block_table, kvl, qsp, PAGE_SIZE)
    expected = reference_causal(Q, K_contig, V_contig, qsp)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass2 = False
    print(f"  Q={ql:3d} KV={kvl:4d} qsp={qsp:4d}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass2 else 'FAILED'}")

# === Test 3: Paged mode (shuffled pages) ===
print("\n=== Test 3: Paged Mode (shuffled pages) ===")
pass3 = True
cases = [(64, 128, 12, 2, 64), (128, 1024, 12, 2, 896), (256, 2048, 12, 2, 1792)]
for ql, kvl, qh, kvh, qsp in cases:
    torch.manual_seed(ql + kvl + qsp + 777)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K_contig = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V_contig = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)

    K_paged, V_paged, block_table = make_paged_kv(K_contig, V_contig, PAGE_SIZE, shuffle=True)
    actual = fwd(Q, K_paged, V_paged, block_table, kvl, qsp, PAGE_SIZE)
    expected = reference_causal(Q, K_contig, V_contig, qsp)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass3 = False
    print(f"  Q={ql:3d} KV={kvl:4d} qsp={qsp:4d} (shuffled): rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass3 else 'FAILED'}")

# === Performance ===
print("\n=== Performance (paged, shuffled) ===")
bench = [(128, 1024, 12, 2, 896), (256, 4096, 12, 2, 3840)]
for ql, kvl, qh, kvh, qsp in bench:
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K_contig = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V_contig = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    K_paged, V_paged, block_table = make_paged_kv(K_contig, V_contig, PAGE_SIZE, shuffle=True)

    N = 5 if kvl >= 4096 else 10
    for _ in range(3):
        fwd(Q, K_paged, V_paged, block_table, kvl, qsp, PAGE_SIZE)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fwd(Q, K_paged, V_paged, block_table, kvl, qsp, PAGE_SIZE)
    torch.xpu.synchronize()
    t = (time.perf_counter() - t0) / N
    print(f"  Q={ql:3d} KV={kvl:4d} (paged): {t*1000:.2f}ms")

overall = pass1 and pass2 and pass3
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
