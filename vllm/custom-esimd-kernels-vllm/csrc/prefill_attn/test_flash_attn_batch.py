"""Task 9: Variable-length batch correctness test.

Tests:
  1. Single seq (batch=1) — should match flash_attn_paged
  2. Multiple seqs with same q_len/kv_len
  3. Multiple seqs with different q_len/kv_len

Usage:
  USE_DOUBLE_GRF=1 KERNEL=flash_attn_batch TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  ZE_AFFINITY_MASK=4 python3 test_flash_attn_batch.py
"""
import ctypes, glob, os, sys, torch, time
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
so = glob.glob(os.path.join(script_dir, "flash_attn_batch*.so"))
if not so:
    print("Build first")
    sys.exit(1)
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_batch.forward
device = torch.device("xpu:0")

PAGE_SIZE = 64
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2


def reference_single_seq(Q, K_contig, V_contig, q_start_pos):
    """Reference for single seq: Q [ql, qh, 256], K/V [kvl, kvh, 256]."""
    qh, kvh = Q.size(1), K_contig.size(1)
    ql, kvl = Q.size(0), K_contig.size(0)
    r = qh // kvh
    K_e = K_contig.repeat_interleave(r, dim=1)
    V_e = V_contig.repeat_interleave(r, dim=1)
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


def make_batch_inputs(seq_configs):
    """Build batched inputs from list of (q_len, kv_len, q_start_pos) tuples."""
    num_seqs = len(seq_configs)
    total_q = sum(ql for ql, _, _ in seq_configs)
    max_kvl = max(kvl for _, kvl, _ in seq_configs)
    max_blocks = (max_kvl + PAGE_SIZE - 1) // PAGE_SIZE

    # Allocate enough blocks for all seqs
    total_blocks = sum((kvl + PAGE_SIZE - 1) // PAGE_SIZE for _, kvl, _ in seq_configs)
    K_cache = torch.zeros(total_blocks * 2, PAGE_SIZE, NUM_KV_HEADS, 256, device=device, dtype=torch.float16)
    V_cache = torch.zeros(total_blocks * 2, PAGE_SIZE, NUM_KV_HEADS, 256, device=device, dtype=torch.float16)

    Q_all = torch.randn(total_q, NUM_Q_HEADS, 256, device=device, dtype=torch.float16)
    block_tables = torch.zeros(num_seqs, max_blocks, device=device, dtype=torch.int32)
    q_start_locs = torch.zeros(num_seqs, device=device, dtype=torch.int32)
    q_seq_lens_t = torch.zeros(num_seqs, device=device, dtype=torch.int32)
    kv_seq_lens_t = torch.zeros(num_seqs, device=device, dtype=torch.int32)
    q_start_positions = torch.zeros(num_seqs, device=device, dtype=torch.int32)

    # Per-seq contiguous K/V for reference
    K_contigs = []
    V_contigs = []
    block_counter = 0
    q_offset = 0

    for s, (ql, kvl, qsp) in enumerate(seq_configs):
        # Generate contiguous K/V
        K_c = torch.randn(kvl, NUM_KV_HEADS, 256, device=device, dtype=torch.float16)
        V_c = torch.randn(kvl, NUM_KV_HEADS, 256, device=device, dtype=torch.float16)
        K_contigs.append(K_c)
        V_contigs.append(V_c)

        # Fill paged cache (sequential allocation with random offset)
        num_pages = (kvl + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(num_pages):
            phys_block = block_counter + p
            block_tables[s, p] = phys_block
            start = p * PAGE_SIZE
            end = min(start + PAGE_SIZE, kvl)
            length = end - start
            K_cache[phys_block, :length] = K_c[start:end]
            V_cache[phys_block, :length] = V_c[start:end]
        block_counter += num_pages

        q_start_locs[s] = q_offset
        q_seq_lens_t[s] = ql
        kv_seq_lens_t[s] = kvl
        q_start_positions[s] = qsp
        q_offset += ql

    return (Q_all, K_cache, V_cache, block_tables, q_start_locs,
            q_seq_lens_t, kv_seq_lens_t, q_start_positions,
            K_contigs, V_contigs, seq_configs)


print(f"Device: {torch.xpu.get_device_name(0)}")

# === Test 1: Single seq ===
print("\n=== Test 1: Single Seq (batch=1) ===")
pass1 = True
cases = [(128, 128, 0), (128, 1024, 896), (256, 2048, 1792)]
for ql, kvl, qsp in cases:
    torch.manual_seed(ql + kvl + qsp + 100)
    inputs = make_batch_inputs([(ql, kvl, qsp)])
    Q_all, K_cache, V_cache, block_tables, q_start_locs, q_seq_lens, kv_seq_lens, q_start_pos, K_cs, V_cs, _ = inputs

    actual = fwd(Q_all, K_cache, V_cache, block_tables, q_start_locs, q_seq_lens, kv_seq_lens, q_start_pos, PAGE_SIZE)
    expected = reference_single_seq(Q_all, K_cs[0], V_cs[0], qsp)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass1 = False
    print(f"  Q={ql:3d} KV={kvl:4d} qsp={qsp:4d}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass1 else 'FAILED'}")

# === Test 2: Multiple seqs, same length ===
print("\n=== Test 2: Multi-Seq Same Length ===")
pass2 = True
configs = [(64, 64, 0), (64, 64, 0), (64, 64, 0), (64, 64, 0)]
torch.manual_seed(2222)
inputs = make_batch_inputs(configs)
Q_all, K_cache, V_cache, block_tables, q_start_locs, q_seq_lens, kv_seq_lens, q_start_pos, K_cs, V_cs, _ = inputs

actual = fwd(Q_all, K_cache, V_cache, block_tables, q_start_locs, q_seq_lens, kv_seq_lens, q_start_pos, PAGE_SIZE)

for s, (ql, kvl, qsp) in enumerate(configs):
    qs = int(q_start_locs[s].item())
    actual_s = actual[qs:qs+ql]
    expected_s = reference_single_seq(Q_all[qs:qs+ql], K_cs[s], V_cs[s], qsp)
    rel = (actual_s.float() - expected_s.float()).abs().max().item() / (expected_s.float().abs().max().item() + 1e-6)
    ok = rel < 0.02
    if not ok: pass2 = False
    print(f"  Seq {s}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass2 else 'FAILED'}")

# === Test 3: Multiple seqs, different lengths ===
print("\n=== Test 3: Multi-Seq Variable Length ===")
pass3 = True
configs = [(32, 128, 96), (64, 256, 192), (128, 512, 384), (256, 1024, 768)]
torch.manual_seed(3333)
inputs = make_batch_inputs(configs)
Q_all, K_cache, V_cache, block_tables, q_start_locs, q_seq_lens, kv_seq_lens, q_start_pos, K_cs, V_cs, _ = inputs

actual = fwd(Q_all, K_cache, V_cache, block_tables, q_start_locs, q_seq_lens, kv_seq_lens, q_start_pos, PAGE_SIZE)

for s, (ql, kvl, qsp) in enumerate(configs):
    qs = int(q_start_locs[s].item())
    actual_s = actual[qs:qs+ql]
    expected_s = reference_single_seq(Q_all[qs:qs+ql], K_cs[s], V_cs[s], qsp)
    rel = (actual_s.float() - expected_s.float()).abs().max().item() / (expected_s.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual_s).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass3 = False
    print(f"  Seq {s} (Q={ql:3d} KV={kvl:4d}): rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass3 else 'FAILED'}")

overall = pass1 and pass2 and pass3
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
