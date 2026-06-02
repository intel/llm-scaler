"""Unit tests for ESIMD prefill FMHA kernel.

Validates against IPEX's flash_attn_varlen_func as reference.
Run: ZE_AFFINITY_MASK=4 python tests/test_prefill_fmha.py

=== Test Case Matrix ===
┌──────┬─────────┬──────────┬───────────┬───────┬────────────┬────────┬────────┬─────────────────────────────────────┐
│ Case │ seq_len │ q_heads  │ kv_heads  │ GQA   │ block_size │ blocks │ causal │ What's new / validation focus       │
├──────┼─────────┼──────────┼───────────┼───────┼────────────┼────────┼────────┼─────────────────────────────────────┤
│  1   │    64   │    1     │     1     │  1:1  │     64     │   1    │   No   │ Basic QK^T→softmax→×V               │
│  2   │    64   │    1     │     1     │  1:1  │     64     │   1    │  Yes   │ + Causal mask                       │
│  3   │   128   │    1     │     1     │  1:1  │     64     │   2    │  Yes   │ + Multi-block paged KV              │
│  4   │   128   │    4     │     1     │  4:1  │     64     │   2    │  Yes   │ + GQA (4 Q heads share 1 KV head)   │
│  5   │   256   │    4     │     1     │  4:1  │     64     │   4    │  Yes   │ + Shuffled block_table (non-seq)    │
│  6   │   256   │    4     │     2     │  2:1  │     64     │   4    │  Yes   │ + Multiple KV heads                 │
│  7   │  2048   │   12     │     2     │  6:1  │     64     │  32    │  Yes   │ Realistic Qwen3.5-27B TP=2 shape   │
│  8   │  2048   │   12     │     2     │  6:1  │     64     │  32    │  Yes   │ Performance benchmark               │
└──────┴─────────┴──────────┴───────────┴───────┴────────────┴────────┴────────┴─────────────────────────────────────┘

=== TODO: Additional test cases to add later ===
- [ ] Batch size > 1 (multiple sequences with different lengths)
- [ ] Variable seq_len_k != seq_len_q (decode has accumulated longer KV)
- [ ] FP8 KV cache (k_scale, v_scale != 1.0)
- [ ] Sliding window attention (window_size_left != -1)
- [ ] Large seq_len (8192, 32768) for stress testing
- [ ] block_size=128 and block_size=512 (vLLM may override to larger block_size)
- [ ] Edge cases: seq_len not aligned to block_size (e.g. seq_len=100, block_size=64)
- [ ] num_q_heads=8, num_kv_heads=1 (GQA 8:1, Qwen3.5-35B-A3B TP=2)
- [ ] num_q_heads=16, num_kv_heads=1 (GQA 16:1, Qwen3.5-122B TP=2)
"""
import time

import torch
import intel_extension_for_pytorch as ipex


def ipex_reference_attention(
    query, key_cache, value_cache, output,
    cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k,
    scale, is_causal, block_table,
):
    """Reference implementation using IPEX flash_attn_varlen_func."""
    ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
        output, query, key_cache, value_cache,
        cu_seqlens_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        scale, is_causal, block_table, None,
    )
    return output


def naive_reference_attention(
    query, key_cache, value_cache,
    cu_seqlens_q, seqused_k,
    scale, is_causal, block_table, block_size,
):
    """Naive PyTorch reference for correctness validation (slow, CPU-like logic)."""
    batch_size = cu_seqlens_q.shape[0] - 1
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = key_cache.shape[2]
    gqa_ratio = num_q_heads // num_kv_heads

    output = torch.zeros_like(query)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        seq_len_q = q_end - q_start
        seq_len_k = seqused_k[b].item()

        # Gather K/V from paged cache using block_table
        k_list = []
        v_list = []
        for t in range(seq_len_k):
            block_idx = block_table[b, t // block_size].item()
            offset_in_block = t % block_size
            k_list.append(key_cache[block_idx, offset_in_block])
            v_list.append(value_cache[block_idx, offset_in_block])

        K = torch.stack(k_list, dim=0)  # [seq_len_k, num_kv_heads, head_dim]
        V = torch.stack(v_list, dim=0)

        for h in range(num_q_heads):
            kv_h = h // gqa_ratio
            Q_h = query[q_start:q_end, h, :]  # [seq_len_q, head_dim]
            K_h = K[:, kv_h, :]                # [seq_len_k, head_dim]
            V_h = V[:, kv_h, :]                # [seq_len_k, head_dim]

            score = torch.matmul(Q_h.float(), K_h.float().T) * scale

            if is_causal:
                seq_diff = seq_len_k - seq_len_q
                for i in range(seq_len_q):
                    for j in range(seq_len_k):
                        if j > i + seq_diff:
                            score[i, j] = float('-inf')

            attn = torch.softmax(score, dim=-1)
            out_h = torch.matmul(attn, V_h.float())
            output[q_start:q_end, h, :] = out_h.half()

    return output


def create_test_data(
    batch_size=1, seq_len=64, num_q_heads=4, num_kv_heads=1,
    head_dim=256, block_size=64, shuffle_blocks=False,
    device='xpu', dtype=torch.float16,
):
    """Create test tensors for attention.

    Args:
        shuffle_blocks: If True, randomize block_table to test non-sequential page access.
    """
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * batch_size + 8  # extra blocks for shuffle

    query = torch.randn(seq_len * batch_size, num_q_heads, head_dim,
                        device=device, dtype=dtype)
    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                            device=device, dtype=dtype)
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                              device=device, dtype=dtype)
    output = torch.empty_like(query)

    # Block table
    block_table = torch.zeros(batch_size, num_blocks_per_seq + 2,
                              device=device, dtype=torch.int32)
    for b in range(batch_size):
        if shuffle_blocks:
            # Randomly assign physical pages (non-sequential)
            perm = torch.randperm(total_blocks, device=device)[:num_blocks_per_seq]
            block_table[b, :num_blocks_per_seq] = perm.to(torch.int32)
        else:
            for i in range(num_blocks_per_seq):
                block_table[b, i] = b * num_blocks_per_seq + i

    cu_seqlens_q = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)],
        device=device, dtype=torch.int32)

    seqused_k = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)

    scale = head_dim ** (-0.5)

    return {
        'query': query,
        'key_cache': key_cache,
        'value_cache': value_cache,
        'output': output,
        'block_table': block_table,
        'cu_seqlens_q': cu_seqlens_q,
        'seqused_k': seqused_k,
        'max_seqlen_q': seq_len,
        'max_seqlen_k': seq_len,
        'scale': scale,
        'block_size': block_size,
    }


# ============================================================================
# Test Cases
# ============================================================================

def run_esimd_kernel(data, is_causal):
    """Helper to run ESIMD prefill FMHA kernel."""
    from custom_esimd_kernels_vllm import esimd_prefill_fmha
    out = data['output'].clone()
    esimd_prefill_fmha(
        out, data['query'], data['key_cache'], data['value_cache'],
        data['block_table'], data['cu_seqlens_q'], data['seqused_k'],
        data['max_seqlen_q'], data['max_seqlen_k'],
        data['scale'], is_causal)
    return out


def test_case1_basic():
    """Case 1: Basic QK^T → softmax → ×V, no causal, single head, single block."""
    data = create_test_data(seq_len=64, num_q_heads=1, num_kv_heads=1, block_size=64)

    ref = naive_reference_attention(
        data['query'], data['key_cache'], data['value_cache'],
        data['cu_seqlens_q'], data['seqused_k'],
        data['scale'], False, data['block_table'], data['block_size'])

    ipex_out = data['output'].clone()
    ipex_reference_attention(
        data['query'], data['key_cache'], data['value_cache'], ipex_out,
        data['cu_seqlens_q'], data['seqused_k'],
        data['max_seqlen_q'], data['max_seqlen_k'],
        data['scale'], False, data['block_table'])

    max_diff = (ref - ipex_out).abs().max().item()
    assert torch.allclose(ref, ipex_out, atol=1e-2, rtol=1e-2), \
        f"naive vs IPEX: max diff = {max_diff}"

    # ESIMD kernel
    esimd_out = run_esimd_kernel(data, False)
    esimd_diff = (ref - esimd_out).abs().max().item()
    assert torch.allclose(ref, esimd_out, atol=1e-2, rtol=1e-2), \
        f"naive vs ESIMD: max diff = {esimd_diff}"
    print(f"  [PASS] naive vs IPEX={max_diff:.6f}, naive vs ESIMD={esimd_diff:.6f}")


def test_case2_causal():
    """Case 2: + Causal mask."""
    data = create_test_data(seq_len=64, num_q_heads=1, num_kv_heads=1, block_size=64)

    ref = naive_reference_attention(
        data['query'], data['key_cache'], data['value_cache'],
        data['cu_seqlens_q'], data['seqused_k'],
        data['scale'], True, data['block_table'], data['block_size'])

    esimd_out = run_esimd_kernel(data, True)
    esimd_diff = (ref - esimd_out).abs().max().item()
    assert torch.allclose(ref, esimd_out, atol=1e-2, rtol=1e-2), \
        f"naive vs ESIMD: max diff = {esimd_diff}"
    print(f"  [PASS] causal mask correct (ESIMD diff={esimd_diff:.6f})")


def test_case3_multi_block():
    """Case 3: + Multi-block paged KV (128 tokens / 64 block_size = 2 blocks)."""
    data = create_test_data(seq_len=128, num_q_heads=1, num_kv_heads=1, block_size=64)

    ref = naive_reference_attention(
        data['query'], data['key_cache'], data['value_cache'],
        data['cu_seqlens_q'], data['seqused_k'],
        data['scale'], True, data['block_table'], data['block_size'])

    esimd_out = run_esimd_kernel(data, True)
    esimd_diff = (ref - esimd_out).abs().max().item()
    assert torch.allclose(ref, esimd_out, atol=1e-2, rtol=1e-2), \
        f"naive vs ESIMD: max diff = {esimd_diff}"
    print(f"  [PASS] multi-block paged (ESIMD diff={esimd_diff:.6f})")


def test_case4_gqa():
    """Case 4: + GQA (4 Q heads share 1 KV head)."""
    data = create_test_data(seq_len=128, num_q_heads=4, num_kv_heads=1, block_size=64)

    ref = naive_reference_attention(
        data['query'], data['key_cache'], data['value_cache'],
        data['cu_seqlens_q'], data['seqused_k'],
        data['scale'], True, data['block_table'], data['block_size'])

    esimd_out = run_esimd_kernel(data, True)
    esimd_diff = (ref - esimd_out).abs().max().item()
    assert torch.allclose(ref, esimd_out, atol=1e-2, rtol=1e-2), \
        f"naive vs ESIMD: max diff = {esimd_diff}"
    print(f"  [PASS] GQA 4:1 (ESIMD diff={esimd_diff:.6f})")


def test_case5_shuffled_blocks():
    """Case 5: + Shuffled block_table (non-sequential physical pages)."""
    data = create_test_data(
        seq_len=256, num_q_heads=4, num_kv_heads=1,
        block_size=64, shuffle_blocks=True)

    ref = naive_reference_attention(
        data['query'], data['key_cache'], data['value_cache'],
        data['cu_seqlens_q'], data['seqused_k'],
        data['scale'], True, data['block_table'], data['block_size'])

    esimd_out = run_esimd_kernel(data, True)
    esimd_diff = (ref - esimd_out).abs().max().item()
    assert torch.allclose(ref, esimd_out, atol=1e-2, rtol=1e-2), \
        f"naive vs ESIMD: max diff = {esimd_diff}"
    print(f"  [PASS] shuffled block_table (ESIMD diff={esimd_diff:.6f})")


def test_case6_multi_kv_heads():
    """Case 6: + Multiple KV heads (GQA 2:1)."""
    data = create_test_data(
        seq_len=256, num_q_heads=4, num_kv_heads=2,
        block_size=64, shuffle_blocks=True)

    ref = naive_reference_attention(
        data['query'], data['key_cache'], data['value_cache'],
        data['cu_seqlens_q'], data['seqused_k'],
        data['scale'], True, data['block_table'], data['block_size'])

    esimd_out = run_esimd_kernel(data, True)
    esimd_diff = (ref - esimd_out).abs().max().item()
    assert torch.allclose(ref, esimd_out, atol=1e-2, rtol=1e-2), \
        f"naive vs ESIMD: max diff = {esimd_diff}"
    print(f"  [PASS] multi KV heads, GQA 2:1 (ESIMD diff={esimd_diff:.6f})")


def test_case7_realistic():
    """Case 7: Realistic Qwen3.5-27B TP=2 shape (correctness only)."""
    data = create_test_data(
        seq_len=2048, num_q_heads=12, num_kv_heads=2, block_size=64)

    ipex_out = data['output'].clone()
    ipex_reference_attention(
        data['query'], data['key_cache'], data['value_cache'], ipex_out,
        data['cu_seqlens_q'], data['seqused_k'],
        data['max_seqlen_q'], data['max_seqlen_k'],
        data['scale'], True, data['block_table'])

    assert ipex_out.isfinite().all(), "IPEX output has NaN/Inf"

    esimd_out = run_esimd_kernel(data, True)
    assert esimd_out.isfinite().all(), "ESIMD output has NaN/Inf"
    max_diff = (ipex_out - esimd_out).abs().max().item()
    assert torch.allclose(ipex_out, esimd_out, atol=1e-2, rtol=1e-2), \
        f"ESIMD vs IPEX mismatch: max diff = {max_diff}"
    print(f"  [PASS] realistic shape (ESIMD vs IPEX diff={max_diff:.6f})")


def test_case8_perf():
    """Case 8: Performance benchmark (same shape as case 7)."""
    data = create_test_data(
        seq_len=2048, num_q_heads=12, num_kv_heads=2, block_size=64)

    # Warmup
    for _ in range(3):
        ipex_reference_attention(
            data['query'], data['key_cache'], data['value_cache'],
            data['output'].clone(),
            data['cu_seqlens_q'], data['seqused_k'],
            data['max_seqlen_q'], data['max_seqlen_k'],
            data['scale'], True, data['block_table'])
    torch.xpu.synchronize()

    # Time IPEX
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        ipex_reference_attention(
            data['query'], data['key_cache'], data['value_cache'],
            data['output'].clone(),
            data['cu_seqlens_q'], data['seqused_k'],
            data['max_seqlen_q'], data['max_seqlen_k'],
            data['scale'], True, data['block_table'])
    torch.xpu.synchronize()
    ipex_ms = (time.perf_counter() - t0) / N * 1000

    # Time ESIMD
    for _ in range(3):
        run_esimd_kernel(data, True)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        run_esimd_kernel(data, True)
    torch.xpu.synchronize()
    esimd_ms = (time.perf_counter() - t0) / N * 1000

    print(f"  [PERF] IPEX:  {ipex_ms:.2f} ms/call")
    print(f"  [PERF] ESIMD: {esimd_ms:.2f} ms/call")
    if esimd_ms > 0:
        print(f"  [PERF] Ratio: {esimd_ms/ipex_ms:.1f}x (ESIMD/IPEX, <1.0 = ESIMD faster)")
    print(f"         (seq_len=2048, q_heads=12, kv_heads=2, head_dim=256, block_size=64)")


# ============================================================================
# Runner
# ============================================================================

ALL_TESTS = [
    ("Case 1: Basic (no causal, 1 head, 1 block)", test_case1_basic),
    ("Case 2: + Causal mask", test_case2_causal),
    ("Case 3: + Multi-block paged KV", test_case3_multi_block),
    ("Case 4: + GQA (4:1)", test_case4_gqa),
    ("Case 5: + Shuffled block_table", test_case5_shuffled_blocks),
    ("Case 6: + Multi KV heads (2:1)", test_case6_multi_kv_heads),
    ("Case 7: Realistic shape (2048 tokens)", test_case7_realistic),
    ("Case 8: Performance benchmark", test_case8_perf),
]


def run_tests():
    print("=" * 70)
    print("Prefill FMHA Unit Tests")
    print("=" * 70)
    passed = 0
    failed = 0
    for name, test_fn in ALL_TESTS:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run specific cases: python test_prefill_fmha.py 1 2 3
        cases = [int(x) for x in sys.argv[1:]]
        selected = [(name, fn) for i, (name, fn) in enumerate(ALL_TESTS, 1) if i in cases]
        print("=" * 70)
        print(f"Prefill FMHA Unit Tests (cases: {cases})")
        print("=" * 70)
        passed = failed = 0
        for name, test_fn in selected:
            print(f"\n[TEST] {name}")
            try:
                test_fn()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {e}")
                failed += 1
        print(f"\nResults: {passed} passed, {failed} failed")
    else:
        run_tests()
