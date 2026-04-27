"""Accuracy tests for Phase-1 non-GEMM kernels added to moe_int4_prefill_ops.

Covers:
  1. gather's new rows_for_experts output
  2. moe_prefill_silu_mul_forward
  3. moe_prefill_scatter_x_forward
  4. moe_topk_softmax

Usage:
    python -m pytest tests/test_moe_prefill_phase1_kernels.py -v -s
    python tests/test_moe_prefill_phase1_kernels.py         # standalone runner
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "xpu"


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


# ───────────────────────── 1. gather rows_for_experts ──────────────────────
def _gather_refs(topk_ids: torch.Tensor, E: int):
    se = topk_ids.cpu().numpy().reshape(-1)
    rows = np.bincount(se, minlength=E).astype(np.int32)
    offsets = np.zeros(E, dtype=np.int32)
    offsets[1:] = np.cumsum(rows[:-1])
    return offsets, rows


def test_gather_rows():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(0)
    M, TK, E = 64, 8, 256
    ti = torch.randint(0, E, (M, TK), dtype=torch.int32, device=DEVICE)

    off, tok, pair_to_perm, rows = ops.moe_prefill_gather_forward_v2(ti, E)

    assert rows.dtype == torch.int32
    assert rows.shape == (E,)

    off_ref, rows_ref = _gather_refs(ti, E)
    assert _max_abs(off.cpu(), torch.from_numpy(off_ref)) == 0
    assert _max_abs(rows.cpu(), torch.from_numpy(rows_ref)) == 0

    # rows + offsets consistency: sum(rows) == total, offsets[e] + rows[e] == offsets[e+1]
    assert int(rows.sum().item()) == M * TK
    off_plus_rows = off[:-1] + rows[:-1]
    assert _max_abs(off_plus_rows.cpu(), off.cpu()[1:]) == 0
    print("[gather_rows] OK")


# ───────────────────────── 2. silu_mul ──────────────────────────────────────
def _silu_mul_ref(gate_up: torch.Tensor) -> torch.Tensor:
    I = gate_up.size(-1) // 2
    g = gate_up[..., :I]
    u = gate_up[..., I:]
    return F.silu(g) * u


def _test_silu_mul(N, I, dtype):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(0)
    x_cpu = (torch.randn(N, 2 * I, dtype=torch.float32) * 1.5).to(dtype)
    x = x_cpu.to(DEVICE)

    y = ops.moe_prefill_silu_mul_forward(x)
    # High-precision reference: compute in fp32 from the fp16 input, then round.
    y_ref = _silu_mul_ref(x_cpu.float()).to(dtype)

    # Use relative tolerance against abs(y_ref) + ULP floor. Kernel uses fp32
    # math with -ffast-math exp; acceptable drift is a few fp16/bf16 ULPs.
    y_f  = y.float().cpu()
    yr_f = y_ref.float()
    denom = yr_f.abs().clamp_min(1e-2)
    rel = ((y_f - yr_f).abs() / denom).max().item()
    max_abs = (y_f - yr_f).abs().max().item()
    # Allow either < 0.5% relative OR tiny absolute (within 4 fp16 ULPs ≈ 1/128).
    ulp_floor = 8.0 / 512.0   if dtype == torch.float16 else 6.0 / 128.0
    rel_tol   = 5e-3          if dtype == torch.float16 else 3e-2
    assert rel < rel_tol or max_abs < ulp_floor, (
        f"silu_mul fail: rel={rel:.2e} max_abs={max_abs:.2e}")
    assert y.shape == (N, I) and y.dtype == dtype


def test_silu_mul_fp16_small():
    _test_silu_mul(N=32, I=128, dtype=torch.float16)
    print("[silu_mul fp16 small] OK")

def test_silu_mul_fp16_122B_shape():
    _test_silu_mul(N=128, I=1024, dtype=torch.float16)
    print("[silu_mul fp16 122B] OK")

def test_silu_mul_fp16_35B_shape():
    _test_silu_mul(N=128, I=512, dtype=torch.float16)
    print("[silu_mul fp16 35B] OK")

def test_silu_mul_bf16():
    _test_silu_mul(N=64, I=512, dtype=torch.bfloat16)
    print("[silu_mul bf16] OK")


# ───────────────────────── 3. scatter_x ─────────────────────────────────────
def _scatter_x_ref(x: torch.Tensor, expert_tokens: torch.Tensor, top_k: int):
    pair_to_token = (expert_tokens.to(torch.int64) // top_k)
    return torch.index_select(x, 0, pair_to_token)


def _test_scatter_x(M, H, TK, E, dtype):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(0)
    x_cpu = torch.randn(M, H, dtype=torch.float32).to(dtype)
    ti = torch.randint(0, E, (M, TK), dtype=torch.int32)

    x = x_cpu.to(DEVICE)
    ti_xpu = ti.to(DEVICE)

    _off, tok, _p2p, _rows = ops.moe_prefill_gather_forward_v2(ti_xpu, E)

    x_perm = ops.moe_prefill_scatter_x_forward(x, tok, TK)
    x_perm_ref = _scatter_x_ref(x_cpu, tok.cpu(), TK).to(dtype)

    # bit-equal: we're just indexing+copying.
    assert _max_abs(x_perm.cpu(), x_perm_ref) == 0
    assert x_perm.shape == (M * TK, H) and x_perm.dtype == dtype


def test_scatter_x_fp16_small():
    _test_scatter_x(M=16, H=256, TK=2, E=8, dtype=torch.float16)
    print("[scatter_x fp16 small] OK")

def test_scatter_x_fp16_122B_shape():
    _test_scatter_x(M=128, H=3072, TK=8, E=256, dtype=torch.float16)
    print("[scatter_x fp16 122B] OK")

def test_scatter_x_fp16_35B_shape():
    _test_scatter_x(M=128, H=2048, TK=8, E=256, dtype=torch.float16)
    print("[scatter_x fp16 35B] OK")

def test_scatter_x_bf16():
    _test_scatter_x(M=64, H=2048, TK=4, E=64, dtype=torch.bfloat16)
    print("[scatter_x bf16] OK")


# ───────────────────────── 4. topk_softmax ─────────────────────────────────
def test_topk_softmax_vs_ipex():
    import intel_extension_for_pytorch  # noqa: F401
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(0)
    M, TK, E = 256, 8, 256
    logits = torch.randn(M, E, dtype=torch.float16, device=DEVICE) * 2.0

    # our kernel
    tw_k, ti_k = ops.moe_topk_softmax(logits, TK, E)

    # IPEX reference (renormalize=True). Runs its own kernel but gives same
    # semantic output (sum-to-1 over the TK winners).
    tw_ref = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
    ti_ref = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    _tei = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw_ref, ti_ref, _tei, logits, True)
    tw_ref = tw_ref.to(torch.float16)

    # ids should match exactly (softmax ordering is deterministic on disjoint logits).
    # Row-wise as *sets* (ipex and our kernel may break ties in opposite order
    # when two experts have numerically identical logits; bottom line: the chosen
    # expert set must be identical).
    ti_k_np  = ti_k.cpu().numpy()
    ti_ref_np = ti_ref.cpu().numpy()
    for r in range(M):
        assert set(ti_k_np[r].tolist()) == set(ti_ref_np[r].tolist()), \
            f"expert set mismatch at row {r}: ours={ti_k_np[r]} ipex={ti_ref_np[r]}"

    # For weight comparison, sort by id then compare.
    tw_k_f = tw_k.float().cpu().numpy()
    tw_r_f = tw_ref.float().cpu().numpy()
    k_perm = np.argsort(ti_k_np, axis=1)
    r_perm = np.argsort(ti_ref_np, axis=1)
    tw_k_sorted = np.take_along_axis(tw_k_f, k_perm, axis=1)
    tw_r_sorted = np.take_along_axis(tw_r_f, r_perm, axis=1)
    diff = float(np.max(np.abs(tw_k_sorted - tw_r_sorted)))
    assert diff < 2e-3, f"topk_weights max_abs_diff={diff}"
    # Each row's weights sum to 1.
    assert np.allclose(tw_k_f.sum(axis=1), 1.0, atol=2e-3)
    print(f"[topk_softmax vs ipex] OK (max_abs_diff={diff:.2e})")


# ───────────────────────── Standalone runner ───────────────────────────────
def _run(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"[{name}] FAIL: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    _run("gather_rows",           test_gather_rows)
    _run("silu_mul_fp16_small",   test_silu_mul_fp16_small)
    _run("silu_mul_fp16_122B",    test_silu_mul_fp16_122B_shape)
    _run("silu_mul_fp16_35B",     test_silu_mul_fp16_35B_shape)
    _run("silu_mul_bf16",         test_silu_mul_bf16)
    _run("scatter_x_fp16_small",  test_scatter_x_fp16_small)
    _run("scatter_x_fp16_122B",   test_scatter_x_fp16_122B_shape)
    _run("scatter_x_fp16_35B",    test_scatter_x_fp16_35B_shape)
    _run("scatter_x_bf16",        test_scatter_x_bf16)
    _run("topk_softmax_vs_ipex",  test_topk_softmax_vs_ipex)
    print("\nAll Phase-1 kernel tests OK")
