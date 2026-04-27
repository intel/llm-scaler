"""Reproduce FP8 ESIMD kernel bug: NaN after 2 async iterations → DEVICE_LOST.

Root cause: ESIMD fp8 kernels (esimd_resadd_norm_gemv_fp8_pert,
esimd_gemv_fp8_pert) submit to the SYCL queue asynchronously. When called
back-to-back sharing pre-allocated buffers (as the vllm decode loop does),
the second iteration reads buffers that the first iteration's kernel hasn't
finished writing. This data race produces NaN, which propagates through
subsequent layers and eventually causes TP allreduce to hang the GPU
(UR_RESULT_ERROR_DEVICE_LOST in multi-card scenarios).

Proof: 1 iteration → correct output. 2 iterations async → NaN.
       2 iterations with sync between → correct output.

Qwen3.6-27B config: hidden=5120, intermediate=17408, 64 layers.

Usage:
    ZE_AFFINITY_MASK=4 python tests/test_fp8_device_lost.py
    ZE_AFFINITY_MASK=4 python -m pytest tests/test_fp8_device_lost.py -v -x -s
"""
import sys
import time
import pytest
import torch

DEVICE = "xpu"
HIDDEN = 5120
INTERMEDIATE = 17408
EPS = 1e-6


def make_fp8_weight(N, K, device=DEVICE):
    w = torch.randn(N, K, dtype=torch.float16, device=device)
    amax = w.float().abs().max().item()
    s = amax / 448.0 if amax > 0 else 1.0
    return (w.float() / s).to(torch.float8_e4m3fn), \
           torch.tensor([s], dtype=torch.float32, device=device)


def run_dense_mlp_loop(tp, n_iters, sync_each=False):
    """Run dense MLP (resadd_norm_gemv + silu + gemv) for n_iters.

    Returns (output_tensor, has_nan).
    """
    from custom_esimd_kernels_vllm import (
        esimd_resadd_norm_gemv_fp8_pert,
        esimd_gemv_fp8_pert,
    )

    inter_tp = INTERMEDIATE // tp
    gate_up_N = 2 * inter_tp

    torch.manual_seed(42)
    gw, gs = make_fp8_weight(gate_up_N, HIDDEN)
    dw, ds = make_fp8_weight(HIDDEN, inter_tp)
    nw = torch.randn(HIDDEN, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

    gb = torch.empty(1, gate_up_N, dtype=torch.float16, device=DEVICE)
    db = torch.empty(1, HIDDEN, dtype=torch.float16, device=DEVICE)
    nb = torch.empty(1, HIDDEN, dtype=torch.float16, device=DEVICE)

    h = torch.randn(1, HIDDEN, dtype=torch.float16, device=DEVICE)
    r = torch.randn(1, HIDDEN, dtype=torch.float16, device=DEVICE)
    torch.xpu.synchronize()

    for _ in range(n_iters):
        esimd_resadd_norm_gemv_fp8_pert(h, r, nw, gw, gs, gb, nb, EPS)
        gate = gb[:, :inter_tp]
        up = gb[:, inter_tp:]
        act = torch.nn.functional.silu(gate) * up
        esimd_gemv_fp8_pert(act, dw, ds, db)
        h = db.clone()
        if sync_each:
            torch.xpu.synchronize()

    torch.xpu.synchronize()
    return h, torch.isnan(h).any().item()


# ---------------------------------------------------------------------------
# Core reproducer: 2 async iterations produce NaN
# ---------------------------------------------------------------------------

class TestAsyncBufferRace:
    """The minimal reproducer: back-to-back ESIMD fp8 with shared buffers."""

    @pytest.mark.parametrize("tp", [4, 2])
    def test_1_iter_ok(self, tp):
        """Single iteration should always work."""
        h, has_nan = run_dense_mlp_loop(tp, n_iters=1, sync_each=False)
        assert not has_nan, "NaN on single iteration — kernel itself is broken"

    @pytest.mark.parametrize("tp", [4, 2])
    def test_2_iters_async_nan(self, tp):
        """2 async iterations produce NaN due to buffer race."""
        h, has_nan = run_dense_mlp_loop(tp, n_iters=2, sync_each=False)
        assert has_nan, (
            "Expected NaN from async buffer race but got clean output. "
            "If this passes, the bug may have been fixed."
        )

    @pytest.mark.parametrize("tp", [4, 2])
    def test_2_iters_sync_ok(self, tp):
        """2 iterations with sync after each → no NaN (proves it's a race)."""
        h, has_nan = run_dense_mlp_loop(tp, n_iters=2, sync_each=True)
        assert not has_nan, "NaN even with sync — different bug"

    @pytest.mark.parametrize("tp", [4, 2])
    def test_64_iters_sync_ok(self, tp):
        """Full 64-layer simulation with sync → no NaN."""
        h, has_nan = run_dense_mlp_loop(tp, n_iters=64, sync_each=True)
        assert not has_nan, "NaN with sync at 64 layers"

    @pytest.mark.parametrize("tp", [4, 2])
    def test_64_iters_async_nan(self, tp):
        """Full 64-layer simulation async → NaN (starts at iter 2)."""
        h, has_nan = run_dense_mlp_loop(tp, n_iters=64, sync_each=False)
        assert has_nan, "Expected NaN from 64 async iters"


# ---------------------------------------------------------------------------
# Threshold finder: exact iteration where NaN first appears
# ---------------------------------------------------------------------------

class TestFindNaNThreshold:

    @pytest.mark.parametrize("tp", [4, 2])
    def test_find_first_nan_iter(self, tp):
        """Find the exact iteration count where NaN first appears."""
        from custom_esimd_kernels_vllm import (
            esimd_resadd_norm_gemv_fp8_pert,
            esimd_gemv_fp8_pert,
        )

        inter_tp = INTERMEDIATE // tp
        gate_up_N = 2 * inter_tp

        torch.manual_seed(42)
        gw, gs = make_fp8_weight(gate_up_N, HIDDEN)
        dw, ds = make_fp8_weight(HIDDEN, inter_tp)
        nw = torch.randn(HIDDEN, dtype=torch.float16, device=DEVICE) * 0.1 + 1.0

        gb = torch.empty(1, gate_up_N, dtype=torch.float16, device=DEVICE)
        db = torch.empty(1, HIDDEN, dtype=torch.float16, device=DEVICE)
        nb = torch.empty(1, HIDDEN, dtype=torch.float16, device=DEVICE)

        h = torch.randn(1, HIDDEN, dtype=torch.float16, device=DEVICE)
        r = torch.randn(1, HIDDEN, dtype=torch.float16, device=DEVICE)
        torch.xpu.synchronize()

        first_nan = None
        for i in range(64):
            esimd_resadd_norm_gemv_fp8_pert(h, r, nw, gw, gs, gb, nb, EPS)
            gate = gb[:, :inter_tp]
            up = gb[:, inter_tp:]
            act = torch.nn.functional.silu(gate) * up
            esimd_gemv_fp8_pert(act, dw, ds, db)
            h = db.clone()
            torch.xpu.synchronize()
            if torch.isnan(h).any().item():
                first_nan = i + 1
                break

        print(f"\n  tp={tp}: first NaN at iteration {first_nan}")
        assert first_nan is not None and first_nan <= 4, \
            f"Expected NaN within first 4 iterations, got {first_nan}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("FP8 ESIMD async buffer race reproducer")
    print(f"Qwen3.6-27B: hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print("=" * 60)

    for tp in [4, 2]:
        inter_tp = INTERMEDIATE // tp
        print(f"\n--- tp={tp} (gate_up N={2*inter_tp}) ---")

        for n, sync in [(1, False), (2, False), (2, True), (64, True), (64, False)]:
            label = f"{n} iters {'sync' if sync else 'async'}"
            try:
                h, has_nan = run_dense_mlp_loop(tp, n, sync)
                status = "NaN!" if has_nan else "OK"
                print(f"  {label:25s} → {status}")
            except RuntimeError as e:
                print(f"  {label:25s} → CRASH: {e}")
                if "DEVICE_LOST" in str(e):
                    sys.exit(1)
