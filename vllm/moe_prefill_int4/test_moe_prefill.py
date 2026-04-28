"""
Test: StandaloneInt4MoE vs IPEX GatedMLPMOE — verify identical output.

Usage: python3 test_moe_prefill.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import torch.nn.functional as F
import numpy as np
import intel_extension_for_pytorch as ipex

from moe_prefill import StandaloneInt4MoE

DEVICE = "xpu"
GROUP_SIZE = 128
PACK_FACTOR = 8

CONFIGS = {
    "small": {"hidden_size": 256, "intermediate_size": 128, "num_experts": 16, "top_k": 4},
    "122B-TP4": {"hidden_size": 3072, "intermediate_size": 256, "num_experts": 256, "top_k": 8},
}


def quantize_int4(weight_fp16, group_size=GROUP_SIZE):
    N, K = weight_fp16.shape
    w = weight_fp16.float().numpy().reshape(N, K // group_size, group_size)
    max_abs = np.abs(w).max(axis=2)
    scale = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)
    quantized = np.round(w / scale[:, :, None].astype(np.float32)).clip(-8, 7).astype(np.int32) + 8
    qf = quantized.reshape(N, K).reshape(N, K // PACK_FACTOR, PACK_FACTOR).astype(np.uint32)
    packed = np.zeros((N, K // PACK_FACTOR), dtype=np.uint32)
    for b in range(PACK_FACTOR):
        packed |= (qf[:, :, b] & 0xF) << (b * 4)
    return torch.from_numpy(packed.view(np.int32)), torch.from_numpy(scale)


def cosine_sim(a, b):
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0)
    ).item()


def test_standalone_vs_ipex():
    print("=" * 60)
    print("Test: StandaloneInt4MoE vs IPEX GatedMLPMOE")
    print("=" * 60)

    for cfg_name, cfg in CONFIGS.items():
        H = cfg["hidden_size"]
        D = cfg["intermediate_size"]
        E = cfg["num_experts"]
        TK = cfg["top_k"]
        print(f"\n  Config: {cfg_name} (H={H}, D={D}, E={E}, top_k={TK})")

        for n_tokens in [1, 4, 128, 8192]:
            torch.manual_seed(42)

            # Create FP16 weights, then quantize
            w13_fp16 = (torch.randn(E, 2 * D, H) * 0.02).half()
            w2_fp16 = (torch.randn(E, H, D) * 0.02).half()

            w13_qw_list, w13_sc_list = [], []
            w2_qw_list, w2_sc_list = [], []
            for e in range(E):
                qw, sc = quantize_int4(w13_fp16[e])
                w13_qw_list.append(qw)
                w13_sc_list.append(sc)
                qw, sc = quantize_int4(w2_fp16[e])
                w2_qw_list.append(qw)
                w2_sc_list.append(sc)

            # GGML N-major: [E, N, K_packed]
            w13_qw = torch.stack(w13_qw_list)  # [E, 2*D, H//8]
            w13_sc = torch.stack(w13_sc_list)   # [E, 2*D, H//128]
            w2_qw = torch.stack(w2_qw_list)     # [E, H, D//8]
            w2_sc = torch.stack(w2_sc_list)      # [E, H, D//128]

            # ── Create IPEX GatedMLPMOE (uses deepcopy to avoid mutating originals) ──
            ipex_w13 = copy.deepcopy(w13_qw).to(DEVICE)
            ipex_w2 = copy.deepcopy(w2_qw).to(DEVICE)
            ipex_s13 = w13_sc.to(DEVICE)
            ipex_s2 = w2_sc.to(DEVICE)

            ipex_moe = ipex.llm.modules.GatedMLPMOE(
                ipex_w13, ipex_w2,
                w1_scale_inv=ipex_s13,
                w2_scale_inv=ipex_s2,
                is_int4=True,
            )

            # ── Create Standalone MoE (uses original GGML weights) ──
            standalone_moe = StandaloneInt4MoE(
                w13_qweight=w13_qw.to(DEVICE),
                w2_qweight=w2_qw.to(DEVICE),
                w13_scales=w13_sc.to(DEVICE),
                w2_scales=w2_sc.to(DEVICE),
                num_experts=E,
            )

            # ── Run both ──
            x = (torch.randn(n_tokens, H) * 0.1).half().to(DEVICE)
            logits = (torch.randn(n_tokens, E) * 0.1).half().to(DEVICE)

            ipex_out = ipex_moe(
                x.clone(), False, TK, logits.clone(), True,
            )

            standalone_out = standalone_moe(
                x.clone(), False, TK, logits.clone(), True,
            )

            cos = cosine_sim(ipex_out, standalone_out)
            mae = (ipex_out.float() - standalone_out.float()).abs().max().item()
            passed = cos > 0.99 and mae < 0.5
            status = "PASS" if passed else "FAIL"
            print(f"    n={n_tokens:>5d}: cos={cos:.6f} mae={mae:.4f} [{status}]")
            if not passed:
                print(f"      ipex[:5]={ipex_out[0, :5].tolist()}")
                print(f"      ours[:5]={standalone_out[0, :5].tolist()}")
                return False

    print("\n  All tests passed!")
    return True


def test_stress_no_hang():
    """Run repeated large-batch forwards to verify no hang."""
    print("\n" + "=" * 60)
    print("Test: Stress test (no hang check)")
    print("=" * 60)

    H, D, E, TK = 3072, 256, 256, 8
    torch.manual_seed(42)

    w13_qw, w13_sc, w2_qw, w2_sc = [], [], [], []
    for e in range(E):
        qw, sc = quantize_int4((torch.randn(2 * D, H) * 0.02).half())
        w13_qw.append(qw)
        w13_sc.append(sc)
        qw, sc = quantize_int4((torch.randn(H, D) * 0.02).half())
        w2_qw.append(qw)
        w2_sc.append(sc)

    moe = StandaloneInt4MoE(
        w13_qweight=torch.stack(w13_qw).to(DEVICE),
        w2_qweight=torch.stack(w2_qw).to(DEVICE),
        w13_scales=torch.stack(w13_sc).to(DEVICE),
        w2_scales=torch.stack(w2_sc).to(DEVICE),
    )

    import time
    x = torch.randn(8192, H, dtype=torch.float16, device=DEVICE)
    logits = torch.randn(8192, E, dtype=torch.float16, device=DEVICE)

    # Warmup
    for _ in range(3):
        moe(x, False, TK, logits, True)
    torch.xpu.synchronize()

    print("  Running 100 iterations at n=8192...")
    for i in range(100):
        t0 = time.perf_counter()
        out = moe(x, False, TK, logits, True)
        torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if i < 3 or i % 20 == 0:
            print(f"    iter {i}: {ms:.1f}ms", flush=True)

    print("  Stress test passed (no hang)!")
    return True


if __name__ == "__main__":
    ok = True
    ok &= test_standalone_vs_ipex()
    ok &= test_stress_no_hang()
    if ok:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
