"""E2E test: compare esimd_gdn_conv_fused_seq against torch.ops._xpu_C.gdn_attention.

Mirrors the Qwen3.5 model's validation pattern:
  - Decode path uses esimd_gdn_conv_fused_seq with sequential [q|k|v|z] layout
  - Prefill/reference uses torch.ops._xpu_C.gdn_attention with interleaved layout

Both kernels share the same conv_state/conv_weight/ssm_state (sequential dim order).
Only qkvz and ba differ in layout.
"""

import torch

NUM_K_HEADS_GLOBAL = 16
TP_SIZE = 4
K = 128
V = 128


def seq_to_interleaved_qkvz(qkvz_seq, H, HV, K, V):
    """Convert qkvz from sequential [q|k|v|z] to GQA-interleaved layout.
    Matches the _gather_qkvz index in Qwen3_5GatedDeltaNet.__init__."""
    HPG = HV // H
    N = qkvz_seq.shape[0]
    qs = H * K
    ks = H * K
    vs = HV * V

    q_all = qkvz_seq[:, :qs].view(N, H, K)
    k_all = qkvz_seq[:, qs:qs+ks].view(N, H, K)
    v_all = qkvz_seq[:, qs+ks:qs+ks+vs].view(N, HV, V)
    z_all = qkvz_seq[:, qs+ks+vs:].view(N, HV, V)

    # Interleaved per group g: [q_g(K) | k_g(K) | v_g_lanes(HPG*V) | z_g_lanes(HPG*V)]
    parts = []
    for g in range(H):
        parts.append(q_all[:, g])
        parts.append(k_all[:, g])
        for lane in range(HPG):
            parts.append(v_all[:, g * HPG + lane])
        for lane in range(HPG):
            parts.append(z_all[:, g * HPG + lane])
    return torch.cat(parts, dim=1)


def seq_to_interleaved_ba(ba_seq, H, HV):
    """Convert ba from sequential [b_all|a_all] to interleaved [b_g|a_g per group].
    Matches the _gather_ba index in Qwen3_5GatedDeltaNet.__init__."""
    HPG = HV // H
    N = ba_seq.shape[0]
    b_all = ba_seq[:, :HV].view(N, H, HPG)
    a_all = ba_seq[:, HV:].view(N, H, HPG)

    parts = []
    for g in range(H):
        parts.append(b_all[:, g])
        parts.append(a_all[:, g])
    return torch.cat(parts, dim=1)


def make_strided_states(num_cache, HV, DIM, device="xpu"):
    """Create strided conv_state/ssm_state matching vLLM's pool layout."""
    SLOT_SIZE = 327680
    pool = torch.zeros(num_cache * SLOT_SIZE, dtype=torch.float16, device=device)
    conv_state = torch.as_strided(
        pool, size=(num_cache, 3, DIM), stride=(SLOT_SIZE, DIM, 1))
    ssm_state = torch.as_strided(
        pool, size=(num_cache, HV, V, K), stride=(SLOT_SIZE, V * K, K, 1),
        storage_offset=3 * DIM)
    return conv_state, ssm_state


def test_e2e_seq(N=1, num_v_heads_global=32, num_cache=32, seed=42):
    H = NUM_K_HEADS_GLOBAL // TP_SIZE
    HV = num_v_heads_global // TP_SIZE
    HPG = HV // H
    DIM = H * K + H * K + HV * V
    QKVZ_DIM = H * (K + K + HPG * V * 2)

    print(f"\n=== E2E Test seq: N={N}, H={H}, HV={HV}, K={K}, V={V}, "
          f"num_v_heads_global={num_v_heads_global} ===")
    torch.manual_seed(seed)
    device = "xpu"

    # Generate data in SEQUENTIAL layout (matching model's decode path)
    qkvz_seq = torch.randn(N, QKVZ_DIM, dtype=torch.float16, device=device) * 0.1
    ba_seq = torch.randn(N, 2 * HV, dtype=torch.float16, device=device) * 0.5

    # conv_weight/conv_bias use same dim layout for both kernels
    conv_weight = torch.randn(DIM, 4, dtype=torch.float16, device=device) * 0.1
    conv_bias_zeros = torch.zeros(DIM, dtype=torch.float16, device=device)
    A_log = torch.randn(HV, dtype=torch.float16, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float16, device=device) * 0.1
    query_start_loc = torch.arange(N + 1, dtype=torch.int32, device=device)
    state_indices = torch.arange(N, dtype=torch.int32, device=device)

    # Identical initial states
    torch.manual_seed(seed + 1000)
    init_conv = torch.randn(num_cache, 3, DIM, dtype=torch.float16, device=device) * 0.01
    init_ssm = torch.zeros(num_cache, HV, V, K, dtype=torch.float16, device=device)

    # Convert qkvz/ba to interleaved layout for reference kernel
    qkvz_intl = seq_to_interleaved_qkvz(qkvz_seq, H, HV, K, V)
    ba_intl = seq_to_interleaved_ba(ba_seq, H, HV)

    # ---- Reference: torch.ops._xpu_C.gdn_attention (interleaved, prefill path) ----
    conv_ref, ssm_ref = make_strided_states(num_cache, HV, DIM, device)
    conv_ref.copy_(init_conv); ssm_ref.copy_(init_ssm)
    out_ref = torch.empty(N, HV, V, dtype=torch.float16, device=device)
    z_ref = torch.empty(N, HV, V, dtype=torch.float16, device=device)

    torch.ops._xpu_C.gdn_attention(
        out_ref, z_ref,
        qkvz_intl.clone(), ba_intl.clone(),
        NUM_K_HEADS_GLOBAL, num_v_heads_global, K, V,
        conv_state=conv_ref, ssm_state=ssm_ref,
        conv_weights=conv_weight, conv_bias=None, activation="silu",
        A_log=A_log, dt_bias=dt_bias,
        num_prefills=0, num_decodes=N, has_initial_state=None,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        num_actual_tokens=N, tp_size=TP_SIZE)
    torch.xpu.synchronize()

    # ---- Our kernel: esimd_gdn_conv_fused_seq (sequential, decode path) ----
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq

    conv_ours, ssm_ours = make_strided_states(num_cache, HV, DIM, device)
    conv_ours.copy_(init_conv); ssm_ours.copy_(init_ssm)
    out_ours = torch.empty(N, HV, V, dtype=torch.float16, device=device)
    z_ours = torch.empty(N, HV, V, dtype=torch.float16, device=device)

    esimd_gdn_conv_fused_seq(
        qkvz_seq, conv_ours, conv_weight, conv_bias_zeros, state_indices,
        A_log, dt_bias, ba_seq,
        ssm_ours, state_indices, out_ours, z_ours,
        N, H, HV, K, V, float(K ** -0.5))
    torch.xpu.synchronize()

    # ---- Compare ----
    def check(name, ref, test, atol=1.0):
        d = (ref.float() - test.float()).abs()
        max_d = d.max().item()
        nan = torch.isnan(test).any().item()
        ok = not nan and max_d < atol
        print(f"  {name:20s}: max_diff={max_d:.6f}  atol={atol}  nan={nan}  [{'PASS' if ok else 'FAIL'}]")
        return ok

    ok = True
    ok &= check("core_attn_out", out_ref, out_ours, atol=1e-3)
    ok &= check("z_out", z_ref, z_ours, atol=1e-3)
    ok &= check("conv_state", conv_ref, conv_ours, atol=1e-3)
    ok &= check("ssm_state", ssm_ref, ssm_ours, atol=1e-3)

    print(f"  Ref  out[0,0,:4]: {out_ref.float()[0,0,:4].cpu().tolist()}")
    print(f"  Ours out[0,0,:4]: {out_ours.float()[0,0,:4].cpu().tolist()}")
    print(f"  Ref  z[0,0,:4]:   {z_ref.float()[0,0,:4].cpu().tolist()}")
    print(f"  Ours z[0,0,:4]:   {z_ours.float()[0,0,:4].cpu().tolist()}")
    return ok


if __name__ == "__main__":
    import vllm_xpu_kernels._xpu_C
    ok = True
    # (label, num_v_heads_global): 35B-A3B=32, 27B=48, 122B-A10B=64
    configs = [("Qwen3.5-35B-A3B", 32), ("Qwen3.5-27B", 48), ("Qwen3.5-122B-A10B", 64)]
    for label, num_v_heads in configs:
        for n in [1, 4, 8]:
            print(f"\n--- {label} (num_v_heads_global={num_v_heads}) ---")
            ok &= test_e2e_seq(N=n, num_v_heads_global=num_v_heads)
    print(f"\nOverall: {'ALL PASS' if ok else 'SOME FAILURES'}")
