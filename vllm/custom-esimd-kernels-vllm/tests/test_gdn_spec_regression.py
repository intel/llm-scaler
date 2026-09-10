"""CPU litmus tests for the 2026-09-07 speculative GDN audit.

These demonstrate counterexamples, NOT passing native-kernel correctness.
No XPU discovery, DSO loading, or vLLM imports are needed.

Sources: gdn_conv_fused_seq_spec.h (ESIMD) and installed wheel revision
a692986, csrc/xpu/gdn_attn/{causal_conv1d,gated_delta_rule}.hpp (fallback).
The fallback's FP16 q/k/v interface and FP32 intra-call SSM carry are modeled
separately from the native implementation, unlike the old native reference.
The packed contract cases cover both Qwen3.8 local geometries: TP8
H=2/HV=6 and TP4 H=4/HV=12.
"""

import collections
import copy
import os

import pytest
import torch


def _conv_step(cache, inputs, accepted, *, packed):
    """One feature lane, width=4, distinct cache_indices = arange(M)."""
    col = max(accepted - 1, 0)
    history = (cache[0, col : col + 3] if packed else cache[col, :3]).clone()
    before = history.clone()
    for t, value in enumerate(inputs):
        if packed:
            if t == 0:
                cache[0, :2] = history[1:]
            cache[0, 2 + t] = value
        else:
            cache[t, :2] = history[1:]
            cache[t, 2] = value
        history = torch.cat((history[1:], value.reshape(1)))
    return before


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("packed", [False, True])
def test_hv0_finishing_before_other_wg_load_is_a_history_counterexample(m, packed):
    """Legal WG scheduling changes shared Q/K; a WG barrier cannot fix it."""
    rows = m + 2 if packed else 3
    for accepted in range(1, m + 1):
        cache = torch.arange(m * rows, dtype=torch.float32).reshape(m, rows)
        inputs = torch.arange(100, 100 + m, dtype=torch.float32)
        col = accepted - 1
        early_load = (cache[0, col : col + 3] if packed else cache[col, :3]).clone()
        # HV0 owns all Q/K stores. Another WG may not have started yet.
        _conv_step(cache, inputs, accepted, packed=packed)
        late_load = cache[0, col : col + 3] if packed else cache[col, :3]
        assert not torch.equal(early_load, late_load), (m, accepted, packed)


@pytest.mark.parametrize("m", range(2, 9))
def test_packed_and_per_slot_caches_are_not_interchangeable_next_step(m):
    """Both directions fail for acceptance >1 even without any WG race."""
    for accepted in range(2, m + 1):
        initial = torch.full((m, m + 2), -99.0)
        initial[0, :3] = torch.tensor([1.0, 2.0, 3.0])
        inputs = torch.arange(10, 10 + m, dtype=torch.float32)
        next_inputs = torch.arange(20, 20 + m, dtype=torch.float32)
        packed_cache, slot_cache = initial.clone(), initial.clone()
        _conv_step(packed_cache, inputs, 1, packed=True)
        _conv_step(slot_cache, inputs, 1, packed=False)
        expected = torch.cat((initial[0, :3], inputs[:accepted]))[-3:]
        correct_packed = _conv_step(
            packed_cache.clone(), next_inputs, accepted, packed=True
        )
        correct_slot = _conv_step(
            slot_cache.clone(), next_inputs, accepted, packed=False
        )
        torch.testing.assert_close(correct_packed, expected, rtol=0, atol=0)
        torch.testing.assert_close(correct_slot, expected, rtol=0, atol=0)
        wrong_after_packed = _conv_step(
            packed_cache.clone(), next_inputs, accepted, packed=False
        )
        wrong_after_slot = _conv_step(
            slot_cache.clone(), next_inputs, accepted, packed=True
        )
        assert not torch.equal(wrong_after_packed, expected)
        assert not torch.equal(wrong_after_slot, expected)


def _delta_call(pool, q, k, v, decay, beta, accepted, *, reload_fp16):
    """FP32 GDN recurrence, FP16 rollback snapshots, no convolution noise."""
    carry = pool[accepted - 1].float()
    outputs = []
    for t in range(q.shape[0]):
        if reload_fp16 and t:
            carry = pool[t - 1].float()
        carry = carry * decay[t]
        delta = (v[t] - carry @ k[t]) * beta[t]
        carry = carry + delta[:, None] * k[t][None, :]
        outputs.append((carry @ q[t]).half())
        pool[t].copy_(carry.half())
    return torch.stack(outputs)


@pytest.mark.parametrize("m", range(2, 9))
def test_multicall_accepted_rollback_distinguishes_fp32_carry_from_fp16_reload(m):
    """All acceptance counts, repeated calls; identical FP16 q/k/v inputs."""
    rng = torch.Generator(device="cpu").manual_seed(138 + m)
    pool = (torch.randn(m, 8, 128, generator=rng) * 0.3).half()
    fallback, native = pool.clone(), pool.clone()
    max_output_error = 0.0
    max_state_error = 0.0
    for call in range(m * 4):
        q = torch.randn(m, 128, generator=rng).half().float()
        k = torch.randn(m, 128, generator=rng).half().float()
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / 128**0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        v = torch.randn(m, 8, generator=rng).half().float()
        decay = torch.full((m,), 0.99)
        beta = torch.full((m,), 0.25)
        accepted = call % m + 1
        want = _delta_call(fallback, q, k, v, decay, beta, accepted, reload_fp16=False)
        got = _delta_call(native, q, k, v, decay, beta, accepted, reload_fp16=True)
        max_output_error = max(max_output_error, (got - want).abs().max().item())
        max_state_error = max(max_state_error, (native - fallback).abs().max().item())
    print(
        f"M={m}: carry output max_abs={max_output_error:.9g}, "
        f"state max_abs={max_state_error:.9g}"
    )
    # This is an isolated numerical mismatch, not evidence of model degradation.
    assert max_output_error > 0
    assert max_state_error > 0
    assert max_output_error < 0.05  # The previous tolerance hides this mismatch.


def test_fallback_fp16_conv_output_boundary_is_absent_in_native_reference():
    """Even with correct FP32 carry, keeping conv output FP32 is not identical."""
    rng = torch.Generator(device="cpu").manual_seed(94)
    x = torch.randn(4, 128, generator=rng).half().float()
    weight = torch.randn(4, 128, generator=rng).half().float()
    conv = (x * weight).sum(0)
    conv = conv / (1 + torch.exp(-conv))
    native_k = conv / torch.sqrt((conv * conv).sum() + 1e-6)
    fallback_conv = conv.half().float()
    fallback_k = fallback_conv / torch.sqrt(
        (fallback_conv * fallback_conv).sum() + 1e-6
    )
    error = (native_k - fallback_k).abs().max().item()
    print(f"FP16 conv interface: normalized K max_abs={error:.9g}")
    assert error > 1e-6


@pytest.mark.parametrize("hv", [6, 16, 24])
def test_fallback_four_lane_ba_reorder_tail_requires_element_mask(hv):
    """a692986 ReorderInput=true writes across rows and allocation for HV=6."""
    m = 4
    writes = []
    a_reads = []
    for token in range(m):
        for start in range(0, hv, 4):
            for e in range(4):
                writes.append(token * hv + start + e)
                a_reads.append(token * hv * 2 + hv + start + e)
    out_of_bounds_writes = sorted({i for i in writes if i >= m * hv})
    out_of_bounds_reads = sorted({i for i in a_reads if i >= m * hv * 2})
    overlaps = [i for i, count in collections.Counter(writes).items() if count > 1]
    if hv == 6:
        assert out_of_bounds_writes == [24, 25]
        assert out_of_bounds_reads == [48, 49]
        assert overlaps == [6, 7, 12, 13, 18, 19]
        print(
            f"HV=6 M=4: b/a OOB={out_of_bounds_writes}, "
            f"mixed_ba OOB reads={out_of_bounds_reads}, overlaps={overlaps}"
        )
    else:
        assert not out_of_bounds_writes
        assert not out_of_bounds_reads
        assert not overlaps


def _padded_pool(shape, padding, rng):
    row_elements = shape[1] * shape[2]
    stride0 = row_elements + padding
    storage = torch.full((shape[0] * stride0,), 123.0, dtype=torch.float16)
    view = storage.as_strided(shape, (stride0, shape[2], 1))
    view.copy_(torch.randn(shape, generator=rng) * 0.5)
    return storage, view


def make_positive_packed_case(m, seed=813, *, H=2, HV=6):
    """CPU-only Qwen3.8 dimensions, positive IDs, padded block strides.

    Exposed for a later GPU harness, but this audit never loads native ops.
    SSM is stored as [block, HV*V, K] with exactly the native addressing.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed + m)
    assert HV % H == 0
    conv_dim = (2 * H + HV) * 128
    qkvz_dim = (2 * H + 2 * HV) * 128
    ids = torch.arange(2, 2 + 3 * m, 3, dtype=torch.int64)
    nblocks = int(ids[-1]) + 2
    conv_storage, conv = _padded_pool((nblocks, m + 2, conv_dim), 256, rng)
    ssm_storage, ssm = _padded_pool((nblocks, HV * 128, 128), 256, rng)
    return {
        "m": m,
        "H": H,
        "HV": HV,
        "ids": ids,
        "accepted": 1,
        "conv": conv,
        "conv_storage": conv_storage,
        "ssm": ssm,
        "ssm_storage": ssm_storage,
        "qkvz": (torch.randn(m, qkvz_dim, generator=rng) * 0.75).half(),
        "weight": (torch.randn(conv_dim, 4, generator=rng) * 0.5).half(),
        "bias": (torch.randn(conv_dim, generator=rng) * 0.5).half(),
        "ba": (torch.randn(m, 2 * HV, generator=rng) * 0.5).half(),
        "A_log": torch.randn(HV, generator=rng) * 0.5 - 1,
        "dt_bias": (torch.randn(HV, generator=rng) * 0.5).half(),
        "token_indices": torch.arange(m - 1, -1, -1),
    }


def packed_contract_reference(case, dtype=torch.float32):
    """v0.26 packed rollback + two-stage XPU math, not the broken fallback.

    Deliberately retains FP32/FP64 carry across tokens; only checkpoints are
    FP16. Conv products AND the q/k/v interface are FP16, A_log is FP32 input.
    FP64 mode supplies an independent higher-precision arithmetic oracle.
    """
    m, H, HV = case["m"], case["H"], case["HV"]
    ids, accepted = case["ids"], case["accepted"]
    key_dim = H * 128
    conv_dim = (2 * H + HV) * 128
    history = case["conv"][ids[0], accepted - 1 : accepted + 2].clone()
    state = case["ssm"][ids[accepted - 1]].to(dtype).reshape(HV, 128, 128)
    outputs = torch.empty((m, HV, 128), dtype=torch.float16)
    z = torch.empty_like(outputs)
    for t, global_t in enumerate(case["token_indices"]):
        raw = case["qkvz"][global_t, :conv_dim]
        window = torch.cat((history, raw[None]), dim=0).to(dtype)
        products = (window.T * case["weight"].to(dtype)).half().to(dtype)
        conv = case["bias"].to(dtype)
        for j in range(4):
            conv = conv + products[:, j]
        conv = (conv / (1 + torch.exp(-conv))).half().to(dtype)
        q = conv[:key_dim].reshape(H, 128).repeat_interleave(HV // H, dim=0)
        k = conv[key_dim : 2 * key_dim].reshape(H, 128).repeat_interleave(
            HV // H, dim=0
        )
        v = conv[2 * key_dim:].reshape(HV, 128)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / 128**0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        ba = case["ba"][global_t].to(dtype)
        x_gate = ba[HV:] + case["dt_bias"].to(dtype)
        softplus = torch.where(x_gate > 20, x_gate, torch.log1p(torch.exp(x_gate)))
        decay = torch.exp(-torch.exp(case["A_log"].to(dtype)) * softplus)
        beta = torch.sigmoid(ba[:HV])
        state = state * decay[:, None, None]
        memory = torch.einsum("hvk,hk->hv", state, k)
        delta = (v - memory) * beta[:, None]
        state = state + delta[:, :, None] * k[:, None, :]
        outputs[global_t] = torch.einsum("hvk,hk->hv", state, q).half()
        z[global_t] = case["qkvz"][global_t, conv_dim:].reshape(HV, 128)
        case["ssm"][ids[t]].copy_(state.reshape(HV * 128, 128).half())
        if t == 0:
            case["conv"][ids[0], :2].copy_(history[1:])
        case["conv"][ids[0], 2 + t].copy_(raw)
        history = torch.cat((history[1:], raw[None]))
    return outputs, z


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize(
    ("H", "HV"), [(2, 6), (4, 12)], ids=["tp8-h2-hv6", "tp4-h4-hv12"]
)
def test_positive_packed_padded_multistep_reference_and_storage_guards(m, H, HV):
    """All acceptance counts, no zero blocks, exact history and padding checks."""
    case = make_positive_packed_case(m, H=H, HV=HV)
    oracle = copy.deepcopy(case)
    assert case["ids"].min() > 0
    conv_dim = (2 * H + HV) * 128
    assert case["conv"].stride(0) > (m + 2) * conv_dim
    assert case["ssm"].stride(0) > HV * 128 * 128
    for accepted in [1, *range(1, m + 1), m, 1]:
        case["accepted"] = oracle["accepted"] = accepted
        before = case["conv"].clone()
        expected_history = before[case["ids"][0], accepted : accepted + 2]
        expected_history = torch.cat(
            (expected_history, case["qkvz"][case["token_indices"], :conv_dim])
        )
        output, z = packed_contract_reference(case)
        expected, expected_z = packed_contract_reference(oracle, torch.float64)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=0.002, atol=0.0002)
        torch.testing.assert_close(case["ssm"], oracle["ssm"], rtol=0.002, atol=0.0002)
        torch.testing.assert_close(z, expected_z, rtol=0, atol=0)
        before[case["ids"][0]].copy_(expected_history)
        torch.testing.assert_close(case["conv"], before, rtol=0, atol=0)
        # Padding is outside tensor numel but inside the real backing storage.
        for name in ("conv", "ssm"):
            storage = case[name + "_storage"]
            rows = storage.reshape(case[name].shape[0], case[name].stride(0))
            assert torch.all(rows[:, -256:] == 123)


@pytest.mark.parametrize(
    ("H", "HV"), [(2, 6), (4, 12)], ids=["tp8-h2-hv6", "tp4-h4-hv12"]
)
@pytest.mark.skipif(os.environ.get("GDN_STRICT_XPU") != "1", reason="opt-in XPU")
def test_native_v2_strict_positive_packed_multistep_xpu(H, HV):
    """正式 v2 的 packed-state 多轮回归；失败时不可放宽数值阈值。"""
    torch.ops.load_library(os.environ["GDN_SPEC_DSO"])
    op = torch.ops.custom_esimd_kernels_vllm.esimd_gdn_conv_fused_seq_spec_v2
    cpu = make_positive_packed_case(4, H=H, HV=HV)
    device = {}
    for name in ("conv", "ssm"):
        storage = cpu[name + "_storage"].to("xpu")
        device[name + "_storage"] = storage
        device[name] = storage.as_strided(cpu[name].shape, cpu[name].stride())
    for name in ("qkvz", "weight", "bias", "ba", "dt_bias"):
        device[name] = cpu[name].to("xpu")
    # Isolate state contract / race from the separate A_log downcast difference.
    cpu["A_log"] = cpu["A_log"].half().float()
    device["A_log"] = cpu["A_log"].half().to("xpu")
    ids = cpu["ids"].int().to("xpu")
    tokens = cpu["token_indices"].int().to("xpu")
    output = torch.empty((4, HV, 128), dtype=torch.float16, device="xpu")
    z = torch.empty_like(output)
    failures = []
    for step, accepted in enumerate((1, 4, 2, 1, 3, 4)):
        cpu["accepted"] = accepted
        expected, expected_z = packed_contract_reference(cpu)
        accepted_gpu = torch.tensor([accepted], dtype=torch.int32, device="xpu")
        op(
            device["qkvz"],
            device["conv"],
            device["weight"],
            device["bias"],
            ids,
            device["A_log"],
            device["dt_bias"],
            device["ba"],
            device["ssm"].view(-1, HV, 128, 128),
            output,
            z,
            tokens,
            accepted_gpu,
            1,
            4,
            H,
            HV,
            128,
            128,
            128**-0.5,
        )
        torch.xpu.synchronize()
        for name, got, want in (
            ("output", output.cpu(), expected),
            ("conv", device["conv"].cpu(), cpu["conv"]),
            ("ssm", device["ssm"].cpu(), cpu["ssm"]),
            ("z", z.cpu(), expected_z),
        ):
            error = (got.float() - want.float()).abs().max().item()
            print(f"step={step} accepted={accepted} {name} max_abs={error:.9g}")
            try:
                exact = name in ("conv", "z")
                torch.testing.assert_close(
                    got,
                    want,
                    rtol=0 if exact else 0.002,
                    atol=0 if exact else 0.0002,
                )
            except AssertionError:
                failures.append((step, name, error))
        for name in ("conv", "ssm"):
            rows = (
                device[name + "_storage"]
                .cpu()
                .reshape(cpu[name].shape[0], cpu[name].stride(0))
            )
            assert torch.all(rows[:, -256:] == 123), f"{name} padding corruption"
    assert not failures, failures
