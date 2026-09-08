"""Opt-in repaired v2 tests against actual standard Triton packed operators.

No _xpu_C fallback is loaded. Set GDN_STRICT_XPU=1 and GDN_SPEC_DSO to a
build-only LGRF DSO; ZE_AFFINITY_MASK must be set by the GPU owner.
"""

import os

import pytest
import torch
from test_gdn_spec_regression import make_positive_packed_case

pytestmark = pytest.mark.skipif(
    os.environ.get("GDN_STRICT_XPU") != "1", reason="requires explicit GPU lease"
)


def _device_case(m, seed=813, a_fp32=True, extra_rows=0):
    cpu = make_positive_packed_case(m, seed)
    case = {"m": m}
    for name in ("conv", "ssm"):
        shape, strides = cpu[name].shape, cpu[name].stride()
        if name == "conv" and extra_rows:
            shape = (shape[0], shape[1] + extra_rows, shape[2])
            strides = (shape[1] * shape[2] + 256, shape[2], 1)
            storage = torch.full(
                (shape[0] * strides[0],), 123.0, dtype=torch.float16, device="xpu"
            )
            view = storage.as_strided(shape, strides)
            view[:, : m + 2].copy_(cpu[name])
        else:
            storage = cpu[name + "_storage"].to("xpu")
            view = storage.as_strided(shape, strides)
        case[name + "_storage"] = storage
        case[name] = view
    for name in ("qkvz", "weight", "bias", "ba", "dt_bias"):
        case[name] = cpu[name].to("xpu")
    case["A_log"] = cpu["A_log"].to(
        device="xpu", dtype=torch.float32 if a_fp32 else torch.float16
    )
    case["ids"] = cpu["ids"].int().view(1, m).to("xpu")
    case["tokens"] = cpu["token_indices"].int().to("xpu")
    case["tokens_long"] = cpu["token_indices"].to("xpu")
    case["starts"] = torch.tensor([0, m], dtype=torch.int32, device="xpu")
    case["accepted"] = {
        n: torch.tensor([n], dtype=torch.int32, device="xpu") for n in range(1, m + 1)
    }
    case["output"] = torch.full((m, 6, 128), -123.0, dtype=torch.float16, device="xpu")
    case["z"] = torch.full_like(case["output"], -123.0)
    return case


def _native(case, accepted):
    torch.ops.custom_esimd_kernels_vllm.esimd_gdn_conv_fused_seq_spec_v2(
        case["qkvz"],
        case["conv"],
        case["weight"],
        case["bias"],
        case["ids"],
        case["A_log"],
        case["dt_bias"],
        case["ba"],
        case["ssm"].view(-1, 6, 128, 128),
        case["output"],
        case["z"],
        case["tokens"],
        case["accepted"][accepted],
        1,
        case["m"],
        2,
        6,
        128,
        128,
        128**-0.5,
    )


def _triton(case, accepted):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    m = case["m"]
    tokens = case["tokens_long"]
    # Same sequential QKV layout as prepare_gdn_attention_core_inputs. The
    # reference materializes its own input since causal_conv mutates x in place.
    mixed = case["qkvz"].index_select(0, tokens)[:, :1280].contiguous()
    mixed = causal_conv1d_update(
        mixed,
        case["conv"].transpose(-1, -2),
        case["weight"],
        case["bias"],
        "silu",
        conv_state_indices=case["ids"][:, 0],
        num_accepted_tokens=case["accepted"][accepted],
        query_start_loc=case["starts"],
        max_query_len=m,
        validate_data=False,
    )
    ba = case["ba"].index_select(0, tokens)
    result, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=case["A_log"],
        a=ba[:, 6:],
        b=ba[:, :6],
        dt_bias=case["dt_bias"],
        q=mixed[:, :256].reshape(1, m, 2, 128),
        k=mixed[:, 256:512].reshape(1, m, 2, 128),
        v=mixed[:, 512:].reshape(1, m, 6, 128),
        initial_state=case["ssm"].view(-1, 6, 128, 128),
        inplace_final_state=True,
        cu_seqlens=case["starts"],
        ssm_state_indices=case["ids"],
        num_accepted_tokens=case["accepted"][accepted],
        use_qk_l2norm_in_kernel=True,
    )
    case["output"].index_copy_(0, tokens, result[0])
    case["z"].copy_(case["qkvz"][:, 1280:].reshape(m, 6, 128))


def _compare(native, reference, label):
    for name in ("output", "ssm", "conv", "z"):
        got, want = native[name].cpu(), reference[name].cpu()
        error = (got.float() - want.float()).abs().max().item()
        print(f"{label} {name} max_abs={error:.9g}")
        exact = name in ("conv", "z")
        torch.testing.assert_close(
            got, want, rtol=0 if exact else 0.002, atol=0 if exact else 0.0002
        )
    for case in (native, reference):
        for name in ("conv", "ssm"):
            rows = (
                case[name + "_storage"]
                .cpu()
                .reshape(case[name].shape[0], case[name].stride(0))
            )
            assert torch.all(rows[:, -256:] == 123)
        # Unused physical conv rows must not be shifted or overwritten.
        assert torch.all(case["conv"][:, case["m"] + 2 :].cpu() == 123)


@pytest.fixture(scope="module", autouse=True)
def _load_native():
    if os.environ.get("GDN_STRICT_XPU") == "1":
        torch.ops.load_library(os.environ["GDN_SPEC_DSO"])


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("a_fp32", [False, True])
def test_all_acceptance_counts_multistep_actual_triton(m, a_fp32):
    native = _device_case(m, a_fp32=a_fp32, extra_rows=2)
    reference = _device_case(m, a_fp32=a_fp32, extra_rows=2)
    rng = torch.Generator(device="cpu").manual_seed(1907 + m)
    for step, accepted in enumerate([1, *range(m, 0, -1), m, 1]):
        # Every forward sees new token projections, while both providers keep
        # their own persistent rollback states across all acceptance patterns.
        qkvz = (torch.randn(m, 2048, generator=rng) * 0.75).half()
        ba = (torch.randn(m, 12, generator=rng) * 0.5).half()
        for case in (native, reference):
            case["qkvz"].copy_(qkvz)
            case["ba"].copy_(ba)
        _native(native, accepted)
        _triton(reference, accepted)
        _compare(native, reference, f"M={m} step={step} accepted={accepted}")


def test_independent_and_handoff_streams_have_no_shared_scratch():
    native_a, reference_a = _device_case(4, 104), _device_case(4, 104)
    native_b, reference_b = _device_case(6, 206), _device_case(6, 206)
    producer, consumer = torch.xpu.Stream(), torch.xpu.Stream()
    producer.wait_stream(torch.xpu.current_stream())
    consumer.wait_stream(torch.xpu.current_stream())
    ready_a, ready_b = torch.xpu.Event(), torch.xpu.Event()
    with torch.xpu.stream(producer):
        _native(native_a, 1)
        _triton(reference_a, 1)
        ready_a.record()
    with torch.xpu.stream(consumer):
        _native(native_b, 1)
        _triton(reference_b, 1)
        ready_b.record()
    with torch.xpu.stream(consumer):
        consumer.wait_event(ready_a)
        _native(native_a, 3)
        _triton(reference_a, 3)
    with torch.xpu.stream(producer):
        producer.wait_event(ready_b)
        _native(native_b, 5)
        _triton(reference_b, 5)
    producer.synchronize()
    consumer.synchronize()
    _compare(native_a, reference_a, "stream-handoff-A")
    _compare(native_b, reference_b, "stream-handoff-B")


@pytest.mark.parametrize("alias", ["z", "qkvz"])
def test_output_alias_is_rejected_before_mutating_state(alias):
    case = _device_case(4)
    before_conv, before_ssm = case["conv"].clone(), case["ssm"].clone()
    before_qkvz = case["qkvz"].clone()
    if alias == "z":
        case["z"] = case["output"]
    else:
        case["output"] = case["qkvz"].view(-1)[: 4 * 6 * 128].view(4, 6, 128)
    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        _native(case, 1)
    torch.testing.assert_close(case["conv"], before_conv, rtol=0, atol=0)
    torch.testing.assert_close(case["ssm"], before_ssm, rtol=0, atol=0)
    torch.testing.assert_close(case["qkvz"], before_qkvz, rtol=0, atol=0)
