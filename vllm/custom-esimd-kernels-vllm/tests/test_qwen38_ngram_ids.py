"""Correctness and safety checks for the Qwen3.8 decode N-gram ID op."""
import random
import sys

import torch

from custom_esimd_kernels_vllm import (
    QWEN38_NGRAM_OFFSETS,
    QWEN38_NGRAM_VOCAB_SIZES,
    esimd_qwen38_ngram_ids_decode,
    esimd_qwen38_ngram_ids_decode_out,
)


MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
FROZEN_OUTPUT = (
    5129529, 22831471, 51687175, 68090399,
    90726650, 118273134, 139593745, 154689607,
    177313122, 183091298, 216586780, 229949691,
    253020612, 276108843, 283808766, 302337060,
)


def require_xpu():
    if not torch.xpu.is_available():
        raise RuntimeError("PyTorch XPU is unavailable")
    return torch.device("xpu:0")


def make_inputs(device, current=271, previous_2=11316, previous=13,
                multipliers=MULTIPLIERS):
    return (
        torch.tensor([current], dtype=torch.int64, device=device),
        torch.tensor([[previous_2, previous]], dtype=torch.int64, device=device),
        torch.tensor(multipliers, dtype=torch.int64, device=device),
    )


def reference(inputs):
    input_ids, ngram_context, multipliers = inputs
    vocab = torch.tensor(
        QWEN38_NGRAM_VOCAB_SIZES, dtype=torch.int64, device=input_ids.device)
    offsets = torch.tensor(
        QWEN38_NGRAM_OFFSETS, dtype=torch.int64, device=input_ids.device)
    current = input_ids[0]
    previous_2 = ngram_context[0, 0]
    previous = ngram_context[0, 1]
    bigram = (current * multipliers[0]) ^ (previous * multipliers[1])
    trigram = bigram ^ (previous_2 * multipliers[2])
    mixed = torch.cat((bigram.expand(8), trigram.expand(8)))
    return (torch.remainder(mixed, vocab) + offsets).reshape(1, 16)


def assert_exact(inputs, label):
    actual = esimd_qwen38_ngram_ids_decode(*inputs)
    expected = reference(inputs)
    torch.xpu.synchronize()
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"{label}:\nactual={actual.cpu()}\nexpected={expected.cpu()}")


def expect_error(inputs, text):
    try:
        esimd_qwen38_ngram_ids_decode(*inputs)
        torch.xpu.synchronize()
    except RuntimeError as exc:
        if text not in str(exc):
            raise AssertionError(f"expected {text!r}, got {exc!r}") from exc
        return
    raise AssertionError(f"expected RuntimeError containing {text!r}")


def test_frozen_capture_and_out_contract():
    device = require_xpu()
    inputs = make_inputs(device)
    expected = torch.tensor(FROZEN_OUTPUT, dtype=torch.int64).reshape(1, 16)

    actual = esimd_qwen38_ngram_ids_decode(*inputs)
    output = torch.empty((1, 16), dtype=torch.int64, device=device)
    raw_return = torch.ops.custom_esimd_kernels_vllm.esimd_qwen38_ngram_ids_decode_out(
        *inputs, output)
    if raw_return is not None:
        raise AssertionError("raw out op must follow its void dispatcher schema")
    returned = esimd_qwen38_ngram_ids_decode_out(*inputs, output)
    if returned.data_ptr() != output.data_ptr():
        raise AssertionError("Python out wrapper did not return the supplied output")

    torch.xpu.synchronize()
    if not torch.equal(actual.cpu(), expected) or not torch.equal(output.cpu(), expected):
        raise AssertionError("frozen capture mismatch")


def test_exact_remainder_edges_and_random_inputs():
    device = require_xpu()
    targets = {-(1 << 63), (1 << 63) - 1, -1, 0, 1}
    for vocab in QWEN38_NGRAM_VOCAB_SIZES:
        targets.update((-vocab - 1, -vocab, -vocab + 1,
                        vocab - 1, vocab, vocab + 1))
    for index, value in enumerate(sorted(targets)):
        assert_exact(
            make_inputs(device, current=value, previous_2=0, previous=0,
                        multipliers=(1, 0, 0)),
            f"bigram_edge_{index}",
        )
        assert_exact(
            make_inputs(device, current=0, previous_2=value, previous=0,
                        multipliers=(0, 0, 1)),
            f"trigram_edge_{index}",
        )

    rng = random.Random(0x38)
    pending = []
    for index in range(128):
        values = tuple(rng.randint(-(1 << 63), (1 << 63) - 1) for _ in range(6))
        inputs = make_inputs(device, *values[:3], multipliers=values[3:])
        pending.append((
            esimd_qwen38_ngram_ids_decode(*inputs), reference(inputs), index))
    torch.xpu.synchronize()
    for actual, expected, index in pending:
        if not torch.equal(actual, expected):
            raise AssertionError(f"random_{index} mismatch")


def test_async_allocating_outputs_are_independent():
    device = require_xpu()
    inputs = make_inputs(device)
    expected = reference(inputs)
    results = [esimd_qwen38_ngram_ids_decode(*inputs) for _ in range(128)]
    if len({tensor.data_ptr() for tensor in results}) != len(results):
        raise AssertionError("allocating calls reused a live output allocation")
    torch.xpu.synchronize()
    for index, result in enumerate(results):
        if not torch.equal(result, expected):
            raise AssertionError(f"repeated_{index} mismatch")


def test_current_stream_and_reused_output():
    device = require_xpu()
    stream = torch.xpu.Stream(device=device)
    output = torch.empty((1, 16), dtype=torch.int64, device=device)
    last_expected = None
    with torch.xpu.stream(stream):
        for value in range(64):
            inputs = make_inputs(
                device, current=value - 32, previous_2=value * 17,
                previous=-value, multipliers=MULTIPLIERS)
            last_expected = reference(inputs)
            returned = esimd_qwen38_ngram_ids_decode_out(*inputs, output)
            if returned.data_ptr() != output.data_ptr():
                raise AssertionError("out wrapper replaced the output tensor")
            del inputs
            torch.empty((4096,), dtype=torch.int64, device=device).fill_(value)
    stream.synchronize()
    if not torch.equal(output, last_expected):
        raise AssertionError("non-default stream or reused-output ordering mismatch")


def test_validation_errors():
    device = require_xpu()
    inputs = list(make_inputs(device))

    bad = inputs.copy()
    bad[0] = bad[0].to(dtype=torch.int32)
    expect_error(tuple(bad), "input_ids must have dtype int64")

    bad = inputs.copy()
    bad[2] = torch.arange(6, dtype=torch.int64, device=device)[::2]
    expect_error(tuple(bad), "layer_multipliers must be contiguous")

    bad = inputs.copy()
    bad[1] = bad[1].reshape(2)
    expect_error(tuple(bad), "ngram_context must have shape [1, 2]")

    bad = inputs.copy()
    bad[1] = bad[1].cpu()
    expect_error(tuple(bad), "ngram_context must be on XPU")

    try:
        esimd_qwen38_ngram_ids_decode_out(
            *inputs, torch.empty((16,), dtype=torch.int64, device=device))
    except RuntimeError as exc:
        if "output must have shape [1, 16]" not in str(exc):
            raise
    else:
        raise AssertionError("out variant accepted the wrong output shape")


def test_cross_device_rejected_when_available():
    if torch.xpu.device_count() < 2:
        return
    inputs = list(make_inputs(torch.device("xpu:0")))
    inputs[1] = inputs[1].to("xpu:1")
    expect_error(tuple(inputs), "same XPU device")


def main():
    print(f"device={torch.xpu.get_device_name(require_xpu())}")
    tests = (
        test_frozen_capture_and_out_contract,
        test_exact_remainder_edges_and_random_inputs,
        test_async_allocating_outputs_are_independent,
        test_current_stream_and_reused_output,
        test_validation_errors,
        test_cross_device_rejected_when_available,
    )
    for test in tests:
        test()
        print(f"{test.__name__}=PASS")
    print("ALL_TESTS=PASS")


if __name__ == "__main__":
    sys.exit(main())
