"""Microbenchmark for the Qwen3.8 fixed-metadata decode N-gram ID op."""
import json
import statistics
import sys

import torch

from custom_esimd_kernels_vllm import (
    QWEN38_NGRAM_OFFSETS,
    QWEN38_NGRAM_VOCAB_SIZES,
    esimd_qwen38_ngram_ids_decode,
    esimd_qwen38_ngram_ids_decode_out,
)


MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)


def reference(inputs, vocab, offsets):
    input_ids, ngram_context, multipliers = inputs
    current = input_ids[0]
    previous_2 = ngram_context[0, 0]
    previous = ngram_context[0, 1]
    bigram = (current * multipliers[0]) ^ (previous * multipliers[1])
    trigram = bigram ^ (previous_2 * multipliers[2])
    mixed = torch.cat((bigram.expand(8), trigram.expand(8)))
    return (torch.remainder(mixed, vocab) + offsets).reshape(1, 16)


def make_inputs():
    device = torch.device("xpu:0")
    return (
        torch.tensor([271], dtype=torch.int64, device=device),
        torch.tensor([[11316, 13]], dtype=torch.int64, device=device),
        torch.tensor(MULTIPLIERS, dtype=torch.int64, device=device),
    )


def time_call(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    warmup = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    repetitions = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    interleaved_repetitions = int(sys.argv[4]) if len(sys.argv) > 4 else 15

    inputs = make_inputs()
    output = torch.empty((1, 16), dtype=torch.int64, device="xpu:0")
    vocab = torch.tensor(
        QWEN38_NGRAM_VOCAB_SIZES, dtype=torch.int64, device="xpu:0")
    offsets = torch.tensor(
        QWEN38_NGRAM_OFFSETS, dtype=torch.int64, device="xpu:0")
    eager_call = lambda: reference(inputs, vocab, offsets)
    kernel_call = lambda: esimd_qwen38_ngram_ids_decode(*inputs)
    kernel_out_call = lambda: esimd_qwen38_ngram_ids_decode_out(*inputs, output)

    expected = eager_call()
    actual = kernel_call()
    actual_out = kernel_out_call()
    torch.xpu.synchronize()
    if not torch.equal(actual, expected) or not torch.equal(actual_out, expected):
        raise AssertionError("kernel output does not match eager reference")

    eager_samples = sorted(
        time_call(eager_call, warmup, iterations) for _ in range(repetitions)
    )
    kernel_samples = sorted(
        time_call(kernel_call, warmup, iterations) for _ in range(repetitions)
    )
    kernel_out_samples = sorted(
        time_call(kernel_out_call, warmup, iterations) for _ in range(repetitions)
    )

    interleaved_eager = []
    interleaved_kernel = []
    interleaved_kernel_out = []
    for _ in range(interleaved_repetitions):
        interleaved_eager.append(time_call(eager_call, warmup, iterations))
        interleaved_kernel.append(time_call(kernel_call, warmup, iterations))
        interleaved_kernel_out.append(time_call(kernel_out_call, warmup, iterations))

    eager_median = statistics.median(eager_samples)
    kernel_median = statistics.median(kernel_samples)
    kernel_out_median = statistics.median(kernel_out_samples)
    interleaved_eager_median = statistics.median(interleaved_eager)
    interleaved_kernel_median = statistics.median(interleaved_kernel)
    interleaved_kernel_out_median = statistics.median(interleaved_kernel_out)
    result = {
        "device": torch.xpu.get_device_name(0),
        "warmup": warmup,
        "iterations": iterations,
        "repetitions": repetitions,
        "eager_samples_ms": eager_samples,
        "kernel_samples_ms": kernel_samples,
        "kernel_out_samples_ms": kernel_out_samples,
        "eager_median_ms": eager_median,
        "kernel_median_ms": kernel_median,
        "kernel_out_median_ms": kernel_out_median,
        "speedup": eager_median / kernel_median,
        "out_speedup": eager_median / kernel_out_median,
        "interleaved_repetitions": interleaved_repetitions,
        "interleaved_eager_samples_ms": interleaved_eager,
        "interleaved_kernel_samples_ms": interleaved_kernel,
        "interleaved_kernel_out_samples_ms": interleaved_kernel_out,
        "interleaved_eager_median_ms": interleaved_eager_median,
        "interleaved_kernel_median_ms": interleaved_kernel_median,
        "interleaved_kernel_out_median_ms": interleaved_kernel_out_median,
        "interleaved_speedup": interleaved_eager_median / interleaved_kernel_median,
        "interleaved_out_speedup": interleaved_eager_median / interleaved_kernel_out_median,
        "max_abs_err": int((actual - expected).abs().max().item()),
    }
    print("RESULT_JSON_START")
    print(json.dumps(result, sort_keys=True))
    print("RESULT_JSON_END")
    print(
        f"eager={eager_median * 1000:.3f} us  "
        f"kernel={kernel_median * 1000:.3f} us  "
        f"speedup={result['speedup']:.2f}x"
    )
    print(
        f"out={kernel_out_median * 1000:.3f} us  "
        f"speedup={result['out_speedup']:.2f}x"
    )
    print(
        f"interleaved eager={interleaved_eager_median * 1000:.3f} us  "
        f"kernel={interleaved_kernel_median * 1000:.3f} us  "
        f"out={interleaved_kernel_out_median * 1000:.3f} us  "
        f"speedup={result['interleaved_speedup']:.2f}x  "
        f"out_speedup={result['interleaved_out_speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
