import pytest
import torch

from omni_xpu_kernel import int8, rotary


pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU unavailable")


def _new_stream():
    try:
        return torch.xpu.Stream()
    except RuntimeError as exc:
        pytest.skip(f"additional XPU streams are unavailable: {exc}")


def _adjacent_reference(x, freqs):
    paired = x.to(freqs.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    output = freqs[..., 0] * paired[..., 0]
    output.addcmul_(freqs[..., 1], paired[..., 1])
    return output.reshape_as(x).type_as(x)


def _split_reference(x, freqs):
    split = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    split = split.to(freqs.dtype)
    output = freqs[..., 0] * split[..., 0] + freqs[..., 1] * split[..., 1]
    return output.movedim(-1, -2).reshape_as(x).type_as(x)


def test_non_default_stream_orders_sycl_and_onednn_kernels():
    """A single custom stream preserves producer/consumer dependencies."""
    torch.manual_seed(123)
    x_cpu = torch.randn(16, 128, dtype=torch.bfloat16)
    weight_cpu = torch.randn(64, 128, dtype=torch.bfloat16)
    down_weight_cpu = torch.randn(128, 64, dtype=torch.bfloat16)
    rope_x_cpu = torch.randn(2, 3, 31, 64, dtype=torch.bfloat16)
    freqs_cpu = torch.randn(2, 1, 31, 32, 2, 2, dtype=torch.float32)

    stream = _new_stream()
    with torch.xpu.stream(stream):
        x = x_cpu.to("xpu")
        weight = weight_cpu.to("xpu")
        qweight, weight_scale = int8.quantize_int8_tensorwise(weight)
        linear_out = int8.int8_linear(
            x, qweight, weight_scale, out_dtype=torch.bfloat16
        )
        qactivation, activation_scale = int8.quantize_int8_rowwise(x)
        prequantized_out = int8.int8_linear_prequantized(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            out_dtype=torch.bfloat16,
        )
        shared1, shared2 = int8.int8_linear_shared_input(
            x,
            qweight,
            weight_scale,
            qweight,
            weight_scale,
            out_dtype=torch.bfloat16,
        )
        gated_q, gated_scale = int8.fused_silu_mul_quantize_rowwise(
            shared1, shared2
        )
        down_weight = down_weight_cpu.to("xpu")
        qdown_weight, down_weight_scale = int8.quantize_int8_tensorwise(
            down_weight
        )
        mlp_out = int8.int8_linear_prequantized(
            gated_q,
            gated_scale,
            qdown_weight,
            down_weight_scale,
            out_dtype=torch.bfloat16,
        )
        convrot_weight, convrot_weight_scale = (
            int8.quantize_int8_convrot_weight(weight, group_size=64)
        )
        convrot_shared1, convrot_shared2 = int8.int8_linear_shared_input(
            x,
            convrot_weight,
            convrot_weight_scale,
            convrot_weight,
            convrot_weight_scale,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=64,
        )
        convrot_gated = int8.fused_silu_mul(
            convrot_shared1, convrot_shared2
        )
        convrot_gated = int8.rotate_convrot(convrot_gated, group_size=64)
        convrot_q, convrot_scale = int8.quantize_int8_rowwise(convrot_gated)
        convrot_down_weight, convrot_down_weight_scale = (
            int8.quantize_int8_convrot_weight(down_weight, group_size=64)
        )
        convrot_mlp_out = int8.int8_linear_prequantized(
            convrot_q,
            convrot_scale,
            convrot_down_weight,
            convrot_down_weight_scale,
            out_dtype=torch.bfloat16,
        )

        rope_x = rope_x_cpu.to("xpu")
        freqs = freqs_cpu.to("xpu")
        rope_out = rotary.apply_kitchen_rope1(rope_x, freqs)

    stream.synchronize()

    # Re-running the oneDNN consumer on the default stream must produce the
    # same result from the custom-stream-produced quantized weight.
    expected_linear = int8.int8_linear(
        x, qweight, weight_scale, out_dtype=torch.bfloat16
    )
    expected_prequantized = int8.int8_linear_prequantized(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        out_dtype=torch.bfloat16,
    )
    expected_shared1, expected_shared2 = int8.int8_linear_shared_input(
        x,
        qweight,
        weight_scale,
        qweight,
        weight_scale,
        out_dtype=torch.bfloat16,
    )
    expected_gated_q, expected_gated_scale = (
        int8.fused_silu_mul_quantize_rowwise(expected_shared1, expected_shared2)
    )
    expected_mlp = int8.int8_linear_prequantized(
        expected_gated_q,
        expected_gated_scale,
        qdown_weight,
        down_weight_scale,
        out_dtype=torch.bfloat16,
    )
    expected_convrot_shared1, expected_convrot_shared2 = (
        int8.int8_linear_shared_input(
            x,
            convrot_weight,
            convrot_weight_scale,
            convrot_weight,
            convrot_weight_scale,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=64,
        )
    )
    expected_convrot_gated = int8.fused_silu_mul(
        expected_convrot_shared1, expected_convrot_shared2
    )
    expected_convrot_gated = int8.rotate_convrot(
        expected_convrot_gated, group_size=64
    )
    expected_convrot_q, expected_convrot_scale = int8.quantize_int8_rowwise(
        expected_convrot_gated
    )
    expected_convrot_mlp = int8.int8_linear_prequantized(
        expected_convrot_q,
        expected_convrot_scale,
        convrot_down_weight,
        convrot_down_weight_scale,
        out_dtype=torch.bfloat16,
    )
    expected_rope = _adjacent_reference(rope_x, freqs)
    torch.testing.assert_close(linear_out, expected_linear, rtol=0, atol=0)
    torch.testing.assert_close(
        prequantized_out, expected_prequantized, rtol=0, atol=0
    )
    torch.testing.assert_close(mlp_out, expected_mlp, rtol=0, atol=0)
    torch.testing.assert_close(
        convrot_mlp_out, expected_convrot_mlp, rtol=0, atol=0
    )
    torch.testing.assert_close(rope_out, expected_rope, rtol=0, atol=0)


def test_two_streams_minimal_independent_rope_validation():
    """Minimal correctness check only; no concurrency/performance guarantee."""
    torch.manual_seed(456)
    x1_cpu = torch.randn(1, 4, 257, 64, dtype=torch.bfloat16)
    x2_cpu = torch.randn(1, 257, 3, 64, dtype=torch.float16)
    f1_cpu = torch.randn(1, 1, 257, 32, 2, 2, dtype=torch.float32)
    f2_cpu = torch.randn(1, 257, 1, 32, 2, 2, dtype=torch.float16)
    stream1, stream2 = _new_stream(), _new_stream()

    with torch.xpu.stream(stream1):
        x1 = x1_cpu.to("xpu")
        f1 = f1_cpu.to("xpu")
        out1 = rotary.apply_kitchen_rope1(x1, f1)
    with torch.xpu.stream(stream2):
        x2 = x2_cpu.to("xpu")
        f2 = f2_cpu.to("xpu")
        out2 = rotary.apply_kitchen_rope_split_half1(x2, f2)

    stream1.synchronize()
    stream2.synchronize()
    torch.testing.assert_close(out1, _adjacent_reference(x1, f1), rtol=0, atol=0)
    torch.testing.assert_close(out2, _split_reference(x2, f2), rtol=0, atol=0)
