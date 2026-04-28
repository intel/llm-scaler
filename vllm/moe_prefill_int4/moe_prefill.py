"""
Standalone INT4 MoE Forward — directly calls torch.ops.torch_ipex ops
without using GatedMLPMOE wrapper (avoids in-place weight mutation and
potential hang issues).

This module reimplements the GatedMLPMOE.forward logic by calling the
same underlying IPEX SYCL kernels (topk_softmax, moe_scatter, moe_gemm,
silu_and_mul, moe_gather) but manages weight layout explicitly.

Usage:
    from moe_prefill_int4 import StandaloneInt4MoE
    moe = StandaloneInt4MoE(w13_qweight, w2_qweight, w13_scales, w2_scales, ...)
    output = moe(hidden_states, router_logits, top_k=8, renormalize=True)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import intel_extension_for_pytorch as ipex  # noqa: F401

DEVICE = "xpu"


def marlin_shuffle_weight(qweight):
    """Reorder nibbles within each int32 to match IPEX XeTLA marlin format.

    Input:  qweight [E, K_packed, N] int32 (natural nibble order)
    Output: shuffled [E, K_packed, N] int32 (marlin nibble order)

    IPEX shuffled_idx = [0, 4, 1, 5, 2, 6, 3, 7]
    """
    E, K, N = qweight.shape
    k = K * 8
    shuffled_idx = np.array([0, 4, 1, 5, 2, 6, 3, 7])
    pack_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    shuffled_weight = torch.zeros(
        [E, k // 8, N], dtype=torch.int32, device=qweight.device
    )

    for e in range(E):
        data = qweight[e][[i // 8 for i in range(k)], :]
        shift = (
            torch.tensor(
                shuffled_idx[[i % 8 for i in range(k)]],
                dtype=torch.int32,
                device=qweight.device,
            )[:, None].expand([-1, N])
            * 4
        )
        dst_data = (data >> shift) & 0xF

        shift_pack = (
            torch.tensor(
                pack_idx[[i % 8 for i in range(k)]],
                dtype=torch.int32,
                device=qweight.device,
            )[:, None].expand([-1, N])
            * 4
        )
        dst_data = dst_data << shift_pack

        for i in range(0, k, 8):
            tmp = dst_data[i, :]
            for j in range(i + 1, i + 8):
                tmp = torch.bitwise_or(tmp, dst_data[j, :])
            shuffled_weight[e, i // 8, :] = tmp

    return shuffled_weight


class StandaloneInt4MoE(nn.Module):
    """Standalone INT4 MoE using IPEX's low-level ops directly.

    Accepts weights in GGML N-major format [E, N, K_packed] (same as
    ggml_quantize_tensor output with transpose=False) and prepares them
    for IPEX's moe_gemm kernel internally.

    Key difference from GatedMLPMOE:
      - Does NOT mutate the original weight Parameter's .data
      - Weight transform (transpose + marlin shuffle) creates new tensors
      - All transforms happen in __init__, not lazily in first forward
    """

    def __init__(
        self,
        w13_qweight,       # [E, 2*D, K_packed] int32 — GGML N-major
        w2_qweight,        # [E, H, D_packed] int32 — GGML N-major
        w13_scales,        # [E, 2*D, K_groups] fp16
        w2_scales,         # [E, H, D_groups] fp16
        num_experts=None,
    ):
        super().__init__()
        self.num_experts = w13_qweight.shape[0] if num_experts is None else num_experts

        # Transform to IPEX K-major format: [E, K_packed, N]
        # Use .clone() to avoid mutating the original tensors
        w13_t = w13_qweight.transpose(1, 2).contiguous()
        w2_t = w2_qweight.transpose(1, 2).contiguous()

        # Marlin shuffle
        w13_shuffled = marlin_shuffle_weight(w13_t)
        w2_shuffled = marlin_shuffle_weight(w2_t)

        # Transpose scales: [E, N, K_groups] -> [E, K_groups, N]
        w13_scales_t = w13_scales.transpose(1, 2).contiguous()
        w2_scales_t = w2_scales.transpose(1, 2).contiguous()

        # Store as buffers (not parameters — won't appear in state_dict)
        self.register_buffer('W13', w13_shuffled)
        self.register_buffer('W2', w2_shuffled)
        self.register_buffer('w13_scale', w13_scales_t)
        self.register_buffer('w2_scale', w2_scales_t)

    def forward(
        self,
        hidden_states,          # [N, H] fp16
        use_grouped_topk,       # bool
        top_k,                  # int
        router_logits,          # [N, E] fp16
        renormalize,            # bool
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=None,
    ):
        num_tokens, hidden_dim = hidden_states.shape

        # ── Step 1: TopK routing ──
        if custom_routing_function is not None:
            routing_weights, selected_experts = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=False,
            )
            routing_weights = routing_weights.to(torch.float)
            selected_experts = selected_experts.to(torch.int32)
        elif not use_grouped_topk:
            routing_weights = torch.empty(
                num_tokens, top_k, dtype=torch.float32, device=hidden_states.device
            )
            selected_experts = torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
            )
            token_expert_indices = torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
            )
            torch.ops.torch_ipex.topk_softmax(
                routing_weights,
                selected_experts,
                token_expert_indices,
                router_logits,
                False,
            )
        elif use_grouped_topk:
            routing_weights, selected_experts = torch.ops.torch_ipex.grouped_topk(
                hidden_states,
                router_logits,
                top_k,
                False,
                num_expert_group,
                topk_group,
                scoring_func,
                e_score_correction_bias,
            )
        else:
            raise ValueError("Invalid routing configuration")

        # ── Step 2: Row counting ──
        rows_for_experts, expert_offsets = torch.ops.torch_ipex.moe_rows_counts(
            selected_experts, 0, self.num_experts
        )

        # ── Step 3: Scatter ──
        reordered_hidden_states, mapped_slot = torch.ops.torch_ipex.moe_scatter(
            hidden_states,
            rows_for_experts,
            selected_experts,
            expert_offsets,
            0,  # experts_start_id
            self.num_experts,
            top_k,
        )

        # ── Step 4: Expert GEMM (W13: gate_up) ──
        reordered_hidden_states = torch.xpu.moe_gemm(
            reordered_hidden_states,
            self.W13,
            rows_for_experts,
            self.num_experts,
            None,  # a1_scale_inv
            self.w13_scale,
            bias=None,
            is_int4=True,
        )

        # ── Step 5: SiLU activation ──
        half = reordered_hidden_states.shape[-1] // 2
        out = torch.empty(
            reordered_hidden_states.shape[:-1] + (half,),
            dtype=reordered_hidden_states.dtype,
            device=reordered_hidden_states.device,
        )
        reordered_hidden_states = torch.ops.torch_ipex.silu_and_mul(
            reordered_hidden_states, out
        )

        # ── Step 6: Expert GEMM (W2: down) ──
        reordered_hidden_states = torch.xpu.moe_gemm(
            reordered_hidden_states,
            self.W2,
            rows_for_experts,
            self.num_experts,
            None,  # a2_scale_inv
            self.w2_scale,
            bias=None,
            is_int4=True,
        )

        # ── Step 7: Gather ──
        moe_output = torch.ops.torch_ipex.moe_gather(
            reordered_hidden_states,
            routing_weights,
            mapped_slot,
            self.num_experts,
            top_k,
            renormalize,
        )
        return moe_output.reshape(num_tokens, hidden_dim)
