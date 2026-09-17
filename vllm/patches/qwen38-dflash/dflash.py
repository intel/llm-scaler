# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            # Request aux hidden states; DFlash turns them into context K/V.
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", {}
        )
        self.sliding_attention_causal = dflash_config.get(
            "sliding_attention_causal", True
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffers_by_gid: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._draft_block_size_by_gid: dict[int, int] = {}
        self._draft_block_tables: dict[int, torch.Tensor] = {}
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # DFlash embeds mask tokens directly.
        self.parallel_drafting_hidden_state_tensor = None
        self._packed_target_hidden_states: torch.Tensor | None = None

        # DSpark: previous-token context for the Markov bigram bias. Set in
        # propose() before the first sampling; None when the draft model has
        # no Markov head (plain DFlash).
        self._markov_prev_tokens: torch.Tensor | None = None

        # DSpark adaptive block length (confidence head). Off by default: for
        # small bandwidth-bound target batches a full draft block is optimal;
        # truncation pays when verify cost grows with width (larger batches,
        # graph padding). Enable with DSPARK_ADAPTIVE_BLOCK=1.
        self._dspark_adaptive = os.environ.get("DSPARK_ADAPTIVE_BLOCK", "0") == "1"
        self._dspark_conf_threshold = float(
            os.environ.get("DSPARK_CONF_THRESHOLD", "0.5")
        )
        self._dspark_min_tokens = max(
            1, int(os.environ.get("DSPARK_MIN_DRAFT_TOKENS", "1"))
        )
        # Per-position predicted acceptance probability, filled by
        # _greedy_sample when adaptive mode is on; consumed by propose().
        self._draft_confidence: torch.Tensor | None = None
        if self._dspark_adaptive:
            logger.info(
                "DSpark adaptive block length enabled "
                "(threshold=%.3f, min_tokens=%d)",
                self._dspark_conf_threshold,
                self._dspark_min_tokens,
            )

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ):
        # For the first draft position the "previous token" is the accepted
        # bonus token, which is known at proposal time. Capture it so
        # _greedy_sample can build the Markov bigram bias. Block-N drafting
        # chains the context semi-autoregressively inside _greedy_sample.
        self._draft_confidence = None
        if getattr(self.model, "markov_head", None) is not None:
            self._markov_prev_tokens = next_token_ids
        else:
            self._markov_prev_tokens = None
        draft_token_ids = super().propose(
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs=mm_embed_inputs,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        )
        if (
            self._dspark_adaptive
            and self._draft_confidence is not None
            and torch.is_tensor(draft_token_ids)
            and draft_token_ids.dim() == 2
        ):
            return self._truncate_by_confidence(draft_token_ids)
        return draft_token_ids

    def _truncate_by_confidence(
        self, draft_token_ids: torch.Tensor
    ) -> list[list[int]]:
        """Per-request prefix stopping driven by the DSpark confidence head.

        The confidence head predicts, per draft position, the probability that
        the position's draft token survives target verification given all
        earlier positions survived (DSpark paper: c_k = sigmoid(w^T [h_k;
        W1[x_{k-1}]]). We propose the longest prefix whose predicted
        acceptance stays at or above the threshold. Variable-width draft lists
        flow through vLLM v1's scheduler and verify paths unchanged (both
        handle per-request draft lengths).
        """
        conf = self._draft_confidence
        keep = (conf >= self._dspark_conf_threshold).long()
        lengths = keep.cumprod(dim=1).sum(dim=1).clamp(min=self._dspark_min_tokens)
        tokens = draft_token_ids.tolist()
        lens = lengths.tolist()
        truncated = [row[:n] for row, n in zip(tokens, lens)]
        if logger.isEnabledFor(10):  # DEBUG
            logger.debug(
                "DSpark adaptive widths: %s (conf=%s)",
                [len(row) for row in truncated],
                conf.tolist(),
            )
        return truncated

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        model = self.model
        markov_head = getattr(model, "markov_head", None)
        if markov_head is None or self._markov_prev_tokens is None:
            return super()._greedy_sample(hidden_states)

        batch_size = self._markov_prev_tokens.shape[0]
        num_spec = self.num_speculative_tokens

        # compute_logits all-gathers to the full vocabulary on XPU, so the
        # full-vocab Markov bias is added directly before argmax.
        if num_spec == 1 or hidden_states.shape[0] == batch_size:
            # Single-token (block-1) case: one previous token (the accepted
            # bonus token) per request.
            logits = model.compute_logits(hidden_states)
            if logits is None:
                return super()._greedy_sample(hidden_states)
            bias = model.compute_markov_bias(self._markov_prev_tokens)
            if bias is not None:
                logits = logits + bias.to(logits.dtype)
            return logits.argmax(dim=-1)

        if hidden_states.shape[0] == batch_size * num_spec and num_spec > 1:
            # Block-N parallel drafting (DFlash/DSpark). hidden_states is laid
            # out positions-contiguous per request: [b0_p0..pN, b1_p0..pN, ...].
            # Apply the Markov bigram bias semi-autoregressively (DSpark paper:
            # at inference each position conditions on the previously *drafted*
            # token): position i's bias is conditioned on position i-1's
            # (biased) prediction, with position 0 conditioned on the accepted
            # bonus token.
            logits = model.compute_logits(hidden_states)
            if logits is None:
                return super()._greedy_sample(hidden_states)
            vocab = logits.shape[-1]
            logits = logits.view(batch_size, num_spec, vocab)
            if os.environ.get("DFLASH_DEBUG") == "1" and not getattr(
                self, "_dbg_logits", False
            ):
                self._dbg_logits = True
                with torch.no_grad():
                    hs = hidden_states.detach().float()
                    lg = logits.detach().float()
                    prev = self._markov_prev_tokens.detach()
                    logger.info(
                        "DFLASH_DBG greedy_blockN hidden shape=%s mean=%.4f "
                        "std=%.4f nan=%s | logits shape=%s mean=%.4f std=%.4f "
                        "max=%.4f | prev(bonus)=%s | top1=%s",
                        tuple(hidden_states.shape),
                        hs.mean().item(),
                        hs.std().item(),
                        bool(torch.isnan(hs).any().item()),
                        tuple(logits.shape),
                        lg.mean().item(),
                        lg.std().item(),
                        lg.max().item(),
                        prev.tolist(),
                        lg[0].argmax(dim=-1).tolist(),
                    )
            conf_head = getattr(model, "confidence_head", None)
            want_conf = self._dspark_adaptive and conf_head is not None
            confidences = None
            block_hidden = None
            if want_conf:
                confidences = torch.empty(
                    batch_size,
                    num_spec,
                    device=logits.device,
                    dtype=torch.float32,
                )
                block_hidden = hidden_states.view(batch_size, num_spec, -1)
            prev_tokens = self._markov_prev_tokens
            for i in range(num_spec):
                latent = markov_head.get_prev_embeddings(prev_tokens)
                bias = markov_head.project_bias(latent)
                logits[:, i, :] = logits[:, i, :] + bias.to(logits.dtype)
                if want_conf:
                    # c_k = sigmoid(w^T [h_k; W1[x_{k-1}]]) (DSpark paper).
                    param_dtype = conf_head.proj.weight.dtype
                    feats = torch.cat(
                        [
                            block_hidden[:, i, :].to(param_dtype),
                            latent.to(param_dtype),
                        ],
                        dim=-1,
                    )
                    confidences[:, i] = torch.sigmoid(conf_head(feats)).float()
                # Next position's context is this position's prediction.
                prev_tokens = logits[:, i, :].argmax(dim=-1)
            self._draft_confidence = confidences
            return logits.argmax(dim=-1)

        return super()._greedy_sample(hidden_states)

    def pack_aux_hidden_states(
        self,
        aux_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
        num_tokens: int,
        token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack target auxiliary states into a reusable contiguous buffer."""
        if not aux_hidden_states:
            raise ValueError("DFlash requires at least one auxiliary hidden state")
        if token_indices is not None:
            assert token_indices.shape[0] == num_tokens

        hidden_size = sum(hidden_state.shape[-1] for hidden_state in aux_hidden_states)
        dtype = self.dtype

        buffer = self._packed_target_hidden_states
        layout_matches = (
            buffer is not None
            and buffer.shape[1] == hidden_size
            and buffer.dtype == dtype
            and buffer.device == self.device
        )
        if not layout_matches or buffer.shape[0] < num_tokens:
            capacity = num_tokens
            if layout_matches:
                capacity = min(
                    self.max_num_tokens,
                    max(num_tokens, buffer.shape[0] * 2),
                )
            self._packed_target_hidden_states = torch.empty(
                (capacity, hidden_size),
                dtype=dtype,
                device=self.device,
            )
            buffer = self._packed_target_hidden_states

        packed = buffer[:num_tokens]
        offset = 0
        for hidden_state in aux_hidden_states:
            width = hidden_state.shape[-1]
            source = (
                hidden_state[:num_tokens]
                if token_indices is None
                else hidden_state.index_select(0, token_indices)
            )
            packed[:, offset : offset + width].copy_(source)
            offset += width
        if os.environ.get("DFLASH_DEBUG") == "1" and not getattr(
            self, "_dbg_pack", False
        ):
            self._dbg_pack = True
            with torch.no_grad():
                stats = []
                for hidden_state in aux_hidden_states:
                    hs = hidden_state[:num_tokens].detach().float()
                    stats.append(
                        f"shape={tuple(hidden_state.shape)} "
                        f"mean={hs.mean().item():.4f} std={hs.std().item():.4f} "
                        f"nan={bool(torch.isnan(hs).any().item())}"
                    )
                pk = packed.detach().float()
                logger.info(
                    "DFLASH_DBG pack_aux n_layers=%d | %s | PACKED shape=%s "
                    "mean=%.4f std=%.4f",
                    len(aux_hidden_states),
                    " || ".join(stats),
                    tuple(packed.shape),
                    pk.mean().item(),
                    pk.std().item(),
                )
        return packed

    @override
    def allow_multiple_draft_kv_cache_groups(self) -> bool:
        return True

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._draft_block_size_by_gid.clear()
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            self._draft_block_size_by_gid[gid] = (
                kernel_block_sizes[gid]
                if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                else attn_group.get_metadata_builder().kv_cache_spec.block_size
            )
        self._ensure_slot_mapping_buffers()

    def clear_draft_block_tables(self) -> None:
        self._draft_block_tables.clear()

    def set_draft_block_table(
        self,
        kv_cache_gid: int,
        block_table: torch.Tensor,
    ) -> None:
        if kv_cache_gid in self._draft_kv_cache_group_ids:
            self._draft_block_tables[kv_cache_gid] = block_table

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=True,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

    def _ensure_slot_mapping_buffers(self) -> None:
        gids = self._draft_kv_gids()

        first_gid = gids[0]
        for gid in gids:
            if gid in self._slot_mapping_buffers_by_gid:
                continue
            if gid == first_gid:
                self._slot_mapping_buffers_by_gid[gid] = (
                    self._context_slot_mapping_buffer,
                    self._slot_mapping_buffer,
                )
            else:
                self._slot_mapping_buffers_by_gid[gid] = (
                    torch.zeros(
                        self.max_num_tokens,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    torch.zeros(
                        self.max_query_tokens,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )

    def _draft_kv_gids(self) -> list[int]:
        return self._draft_kv_cache_group_ids or [
            self.kv_cache_gid if self.kv_cache_gid >= 0 else 0
        ]

    def _get_dflash_block_table(
        self,
        kv_cache_gid: int,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor:
        block_table = self._draft_block_tables.get(kv_cache_gid)
        if block_table is not None:
            return block_table
        if kv_cache_gid == self.kv_cache_gid or self.kv_cache_gid < 0:
            return cad.block_table_tensor
        raise RuntimeError(
            "Missing DFlash KV metadata for draft KV cache group "
            f"{kv_cache_gid}. This is required when DFlash draft layers span "
            "multiple KV cache groups."
        )

    def _get_dflash_context_slot_mapping(
        self,
        num_context: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not self._draft_layer_to_kv_cache_gid:
            return self._context_slot_mapping_buffer[:num_context]
        return {
            layer_name: self._slot_mapping_buffers_by_gid[
                self._draft_layer_to_kv_cache_gid[layer_name]
            ][0][:num_context]
            for layer_name in self._draft_attn_layer_names
        }

    @override
    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._ensure_slot_mapping_buffers()
        if self._draft_layer_to_kv_cache_gid:
            return {
                layer_name: self._slot_mapping_buffers_by_gid[
                    self._draft_layer_to_kv_cache_gid[layer_name]
                ][1][:num_tokens]
                for layer_name in self._draft_attn_layer_names
            }
        return super()._get_slot_mapping(num_tokens, slot_mapping)

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = self.num_speculative_tokens  # OMP: SpecForge DSpark block layout (anchor + k-1 masks)
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context

        # Context preprocessing does not run in a CUDA graph.
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Fill query inputs and per-KV-group slot mappings.
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        self._ensure_slot_mapping_buffers()
        draft_kv_group_ids = self._draft_kv_gids()
        for kv_cache_gid in draft_kv_group_ids:
            context_slot_mapping_buffer, query_slot_mapping_buffer = (
                self._slot_mapping_buffers_by_gid[kv_cache_gid]
            )
            block_table = self._get_dflash_block_table(kv_cache_gid, cad)
            copy_and_expand_dflash_inputs_kernel[grid](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=context_slot_mapping_buffer,
                out_query_slot_mapping_ptr=query_slot_mapping_buffer,
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=block_table,
                block_table_stride=block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                num_rejected_tokens_ptr=(
                    num_rejected_tokens_gpu if has_num_rejected else 0
                ),
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=self._draft_block_size_by_gid.get(
                    kv_cache_gid, self.block_size
                ),
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_NUM_REJECTED=has_num_rejected,
            )

        primary_kv_cache_gid = draft_kv_group_ids[0]
        query_slot_mapping = self._slot_mapping_buffers_by_gid[primary_kv_cache_gid][
            1
        ][:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=self._get_dflash_block_table(primary_kv_cache_gid, cad),
            slot_mapping=query_slot_mapping,
            causal=False,  # Non-causal attention is required for DFlash
        )

        if os.environ.get("DFLASH_DEBUG") == "1" and not getattr(
            self, "_dbg_attn", False
        ):
            self._dbg_attn = True
            with torch.no_grad():
                sl = new_cad.seq_lens.detach()
                ti = token_indices_to_sample.detach()
                bt = new_cad.block_table_tensor.detach()
                csm = self._slot_mapping_buffers_by_gid[primary_kv_cache_gid][0]
                qsm = self._slot_mapping_buffers_by_gid[primary_kv_cache_gid][1]
                logger.info(
                    "DFLASH_DBG attn_meta batch=%d n_query_total=%d "
                    "num_query_per_req=%d seq_lens[:2]=%s token_indices=%s | "
                    "block_table[0,:6]=%s | ctx_slot_map[:8]=%s | "
                    "qry_slot_map[:8]=%s",
                    batch_size,
                    num_query_total,
                    num_query_per_req,
                    sl[:2].tolist(),
                    ti.tolist(),
                    bt[0, :6].tolist() if bt.dim() == 2 else f"dim={bt.dim()}",
                    csm[:8].tolist(),
                    qsm[:8].tolist(),
                )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        context_states = torch.zeros(
            (num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written by the kernel.
        num_context = self._dflash_num_context

        try:
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
                self._context_positions_buffer[:num_context],
                self._get_dflash_context_slot_mapping(num_context),
            )
        finally:
            self._dflash_hidden_states = None
        if os.environ.get("DFLASH_DEBUG") == "1" and not getattr(
            self, "_dbg_inputs", False
        ):
            self._dbg_inputs = True
            with torch.no_grad():
                ids = self.input_ids[:num_input_tokens].detach()
                pos = self._get_positions(num_input_tokens).detach()
                n_mask = int((ids == self.parallel_drafting_token_id).sum().item())
                logger.info(
                    "DFLASH_DBG draft_inputs n_query=%d n_context=%d "
                    "mask_token_id=%s n_mask=%d ids[:16]=%s pos[:16]=%s "
                    "pos_min=%d pos_max=%d",
                    num_input_tokens,
                    num_context,
                    self.parallel_drafting_token_id,
                    n_mask,
                    ids[:16].tolist(),
                    pos[:16].tolist(),
                    int(pos.min().item()),
                    int(pos.max().item()),
                )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        self._ensure_slot_mapping_buffers()
        sliding_layer_names: set[str] = getattr(
            self.model, "sliding_attention_layer_names", set()
        )

        per_group: list[object] = []
        per_layer: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            kv_cache_gid = attn_group.kv_cache_group_id
            group_cad = cad.replace(
                block_table_tensor=self._get_dflash_block_table(kv_cache_gid, cad),
                slot_mapping=self._slot_mapping_buffers_by_gid[kv_cache_gid][1][
                    : cad.num_actual_tokens
                ],
                causal=False,
            )
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_cad,
                draft_index=draft_index,
            )
            per_group.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = attn_metadata

            # DFlash layers consume attention metadata through the per-layer
            # forward context. Keep the non-causal group metadata for
            # group-level spec decode checks, and specialize only the SWA
            # layers that need a causal sliding-window mask.
            causal_layers = (
                sliding_layer_names & set(attn_group.layer_names)
                if self.sliding_attention_causal
                else set()
            )
            if causal_layers:
                causal_attn_metadata = (
                    attn_group.get_metadata_builder().build_for_drafting(
                        common_attn_metadata=group_cad.replace(causal=True),
                        draft_index=draft_index,
                    )
                )
                for layer_name in causal_layers:
                    per_layer[layer_name] = causal_attn_metadata

        for layer_name, attn_metadata in per_layer.items():
            if layer_name in sliding_layer_names and self.sliding_attention_causal:
                assert getattr(attn_metadata, "causal", None) is True, (
                    f"Attention metadata for sliding layer {layer_name} does not have"
                    " causal support, which is required for DFlash SWA."
                )
                continue
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
