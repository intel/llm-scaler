# Draft comment for vllm-project/vllm PR #40128

Target: https://github.com/vllm-project/vllm/pull/40128
"fix: handle non-divisible page sizes in hybrid model KV cache unification" — Sandermage

Not posted yet — post from laptop after final review.

---

Backporting this fix to vLLM 0.19.0 on Intel Arc 140V / Lunar Lake (intel_gpu_lnl_m, 30.9 GiB LPDDR5x unified memory) unblocks Qwen3.5-35B-A3B-AutoRound, which is a different hybrid architecture from the Qwen3.6-35B-A3B-FP8 this PR targets but trips the exact same crash path.

Without the patch:
```
NotImplementedError: The page size of the layer is not divisible by
the maximum page size. Cannot unify by adjusting block_size.
```

With this patch cherry-picked onto `vllm/v1/core/kv_cache_utils.py` in a v0.19.0 venv, Qwen3.5-35B-A3B-AutoRound (21 GiB on disk, 19.26 GiB loaded) runs end-to-end:

- `vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 3 --max-concurrency 1` → **8.94 tok/s** output, TPOT ~92 ms, TTFT median ~1.7 s.

The `page_size_padded` field used by the fix is already present on layer specs in v0.19.0 (`vllm/v1/kv_cache_interface.py:69`), so the PR drops in cleanly on that release branch too — one-file change, no dependency on other main-only refactors as far as I could see.

Useful data point for the hybrid-model matrix: this PR's impact extends beyond Qwen3.6-35B-FP8 to any hybrid attention + Mamba/DeltaNet family, including Qwen3.5-A3B. Would suggest mentioning "hybrid AutoRound INT4 variants" in the PR summary alongside the FP8 TurboQuant case.

Happy to retest against the final merged form when it lands. AI assistance used (per AGENTS.md policy).
