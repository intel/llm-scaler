# Draft comment for vllm-project/vllm PR #32899

Target: https://github.com/vllm-project/vllm/pull/32899
"[WIP][CT][XPU] Add W4A16 (INT4+MXFP4) MoE Support" — Intel Zhenzhong1

Not posted yet — post from laptop after final review.

---

Some Lunar Lake (Xe2-LPG, Arc 140V iGPU, `intel_gpu_lnl_m`) validation data for this approach, in case it's useful for the low-end XPU angle.

Backported the dispatch logic in this PR to vLLM 0.19.0 (monkey-patch onto `CompressedTensorsWNA16MarlinMoEMethod`) and ran on a single Arc 140V / 30.9 GiB LPDDR5x:

| Model | group_size | Output tok/s | TTFT med (ms) | TPOT med (ms) |
|---|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit (compressed-tensors, pack-quantized, sym) | 32 | 18.22 | 1354 | 58 |
| Qwen3-30B-A3B GPTQ-int4 | 128 | 9.97 | 2085 | 100 |
| Qwen3-VL-30B-A3B-int4-autoround | 128 | 9.68 | 2132 | 105 |

Recipe: `vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 3 --max-concurrency 1`. All three are the same Qwen3-30B-A3B architecture (hidden=2048, moe_inter=768, 128 experts, top-k=8, 48 layers), same kernel path (`xpu_fused_moe(is_int4=True)`), back-to-back on the same HW.

Two observations:

1. **Group_size=32 path is ~1.83× faster than group_size=128** on the same architecture. Same `xpu_fused_moe` call, same model size, just different scale tensor shape. Might be worth a look if this gap is expected or if there's headroom on the gs=128 code path.

2. **Lunar Lake gotchas** for anyone retesting on LNL:
   - `vllm-xpu-kernels v0.1.4` (default in v0.19 `requirements/xpu.txt`) rejects `intel_gpu_lnl_m` in `is_xe2_arch()`. v0.1.5 wheel fixes this.
   - `--kv-cache-memory-bytes N` works around the v0.19 memory-profile budget being too pessimistic on 30.9 GiB unified-memory iGPUs.

Happy to retest against the PR's final layout once it's ready. AI assistance used (per AGENTS.md policy).
