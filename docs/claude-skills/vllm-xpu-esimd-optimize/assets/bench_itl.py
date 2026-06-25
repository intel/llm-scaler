"""ITL bench — Phase 3 performance baseline.

Single 16-token prompt, max_tokens=128, 5 runs after a warmup. Always quote
median (sorted()[N//2]); never best, never mean. Watch the run-to-run spread —
if it exceeds ~0.3 ms the test bed is noisy and you should rerun.

The most stable comparisons are env-gate flips on the same binary:
    MODE=fused  python bench_itl.py
    MODE=base   DISABLE_<NAME>=1 python bench_itl.py
back-to-back, on the same EUs (ZE_AFFINITY_MASK), in the same minute.

Knobs (edit before running):
  MODEL_PATH    — full weights path
  TP            — tensor_parallel_size
  PROMPT_IDS    — your model's chat-template-tokenised 16-token prompt
"""
import os
import time
import sys


MODEL_PATH = "/llm/models/weights/gemma-4-26B-A4B-it"   # <-- edit
TP = 2                                                   # <-- edit
# 16-token chat-template prompt for gemma-4-26B. For other models, generate
# once via apply_chat_template on a short user message + paste the resulting
# token id list here. Do NOT change between runs — this is the perf fingerprint.
PROMPT_IDS = [2, 105, 2364, 107, 12553, 54847, 236881, 106, 107, 105, 4368,
              107, 100, 45518, 107, 101]


def main():
    mode = os.environ.get("MODE", "esimd")
    from vllm import LLM, SamplingParams
    print(f"[mode={mode}]", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP,
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype="float16",
        quantization="fp8",
        kv_cache_dtype="fp8",
        trust_remote_code=True,
        max_num_seqs=8,
    )

    sp = SamplingParams(max_tokens=128, temperature=0)

    # warmup (compiles JIT kernels, populates allocator caches)
    llm.generate(prompts=[{"prompt_token_ids": PROMPT_IDS}], sampling_params=sp)

    n_run = 5
    tpots = []
    for i in range(n_run):
        t0 = time.perf_counter()
        out = llm.generate(prompts=[{"prompt_token_ids": PROMPT_IDS}],
                           sampling_params=sp)
        dt = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        itl = dt * 1000 / n
        print(f"  run {i+1}: {n} tokens in {dt*1000:.1f} ms  → "
              f"{n/dt:.2f} tok/s  avg_itl={itl:.2f} ms", flush=True)
        tpots.append(itl)

    median = sorted(tpots)[len(tpots) // 2]
    print(f"[mode={mode}] median ITL = {median:.2f} ms "
          f"({1000/median:.2f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
