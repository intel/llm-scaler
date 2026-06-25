"""Offline chat requests replay — Phase 2 accuracy gate.

Reads /tmp/chat_reqs.json (5 GSM8K few-shot chat prompts), applies the model
chat template locally, hands token-id lists to llm.generate, and prints per-Q
accuracy + token-level outputs.

A 5/5 = 1.000 result is the bar for any kernel-change commit. Less than that
means a different shape (long prompts, prefill chunked into M>1 chunks) is
broken — common failure mode for kernels that pass single-prompt unit tests.

Knobs (edit before running):
  MODEL_PATH       — full path on the container, e.g. /llm/models/weights/<name>
  TP               — tensor_parallel_size
"""
import json
import sys
import regex as re
import ast
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INVALID = -9999999
def get_ans(s: str) -> int:
    s = s.replace(",", "")
    nums = re.findall(r"\d+", s)
    if not nums:
        return INVALID
    try:
        return ast.literal_eval(nums[-1])
    except SyntaxError:
        return INVALID


MODEL_PATH = "/llm/models/weights/gemma-4-26B-A4B-it"   # <-- edit
TP = 2                                                   # <-- edit


def main():
    reqs = json.load(open("/tmp/chat_reqs.json"))
    print(f"Loaded {len(reqs)} requests", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    prompts = []
    labels = []
    for r in reqs:
        # Two-step: tokenize=False returns a string, then tokenize without
        # adding BOS again. apply_chat_template(tokenize=True) returns a
        # BatchEncoding that's awkward to extract list[int] from.
        prompt_str = tok.apply_chat_template(
            r["messages"], add_generation_prompt=True, tokenize=False)
        ids = tok(prompt_str, add_special_tokens=False).input_ids
        prompts.append({"prompt_token_ids": list(ids)})
        labels.append(r["_label"])

    print(f"prompt token lens: {[len(p['prompt_token_ids']) for p in prompts]}", flush=True)
    print(f"first 30 ids of req 0: {prompts[0]['prompt_token_ids'][:30]}", flush=True)

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP,
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype="float16",
        quantization="fp8",
        trust_remote_code=True,
        max_num_seqs=8,
    )

    sp = SamplingParams(temperature=0, max_tokens=512, seed=42)
    outs = llm.generate(prompts, sampling_params=sp)

    print("\n=== SUMMARY ===", flush=True)
    correct = 0
    for i, out in enumerate(outs):
        text = out.outputs[0].text
        pred = get_ans(text)
        label = labels[i]
        try:
            ok = (str(pred) == label.replace(",", ""))
        except Exception:
            ok = False
        if ok:
            correct += 1
        marker = "OK" if ok else "FAIL"
        print(f"Q#{i} {marker} label={label} pred={pred} "
              f"toks={len(out.outputs[0].token_ids)} "
              f"finish={out.outputs[0].finish_reason} chars={len(text)}",
              flush=True)
    print(f"\nAccuracy: {correct}/{len(outs)} = {correct/len(outs):.3f}", flush=True)

    print("\n=== FULL OUTPUTS ===", flush=True)
    for i, out in enumerate(outs):
        print(f"\n--- Q#{i} (label={labels[i]}) ---", flush=True)
        print(out.outputs[0].text, flush=True)

    print("\n=== FIRST 30 TOKEN IDS PER QUESTION ===", flush=True)
    for i, out in enumerate(outs):
        ids = list(out.outputs[0].token_ids)[:30]
        print(f"Q#{i}: {ids}", flush=True)


if __name__ == "__main__":
    main()
