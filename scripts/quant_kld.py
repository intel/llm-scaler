#!/usr/bin/env python3
"""Compare a quantized model against a reference model served by vLLM.

Two metrics, both computed through the OpenAI-compatible API, so the two models
can live on different machines (e.g. BF16 reference on a DGX Spark, MXFP4 or
INT4 candidate on the MSI Claw) or on the same machine at different times:

  1. KL divergence KL(ref || cand) of the next-token distribution, teacher-forced
     on the reference model's own greedy continuation (vLLM `prompt_logprobs`).
     Reported per position; summary = mean, p50, p99, max, and top-1 agreement.
  2. Greedy parity: both models generate at temperature 0 from the same prompt;
     report exact-match rate and the first divergence index.

Workflow (two phases, so only one model needs to be up at a time):

  # phase 1 - reference (any host)
  python quant_kld.py capture --url http://spark:8000/v1 --model <served-name> \
      --out ref.json [--prompts prompts.txt] [--max-tokens 256] [--top-k 20]

  # phase 2 - candidate (e.g. the Claw)
  python quant_kld.py compare --url http://localhost:8080/v1 --model <served-name> \
      --ref ref.json --out cand.json

  # optional: compare a second candidate (e.g. INT4 vs MXFP4) against the same ref
  python quant_kld.py compare --url ... --ref ref.json --out cand2.json
  python quant_kld.py report cand.json cand2.json

Requirements: both models must share the tokenizer (quants of the same base
model). vLLM >= 0.19 for `prompt_logprobs` on /v1/completions. Only `requests`
is needed (pip install requests). Thinking/reasoning should be disabled on both
servers (or enabled on both) so the continuations are comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

# --------------------------------------------------------------------------
# Built-in prompt set: English / Chinese / code / reasoning / tool-ish.
# Replace with --prompts FILE (one prompt per line, blank lines ignored).
# --------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Explain the difference between a process and a thread in one paragraph.",
    "Write a Python function that parses ISO-8601 timestamps and returns a timezone-aware datetime. Handle 'Z' and offsets.",
    "What is the time complexity of binary search and why?",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "Give three reasons why unified memory on an integrated GPU changes how an inference server should budget memory.",
    "Translate to English: 这台掌机的内存是 32GB，GPU 和 CPU 共享同一块内存。",
    "用中文解释什么是量化感知训练（QAT），以及它和训练后量化的区别。",
    "写一个 Bash 脚本，统计当前目录下每种文件扩展名的文件数量，并按数量降序输出。",
    "请用三句话介绍一下量子纠缠。",
    "Implement an LRU cache in Python with O(1) get and put. Include a short docstring.",
    "A train leaves at 14:35 and arrives at 17:10. How long is the journey? Show the calculation.",
    "List the steps to safely rotate an SSH key on a Linux server without locking yourself out.",
    "Write a SQL query that returns the top 3 customers by total order value in 2025, with ties included.",
    "What does the `--gpu-memory-utilization` flag in vLLM control, and what happens if it is set too high?",
    "Fix the bug: `def avg(xs): return sum(xs) / len(xs)` crashes on an empty list. Return None in that case.",
    "In JSON, return an object with keys name, version, and dependencies (array of strings) describing a fictional CLI tool.",
    "Explain why floating point 0.1 + 0.2 != 0.3 in most languages.",
    "Write a regular expression that matches IPv4 addresses and explain each part.",
    "给出一个 Python 例子，展示如何用 asyncio 并发请求三个 URL 并收集结果。",
    "Compare TCP and UDP for a real-time voice chat application. Which would you pick and why?",
    "Write a haiku about memory bandwidth.",
    "A user reports 'No route to host' when SSHing to a laptop on the same Wi-Fi. List likely causes in order of probability.",
    "Convert this to a Python dataclass: a record with id (int), tags (list of str), created_at (datetime), optional note (str).",
    "Explain Mixture-of-Experts routing to a software engineer who knows transformers but not MoE.",
    "What is the derivative of x^3 * ln(x)? Show steps.",
    "用中文写一段 100 字左右的产品说明，介绍一款支持本地语音转写的掌上电脑。",
    "Write a git command sequence that creates a branch from main, cherry-picks commit abc123, and pushes it.",
    "Why might an INT4 quantized model be slower than FP16 on some hardware? Give two reasons.",
    "Design a REST endpoint for uploading an audio file for transcription. Specify method, path, request, and response.",
    "Explain the halting problem in simple terms.",
    "Refactor for readability: `x=[i for i in range(100) if i%2==0 and i%3==0 and i>10]`.",
    "What are the tradeoffs of sliding-window attention versus full attention for long documents?",
    "Write a Dockerfile for a small FastAPI app with a non-root user.",
    "一个列表里有重复元素，怎样在保持原顺序的前提下去重？给出 Python 代码。",
    "Explain what a KV cache is and why it grows with context length.",
    "Given the sequence 2, 6, 12, 20, 30, what is the next term and the general formula?",
    "Write a short systemd unit file that starts a Python server on boot and restarts on failure.",
    "What is the difference between symmetric and asymmetric quantization?",
    "Describe a safe procedure to test a new GPU driver on a laptop that is also your only development machine.",
    "Return only valid JSON: a list of three objects with fields city, country, population (integer).",
]

TIMEOUT = 600


def _post(url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(f"{url.rstrip('/')}/{path}", json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def greedy_generate(url: str, model: str, prompt: str, max_tokens: int) -> str:
    out = _post(
        url,
        "chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 0,
        },
    )
    return out["choices"][0]["message"]["content"] or ""


def render_chat_prompt(url: str, model: str, prompt: str) -> str | None:
    """Ask the server to render the chat template so we can teacher-force the
    exact same token stream through /v1/completions. Uses vLLM's tokenize
    endpoint; returns None if unavailable (then we fall back to raw text)."""
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        r = requests.post(
            f"{base}/tokenize",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "add_generation_prompt": True, "return_token_strs": False},
            timeout=TIMEOUT,
        )
        if r.status_code == 200 and "tokens" in r.json():
            # Detokenize to text so both servers tokenize identically from text.
            d = requests.post(
                f"{base}/detokenize",
                json={"model": model, "tokens": r.json()["tokens"]},
                timeout=TIMEOUT,
            )
            if d.status_code == 200:
                return d.json().get("prompt")
    except Exception:
        pass
    return None


def prompt_logprobs(url: str, model: str, text: str, top_k: int) -> list[dict[str, float] | None]:
    """Teacher-forced top-k next-token logprobs for every position of `text`."""
    out = _post(
        url,
        "completions",
        {
            "model": model,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": top_k,
            "echo": False,
        },
    )
    plp = out["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("server did not return prompt_logprobs (need vLLM >= 0.19)")
    dists: list[dict[str, float] | None] = []
    for pos in plp:
        if pos is None:
            dists.append(None)
            continue
        dists.append({str(tid): v["logprob"] for tid, v in pos.items()})
    return dists


def kl_topk(ref: dict[str, float], cand: dict[str, float]) -> tuple[float, bool]:
    """KL(ref||cand) over the union of both top-k supports, each renormalized.
    Tokens missing from one side get a floor one nat below that side's minimum.
    Returns (kl_nats, top1_agree)."""
    union = set(ref) | set(cand)
    rfloor = min(ref.values()) - 1.0
    cfloor = min(cand.values()) - 1.0
    r = {t: ref.get(t, rfloor) for t in union}
    c = {t: cand.get(t, cfloor) for t in union}
    rz = math.log(sum(math.exp(v) for v in r.values()))
    cz = math.log(sum(math.exp(v) for v in c.values()))
    kl = 0.0
    for t in union:
        p = math.exp(r[t] - rz)
        kl += p * ((r[t] - rz) - (c[t] - cz))
    top1 = max(ref, key=ref.get) == max(cand, key=cand.get)
    return max(kl, 0.0), top1


def cmd_capture(a: argparse.Namespace) -> None:
    prompts = load_prompts(a.prompts)
    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        cont = greedy_generate(a.url, a.model, p, a.max_tokens)
        rendered = render_chat_prompt(a.url, a.model, p) or p
        full = rendered + cont
        lp = prompt_logprobs(a.url, a.model, full, a.top_k)
        # positions that belong to the continuation = last len(lp)-n_prompt_tokens;
        # we don't know the split exactly from text, so record prompt token count.
        n_prompt = len(prompt_logprobs(a.url, a.model, rendered, 1))
        rows.append({"prompt": p, "rendered": rendered, "continuation": cont,
                     "n_prompt_tokens": n_prompt, "logprobs": lp})
        print(f"[capture] {i + 1}/{len(prompts)}  {len(lp)} tok  {time.time() - t0:.0f}s", file=sys.stderr)
    json.dump({"model": a.model, "url": a.url, "top_k": a.top_k, "max_tokens": a.max_tokens,
               "rows": rows}, open(a.out, "w"), ensure_ascii=False)
    print(f"wrote {a.out}: {len(rows)} prompts")


def cmd_compare(a: argparse.Namespace) -> None:
    ref = json.load(open(a.ref))
    top_k = ref["top_k"]
    kls: list[float] = []
    top1 = 0
    npos = 0
    parity = []
    t0 = time.time()
    for i, row in enumerate(ref["rows"]):
        full = row["rendered"] + row["continuation"]
        cand_lp = prompt_logprobs(a.url, a.model, full, top_k)
        ref_lp = row["logprobs"]
        n = min(len(cand_lp), len(ref_lp))
        if len(cand_lp) != len(ref_lp):
            print(f"  warn: token count differs ({len(ref_lp)} vs {len(cand_lp)}) on prompt {i}; "
                  f"tokenizers may differ", file=sys.stderr)
        for pos in range(row["n_prompt_tokens"], n):
            r, c = ref_lp[pos], cand_lp[pos]
            if r is None or c is None:
                continue
            kl, agree = kl_topk(r, c)
            kls.append(kl)
            top1 += agree
            npos += 1
        # greedy parity
        cand_cont = greedy_generate(a.url, a.model, row["prompt"], ref["max_tokens"])
        rc, cc = row["continuation"], cand_cont
        div = next((k for k in range(min(len(rc), len(cc))) if rc[k] != cc[k]),
                   None if len(rc) == len(cc) else min(len(rc), len(cc)))
        parity.append({"prompt": row["prompt"], "identical": div is None,
                       "first_divergence_char": div, "ref_len": len(rc), "cand_len": len(cc),
                       "cand_continuation": cc})
        print(f"[compare] {i + 1}/{len(ref['rows'])}  kl_mean_so_far={statistics.fmean(kls):.4f}  "
              f"{time.time() - t0:.0f}s", file=sys.stderr)
    res = {
        "ref_model": ref["model"], "cand_model": a.model, "cand_url": a.url,
        "positions": npos, "kl_mean": statistics.fmean(kls), "kl_p50": statistics.median(kls),
        "kl_p99": sorted(kls)[int(0.99 * (len(kls) - 1))], "kl_max": max(kls),
        "top1_agreement": top1 / npos if npos else float("nan"),
        "greedy_identical_rate": sum(p["identical"] for p in parity) / len(parity),
        "greedy_median_first_divergence_char": statistics.median(
            [p["first_divergence_char"] for p in parity if p["first_divergence_char"] is not None] or [-1]),
        "parity": parity,
    }
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    _print_summary([res])


def _print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'candidate':<40} {'KL mean':>8} {'p50':>8} {'p99':>8} {'max':>8} {'top1':>6} {'greedy=':>8} {'div@':>6}")
    for r in results:
        div = r["greedy_median_first_divergence_char"]
        div_s = "none" if div < 0 else f"{div:.0f}"
        print(f"{r['cand_model'][:40]:<40} {r['kl_mean']:8.4f} {r['kl_p50']:8.4f} {r['kl_p99']:8.4f} "
              f"{r['kl_max']:8.3f} {r['top1_agreement']:6.1%} {r['greedy_identical_rate']:8.0%} {div_s:>6}")
    print("\nReading: KL mean (nats) vs BF16 — ~0.01 excellent (Q8-class), 0.02-0.05 good INT4, "
          ">0.1 lossy; top1 = next-token agreement; greedy= = fraction of prompts with byte-identical "
          "greedy output; div@ = median character index of first divergence.")


def cmd_report(a: argparse.Namespace) -> None:
    _print_summary([json.load(open(p)) for p in a.files])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="record reference continuations + logprobs")
    c.add_argument("--url", required=True, help="e.g. http://host:8000/v1")
    c.add_argument("--model", required=True, help="served model name")
    c.add_argument("--out", required=True)
    c.add_argument("--prompts", help="file with one prompt per line (default: built-in set)")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--top-k", type=int, default=20)
    c.set_defaults(fn=cmd_capture)
    d = sub.add_parser("compare", help="score a candidate against a capture file")
    d.add_argument("--url", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--ref", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_compare)
    r = sub.add_parser("report", help="print a table for one or more compare outputs")
    r.add_argument("files", nargs="+")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
