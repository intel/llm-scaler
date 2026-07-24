#!/usr/bin/env python3
"""
独立 GSM8K 准确率评测 (Qwen3.6-35B-A3B, BMG)，不依赖 sglang.test.run_eval。
直接打 OpenAI 兼容端点，采样参数(含 repetition_penalty)全部自己掌控。

复现口径与 sglang 的 gsm8k eval 一致:
  - 数据集: openai grade-school-math test.jsonl (缓存到 /tmp/test.jsonl)
  - few-shot: 文件前 num_shots 条做 few-shot, 评测从第 num_shots 条开始 (防泄漏)
  - 抽取: content 里最后一个数字, 与标准答案比对 (同 simple_eval_gsm8k)

两种接口:
  --api chat        -> /v1/chat/completions (套 chat 模板, 可开 --thinking)
  --api completion  -> /v1/completions      (裸 few-shot 文本, 带 stop tokens)

用法示例:
  # chat + thinking (默认开), 20 题, 加 repetition_penalty 压死循环
  python3 run_gsm8k.py --num-examples 20 --api chat \
      --repetition-penalty 1.05 --num-threads 1

  # 真正关闭 thinking (显式下发 enable_thinking=false), 隔离内核计算基线
  python3 run_gsm8k.py --num-examples 20 --api chat --no-thinking --num-threads 8

  # 非 thinking completion 基线 (裸 few-shot, 输出最短最稳)
  python3 run_gsm8k.py --num-examples 20 --api completion --num-threads 8

  # Qwen 官方推荐采样口径
  python3 run_gsm8k.py --num-examples 20 --api chat \
      --temperature 0.6 --top-p 0.95 --top-k 20
"""
import argparse
import ast
import json
import os
import re
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
INVALID = -9999999
COMPLETION_STOP = ["Question", "Assistant:", "<|separator|>"]
# chat 模式的 stop: 阻止模型答完后继续凭空续写 "Question: ..." 假题,
# 否则抽取器取"最后一个数字"会抓到续写题的答案 -> 误判。
CHAT_STOP = ["\nQuestion:", "Question:"]


# ──────────── 数据 / prompt ────────────
def get_one_example(line, include_answer):
    ret = f"Question: {line['question']}\nAnswer:"
    if include_answer:
        ret += f" {line['answer']}"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def load_lines(data_path):
    if data_path and os.path.exists(data_path):
        path = data_path
    else:
        path = "/tmp/test.jsonl"
        if not os.path.exists(path):
            print(f"[info] downloading dataset -> {path}")
            urllib.request.urlretrieve(GSM8K_URL, path)
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# ──────────── 请求 ────────────
def build_sampling(args):
    """把采样参数收集成 dict, 直接放进请求体顶层 (sglang 兼容端点均可读)。"""
    p = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.top_k is not None:
        p["top_k"] = args.top_k
    if args.min_p is not None:
        p["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        p["repetition_penalty"] = args.repetition_penalty
    if args.frequency_penalty is not None:
        p["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        p["presence_penalty"] = args.presence_penalty
    return p


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def call_chat(base_url, model, prompt, sampling, thinking, timeout, stop=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **sampling,
    }
    # 始终显式下发 enable_thinking, 否则模板默认 True, "关思考" 不会生效。
    payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if stop:
        payload["stop"] = stop
    body = post(f"{base_url}/v1/chat/completions", payload, timeout)
    ch = body["choices"][0]
    msg = ch.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = body.get("usage", {}) or {}
    return content, len(reasoning), ch.get("finish_reason"), usage.get("completion_tokens")


def call_completion(base_url, model, prompt, sampling, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "stop": COMPLETION_STOP,
        **sampling,
    }
    body = post(f"{base_url}/v1/completions", payload, timeout)
    ch = body["choices"][0]
    content = ch.get("text") or ""
    usage = body.get("usage", {}) or {}
    return content, 0, ch.get("finish_reason"), usage.get("completion_tokens")


def resolve_model(base_url, model_arg):
    if model_arg:
        return model_arg
    try:
        body = post(f"{base_url}/v1/models", {}, 10) if False else None
    except Exception:
        body = None
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as r:
            return json.loads(r.read().decode())["data"][0]["id"]
    except Exception:
        return "default"


# ──────────── 主流程 ────────────
def classify(ok, content_len, finish):
    if ok:
        return "correct"
    if content_len == 0:
        return "empty_output"
    if finish == "length":
        return "runaway_len"
    return "wrong_answer"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="30000")
    ap.add_argument("--model", default="")
    ap.add_argument("--num-examples", type=int, default=None, help="空=全量(~1319)")
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=8, help="并发请求数")
    ap.add_argument("--api", choices=["chat", "completion"], default="chat")
    ap.add_argument("--thinking", dest="thinking", action="store_true",
                    help="chat 模式开 thinking (默认开)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="chat 模式关 thinking (显式下发 enable_thinking=false)")
    ap.set_defaults(thinking=True)
    ap.add_argument("--chat-stop", dest="chat_stop", action="store_true",
                    help="chat 模式加 stop=['Question:'] 阻止续写假题 (默认开)")
    ap.add_argument("--no-chat-stop", dest="chat_stop", action="store_false",
                    help="chat 模式不加 stop")
    ap.set_defaults(chat_stop=True)
    # 采样
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--frequency-penalty", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    # 其它
    ap.add_argument("--data-path", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    # thinking 默认放大 max_tokens (与 run_gsm8k.sh 行为一致), 但用户显式传了就不覆盖
    if args.thinking and "--max-tokens" not in sys.argv:
        args.max_tokens = 16384

    base_url = f"http://{args.host}:{args.port}"
    model = resolve_model(base_url, args.model)
    sampling = build_sampling(args)

    lines = load_lines(args.data_path)
    few_shot = "".join(
        get_one_example(lines[i], True) + "\n\n" for i in range(args.num_shots)
    )
    eval_lines = lines[args.num_shots:]
    if args.num_examples is not None:
        eval_lines = eval_lines[: args.num_examples]
    n = len(eval_lines)

    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"gsm8k_{ts}"
    examples_path = f"{prefix}_examples.jsonl"
    summary_path = f"{prefix}_summary.txt"

    print("=" * 72)
    print(f"server={base_url}  model={model}")
    print(f"api={args.api}  thinking={args.thinking}  chat_stop={args.chat_stop and args.api=='chat'}  examples={n}  threads={args.num_threads}")
    print(f"sampling={sampling}")
    print("=" * 72)

    results = [None] * n
    done = {"c": 0}
    lock = threading.Lock()
    t_start = time.perf_counter()

    def work(idx):
        line = eval_lines[idx]
        prompt = few_shot + get_one_example(line, include_answer=False)
        gold = get_answer_value(line["answer"])
        try:
            if args.api == "chat":
                content, rlen, finish, ctok = call_chat(
                    base_url, model, prompt, sampling, args.thinking, args.timeout,
                    stop=CHAT_STOP if args.chat_stop else None,
                )
            else:
                content, rlen, finish, ctok = call_completion(
                    base_url, model, prompt, sampling, args.timeout
                )
            err = None
        except Exception as e:
            content, rlen, finish, ctok, err = "", 0, None, None, str(e)
        extracted = get_answer_value(content)
        ok = extracted == gold
        kind = "error" if err else classify(ok, len(content), finish)
        rec = {
            "index": idx, "gold": gold, "extracted": extracted,
            "score": float(ok), "kind": kind,
            "content_len": len(content), "reasoning_len": rlen,
            "finish_reason": finish, "completion_tokens": ctok,
            "error": err, "response": content,
        }
        with lock:
            done["c"] += 1
            print(f"[{done['c']}/{n}] idx={idx} {'PASS' if ok else 'FAIL'} "
                  f"gold={gold} extracted={extracted} kind={kind} "
                  f"clen={len(content)} finish={finish}", flush=True)
        return idx, rec

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = [ex.submit(work, i) for i in range(n)]
        for f in as_completed(futs):
            idx, rec = f.result()
            results[idx] = rec

    elapsed = time.perf_counter() - t_start

    with open(examples_path, "w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    correct = sum(r["score"] for r in results)
    acc = correct / n if n else 0.0
    from collections import Counter
    kinds = Counter(r["kind"] for r in results)

    lines_out = [
        "GSM8K Eval Summary (standalone, no run_eval)",
        "=" * 44,
        f"Timestamp   : {ts}",
        f"Server      : {base_url}",
        f"Model       : {model}",
        f"API         : {args.api}   thinking={args.thinking}   chat_stop={args.chat_stop and args.api=='chat'}",
        f"Examples    : {n}",
        f"Sampling    : {sampling}",
        f"Elapsed     : {elapsed:.1f}s",
        f"Correct     : {int(correct)}/{n}",
        f"Accuracy    : {acc:.4f}",
        f"Kinds       : {dict(kinds)}",
    ]
    summary = "\n".join(lines_out)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    print("\n" + summary)
    print(f"\nPer-example : {examples_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
