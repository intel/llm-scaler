"""HF transformers golden-reference 模板(实战验证:gemma-4-31B-it)。

用途:在 CPU(或跨卡分布式)用 HF transformers 加载与后端相同的权重,做一次 prefill,
在 embed / 选定层 / final_norm / lm_head 处以统一格式 dump hidden_states,
与 vLLM/XPU 后端的 [TRACE] 日志逐行 diff,定位首处数值发散的层/算子。

换模型时改:
  - MODEL_DIR / PROMPT / DUMP_FILE
  - Gemma4ForCausalLM            -> 目标模型的 XXXForCausalLM
  - no_split_module_classes      -> 目标模型的 DecoderLayer 类名
  - 权重前缀 "model.language_model."  -> 目标 ckpt 实际前缀(纯文本模型可能无需 rename)
  - layer_indices_to_trace       -> 想 trace 的层

默认 device_map=CPU(最稳,数值最接近理论值);HF_REF_DEVICE=xpu 切跨卡分布式。
只加载 model.language_model.* 权重,跳过 vision/audio tower,省显存。
"""
import os
import sys
import json
import glob

import torch
import torch.nn as nn

from safetensors.torch import safe_open
from transformers import AutoTokenizer, AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL_DIR = "/llm/models/weights/gemma-4-31B-it"
PROMPT = "AI是什么?"
DUMP_FILE = "/tmp/hf_31b_trace.log"


def stat(name: str, t: torch.Tensor) -> str:
    """格式与 vllm trace 一致。"""
    if t is None:
        return f"[TRACE-HF] {name}: None"
    tf = t.detach().float()
    return (
        f"[TRACE-HF] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={tf.min().item():.4g} max={tf.max().item():.4g} "
        f"mean={tf.mean().item():.4g} absmax={tf.abs().max().item():.4g} "
        f"nan={int(tf.isnan().any())} inf={int(tf.isinf().any())} "
        f"first4={tf.flatten()[:4].tolist()}"
    )


def main():
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    text_cfg = cfg.text_config
    text_cfg._attn_implementation = "eager"  # 不要 sdpa/flash,确保数值可复现
    text_cfg.torch_dtype = torch.float16

    print(f"[HF] num_hidden_layers={text_cfg.num_hidden_layers} "
          f"hidden_size={text_cfg.hidden_size} "
          f"vocab_size={text_cfg.vocab_size}", flush=True)

    # 1. 用 meta device 创建空壳,准备分卡放
    with init_empty_weights():
        model = Gemma4ForCausalLM(text_cfg)
    model.eval()

    # 2a. tied weights (lm_head.weight = model.embed_tokens.weight) 必须显式 tie,
    #    否则 accelerate 在 device_map 下加载会报 'on meta device'。
    model.tie_weights()

    # 2. checkpoint 里 key 前缀是 model.language_model.*,
    #    Gemma4ForCausalLM 期望 model.*,需要 rename。
    ckpt_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
    idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    have_idx = os.path.exists(idx_path)

    # 把 ckpt 重命名好后存成 临时 sharded ckpt(节省显存,不一次性加载到 host RAM)
    rename_dir = "/tmp/hf_31b_text_only_ckpt"
    os.makedirs(rename_dir, exist_ok=True)

    # 仅当目录为空时重新生成
    if not glob.glob(os.path.join(rename_dir, "*.safetensors")):
        print(f"[HF] 重命名 ckpt 到 {rename_dir} (仅 language_model.*)", flush=True)
        from safetensors.torch import save_file
        new_index = {"metadata": {"total_size": 0}, "weight_map": {}}
        per_file_tensors = {}  # fname -> {new_key: tensor}
        for src_file in ckpt_files:
            with safe_open(src_file, "pt") as h:
                for k in h.keys():
                    if not k.startswith("model.language_model."):
                        continue
                    new_k = "model." + k[len("model.language_model."):]
                    t = h.get_tensor(k)
                    out_fname = os.path.basename(src_file)
                    per_file_tensors.setdefault(out_fname, {})[new_k] = t
                    new_index["weight_map"][new_k] = out_fname
                    new_index["metadata"]["total_size"] += t.numel() * t.element_size()
        # tied: lm_head.weight = model.embed_tokens.weight (HF Gemma4ForCausalLM 内部已 tied)
        for fname, td in per_file_tensors.items():
            save_file(td, os.path.join(rename_dir, fname))
            print(f"  saved {fname}: {len(td)} tensors", flush=True)
        with open(os.path.join(rename_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f, indent=2)
        print(f"[HF] 重命名完成,共 {len(new_index['weight_map'])} tensors", flush=True)
    else:
        print(f"[HF] 复用已存在的 {rename_dir}", flush=True)

    # 3. CPU 路径(避免 XPU 加载偶发 device_lost):全模型放 CPU,fp16,
    #    单步 prefill 用 CPU 算,慢但稳。容器有 881GB RAM,fp16 31B(~62GB)够。
    use_xpu = os.environ.get("HF_REF_DEVICE", "cpu").lower() == "xpu"
    if use_xpu:
        n_xpu = torch.xpu.device_count()
        device_map = {
            "model.embed_tokens": "xpu:0",
            "lm_head": "xpu:0",
            "model.norm": f"xpu:{n_xpu - 1}",
        }
        n_layers = text_cfg.num_hidden_layers
        layers_per_device = (n_layers + n_xpu - 1) // n_xpu
        for i in range(n_layers):
            d = min(i // layers_per_device, n_xpu - 1)
            device_map[f"model.layers.{i}"] = f"xpu:{d}"
        print(f"[HF] device_map (XPU): {n_xpu} cards, {layers_per_device} layers/device", flush=True)
    else:
        device_map = {"": "cpu"}
        print(f"[HF] device_map=CPU (set HF_REF_DEVICE=xpu to use XPU)", flush=True)

    model = load_checkpoint_and_dispatch(
        model,
        rename_dir,
        device_map=device_map,
        dtype=torch.float16,
        no_split_module_classes=["Gemma4TextDecoderLayer"],
    )
    print("[HF] 模型加载完成", flush=True)

    # 4. tokenize
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # 同 vllm:用 chat template,与 LLM.generate 一致
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    # input 放到 model 第一层所在的 device(accelerate 会自动 dispatch 后续到正确设备)
    first_dev = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt").to(first_dev)
    input_ids = inputs.input_ids
    print(f"[HF] prompt='{PROMPT}', chat_template_len={input_ids.shape[1]}, "
          f"input_ids[:8]={input_ids[0][:8].tolist()}", flush=True)

    # 5. 注册 hook 抓中间结果
    dump = []  # (name, str)
    layer_outs = {}

    def lstat(name, t):
        s = stat(name, t)
        dump.append(s)
        print(s, flush=True)

    # 5a. embed
    embed_layer = model.model.embed_tokens
    def hook_embed(module, inp, out):
        lstat("embed_out", out)
    h_embed = embed_layer.register_forward_hook(hook_embed)

    # 5b. 每个 decoder layer 前/后,以及内部各子步
    layer_indices_to_trace = sorted(set([0, 1, 2, 5, 11, 17, 23, 29, 35, 41, 47]))
    layer_handles = []

    def make_layer_hook(li):
        def hook(module, inp, out):
            # out 是 hidden_states (Tensor) 或 (hidden_states, ...) tuple
            hs = out[0] if isinstance(out, tuple) else out
            lstat(f"layer{li}_out", hs)
        return hook

    for li in layer_indices_to_trace:
        h = model.model.layers[li].register_forward_hook(make_layer_hook(li))
        layer_handles.append(h)

    # 5c. final norm + lm_head
    h_norm = model.model.norm.register_forward_hook(
        lambda m, i, o: lstat("post_final_norm", o)
    )
    pre_norm_holder = {}
    def pre_norm_hook(m, inp):
        # inp 是 tuple,第一个是 pre_norm 输入
        x = inp[0] if isinstance(inp, tuple) else inp
        lstat("pre_final_norm", x)
    h_pre = model.model.norm.register_forward_pre_hook(pre_norm_hook)

    def lm_head_hook(m, i, o):
        x = i[0] if isinstance(i, tuple) else i
        lstat("compute_logits.input", x)
        lstat("compute_logits.out", o)
        # 不返回 → 不替换输出
    h_lm = model.lm_head.register_forward_hook(lm_head_hook)

    # 6. 单步 forward
    print("[HF] 开始 prefill ...", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    # 7. logits top
    logits = out.logits  # [1, T, V]
    last = logits[0, -1].float()
    top5 = last.topk(5)
    print(f"[TRACE-HF] last_token_logits.absmax={last.abs().max().item():.4g} "
          f"top5_vals={top5.values.tolist()} top5_ids={top5.indices.tolist()}", flush=True)
    dump.append(f"[TRACE-HF] last_token_logits top5_ids={top5.indices.tolist()}")

    # 8. 写日志
    with open(DUMP_FILE, "w") as f:
        f.write("\n".join(dump))
        f.write("\n")
    print(f"[HF] 全部写入 {DUMP_FILE}", flush=True)

    # 9. 试着多生成几步看输出
    print("[HF] 用 generate 跑 16 个 token ...", flush=True)
    gen = model.generate(input_ids, max_new_tokens=16, do_sample=False)
    txt = tok.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[HF] Output: {repr(txt)}", flush=True)


if __name__ == "__main__":
    main()
