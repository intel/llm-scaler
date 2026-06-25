---
name: hf-golden-layerwise-diff
description: 当一个模型在 vLLM/XPU 等自定义后端上"输出不对/精度发散"，用 HuggingFace transformers 在 CPU(或跨卡分布式)加载同一份权重作为 golden reference，逐层 dump hidden_states 并与后端 trace 逐行 diff，定位首处数值发散的层/算子。适用于：用户说"<某模型> 在 vllm/xpu 上输出乱码/不对/和 HF 对不上""帮我对比 HF 和 vllm 的逐层输出""定位是哪一层算错"。前提：拥有 HF 权重路径、一个能跑 transformers 的环境(CPU 足够，无需 GPU)、以及能 patch 后端 forward 打 trace 的能力。
---

# 用 HF transformers 做 golden reference 逐层对比定位精度 bug

这是一套**排查"模型在自定义后端(vLLM/XPU/自写 kernel)上输出不对"的标准流程**。
核心是：HF transformers 的 eager 前向是数值 ground truth，把它和后端的同输入逐层对比，
**找首处发散的层**，把"输出乱码"这种黑盒问题收敛成"第 N 层的某个算子算错了"。

源自 gemma-4-31B / 26B-A4B 在 vLLM-XPU 上输出不对的实战排查。

## 何时用 / 不用

- **用**：后端输出乱码、重复、与参考不符，且怀疑是某个 kernel/算子/权重加载的数值问题。
- **不用**：纯性能优化(用 `vllm-xpu-esimd-optimize`)；或问题明显是采样/模板/tokenize 层面(先排除这些再来)。

## 第 0 步：先做对照实验缩小范围(最省时间，别跳)

下任何结论前，先用"多模型对照"把 bug 关进最小的笼子。这是本方法最关键的一步：

- **同家族不同结构都错 → bug 在共用路径**。例：31B(稠密 MLP)和 26B(128-expert MoE)**都**输出错
  ⇒ 问题必在两者共用代码(Attention QK/V-Norm、RoPE、夹心残差 layernorm、embedding 归一化 `sqrt(hidden)`、`load_weights` 权重名映射/stacked 合并)，**不可能是 MoE 专属**。
- **别家族模型能跑对**(如 Qwen 同后端正常) ⇒ 只能证明"后端基础设施(attention 框架、fp16、KV、采样)是好的"，
  **不能**证明目标模型的共用路径对(不同模型类的 attention/MLP 各自独立实现)。
- 结论：**挑结构最简单、变量最少的那个模型做基准**(稠密 > MoE，TP=1 > TP=N，fp16 > fp8)。
  变量越少，逐层对比越干净。

把对照表写下来再动手，例如：

| 模型 | 结构 | 激活 | 精度/卡数 | 结果 |
|---|---|---|---|---|
| Qwen3.x-MoE | MoE | silu | fp16/4卡 | ✅ |
| gemma-4-31B | 稠密 | gelu_tanh | fp8/2卡 | ❌ |
| gemma-4-26B | MoE | gelu_tanh | fp8/2卡 | ❌ |

→ 选 31B(稠密、变量少)做基准。

## 第 1 步：HF 端造 golden reference

用 `assets/hf_reference.py` 作模板(实战已验证可跑 31B)。改这几个常量后即可用：
`MODEL_DIR` / `PROMPT` / `DUMP_FILE` / 要 trace 的层 `layer_indices_to_trace`。

关键设计点(**照抄，都是踩过的坑**)：

1. **CPU 优先**：默认 `device_map={"": "cpu"}`，fp16 全模型放 CPU RAM。
   慢但**稳**(XPU 加载偶发 device_lost；CPU 数值也最接近"理论值")。
   31B fp16 ≈ 62GB，确认 host RAM 够(`free -g`)。环境变量 `HF_REF_DEVICE=xpu` 可切跨卡。
2. **跨卡分布式(显存不够时)**：`init_empty_weights()` 建空壳 → `load_checkpoint_and_dispatch`
   按 `device_map` 把 N 层摊到多卡，`no_split_module_classes=["<DecoderLayer 类名>"]` 保证单层不被拆。
   HF **没有原生 TP**，这是 pipeline/层间分卡(`accelerate`)，逐层前向时数据自动 dispatch。
3. **只加载语言模型权重**：多模态模型 ckpt key 常是 `model.language_model.*`，而 `XXXForCausalLM`
   期望 `model.*`。需要 rename 前缀并存成临时 sharded ckpt(省 RAM，别一次性 load 进 host)，
   跳过 vision/audio tower。tied weights(`lm_head.weight = embed_tokens.weight`)要先 `model.tie_weights()`，
   否则 `device_map` 下报 "on meta device"。
4. **`_attn_implementation="eager"`**：禁用 sdpa/flash，保证数值**可复现**、与后端逐 op 对齐。
5. **forward hook dump**：在 embed_out / 选定 decoder layer 输出 / pre&post final_norm / lm_head in&out
   注册 hook，打印**统一格式**：
   ```
   [TRACE-HF] {name}: shape=... dtype=... min=.. max=.. mean=.. absmax=.. nan=.. inf=.. first4=[..]
   ```
   `absmax`/`nan`/`inf`/`first4` 是发散最敏感的指标。最后再打 `last_token_logits` 的 top5_ids。
6. **chat template 对齐**：用 `apply_chat_template(..., add_generation_prompt=True)` 让输入 token 和后端
   `LLM.generate` **完全一致**(否则首 token 就对不上，白比)。

## 第 2 步：后端(vLLM)端打同格式 trace

在后端模型文件(如 `vllm/model_executor/models/gemma4.py`)的 `forward` 里注入 trace，
用**环境变量门控 + 只在 rank0 打印**(TP 下每卡都打会乱)：

```python
# ===== TRACE BEGIN =====
import os as _os
_trace = _os.environ.get("GEMMA4_TRACE", "0") == "1"
_is_rank0 = (get_tensor_model_parallel_rank() == 0)
def _t(name, t):
    if not (_trace and _is_rank0): return
    tf = t.detach().float()
    print(f"[TRACE] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={tf.min().item():.4g} max={tf.max().item():.4g} "
          f"mean={tf.mean().item():.4g} absmax={tf.abs().max().item():.4g} "
          f"nan={int(tf.isnan().any())} inf={int(tf.isinf().any())} "
          f"first4={tf.flatten()[:4].tolist()}", flush=True)
# 在 embed 后 / 每个 trace 层后 / final_norm 前后 / logits 处调用 _t(...)
```

格式**必须和 HF 端逐字段一致**(`[TRACE]` vs `[TRACE-HF]` 只差前缀)，这样才能机械 diff。
用 `enforce_eager=True` 跑后端(关 graph，逐层可观测，也匹配基准数值)。

注意 TP 切分：后端 hidden_states 在 rank0 可能只是分片，对比时关注 **mean/absmax/分布**和**首处 nan/inf**，
而非逐元素相等；只有 TP=1 才能严格逐元素比。**优先用 TP=1 跑基准**。

## 第 3 步：逐层 diff 找首处发散

两份 trace 各自存 log(`/tmp/hf_*_trace.log` 与后端 stdout)，按层名对齐：

```bash
# 抽出两边的 TRACE 行，去掉前缀后并排看
grep '\[TRACE-HF\]' /tmp/hf_31b_trace.log     | sed 's/\[TRACE-HF\] //' > /tmp/a.txt
<后端日志> grep '\[TRACE\]' | sed 's/\[TRACE\] //'                     > /tmp/b.txt
diff <(...) <(...)   # 或逐行肉眼对 mean/absmax
```

从 `embed_out` 往后扫，**第一处 mean/absmax 明显偏离、或一边 nan/inf** 的层就是发散起点。
发散前的层都对 ⇒ bug 在发散层的某个子算子(attention/norm/激活/残差)。继续在该层内部加
更细的 trace(QK-Norm 后、RoPE 后、attn_out、各 layernorm 前后)二分定位到具体 op。

## 常见根因清单(从实战)

- 权重加载：stacked qkv/gate_up 合并顺序、权重名映射错位、tied weight 没 tie。
- embedding 归一化(`* sqrt(hidden_size)`)漏了或位置错。
- layernorm 夹心残差结构(pre/post-attn、pre/post-mlp 四个 norm)接错。
- RoPE：滑窗层 vs 全局层用了不同 base/参数，混用导致部分层错。
- QK-Norm / V-Norm 实现或顺序不符。
- 激活函数(gelu_tanh / silu)的 kernel 实现或 `approximate` 模式不符。
- 后端定制 kernel(fp8 GEMM / paged_attn)在该 shape 下数值不对。

## assets

- `hf_reference.py` — 实战验证过的 HF golden-reference 模板(gemma-4-31B)。
  换模型时改 `MODEL_DIR`/`PROMPT`/层列表/`DecoderLayer` 类名/权重前缀即可。
