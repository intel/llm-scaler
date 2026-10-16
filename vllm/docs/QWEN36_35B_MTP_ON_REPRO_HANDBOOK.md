# Qwen3.6-35B-A3B MTP ON 空答案与 matched A/B 异常复现手册

> PR 发布的是本独立手册。下文的 `harnesses/`、`evidence/`、`tests/`、`provenance/` 和 `SHA256SUMS` 属于本机复现包 `/home/intel/guancheng/ima-tests/qwen36-35b-mtp-on-repro`，不在这个 docs 目录。执行命令前进入该复现包；原始证据、脚本、镜像与权重未上传本 PR。

更新时间：2026-10-16 UTC。用途：将一个尚未解决的问题交给另一位工程师独立复现、定位和修复。本次交付只整理证据和工具，没有重新启动 GPU 测试或修改 release 镜像；此副本随文档更新发布到 PR。

## 1. 要复现什么，以及不能下什么结论

问题发生在 **Qwen3.6-35B-A3B、MTP ON、TP2、online FP8、eager、thinking GSM8K 200 题 / 32 并发**。候选镜像首轮出现 1 个空 final answer；同条件 matched A/B 复测出现 2 个空 final answer，thinking correct 从基线的 196/200 降到候选的 191/200。

这里没有启用 LMCache，没有 KV connector，也没有 Attention8。不要把它与 Attention8 缓存量化回放的问题混在一起。

已经证实的是“候选两轮都有空 final，并触发复测/调查条件”，不是“OMIX、Torch、MTP kernel 或 reasoning parser 中某一处已经被证明有 bug”。Thinking 用 temperature=0.6、未设置请求 seed，单个 A/B 对不能证明统计显著退化，更不能保证每次恰好在同一题复现。需同时调查生成、停止条件和 reasoning/final 分离。

## 2. 已有结果：首轮与 matched 复测分开呈现

表内括号为 invalid，所有轮次各 200 题。

| 轮次 | 镜像 | Thinking correct（invalid） | Non-thinking correct（invalid） |
|---|---|---|---|
| 首轮 | baseline overlay | 194（0） | 187（0） |
| 首轮 | candidate r2 full build | 194（1） | 187（0） |
| matched 复测 01 | baseline overlay | 196（0） | 191（0） |
| matched 复测 01 | candidate r2 full build | 191（2） | 194（0） |

matched thinking 差值：correct −5/200，即 −2.5 个百分点；invalid +2。所有轮次 accuracy 都超过 0.8，所以原 `result.json` 和命令退出码为 PASS/0；**它们只代表最低准确率门槛通过，不代表无回归或问题已解决**。调查触发条件为 correct 下降至少 4/200，或新增 invalid；matched 复测两项均触发。

### 三个关键空答案记录

| 候选轮次 | evaluator index（从 0 开始） | 数据集 index / 文件行 | gold | final 字符数 | reasoning 字符数 | completion tokens | finish_reason |
|---|---|---|---|---|---|---|---|
| 首轮 | 54 | 59 / 60 | 187 | 0 | 18283 | 16384 | length |
| matched 复测 01 | 115 | 120 / 121 | 240 | 0 | 708 | 172 | stop |
| matched 复测 01 | 186 | 191 / 192 | 5 | 0 | 16875 | 16384 | length |

三条均 `kind=empty_output`、`response=""`、`error=null`、`extracted=-9999999`。`reasoning_len` 是字符数，不是 tokens；`completion_tokens` 是 API usage 的总生成 token 计数，不是 final answer tokens。

需要区分两个现象：

- `length`：推理消耗了 16384-token 预算，没有留下可提取的 final answer；需看全文是否循环、是否缺少结束标签。
- **`stop` + 172 tokens + 空 final**：不能用“生成预算不够”解释。优先追踪 reasoning 内容、EOS/停止原因和 parser 分离过程。但 `finish_reason=stop` 本身不能证明是哪个 token 或哪个停止分支触发。

题目：

- index 54：`A raspberry bush has 6 clusters of 20 fruit each and 67 individual fruit scattered across the bush. How many raspberries are there total?`
- index 115：`Miguel uses 2 pads of paper a week for his drawing. If there are 30 sheets of paper on a pad of paper, how many sheets of paper does he use every month?`
- index 186：`Ducks need to eat 3.5 pounds of insects each week to survive. If there is a flock of ten ducks, how many pounds of insects do they need per day?`

不是裸题请求：原协议先把同一 test 文件的前 5 题及其完整答案作为 demonstrations，随后评测该文件第 6–205 题。不要换成 train split demonstrations，也不要误用 test 的前 200 行。

### 已排查 / 尚未拿到的证据

已有四组健康检查、实际 warmup、GSM8K 200×2、64K recall 和 MTP 加载证据。三条异常不是客户端 HTTP error；这几组未记录新增服务崩溃/OOM。候选 matched 64K 实际 API prompt tokens=65525、needle 正确，单次耗时 7.473 秒；这不是性能门槛。

MTP 日志实际解析为 `Qwen3_5MoeMTP`，两个 TP worker 均记录共享 target embedding/lm_head。归档 metrics 存在非零 `spec_decode_num_drafts_total` 和 `spec_decode_num_accepted_tokens_total`，证明实际运行过 speculative decoding，不能用仅传入配置代替加载/执行证据。单次 metrics snapshot 不证明整轮平均 acceptance 或加速。

**原 `run_gsm8k.py` 只保存了 final 文本、reasoning 长度、finish reason 和 completion tokens，没有保存完整 reasoning、完整 API body 或原始生成 token。历史缺失不能事后补全。** 本包新增 client-side API capture 工具供新的运行使用。

## 3. 独立复现包与证据位置

本机目录：`/home/intel/guancheng/ima-tests/qwen36-35b-mtp-on-repro`。整个目录可打包转交；原始证据从 OMIX 归档复制，未覆盖原文件。

| 相对路径 | 内容 |
|---|---|
| `harnesses/run_matrix.py` | 原始单组启动、warmup、200×2、64K、归档流程；未修改 |
| `harnesses/run_gsm8k.py` | 原始评测、prompt、sampling、答案提取；未修改 |
| `harnesses/run_large_context.py` | 原始 64K 测试；未修改 |
| `harnesses/gsm8k-test.jsonl` | 原测试文件，不需要联网下载 |
| `harnesses/reproduce_pair.py` | 新增隔离 A/B 入口，默认两对，只运行本问题模型/MTP ON |
| `harnesses/capture_gsm8k_api.py` | 调用原 post()，记录完整请求/响应，保留原 parsing 和 scoring |
| `harnesses/audit_examples.py` | 对齐两组各 200 题，列出 correct/invalid/空答案及调查触发项 |
| `tests/` | API 转发 identity、异常传播、并发 JSONL、启动驱动拦截与资产校验离线测试 |
| `evidence/baseline-eager/` | 基线首轮原始单组目录 |
| `evidence/candidate-functional-r2-eager/` | 候选首轮原始单组目录 |
| `evidence/baseline-final-retest01-eager/` | 基线 matched 复测原始单组目录 |
| `evidence/candidate-functional-r2-retest01-eager/` | 候选 matched 复测原始单组目录 |
| `provenance/` | 权重 inventory、依赖、镜像/源码身份及 patches |
| `SHA256SUMS` | 包内文件完整校验清单（不含清单自身） |

每组 evidence 都保留 `launch.command.json`、thinking/non-thinking command/log/summary/examples、`server*.log`、`container*.log`（JSON inspect）、`result.json`、`large-context.json`、MTP 日志/metrics、live worker 记录。文件中的原绝对路径仍指向原归档，搬走本包后应使用本包对应相对文件，不代表路径失效就是证据缺失。

原归档：`/home/intel/guancheng/omix040-v026-20261015`。原源码：`compatibility-attempt03/sources/vllm`。release 仓库：`/home/intel/guancheng/llm-scaler-omix040-v026`，分支 `feat/omix-040-v026`，commit `b4bd2671048ea412c4837f78a4379fd953c2e888`，关联 draft PR：<https://github.com/intel/llm-scaler/pull/764>。

## 4. 固定环境与启动契约

### 镜像身份：用 image ID，不要只相信可变 tag

Baseline：`llm-scaler-vllm:pr707-review-fix`

```text
sha256:1091b5fc32eb50ec0b58cb00cbde3c1ab5e30deee2a6fc021fd79f13a66f8fc0
```

它是 **Python-only overlay 基线，不是与候选同源码配置的 clean full build**。相关差异不能仅归因于 Ubuntu/Torch 一项。

Candidate：`llm-scaler-vllm:omix040-v026-functional-r2`

```text
sha256:1882a2d8911107e282a0ebb392e13e10bae5cca4093ee8064d0d93c591dde468
```

Candidate 基础镜像：`intel/omix:0.4.0-devel-ubuntu26.04@sha256:59cef13f8e87275846cc08e01d32b49082694a5c01d842d2e845552cbc8c199f`。

| 依赖 / runtime | baseline | candidate r2 |
|---|---|---|
| Python | 3.12.3 | 3.12.14 |
| Torch | 2.12.0+xpu | 2.14.0+xpu |
| Triton XPU implementation | 3.7.1 | 3.8.0（含 triton==3.8.0+xpu shim） |
| Intel SYCL runtime 包 | 2025.3.2 | 2026.1.1 |
| oneCCL | 入口 source 系统 2021.15；pip metadata 无 oneccl 包 | oneccl / oneccl-devel 2022.1.2 |
| vLLM wheel metadata | 0.26.1.dev0+g568afb3a1.d20261012.xpu | 0.26.1.dev0+g568afb3a1.d20261015.xpu |

Candidate 实际 vLLM 源码是 **v0.26.0**、HEAD `568afb3a13806beb53bb2e6bd518269357b237c0`；wheel 的 dirty SCM 开发版本标签不是升到了 v0.31。更多依赖在 `provenance/runtime-final-versions.txt` 与 `provenance/baseline-native-imports-attempt-03.log`。

使用镜像的正常 ENTRYPOINT；不要为了诊断直接绕过 oneAPI 初始化。Baseline 原入口确实 source 2021.15；candidate 原入口 source oneAPI 后优先 `/opt/venv/lib`。不要给 candidate 手工添加旧 CCL source 或把宿主库塞进 LD_LIBRARY_PATH。包里的 image/container inspect 是判断依据。

### 设备、权重、参数

- 仅 GPU **6、7**，顺序 `ZE_AFFINITY_MASK=6,7`。本机 `xpu-smi` 标识为 Intel Graphics `[0xe223]`；原机映射为 GPU6=`0000:c2:00.0`、GPU7=`0000:d6:00.0`，每次运行仍需用 `xpu-smi discovery` 核对。
- 权重 `/home/intel/weights/Qwen3.6-35B-A3B`，挂载 `/home/intel/weights:/models:ro`；served model `Qwen3.6-35B-A3B`。
- TP2、`--quantization fp8 --dtype float16 --enforce-eager`。
- `--gpu-memory-utilization 0.75 --max-num-seqs 32 --max-num-batched-tokens 8192 --block-size 64`。
- `--max-model-len 131072 --mamba-ssm-cache-dtype float16 --enable-prefix-caching`。
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --trust-remote-code`。
- `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}'`。
- Docker：`--privileged --device /dev/dri --shm-size 40g`，`/dev/dri/by-path` 只读挂载；原端口 19160，监听宿主 `127.0.0.1`。
- 无 graph 开关、无 `--kv-transfer-config`、无 LMCache server。

权重 config SHA256=`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`；chat template SHA256=`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`。`provenance/model-input.json` 包含 26 个权重 shards 的 size/mtime 和 JSON/Jinja 哈希；此前没有对全部 shard 内容做 SHA256，不能宣称已做完整权重内容校验。

原显式环境（新入口原样使用）：

```text
ZE_AFFINITY_MASK=6,7
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
CCL_TOPO_P2P_ACCESS=0
CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLTOALL_TMP_BUF=1
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

## 5. 一键 matched A/B：推荐原机复现路径

先读本节，不要直接运行整个十模型矩阵。包内新增入口不修改镜像或 frozen driver，只把 GSM8K 客户端换成记录完整 API 的 wrapper。

```bash
cd /home/intel/guancheng/ima-tests/qwen36-35b-mtp-on-repro
sha256sum -c SHA256SUMS
python3 -m unittest discover -s tests -v
xpu-smi discovery
xpu-smi ps -d 6 -j
xpu-smi ps -d 7 -j
docker image inspect llm-scaler-vllm:pr707-review-fix \
  llm-scaler-vllm:omix040-v026-functional-r2 --format '{{.RepoTags}} {{.Id}}'
```

任何占用先联系资源所有者，不停止别人容器/进程。共享 lock 是原归档的 `gpu67.lock`；它只能协调遵守该 lock 的脚本，不能代替占用检查。当前监控可能在 `xpu-smi ps` 留下名为 `xpu-smi` 的条目；入口忽略这些监控条目，遇到其他进程则退出。

选择**不存在的新结果目录**；下面名称只是示例，请每次改成新的名字。Dry-run 不创建目录、不调用 Docker、不运行 GPU。

```bash
python3 harnesses/reproduce_pair.py \
  --run-dir /home/intel/guancheng/qwen35-mtp-repro-20261016-a \
  --rounds 2 --port 19160 --dry-run

python3 -u harnesses/reproduce_pair.py \
  --run-dir /home/intel/guancheng/qwen35-mtp-repro-20261016-a \
  --rounds 2 --port 19160
```

实际执行前检查四个 frozen 资产 SHA、权重 inventory、两个 image ID、端口与 GPU 占用；获得非阻塞独占 lock 后，依次执行 `round01/baseline → round01/candidate → round02/baseline → round02/candidate`。每组：新容器、health、warmup、thinking 200/32、non-thinking 200/32、64K、MTP 证据、服务日志与 inspect、停止容器释放 GPU。不会删镜像、删旧证据或安装 packages。若启动/硬门槛失败，停止后续运行并保留失败目录。

预计需要数十分钟，实际依机器而变。空答案本身可能仍有 exit code 0：脚本结束提示不是“问题解决”。所有运行独立编号，不覆盖首轮。

新结果结构：`<run-dir>/roundNN/{baseline,candidate}/qwen3.6-35b-a3b-mtp-on-eager/`。新增 `thinking-full-api.jsonl` 和 `non-thinking-full-api.jsonl`，每条保存 question index、完整请求、完整解析 JSON 响应（包括 reasoning/content/usage/finish_reason）、时间、错误与 thread ID。JSONL 按完成顺序，不按题号；用 `question_index` 对齐。

Wrapper 将原 payload 按对象 identity 交给原 post()，把原 response 原样返回原解析器，保留异常类型与 scoring；不添加 seed、不改 stop、不改预算或并发。磁盘记录会增加少量客户端耗时，不能声称与未插桩运行的调度时间完全一致，也不能用该 wrapper 做严格性能基准。

### 固定评测参数（手动接入自己的服务时也要保持）

Thinking：`--api chat --num-shots 5 --num-examples 200 --num-threads 32 --thinking --max-tokens 16384 --temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.05 --no-chat-stop --timeout 600 --min-accuracy 0.8`。

Non-thinking：`--api chat --num-shots 5 --num-examples 200 --num-threads 32 --no-thinking --max-tokens 2048 --temperature 0 --top-p 1 --chat-stop --timeout 600 --min-accuracy 0.8`。后者 stop 数组实际为 `["\nQuestion:", "Question:"]`；thinking 请求不传 stop。

Chat payload 总是传 `chat_template_kwargs.enable_thinking`。两轮均不设置 seed。答案从 **final content 中最后一个数字**提取，不从 reasoning 补答案，不改为“任何位置出现 gold 即正确”。

手动客户端示例，假设服务已按第 4 节启动且 warmup 完成：

```bash
python3 -u harnesses/capture_gsm8k_api.py \
  --api-log /path/to/new-results/thinking-full-api.jsonl \
  --host 127.0.0.1 --port 19160 --model Qwen3.6-35B-A3B --api chat \
  --data-path harnesses/gsm8k-test.jsonl --num-shots 5 --num-examples 200 \
  --num-threads 32 --thinking --max-tokens 16384 --temperature 0.6 \
  --top-p 0.95 --top-k 20 --repetition-penalty 1.05 --no-chat-stop \
  --timeout 600 --min-accuracy 0.8 --out-prefix /path/to/new-results/thinking
```

## 6. 如何检查新结果与定位

先查空答案 / invalid，再看总 correct；不能只看 summary 的 accuracy：

```bash
python3 harnesses/audit_examples.py \
  --baseline /path/to/round01/baseline/qwen3.6-35b-a3b-mtp-on-eager \
  --candidate /path/to/round01/candidate/qwen3.6-35b-a3b-mtp-on-eager

jq -c 'select(.extracted == -9999999) |
  {index,gold,kind,content_len,reasoning_len,completion_tokens,finish_reason,error,response}' \
  /path/to/group/thinking_examples.jsonl

jq -c 'select(.question_index == 115) |
  {question_index,request,response,error}' /path/to/group/thinking-full-api.jsonl

jq -c 'select(.response.choices[0].message.content == null or
             .response.choices[0].message.content == "")' \
  /path/to/group/thinking-full-api.jsonl
```

排查顺序：

1. 对齐同一 question index 的 request SHA/payload，確認 demonstrations、sampling、thinking、stop、模型与设备未变化；检查 HTTP error、服务退出和 MTP 加载/执行证据。
2. 针对新出现的 `stop` 空 final，检查完整 `message.reasoning_content` / `message.reasoning`、`content`、`tool_calls`、`finish_reason`、`stop_reason`（如果有）和 usage。完整 API 仍不是未解析 token stream；若答案在 reasoning 内或 EOS 发生在 `</think>` 前，继续在独立诊断镜像记录原始 token/engine 输出和 parser 输入输出。
3. 针对 `length` 空 final，检查是否推理循环、自我修正失控、缺结束标签，还是正常推理未完成。不要从未保存的历史 reasoning 长度推断全文。
4. 重点代码候选：vLLM `reasoning/qwen3_engine_reasoning_parser.py`，`model_executor/models/qwen3_5_mtp.py`，`v1/spec_decode/llm_base_proposer.py`，`v1/sample/rejection_sampler.py`，`v1/engine/output_processor.py` 及实际 API chat serving/停止路径。它们是检查入口，不是已确定根因。先核对实际安装模块路径与归档 patch，避免诊断另一版源码。
5. 可追加单题重复、固定 seed、MTP OFF、speculative tokens=1、低并发、FP16/不同 backend、禁用 parser 等**独立诊断**，每次只改一个变量，明确标注。单题/seed 改变批处理和采样条件，不保证复现原 32 并发异常；诊断结果不能替代原配置 matched 测试。

仅看到 196→191 不足以宣布某个 kernel 回归；仅看到一次“没复现”也不足以关闭两轮新增空答案。Non-thinking temperature=0 的两轮 correct 也有变化，不应假设温度为 0 就必然跨批处理逐字确定。

## 7. 修复后的回归与交付

修复使用新的分支/镜像标签与 image ID，保留未修改候选和基线。记录问题定位链：异常 API / token 证据 → 原因 → 最小 patch → 新镜像/源码/wheel 身份 → 原条件复测。

至少完成修复前后各同条件 matched pair，推荐重复两对；每组仍需 200×2、32 并发、原预算/采样/stop、实际 MTP、64K。报告 correct/invalid/empty-stop/empty-length/error 的数值和差值，不保证固定随机采样必须得到特定分数，也不把有限复测没有观察到异常写成数学保证。持续触发时保留未解决结论。

禁止以关闭 MTP、关闭 thinking、降低并发、缩小题数、扩大 token 预算、改答案提取或把 reasoning 当 final 的方式宣称原问题已修复。Graph/Attention8 不在本问题原复现路径内。

交付表：

| 轮次/image ID | thinking correct/200 | invalid | empty-stop | empty-length | HTTP error | non-thinking correct/invalid | 64K tokens/recall | MTP 实际证据 | 结论/证据 |
|---|---|---|---|---|---|---|---|---|---|
| 待填写 | | | | | | | | | |

结论分别写 PASS（限定具体检查）、FAIL、BLOCKED、NOT RUN；准确率触发项单列“持续触发 / 未观察到 / 已定位并修复且复测未再观察到”。日志归档后只处理自己的新容器；不得批量清理引用该镜像的其他容器，不能覆盖本包 `evidence/`。

## 8. 异机转交与限制

本包包含文档、脚本、冻结数据和原始证据，**不包含 Docker image tar、模型权重或全部源码**。两个镜像目前只保留在本机、未上传 registry；接手者应在本机使用，或另行安排 `docker save/load` 与权重交付。不要 `docker pull` 同名 tag 然后声称镜像身份相同。

异机需验证 GPU 架构/驱动和两个 image ID、同一权重与配置/模板/数据哈希；若模型目录位置或 mtime 改变，原机专用 `reproduce_pair.py` 权重校验会拒绝执行。应复制为单独的异机入口，记录目录映射、重新确认 shard 内容后采用新的资产校验，不修改本包 frozen driver/hash；设备仍按获得的授权固定，不能偷偷沿用不适用的编号。异机结果与原机问题相关，但不构成严格同机器 matched A/B。

分享前检查 inspect/logs 中的环境变量与内部路径；本包未额外导出宿主完整环境或凭据。脱敏版另存，保留原始证据及其 SHA。

给接手者的任务：

> 请读 HANDBOOK.md，先校验包、镜像和权重。在授权 GPU 6、7 上使用原 online FP8/TP2/eager/MTP ON 配置做同条件 matched A/B，保留 200×2/32 并发/64K 和完整 API。重点定位 `stop` + 172 tokens + 空 final 与 `length` + 16384 tokens + 空 final；不要将它与 Attention8 混为一谈。保留第一次失败，最小修复后用新镜像原条件复测，交付根因、patch 和逐项原始证据。降低测试要求或关闭 MTP 的结果只记诊断，不算修复。
