# [XPU] Add fused speculative GDN kernel for Qwen3.5/3.6

## Summary

为 Qwen3.5/3.6 在 XPU 上的 speculative decode 增加 sequential-layout GDN fused kernel。

该路径将 speculative GDN 中原本分离的：

- causal Conv1d；
- GDN state update；
- output / gate 写回；
- speculative Conv state checkpoint；

合并为一个 kernel，减少中间 tensor、Python 编排和 kernel dispatch 开销。普通 decode GDN 路径保持不变，不支持的 geometry 会显式 fallback 到原始实现。

## Motivation

Qwen3.6-35B 的 MTP target verify 使用 `M=3` speculative GDN。原始路径需要为每个 draft token 单独处理 Conv 和 GDN state，并产生额外的中间 buffer 和 dispatch。

之前的 profile 显示：

- GDN core 约 `0.06 ms/layer`；
- 30 个 linear-attention 层合计约 `1.8 ms`；
- GDN 不是 target verify 的主要瓶颈，但属于可以通过融合减少的固定调度成本。

## Design

### Kernel geometry

当前只支持经过验证的 geometry：

| Parameter | Supported value |
| --- | --- |
| Q/K heads `H` | `8` |
| V heads `HV` | `16` 或 `24` |
| `K` | `128` |
| `V` | `128` |
| Work-group size | `64` |

`HV=16` 覆盖 Qwen3.6-35B TP=2，`HV=24` 覆盖对应的 Qwen3.6 geometry。其他 geometry 在 host dispatcher 中通过 `TORCH_CHECK` 失败，不会静默使用错误 layout。

### Main kernel

`esimd_gdn_conv_fused_seq_spec` 为每个 `(speculative sequence, value head)` 分配一个 work-group，并按 token 顺序处理：

1. 从 sequential `[q|k|v|z]` projection 中执行 Conv1d；
2. 在 SLM 中交换 Q/K/V；
3. 执行 GDN delta-rule update；
4. 写出 attention output 和 `z`；
5. 将每个 speculative token 的 SSM state 写入对应 checkpoint；
6. 保留下一 token 需要的 Conv state。

按 sequence 顺序处理 token 是为了保持 rollback 语义，初始 state 由上一步接受 token 数决定。

### Conv state checkpoint ownership

主 kernel 按 value head 并行，因此 Q/K lanes 会被多个 work-group 重复执行。为避免多个 work-group 写同一 Conv checkpoint：

- `HV=16/24` 的每个 value-head work-group 只写自己的 V slice；
- 只有 `hv=0` 的 work-group 写 replicated Q/K slices。

这样每个 checkpoint slice 都有唯一 writer，不需要额外的 state-checkpoint kernel，也不会重复读取 `qkvz`。

### vLLM integration

vLLM 只在以下条件下启用新路径：

- speculative decode；
- 当前没有 prefill 或普通 decode token；
- `num_spec_decodes == 1`；
- `0 < num_spec_decode_tokens <= 128`；
- sequential Qwen layout；
- Conv/SSM cache 为 FP16；
- `DISABLE_ESIMD_GDN_SPEC=1` 未设置；
- active extension 导出了新 op。

不满足条件时继续走原始 GDN 实现。普通 `esimd_gdn_conv_fused` 和 `esimd_gdn_conv_fused_seq` 没有替换或修改其计算路径。

## Performance

配置：Qwen3.6-35B-A3B、FP8、TP=2、XPU card 4/5、oneCCL 2021.17、eager、speculative tokens=2、20 条 ShareGPT、并发 1。

| Configuration | Throughput | Median TPOT | Acceptance length |
| --- | ---: | ---: | ---: |
| Speculative GDN disabled | 35.27 tok/s | 28.76 ms | 2.38 |
| Speculative GDN enabled | 36.77 tok/s | 27.17 ms | 2.43 |

初步端到端收益约 `+4.3%`。该数据来自本次单 kernel ownership 优化前的 A/B，合入前应重新测量并确认收益没有被新 kernel geometry 改变。

注意：后续 M=3 all-reduce retry guard 修复将整体 MTP 性能从约 `57 tok/s` 提升到 `102.55 tok/s`，这是独立的 vLLM communication-path 修复，不能归因于本 kernel。

## Correctness

已有验证：

- 同一确定性短 prompt 下，enabled/disabled 输出文本一致；
- Qwen3.6-35B GSM8K 5-shot，`max_tokens=2048`：`5/5`；
- speculative path 的 workspace 对拍矩阵已通过，覆盖 accepted token 数 `0/1/3` 和 token permutation；
- 现有普通 GDN regression test 继续保留。

本 PR 同时补齐了新 dispatcher schema 的 mutation 标注：

- `conv_state`：`Tensor(a!)`；
- `ssm_state`：`Tensor(b!)`；
- `output`：`Tensor(c!)`；
- `z_out`：`Tensor(d!)`；
- 返回值 alias 到 `output`：`-> Tensor(c!)`。

这与 kernel 的实际原地写行为一致，避免 dispatcher alias analysis、functionalization 或后续 graph 路径误判 tensor 没有被修改。

## Performance and correctness risks

### 1. Draft token 数增加时延迟线性增加

主 kernel 在一个 work-group 内顺序遍历所有 speculative token。该设计针对当前 `M=3`，不适合直接推广到很大的 speculative window。

### 2. Q/K 计算在 value-head work-group 之间重复

每个 value-head work-group 都会读取和处理对应的 Q/K。当前实现优先保证 geometry 简单和 state isolation，尚未做跨 value-head 的 Q/K 共享。

### 3. State ownership 依赖固定 geometry

checkpoint ownership 依赖 `H=8`、`HV=16/24` 的线程映射。若扩大 geometry，必须同步重新证明 Q/K/V slice 没有重复 writer 或遗漏 writer。

### 4. SLM、barrier 和寄存器压力

- 每个 work-group 使用约 2 KB SLM；
- 每个 token 至少经过两次 barrier；
- `simd<fp16, 256>` 和多个 `simd<float, 64>` 临时值可能提高 GRF 压力；
- `gdn_spec_update_seq` 中 Q/K normalization 在各线程中重复计算。

这些因素可能限制 occupancy，必须以 kernel-level profile 为准，不能只根据单次端到端结果决定进一步融合。

### 5. Geometry 和 dtype 限制

当前只支持 `H=8`、`HV=16/24`、`K=V=128` 和 FP16 state。其他 TP、模型变体或 FP32 state 会 fallback，不应强行放宽 host gate。

### 6. Metadata contract

kernel 假设 vLLM 已保证以下 metadata 有效：

- `spec_state_indices` 的 index 范围；
- `token_indx` 的 token 范围；
- `num_accepted_tokens` 与 state checkpoint 布局匹配；
- input/output/state tensor contiguous 且位于同一 XPU device。

当前 wrapper 没有重复执行完整的 shape、dtype、device 和 index 检查；这些条件由 vLLM caller 保证。若未来该 op 被其他 caller 复用，应增加 fail-fast validation。

## Scope and impact on other kernels

### No direct change to ordinary GDN behavior

新 op 是 additive registration。普通 decode 仍使用原有 `esimd_gdn_conv_fused` 或 `esimd_gdn_conv_fused_seq`，不支持 speculative geometry 时显式 fallback。因此该改动不会改变普通 GDN 的默认 dispatch。

### MoE profiling cleanup

`VLLM_MOE_STAGE_PROFILE` 诊断代码已删除，不再改变 MoE 默认路径或引入额外 `wait_and_throw()`。

## Test gaps before merge

当前已提交的 `test_gdn_conv_fused_tp_fix.py` 测试的是普通 `esimd_gdn_conv_fused`，并没有直接调用新的 speculative op。合入前建议补充专用回归测试，至少覆盖：

- `HV=16` 和 `HV=24`；
- `num_spec_decodes=1/2`；
- `num_spec_tokens=1/2/3`；
- accepted token 数 `0/1/3`；
- token permutation；
- output、`z`、Conv state、SSM state；
- 不支持 geometry 的 `TORCH_CHECK`；
- contiguous、dtype 和 state-layout 边界。

还应补充 kernel-level A/B：

- `M=1/2/3/4`；
- fused single-kernel path 与原始 sequential GDN path 对比；
- 确认 `DISABLE_ESIMD_GDN_SPEC=1` 能够精确回退。

## Validation checklist

- [x] Build GDN/LGRF extension；
- [x] Qwen3.6-35B deterministic fingerprint；
- [x] GSM8K 5-shot accuracy `5/5`；
- [x] ShareGPT enabled/disabled A/B；
- [x] Unsupported geometry fails explicitly；
- [x] Workspace speculative GDN matrix（HV=24）；
- [ ] Add in-repository speculative GDN matrix test；
- [ ] Repeat performance A/B with stable median；
- [x] Rebuild the active `.so` after the dispatcher mutation-schema change.

## Commits

- `1c2920c` — `[XPU] Add Qwen speculative GDN ESIMD kernel`
- `65b460e` — `[XPU] Support Qwen35 speculative GDN HV16`
- 后续 cleanup commit — 删除 `VLLM_MOE_STAGE_PROFILE` 诊断代码，并将 Conv checkpoint 写回主 kernel
