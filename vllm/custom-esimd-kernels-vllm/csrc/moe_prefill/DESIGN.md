# MoE Prefill INT4 Kernel 设计文档

目标：在 llm-scaler 现有 `custom-esimd-kernels-vllm` 工程下，新增一套**专为 prefill 场景**设计的 INT4 MoE kernel，与现有的 decode 版 `moe_int4_ops` 并存，按 shape 分发。

- **首要目标**：替换 ipex `ipex.llm.modules.GatedMLPMOE` 在 sym_int4 prefill 路径上的调用，修复 `TORCH_LLM_ALLREDUCE=1` + 32k×batch8 100% hang 问题。
- **次要目标**：长期去 ipex 化，prefill 性能对标或超过 ipex xetla marlin。
- **模型目标**：Qwen3.5-122B-A10B（sym_int4, TP=4）。

---

## 1. 背景与定位

### 1.1 现有实现盘点

| 实现 | 位置 | 目标场景 | 能否跑 prefill | 说明 |
|---|---|---|---|---|
| ipex marlin int4 | `frameworks.ai.pytorch.ipex-gpu/csrc/gpu/aten/operators/XEGEMM_INT4_marlin.cpp` | 通用 MoE，含 prefill | 能，但当前 hang | xetla DPAS 一次性 grouped kernel；怀疑 host-side atomic_buffer/event 路径导致 rank 间不对称 |
| llm-scaler `moe_int4_ops` | `llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl` | **Decode (n_tokens ≤ 4)** | 能跑但性能差 30-100× | 无 DPAS，纯 `simd<T,32> block_load + 标量 FMA`；每 WI 算 16 个 N 输出，GEMV 外积分布 |
| 套 A prefill_v2 | `frameworks.ai.client-ai.esimd-kernels/.../custom-esimd-kernels-vllm-moeops/csrc/moe_prefill_v2.cpp` | **Prefill (n_tokens ≫ 4)** | 未编译，设计为 4-op 管线 | **主参考**：`xmx::dpas<8,8,fp16>` + `lsc_load_2d` + SLM atomic gather |

**本设计基于套 A，但集成到 llm-scaler、兼容 IPEX marlin 权重格式、对接现有 `moe_forward_full_int4` 风格的 API。**

### 1.2 Qwen3.5-122B-A10B 关键 shape（TP=4 每 rank 视角）

| 参数 | 全局 | 每 rank |
|---|---|---|
| hidden_size H | 3072 | 3072（不切） |
| moe_intermediate_size I | 1024 | **256**（切 4 份） |
| shared_expert_intermediate_size | 1024 | **256** |
| num_experts E | 256 | **64**（切 4 份） |
| top_k | 8 | 8 |
| num_shared_experts | 1 | 1 |
| quant group_size GS | 128 | - |

**prefill 输入**：M = 32768 tokens/batch × 8 batch × top_k = **262144 pair_idx 行**（极端情况）；实际 `max-num-batched-tokens=8192` 时单次 M ≈ 65536 pair_idx。

每层分摊到 64 个 routed experts，平均每 expert ≈ **1024 行**（远大于 MAX_M=32），这是 **M-大** 场景，用 DPAS grouped GEMM 才能摸到 peak。

### 1.3 为什么一定要重做

- 现有 `moe_int4_ops` kernel 在 M=32768 下启动 **~16M 个 WI**、每 WI 标量累加 K=3072，毫无 tile 复用；估算 2-5 TFLOPS vs DPAS 路径 80-150 TFLOPS，**prefill TTFT 会慢几十分钟**，完全不可用。
- ipex marlin 路径虽然快但 hang，且长期规划要去 ipex，搬迁成本高（xetla 是内部模板库）。
- 套 A prefill_v2 源码清晰、已经是 DPAS + int4，但用的是**自己的权重 layout**（与 IPEX marlin shuffle 不兼容），需要改造。

---

## 2. 目录结构与文件规划

新目录：`llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_prefill/`

```
moe_prefill/
├── moe_prefill_int4.sycl       # SYCL+ESIMD kernel 主文件 + Torch wrapper + 注册
├── moe_prefill_int4.h          # 头文件，kernel 签名/内部常量
├── layout.h                    # IPEX marlin shuffle 解包辅助 (和 moe_int4.sycl 共用，单独抽出)
└── README.md                   # 编译、使用、benchmark 说明
```

`setup.py` 追加一个 `SyclExtension` 条目：`custom_esimd_kernels_vllm.moe_int4_prefill_ops`。

**不动**现有 `moe_int4_ops`，两者作为独立 python 子模块并存，由上层 Python 调度器按 shape 选择（见 §6）。

---

## 3. 权重格式约定（必须与 IPEX marlin 兼容）

### 3.1 IPEX marlin shuffle 的 packing

来自 `ipex-gpu` 的 `GatedMLPMOE.marlin_shuffle_weight`（`linear_fusion.py:169`）：

- 原始 int4 权重 `[E, K, N]`（uint32 packed，8 个 int4 nibble 打包成一个 uint32）
- **Marlin shuffle**：把每 8 个连续的 int4 元素的 nibble 顺序从 `[0,1,2,3,4,5,6,7]` 重排为 `[0,4,1,5,2,6,3,7]`
- **Transpose**：W13 shuffle 后做 `transpose(1,2)` → `[E, K_packed, 2*I]`（K-major）；W2 同理 → `[E, I_packed, H]`

对应 llm-scaler `moe_int4.sycl:247` 的注释：
```
Weight layout (IPEX): [E, K_packed, 2*d_ff] int32, marlin shuffled nibbles
Scale layout (IPEX):  [E, K_groups, 2*d_ff] fp16
```

### 3.2 新 prefill kernel 的权重入参

**完全采纳 IPEX 布局，不做额外转换**（这是与套 A 最大的差异；套 A 用 `[E, 2*I, K/2]` + 内部 interleave，不兼容）：

- `gate_up_qweight`: `[E, K_packed, 2*I] int32`  (K_packed = H/8，marlin shuffled)
- `gate_up_scales`:  `[E, K_groups, 2*I] fp16`   (K_groups = H/128)
- `down_qweight`:    `[E, I_packed, H] int32`    (I_packed = I/8，marlin shuffled)
- `down_scales`:     `[E, I_groups, H] fp16`     (I_groups = I/128)

这样上层 Python 层调用我们的 kernel 时，**不需要任何权重重排**——`layer.W13 / layer.W2` 已经是 marlin shuffle 后的格式（ipex 做过）。

### 3.3 Marlin unshuffle（kernel 内部）

DPAS 期望 int4 解包顺序是 `[0,1,2,3,4,5,6,7]`，而 marlin shuffle 后是 `[0,4,1,5,2,6,3,7]`。

在 kernel 内部解 4bit 的地方，用一个 `constexpr int unshuffle[8] = {0,2,4,6,1,3,5,7}`（逆置换）来还原——参考 `moe_int4.sycl` 内部：

```cpp
int shift = unshuffle[b] * 4;
nibble = (packed >> shift) & 0xF;
```

---

## 4. Kernel 设计

整个 MoE prefill 管线拆成 4 个 kernel，对应套 A 的 4-op 结构，但每个 kernel 都改造为 IPEX 权重格式并适配 Qwen3.5-122B shape。

### 4.1 Kernel K0：`moe_prefill_gather_kernel`

**职责**：从 `selected_experts[M, top_k]` 构建 `expert_offsets[E]`（前缀和）和 `expert_tokens[M*top_k]`（按 expert 排序的 pair_idx 列表）。

**算法**：套 A 原样复用（`moe_prefill_v2.cpp:34-82`），Phase 0-4：
1. SLM 清零 counters + offsets
2. SLM atomic histogram
3. 单线程 exclusive prefix sum（E=64 时极快）
4. 全局 offset 写回
5. SLM atomic scatter 得 `expert_tokens`

**并行度**：单 WG，`GS=1024` WI（套 A 默认值）。对 M×top_k ≤ 1M 的场景足够。对更大的 M 考虑 `GS=2048`。

**修改点**：
- `num_experts` 改为 `num_local_experts`（TP 切分后每 rank 只看自己的 64 个 expert）
- `selected_experts` 已在上游做过 `- experts_start_id` 偏移（本 rank 视角的 local expert id）

**耗时估算**：prefill M=8192 下 <20 μs，瓶颈是 SLM atomic contention，E=64 下不是瓶颈。

### 4.2 Kernel K1：`moe_prefill_up_int4_kernel`（核心之一）

**职责**：per expert 做 `x @ gate_up^T`，融合 SiLU(gate) * up，输出 `intermediate[M*top_k, I]`。

**Tile 设计**（基于套 A，适配 IPEX K-major 权重）：

| 常量 | 值 | 来源/理由 |
|---|---|---|
| `MAX_M` | 32 | DPAS `dpas<8,8,...>` 天然 M=8 倍数；MAX_M=32 → MS=2 个 M tile，寄存器压力适中 |
| `N`     | 16 | 单 WI 沿 N 覆盖 16；W13 总 N=2*I=512（TP 切后），一个 WG 启 `I/N=16` 个 WI |
| `BS`    | 128 | = group_size；一个 group 内 scale 相同，K 方向 blocking 单位 |
| `MS = MAX_M/16` | 2 | M 方向 tile 数量（DPAS 一次吃 M=8×2 rows） |
| `NS = N/8`      | 2 | N 方向 tile 数量 |
| `ACC_SZ = MS*NS*128` | 512 | 累加器 register footprint（per WI） |

**DPAS 调度**：
- `xmx_ns::dpas<8, 8, fp16, fp16, fp16, fp16>(acc, b_tile, a_tile)`
  - 8 (repeat_count) × 8 (sub_group_size) → 每次 DPAS 做 8×8×16 fp16×fp16→fp16 的 systolic matmul
  - `b_tile`: 来自输入 x 的 VNNI 排列（K 维 gather）
  - `a_tile`: 来自 int4 weight 解包 + scale
  - `acc`: 寄存器累加器

**K 循环**：`for blk in K/BS: for k_base in 0..BS/2 step 8`（BS=128 → 外循环 24 次，内循环 8 次 → 每 WI 192 次 DPAS 对）。H=3072 时每 WI 算 **192 次 gate DPAS + 192 次 up DPAS = 384 次 DPAS**，覆盖整条 K。

**Weight load**：
- 原套 A 用 `lsc_load_2d<uint8_t, 8, 8, 1>(w_ptr, …)` 从 `[E, 2*I, K/2]` N-major 布局取 `[N_tile=8, K_bytes=8]`
- **本设计改**：权重是 IPEX K-major `[E, K_packed, 2*I]`（int32）
  - 沿 K 维 stride 大（2*I 个 int32 = 2048 bytes/行）
  - 沿 N 维连续（int32 连续存放，8 个 nibble 一个 uint32）
  - 用 **`lsc_load_2d<uint32_t, 8, ..., BlockHeight=1, BlockWidth=32>`** 加载 `[K_rows=8 (=1 int32 per N-group × 8 rows of K_packed), N_cols=32]`：一次加载 32 个连续的 N 方向列上 8 个 K_packed 行 = 256 个 uint32 = 2048 个 int4
  - 解包时需按 marlin unshuffle 序 `[0,2,4,6,1,3,5,7]` 展开 nibble

**Input gather**（VNNI b_tile）：
- x 是 `[M, H] fp16` row-major
- 每个 WI 处理 MAX_M=32 行（pair_idx → token_idx 映射后），沿 K 维取 VNNI
- 套 A 原写法 `lsc_gather<uint32_t, 8, ...>` 从 `in_off + k_off` 取 32 个 uint32（= 64 个 fp16）构成 b_tile
- **保持不变**（与权重 layout 无关）

**SiLU-and-mul**：
```cpp
simd<fp16, ACC_SZ> sv =
    (g_acc / (fp16(1) + sycl::ext::intel::esimd::exp(-g_acc))) * u_acc;
```
fused 在寄存器完成，不落 HBM。

**Scatter output**：
- `intermediate[pair_idx, n_start..n_start+N]` 写回，pair_idx 由 `expert_tokens` 决定
- scatter 用 `simd` + `scatter<IT, 16>`（套 A 原写法）

**处理 pair_idx 不足 MAX_M**：
```cpp
simd<uint32_t, MAX_M> sorted_idxs = min(
    simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)m_base,
    simd<uint32_t, MAX_M>((uint32_t)(t1 - 1)));  // clamp 到 expert 内最后一行
```
写回时被 clamp 的 lane 会**重复写**到同一个 pair_idx 位置（幂等覆盖，结果正确）。

**并行度**：`parallel_for(range<2>(num_experts, intermediate_size / N))`
- Qwen3.5-122B TP=4：`(64, 256/16=16)` = 1024 WI 大 grid
- 每 WI 处理 1 个 expert 的 N=16 列、所有 M rows（通过 MAX_M=32 内循环）
- **对 expert rows==0 安全**：`if (t0 == t1) return;` early exit，但**仍然提交 kernel**（event 被 signal），rank 间对称——**这是绕开 ipex hang 的关键**。

**性能估算**（H=3072, N_total=512, 平均每 expert M ≈ 1024 rows）：
- 每个 expert 需要 `M/MAX_M × N/N_tile × K/BS × BS/(2*8)` DPAS 次数 = `32 × 32 × 24 × 8 = 196608` DPAS
- 单 DPAS 做 `8*8*16 = 1024` MAC → 每 expert 200 MMAC
- 64 expert / layer × 2 projections = 128 gate/up 合一 → prefill one layer ~13 TMAC
- BMG peak 150 TFLOPS fp16 DPAS → 理论 ≈ 0.09s / layer，实测预期 1.5-3× 慢 = **0.15-0.25s/layer**
- 48 layer × 0.2s ≈ **10s prefill**（vs decode kernel 跑 prefill 估 ~5 分钟）

### 4.3 Kernel K2：`moe_prefill_down_int4_kernel`

**职责**：per expert 做 `intermediate @ down^T`，输出 `expert_output[M*top_k, H]`。

**设计**：与 K1 几乎对称，差异：
- 输入是 `intermediate[M*top_k, I]`（pair_idx 直接做行索引，不再二次映射）
- 权重是 `down_qweight [E, I_packed=I/8, H] int32`（IPEX K-major，K 是 I，N 是 H）
- 输出 `expert_output[pair_idx, :H]`
- `N=32`（套 A 原选择，因为 H=3072 比 2I=512 大得多，需要更多 N tile 才能让 WI 数目饱和）

**Tile 参数**：`MAX_M=32, N=32, BS=128`；`MS=2, NS=4, ACC_SZ=1024`

**并行度**：`parallel_for(range<2>(64, 3072/32=96))` = 6144 WI

**注意**：`expert_output` 是 `[M*top_k, H]` 行存每个 expert 的输出，**pair_idx 决定写哪一行**，同一个 token 的 top_k 份输出会分散在 top_k 个不同的 pair_idx 行里，**不重叠**，不需要原子写。

### 4.4 Kernel K3：`moe_prefill_finalize_kernel`（取代套 A 的 accumulate）

**职责**：用 `routing_weights[M, top_k]` 对 `expert_output[M*top_k, H]` 做加权求和，**同时融合 shared expert**，输出 `final_output[M, H]`。

**套 A 的 accumulate kernel**（`moe_prefill_v2.cpp` 文末）只做了加权 sum，没处理 shared expert。Qwen3.5-122B 有 1 个 shared expert，必须补上。

**两种 shared expert 路径**：

- **选项 A（推荐，简单）**：shared expert **在上层 Python 里单独用 SYCL int4 linear kernel 算**（复用 llm-scaler 已有的 `custom_esimd_kernels_gemm` 或 `int4_gemv.cpp`），结果与 routed 输出相加。
- **选项 B（合并）**：在 K3 finalize kernel 里 inline shared expert 的 matmul。参考 `moe_int4.sycl:914` `moe_down_finalize_int4_kernel`（decode 版的写法），搬过来适配 prefill shape。

**初版选 A**：finalize kernel 纯粹做 weighted sum（像套 A），shared expert 单独调一次现有 int4 GEMM。这样新 kernel 的复杂度最小，也让 shared expert 的调用路径可替换。

**Finalize kernel 伪码**：
```cpp
for tok in parallel(M):
    for hoff in 0..H step 64:
        acc = 0
        for s in 0..top_k:
            w = routing_weights[tok, s]
            pi = tok * top_k + s
            acc += w * block_load<fp16, 64>(expert_output + pi*H + hoff)
        block_store<fp16, 64>(final_output + tok*H + hoff, acc)
```

这里**不引入 shared expert**，由调用方在外面加一次 `final_output += shared_expert(x)`。

### 4.5 Kernel 汇总表

| Kernel | M 并行度 | N 并行度 | 主指令 | SLM 用量 | 估算单层时间 |
|---|---|---|---|---|---|
| K0 gather | 全部 M×top_k 在一个 WG | 1 WG | SLM atomic histogram + prefix sum | 2×E×4 = 512B | <20 μs |
| K1 up | expert × MAX_M tile | I/N = 16 | `xmx::dpas<8,8,fp16>` | 0（纯寄存器） | ~120 ms/layer |
| K2 down | expert × MAX_M tile | H/N = 96 | `xmx::dpas<8,8,fp16>` | 0 | ~80 ms/layer |
| K3 finalize | 1 WI per token | 1 WI per (token, 64-block) | block_load/store + scalar FMA | 0 | <2 ms |

每层 ≈ 200ms, 48 层 ≈ **10s prefill**。对比 decode kernel 估算 5 分钟，**快 ~30x**。

---

## 5. Torch wrapper 与 op 注册

### 5.1 API 设计

参考 llm-scaler 现有 `moe_forward_full_int4` 的命名风格：

```cpp
// Low-level (每个 kernel 一个 op，便于单测和 benchmark)
moe_prefill_gather_v2(
    Tensor selected_experts,   // [M, top_k] int32
    int64_t num_local_experts
) -> (Tensor expert_offsets, Tensor expert_tokens);

moe_prefill_up_int4(
    Tensor x,                  // [M, H] fp16
    Tensor gate_up_qweight,    // [E_local, K_packed, 2*I] int32 (IPEX K-major + marlin shuffled)
    Tensor gate_up_scales,     // [E_local, K_groups, 2*I] fp16
    Tensor expert_offsets,     // [E_local] int32
    Tensor expert_tokens,      // [M*top_k] int32
    int64_t top_k
) -> Tensor;                   // [M*top_k, I] fp16

moe_prefill_down_int4(
    Tensor intermediate,       // [M*top_k, I] fp16
    Tensor down_qweight,       // [E_local, I_packed, H] int32
    Tensor down_scales,        // [E_local, I_groups, H] fp16
    Tensor expert_offsets,
    Tensor expert_tokens
) -> Tensor;                   // [M*top_k, H] fp16

moe_prefill_finalize(
    Tensor expert_output,      // [M*top_k, H] fp16
    Tensor routing_weights,    // [M, top_k] fp16
    int64_t top_k
) -> Tensor;                   // [M, H] fp16

// High-level (组合 4 个，给 vllm 接入)
moe_prefill_forward_int4(
    Tensor x,                  // [M, H] fp16
    Tensor selected_experts,   // [M, top_k] int32
    Tensor routing_weights,    // [M, top_k] fp16
    Tensor gate_up_qweight,
    Tensor gate_up_scales,
    Tensor down_qweight,
    Tensor down_scales,
    int64_t top_k,
    int64_t num_local_experts
) -> Tensor;                   // [M, H] fp16
```

**设计取舍**：
- 4 个 low-level op 单独暴露，便于 pytest 分模块对拍 ipex 输出
- high-level op 做 one-call 方便集成；内部直接调 4 个 kernel launch，不走 Python 多次 dispatch（减少 host overhead）
- **router/topk 不放在本包内**——沿用 llm-scaler 现有的 `moe_router_forward_int4` 或 ipex `topk_softmax`，保持 topk 路径和 decode 一致

### 5.2 注册方式

和现有 `moe_int4_ops` 一样：

```cpp
TORCH_LIBRARY_FRAGMENT(moe_int4_prefill_ops, m) {
    m.def("moe_prefill_gather_v2(Tensor selected_experts, int num_local_experts) -> (Tensor, Tensor)");
    m.def("moe_prefill_up_int4(Tensor x, Tensor qw, Tensor sc, Tensor off, Tensor tok, int top_k) -> Tensor");
    m.def("moe_prefill_down_int4(...) -> Tensor");
    m.def("moe_prefill_finalize(...) -> Tensor");
    m.def("moe_prefill_forward_int4(...) -> Tensor");
}
TORCH_LIBRARY_IMPL(moe_int4_prefill_ops, XPU, m) { ... }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { ... }
```

**namespace 用 `moe_int4_prefill_ops`**，和现有 `moe_int4_ops` 不冲突。

---

## 6. 上层 Python 集成（vllm-xpu 侧）

### 6.1 新类 `ESIMDGatedMLPMOE`

在 `llm-scaler-vllm-xpu/vllm/model_executor/layers/quantization/` 下加 `esimd_moe.py`：

```python
class ESIMDGatedMLPMOE(nn.Module):
    """Drop-in replacement for ipex.llm.modules.GatedMLPMOE, sym_int4 only."""
    def __init__(self, W13, W2, w1_scale_inv, w2_scale_inv, is_int4=True, ...):
        # 权重已经是 ipex 的 marlin shuffle + transpose 后格式
        self.W13, self.W2 = W13, W2
        self.w13_scale, self.w2_scale = w1_scale_inv, w2_scale_inv
        self.num_local_experts = W13.shape[0]

    def forward(self, hidden_states, use_grouped_topk, top_k,
                router_logits, renormalize, ...):
        # 1. topk: 复用 ipex 或 llm-scaler 现有 topk kernel
        routing_weights, selected_experts = self._route(
            router_logits, top_k, use_grouped_topk, ...)

        M, H = hidden_states.shape
        # 2. prefill / decode 分发
        if M > 64:  # prefill 阈值（见 §6.2）
            return torch.ops.moe_int4_prefill_ops.moe_prefill_forward_int4(
                hidden_states, selected_experts, routing_weights,
                self.W13, self.w13_scale, self.W2, self.w2_scale,
                top_k, self.num_local_experts)
        else:
            # 走现有 decode kernel
            return torch.ops.moe_int4_ops.moe_forward_full_int4(...)
```

### 6.2 prefill/decode 分发阈值

**阈值 M=64**：
- M ≤ 4：decode kernel Option A（weight-reuse）最优
- M = 5..64：decode kernel Option C 尚可，prefill kernel MAX_M=32 下 tile 利用率低（填不满）
- M > 64：prefill kernel 的 DPAS 吞吐明显领先

后续可做 per-config auto-tune，但初版先用硬阈值。

### 6.3 接入点

修改 `llm-scaler-vllm-xpu/vllm/model_executor/layers/quantization/sym_int4.py:394`：

```python
if os.environ.get("USE_ESIMD_MOE", "0") == "1":
    from .esimd_moe import ESIMDGatedMLPMOE
    layer.ipex_fusion = ESIMDGatedMLPMOE(
        layer.w13_weight, layer.w2_weight,
        w1_scale_inv=layer.w13_scales, w2_scale_inv=layer.w2_scales,
        is_int4=True)
else:
    layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
        layer.w13_weight, layer.w2_weight,
        w1_scale_inv=layer.w13_scales, w2_scale_inv=layer.w2_scales,
        is_int4=True)
```

`USE_ESIMD_MOE=1 bash 122b.sh` 一键切换。

---

## 7. 实施路线（7 个阶段）

| Phase | 目标 | 工作量 | 验证方式 |
|---|---|---|---|
| **P0** | 建目录、写空壳 `.sycl` + `.h`、改 setup.py、`build.sh` 跑通**空 kernel** 编译 | 0.5 day | `python -c "from custom_esimd_kernels_vllm import moe_int4_prefill_ops"` 成功 |
| **P1** | 搬 K0 (gather)。直接 copy 套 A `moe_prefill_gather_forward_kernel` | 0.5 day | pytest 用 Python 纯 numpy 实现对拍，num_experts=64, M=8192 全对 |
| **P2** | 写 K1 (up int4 IPEX layout)。参考套 A `moe_prefill_up_forward_kernel`，改 weight load + marlin unshuffle | 2 days | pytest：fix seed 生成权重，对比 ipex `group_mm_int4_out_marlin` + silu_and_mul 的输出，max_diff < 1e-3 |
| **P3** | 写 K2 (down int4)。复用 K1 的 DPAS 模板，改 N=32 | 1 day | pytest 对比 ipex down 输出 |
| **P4** | 写 K3 (finalize)，wrapper `moe_prefill_forward_int4` 组合 4 个 kernel | 0.5 day | pytest 端到端对拍，所有 shape 在 `test_moe_int4_kernel.py` 的 `122B-A10B-TP4` 配置下通过 |
| **P5** | 接入 vllm-xpu，`USE_ESIMD_MOE=1` 跑 `122b.sh`，验证 **hang 消失** | 0.5 day | TP=4 `max-num-batched-tokens=8192` 完整跑完 32768-2048 benchmark |
| **P6** | 性能 benchmark，与 ipex marlin 对拍 TTFT 和 TFLOPS | 0.5 day | `benchmark_prefill.py` 适配 Qwen3.5-122B shape，跑单层 MoE；估算全模型 TTFT |

**总周期 ~6 个工作日**。最大风险在 P2（weight layout 适配）。

### 7.1 风险点与缓解

| 风险 | 缓解 |
|---|---|
| IPEX marlin shuffle 的 unshuffle 序写错，精度挂 | P2 第一步：用固定 uint32 权重 + numpy ref dequant，对拍单个 expert M=1 的输出，先做到 bit-exact 再堆 DPAS |
| `lsc_load_2d` 在 K-major 布局下 stride 计算错导致越界 | 用 `ZE_DEBUG=-1` + `ONEAPI_DEVICE_SELECTOR=level_zero:0` 跑单 WI 单 expert 的标量版，先对；然后把标量版逐步 DPAS 化 |
| DPAS 累加器寄存器溢出（MS=2, NS=4 → ACC_SZ=1024 fp16 = 2KB/WI） | BMG GRF=128×32B=4KB/thread 足够；若报 spill 先把 K2 的 N 从 32 降到 16 重编译 |
| prefill kernel 对 expert rows < MAX_M 的 tail 处理不完备 | 用 clamp 到 `t1-1` 的 sorted_idxs trick（套 A 原设计，已验证正确） |
| hang 依旧存在 | 说明根因不在 ipex MoE kernel，而在 allreduce / 其他路径；这时依然有 prefill 加速收益，同时缩小了 hang 的嫌疑范围 |

---

## 8. 测试设计

### 8.1 单 kernel 单元测试

`tests/test_moe_prefill_int4_kernel.py`，复用 `test_moe_int4_kernel.py` 的 INT4 quant/dequant helper + `ipex_transform_expert_weights`：

```python
CONFIGS = {
    "tiny":        {"M": 64,    "H": 256,  "I": 64,   "E": 4,   "top_k": 2},
    "122B-TP4":    {"M": 8192,  "H": 3072, "I": 256,  "E": 64,  "top_k": 8},
    "122B-TP4-max": {"M": 32768, "H": 3072, "I": 256, "E": 64, "top_k": 8},
}

def test_moe_prefill_up_int4(config):
    x, qw, sc = make_inputs(config)
    selected = random_topk(config)
    offsets, tokens = ops.moe_prefill_gather_v2(selected, config["E"])
    out = ops.moe_prefill_up_int4(x, qw, sc, offsets, tokens, config["top_k"])
    ref = ref_up_silu(x, dequant(qw, sc), selected, ...)
    assert (out - ref).abs().max() < 1e-2
```

### 8.2 E2E 对拍 ipex

对比 `ipex.llm.modules.GatedMLPMOE`(int4) 的输出：
```python
ref = ipex_moe.forward(x, False, top_k, router_logits, True)
ours = esimd_moe.forward(x, False, top_k, router_logits, True)
assert (ref - ours).abs().max() < 0.05  # 允许小精度偏差（DPAS fp16 中间累加）
```

### 8.3 Benchmark

`tests/bench_moe_prefill_int4.py`：
- warmup 10 iter + timed 100 iter，`torch.xpu.synchronize()` 包围
- 报告 up / down / finalize / total 四段 μs + TFLOPS + 带宽
- 分别跑 `M ∈ {64, 256, 1024, 8192, 32768}`，输出一张性能表
- 在同一脚本里顺便跑 ipex 对照

---

## 9. Open questions（需确认后才能动手 P2）

1. **sym_int4 scale dtype 确认**：代码里既有 fp16 也有 bf16 签名；Qwen3.5-122B 实际跑是哪个？需要看 `sym_int4.py` 的 `layer.w13_scales.dtype`。
2. **ipex marlin shuffle 的 transpose 维度**：`W13.shape = [E, 2*I, H]` 原始 → `[E, H, 2*I]` transpose 后。我们 kernel 按 `[E, H/8, 2*I] int32` 读。需要看 vllm-xpu 的 `apply_weights` 实际存的是 packed 形式还是原始。
3. **shared expert 权重是否同样 marlin shuffled**：`moe_int4.sycl:1228` 注释说 shared expert 用 "IPEX column-major repack, no marlin shuffle"——如果 shared expert 也走我们的 prefill kernel，需要分层处理。**本设计 §4.4 选项 A 把 shared expert 外置**避开这个问题。

这三个问题在 P0 的 empty kernel 编译通过后、动手写 K1 前用一个 20 行 Python 脚本 dump 一遍真实权重 shape/dtype 就能回答。

---

## 附录 A：DPAS 参数速查

`xmx::dpas<SD, RC, T_acc, T_a, T_b, T_dst>(acc, b, a)`
- SD=8 (Systolic Depth)：K 维度每次吃 8（fp16 下 = 16 个 fp16 元素）
- RC=8 (Repeat Count)：M 维度一次算 8 行
- 单次 DPAS 完成 `[8, 16] fp16 × [16, 8] fp16 → [8, 8] fp16 acc`（VNNI packed 下）
- a_tile 存 weight（K×N），b_tile 存 input VNNI（M×K），acc 存 M×N accumulator

## 附录 B：参考代码精确坐标（便于复制/对照）

| 功能 | 文件 | 行号 |
|---|---|---|
| gather kernel 完整实现 | `frameworks.ai.client-ai.esimd-kernels/.../moe_prefill_v2.cpp` | 25-82 |
| up DPAS kernel | 同上 | 87-227 |
| down DPAS kernel | 同上 | 230-? |
| accumulate kernel | 同上 | 文末 |
| IPEX marlin shuffle 权重格式注释 | `llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl` | 239-246 |
| IPEX marlin_shuffle_weight Python | `frameworks.ai.pytorch.ipex-gpu/intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py` | 169-208 |
| 现有 moe_forward_full_int4 host wrapper | `llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl` | 1195-1260 |
| vllm-xpu 调用 ipex.GatedMLPMOE (sym_int4) | `llm-scaler-vllm-xpu/vllm/model_executor/layers/quantization/sym_int4.py` | 394 |
| hang 的启动脚本 | `llm/models/test/122b.sh` | - |

