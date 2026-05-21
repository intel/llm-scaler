# ESIMD Prefill FMHA 开发记录

## 项目目标
用 ESIMD 实现 prefill Flash Attention kernel，替换 IPEX 的 xetla FMHA，最终超越其性能。

## 当前代码位置
- Branch: `feature/prefill-fmha-esimd` on `/llm/shaojun/code/llm-scaler`
- Kernel: `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/prefill_fmha.h`
- Torch binding: `csrc/xpu/esimd_kernel_fmha.sycl` + `csrc/xpu/torch_extension_fmha.cc`
- UT: `tests/test_prefill_fmha.py` (8 cases, 全部通过)
- Python wrapper: `python/custom_esimd_kernels_vllm/ops.py` (esimd_prefill_fmha)

## 编译方式

### AOT (当前 setup.py 配置为 JIT):
```bash
# 需要改 setup.py 的 sycl flags: 去掉 -fsycl -fsycl-targets=spir64，加 -doubleGRF
cd /llm/shaojun/code/llm-scaler/vllm/custom-esimd-kernels-vllm
rm -rf dist/ build/
pip uninstall -y custom-esimd-kernels-vllm
TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python3 setup.py bdist_wheel
pip install dist/*.whl --no-deps
ZE_AFFINITY_MASK=4 python tests/test_prefill_fmha.py 1 2 3 4 5 6 7 8
```

### JIT (当前配置):
```bash
# setup.py 的 FMHA sycl flags: "-fsycl", "-fsycl-targets=spir64"
# 同上编译命令
# 注意: esimd_build_extention.py 已修改，检测 spir64 时跳过 AOT dlink flags
```

### 已知编译问题:
- `moe_int4_prefill_ops` module AOT 会 segfault → 已在 setup.py 中注释掉
- `esimd_build_extention.py` line 813 附近: 加了 JIT 检测逻辑跳过 `-Xs -device` flags

## 性能演进

| 版本 | kBr | 编译 | 特性 | 性能 | 正确性 |
|------|-----|------|------|------|--------|
| Phase 1 | 1 | AOT | 标量 dot, 逐 token | 20x | PASS |
| Phase 2 | 4 | AOT+doubleGRF | Q 预加载, V 共享 | 14x | PASS |
| Phase 2 | 4 | AOT+doubleGRF | split-KV | 15x | PASS |
| Phase 2 | 4 | AOT+doubleGRF | split-KV + DPAS + batch softmax | 15x | PASS |
| kBr=8 DPAS | 8 | JIT | DPAS score + SIMD reduce (buggy) | 5x | NaN |
| kBr=8 DPAS | 8 | JIT | DPAS score + 标量 softmax | 93x | PASS |
| IPEX baseline | — | JIT | xetla FMHA | 1x (1ms) | — |

## 关键发现：JIT 编译器 Bug

### Bug 1: `reduce<float>(simd<float,16>, maximum<>())` 返回错误值

**验证**:
```cpp
simd<float, 16> test_vec = 0.0f;
test_vec[0] = 1.0f; test_vec[1] = 2.0f; test_vec[8] = -1000.0f;
float test_max = reduce<float>(test_vec, maximum<>());
// 期望: 2.0, 实际返回: 0.0
```

**影响**: softmax 的 chunk_max 计算错误 → exp(score - wrong_max) 溢出 → NaN
**绕过**: 用标量 for 循环找 max

### Bug 2: pointer array 间接访问 simd 产生 NaN

**验证**:
```cpp
simd<float,16>* out_all[8] = {out_r0, out_r1, ...};
out_all[r][i] * inv_sum;  // 产生 NaN
// 改为手动 WR macro 展开每行 → 正确
```

**绕过**: 不用 pointer array，手动展开每行的读写

### 无 Bug 的操作:
- `reduce<float>(simd, std::plus<>())` — 求和正确
- `__ESIMD_NS::exp(simd)` — SIMD exp 正确（需要 merge 处理极端负值）
- `simd[c] = value` (c 编译期常量) — 正确
- `simd.select<1,1>(c)[0]` (c 编译期常量) — 正确但大量使用会让 JIT 生成慢代码
- DPAS — 正确

## 性能瓶颈分析

### AOT kBr=4 版本 (14-18x):
- Stall 44% (和 IPEX 一样)
- XMX = 0% (没用 DPAS)
- 瓶颈: 每个 KV token 的标量 dot product (reduce<float>(q*k, plus))

### AOT kBr=4 + split-KV + DPAS (15x):
- Split-KV 没有帮助: GPU EU 已被大量 WG 饱和
- DPAS 只加速 score 计算 (~5% of time), V accumulate 占 65%
- V accumulate 瓶颈: 每 token 16 次 block_load<fp16,16> 的 L3 延迟

### JIT kBr=8 (93x 正确版 / 5x NaN版):
- 93x: 大量 select<1,1>(c)[0] 让 kernel body 巨大 → instruction cache thrashing
- 5x (NaN): 纯 SIMD 操作 (reduce+exp) 让 kernel 紧凑 → JIT 高度优化
- 差距根因: kernel body 指令数量 (紧凑 SIMD vs 展开的标量)

## 核心矛盾

要正确 → 不能用 `reduce<float>(simd, maximum<>())` → 必须用标量 max → 需要 `select<1,1>(c)[0]` → kernel body 膨胀 → JIT 生成慢代码 → 93x

要快 → 必须紧凑 SIMD 操作 → 需要 `reduce` → 有 bug → NaN

## 下一步探索方向

### 方向 1: 手动 tree reduction (替代 reduce intrinsic)
```cpp
simd<float, 8> h = max(sr.select<8,1>(0), sr.select<8,1>(8));
simd<float, 4> q = max(h.select<4,1>(0), h.select<4,1>(4));
simd<float, 2> d = max(q.select<2,1>(0), q.select<2,1>(2));
float m = max(d.select<1,1>(0)[0], d.select<1,1>(1)[0]);
```
- 不用 `reduce` intrinsic
- 用 SIMD `max` 操作 (不是标量循环)
- 只需要最后一步提取标量 (2 次 select, 不是 16 次)
- **未测试**, 可能绕过 reduce bug 且保持紧凑

### 方向 2: AOT kBr=4 + head-dim 并行 (PR#368 思路)
- 多个线程分摊 V 的 256 dim load
- 每个线程只处理 16 dim → V load 只需 1 次 block_load
- Q×K dot product 需要 SLM 跨线程 reduce
- 最接近 PR#368 Phase2 的设计

### 方向 3: AOT + DPAS for P×V
- 用 DPAS 做 P[4x16] × V[16x16] → output partial
- V 需要 VNNI pack (从 paged cache 加载后转置)
- 理论上替代 256 次 block_load → 16 次 DPAS
- 复杂度高

## 理论性能分析

2048 tokens, 12 heads, head_dim=256:
- 总读取量: ~48GB (大部分在 L3 cache, hit rate 99%)
- 总计算量: ~48 GFLOPs
- BMG L3 BW: ~2-4 TB/s → memory bound 理论下限: ~24us
- BMG XMX: ~100 TFLOPS → compute bound: ~0.5ms
- BMG ALU FP32: ~10 TFLOPS → ALU bound: ~5ms
- IPEX 实际: 1ms (介于 XMX 和 ALU bound)

## 文件修改清单

### 新增:
- `csrc/xpu/esimd_kernels/prefill_fmha.h` — kernel 实现
- `csrc/xpu/esimd_kernel_fmha.sycl` — torch extension 接口
- `csrc/xpu/torch_extension_fmha.cc` — op 注册 + PyInit
- `tests/test_prefill_fmha.py` — 8 个 UT case

### 修改:
- `setup.py` — 新增 FMHA module, 注释 moe_prefill
- `esimd_build_extention.py` — JIT 检测逻辑 (跳过 AOT dlink flags)
- `include/kernel_ops.h` — esimd_prefill_fmha 声明
- `python/custom_esimd_kernels_vllm/ops.py` — Python wrapper
- `python/custom_esimd_kernels_vllm/__init__.py` — export

### 上下文文档:
- `/llm/shaojun/TTFT_OPTIMIZATION_CONTEXT.md` — 完整项目上下文
- `/llm/shaojun/TTFT_Profiling_Report.md` — 给老板的报告

## UT 使用方法

```bash
ZE_AFFINITY_MASK=4 python tests/test_prefill_fmha.py 1 2 3 4 5 6 7 8  # 全部
ZE_AFFINITY_MASK=4 python tests/test_prefill_fmha.py 7 8              # 只跑 realistic + perf
```

## split-KV 实现 (在之前的 commit 中)

- Sub-kernel: 每个处理 PARTITION_SIZE(512) tokens 的 KV range
- Reduce kernel: log-sum-exp merge partial results
- 中间 buffer: partial_out(float32) + partial_max + partial_sum
- 用 sycl::malloc_device/free 分配
- 对 2048 tokens 没有帮助 (EU 已饱和), 对长输入 (64K+) 可能有用
