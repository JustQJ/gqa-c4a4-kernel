# C4A4 MXFP4 PA_BBND 性能优化开发文档

本文档分两部分：

- **第一部分（A1–A8）是当前生效实现的完整开发说明**：设计思想、优化路径、具体改动、优势、限制、使用注意。以代码现状与 2026-11-25 的完整验证为依据。
- **第二部分（附录 B，原第 1–12 章）保留设计推演与历史实验记录**。其中若干结论已被后续实测推翻（尤其是 B3「Config 6 不派发」、B7「合并方案净退化」、B7.9「Q scale 半 fractal 写入」），**阅读时一律以第一部分为准**。保留它们是为了让后来者不重复已经证伪的路径。

配套文档：[验证报告](validation-report-20261125.md) · [测试报告](test-report.md) · [官方优化记录](../attention/quant_flash_attn/docs/kernel-optimized.md)

---

## A1. 概述

| 项 | 内容 |
| --- | --- |
| 优化对象 | `attention/quant_flash_attn` arch35 MXFP4（C4A4）实现 |
| 目标平台 | Ascend 950PR（28 cube core / 56 vector core，L2 128 MB） |
| 核心场景 | **PA_BBND 布局下的 ND decode**（`layout_q_descale=N2TGD`、`N_kv=2`、`D=Dv=256`） |
| 度量 | 同用例、同输入、同设备、同 Graph 计时口径下的 C8/C4 耗时比 |
| 目标 | 1.4～2.0 |
| 当前达成 | PA_BBND decode 几何均值 **1.3918**（基线 1.0875），整体 C4 耗时 **-8.02%** |

**一句话结论**：PA_BBND 的 decode 瓶颈不是 DRAM 带宽不足，而是**单 head 行宽 128 B 填不满硬件 256 B 的取回粒度**，一半取回的字节被丢弃。把同一 page 的两个 KV head 合进一个 task，让每次 256 B 取回全部有用，DRAM 流量减半，长 KV decode 提速 35%～43%。

**适用**：PA_BBND + N2TGD + `N_kv=2` + `D=Dv=256` + `N_q/N_kv` 为 ≥4 的 2 的幂。
**不适用**：DN prefill、PA_BNBD、PA_NZ、`N_kv≠2`、`D=128`、gSize 非 2 的幂——这些一律走基线路径，行为与优化前一致。

---

## A2. 设计思想

### A2.1 问题的本质：事务粒度浪费，不是带宽不足

基线 `bench16k_decode_1` 的 `msprof op / Default`：

| 指标 | 值 | 含义 |
| --- | ---: | --- |
| `aic_mte2_ratio` | 0.954 | GM→L1 搬运几乎占满 |
| `GM_to_L1_bw_usage_rate` | 30.9% | 但带宽只用了三成 |
| `aic_cube_ratio` | 0.066 | Cube 几乎空转 |
| `aic_cube_wait_ratio` | 0.939 | Cube 全程在等数据 |

「搬运占满 + 带宽只用三成」同时成立，说明受限于**事务效率**而非带宽。

PA_BBND 的 GM 布局是 `[page][token][head][D/2]`，同一 token 的各 KV head 交织。C4 只搬本 head 时，ND2NZ 的每行是「128 B 有效 / 256 B 源跨步」，`srcDValue != dValue`，无法合并成长 burst。硬件按 256 B 整行取回，另一半直接丢弃。

**决定性证据**：C4 与 C8 的 `read_main_memory_datas` 几乎完全相同（decode_1 5324.8 vs 5328.1 KB，decode_2 14255.1 vs 14254.5 KB）。C8 的行宽是 256 B、正好填满粒度，C4 行宽 128 B 却读了同样多的 DRAM 字节——**浪费的那一半被实测坐实**。

### A2.2 为什么「多读再挑」必然失败

直觉方案是「一次读整页全 head，消费侧按 head 偏移挑」。两个变体都实测过：

| 变体 | 做法 | decode 几何均值 | 结论 |
| --- | --- | ---: | --- |
| 基线 | 单 head 跨步读 | 1.1102 | — |
| A | payload 整页全 head，scale 仍单 head | **1.0889** | 退化 |
| B | payload + scale 都整页全 head | 更差 | 退化 |

原因：行连续带来的事务效率提升，被「同一 page 的两个 head 各有一个 task、各读一遍整页」造成的**访问次数翻倍**完全抵消。

**由此确立第一原则：基线的 GM 字节数已经最小，任何「多读再挑」的改写都不可能净收益。必须让同一 page 的两个 KV head 只被读一次。**

### A2.3 块对角合并：让每次取回全部有用

一个 task 覆盖**一个 KV page 的全部 head**：

- **M 轴** = `seqQ × N_q`（展开全部 Q head，不再按 KV head 拆 task）；
- **N 轴** = `n2Size × kvTokenBlock` = 2 × 128 = 256 列，列布局 `[head0 的 128 token; head1 的 128 token]`；
- softmax 阶段，「行所属 KV head ≠ 列所属 KV head」的整块**不参与计算**；
- 非对角块对结果无贡献，因此 PV 用同样堆叠的 V 得到的每一行与单 head 计算完全一致。

关键在于**「跳过」而不是「置 -inf 后照算」**：跳过的行 P 保持预置 0、指数保持 INVALID，与置 -inf 等价（该组 Pscale 为 0），但省掉一半向量计算。这是本方案能成立的第二个支点。

### A2.4 三个必须同时成立的条件

| 条件 | 含义 | 违反的后果 |
| --- | --- | --- |
| **字节守恒** | 每 page 的 GM payload+scale 字节与基线相同 | 多读即退化（A2.2 已证伪） |
| **片上开销不翻倍** | V2/Pscale 总行数、VF 行数不能因为 M/N 抬高而翻倍 | 事务收益被片上开销吃掉（B7.14 的历史失败） |
| **流水约束** | `GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N` | 单核 task 数 ≥ `PRELOAD_N` 时 AIC/AIV 互等死锁 |

第二条决定了 `kvTokenBlock` 必须等于基线每 head 的 token 块（128），从而 **task 数减半、V2/Pscale 总行数不变**。第一条由硬件粒度天然满足。第三条是本轮定位到的真实缺陷之一。

### A2.5 为什么最终形状是 M64/N256/D256

| 候选形状 | 结论 | 理由 |
| --- | --- | --- |
| M128/N128（`kvTokenBlock=64`） | 放弃 | task 数与基线相同但每 task Q 行数翻倍 → V2/Pscale 变成 2 倍工，decode_4 退化 |
| M128/N256 | 放弃 | UB 需求 273408 B 超出 248 KiB 预算，静态断言不通过 |
| **M64/N256/D256** | **采用** | `kvTokenBlock=128` 与基线一致；L1/L0 预算完全沿用，无需扩容 |

一个关键巧合让 M64/N256 恰好合身：`mmKVSize = 256×256/2 = 32768 B`、`mmKVScaleSize = 256×256/32 = 2048 B`，**正好等于一个 BBND 整页双 head 的 payload 与 scale 字节数**。

---

## A3. 优化路径

PA_BBND 今天的性能是**七层优化叠加**的结果，块对角合并只是最上面一层。按时间顺序：

| # | 阶段 | 提交 | 机制 | 受益范围 | 实测 |
| --- | --- | --- | --- | --- | --- |
| 1 | 普通 Div + 倒数复用 | `98e3d8d1a` | 关闭输出 Div 高精度修正；DN/ND 一次除法后各通道乘法 | 全布局 | 12 例几何均值 1.0611× |
| 2 | ExpSub + 乘加融合 | `beb2b64d7` | 37 处 Sub→Exp 用 `ExpSub`，11 处乘加用 `MulDstAdd`/`MulAddDst` | 全布局 | 累计 1.0663× |
| 3 | D256 QK Chunk128 | `00e1713d7` | DN/ND 均用 QK Chunk128，QK L0 单槽，输出暂存复用已消费的 PV 缓冲 | 全布局 D256 | 相对 Base 累计 **1.1910×**（-16.04%） |
| 4 | V 补列紧凑化 | `bc68af786` | `InitPaddingL0` 只搬 16 通道补列，L1 常数区 17–19 KiB → 1–2.25 KiB | 全布局 | C4 几何均值 1.0217×，重 prefill -4.4%～-6.4% |
| 5 | 统一一拍 KV 预取 | `8e0989914` | K/V 创建指针领先执行 1 拍，每流在途 2 槽 | 全布局 | 36 组 C4 耗时 -8.21% |
| 6 | 按 Q 行派发任务 | `2e4ad3565` | AIC/AIV 分别按 Q 行生成任务描述符，满块 QK/PV 专用路径 + ND 尾块双目标 Fixpipe | 全布局 | 36 组 C4 耗时 -3.04%，AIC Scalar 降到 AIC MAC 以下 |
| 7 | **同页双 KV head 合并** | `f1d13043b` | 本文 A2/A4 | **仅 PA_BBND ND decode** | decode C8/C4 1.0875 → **1.3918** |
| 8 | 派发条件收紧 | `6b9da048a` | gSize 必须为 ≥4 的 2 的幂 | 修正正确性 | gSize 扫描 34/72 → **72/72** |

**路径的内在逻辑**：1–3 层降低向量与标量开销，4–6 层降低搬运与任务管理开销，把瓶颈一步步推到 GM 事务效率上；第 7 层才动事务粒度本身。顺序不能颠倒——在向量/标量还是瓶颈时改搬运，收益会被掩盖（B7 的历史失败正是发生在第 5、6 层之前）。

第 7 层内部还包含两个**独立缺陷的修复**，它们才是合并方案从「看起来不可用」变成「可用」的真正原因：

- **流水缺陷**：M64 曾用 `GROUP_N=20`，`20+3 > PRELOAD_N=20`，单核 task ≥20 时必挂死；
- **标量缺陷**：softmax VF 扁平行循环里写了 `spanOffset / rowSpan`（运行时除数），每 4 行一次整数除法，使整个 VF 慢约 1.9 倍，且污染所有路径的对照实验。

---

## A4. 具体改动清单

### A4.1 派发条件（三处必须逐字一致）

```cpp
// host tiling 与 AICPU metadata
const int64_t mergeGroupSize = n2Size > 0 ? n1Size / n2Size : 0;
const bool mergeGroupAligned = mergeGroupSize >= 4 && (mergeGroupSize & (mergeGroupSize - 1)) == 0;

mergeKvHeads = decodeS1GMerge            // layout_q_descale == N2TGD
            && pageAttentionFlag
            && kvLayout == PA_BBND
            && n2Size == 2
            && qkHeadDim == 256 && vHeadDim == 256
            && mergeGroupAligned;
```

三处位置：

| 位置 | 文件 | 作用 |
| --- | --- | --- |
| host tiling | [quant_flash_attn_tiling_mxfp4.cpp](../attention/quant_flash_attn/op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp) `SplitPolicy()` | 选 `sOuterFactor_=32`、`sInnerFactor_=256`，进而选 Config 6 |
| AICPU metadata | [quant_flash_attn_metadata_aicpu.cpp](../attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp) `ParamsInit()` | 生成与 host 一致的任务划分 |
| golden 调度模型 | [mxfp4_schedule.py](../attention/quant_flash_attn/tests/pytest/qfa_mxfp4_test/common/mxfp4_schedule.py) `decode_sections()` | 参考侧反推同一调度 |

**任何一处不一致都会立即死锁或算错**，这是本设计最重要的工程约束。

### A4.2 tiling key 与编译期开关

新增专用配置编号，让 kernel 在编译期就能区分合并语义：

```cpp
#define Config_MergeKvHeads_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 6
```

[quant_flash_attn.cpp](../attention/quant_flash_attn/op_kernel/quant_flash_attn.cpp) 依 `config` 计算 `mergeKvHeads`，作为模板参数传给 **Cube / Vector 及其 Dummy 四个类**：

```cpp
constexpr bool mergeKvHeads = (config == Config_MergeKvHeads_...);
```

三个模板类导出同一常量：

```cpp
static constexpr bool MERGE_KV_HEADS = mergeKvHeads && !useDn && KvLayoutType == 1;
static constexpr uint32_t kvTokenBlock = MERGE_KV_HEADS ? s2BaseSize / 2U : s2BaseSize;
static constexpr uint32_t mergePartRows = mBaseSize / 2U;   // = 32
```

**踩坑记录**：AIV 侧实例化的是 `CubeBlockDummy`。若开关只由真实 Cube 导出，AIV 取到的恒是 `false`，两侧任务枚举与 score 槽数不同 → 跨核 flag 失衡 → 立即死锁。merge4/5/6 三轮隔离实验都因此走错方向。**Dummy 必须导出同一值。**

### A4.3 metadata 调度改写

命中合并时：

| 字段 | 值 | 效果 |
| --- | --- | --- |
| `sOuterFactor` | 32 | `mBaseSize_ = 32 × aiv/aic = 64` |
| `sInnerFactor` | 128 | 每 head 的 token 块（调度语义） |
| `baseInfo.kvHeadNum` | **1** | KV head 不再是独立任务轴 |
| `s2BaseSize_`（header） | `128 × 2 = 256` | header 写 **score 宽度**，便于下游区分合并与基线 |
| `param.s2BaseSize` | 128 | 调度用 token 块 |

于是 `GetGroupSize() = N_q`、`GetMSize() = seqQ × N_q`、`GetHeadNum() = 1`，调度天然变成「每 (batch, 128-token 块) 一个 task」。

下游可用 header 区分：合并 `[1,*,64,256,28]`，基线 `[1,*,128,128,28]`。

### A4.4 kernel 常量与任务枚举

[quant_flash_attn_kernel_mxfp4.h](../attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_kernel_mxfp4.h) `InitConstInfo` ND 分支：

```cpp
realN2Size = MERGE_KV_HEADS ? 1U : n2Size;
realGSize  = MERGE_KV_HEADS ? n2Size * gSize : gSize;
```

由此**自动正确、无需改动**的部分：`bIdx`/`realHead` 推导、M 行数、Q 的 token 跨步、输出行映射、LSE 的 queryHead、S2 块↔token 换算。

需要把 `s2BaseSize` 换成 `kvTokenBlock` 的位置：`DispatchRows` 的 `task.s2Idx` 与 `actSingleLoopS2Size`、`GetTaskDealMode` 的 `s2LoopTimes`、`CalcCurS2StartEnd*` 的全部换算。

### A4.5 GM→L1 搬运：`CopyMergedKvBlock`

[quant_flash_attn_block_cube_mxfp4.h](../attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h)：

- payload 按 `fullRowBytes = (D/2) × 2 = 256 B` 整行搬，`srcDValue == dValue` → **行连续，填满取回粒度**；
- 按 page 边界分段（`pageOffset = logicalToken & (blockSize-1)`），因此 page=128/256/512 都正确，page 不是 128 整数倍时也能跨页拼接；
- K scale 用 `Dn2Nz(nValue = (D/64)×2 = 8)`、V scale 用 `Nd2Nz(dValue = D×2 = 512)`，同样整行连续；
- 槽内 head h 的字节偏移：payload `h × kvTokenBlock × D/2`，K scale `h × (D/64) × 32`，V scale `h × (D/16) × (kvTokenBlock/64) × 32`。

消费侧 `LoadK2L0`/`LoadV2L0` 按 `headIdx` 加基址偏移，`srcStride` 相应放大。

### A4.6 L1 Q tile 采用 part-major 行序

**硬件约束**：GM→L1 的 MX scale 搬运只能以完整 16 行 fractal 落地，目的地址必须 32 B 对齐。半 fractal 写入触发 AI Core error 80。

因此 part p（= KV head p）占 `[p×32, (p+1)×32)` 行，内部为 `[token][gSize 个 head]`：

- payload：每 part 一次 `Nd2Nz(ndNum=tokenCount, nValue=gSize, dValue=D/2, dstNzMatrixStride=gSize×32)`；
- scale：每 part 一次 `Dn2Nz(dnNum=1, dValue=tokenCount×gSize, nValue=D/64)`，目的基址 `p × (32/16) × (D/64) × 32`（32 B 对齐）；
- `LoadQ2L0(partIdx)` 用 `mStartPosition = partIdx × 32/16` 取本 part。

### A4.7 QK 与 Fixpipe

每个 chunk 只计算本 KV head 的 Q part（`Mmad m = 32`），**与基线单 head 任务同规模，不增加 Mmad 量**。L0C 为 part-major，用 `FixpipeMm1Merged` 写回自然行序的 score UB。

### A4.8 softmax VF：跳段而非置 -inf

[vf_softmax_fp32_mxfp4.h](../attention/quant_flash_attn/op_kernel/arch35/vf/vf_softmax_fp32_mxfp4.h)：

行归属以 `qGroupSize = realGSize` 为周期，每周期内连续 `realGroupSize` 行同属一个 KV head。本 chunk 只处理属于该 head 的行段：

```cpp
rowPeriod = realGroupSize * mergeHeadCount;
rowSpan   = realGroupSize;
phaseOffset = qRowPhase % rowPeriod;      // 取模只在每次 VF 调用做一次
rowBegin = mergeHeadIdx * realGroupSize + rowPeriod - phaseOffset;
rowBegin -= (rowBegin >= rowPeriod ? rowPeriod : 0U);
...
spanOffset += 4U;
const uint32_t carry = spanOffset >= rowSpan ? 1U : 0U;   // 比较，不是除法
spanOffset -= carry * rowSpan;
rowBase += carry * rowPeriod;
```

两条硬性要求：

1. **向量体内不得有数据相关分支或额外 MaskReg**——实测会让 VF 慢 3.5 倍；
2. **行循环内不得有运行时除数的整数除法**——实测让整个 VF 慢 1.9 倍。

### A4.9 流水常量统一

[quant_flash_attn_mxfp4_common.h](../attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_mxfp4_common.h)：

```cpp
static constexpr uint32_t MXFP4_TASK_GROUP_N = 16U;
```

主控 `GROUP_N`、Cube `PV_TASKS_PER_GROUP`、Vector `TASKS_PER_GROUP` 全部取它，并在主控加：

```cpp
static_assert(GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N,
              "Pscale of a group must be published before Cube's first C2 of that group");
```

### A4.10 参考模型同步

`mxfp4_schedule.py` 靠读 metadata header 反推调度（`m_base==64 && task_n==256` → 合并，`task_n //= N_kv`），`cpu_mxfp4_golden_kernel.py` 与 Triton golden 读同一 schedule。**golden 不得写死组大小或派发条件**，否则回退路径会算错。

### A4.11 改动文件汇总

| 文件 | 角色 |
| --- | --- |
| `op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp` / `.h` | 派发条件、分块因子、Config 选择 |
| `op_kernel/arch35/quant_flash_attn_common_def.h` | Config 6 编号 |
| `op_kernel/arch35/quant_flash_attn_template_tiling_key.h` | tiling key 组装 |
| `op_kernel/quant_flash_attn.cpp` | 编译期开关传入四个模板类 |
| `op_kernel/arch35/quant_flash_attn_kernel_mxfp4.h` | `realN2Size`/`realGSize`/`kvTokenBlock`、任务枚举、流水断言 |
| `op_kernel/arch35/quant_flash_attn_mxfp4_common.h` | `MXFP4_TASK_GROUP_N` |
| `op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h` | 搬运、Q part-major、QK/Fixpipe、PV chunk |
| `op_kernel/arch35/quant_flash_attn_block_vec_mxfp4.h` | V1 chunk、score 槽、`mergeHeadIdx` 传递 |
| `op_kernel/arch35/vf/vf_softmax_fp32_mxfp4.h` | 跳段行循环、无除法进位 |
| `quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp` | 调度改写 |
| `tests/.../common/mxfp4_schedule.py` | 参考侧调度反推 |
| `tests/.../common/cpu_mxfp4_golden_kernel.py` | 块对角参考计算 |
| `tests/.../common/triton_mxfp4_golden_kernel.py` | Triton 参考同步 |
| `tests/.../common/perf_convert.py` | 性能输入布局转换 |

---

## A5. 优势

### A5.1 实测收益（2026-11-25，NPU7，Graph，同轮基线对照）

| 用例 | 基线 C4 (µs) | 候选 C4 (µs) | 变化 | 基线 C8/C4 | 候选 C8/C4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_1 | 281.911 | 182.877 | **-35.13%** | 1.1079 | **1.6999** |
| decode_2 | 790.869 | 451.556 | **-42.90%** | 1.0030 | **1.7575** |
| decode_3 | 34.384 | 29.484 | -14.25% | 1.2093 | 1.4064 |
| decode_4 | 46.386 | 54.138 | +16.71% | 1.0406 | 0.8931 |

| 范围 | 基线 C8/C4 | 候选 C8/C4 |
| --- | ---: | ---: |
| PA_BBND decode（4） | 1.0875 | **1.3918** |
| PA_BBND all（12） | 1.0802 | 1.1745 |
| PA_BBND prefill（6） | 1.1106 | 1.1129 |
| PA_BBND mixed（2） | 0.9806 | 0.9832 |

PA_BBND 整体 C4 耗时相对同轮基线 **-8.02%**。

### A5.2 机理层面的优势

| 项 | 基线 | 合并 |
| --- | --- | --- |
| GM 源行宽 / 跨步 | 128 B / 256 B（跨步） | **256 B / 256 B（连续）** |
| decode_1 DRAM 读量 | 5325 KB | **2802 KB** |
| task 数 | 2 × page 数 | **page 数**（减半） |
| 每 page GM 字节 | 69632 B | 69632 B（**相同**） |
| V2/Pscale 总行数 | — | **不变** |
| QK Mmad | 每 task 一个 chunk | 每 task 两个 chunk，各只算本 head 的 Q part（`m = mergePartRows = 32`），**不增加有效 Mmad 量** |

DRAM 读量取自 `f1d13043b` 提交时的 `msprof op / Memory` 采集；其余为设计推导与本轮 Graph 实测。

### A5.3 工程层面的优势

- **布局隔离**：PA_BNBD / PA_NZ 的 C4 耗时比 0.9923 / 0.9972，metadata header 仍是 `[1,*,128,128,28]`，改动不外溢；
- **精度零回归**：744 例矩阵 + document full 4860 例，合计 **5604 例全通过**；
- **无需扩容**：L1/L0/UB 预算完全沿用，不挤占其他配置；
- **可回退**：不满足派发条件即走基线，行为与优化前逐位一致。

---

## A6. 限制与已知问题

### A6.1 适用范围窄

只有 `PA_BBND + N2TGD + N_kv=2 + D=Dv=256 + gSize 为 ≥4 的 2 的幂` 才生效。DN prefill 完全不派发。**PA_BNBD / PA_NZ 的高 C8/C4 比值来自布局本身，与本优化无关**——它们的基线就已经是 1.4036 / 1.4130。

### A6.2 gSize 必须是 ≥4 的 2 的幂（正确性硬约束）

M64 tile 按 `mergePartRows = 32` 拆成两个 KV head part，head 边界落在每 `gSize/2` 行；只有 gSize 为 ≥4 的 2 的幂时该边界才与 Q 装载粒度和 VF 的 4 行分组对齐。

实测（`N_kv=2, D=256, PA_BBND, N2TGD`，72 组）：

| N_q | gSize | 修复前 | 修复后 |
| ---: | ---: | ---: | ---: |
| 8 / 16 / 32 / 64 | 4 / 8 / 16 / 32 | 全通过 | 全通过 |
| 2 / 4 / 6 / 10 / 12 / 20 / 24 / 48 | 1/2/3/5/6/10/12/24 | **全失败** | 全通过（回退基线） |

**这是一个曾经存在的真实缺陷**：`f1d13043b` 的派发条件不约束 gSize，`6b9da048a` 才收紧。注意 `%4` 判据也是错的（会误放行 gSize=12）。

### A6.3 decode_4 回退约 17%

单核约 24 个 task，刚越过 `PRELOAD_N=20`，C1/C2 爬坡开销开始显现；同时 `kvHeadNum 2→1` 让负载均衡粒度变粗。PA_BNBD/NZ 上也有 5.6%/6.6% 的小幅回退，来自共享改动（`GROUP_N` 20→16、VF 进位改写）。

### A6.4 未达 1.4 目标

PA_BBND decode 几何均值 1.3918，差 0.008。分档回退可以补上（离线验证二维判据可达 1.4460），但**需要跨算子接口改动**，见 A6.5。

### A6.5 分档回退受阻于接口

分档判据依赖真实 `seqused_kv` 分布，四条通道全堵：

1. host tiling **不得解引用 `GetData()`**（返回设备地址，解引用会段错误 `PYTEST_EXIT=139`）；
2. kernel 运行时切换需实例化两套，UB/L1 翻倍；
3. AICPU metadata 能读真实数据，但决策写在设备内存，host 读不到；
4. metadata 的 shape 由调用方预分配，不能当传信通道；`block_table` 列数与 `maxKV` 同样被长尾污染。

详见附录 B12。

### A6.6 Config 7（M32/N128）死锁未修

保留实现但不派发。单核 task 数 ≥ `PRELOAD_N=20` 时挂死，已排除组大小、QK 槽数、PV 分段、P 槽数四类因素。详见附录 B3.3。

### A6.7 判据必须是二维的

单看 task 数会误伤 decode_3（task/核仅 12 却达标 1.4211，因为 `seqQ=1` 使 M 行仅 8）。一维阈值会把几何均值从 1.3939 拉低到 1.3833。正确判据：

```
派发合并 ⟺ (合并后单核 task 数 >= 2 * PRELOAD_N) 或 (基线单 task M 行 <= 16)
```

---

## A7. 使用注意事项

### A7.1 调用方 checklist

- [ ] `layout_q_descale` 必须是 `N2TGD`，`layout_kv` 必须是 `PA_BBND`；
- [ ] `N_kv == 2`、`D == Dv == 256`；
- [ ] `N_q / N_kv` 是 **≥4 的 2 的幂**（8/16/32/64…）。不满足时算子自动回退基线，**不会报错但也没有加速**；
- [ ] page 大小 128/256/512 均可；合并的 token 块固定 128，与 page 解耦；
- [ ] 不要期望 prefill 或 PA_BNBD/PA_NZ 有收益。

### A7.2 修改代码时的八条红线

1. **host tiling / AICPU metadata / golden 三处派发条件必须逐字一致**，否则立即死锁或算错。
2. **编译期开关必须同时传给 Cube/Vector 及其 Dummy**，AIV 取的是 Dummy。
3. **`GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N`** 必须成立，且三处组大小取同一常量。
4. **VF 行循环内不得有运行时除数的整数除法**，进位一律用比较。
5. **向量体内不得加数据相关分支或额外 MaskReg**。
6. **MX scale 只能整 16 行 fractal 落地，目的地址 32 B 对齐**，半 fractal 写入触发 AI Core error 80。
7. **host tiling 不得解引用 `GetData()`**，只能做 null 判断。
8. **golden 不得写死组大小/派发条件**，必须读实际 schedule。

### A7.3 验证要求

按 [kernel-optimized.md](../attention/quant_flash_attn/docs/kernel-optimized.md) 的「优化遵守标准」：

| 改动规模 | 必跑 |
| --- | --- |
| 小改动 | `document small` + `bench16k`，**PA_BBND/PA_BNBD/PA_NZ 三布局都要过** |
| 多次大改动 | 追加一次 `document full` |

**血泪教训**：744 例矩阵里真正走到合并路径的只有 **4 个用例**（bench16k 的 4 个 decode，且只有 BBND 实际合并）。`document small` 和 Qwen3-8B 两个集合**一个都不命中**。gSize 缺陷因此历次全绿都没暴露，只有 `document full` 的 270 个 `qwen38_27b_TP2` 用例能命中。

**结论：改了合并路径，`document full` 不可省略。** 另建议对 gSize ∈ {2,4,6,8,10,12,16,24,32} 做 shape 扫描，用 `--reference-backend global`（与 metadata 无关）交叉验证。

### A7.4 测量与环境口径

- 性能必须 Graph 回放 + Event 计时，预热 3、5 轮 × 20 次取中位数；输入准备、metadata、布局转换在计时外；
- 基线与候选必须**同轮、同卡、同输入 SHA256、同 case 指纹**；跨批次数字不可直接比；
- 长 prefill 用例五轮极差常超 5%，解读单例时先看 `spread_pct`；decode 四例极差都在 4% 以内；
- **并行跑精度时按测试集最大单例显存选 worker 数**：Qwen3-8B 单例 9.13 GB，16 worker 会 OOM，必须降到 6；document/bench16k 可用 16；
- **共享机器上卡的可用性会变**：任务开始前探测通过不代表运行期一直可用。本轮 NPU4 中途被其他用户占用后不可 retain，导致 7 个分片失败。长跑建议分片失败时自动换卡重试。

---

## A8. 复现与证据

| 内容 | 位置 |
| --- | --- |
| 完整验证报告 | [validation-report-20261125.md](validation-report-20261125.md) |
| 原始证据（433 文件） | `evidence/ev21/`（accuracy / perf / prof / sweep / build） |
| 逐例性能 CSV | `evidence/ev21/perf/bench16k.csv` |
| gSize 扫描修复前后对照 | `evidence/ev21/sweep/` |
| 优化提交 | `f1d13043b`（合并）、`6b9da048a`（派发收紧） |

关键提交链：`98e3d8d1a` → `beb2b64d7` → `00e1713d7` → `bc68af786` → `8e0989914` → `2e4ad3565` → `f1d13043b` → `6b9da048a`。

---
---

# 附录 B：设计推演与历史实验记录

> **以下是本文档的原始内容（原第 1–12 章），按时间顺序保留了探索过程。**
> 其中 B3「Config 6/7 不派发」、B7 整章「合并方案净退化」、B7.9「Q scale 半 fractal 写入」、B7.14「合并不能减少 DRAM 字节」**均已被后续实测推翻**，当前生效设计见第一部分 A2–A4。
> 保留这些记录的价值在于：它们给出了硬件约束的实测边界和已证伪的路径，避免后来者重走。

## B1. 目标与约束

优化对象为 `attention/quant_flash_attn` 的 arch35 MXFP4（C4A4）实现，目标平台为 Ascend 950PR，重点覆盖 ND decode 小 M 场景。

- 保持 PA_BBND、PA_BNBD、PA_NZ 和现有量化路径的精度要求不变；
- C8/C4 耗时比以同一用例、同一输入、同一设备、同一 Graph 计时口径比较，目标为 1.4～2.0 倍以上；
- 不把输入生成、格式转换和 metadata 生成计入 attention 耗时；
- 真实收益必须以宿主机 `k831` 容器内的独立构建和 NPU 实测为准。

## B2. 基线瓶颈

基线提交为 `5f2d913c3`。D256 ND 使用 M128/N128，QK 和 PV 都按 128 token 处理。

D256 下 QK 的 L0A/L0B/L0C 只能使用单槽，任务派发和 Pscale/V2 的管理成本较明显。减小 M 可以减少无效 Q 行计算；调整任务 N 和 QK 槽数则用于降低任务管理成本并改善 QK Mmad 与 Fixpipe 的重叠。

小 M 不会减少有效 KV token 数，也不保证降低 KV 搬运量。最终收益必须来自实际 kernel 执行效率，不能通过减少实际 KV token 获得虚假收益。

## B3. 配置设计与当前派发状态

**当时的结论：小 M 方向实测未达目标，Config 6/7 的 kernel 实现保留在代码中，但 host tiling 与 AICPU metadata 均不再派发它们。** ND decode D=Dv=256 仍走基线 Config 5（M128/N128），与 `5f2d913c3` 行为一致。

> **状态已更新**：Config 6 后来被改造为块对角合并专用配置并重新派发（见 A4.1/A4.2），编号改为 `Config_MergeKvHeads_S1Aligned64_S2Aligned256_DAligned256_DVAligned256`。Config 7 仍保留不派发，B3.3 的死锁未修。本节以下内容描述的是改造前的中间状态。

### B3.1 Config 6：M64/N256/D256（当时保留不派发；现已改造为合并专用配置并重新派发）

新增 `Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 = 6`：

- AIC Q 行块为 32，经 CV_RATIO=2 后合并为 M64；
- task N 为 256，QK/PV 都按 128 token 分段；
- QK 仍为单槽，PV 使用双槽；
- 每组最多 20 个 task；
- 曾用于 `maxSeqQ*gSize>32` 的 ND decode D=Dv=256 场景，现已停止派发。

### B3.2 Config 7：M32/N128/D256（保留，不派发，存在已知死锁）

新增 `Config_S1Aligned32_S2Aligned128_DAligned256_DVAligned256 = 7`：

- 曾在 ND decode、D=Dv=256 且 `maxSeqQ*gSize<=32` 时选择，现已停止派发；
- AIC Q 行块为 16，经 CV_RATIO=2 后合并为 M32；
- task N 为 128，QK/PV 均按 128 token 分段；
- 保留代码为 QK 单槽 + PV128 双槽 + 40 个 L1 P/统计槽（最后一次隔离实验的状态）；
- 每组最多 16 个 task。

实验过程中先后试过 QK 双槽 + PV128 单槽、组大小 16、PV64 双槽、QK 单槽、P 槽 40，五种组合均在同一边界挂死，详见 B3.3。

host tiling 与 AICPU metadata 曾使用完全相同的选档条件（`maxSeqQ*gSize<=32` 选 Config 7，否则 Config 6），现已移除该派发分支。Config 7 的 metadata 在 CV 合并前为 `mBase=16/taskN=128`，合并后为 `mBase=32/taskN=128`；Config 6 对应合并前 `32/256`、合并后 `64/256`。旧配置编号和其他量化模式保持不变。

### B3.3 M32 已知缺陷：C1/C2 同 tick 并发死锁

实测边界（NPU 7，eager 与 graph 均复现）：

| 形状 | 单核最大 task 数 | 结果 |
| --- | ---: | --- |
| B4 / KV8192 | 19 | 通过 |
| B1 / KV16384 | 10 | 通过 |
| B1 / KV32768 | 10 | 通过 |
| B5 / KV256 | 2 | 通过 |
| B5 / KV8192 | 23 | 挂死 |
| B4 / KV16384 | 37 | 挂死 |

分界不是 batch 数、KV 长度或 FD 开关，而是**单核 task 数是否达到 `PRELOAD_N=20`**。`ExecuteTick` 在 `tick>=20` 时首次同时发射 C2(task t−20) 与 C1(task t)；task 数 ≤19 时 C2 独占执行，因此不触发。

已逐一排除且均无效的因素：组大小（16 / 20）、QK L0 槽数（1 / 2）、PV 分段（64 / 128）、L1 P 与统计槽数（20 / 40）。M32 的数值正确性本身没有问题：`short1/3/4` 三例 Attention 与 LSE 全部通过。该缺陷未修复，故不派发。

模板生成器仍会为 Config 6/7 生成 DN/ND 变体，因此小 M 配置在编译期满足两种变体的容量和分段约束；host 目前不派发任何小 M 配置。

## B4. 内存预算

### B4.1 M128/N128/D256 基线

| 资源 | 近似占用 | 预算 |
| --- | ---: | ---: |
| L0A | 32 KiB | 64 KiB |
| L0B | 50 KiB | 64 KiB |
| L0C | 200 KiB | 256 KiB |
| L1 | 261 KiB | 512 KiB |
| Vector UB | 约 233～239 KiB | 248 KiB |

### B4.2 M64/N256/D256

| 资源 | 计算方式 | 近似占用 | 预算 |
| --- | ---: | ---: | ---: |
| L0A | QK-Q 8 KiB + 2×PV-P 8 KiB | 24 KiB | 64 KiB |
| L0B | QK-K 16 KiB + 2×PV-V 17 KiB | 50 KiB | 64 KiB |
| L0C | QK 32 KiB + PV 68 KiB | 100 KiB | 256 KiB |
| L1 | P 160 KiB + Pscale 12.5 KiB + Q 10 KiB + KV 68 KiB + padding | 约 252 KiB | 512 KiB |
| Vector UB | 独立输出暂存、Pscale、history、score 和 P 双槽 | 约 234 KiB | 248 KiB |

M64 的 PV 如果不按 128 分段，V 双槽会达到约 68 KiB，超过 L0B 的 64 KiB 预算。因此保留 128 token 的 PV 分段是正确性的必要条件。

### B4.3 M32/N128/D256

| 资源 | 计算方式 | 占用 | 预算 |
| --- | ---: | ---: | ---: |
| L0A | 2×QK-Q 4096B + 2×PV64-P 1024B | 10240B | 64 KiB |
| L0B | 2×QK-K 16384B + 2×PV64-V 8704B | 50176B | 64 KiB |
| L0C | 2×QK 16384B + PV 34816B | 67584B | 256 KiB |
| L1 | 20 组 P/Pscale + Q + 4 个 KV 槽 + padding | 119872B | 512 KiB |
| Vector UB | cross 33280B + persist 105472B + transient 38912B | 177664B | 248 KiB |

如果 M32 同时启用 QK 双槽和 PV128 双 V 槽，L0B 将达到 67584B，超过 65536B 上限；PV128 单槽又在第 20 个 task 后的 C1/C2 并发中发生等待。因此当前设计选择 PV64 双槽。

## B5. 寻址与组边界

M64 的 PV 第二段（chunkOffset=128）的源地址关系为：

- P：`kStartPosition=128/64=2`；
- Pscale：`yStartPosition=128/64=2`；
- V：`mStartPosition=128/16=8`；
- Vscale：`yStartPosition=128/64=2`。

M32 的 PV64 第二段使用同一寻址公式：`chunkOffset=64` 时 P/Pscale/Vscale 的起点为 1，V 的 `mStartPosition=64/16=4`。

源 stride 仍按完整任务 N 设置。L0C 仅在当前组首个 task 的首个 chunk 初始化，后续 chunk 和 task 全部累加，仅在组尾执行 Fixpipe。

M64 的 Cube、Vector、kernel 和 CPU/Triton golden 统一使用 `group_tasks=20`，M32 与旧 M128 配置使用 16。CPU 和 Triton golden 不得写死组大小，必须读取实际 schedule。

## B6. BBND 搬运优化：已淘汰的两个变体

### B6.1 依据

基线 `bench16k_decode_1` 的 `msprof op / Default` 显示 `aic_mte2_ratio=0.954`、`GM_to_L1_bw_usage_rate=30.9%`、`aic_cube_ratio=0.066`、`aic_cube_wait_ratio=0.939`。即 decode 完全由 GM→L1 搬运受限，且受限于事务效率而非带宽。

PA_BBND 的 GM 布局为 `[page][token][head][D/2]`，同一 token 的各 KV head 交织。只搬本 head 时，ND2NZ 的每行为「128B 有效 / 256B 源跨步」；`srcDValue != dValue`，无法合并为长 burst。

### B6.2 变体 A：payload 整页全 head（已淘汰）

把同一 token 的全部 KV head 一次连续读入同一个 L1 KV 槽，消费侧按 head 偏移取用；scale 保持单 head。
门控 `KvLayoutType == PA_BBND && n2Size <= 2 && blockSize == s2BaseSize`。

实测（M128 干净构建）decode C4 = 275.462 / 794.420 / 35.696 / 46.479，几何均值 **1.0889**，低于基线 1.1102。
原因：行连续带来的事务效率提升，被「同一 page 的两个 head 各有一个 task、各读一遍整页」造成的 GM 流量翻倍完全抵消。

### B6.3 变体 B：payload + scale 都整页全 head（已淘汰）

K scale 用 `Dn2Nz(nValue = D/64 * heads)`、V scale 用 `Nd2Nz(dValue = D * heads)` 使源行连续，消费侧按 head 偏移与放大的 `srcStride` 取用。

实测 decode C4 = 283.251 / 802.608 / 35.960 / 46.356，比变体 A 更差。说明 scale 的小跨步读并不是主要瓶颈，多搬的字节反而有害。

**共同结论：基线的 GM 字节数已经最小，任何「多读再挑」的搬运改写都不可能净收益。必须让同一 page 的两个 KV head 只被读一次。**

## B7. 同页双 KV head 块对角合并（历史轮次，最终形态见 A2–A4 与第 B11 节）

**本节记录的是 merge2～merge8 轮次的中间结论，已被第 B11 节与第一部分推翻并取代。**
当时的判断是「净退化 + 单核 task 数 ≥ `PRELOAD_N=20` 时挂死」，因此 host tiling 与 AICPU metadata 停止派发。后续定位到两个独立缺陷（组大小违反流水约束、行循环内的整数除法），修复后本方案在长 KV 上取得 1.5～1.76 倍加速。**阅读本节时请以第一部分 A2–A4 为准**，以下保留设计推演与硬件约束，供后续参考。

### B7.1 核心思想

一个 task 覆盖**一个 KV page 的全部 head**：

- M 轴 = `seqQ * N_q`（全部 Q head，不再按 KV head 拆分）；
- N 轴 = `n2Size * kvTokenBlock` = 2 × 128 = 256 列，列布局 `[head0 的 128 token; head1 的 128 token]`；
- softmax 前把「行所属 KV head ≠ 列所属 KV head」的整块置 `-inf`（块对角 mask）；
- 由于 softmax 后非对角块恒为 0，PV 用同样堆叠的 V 得到的每一行结果与单 head 计算完全一致。

收益来源：

| 项 | 基线（Config 5） | 合并（Config 6） |
| --- | --- | --- |
| 每 page 的 GM payload+scale 字节 | 2 task × 2 × (16384+1024) = 69632 | 1 task × 2 × (32768+2048) = 69632（**相同**） |
| GM 源行宽 / 跨步 | 128B / 256B（跨步） | 256B / 256B（**连续**） |
| QK Mmad 规模 | 2 × (128×128×256) | 64×256×256（**减半**） |
| task 数 | 2 × page 数 | page 数（**减半**） |

关键巧合：**Config 6（M64/N256/D256）的 `mmKVSize = 256*256/2 = 32768B`、`mmKVScaleSize = 256*256/32 = 2048B`，正好等于一个 BBND 整页双 head 的 payload 与 scale 字节数**，因此 L1/L0 预算完全沿用 Config 6，无需扩容。

### B7.2 派发条件（tiling 与 metadata 必须完全一致）

```
quantMode == MXFP4(5)
&& decodeS1GMerge (layout_q_descale == "N2TGD")
&& layout_kv == PA_BBND && pageAttention
&& qkHeadDim == 256 && vHeadDim == 256
&& n2Size == 2
&& gSize % 4 == 0            // VF 按 4 行一组做块对角判定
```

metadata 侧看不到 `block_size`，因此**合并任务的 token 块固定为 128**（`kvTokenBlock = s2BaseSize / n2Size`），与 page 大小解耦；page 为 128/256/512 时 128 token 必落在单页内，page=64 时按页分段搬运。

### B7.3 metadata 改动

`attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp`，`ParamsInit` 内（约 L296-341）：

- 新增成员 `mergeKvHeads_`，按 B7.2 判定（用 `numHeadsKv_`、`numHeadsQ_`、`headDim_`、`headDimV_`、`layoutKv_`、`layoutQDescale_`、`quantMode_`）；
- 命中时：`sOuterFactor = 32`（使 `mBaseSize_ = 32 * aiv/aic = 64`），`sInnerFactor = 128` 保持不变；
- 命中时把 `baseInfo.kvHeadNum = 1`（`numHeadsQ_` 不变），于是 `GetGroupSize() = N_q = 16`、`GetMSize() = seqQ*16`、`GetHeadNum() = 1`，调度天然变成「每 (batch, 128-token 块) 一个 task」；
- `KernelSplitMode` 仍为 `BN2_S1G_S2`；`param.mBaseSize = 64`、`param.s2BaseSize = 128`。

metadata header 的 `HEAD_M_BASE_SIZE_INDEX = 64`、`HEAD_S2_BASE_SIZE_INDEX = 128`（token 语义）。

### B7.4 tiling 改动

`op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp`：

- `SplitPolicy()`：命中 B7.2 时 `sOuterFactor_ = 32`、`sInnerFactor_ = 128`；
- `UpdateTilingKeyConfig()`：新增分支 `sOuter == 64 && sInner == 128 && dSize == 256 && dVsize == 256 → Config 6`；
- tiling data 的 `n2Size / gSize` **保持真实值 2 / 8**（GM 寻址需要），不写假值。

### B7.5 kernel 侧编译期开关

在 cube / vec / kernel 三个模板类中定义同一常量：

```cpp
static constexpr bool MERGE_KV_HEADS = M64_CONFIG && !USE_DN && KvLayoutType == 1;
```

Config 6 目前不被任何其它场景派发，因此不需要新增 tiling key 位，也不需要新增模板变体。

`InitConstInfo`（`quant_flash_attn_kernel_mxfp4.h` L274-280）ND 分支改为：

```cpp
realN2Size = MERGE_KV_HEADS ? 1U : n2Size;
realGSize  = MERGE_KV_HEADS ? n2Size * gSize : gSize;
```

由此**自动正确**的部分（无需改动）：

- `bIdx = bN2Cur / realN2Size`、`realHead = bN2Cur % realN2Size = 0`；
- M 行数 `actSeqLensQ_ * realGSize`；
- `CopyQGm2L1Nd` 的 `tokenByteStride = realN2Size*realGSize*rowBytes = 16*128`、head 偏移 `= h*128`，与真实 TND Q 布局一致；
- `WriteAttentionResult` 的 `outputRow = (base + row/16)*16 + row%16`、LSE 的 `queryHead = h`；
- `CalcCurS2StartEndWithSparse` 的 token 换算 `gS1Idx / realGSize`。

### B7.6 token 块与 score 宽度解耦

新增 `kvTokenBlock = MERGE_KV_HEADS ? s2BaseSize / constInfo_.n2Size : s2BaseSize`（=128）。
以下位置把 `s2BaseSize` 换成 `kvTokenBlock`：

- `DispatchRows`：`task.s2Idx = s2 * kvTokenBlock`；`task.actSingleLoopS2Size = min(kvTokenBlock, actSeqLensKv_ - s2Idx)`（**每 head 的有效 token 数**）；
- `GetTaskDealMode`：`s2LoopTimes = ceil(actSeqLensKv_ / kvTokenBlock)`；
- `CalcCurS2StartEndNoSparse` / `WithSparse` 中所有 S2 块↔token 换算。

`task.isGroupLast` 仍按 `GROUP_N`（Config 6 为 20）个 task 计数，语义不变。

### B7.7 chunk 循环统一改写

cube 的 QK、cube 的 PV、vector 的 V1 三处循环统一为：

```cpp
const uint32_t chunkCount = MERGE_KV_HEADS ? constInfo_.n2Size
                                           : (tokens + SPLIT - 1) / SPLIT;
for (uint32_t c = 0; c < chunkCount; ++c) {
    const uint32_t slotOffset = c * SPLIT;                 // P / score 列偏移，0 或 128
    const uint32_t tokenOffset = MERGE_KV_HEADS ? 0 : slotOffset;
    const uint32_t valid = MERGE_KV_HEADS ? tokens : Min(SPLIT, tokens - slotOffset);
    const uint32_t headIdx = MERGE_KV_HEADS ? c : 0;       // L1 KV 槽内 head 偏移
    ...
}
```

QK 的 `SPLIT = s2SplitSize = 128`，PV 的 `SPLIT = PV_SPLIT_SIZE = 128`（Config 6）。
`MatmulQK` 的 `n = valid`，`LoadP2L0` 的 `chunkOffset = slotOffset`（P 是完整的 M64×256 tile，寻址公式不变）。

### B7.8 L1 KV 槽内布局与消费侧偏移

搬运（`CopyFullKvPage`，合并模式下 `kvSlotHeads_ = 1`，因为 `s2BaseSize` 已含 2 head）：

- payload：`Nd2Nz(nValue = kvTokenBlock = 128, dValue = 128B * n2Size = 256, srcDValue = 256, dstNzC0Stride = 128)`
  → 槽内 `[D/32 = 8][128/16 = 8]` 个 512B 单元；head h 占 k-block `[h*4, h*4+4)`，即**字节偏移 `h * 16384`，head 内连续**。
- K scale：`Dn2Nz(dValue = 128, nValue = dstNzC0Stride = (D/64)*n2Size = 8, srcDValue = 8)`
  → `[128/16 = 8][8]` 个 32B 单元；head h 字节偏移 `h * 128`，消费侧 `srcStride = 8`。
- V scale：`Nd2Nz(nValue = dstNzC0Stride = kvTokenBlock/64 = 2, dValue = D*n2Size = 512, srcDValue = 512)`
  → `[512/16 = 32][2]` 个 32B 单元；head h 字节偏移 `h * 1024`，消费侧 `srcStride = 2` 不变。

消费侧（`LoadK2L0` / `LoadV2L0`）：

- payload 基址 `+= headIdx * (kvTokenBlock * dBaseSize / 2)`，`srcStride = kvTokenBlock/16 = 8`，`mStartPosition = tokenOffset/16 = 0`；
- K scale 基址 `+= headIdx * (dBaseSize/64) * 32`，`srcStride = (dBaseSize/64) * n2Size = 8`；
- V scale 基址 `+= headIdx * (dVBaseSize/16) * (kvTokenBlock/64) * 32`，`srcStride = kvTokenBlock/64 = 2`，`yStartPosition = 0`。

page 不是 128 的整数倍时（page=64），按页分段：每段 `Nd2Nz(nValue = 该页 token 数, dstNzC0Stride = kvTokenBlock)`，目的基址加 `(tokenOffset/16) * 512B`。

### B7.9 Q scale 寻址修正（已被实测推翻，见 B7.12）

Q scale 的 GM 布局是 `N2TGD = [n2Size=2, T, gSize=8, D/64, 2]`，而合并后 `realN2Size=1 / realGSize=16`，现有公式（`block_cube_mxfp4.h` L941-943）不再成立。

真实地址：`(token t, q head h)` → `pairOffset = ((h/8)*T_total + t)*8 + (h%8)`，再乘 `D/64`。
合并后 M 行为 token-major/head-minor，故每个 token 的 16 行 scale 由 **两段各 8 行**组成。原设计拟按 token 循环、每 token 两次 `Dn2Nz`（`dValue = 8`、`nValue = D/64`），目的地址偏移 16B 写入 lane 8..15。

**该做法在硬件上非法**：MTE2 目的地址必须 32B 对齐，半 fractal 写入触发 AI Core error 80。详见 B7.12。

### B7.10 块对角 mask

`vf/vf_softmax_fp32_mxfp4.h` 的 ND 分支（L676-724）：

- `SoftmaxFp32Mxfp4Params` 新增 `mergeHeadIdx`（当前 chunk 的 KV head）与 `realGroupSize`（真实 gSize=8）；
- `params.kvStart = runInfo.s2Idx`（两个 chunk 覆盖**同一段 token**，不再加 chunkOffset）；
- 行 `r` 的 KV head = `((qGs1Start + r) % qGroupSize) / realGroupSize`，`qGroupSize = realGSize = 16`；
- 因 `gSize % 4 == 0`，VF 的 4 行一组内 head 相同，可用标量判定：不匹配时把该 4 行的 `visible` 换成一个全 false 的 `MaskReg`（chunk 开头用 `Compares<LT>(none, chunkIndices, 0)` 构造一次），其余逻辑（`Select` 置 `-FLT_MAX`、组 max、exp、量化）完全不变；
- 尾块 `validKv` 与 causal 判定保持原样，两个 head 共用同一 `validKvTokens`。

全 `-inf` 行由既有 `INVALID_GROUP_EXPONENT` 路径处理，与 causal 空行一致。

### B7.11 风险与验证顺序

1. Config 6 的 `PRELOAD_N = 20` C1/C2 并发路径曾在 M32 上死锁；合并后单核 task 数减半，需先用小形状冒烟确认不挂死。
2. 先跑 ND 冒烟 + CPU golden 单例（`bench16k_decode_3`，M=16 尾块），再跑 `bench16k` 12 例精度，最后跑性能。
3. 回归：`document small`（N_kv=1，不命中合并）与 PA_BNBD / PA_NZ 必须与基线逐位一致。

### B7.12 实测约束与 part-major 修正

**约束（NPU 7 实测）**：GM→L1 的 MX scale 搬运只能以完整 16 行 fractal 为单位落地，目的地址必须 32B 对齐。

| 尝试 | 做法 | 实测结果 |
| --- | --- | --- |
| merge2 | 每 token 两次 `Dn2Nz(dValue=8)`，第二次目的偏移 16B | AI Core error 80 `The MTE non-atomic instruction address is not aligned`（全部 AIC） |
| merge3 | `dnNum=2`、`srcDnMatrixStride=T*32`、`dstNzMatrixStride=16B` | aicore timeout |

因此一个 scale fractal 的 16 行必须来自 GM 中**步长一致的一段连续行**。在 `N2TGD` 布局下，这样的 16 行只能是「同一 KV head、相邻 2 个 token、各 8 个 Q head」。

**修正：L1 Q tile 采用 part-major 行序。** part p（= KV head p）占 `[p*mBaseSize/2, (p+1)*mBaseSize/2)` 行，内部为 `[token][gSize 个 head]`：

- payload：每 part 一次 `Nd2Nz(ndNum=seqQ, nValue=gSize, dValue=D/2, srcDValue=D/2, srcNdMatrixStride=N_q*D/2, dstNzC0Stride=actMSizeAlign32, dstNzMatrixStride=gSize*32)`，目的基址 `p*(mBaseSize/2)*32`；
- scale：每 part 一次 `Dn2Nz(dnNum=1, dValue=seqQ*gSize, nValue=dstNzC0Stride=D/64, srcDValue=D/64)`，目的基址 `p*(mBaseSize/2/16)*(D/64)*32`（32B 对齐）；
- `actMSizeAlign32` 在合并模式下固定为 `mBaseSize`，尾部无效行不参与输出；
- `LoadQ2L0(partIdx)` 用 `mStartPosition = xStartPosition = partIdx*(mBaseSize/2)/16`、`mStep = (mBaseSize/2)/16` 取本 part。

**QK 与 score 行序**：每个 chunk 只计算本 KV head 的 Q part（`Mmad m = mBaseSize/2`，与基线单 head 任务同规模，不增加 Mmad 量）。L0C 为 part-major，需要写回自然行序的 score UB，原设计用逐 token 的 `Fixpipe(mSize=gSize, dualDstCtl=0, subBlockId=...)`。

`merge4`（逐 token Fixpipe）实测 `bench16k_decode_3` aicore timeout，AIC/AIV 全部停在同一 PC。`merge5` 把该 Fixpipe 换成已验证的单次 dualDst Fixpipe 以隔离根因，结果见测试报告第 6 节。

若逐 token Fixpipe 确为根因，备选方案是**全链路 part-major**：score/P/PV/输出/LSE 全部按 part-major 行序，每个 AIV 恰好独占一个 part（`cap = mBaseSize/2`），此时 `CopyProbabilityToL1` 与 `UpdatePScale` 无需改动（各 AIV 写整段连续 fractal），只需改 `WriteAttentionResult`/LSE 的行映射，并对合并路径关闭 FlashDecode（FD 的 workspace 行映射依赖自然行序）。

### B7.13 真正的挂死根因：AIV 侧编译期开关取自 Dummy

`quant_flash_attn.cpp` 在 AIV 侧实例化 `QuantFlashAttnKernelMxfp4<CubeBlockDummy, VecFaBlock, VecFdBlock>`。
若 kernel 写 `MERGE_KV_HEADS = CubeBlockType::MERGE_KV_HEADS`，AIV 取到的恒是 Dummy 的 `false`：

| 侧 | realN2Size | realGSize | kvTokenBlock | 每 task chunk 数 |
| --- | ---: | ---: | ---: | ---: |
| AIC | 1 | 16 | 128 | 2 |
| AIV（错误） | 2 | 8 | 256 | 1 |

两侧任务枚举与 score 槽数不同 → 跨核 flag 失衡 → 立即死锁。merge4/5/6 三轮隔离（改 Fixpipe、改 KV 搬运）都无效，正是因为根因不在 cube 侧。

**正确做法**：cube 与 cube Dummy 都导出 `KV_LAYOUT_TYPE`，kernel 用自身模板常量推导
`MERGE_KV_HEADS = M64_CONFIG && !USE_DN && CubeBlockType::KV_LAYOUT_TYPE == 1`。
修复后 `decode_3` 精度通过，逐 token Fixpipe 与整块连续 KV 搬运本身都是正确的。

### B7.14 性能证伪与后续方向

合并后 GM payload+scale 字节与基线完全相同（每 page 69632 B），但 M 从 32 抬到 64、N 从 128 抬到 256，一半 score 列被块对角 mask 恒置 0：L0C→UB 的 score 字节、VF 行数、P 规模、PV Mmad 全部翻倍。事务效率收益不足以抵消翻倍的片上开销。

**更根本的原因（见测试报告第 10 节的双侧 msprof）**：PA_BBND 下 C4 单 head 行只有 128 B，硬件按 256 B 整行取回，另一半被丢弃。实测 C4 与 C8 的 `read_main_memory_datas` 几乎完全相同（decode_1 5324.8 vs 5328.1 KB，decode_2 14255.1 vs 14254.5 KB）。**因此「让同页两个 head 只读一次」并不能减少任何 DRAM 字节**——读的仍是同样的 256 B 行。这否定了本章方案，也否定了「合并 + 跳过被 mask 的行」的变体。

后续方向见测试报告 10.4：算子内唯一可能减少 DRAM 字节的是提高 L2 复用（两 head 访问时间相邻化）；加深 KV 预取只对 PA_NZ / PA_BNBD 有空间；改用 GM 连续的 KV 布局属上层决策。

> **本节结论已被推翻（2026-11-25）。** 上面的推理漏掉了一点：C4 与 C8 读同样多的 DRAM 字节，恰恰说明 C4 有一半取回被浪费；把两个 head 合进一个 task 后，同样的 256 B 行只被取回一次而不是两次。修复流水与标量缺陷后实测 `decode_1` 的 DRAM 读量从 **5325 KB 降到 2802 KB**，MTE2 不再是瓶颈，decode C8/C4 从 1.0875 升到 1.3918。当时的「净退化」观测来自 `GROUP_N=20` 违反流水约束与 VF 行循环内的整数除法这两个独立缺陷，不是方案本身的问题。当前生效设计见 A2–A4。

## B8. 兼容性设计

- 原 M128 的 buffer 尺寸、QK/PV 分段和 16-task 组边界不变；
- DN/C8 计算逻辑不被 host 选择到小 M decode 配置；
- 新配置不改变 FP4/E8M0 编码、Softmax scale、P 量化 schema、LSE 和 FD 合并语义；
- PA_BBND、PA_BNBD、PA_NZ 共用现有 GM→L1 布局转换；
- 输出复用 PV 的优化仅保留给旧配置，小 M 配置使用独立输出暂存区，避免超过 UB 容量；
- Config 6/7 与第 B7 章的块对角合并方案均保留实现但当前不派发；host tiling 与 AICPU metadata 的 `mergeKvHeads` 开关同时置 false，ND decode D=Dv=256 走基线 Config 5。合并路径只改变 GM→L1 的读取方式、L1 槽内布局与 M/N 分块，不改变 FP4/E8M0 编码与 softmax 语义。

> **本节的「不派发」状态已过期。** Config 6 现已作为合并专用配置重新派发（条件见 A4.1），Config 7 仍不派发。

## B9. 验证口径

### 性能

执行 `test_quant_flash_attn_perf_rdv.py` 的 bench16k 12 用例：

```text
--run-perf --perf-backends c4,c8 --perf-mode graph
--perf-warmup 3 --perf-rounds 5 --perf-repeats 20
```

每轮记录单次 attention 调用耗时，取 5 轮中位数。基线和候选使用相同 case 指纹、输入 SHA256、设备和计时参数。重点观察 4 个 decode 用例及全部 12 用例的等权几何平均 C8/C4。

### 精度

按 `kernel-optimized.md` 的要求，执行：

- document small × PA_BBND/PA_BNBD/PA_NZ；
- bench16k × PA_BBND/PA_BNBD/PA_NZ；
- 三种 schema（0/1/2）的必要覆盖；
- 先进行 ND 冒烟和 CPU golden 对照，再运行完整矩阵；
- Attention 和 LSE 都检查。

`short11` 在 G=8 时有 88 个有效 Q 行，只覆盖 M64 回退；M32 必须使用 `short1/3/4` 或其他满足 `maxSeqQ*gSize<=32` 的用例验证。精度不通过时先修复寻址和边界，不以性能结果替代精度结论。

## B10. 测试环境辅助处理

当前容器缺少可用的 Triton Ascend Python 3.12 wheel。性能输入的 FP4→FP8/BF16 展开增加等价后备实现：

- 优先沿用原 Triton 路径；
- 无 Triton 时在 CPU 上按相同 E2M1 位级映射展开，再回传 NPU；
- 转换在计时区域之外；
- 已对全部 256 种输入字节编码验证 FP8 位模式和 BF16 数值完全一致。

该后备路径只影响测试输入准备，不参与 C4/C8 kernel 计算，也不改变两者的输入数值。

Python 测试环境必须隔离不同 vendor 包，避免既有 `cann_ops_transformer_c4a4` 的 entry point 覆盖候选实现。运行时同时检查包路径、`ASCEND_CUSTOM_OPP_PATH` 和包内 kernel JSON；缺少新增 tiling key 时不进入性能对比。

## B11. 最终生效设计：块对角合并 + 两个流水/标量缺陷修复

对应测试报告第 11 章。本节描述**当前代码实际派发**的设计。

### B11.1 缺陷一：组大小必须满足 `GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N`

C1/C2 流水中：

- AIV 在 `tick = 组内最后一个 task + DELAY_P_SCALE_N` 发布该组 Pscale；
- Cube 在 `tick = 组内第一个 task + PRELOAD_N` 的首个 C2 等待它，等待落在 **MTE1** 流水，会连带挡住后续 Bmm1 的 L1→L0 搬运；
- AIV 在同一 tick 的 V1 又在等 Cube 该 tick 才能产出的 score。

组过大即形成互等。M128/M32 历史用 `GROUP_N=16`（16+3=19 ≤ 20）恰好成立，M64 曾用 20（23 > 20）必挂；单核 task 数不足 `PRELOAD_N` 时 C2 只在排空阶段运行，不与 C1 并发，故不暴露。

**修复**：主控 `GROUP_N`、Cube `PV_TASKS_PER_GROUP`、Vector `TASKS_PER_GROUP` 统一为
[quant_flash_attn_mxfp4_common.h](file:///d:/Project/GQA-C4A4-A5/gqa-c4a4-kernel/attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_mxfp4_common.h) 中的 `MXFP4_TASK_GROUP_N = 16`，并在主控加
`static_assert(GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N)`。

### B11.2 缺陷二：softmax VF 行循环内不得有整数除法

为让「跳过 head 不匹配的整段行」不在向量体内产生数据相关分支，行循环被改成单层扁平循环。若跨段进位写成 `spanOffset / rowSpan`（`rowSpan` 为运行时值），则**每个 4 行单元一次 32 位整数除法**，实测使整个 VF 慢约 1.9 倍，且该循环对所有路径生效，会污染任何以它为基础的对照实验。

**修复**（[vf_softmax_fp32_mxfp4.h](file:///d:/Project/GQA-C4A4-A5/gqa-c4a4-kernel/attention/quant_flash_attn/op_kernel/arch35/vf/vf_softmax_fp32_mxfp4.h)）：

```cpp
spanOffset += 4U;
const uint32_t carry = spanOffset >= rowSpan ? 1U : 0U;  // 比较，不是除法
spanOffset -= carry * rowSpan;
rowBase += carry * rowPeriod;
```

唯一的取模 `qRowPhase % rowPeriod` 提到块循环之外，每次 VF 调用只做一次；块内回绕同样用比较完成。

### B11.3 跳段而非置 -inf

行归属以 `qGroupSize = realGSize` 为周期，每周期内连续 `realGroupSize` 行同属一个 KV head。本 chunk 只处理属于该 head 的行段：

- 跳过的行 **P 保持预置 0、指数保持 INVALID**，与「置 -inf 后照算」等价（该组 Pscale 为 0），但省掉一半向量计算；
- 向量体内不保留任何数据相关分支与额外掩码——实测在向量体内加分支 + 额外 `MaskReg` 会让 VF 慢 3.5 倍。

### B11.4 分块形状与派发

| 项 | 值 | 理由 |
| --- | --- | --- |
| Config | 专用 6 = M64/N256/D256 | 与基线 Config 5 区分，让 kernel 编译期得到 `mergeKvHeads` |
| `kvTokenBlock` | `s2BaseSize/2 = 128` | 与基线每 head 的 token 块一致 → task 数减半 → V2/Pscale 总行数不变 |
| M 轴 | `seqQ * n2Size * gSize` | `realN2Size=1`、`realGSize=n2Size*gSize`，展开全部 Q head |
| L1 Q tile | part-major，每 part `mBaseSize/2 = 32` 行 | 32 是 16 的整数倍，满足 MX scale 只能整 16 行 fractal 落地的约束 |
| score 槽 | 每个 KV head 独占一槽、一次 VF 调用 | 与基线的 chunk 语义一致 |
| metadata | `kvHeadNum=1`，`param.s2BaseSize=128`，header 的 s2 字段写 256 | header 写 score 宽度以便下游区分合并与基线 |

派发条件（host tiling 与 AICPU metadata 完全一致）：
`layout_q_descale=N2TGD` + `PA_BBND` + `n2Size==2` + `D=Dv=256`。

编译期开关由 `quant_flash_attn.cpp` 依 `config` 计算后作为模板参数传入 Cube / Vector **及其 Dummy**，主控直接取 `CubeBlockType::MERGE_KV_HEADS`——两侧必须导出同一值，否则 AIC/AIV 调度不一致会立即死锁。

**曾评估并放弃的形状**：
- M128/N128（`kvTokenBlock=64`）：task 数与基线相同但每 task Q 行数翻倍，V2/Pscale 变成 2 倍工，decode_4 退化；
- M128/N256：UB 需求 273408 B 超出 248 KiB 预算，静态断言不通过。

### B11.5 效果与遗留

decode_1/2/3 的 C8/C4 达到 1.69 / 1.76 / 1.42，绝对耗时相对基线降低 16%～43%；decode 几何均值 1.39（基线 1.11）。三布局精度各 12 例全通过，DN prefill 路径未派发合并、无回归。

遗留：decode_4（单核约 24 task，刚越过 `PRELOAD_N`，且 `kvHeadNum` 减半使均衡粒度变粗）仍退化约 15%。下一步是**按单核 task 数分档回退基线**，但判据依赖真实 `seqused_kv` 分布，只有 AICPU metadata 侧可得，需要先把分档决策统一到 metadata 再驱动 host 选 tiling key。

> **2026-11-25 完整验证后的更新**：同轮基线对照下 decode_1/2/3/4 的 C8/C4 为 1.6999 / 1.7575 / 1.4064 / 0.8931，decode 几何均值 **1.3918**（基线 1.0875），PA_BBND 整体 C4 耗时 -8.02%。精度覆盖扩大到 744 例矩阵 + document full 4860 例，合计 5604 例全通过。另发现并修复了派发条件过宽的正确性缺陷（gSize 必须为 ≥4 的 2 的幂，见 A6.2）。分档仍未实现，原因见 B12。

## B12. 分档设计的两条硬约束（2026-11-25 实测确立）

对应测试报告第 12 章。本节记录做分档前必须满足的两个约束，供后续实现参考。

### B12.1 判据必须是二维的：task 数 + M 行

合并后单核 task 数少并不必然意味着净退化。反例 decode_3：

| 用例 | 合并 task/核 | 基线单 task M 行 | 合并比值 | 基线比值 |
| --- | ---: | ---: | ---: | ---: |
| decode_3 | **12.0** | **8** | **1.4211** | 1.1900 |
| decode_4 | 24.7 | 32 | 0.8945 | 1.0360 |

decode_3 的 task 数比 decode_4 更少却达标：`seqQ=1` 使 M 行仅 8，单 task 计算量极小、纯 DRAM 受限，C1/C2 爬坡的**绝对时间**也短。爬坡开销只在 M 行占满、计算足以与搬运竞争时才吃掉 DRAM 收益。

因此判据为：

```
派发合并 ⟺ (合并后单核 task 数 >= 2 * PRELOAD_N) 或 (基线单 task M 行 <= 16)
```

离线用四用例实测比值验证：一维阈值（≥40 或 ≥30）会把 decode_3 一起回退，几何均值反降至 **1.3833**；二维判据达 **1.4460**。

### B12.2 host tiling 不得解引用 GetData()

`gert::Tensor::GetData<T>()` 返回**设备地址**。host 侧只能用它做 null 判断（区分静态/动态编译），**不能读取内容**。违反此约定会在算子调用内段错误（`PYTEST_EXIT=139`），且只在进入该分支的布局上崩溃，容易误判为 kernel 问题。

推论：**分档所需的真实 `seqused_kv` 分布在 host 侧不可得**。而静态可见量（`B`、`max_seqlen_q`、`max_seqlen_kv`）不足以判档——长尾 batch 会把 `maxKV` 拉高（decode_4 均值 2618 而 maxKV 16409，偏斜 6.27），使其 task 数上界 152.0 与 decode_1 的 163.4 无法区分。

AICPU metadata 可读真实数据，但其决策写在设备内存中，host 读不到；metadata 的 shape 由调用方按 `16 + (aicNum+aivNum)*16*batch*numHeadsKv` 预分配，不随分档变化，也不能用作传信通道。而 host 与 metadata 的分块必须完全一致（否则 AIC/AIV 调度不匹配立即死锁）。

**故分档必须先改调度接口**，让 metadata 的分档决策能驱动 host 选 tiling key。可选方向：
1. 新增 host 可见的分档输入（如把档位作为标量属性传入）；
2. 约定 metadata 的 shape 编码档位，host 据 shape 反推；
3. 由上层框架在调用前完成分档并分别传入 `max_seqlen_kv` 等属性。

三者都属跨算子接口变更，需同步修改 metadata 算子、host tiling、调用方与 pytest golden。
