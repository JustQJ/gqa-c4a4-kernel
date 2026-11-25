# C4A4 小 M 分块优化设计

## 1. 目标与约束

优化对象为 `attention/quant_flash_attn` 的 arch35 MXFP4（C4A4）实现，目标平台为 Ascend 950PR，重点覆盖 ND decode 小 M 场景。

- 保持 PA_BBND、PA_BNBD、PA_NZ 和现有量化路径的精度要求不变；
- C8/C4 耗时比以同一用例、同一输入、同一设备、同一 Graph 计时口径比较，目标为 1.4～2.0 倍以上；
- 不把输入生成、格式转换和 metadata 生成计入 attention 耗时；
- 真实收益必须以宿主机 `k831` 容器内的独立构建和 NPU 实测为准。

## 2. 基线瓶颈

基线提交为 `5f2d913c3`。D256 ND 使用 M128/N128，QK 和 PV 都按 128 token 处理。

D256 下 QK 的 L0A/L0B/L0C 只能使用单槽，任务派发和 Pscale/V2 的管理成本较明显。减小 M 可以减少无效 Q 行计算；调整任务 N 和 QK 槽数则用于降低任务管理成本并改善 QK Mmad 与 Fixpipe 的重叠。

小 M 不会减少有效 KV token 数，也不保证降低 KV 搬运量。最终收益必须来自实际 kernel 执行效率，不能通过减少实际 KV token 获得虚假收益。

## 3. 配置设计与当前派发状态

**当前结论：小 M 方向实测未达目标，Config 6/7 的 kernel 实现保留在代码中，但 host tiling 与 AICPU metadata 均不再派发它们。** ND decode D=Dv=256 仍走基线 Config 5（M128/N128），与 `5f2d913c3` 行为一致。

### 3.1 Config 6：M64/N256/D256（保留，不派发）

新增 `Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 = 6`：

- AIC Q 行块为 32，经 CV_RATIO=2 后合并为 M64；
- task N 为 256，QK/PV 都按 128 token 分段；
- QK 仍为单槽，PV 使用双槽；
- 每组最多 20 个 task；
- 曾用于 `maxSeqQ*gSize>32` 的 ND decode D=Dv=256 场景，现已停止派发。

### 3.2 Config 7：M32/N128/D256（保留，不派发，存在已知死锁）

新增 `Config_S1Aligned32_S2Aligned128_DAligned256_DVAligned256 = 7`：

- 曾在 ND decode、D=Dv=256 且 `maxSeqQ*gSize<=32` 时选择，现已停止派发；
- AIC Q 行块为 16，经 CV_RATIO=2 后合并为 M32；
- task N 为 128，QK/PV 均按 128 token 分段；
- 保留代码为 QK 单槽 + PV128 双槽 + 40 个 L1 P/统计槽（最后一次隔离实验的状态）；
- 每组最多 16 个 task。

实验过程中先后试过 QK 双槽 + PV128 单槽、组大小 16、PV64 双槽、QK 单槽、P 槽 40，五种组合均在同一边界挂死，详见 3.3。

host tiling 与 AICPU metadata 曾使用完全相同的选档条件（`maxSeqQ*gSize<=32` 选 Config 7，否则 Config 6），现已移除该派发分支。Config 7 的 metadata 在 CV 合并前为 `mBase=16/taskN=128`，合并后为 `mBase=32/taskN=128`；Config 6 对应合并前 `32/256`、合并后 `64/256`。旧配置编号和其他量化模式保持不变。

### 3.3 M32 已知缺陷：C1/C2 同 tick 并发死锁

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

## 4. 内存预算

### 4.1 M128/N128/D256 基线

| 资源 | 近似占用 | 预算 |
| --- | ---: | ---: |
| L0A | 32 KiB | 64 KiB |
| L0B | 50 KiB | 64 KiB |
| L0C | 200 KiB | 256 KiB |
| L1 | 261 KiB | 512 KiB |
| Vector UB | 约 233～239 KiB | 248 KiB |

### 4.2 M64/N256/D256

| 资源 | 计算方式 | 近似占用 | 预算 |
| --- | ---: | ---: | ---: |
| L0A | QK-Q 8 KiB + 2×PV-P 8 KiB | 24 KiB | 64 KiB |
| L0B | QK-K 16 KiB + 2×PV-V 17 KiB | 50 KiB | 64 KiB |
| L0C | QK 32 KiB + PV 68 KiB | 100 KiB | 256 KiB |
| L1 | P 160 KiB + Pscale 12.5 KiB + Q 10 KiB + KV 68 KiB + padding | 约 252 KiB | 512 KiB |
| Vector UB | 独立输出暂存、Pscale、history、score 和 P 双槽 | 约 234 KiB | 248 KiB |

M64 的 PV 如果不按 128 分段，V 双槽会达到约 68 KiB，超过 L0B 的 64 KiB 预算。因此保留 128 token 的 PV 分段是正确性的必要条件。

### 4.3 M32/N128/D256

| 资源 | 计算方式 | 占用 | 预算 |
| --- | ---: | ---: | ---: |
| L0A | 2×QK-Q 4096B + 2×PV64-P 1024B | 10240B | 64 KiB |
| L0B | 2×QK-K 16384B + 2×PV64-V 8704B | 50176B | 64 KiB |
| L0C | 2×QK 16384B + PV 34816B | 67584B | 256 KiB |
| L1 | 20 组 P/Pscale + Q + 4 个 KV 槽 + padding | 119872B | 512 KiB |
| Vector UB | cross 33280B + persist 105472B + transient 38912B | 177664B | 248 KiB |

如果 M32 同时启用 QK 双槽和 PV128 双 V 槽，L0B 将达到 67584B，超过 65536B 上限；PV128 单槽又在第 20 个 task 后的 C1/C2 并发中发生等待。因此当前设计选择 PV64 双槽。

## 5. 寻址与组边界

M64 的 PV 第二段（chunkOffset=128）的源地址关系为：

- P：`kStartPosition=128/64=2`；
- Pscale：`yStartPosition=128/64=2`；
- V：`mStartPosition=128/16=8`；
- Vscale：`yStartPosition=128/64=2`。

M32 的 PV64 第二段使用同一寻址公式：`chunkOffset=64` 时 P/Pscale/Vscale 的起点为 1，V 的 `mStartPosition=64/16=4`。

源 stride 仍按完整任务 N 设置。L0C 仅在当前组首个 task 的首个 chunk 初始化，后续 chunk 和 task 全部累加，仅在组尾执行 Fixpipe。

M64 的 Cube、Vector、kernel 和 CPU/Triton golden 统一使用 `group_tasks=20`，M32 与旧 M128 配置使用 16。CPU 和 Triton golden 不得写死组大小，必须读取实际 schedule。

## 6. BBND 搬运优化：已淘汰的两个变体

### 6.1 依据

基线 `bench16k_decode_1` 的 `msprof op / Default` 显示 `aic_mte2_ratio=0.954`、`GM_to_L1_bw_usage_rate=30.9%`、`aic_cube_ratio=0.066`、`aic_cube_wait_ratio=0.939`。即 decode 完全由 GM→L1 搬运受限，且受限于事务效率而非带宽。

PA_BBND 的 GM 布局为 `[page][token][head][D/2]`，同一 token 的各 KV head 交织。只搬本 head 时，ND2NZ 的每行为「128B 有效 / 256B 源跨步」；`srcDValue != dValue`，无法合并为长 burst。

### 6.2 变体 A：payload 整页全 head（已淘汰）

把同一 token 的全部 KV head 一次连续读入同一个 L1 KV 槽，消费侧按 head 偏移取用；scale 保持单 head。
门控 `KvLayoutType == PA_BBND && n2Size <= 2 && blockSize == s2BaseSize`。

实测（M128 干净构建）decode C4 = 275.462 / 794.420 / 35.696 / 46.479，几何均值 **1.0889**，低于基线 1.1102。
原因：行连续带来的事务效率提升，被「同一 page 的两个 head 各有一个 task、各读一遍整页」造成的 GM 流量翻倍完全抵消。

### 6.3 变体 B：payload + scale 都整页全 head（已淘汰）

K scale 用 `Dn2Nz(nValue = D/64 * heads)`、V scale 用 `Nd2Nz(dValue = D * heads)` 使源行连续，消费侧按 head 偏移与放大的 `srcStride` 取用。

实测 decode C4 = 283.251 / 802.608 / 35.960 / 46.356，比变体 A 更差。说明 scale 的小跨步读并不是主要瓶颈，多搬的字节反而有害。

**共同结论：基线的 GM 字节数已经最小，任何「多读再挑」的搬运改写都不可能净收益。必须让同一 page 的两个 KV head 只被读一次。**

## 7. 同页双 KV head 块对角合并（历史轮次，最终形态见第 11 节）

**本节记录的是 merge2～merge8 轮次的中间结论，已被第 11 节推翻并取代。**
当时的判断是「净退化 + 单核 task 数 ≥ `PRELOAD_N=20` 时挂死」，因此 host tiling 与 AICPU metadata 停止派发。后续定位到两个独立缺陷（组大小违反流水约束、行循环内的整数除法），修复后本方案在长 KV 上取得 1.5～1.76 倍加速。**阅读本节时请以第 11 节为准**，以下保留设计推演与硬件约束，供后续参考。

### 7.1 核心思想

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

### 7.2 派发条件（tiling 与 metadata 必须完全一致）

```
quantMode == MXFP4(5)
&& decodeS1GMerge (layout_q_descale == "N2TGD")
&& layout_kv == PA_BBND && pageAttention
&& qkHeadDim == 256 && vHeadDim == 256
&& n2Size == 2
&& gSize % 4 == 0            // VF 按 4 行一组做块对角判定
```

metadata 侧看不到 `block_size`，因此**合并任务的 token 块固定为 128**（`kvTokenBlock = s2BaseSize / n2Size`），与 page 大小解耦；page 为 128/256/512 时 128 token 必落在单页内，page=64 时按页分段搬运。

### 7.3 metadata 改动

`attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp`，`ParamsInit` 内（约 L296-341）：

- 新增成员 `mergeKvHeads_`，按 7.2 判定（用 `numHeadsKv_`、`numHeadsQ_`、`headDim_`、`headDimV_`、`layoutKv_`、`layoutQDescale_`、`quantMode_`）；
- 命中时：`sOuterFactor = 32`（使 `mBaseSize_ = 32 * aiv/aic = 64`），`sInnerFactor = 128` 保持不变；
- 命中时把 `baseInfo.kvHeadNum = 1`（`numHeadsQ_` 不变），于是 `GetGroupSize() = N_q = 16`、`GetMSize() = seqQ*16`、`GetHeadNum() = 1`，调度天然变成「每 (batch, 128-token 块) 一个 task」；
- `KernelSplitMode` 仍为 `BN2_S1G_S2`；`param.mBaseSize = 64`、`param.s2BaseSize = 128`。

metadata header 的 `HEAD_M_BASE_SIZE_INDEX = 64`、`HEAD_S2_BASE_SIZE_INDEX = 128`（token 语义）。

### 7.4 tiling 改动

`op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp`：

- `SplitPolicy()`：命中 7.2 时 `sOuterFactor_ = 32`、`sInnerFactor_ = 128`；
- `UpdateTilingKeyConfig()`：新增分支 `sOuter == 64 && sInner == 128 && dSize == 256 && dVsize == 256 → Config 6`；
- tiling data 的 `n2Size / gSize` **保持真实值 2 / 8**（GM 寻址需要），不写假值。

### 7.5 kernel 侧编译期开关

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

### 7.6 token 块与 score 宽度解耦

新增 `kvTokenBlock = MERGE_KV_HEADS ? s2BaseSize / constInfo_.n2Size : s2BaseSize`（=128）。
以下位置把 `s2BaseSize` 换成 `kvTokenBlock`：

- `DispatchRows`：`task.s2Idx = s2 * kvTokenBlock`；`task.actSingleLoopS2Size = min(kvTokenBlock, actSeqLensKv_ - s2Idx)`（**每 head 的有效 token 数**）；
- `GetTaskDealMode`：`s2LoopTimes = ceil(actSeqLensKv_ / kvTokenBlock)`；
- `CalcCurS2StartEndNoSparse` / `WithSparse` 中所有 S2 块↔token 换算。

`task.isGroupLast` 仍按 `GROUP_N`（Config 6 为 20）个 task 计数，语义不变。

### 7.7 chunk 循环统一改写

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

### 7.8 L1 KV 槽内布局与消费侧偏移

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

### 7.9 Q scale 寻址修正（已被实测推翻，见 7.12）

Q scale 的 GM 布局是 `N2TGD = [n2Size=2, T, gSize=8, D/64, 2]`，而合并后 `realN2Size=1 / realGSize=16`，现有公式（`block_cube_mxfp4.h` L941-943）不再成立。

真实地址：`(token t, q head h)` → `pairOffset = ((h/8)*T_total + t)*8 + (h%8)`，再乘 `D/64`。
合并后 M 行为 token-major/head-minor，故每个 token 的 16 行 scale 由 **两段各 8 行**组成。原设计拟按 token 循环、每 token 两次 `Dn2Nz`（`dValue = 8`、`nValue = D/64`），目的地址偏移 16B 写入 lane 8..15。

**该做法在硬件上非法**：MTE2 目的地址必须 32B 对齐，半 fractal 写入触发 AI Core error 80。详见 7.12。

### 7.10 块对角 mask

`vf/vf_softmax_fp32_mxfp4.h` 的 ND 分支（L676-724）：

- `SoftmaxFp32Mxfp4Params` 新增 `mergeHeadIdx`（当前 chunk 的 KV head）与 `realGroupSize`（真实 gSize=8）；
- `params.kvStart = runInfo.s2Idx`（两个 chunk 覆盖**同一段 token**，不再加 chunkOffset）；
- 行 `r` 的 KV head = `((qGs1Start + r) % qGroupSize) / realGroupSize`，`qGroupSize = realGSize = 16`；
- 因 `gSize % 4 == 0`，VF 的 4 行一组内 head 相同，可用标量判定：不匹配时把该 4 行的 `visible` 换成一个全 false 的 `MaskReg`（chunk 开头用 `Compares<LT>(none, chunkIndices, 0)` 构造一次），其余逻辑（`Select` 置 `-FLT_MAX`、组 max、exp、量化）完全不变；
- 尾块 `validKv` 与 causal 判定保持原样，两个 head 共用同一 `validKvTokens`。

全 `-inf` 行由既有 `INVALID_GROUP_EXPONENT` 路径处理，与 causal 空行一致。

### 7.11 风险与验证顺序

1. Config 6 的 `PRELOAD_N = 20` C1/C2 并发路径曾在 M32 上死锁；合并后单核 task 数减半，需先用小形状冒烟确认不挂死。
2. 先跑 ND 冒烟 + CPU golden 单例（`bench16k_decode_3`，M=16 尾块），再跑 `bench16k` 12 例精度，最后跑性能。
3. 回归：`document small`（N_kv=1，不命中合并）与 PA_BNBD / PA_NZ 必须与基线逐位一致。

### 7.12 实测约束与 part-major 修正

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

### 7.13 真正的挂死根因：AIV 侧编译期开关取自 Dummy

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

### 7.14 性能证伪与后续方向

合并后 GM payload+scale 字节与基线完全相同（每 page 69632 B），但 M 从 32 抬到 64、N 从 128 抬到 256，一半 score 列被块对角 mask 恒置 0：L0C→UB 的 score 字节、VF 行数、P 规模、PV Mmad 全部翻倍。事务效率收益不足以抵消翻倍的片上开销。

**更根本的原因（见测试报告第 10 节的双侧 msprof）**：PA_BBND 下 C4 单 head 行只有 128 B，硬件按 256 B 整行取回，另一半被丢弃。实测 C4 与 C8 的 `read_main_memory_datas` 几乎完全相同（decode_1 5324.8 vs 5328.1 KB，decode_2 14255.1 vs 14254.5 KB）。**因此「让同页两个 head 只读一次」并不能减少任何 DRAM 字节**——读的仍是同样的 256 B 行。这否定了本章方案，也否定了「合并 + 跳过被 mask 的行」的变体。

后续方向见测试报告 10.4：算子内唯一可能减少 DRAM 字节的是提高 L2 复用（两 head 访问时间相邻化）；加深 KV 预取只对 PA_NZ / PA_BNBD 有空间；改用 GM 连续的 KV 布局属上层决策。

## 8. 兼容性设计

- 原 M128 的 buffer 尺寸、QK/PV 分段和 16-task 组边界不变；
- DN/C8 计算逻辑不被 host 选择到小 M decode 配置；
- 新配置不改变 FP4/E8M0 编码、Softmax scale、P 量化 schema、LSE 和 FD 合并语义；
- PA_BBND、PA_BNBD、PA_NZ 共用现有 GM→L1 布局转换；
- 输出复用 PV 的优化仅保留给旧配置，小 M 配置使用独立输出暂存区，避免超过 UB 容量；
- Config 6/7 与第 7 章的块对角合并方案均保留实现但当前不派发；host tiling 与 AICPU metadata 的 `mergeKvHeads` 开关同时置 false，ND decode D=Dv=256 走基线 Config 5。合并路径只改变 GM→L1 的读取方式、L1 槽内布局与 M/N 分块，不改变 FP4/E8M0 编码与 softmax 语义。

## 9. 验证口径

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

## 10. 测试环境辅助处理

当前容器缺少可用的 Triton Ascend Python 3.12 wheel。性能输入的 FP4→FP8/BF16 展开增加等价后备实现：

- 优先沿用原 Triton 路径；
- 无 Triton 时在 CPU 上按相同 E2M1 位级映射展开，再回传 NPU；
- 转换在计时区域之外；
- 已对全部 256 种输入字节编码验证 FP8 位模式和 BF16 数值完全一致。

该后备路径只影响测试输入准备，不参与 C4/C8 kernel 计算，也不改变两者的输入数值。

Python 测试环境必须隔离不同 vendor 包，避免既有 `cann_ops_transformer_c4a4` 的 entry point 覆盖候选实现。运行时同时检查包路径、`ASCEND_CUSTOM_OPP_PATH` 和包内 kernel JSON；缺少新增 tiling key 时不进入性能对比。

## 11. 最终生效设计：块对角合并 + 两个流水/标量缺陷修复

对应测试报告第 11 章。本节描述**当前代码实际派发**的设计。

### 11.1 缺陷一：组大小必须满足 `GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N`

C1/C2 流水中：

- AIV 在 `tick = 组内最后一个 task + DELAY_P_SCALE_N` 发布该组 Pscale；
- Cube 在 `tick = 组内第一个 task + PRELOAD_N` 的首个 C2 等待它，等待落在 **MTE1** 流水，会连带挡住后续 Bmm1 的 L1→L0 搬运；
- AIV 在同一 tick 的 V1 又在等 Cube 该 tick 才能产出的 score。

组过大即形成互等。M128/M32 历史用 `GROUP_N=16`（16+3=19 ≤ 20）恰好成立，M64 曾用 20（23 > 20）必挂；单核 task 数不足 `PRELOAD_N` 时 C2 只在排空阶段运行，不与 C1 并发，故不暴露。

**修复**：主控 `GROUP_N`、Cube `PV_TASKS_PER_GROUP`、Vector `TASKS_PER_GROUP` 统一为
[quant_flash_attn_mxfp4_common.h](file:///d:/Project/GQA-C4A4-A5/gqa-c4a4-kernel/attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_mxfp4_common.h) 中的 `MXFP4_TASK_GROUP_N = 16`，并在主控加
`static_assert(GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N)`。

### 11.2 缺陷二：softmax VF 行循环内不得有整数除法

为让「跳过 head 不匹配的整段行」不在向量体内产生数据相关分支，行循环被改成单层扁平循环。若跨段进位写成 `spanOffset / rowSpan`（`rowSpan` 为运行时值），则**每个 4 行单元一次 32 位整数除法**，实测使整个 VF 慢约 1.9 倍，且该循环对所有路径生效，会污染任何以它为基础的对照实验。

**修复**（[vf_softmax_fp32_mxfp4.h](file:///d:/Project/GQA-C4A4-A5/gqa-c4a4-kernel/attention/quant_flash_attn/op_kernel/arch35/vf/vf_softmax_fp32_mxfp4.h)）：

```cpp
spanOffset += 4U;
const uint32_t carry = spanOffset >= rowSpan ? 1U : 0U;  // 比较，不是除法
spanOffset -= carry * rowSpan;
rowBase += carry * rowPeriod;
```

唯一的取模 `qRowPhase % rowPeriod` 提到块循环之外，每次 VF 调用只做一次；块内回绕同样用比较完成。

### 11.3 跳段而非置 -inf

行归属以 `qGroupSize = realGSize` 为周期，每周期内连续 `realGroupSize` 行同属一个 KV head。本 chunk 只处理属于该 head 的行段：

- 跳过的行 **P 保持预置 0、指数保持 INVALID**，与「置 -inf 后照算」等价（该组 Pscale 为 0），但省掉一半向量计算；
- 向量体内不保留任何数据相关分支与额外掩码——实测在向量体内加分支 + 额外 `MaskReg` 会让 VF 慢 3.5 倍。

### 11.4 分块形状与派发

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

### 11.5 效果与遗留

decode_1/2/3 的 C8/C4 达到 1.69 / 1.76 / 1.42，绝对耗时相对基线降低 16%～43%；decode 几何均值 1.39（基线 1.11）。三布局精度各 12 例全通过，DN prefill 路径未派发合并、无回归。

遗留：decode_4（单核约 24 task，刚越过 `PRELOAD_N`，且 `kvHeadNum` 减半使均衡粒度变粗）仍退化约 15%。下一步是**按单核 task 数分档回退基线**，但判据依赖真实 `seqused_kv` 分布，只有 AICPU metadata 侧可得，需要先把分档决策统一到 metadata 再驱动 host 选 tiling key。
