# C4A4 Attention 性能优化测试报告

本报告记录 `attention/quant_flash_attn` arch35 MXFP4（C4A4）在 Ascend 950PR 上的优化实验、真实测量结果与结论。所有数据均来自宿主机 `ascend-950pr-223-2-2` 的 `k831` 容器、物理 NPU 7，未使用估算值替代实测。

## 1. 测试环境与统一口径

| 项 | 值 |
| --- | --- |
| 宿主机 | `ascend-950pr-223-2-2` |
| 容器 | `k831`（仅写入 `/home/k831`） |
| 设备 | 物理 NPU 7（Ascend950PR），测试前确认空闲、测试后确认无残留进程 |
| CANN | `/usr/local/Ascend/cann-9.2.1` |
| Python / Torch / Torch-NPU | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 |
| 基线提交 | `5f2d913c3` |
| 基线包 | `cann_ops_transformer_qfa_base_m128` |
| 用例集 | `bench16k` 12 例（6 prefill + 4 decode + 2 mixed） |
| 固定形状 | Nq=16、Nkv=2、D=Dv=256、page=128、mask=3、LSE 关闭、seed=421、schema=2 |
| KV 池 | 固定 31374 页，逻辑形状 `[31374, 128, 512]` |
| 计时 | Graph 回放 + NPU Event，warmup 3、rounds 5、repeats 20，取每轮单次调用耗时的中位数 |
| 计时范围 | 仅 attention 调用；输入生成、FP4→FP8/BF16 转换、metadata 生成均在计时外 |

C4 与 C8 使用同一 case 指纹、同一输入 SHA256、同一设备与同一计时参数。比值定义 `C8/C4`，大于 1 表示 C4 更快；分组按 case 等权取几何平均。

原始证据：`evidence/base/`（逐例 JSON、JUnit、日志、summary）。

## 2. 基线结果（M128/N128，PA_BBND）

| case | C4 (μs) | C8 (μs) | C8/C4 |
| --- | ---: | ---: | ---: |
| decode_1 | 282.411 | 313.510 | 1.1101 |
| decode_2 | 790.772 | 794.744 | 1.0050 |
| decode_3 | 32.527 | 41.181 | 1.2661 |
| decode_4 | 46.262 | 49.753 | 1.0755 |

| 范围 | case 数 | C8/C4 几何平均 |
| --- | ---: | ---: |
| all | 12 | 1.1064 |
| prefill | 6 | 1.1464 |
| decode | 4 | **1.1102** |
| mixed | 2 | 0.9876 |

目标为 decode 及整体达到 1.4～2.0，基线距目标有明显差距。要达到 1.4，四个 decode 的 C4 需分别降到 223.936 / 567.674 / 29.415 / 35.538 μs。

## 3. 小 M 分块实验（延续 M 分块缩小思路）

### 3.1 M64/N256（Config 6）三轮

| case | 基线 C4 | 轮1 QK128/PV128 G8 | 轮2 QK256 G20 | 轮3 QK128/PV128 G20 |
| --- | ---: | ---: | ---: | ---: |
| decode_1 | 282.411 | 277.042 | 278.293 | 279.444 |
| decode_2 | 790.772 | 791.956 | **2204.977** | 788.761 |
| decode_3 | 32.527 | 31.326 | 31.024 | 30.941 |
| decode_4 | 46.262 | 47.825 | 48.114 | 48.474 |

- 轮 2（QK 直接 N256 + 单 V 槽）使 `decode_2` 退化到 2.79 倍耗时，**已淘汰**。
- 轮 1 / 轮 3 相对基线在 ±5% 内波动，`decode_4` 反而回退，**没有稳定收益**。
- 三轮均通过 `qwen36_35b_TP2_short11_N2TGD_M0_LSE1` 的 Attention 与 LSE 单例精度。

结论：M64 不减少有效 KV token 与搬运量，只改变任务粒度；实测不足以支撑目标。

### 3.2 M32/N128（Config 7）：精度通过但存在并发死锁

真实命中 M32 的用例（`maxSeqQ*gSize<=32`）`short1 / short3 / short4`，Attention 与 LSE **3/3 通过**。
注意 `short11` 在 G=8 时有 88 个有效 Q 行，只会命中 M64 回退，不能用于验证 M32。

性能用例无法完成，边界如下（eager 与 graph 均复现）：

| 形状 | 单核最大 task 数 | 结果 |
| --- | ---: | --- |
| B5 / KV256 | 2 | 通过 |
| B1 / KV16384 | 10 | 通过 |
| B1 / KV32768 | 10 | 通过 |
| B4 / KV8192 | 19 | 通过 |
| B5 / KV8192 | 23 | **挂死** |
| B4 / KV16384 | 37 | **挂死** |

分界是单核 task 数是否达到 `PRELOAD_N=20`：`ExecuteTick` 在 `tick>=20` 时首次同时发射 C2(task t−20) 与 C1(task t)。已逐一排除且均无效：

| 隔离项 | 取值 | 结果 |
| --- | --- | --- |
| 每组 task 数 | 20 / 16 | 均挂死 |
| QK L0 槽数 | 2 / 1 | 均挂死 |
| PV 分段 | 128 单槽 / 64 双槽 / 128 双槽 | 均挂死 |
| L1 P 与统计槽数 | 20 / 40 | 均挂死 |

因此 M32 的 C1/C2 并发路径存在未修复缺陷。**Config 6/7 的 kernel 实现保留在代码中，但 host tiling 与 AICPU metadata 均已停止派发**，ND decode D256 回到基线 Config 5（M128/N128）。

原始证据：`evidence/candidate-iterations/`。

## 4. 瓶颈定位：BBND decode 由 GM→L1 搬运受限

对基线包 `bench16k_decode_1`（C4、PA_BBND、NPU 7）采集 `msprof op --aic-metrics=Default`，5 次抓取，任务耗时中位数 298.824 μs。逐核均值：

| 指标 | 值 | 含义 |
| --- | ---: | --- |
| `aic_mte2_ratio` | **0.954** | GM→L1 搬运流水几乎满负荷 |
| `GM_to_L1_bw_usage_rate` | **30.9%** | 但带宽只用到三成 |
| `aic_mte2_active_bw` | 55.3 GB/s | 有效带宽偏低 |
| `aic_cube_ratio` | 0.066 | Cube 基本空转 |
| `aic_cube_wait_ratio` | 0.939 | Cube 绝大部分时间在等数据 |
| `aic_mte1_ratio` / `wait_ratio` | 0.168 / 0.931 | L1→L0 同样被饿死 |
| `aic_scalar_ratio` | 0.363 | 标量开销次要 |
| `aiv_vec_ratio` | 0.420 | Vector 不是瓶颈 |

结论：decode 的瓶颈是 **MTE2 事务效率**，不是算力、不是 Vector、也不是 DRAM 带宽。PA_BBND 的 GM 按 token 交织各 KV head，只取单 head 时每行读为「128B 有效 / 256B 跨步」的小事务。

旁证：`kernel-optimized.md` 记录同一 kernel 在 GM 连续的 PA_NZ / PA_BNBD 上 decode 比值分别为 1.3573 / 1.3186，而 PA_BBND 仅 1.1060。

原始证据：`evidence/bbnd-decode1-c4/`、`evidence/bbnd-decode1-c4.log`。

## 5. BBND 整页全 head 连续搬运

### 5.1 payload-only 变体（scale 仍单 head）

改动只作用于 `KvLayoutType == PA_BBND && n2Size <= 2 && blockSize == s2BaseSize` 的整页任务：payload 的 `dValue = srcDValue = rowBytes * n2Size`，一次连续读入全部 KV head，消费侧按 head 偏移取用；scale 保持原单 head 路径。

首次测量曾得到 decode 几何均值 1.1434，但复核发现该轮安装包仍派发了 M32（Config 7），结果不可用。使用全新构建目录 `bbnd-clean-build` / `bbnd-clean-install` 重测，四个 decode 的 `metadata_header` 均为 `[1, 1, 128, 128, 28, 56]`（decode_3 为 `[1, 0, ...]`），确认回到 M128/N128 基线分块。

| case | 基线 C4 | payload-only C4 | C8 | C8/C4 | 相对基线 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_1 | 282.411 | 275.462 | 313.510 | 1.1381 | -2.46% |
| decode_2 | 790.772 | 794.420 | 794.744 | 1.0004 | +0.46% |
| decode_3 | 32.527 | 35.696 | 41.181 | 1.1537 | **+9.74%** |
| decode_4 | 46.262 | 46.479 | 49.753 | 1.0704 | +0.47% |

decode 几何均值 **1.0889**，低于基线 1.1102。

原因分析：整页读消除了 head 交织跨步，但每个 task 读入的 payload 字节翻倍（16 KiB → 32 KiB），而同一 page 的两个 head 各有一个 task，总 GM 流量变为基线的 2 倍。事务效率提升被字节翻倍抵消，`decode_3`（KV 短、page 数少、尾块占比高）因额外的无效 head 搬运而净退化。

结论：**payload 单独整页化不成立**。原始证据：`evidence/bbnd-payload-only/`。

### 5.2 payload + scale 全 head 变体

在 5.1 基础上把 scale 也整页化：K scale 用 `Dn2Nz(nValue = dstNzC0Stride = D/64 * heads, srcDValue = D/64 * heads)`，V scale 用 `Nd2Nz(dValue = srcDValue = D * heads)`，使 GM 源行同样连续；消费侧按 head 偏移并放大 `srcStride`。

| case | 基线 C4 | payload+scale C4 | C8 | C8/C4 | 相对基线 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_1 | 282.411 | 283.251 | 313.510 | 1.1068 | +0.30% |
| decode_2 | 790.772 | 802.608 | 794.744 | 0.9902 | +1.50% |
| decode_3 | 32.527 | 35.960 | 41.181 | 1.1452 | **+10.55%** |
| decode_4 | 46.262 | 46.356 | 49.753 | 1.0733 | +0.20% |

decode 几何均值 **1.0773**，比 5.1 的 1.0889 更差，也低于基线 1.1102。

结论：scale 的小跨步读不是主要瓶颈，把 scale 也翻倍搬运只会增加 GM 流量。**两个变体共同证明：基线的 GM 字节数已经最小，任何「多读再挑」的搬运改写都不可能净收益；必须让同一 page 的两个 KV head 只被读一次。**

原始证据：`evidence/bbnd-payload-only/`（含 payload+scale 轮次）。

## 6. 同页双 KV head 块对角合并（当前方向）

### 6.1 设计与已实现部分

一个 task 覆盖一个 128-token 块的**全部** KV head：N 轴 256 列 = `[head0 的 128 token; head1 的 128 token]`，softmax 前把「行所属 KV head ≠ 列所属 KV head」的整块置 `-inf`。GM 侧每 page 只读一次，源行宽 256B 连续（`srcDValue == dValue`），task 数减半。

派发条件（host tiling 与 AICPU metadata 完全一致）：`MXFP4 + N2TGD decode + PA_BBND + D=Dv=256 + Nkv=2 + gSize%4==0`；命中时 `sOuter=32`（CV 合并后 M64）、`sInner=128`、`baseInfo.kvHeadNum=1`，tiling key 走 Config 6。

已实现：metadata/tiling 派发、`kvTokenBlock` 与 score 宽度解耦、L1 KV 槽内按 head 连续分段的整块搬运（`CopyMergedKvBlock`，含按页分段与 V 尾块清零）、cube/vec 的 chunk 循环、VF 块对角 mask、CPU golden 与 schedule 的合并语义。

### 6.2 实测发现的硬件约束：MX scale 的 16 行 fractal 不可半写

合并后 M 轴为 `seqQ * N_q = 64` 行（token-major、head-minor），而 Q scale 的 GM 布局是 `N2TGD = [n2Size=2, T, gSize=8, D/64, 2]`。一个 16 行 scale fractal 需要「同一 token 的 16 个 Q head」，但它们在 GM 中分成相距 `T*32` pair 的两段各 8 行。

- 尝试 1：两次 `Dn2Nz(dValue=8)`，第二次目的地址偏移 16B（lane 8..15）。
  实测 **AI Core error 80：`The MTE non-atomic instruction address is not aligned`**（`merge2`，全部 AIC）。MTE2 目的地址必须 32B 对齐，半 fractal 写入非法。
- 尝试 2：`dnNum=2 + srcDnMatrixStride=T*32 + dstNzMatrixStride=16B`。
  实测 **aicore timeout**（`merge3`），矩阵 stride 同样不支持 16B。

结论：**GM→L1 的 MX scale 搬运只能以完整 16 行 fractal 为单位，且这 16 行必须是 GM 中步长一致的一段连续行。** 因此 L1 Q tile 只能采用 **part-major** 行序（part p = 一个 KV head 的 `seqQ*gSize` 行，GM 中每 token 一段连续 8 行）。

### 6.3 part-major 实现与挂死根因定位

为保留 score/P/PV/输出/LSE/FD 的自然行序，实现为：L1 Q tile 用 part-major（两次对齐的 `Dn2Nz`），QK 按 chunk 只算本 head 的 Q part（`Mmad m=32`，与基线单 head 任务同规模），再用逐 token 的 `Fixpipe(mSize=gSize, dualDstCtl=0, subBlockId=...)` 把结果写回自然行序。

`merge4` 实测 `bench16k_decode_3` **aicore timeout**（AIC 与 AIV 全部停在同一 PC）。逐项隔离：

| 轮次 | 隔离改动 | 结果 |
| --- | --- | --- |
| merge5 | 逐 token Fixpipe → 已验证的单次 dualDst Fixpipe | 仍 timeout |
| merge6 | 整块连续 KV 搬运 → 逐 head 复用已验证的单 head 搬运 | 仍 timeout |
| merge5 同包 | `document small`（N_kv=1，不命中合并）回归 | 55 例连续通过、无错误 |

**根因**：`quant_flash_attn.cpp` 在 AIV 侧实例化 `QuantFlashAttnKernelMxfp4<CubeBlockDummy, VecFaBlock, VecFdBlock>`，而 kernel 里把 `MERGE_KV_HEADS` 定义成 `CubeBlockType::MERGE_KV_HEADS`。Dummy 类该常量恒为 `false`，于是 **AIV 的 `realN2Size=2 / realGSize=8 / kvTokenBlock=256`，AIC 的 `realN2Size=1 / realGSize=16 / kvTokenBlock=128`**，两侧任务枚举与 score 槽数完全不同，必然跨核死锁。这也解释了为什么改动 cube 侧任何一处都无效。

修复：在 cube 与 cube Dummy 中都导出 `KV_LAYOUT_TYPE`，kernel 用自身模板常量推导 `MERGE_KV_HEADS = M64_CONFIG && !USE_DN && KV_LAYOUT_TYPE == 1`。

### 6.4 修复后的实测：精度可过，但性能净退化

`merge7`（修复 AIV 侧开关 + 整块连续 KV 搬运 + part-major Q tile + 逐 token Fixpipe）：

| 用例 | 结果 |
| --- | --- |
| `bench16k_decode_3` 精度（Attention + LSE，CPU golden） | **通过**（41.31 s） |
| `bench16k` 6 个 prefill 精度 | **通过**（DN 路径不命中合并） |
| `bench16k_decode_1` | aicore timeout，其后用例因设备进入错误态连带失败 |
| `bench16k_decode_4` 单例 | aicore timeout |

`decode_3` 是唯一能跑完的 decode 用例，其性能实测（graph、warmup 3 / rounds 5 / repeats 20、NPU 7）：

| 项 | 基线 Config 5 | 合并 Config 6 |
| --- | ---: | ---: |
| `metadata_header` | `[1, 0, 128, 128, 28, 56]` | `[1, 0, 64, 128, 28, 56]` |
| C4 (μs) | 32.527 | **89.240** |
| C8 (μs) | 41.181 | 41.308 |
| C8/C4 | 1.2661 | **0.4629** |

**C4 退化 2.74 倍**，比值远低于基线，方向证伪。

退化原因（与 5.1/5.2 的结论一致）：合并把 M 从 32 抬到 64、N 从 128 抬到 256，而其中一半的 score 列被块对角 mask 恒置 0。于是 L0C→UB 的 score 字节、VF 的行数、P 的规模、PV 的 Mmad 全部翻倍；GM payload+scale 字节数与基线完全相同（每 page 69632 B），只把「128B 有效 / 256B 跨步」换成「256B 连续」。事务效率的收益远不足以抵消翻倍的片上计算与搬运。

挂死边界与 M32 完全一致：**单核 task 数是否达到 `PRELOAD_N = 20`**。`decode_3` 约 11.7 task/核（通过），`decode_4` 约 24 task/核（挂死），`decode_1`/`decode_2` 更高（挂死）。该 C1/C2 并发缺陷未修复。

### 6.5 处置

**host tiling 与 AICPU metadata 均已停止派发合并方案**（`mergeKvHeads_ = false` / `mergeKvHeads = false`），ND decode D=Dv=256 回到基线 Config 5（M128/N128）。kernel 侧实现（`CopyMergedKvBlock`、part-major Q tile、`FixpipeMm1Merged`、VF 块对角 mask、CPU golden 合并语义）保留在代码中但不被派发，与 Config 6/7 的处置方式一致。

原始证据：`evidence/merge-evidence/`（merge2～merge7 逐轮日志、`results-merge7/` 逐例 JSON）。

## 7. 结论

1. **目标未达成。** 截至本轮，`bench16k` decode 的 C8/C4 几何均值仍为基线 **1.1102**，没有任何一轮改动稳定超过基线，距 1.4～2.0 的目标仍有明显差距。
2. **瓶颈已定位清楚**：PA_BBND decode 受限于 GM→L1 搬运。第 4 节初判为「MTE2 事务效率」；第 9/10 节的双侧 msprof 进一步定位为 **256 B 取回粒度浪费**——C4 单 head 行只有 128 B，另一半被丢弃，导致 C4 与 C8 的实际 DRAM 字节数相同（见 10.1）。这是比值上不去的结构性原因。
3. **四条优化路线全部实测淘汰**：
   - 小 M（M64/N256）三轮：±5% 波动，无稳定收益；
   - 小 M（M32/N128）：精度通过但单核 task ≥ 20 时 C1/C2 并发死锁；
   - BBND 整页全 head 搬运（payload-only / payload+scale）：GM 字节翻倍，几何均值 1.0889 / 1.0773，均低于基线；
   - 同页双 KV head 块对角合并：GM 字节不变但片上计算与 score/P/PV 规模翻倍，`decode_3` C4 退化 2.74 倍，且同样在 task ≥ 20 时挂死。
4. **本轮新增的两条硬约束**（对后续方案有直接指导意义）：
   - GM→L1 的 MX scale 只能整 16 行 fractal 落地，目的地址必须 32B 对齐；半 fractal（16B）写入触发 AI Core error 80，`dstNzMatrixStride=16B` 同样不可用。因此 `N2TGD` 的 Q scale 无法按「同 token 的 16 个 Q head」拼装，只能 part-major。
   - 派生常量必须由各自模板类推导：AIV 侧 `CubeBlockType` 是 Dummy，任何取自它的编译期开关都会让 AIC/AIV 调度不一致并立即死锁。
5. **后续方向建议**：见 10.4。在 PA_BBND 布局下，减少 DRAM 字节的唯一算子内路径是**提高 L2 复用**（让两个 head 的访问在时间上相邻），弹性外推上限 ≈1.7～1.8，但依赖 L2 行为、风险高。加深 KV 预取只对 NZ/BNBD 有空间且上限有限。改用 PA_NZ / PA_BNBD 存 KV 有实测支撑（1.3186 / 1.3573），但属上层决策。单纯缩小 M 或合并 head 都已被实测否定。
6. **代码当前状态**：ND decode D256 与基线 `5f2d913c3` 行为一致；Config 6/7 与合并方案仅保留实现、不派发。本轮未 commit / push。

## 8. 最终代码状态验证（merge8）

停止派发合并方案后，用全新构建目录 `merge8-build` / `merge8-install` 重新验证。

### 8.1 性能：回到基线分块

`bench16k` 12 例、C4+C8、graph、warmup 3 / rounds 5 / repeats 20、NPU 7、PA_BBND、schema 2。全部 12 例的 C4 `metadata_header` 前 6 项均为 `[*, *, 128, 128, 28, 56]`，确认回到 M128/N128。

| case | C4 (μs) | C8 (μs) | C8/C4 |
| --- | ---: | ---: | ---: |
| decode_1 | 279.852 | 311.991 | 1.1148 |
| decode_2 | 793.029 | 795.205 | 1.0027 |
| decode_3 | 35.070 | 41.434 | 1.1815 |
| decode_4 | 46.229 | 49.501 | 1.0708 |
| single_prefill_1 | 249.690 | 242.907 | 0.9728 |
| single_prefill_2 | 950.599 | 1067.812 | 1.1233 |
| single_prefill_3 | 1302.868 | 1559.777 | 1.1972 |
| dual_prefill_1 | 1231.840 | 1292.957 | 1.0496 |
| dual_prefill_2 | 1272.970 | 1484.902 | 1.1665 |
| dual_prefill_3 | 2274.837 | 2706.735 | 1.1899 |
| mixed_1 | 1335.693 | 1345.139 | 1.0071 |
| mixed_2 | 3806.227 | 3611.333 | 0.9488 |

| 范围 | case 数 | C8/C4 几何平均 | 基线 |
| --- | ---: | ---: | ---: |
| all | 12 | 1.0820 | 1.1064 |
| prefill | 6 | 1.1135 | 1.1464 |
| decode | 4 | **1.0905** | **1.1102** |
| mixed | 2 | 0.9775 | 0.9876 |

与基线的差异在运行间波动范围内（宿主机为多用户共享，`decode_3` 这类 30 μs 量级用例波动可达 ±8%），**没有引入性能回退，也没有取得收益**。

### 8.2 精度

| 集合 | 布局 | 结果 |
| --- | --- | --- |
| `bench16k` 12 例（Attention + LSE，CPU golden） | PA_BBND | **12 passed**（240.39 s） |
| `document --scope small` | PA_BBND | 连续 126 例通过、零失败，随后被本轮 30 min 运行上限截断（`PYTEST_EXIT=124`），未跑完全部用例 |
| `bench16k` 12 例 | PA_BNBD | **12 passed**（280.80 s） |
| `bench16k` 12 例 | PA_NZ | **12 passed**（292.75 s） |

`document small` 的 Decode/MTP 用例 KV 最长 262144，CPU golden 生成很慢，30 min 内只能覆盖前 126 例；这一轮是**回归抽查**而非完整集合，如实记录。

Triton 参考在本环境不可用，未纳入本轮；`--triton-reference` 未启用。schema 覆盖本轮固定为 2（`bench16k` 用例自带），schema 0/1 未跑。

原始证据：`evidence/merge-evidence/results-merge8/`。

## 9. 三种 KV 布局的 msprof 对照：缩小 M 是否有收益

为回答「PA_NZ / PA_BNBD 上缩小 M 是否有收益」，用**同一基线包** `cann_ops_transformer_qfa_base_m128`、同一用例 `bench16k_decode_1`、同一设备 NPU 7、同一采集参数（`msprof op --aic-metrics=Default`、eager、warmup 20 / repeat 5、prof-warmup 10 / launch-count 5）分别采集三种布局，逐核均值再对 5 次抓取取平均。

| 指标 | PA_BBND | PA_BNBD | PA_NZ |
| --- | ---: | ---: | ---: |
| `aic_time` (μs) | 286.509 | 182.743 | 176.290 |
| `aic_mte2_ratio` | **0.956** | **0.931** | **0.891** |
| `aic_mte2_wait_ratio` | 0.857 | 0.906 | 0.916 |
| `GM_to_L1_bw_usage_rate` (%) | **31.04** | **24.57** | **25.12** |
| `aic_mte2_instructions` | 1922 | 1922 | **979** |
| `read_main_memory_datas` (KB) | **5324.8** | 2735.3 | 2736.4 |
| `GM_to_L1_datas` (KB) | 15815.6 | 8044.9 | 8050.5 |
| `aic_main_mem_read_bw` (GB/s) | 50.52 | 39.99 | 40.89 |
| `aic_cube_ratio` | 0.066 | 0.103 | 0.107 |
| `aic_cube_wait_ratio` | 0.941 | 0.930 | 0.923 |
| `aic_mte1_ratio` / `wait` | 0.168 / 0.933 | 0.260 / 0.915 | 0.267 / 0.890 |
| `aic_scalar_ratio` | 0.360 | 0.568 | 0.579 |
| `aic_fixpipe_ratio` | 0.067 | 0.107 | 0.111 |
| `aiv_vec_ratio` | 0.422 | 0.656 | 0.661 |

### 9.1 三条结论

1. **NZ/BNBD 同样是 MTE2 受限，不是算力或 Vector 受限。** `mte2_ratio` 仍高达 0.89～0.93，`cube_ratio` 只有 0.10。缩小 M 不会减少 MTE2 的搬运量与事务数（MTE2 指令数由 KV 拷贝粒度决定，与 task 数无关），因此**预期收益很小**，与 BBND 上 M64 三轮 ±5% 的实测一致。
2. **三种布局都远未打满带宽**：`GM_to_L1_bw_usage_rate` 只有 25～31%，而 `mte2_wait_ratio` 高达 0.86～0.92。即 MTE2「忙但一直在等」，属**事务/延迟受限（memory-level parallelism 不足）**，不是带宽受限。真正的杠杆是**加深 KV 预取、增加在途 GM→L1 请求数**，而不是缩小 M。当前 `KV_PREFETCH_LEAD = 1`（每流 2 槽、共 4 槽），M128/N128/D256 的 L1 只用了约 261 KiB / 512 KiB，还有约 251 KiB 可用于加深预取。
3. **修正第 5/6 章的一处表述**：BBND 的「有效字节」确实已最小，但**实际 DRAM 流量是 NZ/BNBD 的约 1.95 倍**（5324.8 KB vs 2735.3 KB），因为 128B 有效 / 256B 跨步会整行取回。这解释了 BBND 为何慢 1.6 倍，也说明「让同页两个 head 只读一次」在长 KV 用例上仍有理论价值——本轮合并方案只在短 KV 的 `decode_3` 上测到净退化，长 KV 的 `decode_1/2` 因挂死未能测到，**不能据此断言合并方案在所有形状上都无收益**。

### 9.2 NZ 为什么最快

PA_NZ 的 GM 已是 NZ 分形，走直拷 `DataCopy(blockCount = D/64, blockLen = pageTokenCount)`，`mte2_instructions` 只有 979（BBND/BNBD 为 1922），单条指令搬运更大，因此同样字节数下耗时最低。这与「事务效率决定 decode 性能」的判断一致。

原始证据：`evidence/layout-prof-evidence/`（NZ / BNBD 逐核 CSV 与 `profile_summary.json`）、`evidence/bbnd-decode1-c4/`（BBND 对照）。

## 10. decode_1 / decode_2 的 C4+C8 双侧 msprof：BBND 行粒度是结构性天花板

第 9 节把瓶颈统一归为「在途请求不足」。补采 `bench16k_decode_1` / `bench16k_decode_2` 的 **C4 与 C8 双侧** msprof（同一基线包 `cann_ops_transformer_qfa_base_m128`、同一设备 NPU 7、PA_BBND、eager、warmup 20 / repeat 5、prof-warmup 10 / launch-count 5、5 次抓取逐核均值）后，该结论需要**分场景修正**，第 9.1 条第 3 点的「合并可省一半流量」推断被**证伪**。

### 10.1 决定性事实：C4 与 C8 的实际 DRAM 字节数相同

输入 shape 已核对：C4 的 K/V 为 `[31374, 128, 2, 128]`（128 B/token/head），C8 为 `[31374, 128, 2, 256]`（256 B/token/head），**逻辑字节 C8 是 C4 的 2 倍**。

| 指标 | d1-C4 | d1-C8 | d2-C4 | d2-C8 |
| --- | ---: | ---: | ---: | ---: |
| `aic_time` (μs) | 286.509 | 305.629 | 776.612 | 786.145 |
| **`read_main_memory_datas` (KB)** | **5324.8** | **5328.1** | **14255.1** | **14254.5** |
| `GM_to_L1_datas` (KB) | 15815.6 | 15788.3 | 42608.0 | 42579.1 |
| `aic_mte2_instructions` | 1922 | 1912 | 5107 | 5102 |
| `aic_mte2_ratio` | 0.956 | 0.965 | **0.984** | **0.987** |
| `aic_mte2_wait_ratio` | 0.857 | 0.690 | 0.886 | 0.713 |
| `GM_to_L1_bw_usage_rate` (%) | 31.04 | 29.12 | 31.36 | 31.21 |
| `aic_cube_ratio` | 0.066 | 0.119 | 0.065 | 0.125 |
| `aiv_vec_ratio` | 0.422 | 0.264 | 0.420 | 0.277 |
| `aic_fixpipe_ratio` | 0.067 | 0.098 | 0.066 | 0.102 |
| `aic_read_hit_rate` (%) | **11.13** | 0.04 | **4.29** | 0.015 |
| `aic_read_close_hit` | 14057 | 51 | 14188 | 48 |

两个用例上 C4 与 C8 的 `read_main_memory_datas` 差异都 < 0.1%，`GM_to_L1_datas` 与 `mte2_instructions` 也几乎一致。

机制：PA_BBND 同一 token 的两个 KV head 在 GM 中相邻。

- **C8**：单 head 行 256 B，正好填满一个 256 B 取回粒度 → 字节全部有效；
- **C4**：单 head 行 128 B，只占半个粒度 → **另一半取回的字节被丢弃**。

因此 **C4 在 PA_BBND 上「省一半字节」的理论优势被行粒度浪费完全吃掉**，这是 C8/C4 只有 1.00～1.11 的根本原因。C4 剩余的那点优势（d1 快 6.6%、d2 快 1.2%）来自更低的 cube / VF / Fixpipe 占用，而非带宽。

decode_1 还能有 1.11，是因为 L2 命中率达 11.13%（14057 次 close hit）吸收了部分重复行；decode_2 命中率掉到 4.29%，DRAM 直接顶到 C8 水平。对照 PA_NZ / PA_BNBD：C4 的行在 token 方向连续、粒度被填满，DRAM 只有 2735～2736 KB（BBND 的一半），耗时 176～183 μs vs 286 μs。

### 10.2 MTE2「忙 / 等 / 空闲」分解：加深预取的天花板

`aic_mte2_ratio` 统计「有指令在飞」的周期（含等待）；真正无指令可发的空闲时间是 `aic_time − aic_mte2_time`。

| 用例 | `aic_time` | MTE2 忙 | **MTE2 空闲** | `mte2_wait_ratio` | 搬运能力利用率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| d1 BBND C4 | 286.509 | 273.896 | 12.613 (4.4%) | 0.857 | 31.0% |
| d1 BBND C8 | 305.629 | 294.895 | 10.734 (3.5%) | 0.690 | 29.1% |
| d1 NZ C4 | 176.290 | 156.990 | 19.300 (10.9%) | 0.916 | 25.1% |
| d2 BBND C4 | 776.612 | 764.117 | **12.495 (1.6%)** | 0.886 | 31.4% |
| d2 BBND C8 | 786.145 | 775.572 | **10.573 (1.3%)** | 0.713 | 31.2% |

- **NZ / BNBD**：搬运能力只用到 25%，MTE2 空闲 10.9%，「在途请求不足（延迟/MLP 受限）」成立，加深预取有空间。
- **BBND decode_2**：MTE2 空闲仅 **1.6%**。即使消除全部等待、字节瞬间到达，也只能省 1.6%。**第 9.1 条第 2 点把「加深预取」当通用杠杆的说法在此收回**——它在 BBND 长 KV 上天花板极低。

### 10.3 对第 9.1 条第 3 点的证伪

第 9.1 条曾推断「让同页两个 head 只读一次 → DRAM 流量减半 → decode_1 C4 可到 180～200 μs、比值 1.6～1.75」。**该推断错误。**

那 2 倍 DRAM 流量是**单 head 读被硬件按 256 B 整行取回**造成的，不是「两个 task 各读一遍」造成的。把两个 head 合并成一次读，读的仍是同样的 256 B 行、同样的 DRAM 字节，**一个字节都省不下来**。这从 DRAM 层面解释了第 6 章块对角合并为何没有收益，也说明「合并 + 跳过被 mask 的行」同样无法减少 DRAM 流量——它只能省片上开销，而片上开销本就不是关键路径（`cube_ratio` 0.065）。

### 10.4 修正后的可行方向

能真正减少 BBND 的 DRAM 字节、又不改 KV cache 物理布局的，只有**提高 L2 复用**：让被整行白搬进来的另一半真的被用上，需要两个 head 的访问在时间上足够近。decode_1 已自发拿到 11.13%，decode_2 只有 4.29%，说明当前调度下 L2 容量/局部性不足。

收益上限可按实测弹性外推：BBND 与 NZ/BNBD 的字节比 1.95 → 耗时比 1.63，指数 `log(1.63)/log(1.95) ≈ 0.73`；若流量减半，耗时约 ×`0.5^0.73 ≈ 0.60`，decode_1 ≈ 172 μs（比值 ≈1.82）、decode_2 ≈ 466 μs（≈1.71）。**这是估计值，不是实测**，且完全依赖 L2 行为能否被调度改善。

| 方向 | 依据 | 判断 |
| --- | --- | --- |
| 缩小 M（M64/M32） | 既不减 DRAM 行流量，也不增在途请求；M32 另有并发死锁 | **否定**（已实测） |
| 整页多读再挑 / 块对角合并 | DRAM 字节不变，片上开销增加 | **否定**（已实测） |
| 合并 + 跳过 mask 行 | 同上，DRAM 字节仍不变 | **否定**（10.3 推论） |
| 加深 KV 预取 | NZ/BNBD 空闲 10.9% + 带宽 25%；BBND 空闲仅 1.6% | 仅 NZ/BNBD 有空间；BNBD→NZ 指令数减半只换 3.5%，上限有限 |
| 提高 L2 复用（head 访问相邻化） | L2 命中 4.3～11.1%，弹性外推上限 ≈1.7～1.8 | 唯一可能跨过 1.4 的算子内方向，风险高、依赖 L2 |
| 上层改用 PA_NZ / PA_BNBD 存 KV | DRAM 减半，实测 decode 比值 1.3186 / 1.3573 | 有实测支撑，但超出算子优化范围，需上层决策 |

**结论不变且更强：目标 1.4～2.0 未达成。在 PA_BBND 布局下，C4 的字节优势被硬件 256 B 取回粒度抹平，这是比值上不去的结构性原因；单靠 kernel 分块或预取调整无法跨越。**

原始证据：`evidence/layout-prof-evidence/{bbnd-decode1-c8,bbnd-decode2-c4,bbnd-decode2-c8,nz-decode1-c4,bnbd-decode1-c4}/`（各 5 次抓取的逐核 CSV + `profile_summary.json` + 运行日志），BBND decode_1 C4 对照见 `evidence/bbnd-decode1-c4/`。

> **本节 10.3 / 10.4 的推断已被第 11 章的实测证伪**：块对角合并确实把 DRAM 字节减半（decode_1 5325.03 → 2802.39 KB）。10.3 假设「两个 head 的整行取回会被彼此复用」，但实测 L2 命中只有 4～11%，两次取回绝大多数各自落到 DRAM，因此合并成一个 task 后字节真的省下来了。10.4 表格中「整页多读再挑 / 块对角合并」「合并 + 跳过 mask 行」两行的**否定理由（DRAM 字节不变）不成立**，请以第 11 章为准。

## 11. 修掉 task ≥ 20 的 C1/C2 挂死，并重新评估块对角合并

### 11.1 挂死根因：组大小与 C1/C2 流水深度冲突（已修复）

第 3.2 / 6.3 节把挂死记为「单核 task 数达到 `PRELOAD_N=20` 即挂」，但未能定位。真实根因是**组大小约束**：

- AIV 在 `tick = 组内最后一个 task + DELAY_P_SCALE_N` 才发布该组 Pscale；
- Cube 在 `tick = 组内第一个 task + PRELOAD_N` 的首个 C2 就要等这个 Pscale，且该等待落在 **MTE1** 流水上，会一并挡住后续 Bmm1 的 L1→L0 搬运；
- 而 AIV 在同一 tick 的 V1 又在等 Cube 该 tick 才能产出的 score。

因此必须满足 `GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N`。M128/M32 用 `GROUP_N=16`（16+3=19 ≤ 20）恰好成立，这解释了基线为何一直正常；**M64 用 `GROUP_N=20`（20+3=23 > 20）违反约束**，组内最后一个 task 的 Pscale 要到 `first+22` 才发布，而 Cube 在 `first+20` 已经开始等 → 两侧互等。单核 task 数不足 20 时 C2 只在排空阶段运行、不与 C1 并发，所以不暴露。

修复：把主控 `GROUP_N`、Cube `PV_TASKS_PER_GROUP`、Vector `TASKS_PER_GROUP` 统一为共享常量 `MXFP4_TASK_GROUP_N = 16`（[quant_flash_attn_mxfp4_common.h](file:///d:/Project/GQA-C4A4-A5/gqa-c4a4-kernel/attention/quant_flash_attn/op_kernel/arch35/quant_flash_attn_mxfp4_common.h)），并在主控加 `static_assert(GROUP_N + DELAY_P_SCALE_N <= PRELOAD_N)` 锁住该约束。

验证（NPU7，容器 k831）：

| 用例 | 修复前 | 修复后 |
| --- | --- | --- |
| decode_4（约 24 task/核） | 挂死，`timeout` 终止（`merge7-acc-decode4.log` PYTEST_EXIT=143） | **1 passed in 48.55s** |
| bench16k 全 12 例 × PA_BBND | 无法跑完 | **12 passed in 251.78s** |

### 11.2 合并重新派发后：DRAM 字节确实减半

派发条件（host tiling 与 AICPU metadata 一致）：ND decode（`layout_q_descale=N2TGD`）+ `PA_BBND` + `n2Size==2` + `D=Dv=256`。此时 `mBaseSize=64`、`s2BaseSize=256`、`kvTokenBlock=128`，`realN2Size=1`、`realGSize=n2Size*gSize`，一个 task 覆盖同页两个 KV head 的同一批 token。

msprof（eager，5 次抓取，逐核平均）：

| 指标 | decode_1 基线 | decode_1 合并 | decode_2 基线 | decode_2 合并 |
| --- | --- | --- | --- | --- |
| `read_main_memory_datas(KB)` | 5325.03 | **2802.39** | 14255.01 | **7333.67** |
| `aic_mte2_ratio` | 0.9569 | **0.4655** | 0.9839 | **0.4202** |
| `aic_main_mem_read_bw(GB/s)` | 51.04 | 27.42 | 51.00 | 26.80 |
| `aiv_vec_ratio` | 0.4246 | **0.8986** | 0.4203 | **0.9621** |

**DRAM 字节降到 0.52 倍，MTE2 不再是瓶颈**——第 10 章预期的机制成立。但瓶颈整体转移到 **AIV 向量**，且向量时间反而涨到约 2 倍，把 DRAM 收益全部吃掉。

### 11.3 合并路径向量开销的三轮修正

| 轮次 | 改动 | decode_1 AIV `vec_time` | decode_1 C4 graph |
| --- | --- | --- | --- |
| 基线 M128 | — | 126.3 μs | 282.411 μs |
| merge9 | 块对角 mask 置 -inf 后**照算**，向量体内含数据相关分支 + 额外掩码 | 974.7 μs | 1004.029 μs |
| merge10 | 标量边界**整段跳过** head 不匹配的行（双层循环） | 279.2 μs | 305.512 μs |
| merge11 | 改为**单层扁平循环**，跨段用整除进位，迭代次数与基线一致 | 251 μs | 279.908 μs |

去掉向量体内的数据相关分支与额外掩码带来 3.5 倍改善（974.7 → 279.2 μs），说明该写法会严重破坏向量流水。但此后仍稳定残留约 2 倍开销。

### 11.4 残留 2 倍的定位实验

**实验 A：段长/循环结构无关。** 手工形状 B32 / seqQ4 / KV12288 / D256 / page128 / mask3 / schema2，只改 `N_q`（`N_q=16` → gSize=8，跳段为 2×8 行；`N_q=32` → gSize=16，跳段为 1×16 行连续）。按「每 AIV 处理的 4 行×64 token 单元数」归一：

| 配置 | 单元数/核 | `vec_time` | 每单元耗时 |
| --- | --- | --- | --- |
| 基线 N_q=16 | 1755 | 116.9 μs | 0.0666 μs |
| 合并 N_q=16 | 1755 | 234.2 μs | 0.1334 μs（2.00×） |
| 基线 N_q=32 | 3510 | 197.0 μs | 0.0561 μs |
| 合并 N_q=32 | 3510 | 429.1 μs | 0.1223 μs（2.18×） |

合并与基线的 **VF 调用次数、每次调用的行单元数完全相同**，段长与循环形态却相差很大，倍率仍恒为 2.0～2.2。**「短内层循环导致流水建立开销」假设被否定。**

**实验 B：VF 占比与收益上限。** 临时构建 diag12（`DIAG_SKIP_MERGE_VF=true`，跳过合并路径的 softmax VF；**结果无效，仅用于计时**）：

| 配置 | AIV `vec_time` | AIV 总时间 | `aic_mte2_ratio` |
| --- | --- | --- | --- |
| merge11 手工 N_q=16 | 234.2 μs | 263.7 μs | 0.470 |
| diag12 手工 N_q=16 | **10.7 μs** | 171.6 μs | 0.918 |
| diag12 decode_1 | **11.3 μs** | 179.1 μs | 0.929 |

结论：合并路径的向量时间里 **223.5 / 234.2 μs 全部来自 softmax VF**，其余向量步骤（Pscale 更新、V2、FD）合计仅约 11 μs。且**一旦 VF 开销降到基线水平，合并路径 decode_1 约 179 μs、MTE2 重新饱和到 0.93**，对应比值 ≈ 313/179 ≈ **1.75**——与第 10.4 节按弹性外推的 1.7～1.8 一致。**目标区间在机制上可达，卡点已收敛到 softmax VF 内部。**

VF 体内唯一随 `TaskN`（合并 256 / 基线 128）变化的量是 `GROUPS_PER_TASK = TaskN/32`（8 / 4），它只出现在两处：行循环里写 `taskExponent[row*GPT+group]` 的 Scatter，以及尾部按 `rowOffsets = arange*GPT` 做的 4 次跨行 `Gather`。逐一隔离（手工 N_q=16，基线 116.9 μs / merge11 234.2 μs）：

| 诊断构建 | 改动（结果无效，仅计时） | `vec_time` | 相对 merge11 |
| --- | --- | --- | --- |
| diag13 | 去掉合并路径尾部 4 次跨行 Gather | 228.0 μs | −6.2 μs |
| diag14 | 去掉合并路径行循环内 4 次指数 Scatter | 225.0 μs | −9.2 μs |

**两者合计只解释 15 μs，Gather / Scatter 假设均被否定。**

**实验 C：关闭合并语义、只保留 M64/N256 分块。** diag15 让 host/metadata 对 `PA_NZ` 也派发 Config6（`sOuter=64, sInner=256`，`kvHeadNum` 保持 2）。此时 kernel 的 `MERGE_KV_HEADS=false`（NZ），VF 走**与基线完全相同的普通行循环**，无跳段、无进位除法，但 `TaskN` 仍为 256、每 task 仍 2 个 chunk：

| 配置 | VF 调用/核 | 行单元/核 | `vec_time` | 每单元 |
| --- | --- | --- | --- | --- |
| Config5 M128/N128（基线） | 219.4 | 1755 | 116.9 μs | 0.0666 μs |
| Config6 M64/N256，**不合并**（NZ） | 219.4 | 1755 | **215.5 μs** | 0.1228 μs（1.84×） |
| Config6 M64/N256，合并（BBND） | 219.4 | 1755 | 234.3 μs | 0.1335 μs（2.00×） |

**结论（后被 11.5 修正）：** 当时据此判断「1.84 倍来自小 M 分块形态」。**这个判断是错的**——diag15/diag16 虽然关闭了合并语义，但仍然走 merge10/11 引入的新行循环，而该循环里含一次整数除法（见 11.5）。两个对照被同一个自变量污染，因此不能用来区分 M/N 的影响。

### 11.5 真正的根因：行循环里的整数除法（已修复，长 KV 进入目标区间）

merge10/11 为了让「跳段」不产生向量体内分支，把行循环改成单层扁平循环，跨段进位写成：

```cpp
spanOffset += 4U;
const uint32_t carry = spanOffset / rowSpan;   // ← 每次行迭代一次 32 位整数除法
```

`rowSpan` 是运行时值，编译器无法强度削减，于是**每个 4 行单元都要做一次整数除法**。这条循环对**所有路径**生效（`MergeHeads=false` 时也走），所以：

- diag15（Config6/NZ，不合并）、diag16（Config7/M32，不合并）同样慢 ~1.9 倍——不是分块形态的问题；
- 11.4 中「小 M 导致 2 倍」的推断随之作废。

修复：进位改用比较得到 0/1，并把唯一的取模从块循环里提到每次 VF 调用只做一次：

```cpp
const uint32_t carry = spanOffset >= rowSpan ? 1U : 0U;
spanOffset -= carry * rowSpan;
rowBase += carry * rowPeriod;
```

同时把合并搬到**基线同形状 M128/N128**（专用 Config 6，形状与 Config 5 相同，仅用于让 kernel 编译期区分合并语义），`kvTokenBlock = s2BaseSize/2 = 64`：一个 task 覆盖 64 token × 2 head = 128 列，两个 head 共用一个 score 槽、**每 task 一次 VF 调用**，`TaskN=128`、`P_PACK_BYTES=4096`、`TASK_GROUP_MAX_ELEMENTS=256` 全部与基线一致。GM 侧每行仍是两 head 相邻的 256 B 连续行，DRAM 减半机制完整保留。

实测（手工 B32/Q4/KV12288，eager）：

| 构建 | 分块 / 合并 | 行循环 | AIV `vec_time` |
| --- | --- | --- | --- |
| merge8 基线 | M128/N128，不合并 | 原始 `row += 4` | 116.9 μs |
| merge17 | M128/N128，合并 | 含整数除法 | 291.4 μs |
| **merge18** | M128/N128，合并 | **比较进位，无除法** | **144.1 μs** |

graph 模式 decode 全量（NPU7，warmup3/rounds5/repeats20，schema2，PA_BBND）：

| 用例 | merge17 C4 | **merge18 C4** | C8 | **C8/C4** | 基线 C4 | 相对基线 |
| --- | --- | --- | --- | --- | --- | --- |
| decode_1 | 340.656 | **200.423** | 312.127 | **1.5573** | 282.411 | ×1.409 |
| decode_2 | 865.349 | **494.322** | 796.153 | **1.6106** | 790.772 | ×1.600 |
| decode_3 | 44.353 | **31.678** | 40.973 | 1.2934 | 32.527 | ×1.027 |
| decode_4 | 85.066 | **193.781** | 48.799 | 0.2518 | 46.262 | ×0.239 |

精度：bench16k 12 例 × PA_BBND **12 passed in 263.92s**。

**长 KV（decode_1 / decode_2）已进入目标区间 1.4～2.0**，且相对基线分别快 1.41 / 1.60 倍——第 10.4 节按弹性外推的 1.7～1.8 上限基本兑现。decode_3 追平基线。decode_4 首轮 graph 采样出现 193.781 μs 的异常值，单独重跑后为 53.170 μs（分轮 53.0～53.8，稳定），确认首轮是测量干扰而非算子行为。

### 11.7 最终方案与结果（merge19）

11.5 的 M128/N128 合并把 `kvTokenBlock` 降到 64，task 数与基线相同，但每 task 的 Q 行数翻倍，导致 **V2 / Pscale 更新的总行数变成基线的 2 倍**（V1 靠跳段抵消了，V2/Pscale 没有）。最终改回 **M64/N256**（`kvTokenBlock = 256/2 = 128`，与基线每 head 的 token 块一致）：

- task 数减半 → V2/Pscale 总行数与基线相同；
- M 轴展开全部 `N_q` 个 Q head（`realN2Size=1`、`realGSize=n2Size*gSize`），L1 Q tile 为 part-major，每 part 32 行（16 的整数倍，满足 MX scale 整 fractal 落地约束）；
- 每个 KV head 独占一个 score 槽与一次 VF 调用，`TaskN=256`；
- 行循环使用 11.5 的**无除法**跳段。

曾尝试 M128/N256 以同时保留大 M，但静态断言显示其 UB 需求 273408 B > 248 KiB 预算（score 双槽 65536 + PV 67584 + persist 134144 + transient 6144），故放弃。

**最终实测**（graph，NPU7，容器 k831，warmup3 / rounds5 / repeats20，bench16k decode，schema2，PA_BBND；基线为同机同轮重跑的 merge8）：

| 用例 | C4 (μs) | C8 (μs) | **C8/C4** | 基线 C4 | 相对基线 | 是否达标(1.4~2.0) |
| --- | --- | --- | --- | --- | --- | --- |
| decode_1 | **184.971** | 312.893 | **1.6916** | 278.743 | **×1.507** | 是 |
| decode_2 | **452.596** | 794.515 | **1.7555** | 793.662 | **×1.754** | 是 |
| decode_3 | **29.264** | 41.589 | **1.4211** | 34.795 | **×1.189** | 是 |
| decode_4 | 54.409 | 48.667 | 0.8945 | 46.563 | ×0.856 | 否 |
| **几何均值** | | | **1.3939** | | | 差 0.006 |

metadata_header 全部为 `[1, *, 64, 256, 28]`（合并配置），精度 bench16k 12 例 × PA_BBND **12 passed in 286.87s**。

同一轮重跑的基线（merge8，`evidence/ev19/merge8-decode-rerun/`）C8/C4 依次为 1.1176 / 0.9997 / 1.1900 / 1.0360，**几何均值 1.0833**；同轮对比下合并带来 1.3939 / 1.0833 = **+28.7%**。

**达标判定：4 个 decode 用例中 3 个（decode_1/2/3）进入目标区间 1.4～2.0，几何均值 1.3939，尚未整体跨过 1.4。** 相对第 2 章基线（几何均值 1.1102）提升 25.6%，长 KV 用例绝对耗时降低 34%～43%。

**decode_4 为什么仍退化：** 该用例 B33、seqQ4、KV 为 32 条 2048～2327 加 1 条 16409。合并后单核 task 数约 `673/28 ≈ 24`，刚好越过 `PRELOAD_N=20`——C1/C2 流水几乎全程处于「20 拍爬坡 + 4 拍稳态 + 20 拍排空」，爬坡占比远高于基线（约 48 task/核）。同时 `kvHeadNum` 由 2 降为 1 使负载均衡的可分配单元从 66 降到 33，短 batch 的打包粒度变粗。二者叠加抵消了 DRAM 减半的收益。decode_3（seqQ1，M 轴仅 16 行）计算量极小、纯 DRAM 受限，因此同样 task 数很少却仍然获益。

后续若要整体跨过 1.4，方向是**按单核 task 数分档**：task/核 处于 `PRELOAD_N` 附近（约 20～40）且 M 轴占满时回退基线分块。该判据依赖 KV 长度分布，只有 AICPU metadata 侧能拿到真实 `seqused_kv`，host tiling 在 graph 模式下只有 `max_seqlen_kv`（decode_4 的 16409 会误判），因此需要先把分档决策统一到 metadata 并让 host 依据同一信息选 tiling key，属于调度接口改动，本轮未做。

### 11.8 复测、完整精度矩阵与证据

**decode 复测**（同口径独立重跑一次，确认稳定性）：

| 用例 | C4 (μs) | C8 (μs) | C8/C4 | 相对基线 |
| --- | --- | --- | --- | --- |
| decode_1 | 183.928 | 312.985 | 1.7017 | ×1.515 |
| decode_2 | 451.222 | 794.322 | 1.7604 | ×1.759 |
| decode_3 | 29.288 | 41.256 | 1.4086 | ×1.188 |
| decode_4 | 55.983 | 49.004 | 0.8753 | ×0.832 |
| **几何均值** | | | **1.3863** | |

两轮几何均值 1.3939 / 1.3863，波动 <0.6%，结论稳定。

**精度矩阵**（merge19，`--accuracy-case-set bench16k`，func_rdv，NPU7）：

| KV 布局 | 结果 |
| --- | --- |
| PA_BBND（合并生效） | **12 passed in 286.87s** |
| PA_BNBD（合并未派发，走基线） | **12 passed in 292.94s** |
| PA_NZ（合并未派发，走基线） | **12 passed in 275.80s** |

**非 decode 用例无回归**（`--perf-kind all`，合并只在 ND decode 派发，DN prefill 路径不变）：

| 用例 | C4 (μs) | C8 (μs) | C8/C4 |
| --- | --- | --- | --- |
| dual_prefill_1 | 1233.839 | 1292.360 | 1.0474 |
| dual_prefill_2 | 1269.685 | 1481.325 | 1.1667 |
| dual_prefill_3 | 2292.606 | 2660.933 | 1.1607 |
| single_prefill_1 | 249.227 | 242.462 | 0.9729 |
| single_prefill_2 | 949.417 | 1068.246 | 1.1252 |
| single_prefill_3 | 1306.948 | 1546.693 | 1.1834 |

**原始证据**：`evidence/ev19/`
- `merge{9,10,11,17,18,19}-*.log`：各轮 configure / build / 精度 / 性能日志；
- `diag{12,13,14,15,16}.log`、`prof-manual.log`、`prof-heads.log`：隔离实验运行日志；
- `results/merge*-decode*/`、`results/merge19-all/`：逐用例 perf JSON（含 `samples_us`、`metadata_header`、`input_sha256`、`loaded_operator_libraries`）；
- `prof/`：15 组 msprof 抓取的逐核 CSV（PipeUtilization / Memory / MemoryUB / ArithmeticUtilization / ResourceConflictRatio）与 `profile_summary.json`。

### 11.9 中间轮次存档（merge11，已被 11.5 / 11.7 取代）

graph 模式、NPU7、warmup3 / rounds5 / repeats20、bench16k decode、schema2、PA_BBND：

| 用例 | C4 (μs) | C8 (μs) | C8/C4 | 基线 C4 | 相对基线 |
| --- | --- | --- | --- | --- | --- |
| decode_1 | 279.908 | 312.167 | 1.1153 | 282.411 | ×1.009 |
| decode_2 | 700.938 | 794.669 | 1.1337 | 790.772 | ×1.128 |
| decode_3 | 37.727 | 41.646 | 1.1039 | 32.527 | ×0.862 |
| decode_4 | 74.872 | 48.353 | 0.6458 | 46.262 | ×0.618 |
| **几何均值** | | | **0.9744** | | |

精度：bench16k 12 例 × PA_BBND **12 passed**（merge9 / merge10 / merge11 各自独立构建均通过）。

**判定：目标 1.4～2.0 仍未达成。** 长 KV（decode_1/2）已追平或略超基线，短 KV（decode_3/4）仍净退化——合并把 task 数减半，短 KV 下每核 task 数本就接近流水深度，减半后并行度与 FD 切分都变差。后续必须（a）消除 softmax VF 的 2 倍开销，(b) 按 KV 长度分档，短 KV 回退基线分块。

