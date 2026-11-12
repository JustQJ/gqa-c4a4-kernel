# C4A4 FP32 Softmax Kernel 实现说明

本文记录 2026-11-11 正式 arch35 实现，包含基线提交 `f8034b9db` 之后的可选预缩放/默认双目标改动（第 9 节）。资源数字来自当前分配公式，单位 **KiB = 1024 字节**；L1/L0 按每 AIC，UB 按每 AIV 统计。一个 AIC 对应两个 AIV，两个 AIV 的 UB 独立，不能把表内数字当成两核合计。接口和输入 shape 见 [API 说明](c4a4-api-softmax-fp32.md)。

## 1. 计算划分与主要文件

| 项目 | D128 | D256 |
|---|---|---|
| 完整 task（Q 行 × KV token） | M128 × N256 | M128 × N128 |
| QK 子块 | 两次 M128 × N128 | 两次 M128 × N64 |
| 一个 AIV 的最大 Q 行数 | 64 | 64 |
| QK/PV 累加 dtype | FP32 | FP32 |
| Q/K/V/P payload | FP4 E2M1，两个数一字节 | 同左 |
| 随路量化 scale | E8M0，32 元素一组 | 同左 |

DN 对应 `layout_q_descale=TND`，一个 Q head 独立调度；MM1 实际计算 `K·Qᵀ`，MM2 计算 `Vᵀ·Pᵀ`。ND 对应 `N2TGD`，合并 Q token/GQA 组的行轴，计算 `Q·Kᵀ`、`P·V`。DN/ND 是计算方向，不是公开 Q payload layout；公开 Q 始终为 TND。

| 文件 | 职责 |
|---|---|
| [quant_flash_attn.cpp](../op_kernel/quant_flash_attn.cpp) | 按 tiling key 选择 C4 模板、连接正式入口 |
| [kernel_mxfp4](../op_kernel/arch35/quant_flash_attn_kernel_mxfp4.h) | Metadata、task 环、跨 Q 调度、section/FD |
| [block_cube_mxfp4](../op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h) | GM/L1/L0 搬运、QK/PV、Fixpipe |
| [block_vec_mxfp4](../op_kernel/arch35/quant_flash_attn_block_vec_mxfp4.h) | V1、Pscale、V2、输出与组状态 |
| [mxfp4_common](../op_kernel/arch35/quant_flash_attn_mxfp4_common.h) | QuantSchema、CVCrossBuffer、共享地址与跨核 flag |
| [vf_softmax_fp32_mxfp4](../op_kernel/arch35/vf/vf_softmax_fp32_mxfp4.h) | DN/ND FP32 score、mask、P 量化 |
| [vf_update_fp32_mxfp4](../op_kernel/arch35/vf/vf_update_fp32_mxfp4.h) | 整组 Pscale、参考指数与 alpha |
| [vf_output_update_fp32_mxfp4](../op_kernel/arch35/vf/vf_output_update_fp32_mxfp4.h) | PV/分母累积、归一化、BF16 输出、LSE |
| [block_vec_flashdecode](../op_kernel/arch35/quant_flash_attn_block_vec_flashdecode.h)、[vf_flash_decode_mxfp4](../op_kernel/arch35/vf/vf_flash_decode_mxfp4.h) | FD workspace 归并与 C4 专用数值处理 |

## 2. 主循环和跨 Q 流水

`Process()` 初始化 Cube/Vector buffer 和事件后逐 section 执行。`FlashAttention()` 根据 Metadata 指定的 batch/head、Q 块和 S2 范围生成有效 task。跳过坐标不推进 task 编号；有效 task 使用 21 槽描述符环 `taskRunInfo[PRELOAD_N+1]`。

核心常量：`PRELOAD_N=20`、`GROUP_N=16`、`DELAY_P_SCALE_N=3`。每个流水 tick 的执行顺序为：

```text
1. 若已有第 t-20 个 task：
   AIC 执行 C2/PV；组末 AIV 执行 V2；随后退休该 task。
2. 若第 t-3 个 task 是组末：
   AIV 执行该组 UpdatePScale，发布 scale-ready。
3. 当前 task 有效时：
   AIC 执行 C1/QK；两个 AIV 分别执行 V1。
```

C2/V2 先退休、V1 后申请新槽，避免同 tick 先占用后释放造成碰撞。**Q 块边界不排空流水**；同 section 连续发射下一 Q 块，只有 section 末尾将 pending task 排空。组在 16 个 task 或当前 Q/本核 S2 分片结束时封口，因此可以出现单 task 短组。

不同“槽”对应不同生命周期：

| 编号 / 状态 | 槽数 | 选择方式和生命周期 |
|---|---|---|
| task 描述符 | 21 | 按 pipeline tick 轮转，退休后可重用 |
| L1 P/Pscale、UB task 指数 | 20 | `runInfo.loop % 20`，从 V1 保留至 C2 消费 |
| 组 reference/alpha/owner | 20 | `v1GroupId % 20`，从组首 V1 保留到组末 V2/输出结束 |
| score UB | 2 | 连续 `scoreSequence_ % 2`，每个 QK 子块轮转 |
| UB packed P | 2 | `runInfo.loop % 2`，MTE3 搬运完成后可重新写入 |
| MM1 L0A/L0B/L0C | 各 2 | 三者成对轮转；L0C 由本槽 FIX 完成信号保护 |

`GroupOwner` 记录真实组 id、首 task、task 数、是否当前 Q 首组，状态依次为 `FREE → ACCUMULATING → SEALED → READY → FREE`。`runInfo.v1GroupId` 在 V1 写入，Pscale 和 V2 读取同一记录，不能用 `loop/16` 推导短组的身份。`groupOwners_`、task 环和标量控制变量是控制结构，不是额外 `InitBuffer` 分配的 UB Tensor，下面业务 UB 表不把它们计作数据缓冲区。

Q payload 在 L1 单槽缓存，当前 Q 块结束后释放；Q 的专用 L0 双槽前两次装入后可复用。KV 的四槽为 K/V 共用的池，C1/C2 按调用顺序取槽，不是 K 四槽再加 V 四槽。

## 3. L1：每 AIC，512 KiB 预算

设 `M=128`、`N=s2BaseSize`、`D=Dv`。以下槽大小含代码显式分配的 scale/padding。

| buffer | 单槽公式（字节） | D128 单槽 × 槽数 = 合计 KiB | D256 单槽 × 槽数 = 合计 KiB | 作用 |
|---|---|---|---|---|
| `crossBuffers_.l1PBuffers_` | `M*N/2` | `16 × 20 = 320` | `8 × 20 = 160` | 20 task 的 FP4 P，两个 AIV 写各自 Q 行 |
| `crossBuffers_.l1PScaleBuffers_` | `(M/16)*(N/64+1)*32` | `1.25 × 20 = 25` | `0.75 × 20 = 15` | P 的 E8M0 scale；每 16 Q 行增加一组 scale padding |
| `l1QBuffers_` | `M*D/2 + M*D/32` | `8.5 × 1 = 8.5` | `17 × 1 = 17` | Q payload + Qscale |
| `l1KVBuffers_` | `N*D/2 + N*D/32` | `17 × 4 = 68` | `17 × 4 = 68` | K 或 V payload + scale |
| `l1VPaddingBuffers_` | `N*(D+16)/2 + N*(D+16)/32` | `19.125 × 1 = 19.125` | `18.0625 × 1 = 18.0625` | V 补列模板，为 Cube 计算 P 的行和提供常数 1 |
| **业务分配合计** | | **440.625** | **278.0625** | DN/ND 相同 |
| **距 512 KiB 余量** | | **71.375** | **233.9375** | |

常数 1 的编码为 FP4 payload `0x6`（4）乘 E8M0 `0x7D`（1/4）。Pscale 的额外 padding 与 V 的额外通道是两个不同用途的区域，不应合并计数。

L1 为 Cube 和配对 Vector 交换数据的地址空间。Vector 侧重复建立 L1 BufferManager/视图用于确定 P 地址，并不是在每个 AIV 再分配一套独立的 512 KiB L1。

## 4. L0A / L0B / L0C：每 AIC

L0A、L0B 各 64 KiB，L0C 为 256 KiB。表内为 `单槽 KiB × 槽数 = 总 KiB`，ND/DN 的左右操作数互换。L0A/B 的显式容量按 FP4 payload 公式计算；`LoadData(..., scale, ...)` 随路加载 MX scale，不存在额外显式申请的 `*ScaleBuffers_`。

| 存储 / buffer | D128 ND | D128 DN | D256 ND | D256 DN | 作用 |
|---|---|---|---|---|---|
| L0A `mm1LeftBuffers_` | `8×2=16` | `8×2=16` | `16×2=32` | `8×2=16` | QK 左操作数；ND=Q，DN=K |
| L0A `mm2LeftBuffers_` | `16×2=32` | `18×2=36` | `8×2=16` | `17×2=34` | PV 左操作数；ND=P，DN=V 含补列 |
| **L0A 合计 / 余量** | **48 / 16** | **52 / 12** | **48 / 16** | **50 / 14** | |
| L0B `mm1RightBuffers_` | `8×2=16` | `8×2=16` | `8×2=16` | `16×2=32` | QK 右操作数；ND=K，DN=Q |
| L0B `mm2RightBuffers_` | `18×2=36` | `16×2=32` | `17×2=34` | `8×2=16` | PV 右操作数；ND=V 含补列，DN=P |
| **L0B 合计 / 余量** | **52 / 12** | **48 / 16** | **50 / 14** | **48 / 16** | |
| L0C `mm1ResBuffers_` | `64×2=128` | `64×2=128` | `32×2=64` | `32×2=64` | QK FP32 子块，`128*s2SplitSize*4` 字节/槽 |
| L0C `mm2ResBuffers_` | `72×1=72` | `72×1=72` | `136×1=136` | `136×1=136` | `128*(D+16)*4`，组内 PV + 行和累积 |
| **L0C 合计 / 余量** | **200 / 56** | **200 / 56** | **200 / 56** | **200 / 56** | |

L0A 的左槽负责左右操作数的配对 `MTE1↔M` 事件；L0B 同步策略为 `NO_SYNC`，不重复申请事件。MM1 使用两槽 `FIX_M/M_FIX`；MM2 单槽在一组至多 16 个 task 中累积，仅组首初始化、组末 Fixpipe。双缓冲是允许不同流水交错的条件，不是实测重叠率的结论。

## 5. FA UB：每 AIV

以 248 KiB 为当前代码预算。score、PV 在 Cube/Vector 两侧建立相同 UB 地址视图，实际容量只在对应 AIV 的 UB 计一次。以下包含预留但当前未读取的索引表。

| buffer | 槽数 | D128 单槽 KiB | D256 单槽 KiB | 用途 |
|---|---:|---:|---:|---|
| `crossBuffers_.ubBmm1Buffers_` | 2 | 32 | 16 | FP32 QK 接收；ND `[64,s2SplitSize]`，DN `[s2SplitSize,64]` |
| `crossBuffers_.ubBmm2Buffers_`，ND | 1 | 34 | 66 | FP32 PV/行和，`[64,D+8]` |
| `crossBuffers_.ubBmm2Buffers_`，DN | 1 | 32.25 | 64.25 | FP32 PV/行和，`[D+1,64]`；与上一行二选一 |
| `historyOBuf_` | 1 | 32 | 64 | 未归一化累计 O；ND `[64,D]`，DN `[D,64]`，FP32 |
| `globalRowSum_` | 1 | 0.25 | 0.25 | 64 行增强分母，FP32 |
| `taskGroupMax_` | 20 | 2 | 1 | 每 task 的 S32 量化指数，`64*(N/32)` 个 FP32 |
| `globalCurrentMax_` | 20 | 0.25 | 0.25 | 每组每行 reference，64 个 FP32；数值语义为二进制参考指数 |
| `globalOldMax_` | 1 | 0.25 | 0.25 | 按顺序 Pscale 更新所用的上一组参考，FP32 |
| `alphaUpdate_` | 20 | 0.25 | 0.25 | 历史 O/分母重缩放系数，FP32 |
| `pNd2NzIdx_` | 1 | 0.25 | 0.25 | 历史 uint8 打包索引预留，当前固定 lane 打包不读取 |
| `pNd2NzHalfwordIdx_` | 1 | 0.25 | 0.25 | 历史 uint16 索引预留，当前不读取 |
| `pBuffers_[0/1]` | 2 | 8 | 4 | 独立 FP4 P，`64*N/2` 字节/槽；V1 计算与上一槽 MTE3 可交错 |
| `tempCommBuf_` | 1 | 38 | 38 | Pscale 编码、输出转换、统计，详见下表 |

| 汇总 KiB | D128 ND | D128 DN | D256 ND | D256 DN |
|---|---:|---:|---:|---:|
| score 两槽 + PV | 98 | 96.25 | 98 | 96.25 |
| 其他常驻区（含 P 双槽，不含 temp） | 99 | 99 | 103 | 103 |
| temp | 38 | 38 | 38 | 38 |
| **总计** | **235** | **233.25** | **239** | **237.25** |
| **距 248 KiB 余量** | **13** | **14.75** | **9** | **10.75** |

### 5.1 `tempCommBuf_` 内部视图

这些是同一个 38 KiB buffer 的视图，不增加 UB 总量。

| 阶段 / 视图 | 相对 byte offset | 容量 | 使用者 |
|---|---:|---|---|
| Pscale `encodedScale` | 0 | 最多 16 task：D128 10 KiB、D256 6 KiB | `UpdatePScale()`，编码并搬到 L1 |
| 输出 `convertedOutput` / DN FD `outputTile` | 0 | 预留 32 KiB；BF16 D128 实用 16 KiB、D256 32 KiB | `WriteAttentionResult()` → `processOutDn/Nd` |
| `lse` | 32768 | 2 KiB | 64 行，每行 8 个 FP32 的对齐空间；只搬 LSE 第 0 列 |
| `broadcastSum` | 34816 | 2 KiB | FD 的每行分母 |
| `broadcastReference` | 36864 | 2 KiB | FD 的每行二进制参考 |

输出转换是将 FP32 历史 O 按分母归一化并 Cast 为 BF16；DN 同时从 `[D,64]` 整理为 `[64,D]`。FD 分片输出保留原始 FP32 PV：ND 可直接从 history 搬出，DN 每 16 通道使用 temp 转排再搬出。P 双槽与 temp 独立，D256 输出不复用刚消费的 PV 接收区。

## 6. FD UB：与 FA 分时复用

FA section 排空后，FD 从 `GetFdReuseUbOffset()` 开始建立静态视图，复用**相邻的 PV 接收区 + history O**。D128 起点为 64 KiB，D256 为 32 KiB，即 score 两槽之后。不会 `TPipe::Reset()`，也不会新占用同等大小 UB。

| FD buffer | 槽数 | D128 单槽 KiB | D256 单槽 KiB | 作用 |
|---|---:|---:|---:|---|
| `fdSumBuf1_ / fdSumBuf2_` | 2 | 6 | 6 | 分片 sum 双缓冲 |
| `fdMaxBuf1_ / fdMaxBuf2_` | 2 | 6 | 6 | 分片二进制 reference 双缓冲 |
| `fdLseExpBuf_` | 1 | 6 | 6 | 各分片合并倍率：先暂存 `a_i`，再原地改写为 `a_i/S` |
| `fdMm2ResBuf1_ / fdMm2ResBuf2_` | 2 | 4 | 8 | 分片 FP32 O，`8*D*4` 字节/槽 |
| `fdReduceBuf_` | 1 | 4 | 8 | 已归一化的合并 FP32 O |
| `fdOutputBuf_` | 1 | 4 | 8 | BF16 输出转换与搬运暂存；不参与 PV 归并 |
| `fdLseMaxUbBuf1_ / fdLseMaxUbBuf2_` | 2 | 0.25 | 0.25 | 合并 reference 暂存 |
| `fdLseUbBuf_` | 1 | 0.25 | 0.25 | 最终 LSE |
| `fdMinCheckBuf_` | 1 | 0.25 | 0.25 | 共用布局保留，C8 用于 min-check；C4 不再使用 |
| **FD 视图总量** | | **47** | **63** | `5*6 + 4*(8*D*4/1024) + 1` |

可复用容量：D128 ND/DN 为 66/64.25 KiB，D256 为 130/128.25 KiB，均覆盖 FD 需求。FD 绝不能覆盖 score：AIC 可以提前开始下一 section 的 C1，score 双槽和握手必须保持有效。

## 7. 核间与核内同步

### 7.1 AIC ↔ 两个 AIV

采用 `CrossCoreSetFlag/WaitFlag` mode 4。表内是本地 base id；AIC 对两个 AIV 显式使用 `id` 和 `id+16`，AIV 使用自身通道。

| 通知 | base id | 生产流水 → 消费流水 | 语义 |
|---|---|---|---|
| score 槽可写 `V1_C1[slot]` | 0 / 1 | AIV V → AIC FIX | 两个 AIV 均消费完本槽，AIC 才能覆盖 |
| score 就绪 `C1_V1[slot]` | 2 / 3 | AIC FIX → AIV V | 本槽 Fixpipe 完成后两个 AIV 可读取 |
| 整组 Pscale 就绪 `PSCALE_C2[groupId%2]` | 4 / 7 | AIV MTE3 → AIC MTE1 | 组首 C2 等待两侧；后续 C2 复用本组 scale |
| PV 槽可写 `V2_C2` | 5 | AIV V → AIC FIX | V2 已消费上一组 PV |
| PV 就绪 `C2_V2` | 6 | AIC FIX → AIV V | 组末 C2 的 Fixpipe 已完成 |

初始化时 AIV 发布两个 score 槽和 PV 槽的 free credit。即使某 AIV 当前有效 Q 行为 0，也必须 wait/归还每个 score 槽。Pscale 使用两条计数通道，避免跨 Q 大量短组令单条通道累计超过硬件 15 次上限。退出时 AIC 回收 free credit，再释放本地事件。

L1 P 没有单独逐 task 的跨核 ready flag。V1 的 P 搬运、稍后的整组 Pscale 搬运在同一 MTE3 流水排序；Pscale ready 在这些搬运之后发布，组首 C2 的等待保证 P 和 scale 都可读。20 槽生命周期和先退休再生产的主循环防止环绕覆盖。

### 7.2 单核流水

| buffer / 阶段 | 同步 |
|---|---|
| L1 Q/KV | `MTE1_MTE2` 等消费完成后写入；`MTE2_MTE1` 等生产完成后 LoadData |
| 成对 L0A/B | 左槽 `M_MTE1` 防止加载覆盖，`MTE1_M` 保证 Mmad 输入可读；右槽不重复分配事件 |
| L0C QK 两槽 | `FIX_M` 保护源槽再次 Mmad；`M_FIX` 保护 Fixpipe 读取；释放在搬运之后 |
| UB P 两槽 | 每槽独立 `V_MTE3` 和 `MTE3_V`；搬出后只 Set free，直到下次写同槽才 Wait |
| temp Pscale/输出 | 单独一对 `V_MTE3/MTE3_V`，复用前等上一搬运结束 |
| FD | MTE2/V/MTE3 自有流水 flag，FA 和 FD 的 workspace 交接由 `SyncAll()` 保护 |

持久 P/temp 事件使用 `AllocEventID` 分配，各方向 3 个；不能用只查询空闲 ID 的 `FetchEventID` 代替。释放时先 Wait，再 `ReleaseEventID`。

FD 由 AIV 执行：第一个 `SyncAll()` 等本 section 各核 FA workspace 写完；第二个等所有 FD 读取完成，防止下一 section 重用 workspace。AIC 不等待这两个 AIV 阶段屏障，而受跨核 score/Pscale/PV credit 限制前进。

FD 统计槽在 `ComputeScaleValue` 消费完成后释放 V→MTE2 credit；输出 credit 在 BF16 Cast 前获取，Cast 后交给 MTE3，搬运结束后归还。reference 和输出缓冲均不保存残差。

当前同步允许 L0C 槽 0 的 FIX 与槽 1 的 Mmad 交错，以及 P 槽 0 的搬运与槽 1 的 V1 交错。是否真实重叠及重叠比例需要设备时间线，不能用双槽定义代替测量。

## 8. Vector 数值精度

| 阶段 | 计算 / 存储精度 | 说明 |
|---|---|---|
| 接收 QK score | FP32 | Cube MXFP4 点积累积为 FP32，无 FP16 score 缓冲 |
| V1 缩放、mask、S32 max、指数变换、Exp | FP32 寄存器和 UB | DN/ND 各自独立 VF；无效 KV/causal 元素不进入有效概率 |
| task/组 reference | FP32 存储整数意义的二进制指数 | `taskGroupMax_` 与 `globalCurrentMax_`，不是原始自然底 score max |
| P 量化中间值 | BF16 寄存器 | DN/ND 均将 FP32 Exp 结果 Cast 为 BF16，再进行 FP4 Cast；BF16 P 不落 UB |
| P payload | FP4 E2M1 | `FP32 → BF16 → FP4`，随后固定 lane 打包为 uint8；UB/L1 只存最终 packed FP4 P |
| Pscale | FP32 算指数，int32 中转，uint8 E8M0 存储 | 按组统一参考，编码结果送 L1 |
| alpha、history O、row sum | FP32 | V2 合并最多 16 task 的 PV 与增强分母 |
| 归一化 | FP32 | 非 FD 对累计 O 除以 rowsum；FD 对合并倍率除以总 rowsum，均用 `Div<float, &MXFP4_OUTPUT_DIV>` 请求精确舍入 |
| 最终输出 | BF16 | `MXFP4_CAST_ZERO/ONE`：`SAT`、`ZEROING`、`CAST_ROUND` |
| LSE | FP32 | `ln(sum) + (reference+3)*ln(2) - ln(delta)`，CX 补偿只应用一次 |
| FD | FP32 | 先计算总分母及单个 FP32 倍率 `a_i/S`，直接乘加合并 PV，最后转 BF16 |

默认 CX Delta 的 `x=7.25`、`delta=1.5` 为编译期算法常量。每 S32 子组产生 P 和局部指数，整组参考统一后得到 E8M0 Pscale。分母由 Cube 的补 1 通道计算量化 P 的行和，因此 P 的量化误差同时进入分子和分母；这不是原始 FP32 Softmax 的无损实现。

BF16 中间 Cast 是当前真实量化链的一部分，会影响 FP4 中点附近的舍入；对拍不能直接用一次 FP32→FP4 Cast 代替。它不改变 score、max、指数、Exp 和历史累积使用 FP32 的事实，也不额外占用一个 BF16 P 的 UB buffer。

Pscale 更新时按新旧参考差计算 alpha，V2 对历史 O 和分母共同重缩放。FD 保存整数意义的二进制 reference；正规范围内直接构造 `2^(reference-max)` 的 FP32 位模式，次正规范围保留 Exp 分支，然后合并。输出转换前不将分片 PV 提前降为 BF16。

FD 的 `PrepareFlashDecodeVf` 先计算 `a_i=2^(R_i-Rmax)` 和总分母 `S=sum(a_i*s_i)`，再将现有倍率缓冲原地改写为 `a_i/S`。倍率写后读之间使用 `LocalMemBar<VEC_STORE,VEC_LOAD>`；`ReduceFinalRes` 直接得到 `sum((a_i/S)*PV_i)`，不再调用独立的逐通道归一化 VF，也不再将最终 S 暂存到 UB。LSE 仍由原始 S 和参考计算。S 为零时使用安全分母 1 并将倍率置零；空分片 PV 在 FA 阶段已清零。

当前使用单个 FP32 倍率，`ReduceFinalRes` 调用共用 `FaVectorApi::ReduceFinalRes_VF` 直接合并 PV。按用户要求已删除额外的倍率低位、乘法残差和 TwoSum 计算，不增加最终逐通道除法。提前除法改变浮点舍入次序；初轮设备测试已观察到少量 BF16 中点翻转。补偿版本的 236/236 通过结果不适用于当前版本，精度门限保持不变。

非 FD 空行、FD 空行均输出零，LSE 使用 `+inf`。边界 BF16 中点仍可能放大 FP32 累积差异；“FP32 Softmax”不意味着所有输出对所有参考逐位一致。

## 9. Fixpipe MM1 双目标与预缩放参数

内部入口 `quant_flash_attn_mxfp4<..., isFd, enableQf322F32Pre=false, QuantSchema QUANT_SCHEMA=QuantSchema::CX_DELTA_QUANT>` 的预缩放 bool 参数默认 false。它传给 Cube、Vector 和对应 Dummy 类型；主控用 `static_assert` 检查两侧配置一致。预缩放开关没有公开为 Python/ACLNN 参数，也不增加 tiling key；内部显式实例化 `enableQf322F32Pre=true` 并重新编译才能开启。P量化方案则独立由公开 `quant_schema` 选择，见第10节。

| 开关 | Fixpipe MM1 | score 缩放位置 |
|---|---|---|
| `false`（默认） | 单次双目标：DN `dualDstCtl=2`，ND `dualDstCtl=1`；始终 `NoQuant` | V1 以 FP32 乘 `softmax_scale` |
| `true`，scale 可精确表达 | 保留原逐 AIV 循环，`dualDstCtl=0`、`subBlockId=0/1`，使用 `QF322F32_PRE` | Fixpipe；V1 使用 `ScorePreScaled=true` |
| `true`，scale 不可精确表达 | 仍走逐 AIV 循环，但 `quantPre=NoQuant` | V1；不截断任意 API scale |

这里 true 表示允许尝试 `QF322F32_PRE`，并不取消原来的精确表达检查。任何分支都只能应用一次 attention scale。

### 9.1 默认双目标的尺寸与尾块

设当前实际 Q 行数为 `q`，当前 QK 子块有效 KV 数为 `n`。

| 字段 | DN（按 N 拆 Q） | ND（按 M 拆 Q） |
|---|---|---|
| `mSize` | `n` | `align_up(q,32)` |
| `nSize` | `128` | `align_up(n,8)` |
| `srcStride` | `align_up(n,16)` | `align_up(q,32)` |
| `dstStride` | `64` | `s2SplitSize` |
| AIV 0 / 1 的 Q 起点 | `0 / 64` | `0 / (align_up(q,32)/2)` |

DN 固定每 AIV 64 Q 行，尾 Q 时双目标读取的填充列仍位于当前 MM1 L0C 槽内，无效 Q 不参与有效行输出。ND 的每 AIV 容量必须保持 Q16 对齐，所以默认分支同时将 Q 的 `LoadQ2L0.mStep` 和 `MatmulQK.m` 扩展至 `align_up(q,32)`；Fixpipe 使用同样的 NZ 行距，Vector 的 `actMSize`、有效行数和输出范围保持实际值。新增计算只涉及填充行，不增加 L0/UB 分配。

原有 score 槽 ready/free、L0C `M_FIX/FIX_M` 和独立 P 双槽同步保持原样。两个 AIV 都收到 score-ready，即使某侧没有有效 Q，也继续归还 credit。

### 9.2 基线差异与 SDK 约束

基线 `f8034b9db` 的 `FixpipeMm1` 按 `subBlockId=0/1` 逐 AIV 搬运。在 `CanScaleInFixpipe(scale)` 成立时使用 `QF322F32_PRE`，否则在 V1 乘 scale；谓词要求 scale 为正有限数且 FP32 位模式低 13 位为零，避免 Fixpipe 截断 scale 精度。Cube 和 Vector 必须共享“score 是否已缩放”的判断。

CANN 9.2.0 实际 `dav_3510/kernel_operator_fixpipe_impl.h` 对 L0C→UB 明确要求：双目标不能同时启用非 `NoQuant`、ReLU 或 COLUMN_MAJOR；按 M 拆分要求 M 为偶数，按 N 拆分要求 N 为 32 的倍数。因此不能只把 `dualDstCtl` 改为 1/2 而保留预缩放。

历史基线有 436 次生产入口直调与 615 次公开 API 检查通过，后续测试曾暂停。双目标修改曾完成 98,304 组地址模型检查及资源复算；地址模型不模拟硬件执行或数值精度。此前默认双目标随 CX Pscale/LSE 和 FD 残差补偿在 200 机器完成小量公开 API 验证（236/236 通过）。用户随后要求删除残差补偿，当前代码已恢复直接乘加；上述结果仅属于补偿版本。历史结果见[本轮精度验证报告](../../../operators/quant_flash_attn/cx_pscale_lse_small_20261111/README.md)，不继承历史设备结果。旧大 scale 的 13/112 项失败不在本轮范围内；没有性能或流水重叠时间线结论。

## 10. 三种 P 量化模式：普通 MXFP4、CX 与 CX Delta

三种算法都使用公开 `quant_mode=5`，Q/K/V 输入仍为 MXFP4 E2M1 + E8M0。DN/ND、D128/D256、mask、V2 与 FD 均沿用相同的任务尺寸、buffer 和同步。公开 API 使用可选整数 `quant_schema` 选择 P 量化模式，默认2；同一个包支持三种方案。

### 10.1 模板选择

量化枚举定义于 `op_kernel/arch35/quant_flash_attn_mxfp4_common.h` 的 `BaseApi` 命名空间：

```cpp
enum class QuantSchema : uint8_t {
    OCP_QUANT = 0,
    CX_QUANT = 1,
    CX_DELTA_QUANT = 2,
};
```

| quant_schema | 模式 | QUANT_SCHEMA |
|---|---|---|
| 0 | 普通 MXFP4 / OCP | QuantSchema::OCP_QUANT |
| 1 | CX（cx_legacy），不乘1.5 | QuantSchema::CX_QUANT |
| 2（默认） | CX Delta（cx_delta） | QuantSchema::CX_DELTA_QUANT |

原 `UseCxQuant` / `UseCxDeltaQaunt` 两个bool模板参数已删除，统一为 `QuantSchema QUANT_SCHEMA = QuantSchema::CX_DELTA_QUANT`。入口、Vector真实/Dummy block、Softmax VF、Pscale、FA统计和FD沿同一枚举传递，Vector通过 `QUANT_SCHEMA_VALUE` 暴露给主控。共享FD block仅在MXFP4输入分支使用该选择，C8行为保持。

`if constexpr` 按枚举选择OCP的floor与固定−2、CX的ceil与无1.5、CX Delta的ceil与1.5。两种CX共用Pscale +124和LSE +3ln2；只有CX Delta使用−ln(1.5)。没有可同时开启两种方案的布尔组合。量化公式、FP32运算顺序、buffer与同步保持不变。

host读取公开 `quant_schema`（缺省2），校验0/1/2并写入 `QuantFlashAttnBaseParams.quantSchema`。该uint8字段使用原三个uint8标志后的对齐字节，结构总大小和后续字段偏移保持。C4外层入口将其转换为枚举，在构建TPipe前统一分派三套模板，主循环中不增加运行时方案判断。tiling key及metadata布局不增加新维度，Q/K/V和FD调度不变。非C4模式仅允许默认2。

`enableQf322F32Pre` 仍独立控制 score 缩放和 Fixpipe 搬运；`p_scale` 也不能用于切换 P 量化模式。

### 10.2 量化公式

对每个 Q 行的有效 score `s_i=QK*softmax_scale`，按连续 32 个 KV 为一个 S32 小组，令 `m_g=max(s_g)`。`R` 是当前至多 16 个 task 与历史合并后的逐行参考 `max(R_old,max_g(k_g))`。S32 小组与 16-task 累积组不是同一种分组。

| 项目 | 普通 MXFP4 | `cx_legacy` | `cx_delta`（默认） |
|---|---|---|---|
| 小组指数 `k_g` | `floor(m_g/ln2)` | `ceil(m_g/ln2-log2(x))` | `ceil(m_g/ln2-log2(x/delta))` |
| 量化前值 `u_i` | `exp(s_i-(k_g-2)*ln2)` | `exp(s_i-k_g*ln2)` | `exp(s_i-k_g*ln2+ln(delta))` |
| P payload | `FP4_E2M1(BF16(u_i))` | 同左 | 同左 |
| scale 偏移 `c` | 2 | `ceil(log2(x))=3` | `ceil(log2(x/delta))=3` |
| E8M0 编码 | `clamp(k_g-R+127-c,0,254)` | 同左 | 同左 |
| 未量化 `u_i*Pscale` | `exp(s_i-R*ln2)` | `1/8*exp(s_i-R*ln2)` | `delta/8*exp(s_i-R*ln2)` |
| 标准 LSE 还原 | `ln(Z)+R*ln2` | `ln(Z)+(R+3)*ln2` | `ln(Z)+(R+3)*ln2-ln(delta)` |

`x=7.25`；`cx_delta` 的 `delta=1.5`，`cx_legacy` 的 `delta=1`。普通模式编码偏置为 125，两种 CX 均为 124。当前固定参数下 `ceil(log2(x))` 和 `ceil(log2(x/delta))` 恰好都是 3；若以后更改常量，必须同步检查 Pscale 与 LSE，不能假定仍相同。

三条路径都使用 FP32 score、max、指数与 Exp，经过 BF16 寄存器中间舍入后再 Cast 到 FP4 E2M1。P 与最终 BF16 输出沿用当前饱和和中点舍入配置。`Z` 是量化 P 与真实 E8M0 scale 经 Cube 补 1 通道计算的分母，不能视为未量化 Softmax 的精确分母。

默认 `cx_delta` 的伪代码：

```text
对每个有效 S32 小组：
  k_g = ceil(max(s_g)/ln2 - log2(7.25/1.5))
  p4_g = E2M1(BF16(exp(s_g - k_g*ln2 + ln(1.5))))
对当前至多 16 个 task：
  R = max(R_old, 所有有效 k_g)
  scale_g = E8M0(clamp(k_g-R+124, 0, 254))
  W = decode(p4_g) * decode(scale_g)
  groupPV = sum_task(W @ V)
  groupSum = sum_task(sum(W))
  alpha = exp((R_old-R)*ln2)  # 首组/无有效历史为 0
  PV = PV_old*alpha + groupPV
  Z = Z_old*alpha + groupSum
最终 O = PV/Z
最终 LSE = ln(Z)+(R+3)*ln2-ln(1.5)
```

### 10.3 三种模式的区别与边界

普通模式的 S32 最大量化输入在 `[4,8)`；不含1.5的 CX 与默认 CX Delta 均在 `(x/2,x] = (3.625,7.25]`。这些为忽略浮点舍入的实数区间。FP4 E2M1 最大有限正值为 6，因此新模式仍可能发生饱和，不能描述成完全无截断。

`cx_delta` 使用 `-log2(7.25/1.5)`（`MXFP4_NEG_LOG2_CX_DELTA`），Exp 中保留 `ln(1.5)`。`UseCxQuant` 使用 `-log2(7.25)`（`MXFP4_NEG_LOG2_CX`），Exp 不含 `ln(1.5)`；DN resident、DN lookahead 和 ND 四行/尾行都遵守此规则。2026-11-12按用户要求删除旧 CX 的1.5，CLI 名 `cx_legacy` 保留以避免参数改名。该模式与 MindIE-SD CX 的指数、P 与 Pscale 公式一致，但我们用 FP32 Softmax，MindIE-SD 对应实现用 FP16，不保证逐位相同。

空 S32 组的 P 清零、scale 编码字节写 0。编码字节 0 不表示浮点 scale 为零，空组无贡献依靠 P 为零。E8M0 钳制边界必须按实际编码计算；表中实数倍率等式只在未触发钳制时适用。空输出行保持 O=0、LSE=+inf。

### 10.4 V2、FD 与验证参考

V2 每个累积组只更新一次历史，使用 FP32 `exp((R_old-R)*ln2)` 重缩放原始 PV 和分母，不增加残差补偿。FD 保留整数 R，按 `a_i=2^(R_i-Rmax)` 计算总分母 `S=sum(a_i*Z_i)`，在 Prepare 中生成 `a_i/S`，随后直接合并原始 split PV。两种 CX 的 FA/FD 都补偿 `+3ln2`；仅 `cx_delta` 再减 `ln(1.5)`。MindIE-SD CX 本身不输出 LSE；本实现仍按 API 保留标准 logsumexp 的量化近似语义。

`qfa_mxfp4_test` 保留全局 CPU golden，新增实际 metadata/16-task CPU 参考、同分块 Triton 参考和不量化 P 的 FP64 参考。CPU/Triton 参考模拟任务分段、组参考和 FD 数学语义，不声称逐指令还原硬件舍入。FP64 用于在量化参考验收通过后统计差距。运行方法与通过标准见 [测试 README](../tests/pytest/qfa_mxfp4_test/README.md)。历史通过记录不能代替本次默认 `cx_delta` 的设备验收。

| 日期 | 修订者 | 内容 |
|---|---|---|
| 2026-11-11 | Codex | 按正式实现整理主循环、20 槽状态、L1/L0/UB预算、FD别名、同步和 FP32/CX 精度；替代原设计文档的计划性描述 |
| 2026-11-11 | Codex | 增补默认双目标、可选 `QF322F32_PRE`、ND 尾块行距及 Cube/Vector 缩放一致性 |
| 2026-11-11 | Codex | 新增普通/CX 两种 P 量化模式的公式、编码、饱和特征、输出/FD补偿、切换位置及验证边界 |
| 2026-11-11 | Codex | FD 改为 Prepare 内归一化合并倍率，删除独立输出归一化 VF；保留 LSE 与空行语义 |
| 2026-11-11 | Codex | CX 编码修正为 124 并恢复标准 LSE 补偿；FD 增加 FP32 权重/累加残差补偿，复用原有 UB 并延长对应 credit 生命周期 |

| 2026-11-11 | Codex | 按用户要求删除 FD 残差补偿与专用 Reduce VF，恢复共用直接乘加及原缓冲同步；保留 Pscale/LSE 修正 |
