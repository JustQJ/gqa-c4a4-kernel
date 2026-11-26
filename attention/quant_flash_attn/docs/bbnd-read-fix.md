# PA_BBND 的 ND2NZ 读放大与修复

归档于 2026-11-26。实测环境为 Ascend 950DT、CANN 9.2.1（`dav_3510`）。主算例为 `bench16k_decode_2` 的 C4 D256 路径，batch 24，16 个 Q head、2 个 KV head，Q 长度 4，最大 KV 长度 69,532，page 128，量化 schema 2。

## 1. 结论与实现状态

- **原 BBND 确实多读了 HBM。** 基线 C4 `decode_2` 的 BBND 和 BNBD 实读分别约为 1.089 和 0.571 GiB/次，相差 1.91 倍。单独搬运实验将问题定位到 `DataCopy<uint8_t>(Nd2NzParams)` 的 **128 B 行宽与 256 B 矩阵内源步长**组合。
- **保留 BBND 时，将奇偶 token 各自改为 512 B 步长。** 本地工作树已实现两次单矩阵 `ND2NZ`。D256 `decode_2` 的 GM→L1 读量下降约 48%，MTE2 耗时仍约 468 µs。独立验证版用一次 `ndNum=2` 调用处理偶数行块，读量维持在 0.596 GiB/次，MTE2 耗时进一步降至约 396–397 µs。
- **双矩阵 BBND 已接近 BNBD。** `decode_2` 同包对照中，GM→L1 和 HBM 实读均比 BNBD 高约 3%，MTE2 耗时高 6.8%，关闭 profiler 的 Graph 算子时延高 6.1%。D128 四 KV head 的有效行只有 64 B，奇偶拆分后仍有约两倍的 sector 开销。

基线提交为 `5f2d913c3`，`bbnd-fix` 分支包含两次调用的实现。单次双矩阵版本在远端独立包中完成编译、精度和性能验证。对应实现入口见 [MXFP4 Cube 搬运](../op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h) 和第 7 节。

## 2. 布局与触发条件

设 `R` 为一个 token、一个 KV head 的 payload 字节数，`Nkv` 为本 rank 的 KV head 数。普通 ND 物理排布如下，`Bn`、`Bs` 分别表示物理页数和页内 token 数。[C4A4 PA_NZ / PA_BNBD 布局定义](c4a4-pa-nz-bnbd-development.md)

| 布局 | C4 payload 物理形状 | 固定一个 KV head 时相邻 token 的源步长 |
| --- | --- | --- |
| `PA_BBND` | `[Bn, Bs, Nkv, D/2]` | `Nkv × R` |
| `PA_BNBD` | `[Bn, Nkv, Bs, D/2]` | `R` |
| `PA_NZ` | `[Bn, Nkv, D/64, Bs, 32]` | 数据在 GM 已是 NZ 分形，按分形搬入 L1 |

C4 D256 的 `R=128 B`。两个 KV head 在 BBND 下按 `t0h0, t0h1, t1h0, t1h1, …` 交错。固定 `h0` 搬运时，API 用 `dValue=128` 请求每行 128 B，用 `srcDValue=256` 表达连续两行的**起始地址差**。对 `uint8_t`，这两个参数的元素数等于字节数。`nValue` 是每个矩阵的行数，`ndNum` 是矩阵数。参数语义见 [Ascend C 随路转换 ND2NZ 搬运](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_00127.html)。

```text
地址偏移      0 B         128 B       256 B       384 B       512 B
BBND         [t0 h0]     [t0 h1]     [t1 h0]     [t1 h1]     [t2 h0] ...
h0 行起点     ↑                       ↑                       ↑
             └──────── 256 B ────────┘
```

原版 D256 `decode_2` 的 C8/C4 Graph 时延比在既有报告里是 BBND **0.988×**、BNBD **1.274×**、NZ **1.326×**。从 payload 解释，C4 的 128 B 有效行在原 BBND 取数路径上约产生 256 B HBM 流量，量级已接近 C8 的 256 B 有效行。C4 的低位宽优势在该 BBND 实验中被读放大抵消。该比值是算子时延比，单独的比值无法定位硬件瓶颈。原 BBND `decode_2` 的 MTE2 流水占用约 98%，需要结合独立 HBM 计数判断流量。[原始 C4/C8 性能对照](kernel-optimized.md#按-q-行派发-aicaiv-任务后的主分支验证2026-11-23)

## 3. 定位额外流量

### 3.1 HBM 交叉采集与字段对照

在 NPU 7 上使用同一份 `decode_2` 逻辑输入，按 BBND、BNBD、BNBD、BBND 顺序采集设备 HBM，单轮回放 2,400 次。用轮次中间 50% 的带宽积分除以区间内算子次数，BBND 两轮为 **1.0905、1.0879 GiB/次**，BNBD 为 **0.5717、0.5700 GiB/次**。最后一轮 BBND 的较宽窗口出现一次超峰值样本，该样本位于统计窗口外。

在 NPU 6 上将其缩为单 head 的独立 `ND2NZ`。固定 `nValue=1`、`dValue=128`、每条指令的 GM 起始地址实际增加 256 B，绕过 L2，只改描述符里的 `srcDValue`。每个任务请求 448 MiB 有效数据。

| `srcDValue` 字段 | 单次 HBM 实读 | 相对于有效数据 |
| --- | ---: | ---: |
| 128 B | 约 449 MiB | 约 1× |
| 129、192、224、255 B | 各约 896 MiB | 约 2× |
| 256 B | 894.2 MiB | 约 2× |
| 257 B | 445.6 MiB | 约 1× |
| 288、384 B | 各约 448 MiB | 约 1× |
| 512 B | 448.6 MiB | 约 1× |

单行情况下实际读地址序列完全相同，仅描述符字段改变，256 B 与 257 B 之间已经出现清晰的流量跳变。257 B 组跳过了两条高于设备峰值的采样。多行版本也复现了同一趋势，128、256、512 B 实际步长的 HBM 实读分别约为 446、896、448 MiB。软件固定请求 128 B，额外字节来自这组参数下的 DMA 取数行为。

### 3.2 并包、sector 和额外字节的去向

950 的 L2 支持 512 B cache line 和 128 B sector。昇腾 CANN 开发者社区介绍了 ND2NZ Advance 模式在步长不大于 256 B 时的并包能力。并包描述的是合并请求，内存侧取数范围仍由硬件选择，可以覆盖有效请求之间的 sector。现有 `Nd2NzParams` 没有独立的 Advance 开关。SDK 将行宽和步长传给 DMA。本实验确认了字段边界和流量效果，公开资料对该用例的内部模式判定与 sector mask 只提供架构背景。[《昇腾 950 NPU 架构白皮书》§4.3.2](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf#page=26) · [昇腾 CANN 社区架构详解](https://cann.csdn.net/69d8a96e54b52172bc684f2e.html)

对 128 B 行，实测流量可用 **每个有效行带回两个 128 B sector** 解释，其中相邻 sector 对应同一 token 的另一个 head。256 B 是此组合的等效取数大小。128 B 和 512 B 步长组均证明这条 GM→L1 路径可以只取 128 B 有效行所需的流量。

进一步让每个 tile 先搬 head0，经 `PIPE_MTE2` barrier 再搬 head1。两者共请求 896 MiB 有效数据，普通缓存模式的 HBM 实读约 896 MiB，绕过 L2 时约 1,792 MiB。先遍历全部 head0、再遍历全部 head1 时，普通缓存模式也约读 1,793 MiB。这说明额外取回的相邻 sector 在短间隔内可经 L2 复用。每次 ND2NZ 在 L1 仍只为该 head 生成自己的有效 NZ 行。现有计数器可核对额外流量与复用效果，硬件回包筛选和缓存分配的逐字节轨迹尚未定位。

### 3.3 量化类型和行宽的边界

| 路径 | 每个 head 的有效行宽 | BBND 源步长 | 控制实验 |
| --- | ---: | ---: | --- |
| C8 D128、2 KV head | 128 B | 256 B | BBND / BNBD HBM 为 1.022 / 0.515 GiB，约 1.99× |
| C8 D128、8 KV head | 128 B | 1,024 B | BBND / BNBD HBM 为 0.564 / 0.514 GiB，约 1.10× |
| C4 D128、4 KV head | 64 B | 256 B | BBND / BNBD HBM 为 2.088 / 0.530 GiB，约 3.94× |

C8 D128 使用 MXFP8 的 `CopyKvGmToL1`，2 KV head 时和 C4 D256 具有同样的 128 B 行宽与 256 B 步长。C8 D128 仓库的 Qwen3-8B TP1 用例用 8 KV head。以上 head 数对照用 Q=4、KV=32,768、page 128，固定 0.5 GiB 逻辑 K/V payload，在同一 head 数下比较两种布局。

C4 D128 每行 64 B，独立单 head 基准在 64、128、256、512 B 源步长的 HBM 读量分别为 **0.437、0.875、1.751、0.875 GiB**，约为有效请求量的 **1、2、4、2 倍**。512 B 步长消除了第二个 sector，稀疏访问同一个 128 B sector 中 64 B 的开销仍在。BNBD 让同一 head 的 64 B 行连续。固定 BBND 如果还需降低这部分流量，候选思路是让相邻 head 共用一次 sector 读取，当前没有相应的整算子实现。

## 4. 修复方案

### 4.1 两次单矩阵 ND2NZ

从每个页内片段的首行开始按局部行号分组，偶数行从原 GM 基址开始，奇数行从 `GM+256 B` 开始。两组内部的 `srcDValue` 都设成 512 B，`dstNzNStride=2`，即按两个 32 B C0 的步长写下一行。奇数组的目标基址为 `L1+32 B`。偶数组搬 `ceil(rows/2)` 行，奇数组搬 `floor(rows/2)` 行，只有一行时只发偶数组。

当前实现位于 `CopyFp4RowsGm2L1<IsBbndKvPayload>`。`CopyFullKvPage` 的整页路径和 `CopyKvPayloadGm2L1` 的跨页路径都传入 BBND 选择。启用条件为 BBND KV payload、源行步长等于 256 B、有效行宽小于源行步长。跨页按片段处理，保留各片段调用传入的 GM、L1 基址。Q、K/V scale、C8 及其他布局沿用各自搬运路径。[内核实现](../op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h)

### 4.2 一次双矩阵 ND2NZ

偶数行块的两组行数相同，可用 `ndNum=2` 描述，并将第二组的基址偏移放入矩阵步长。下例参数用于 `DataCopy<uint8_t>`，`destinationRowStride` 取现有 tile 的 C0 列间步长。

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `ndNum` | `2` | 偶数行和奇数行各一个矩阵 |
| `nValue` | `rowCount / 2` | 两个矩阵各自的行数，128 token 时为 64 |
| `dValue` | `rowByteCount` | 每个 head 的有效行宽 |
| `srcDValue` | `512` | 一个矩阵内相邻两行的 GM 地址差 |
| `srcNdMatrixStride` | `256` | 第二个矩阵的 GM 基址偏移 |
| `dstNzNStride` | `2` | 一个矩阵的相邻 NZ 行隔两个 C0 |
| `dstNzMatrixStride` | `32` | 第二个矩阵的 L1 基址偏移一个 C0，单位为 `uint8_t` 元素 |
| `dstNzC0Stride` | `destinationRowStride` | 同一行在不同 C0 列之间的间距，单位为 C0 |

令 `m` 为矩阵号，`r` 为矩阵内行号，`c` 为 C0 列号。下式给出两个接口写法共同访问的地址。`m=0` 写第 `0, 2, 4…` 行，`m=1` 写第 `1, 3, 5…` 行。

```text
GM(m, r, c) = GM_base + m*256 + r*512 + c*32
            = GM_base + (2*r + m)*256 + c*32
L1(m, r, c) = L1_base + (2*r + m + c*destinationRowStride)*32
```

`srcNdMatrixStride=256` 只决定奇数组的起点。矩阵内取数仍以 512 B 为步长，实测保持两次调用的读量。CANN 9.2.1 的 `impl/dav_3510/kernel_operator_data_copy_impl.h` 将矩阵数与目标步长编码到 `MTE2_NZ_PARA`，再调用一次 `copy_gm_to_cbuf_multi_nd2nz`。两个矩阵共用 `nValue`，所以独立验证版仅合并非零偶数行块，奇数行块继续用原奇偶拆分。[Ascend C 多矩阵参数](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_00127.html)

### 4.3 普通跨步 DataCopy 对照

同一组 GM 地址、实际行步长 256 B、绕过 L2、每次有效请求 448 MiB 时，普通跨步拷贝与奇偶拆分的 HBM 实读接近。结果针对该调用方式和测试尺寸。

| 有效行宽 | 原 ND2NZ，GiB | 普通跨步 DataCopy，GiB | 两次奇偶 ND2NZ，GiB |
| --- | ---: | ---: | ---: |
| 128 B | 0.87537 | 0.43761 | 0.43740 |
| 64 B | 1.74914 | 0.87497 | 0.87516 |

普通跨步 `DataCopy` 按整行拷入 L1 时得到 ND，Cube 还需转换为 NZ。把普通拷贝按 32 B 列片分发可以直接填入 NZ，但 64 B 行的实测耗时高于原 ND2NZ。该列片方案绕过 L2 时读放大也会增至约 4 倍。使用 BNBD 可在生产者侧改变物理排布，使用 NZ 可在搬入 L1 时直接按分形拷贝，两者都是独立的布局方案。

## 5. 实测结果

### 5.1 两次调用先解决读量

独立搬运微基准固定每次 448 MiB 有效数据和每 tile 128 行，按原 ND2NZ、奇偶拆分、奇偶拆分、原 ND2NZ 的顺序各跑 2,200 次。普通 L2 下，128 B 行的 HBM 读量从 896.64 降到 447.58 MiB，任务中位时延从 408.50 降到 280.29 µs，下降 31.4%。64 B 行从 1,792.56 降到 894.99 MiB，从 828.48 降到 562.34 µs，下降 32.1%。绕过 L2 后两种行宽的时延也分别下降约 29.9% 和 31.5%。128 B 普通 L2 组剔除了一条高于设备峰值的采样。

整算子使用 NPU 7 的 `msprof op --aic-metrics=Default`。每组取三次采样的中位数，GM→L1 读量按 28 个 AIC 求和，MTE2 时间按 AIC 取平均。`GM_to_L1_datas(KB)` 是算子搬运计数，包含 KV payload 之外的流量。HBM 实读另用设备带宽计数验证。[MTE2 流水计时字段](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/optool/atlasopdev_16_0099.html) · [内存搬运计数字段](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0095.html)

| C4 用例 | 原 BBND，GiB / µs | 两次调用，GiB / µs | NZ，GiB / µs |
| --- | ---: | ---: | ---: |
| D256 `decode_2` | 1.13775 / 467.99 | 0.59568 / 468.02 | 0.57759 / 311.85 |
| D256 `dual_prefill_1` | 3.20640 / 959.94 | 1.69211 / 718.21 | 1.64180 / 557.39 |
| D128 四 KV head decode | 2.08638 / 838.67 | 1.08252 / 838.92 | 0.53270 / 372.23 |

NPU 6 的长 Graph 回放中，D256 `decode_2` 的实际 HBM 读量从 1.08505 降到 0.59110 GiB/次，NZ 为 0.57779 GiB/次。D128 四 KV head 用例对应 2.09074、1.08065 和 0.53207 GiB/次。

`decode_2` 两次调用使整个算子的 MTE2 指令数从每 AIC 5,107 增至 7,648。有效带宽按实际计数读量和 MTE2 时间计算，因此读量降约 48%、时间持平时，计数带宽从 86.83 降至 45.46 GB/s/AIC。同样 468 µs 内完成的有效 KV 行数近似相同。普通 `msprof` 复测约为 482 / 480 µs，关闭 profiler 的 Graph 复测为 501.7 / 492.4 µs，算子时延改善约 1.8%。双请求 prefill 的 MTE2 时间则下降约 25%。

独立的等流量对照固定 512 B 步长、读写地址和行顺序，一条搬 128 行与两条各搬 64 行的 HBM 实读均约 446 MiB。普通 L2 下耗时为 251.0 / 276.7 µs，绕过 L2 为 222.7 / 244.0 µs。切分成本约为 10%，包含额外发指令和改变硬件处理批次的综合影响，现有采集无法分出各项成本。

### 5.2 双矩阵进一步缩短 MTE2 时间

使用相同 BBND 地址和奇偶写回方式，NPU 6 上独立搬运微基准按两次、双矩阵、双矩阵、两次的顺序采集，每轮 1,500 次。HBM 实读在约 444–446 MiB。启用普通 L2 时，延迟从 278.18 降为 253.87 µs，改善 8.74%。绕过 L2 时从 248.28 降为 226.87 µs，改善 8.62%。

NPU 7 的 D256 `decode_2` 整算子中，两次调用与双矩阵的 GM→L1 读量分别为 0.59568 和 0.59559 GiB，指令数从每 AIC 7,647.57 降至 5,107 条，MTE2 时间从 468.02 降至 396.74 µs，改善 **15.23%**。同输入 NZ 为 311.85 µs，双矩阵仍高约 27.2%。

### 5.3 双矩阵 BBND 与 BNBD 同包对照

| D256 `decode_2` 指标 | 双矩阵 BBND | BNBD | BBND 高出 |
| --- | ---: | ---: | ---: |
| GM→L1 读量，GiB/次 | 0.59558 | 0.57754 | 3.12% |
| HBM 实读量，GiB/次 | 0.59136 | 0.57329 | 3.15% |
| MTE2 耗时，µs | 396.28 | 371.00 | 6.81% |
| MTE2 指令数，每 AIC | 5,107 | 5,107 | 0% |
| Graph 算子时延，关闭 profiler，µs | 421.64 | 397.40 | 6.10% |

表中每行都是同卡、同候选包、同逻辑输入对照，输入 SHA 一致。MTE2 在 NPU 7 各采三次。HBM 在 NPU 6 按 BBND、BNBD、BNBD、BBND 交叉采集，每轮 2,100 次 Graph 回放，以中间 50% 区间积分后取两轮均值，四轮均无剔除点。关闭 profiler 的 Graph 用 NPU 5，预热 3 次、9 轮、每轮 20 次回放，取中位数。双矩阵与 BNBD 的 MTE2 指令数已相同，该用例仍有约 25 µs 的 MTE2 时间差。

## 6. 精度与验证范围

奇偶两次调用和单次双矩阵隔离候选均通过 BBND 的 **220 项精度用例**，包括 204 个 `document` 小用例、12 个 `bench16k` 小用例以及 4 个针对性奇数尾页用例。尾页覆盖 C4 D128 四 KV head、D256 两 KV head、page 128/256 和 1/3 token 的剩余长度。编译候选的 kernel SHA 与 profiler dump 中实际加载的 kernel 匹配。

双矩阵的整算子性能结论来自 D256 `decode_2`。D128 四 KV head 和 D256 双请求 prefill 的表格结果来自两次调用版本。HBM 读放大依赖行宽、描述符步长与 L2 复用，其他形状应按这三个条件和实际计数判断。

## 7. 原始数据与复现

原始文件在 `195.121.2.18` 上的 `tp_a4c4_gqa_kernel_ops` 容器。工作目录为 `/home/t00970481/C4A4-GQA`，下表路径均相对于其 `work/` 子目录。

| 实验 | 目录 |
| --- | --- |
| 原 BBND / BNBD 的 HBM 交叉复测 | `bbnd_c4_hbm_layout_20261126.khG8jV/` |
| 128 B 行宽、描述符字段及 L2 复用 | `nd2nz_stride_c4_20261126.XWTHAR/` |
| 256/257 B 字段边界 | `nd2nz_stride_c4_20261126.XWTHAR/stride_boundary.SOzXfT/` |
| C8 D128、C4 D128 控制实验 | `c8_d128_bbnd_20261126.fL4raV/`、`nd2nz_64b_20261126.82leOb/`、`c4_d128_bbnd_20261126.R5TF6i/` |
| 普通拷贝对照、单次与两次调用微基准 | `plain_copy_nd2nz_20261126.4RRbdJ/`、`nd2nz_single_vs_split_20261126.ZQBWt1/` |
| 等流量切分指令条数对照 | `nd2nz_instruction_count_20261126.sXlYyr/` |
| 两次调用整算子构建、精度和性能 | `bbnd_evenodd_20261126.2VVT56/` |
| 双矩阵微基准与隔离候选 | `nd2nz_two_matrices_20261126.J7UKfH/`，候选源码位于 `operator/src/quant_flash_attn/op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h` |
| 双矩阵与 BNBD 同包数据 | `nd2nz_two_matrices_vs_bnbd_20261126.vGHZqk/` |

在容器内复算最后一组对照数据。

```bash
work=/home/t00970481/C4A4-GQA/work
ref="$work/nd2nz_two_matrices_vs_bnbd_20261126.vGHZqk"
python3 "$ref/summarize_measurements.py" mte2
python3 "$ref/summarize_measurements.py" graph
```

在同一环境有空闲设备时，可将现有对照脚本复制到新目录重跑。`mte2` 输出 `Memory` 和 `PipeUtilization` 等 CSV，`graph` 输出设备 HBM 和任务数据库，`plain` 输出关闭 profiler 的 Graph 性能 JSON。

```bash
pkg="$work/nd2nz_two_matrices_20261126.J7UKfH/operator"
run=$(mktemp -d "$work/bbnd-bnbd-replay.XXXXXX")
mkdir -p "$run/tmp" "$run/logs"
cp "$ref/compare_layouts.sh" "$ref/summarize_measurements.py" "$run/"
bash "$run/compare_layouts.sh" mte2 7 "$pkg"
bash "$run/compare_layouts.sh" graph 6 "$pkg"
bash "$run/compare_layouts.sh" plain 5 "$pkg"
```
