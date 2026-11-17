# 性能优化记录

## 优化遵守标准

1. 精度要严格保证通过

    小的改动通过 tests\pytest\qfa_mxfp4_test\test_quant_flash_attn_func_rdv.py 的 --accuracy-case-set document --scope small 测试集 + --accuracy-case-set bench16k 测试集

    多次大的改动要进行一次 tests\pytest\qfa_mxfp4_test\test_quant_flash_attn_func_rdv.py 的 --accuracy-case-set document --scope full 测试集

## 优化方向

1. vector的vf指令优化

2. scalar 的优化，调整基本块，减少循环的scalar指令

3. 小shape的特化


## D256 优化记录过程

性能测试使用 tests/pytest/qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py 的 `--perf-case-set bench16k --run-perf --perf-backends c4,c8 --perf-mode graph --perf-warmup 3 --perf-rounds 5 --perf-repeats 20` 12 个case进行测试。

使用 tests\pytest\qfa_mxfp4_test\prof_quant_flash_attn.py  的 msprof 和 msprof op （ Default 和 InstrTimeline 两个指标） 采集 bench16k_dual_prefill_1 和 bench16k_decode_4  的性能数据

### 总表记录

| 阶段 | 改动 | 数据来源 |
| --- | --- | --- |
| Base | 输出高精度 Div，`precisionMode=true` | 2026-11-16 的 A/B/C 三个批次汇总；`decode_3` 使用短尾最新参数 |
| 普通 Div + 倒数复用 | 关闭输出 Div 高精度修正，`precisionMode=false`；DN/ND 输出复用倒数，一次除法后各通道乘法 | 2026-11-17 的 `performance-candidate`，包 `cann_ops_transformer_qfa_recip_bench16k_20261117` |
| 再加 ExpSub + 乘加 | 在普通 Div + 倒数复用基础上，追加 37 处 Sub→Exp 和 11 处乘加融合 | 2026-11-17 融合实验的 `performance-compound`，包 `cann_ops_transformer_qfa_fuse_cmp_20261117` |

三个阶段均为 200 / `tp_a4c4_gqa_kernel` / NPU7，schema2，Graph 回放 + Event，3 次预热、5 轮 × 20 次调用取中位数；输入准备、metadata 和格式转换在计时外。固定 Nq16/Nkv2/D256/page128、mask3、LSE 关闭，K/V 各使用 31374 页固定池。

Base 是历史三个批次的汇总，后两列分别使用倒数复用阶段和 ExpSub + 乘加联合融合阶段的实测结果。已核对三个阶段的 12 项完整 case 指纹一致。下表耗时单位为 μs，case 省略 `bench16k_` 前缀；百分比使用原始未舍入值计算，正值表示耗时减少，负值表示耗时增加。“较 Base 减少”指联合融合阶段相对原始 Base 的累计变化。

本次融合实验仅测 C4。总表的 C8 对照固定使用 Base 的历史 C8 数据，“Base C8/融合 C4”为该历史 C8 耗时除以联合融合阶段 C4 耗时，大于 1 表示融合后的 C4 更快；这是跨批次参考。融合本身的增量收益使用下节同轮重测基线比较。

| case | Base C4 | Base C8 | 普通 Div + 倒数复用 C4 | 再加 ExpSub + 乘加 C4 | 较 Base 减少 | Base C8/融合 C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 318.742 | 212.413 | 246.134 | 240.429 | 24.57% | 0.883 |
| single_prefill_2 | 1045.231 | 937.282 | 994.794 | 991.298 | 5.16% | 0.946 |
| single_prefill_3 | 1435.241 | 1300.633 | 1385.437 | 1381.087 | 3.77% | 0.942 |
| dual_prefill_1 | 1544.096 | 1132.345 | 1230.413 | 1212.775 | 21.46% | 0.934 |
| dual_prefill_2 | 1431.694 | 1251.043 | 1324.859 | 1319.603 | 7.83% | 0.948 |
| dual_prefill_3 | 2273.021 | 2089.713 | 2194.643 | 2188.596 | 3.71% | 0.955 |
| decode_1 | 330.693 | 355.821 | 330.377 | 330.917 | -0.07% | 1.075 |
| decode_2 | 715.852 | 914.377 | 712.098 | 712.165 | 0.52% | 1.284 |
| decode_3 | 48.793 | 41.291 | 48.682 | 48.386 | 0.83% | 0.853 |
| decode_4 | 52.640 | 50.420 | 53.858 | 54.066 | -2.71% | 0.933 |
| mixed_1 | 1361.233 | 1371.783 | 1322.265 | 1316.726 | 3.27% | 1.042 |
| mixed_2 | 3587.749 | 2972.840 | 3536.840 | 3529.779 | 1.62% | 0.842 |

分组按 case 等权计算耗时比的几何平均。`Base/倒数复用` 和 `Base/融合后` 大于 1 表示相应阶段比原始 Base 更快；两列 C8 比值均使用固定的 Base C8 数据。

| 范围 | case 数 | Base/倒数复用 | Base/融合后 | Base C8/C4 | Base C8/融合 C4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 12 | 1.0611 | 1.0663 | 0.9037 | 0.9636 |
| prefill | 6 | 1.1206 | 1.1302 | 0.8266 | 0.9342 |
| decode | 4 | 0.9964 | 0.9965 | 1.0274 | 1.0238 |
| mixed | 2 | 1.0219 | 1.0251 | 0.9138 | 0.9367 |

联合融合阶段相对 Base 的 C4 耗时比几何平均为 1.0663，折合整体耗时减少约 6.21%；主要累计收益出现在 prefill（1.1302），decode 为 0.9965。联合融合阶段与 Base C8 比较，C4 耗时更短用例为 3/12。

数据来源：[Base 原始汇总](../../../operators/quant_flash_attn/bench16k_performance_summary_20261116/README.md)、[倒数复用阶段](../../../operators/quant_flash_attn/reciprocal_bench16k_20261117_card7_01/README.md)、[联合融合结果快照](../../../operators/quant_flash_attn/vf_compound_fusion_20261117_card7_01/remote-status-20261117.json)。

### 去掉高精度除法设置

#### 实现改动

修改 [vf_softmax_fp32_mxfp4.h](../op_kernel/arch35/vf/vf_softmax_fp32_mxfp4.h) 的输出除法配置：

```cpp
// 优化前
constexpr DivSpecificMode MXFP4_OUTPUT_DIV = {MaskMergeMode::ZEROING, true};
// 优化后
constexpr DivSpecificMode MXFP4_OUTPUT_DIV = {MaskMergeMode::ZEROING, false};
```

`precisionMode=false` 使用普通硬件 Div，取消输出归一化除法的高精度修正。当前优化同时包含 DN/ND 输出倒数复用；P 量化、softmax scale、输出类型和精度比较门限沿用原实现。

修改 [vf_output_update_fp32_mxfp4.h](../op_kernel/arch35/vf/vf_output_update_fp32_mxfp4.h) 的输出归一化：

- DN 在通道循环外，用一次向量 Div 求出 Q64 的 `1 / rowsum`，通道循环内改用 Mul。
- ND 在行循环外，用一次向量 Div 求出 Q64 全部行的倒数，再逐行 Gather 广播并用 Mul 归一化。
- 保留空行、尾行和 BF16 转换处理；FD 分段的原始 PV 输出继续跳过局部归一化。

D256 满 Q64 输出块的归一化，在源码 API 层面由 256 次向量 Div 改为 1 次 Div + 256 次 Mul；这不是整个 kernel 的机器指令计数。ND 还增加了逐行 Gather，实际收益以设备计时为准。

普通 Div + 倒数复用阶段的 12 项 Graph 结果记录在总表对应列。`dual_prefill_1` 的 C4 从指定 Base 的 1544.096 μs 降至 1230.413 μs，跨批次耗时减少 20.31%，当前 C8 为 1132.407 μs；`single_prefill_1` 减少 22.78%。`decode_4` 从 52.640 μs 增至 53.858 μs，耗时增加 2.31%。该阶段未包含已撤销的 DN TaskN256 打包和 ND ExpSub 融合；普通 Div 与 DN/ND 倒数复用已提交为 `98e3d8d1a`，未推送。后续融合候选的本轮对照结果见下节。

#### 精度与采集覆盖

| 版本 | 已完成验证 | 范围说明 |
| --- | --- | --- |
| 优化后（普通 Div + 倒数复用） | schema2 的 bench16k 三方精度 12/12、ND 补充精度 8/8；C4/C8 Graph 性能 12/12 | CPU/kernel/Triton 沿用现有参考与门限；ND 补充覆盖 D128/D256、G3/G8、Q17/1041、KV257、mask0/3，开启 LSE |

bench16k 本身包含 8 个 DN（6 个 prefill + 2 个 mixed）和 4 个 ND decode；`decode_3` 的 FD 关闭，已覆盖 `processOutNdVF`。额外 8 项 ND 用于扩展边界覆盖。倒数乘法与直接除法存在 BF16 舍入差异，以上结论为现行容差内通过。

按本文开头的验收标准，上述普通 Div + 倒数复用基线尚待补跑 `--accuracy-case-set document --scope small`；本轮也未执行 `document full` 或其他 schema。已完成的 bench16k + ND 补充不能替代该矩阵。

#### dual_prefill_1 的 C4 采集（2026-11-17）

在 200 / `tp_a4c4_gqa_kernel` / NPU7 对当前优化版本采集，仅 C4。参数、固定页池、输入 SHA 和 case 指纹与总表的 Graph 测试一致；下表为 eager 诊断口径，与 Graph/Event 分开统计。

| 采集 | 状态 | 主算子耗时中位数/μs | 说明 |
| --- | --- | ---: | --- |
| 普通 msprof / PipeUtilization | validated | 1229.123 | 20 次应用预热 + 5 次正式调用；Vector 中位数 1007.072 μs |
| msprof op / BasicInfo | validated | 1233.267212 | MSTX 内捕获 5 次，工具预热 3 次；当前/额定频率均为 1650/1650 MHz |
| msprof op / InstrTimeline，全管道 | failed | — | 18 条 data loss，aivec error 259 / retCode 0x26，worker 未完成 |
| msprof op / InstrTimeline，仅 vector 补采 | failed | — | 13 条 data loss，重放预热未完成，仅结束本次 worker 后退出 |

两组成功采集的输出有限、allocated 前后均为 0。全管道及 vector trace 分别保留 408918、158502 个诊断区间，本轮没有获得完整有效的 InstrTimeline；不能用这些区间作完整指令耗时依据。结束 NPU7 健康 OK、无残留进程，未重置设备。

235 项生产源码、33 项测试、实际编译复制头及 18 项安装对象核对通过；239 项原始文件及清单下载逐 SHA 核验，并提供 CSV/trace 短路径副本。详见[本轮 C4 报告与原始数据](../../../operators/quant_flash_attn/dual_prefill1_optimized_profiler_20261117_card7_01/README.md)。本轮未采 C8 或 `decode_4`，未重新执行精度矩阵。

### Sub→Exp 与乘加融合（2026-11-17）

#### 改动与本轮基线

本轮以普通 Div + DN/ND 倒数复用版本（`98e3d8d1a`）为基线，在 200 / `tp_a4c4_gqa_kernel` / NPU7 上独立构建两套候选：

- `exp_sub`：softmax 的 37 处相邻 Sub→Exp 改用 `AscendC::Reg::ExpSub`，覆盖 DN/ND 路径。
- `compound`：在 `exp_sub` 上追加 11 处乘加融合，使用 `MulDstAdd` / `MulAddDst`，覆盖历史 rowsum/PV 更新及 FD 合并。

以上数量是源码替换位置数。掩码及 Cast 链沿用原实现；带中间 Select 或缩放的序列保留运算顺序。候选包分别为 `cann_ops_transformer_qfa_fuse_exp_20261117` 和 `cann_ops_transformer_qfa_fuse_cmp_20261117`。

本轮基线、ExpSub、ExpSub + 乘加依次完成完整 12 项 C4 Graph 测试，均为 schema2、3 次预热、5 轮 × 20 次调用取中位数，计时范围与前述 attention-only 口径一致。本轮重新测量的基线与总表中倒数复用阶段的历史值属于不同批次；以下降幅统一相对本轮基线计算。本轮仅测 C4。

#### 12 项 Graph 结果

耗时单位为 μs，case 省略 `bench16k_` 前缀。百分比按原始未舍入值计算，正值表示耗时减少，负值表示耗时增加。

| case | 本轮基线 C4 | ExpSub C4 | 较基线减少 | ExpSub + 乘加 C4 | 较基线减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 246.364 | 240.045 | 2.57% | 240.429 | 2.41% |
| single_prefill_2 | 992.322 | 989.405 | 0.29% | 991.298 | 0.10% |
| single_prefill_3 | 1386.154 | 1381.742 | 0.32% | 1381.087 | 0.37% |
| dual_prefill_1 | 1228.242 | 1214.742 | 1.10% | 1212.775 | 1.26% |
| dual_prefill_2 | 1324.406 | 1320.729 | 0.28% | 1319.603 | 0.36% |
| dual_prefill_3 | 2192.156 | 2189.479 | 0.12% | 2188.596 | 0.16% |
| decode_1 | 330.171 | 331.701 | -0.46% | 330.917 | -0.23% |
| decode_2 | 711.832 | 712.297 | -0.07% | 712.165 | -0.05% |
| decode_3 | 48.494 | 48.430 | 0.13% | 48.386 | 0.22% |
| decode_4 | 54.570 | 54.045 | 0.96% | 54.066 | 0.92% |
| mixed_1 | 1322.262 | 1317.999 | 0.32% | 1316.726 | 0.42% |
| mixed_2 | 3536.609 | 3530.323 | 0.18% | 3529.779 | 0.19% |

分组按 case 等权计算耗时比的几何平均，大于 1 表示候选更快。

| 范围 | case 数 | 基线/ExpSub | 基线/(ExpSub + 乘加) |
| --- | ---: | ---: | ---: |
| all | 12 | 1.004837 | 1.005174 |
| prefill | 6 | 1.007893 | 1.007866 |
| decode | 4 | 1.001431 | 1.002199 |
| mixed | 2 | 1.002507 | 1.003069 |

ExpSub 和 ExpSub + 乘加的整体耗时分别减少约 0.48% 和 0.51%。单例改善最大的是 `single_prefill_1`（2.57% / 2.41%），`dual_prefill_1` 减少 1.10% / 1.26%；`decode_1/2` 略有回退。追加乘加相对单独 ExpSub 的变化有升有降，单例约 -0.19% 至 +0.24%；本轮顺序采样尚不能确认稳定的独立增益。

#### 精度与结果核验

两套候选均通过独立构建及实际编译副本审计；精度使用现有 kernel + Triton 参考及原比较门限。

| 候选 | document small 精度 | bench16k 精度 | C4 Graph 性能 |
| --- | ---: | ---: | ---: |
| ExpSub | 204/204 | 12/12 | 12/12 |
| ExpSub + 乘加 | 204/204 | 12/12 | 12/12 |

精度测试无失败、错误或跳过；基线本轮仅重测 C4 Graph 性能 12/12。12 项 case 指纹及 CPU 输入 SHA 在三个版本间一致，36 路性能结果输出有限、Graph 回放抽样一致，释放后 allocated 均为 0。两套融合候选已覆盖本文要求的 document small + bench16k；本轮未执行 document full 或其他 schema，不将候选成绩作为基线的独立精度验收。

#### dual_prefill_1 的 msprof 对照

本轮三个版本另采普通 msprof / PipeUtilization，仅 C4、eager 诊断口径；每版 20 次应用预热 + 5 次正式调用。原始 CSV 的 SHA、精确 25 次 `QuantFlashAttn` 任务及后 5 次正式样本均已复核，排除 Metadata 任务；三套采集均 validated，无 data loss，释放后 allocated 为 0。case 指纹和输入 SHA 与本轮 Graph 对应。

| 版本 | 总耗时/μs | 较基线减少 | Vector/μs | Vector 较基线减少 |
| --- | ---: | ---: | ---: | ---: |
| 本轮基线 | 1229.231 | 0.00% | 1007.497 | 0.00% |
| ExpSub | 1214.555 | 1.19% | 946.883 | 6.02% |
| ExpSub + 乘加 | 1213.535 | 1.28% | 944.203 | 6.28% |

本轮 msprof 未采其他 11 个 case，也未新增 msprof op / BasicInfo / InstrTimeline。VF 计算部分的反汇编核对尚未完成，当前证据为融合 API 的源码改动及本轮精度、性能结果，尚不能逐条确认目标融合机器指令。

实验于 2026-11-17 13:37 完成，控制器为 passed；14:22 检查时 NPU7 健康 OK、利用率 0、无实验或设备进程。以上为当时检查状态。结果来源为[本轮检查报告](../../../operators/quant_flash_attn/vf_compound_fusion_20261117_card7_01/remote-check-20261117.md)与[原始样本、JUnit 摘要及远端文件 SHA 清单](../../../operators/quant_flash_attn/vf_compound_fusion_20261117_card7_01/remote-status-20261117.json)。完整实验归档尚留远端。

#### 联合融合版本补采 msprof 与 msprof op（2026-11-17）

本次对当前 ExpSub + 乘加联合融合代码采集 `bench16k_dual_prefill_1`，200 / `tp_a4c4_gqa_kernel` / NPU7，仅 C4、schema2。235 项生产源码、实际编译副本和 18 项安装对象核对通过，实际加载包为 `cann_ops_transformer_qfa_fuse_cmp_20261117`；本轮沿用远端已有测试快照，目标 case 参数、输入生成及执行文件与本机一致。参数、31374 页固定池、输入 SHA 和 case 指纹与前述 Graph 结果一致。

| 采集 | 状态 | 主算子耗时中位数/μs | 说明 |
| --- | --- | ---: | --- |
| 普通 msprof / PipeUtilization | validated | 1212.558 | 20 次预热 + 5 次正式，25 次目标任务完整；Vector 944.296 μs |
| msprof op / BasicInfo | validated | 1217.428223 | MSTX 内 5 次，工具预热 3 次；当前/额定频率均 1650/1650 MHz |
| msprof op / InstrTimeline | failed | — | 18 条 data loss，aivec error 259；worker 未完成 |

两组成功采集输出有限、allocated 前后均为 0；以上为 eager 诊断口径。InstrTimeline 留下 407808 个诊断区间，不能作为完整有效指令时间线。工具退出后 15:11 检查 NPU7 健康 OK、利用率 0、无残留进程。未重置设备，本轮未采 C8 或其他 case。

215 项原始文件加清单已下载并逐 SHA/尺寸/精确集合核验，提供 7 份 CSV/trace 短路径副本；自动 dump 保留远端并入清单。详见[本轮联合融合版本采集报告与原始数据](../../../operators/quant_flash_attn/dual_prefill1_compound_profiler_20261117_card7_01/README.md)。


### DN Pscale：Scatter 替换为分块 Store（2026-11-17）

基于 `cafc1498c` 的联合融合版本，仅将 `UpdatePScaleDnVf` 的 5 处静态 Scatter 调用改成 `DATA_BLOCK_COPY StoreAlign`，删除专用索引构造，保留寄存器 Gather 压紧、量化公式、参考指数和 alpha。每条 Store 启用 4 个完整 32B 块，步长为 3/5，保持原 `[Q16块,S64块,Q16,2]` 布局和每个 Q16 块末尾 32B 零 padding。

3510 的 bank group 为 `(byte_address >> 5) & 7`。4 块相对 group 分别为 `0,3,6,1` 或 `0,5,2,7`，因此本次每条 Store 的有效块分散到 4 个不同组；此结论不覆盖整个 kernel 或跨流水访问。CANN 9.2 的独立 DN Task128/256 实例 LLVM IR 已确认：8 个 `vscatter` 调用换成 8 个 `vsstb`，步长控制字分别为 `3 << 16`、`5 << 16`，未将不完整的 objdump Reg 解码作为机器指令证据。

本地新旧实际 VF 字节模型各 7,680 组、独立 bank/对齐/边界检查 64 组通过。200 / `tp_a4c4_gqa_kernel` / 仅 NPU7：独立 Pscale 256/256，通过 document small 204/204 + bench16k 12/12。按用户要求，后两组直接复用已核对版本、完整参数 fingerprint 和 input_id 的历史 CPU reference，运行 `--golden-mode npu,compare`，不再运行 Triton reference。

配对性能使用同一轮测试代码和输入、schema2、C4/C8 Graph，3 次预热、5 轮 × 20 次取中位数。基线包 `cann_ops_transformer_qfa_fuse_cmp_20261117`，候选包 `cann_ops_transformer_qfa_dn_store_20261117`；12 项 case fingerprint 和输入 SHA 全部匹配。C4 耗时几何平均减少 0.167%，C8 对照减少 -0.155%；C4 prefill 变化 -0.18%..+0.18%，没有测到明确整算子加速。不能将未修改路径的小幅变化归因于 DN Store。

| case | 采集指标 | 状态 | 主算子中位数/μs |
| --- | --- | --- | ---: |
| dual_prefill_1 | PipeUtilization | validated | 1215.470 |
| dual_prefill_1 | Default | validated | 1218.495 |
| dual_prefill_1 | InstrTimeline | 无效，data loss 18 条 | — |
| decode_4 | PipeUtilization | validated | 51.790 |
| decode_4 | Default | validated | 63.262 |
| decode_4 | InstrTimeline | 无效，data loss 18 条 | — |

以上 profiler 只采候选，用于诊断，不与 Graph 计时混用。失效 InstrTimeline 不用于证明 bank 无冲突或逐指令性能收益。源码身份、24 份性能 JSON、JUnit、采集摘要及失败边界见[本次替换验证记录](../../../operators/quant_flash_attn/dn_pscale_store_20261117_card7_01/README.md)，逐例数字见[配对性能表](../../../operators/quant_flash_attn/dn_pscale_store_20261117_card7_01/performance-table.md)。


### ND Pscale 与 Softmax 指数：Scatter 替换为 Store（2026-11-17）

基线为上一轮已验收的 DN Store 包 `qfa_dn_store_20261117`，候选为 `qfa_nd_store_20261117`。`UpdatePScaleNdVf` 直接将 Cast 编码重排到物理 Q16 块；Task128 连续写 96B，Task256 合并相邻两路 Q8 后连续写 160B，均包含原有 32B 零 padding。每条 Store 的 3/5 个连续 32B 块落入不同 bank group；不将其外推为整个 kernel 无 bank 冲突。`SoftmaxQuantizeNdChunkFp32Vf` 的 5 处两 float Scatter 改为 `Reg::Store<float>(...,2)`，内部走非对齐 Store 和收尾；物理布局、量化公式及已有 DN 修改保留。

新旧实际 VF 字节模型各 19,200 组、地址审计 160 组通过。200 / `tp_a4c4_gqa_kernel` / 仅 NPU7：两版各 640 项独立 Pscale 通过，1,344 项 ND Softmax 全部输出字节一致；候选 document small 204/204 + bench16k 12/12 通过。复用已核验 CPU 缓存，使用 `--golden-mode npu,compare`，不重算 CPU reference、不跑 Triton。

独立 ND 实例 CANN 9.2 LLVM IR：70 个 Scatter 调用变为 0，Softmax 使用 20 组非对齐写/收尾，Pscale 使用对齐 Store。这里只说明编译器 lowering，不是最终机器逐指令时长。

同轮两版各 12 项 C4/C8 Graph，3 预热、5 轮×20 回放，完整 case 指纹与输入 SHA 一致，48 路输出有限/回放抽样一致/allocated 释放为 0。C4 耗时几何平均减少 -0.386%（decode -1.161%），C8 对照减少 +0.066%；负值代表变慢。未修改路径的小幅波动不能归因于 ND 替换。

| case | 指标 | 状态 | 主算子中位数/μs |
| --- | --- | --- | ---: |
| dual_prefill_1 | PipeUtilization | validated | 1213.502 |
| dual_prefill_1 | Default | validated | 1217.152 |
| dual_prefill_1 | InstrTimeline | 无效，data loss 18 条 | — |
| decode_4 | PipeUtilization | validated | 52.403 |
| decode_4 | Default | validated | 63.885 |
| decode_4 | InstrTimeline | 无效，data loss 18 条 | — |

profiler 仅候选 C4/eager，不与 Graph 混算收益；失效 InstrTimeline 不用于 bank 或完整指令时长归因。详见[本次 ND 验证记录](../../../operators/quant_flash_attn/nd_store_20261117_card7_01/README.md)及[12项配对性能](../../../operators/quant_flash_attn/nd_store_20261117_card7_01/performance-table.md)。
