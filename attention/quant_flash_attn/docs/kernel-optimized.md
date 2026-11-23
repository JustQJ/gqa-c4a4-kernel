# 性能优化记录

## 优化遵守标准

1. 精度要严格保证通过

    小的改动通过 tests\pytest\qfa_mxfp4_test\test_quant_flash_attn_func_rdv.py 的 --accuracy-case-set document --scope small 测试集 + --accuracy-case-set bench16k 测试集 + PA_BBND/PA_NZ/PA_BNBD 这几个layout都要通过

    多次大的改动要进行一次 tests\pytest\qfa_mxfp4_test\test_quant_flash_attn_func_rdv.py 的 --accuracy-case-set document --scope full 测试集

## 优化方向

1. vector的vf指令优化

2. scalar 的优化，调整基本块，减少循环的scalar指令

3. 小shape的特化


## D256 优化记录过程

截至 2026-11-18，`fea334e` 的 D256 DN 和 ND 路径均采用 QK Chunk128。[D256 Chunk128 最终结果](#d256-chunk128-最终结果)

性能测试使用 tests/pytest/qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py 的 `--perf-case-set bench16k --run-perf --perf-backends c4,c8 --perf-mode graph --perf-warmup 3 --perf-rounds 5 --perf-repeats 20` 12 个case进行测试。

使用 `prof_quant_flash_attn.py` 对重点用例做 eager 采集。最终版 `dual_prefill_1` 使用 msprof 的 PipeUtilization 和 msprof op 的 Default，两种工具均采集了 C4 和 C8。[采集脚本](../tests/pytest/qfa_mxfp4_test/prof_quant_flash_attn.py)

### 总表记录

| 阶段 | 改动 | 数据来源 |
| --- | --- | --- |
| Base | 输出高精度 Div，`precisionMode=true` | 2026-11-16 的 A/B/C 三个批次汇总；`decode_3` 使用短尾最新参数 |
| 普通 Div + 倒数复用 | 关闭输出 Div 高精度修正，`precisionMode=false`；DN/ND 输出复用倒数，一次除法后各通道乘法 | 2026-11-17 的 `performance-candidate`，包 `cann_ops_transformer_qfa_recip_bench16k_20261117` |
| 再加 ExpSub + 乘加 | 在普通 Div + 倒数复用基础上，追加 37 处 Sub→Exp 和 11 处乘加融合 | 2026-11-17 融合实验的 `performance-compound`，包 `cann_ops_transformer_qfa_fuse_cmp_20261117` |
| DN128/ND128（最终版） | D256 的 DN 和 ND 均采用 QK Chunk128，QK L0 使用单槽，输出暂存复用已消费的 PV 缓冲区 | 2026-11-18 ND 配对实验的 B 版，提交 `fea334e` |

各阶段均在 200 / `tp_a4c4_gqa_kernel` / NPU7 上使用 schema 2 和 Graph 回放 + Event 计时。每版预热 3 次，测量 5 轮，每轮 20 次调用，取中位数。输入准备、metadata 和格式转换在计时外。固定 Nq16/Nkv2/D256/page128、mask3、LSE 关闭，K/V 各使用 31374 页固定池。

Base 汇总了三个历史批次。12 个 case 的指纹在各阶段一致，Base 和最终版的输入 SHA 也逐项匹配。表中耗时单位为 μs，case 省略 `bench16k_` 前缀。降幅使用未舍入值计算，正值表示耗时减少，负值表示耗时增加。相对 Base 的累计降幅来自跨批次比较，Chunk128 的增量收益见各自配对结果。[分阶段配对结果](#graph-性能)

「Base C8/融合 C4」和「Base C8/最终 C4」均用表中的历史 Base C8 耗时除以对应阶段的 C4 耗时。比值大于 1 表示对应 C4 的耗时低于历史 Base C8。

| case | Base C4 | Base C8 | 普通 Div + 倒数复用 C4 | ExpSub + 乘加 C4 | 融合较 Base 降幅 | Base C8/融合 C4 | DN128/ND128 C4 | 最终较 Base 降幅 | Base C8/最终 C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 318.742 | 212.413 | 246.134 | 240.429 | 24.57% | 0.883 | 219.154 | 31.24% | 0.9692 |
| single_prefill_2 | 1045.231 | 937.282 | 994.794 | 991.298 | 5.16% | 0.946 | 831.003 | 20.50% | 1.1279 |
| single_prefill_3 | 1435.241 | 1300.633 | 1385.437 | 1381.087 | 3.77% | 0.942 | 1149.859 | 19.88% | 1.1311 |
| dual_prefill_1 | 1544.096 | 1132.345 | 1230.413 | 1212.775 | 21.46% | 0.934 | 1085.216 | 29.72% | 1.0434 |
| dual_prefill_2 | 1431.694 | 1251.043 | 1324.859 | 1319.603 | 7.83% | 0.948 | 1122.707 | 21.58% | 1.1143 |
| dual_prefill_3 | 2273.021 | 2089.713 | 2194.643 | 2188.596 | 3.71% | 0.955 | 1814.609 | 20.17% | 1.1516 |
| decode_1 | 330.693 | 355.821 | 330.377 | 330.917 | -0.07% | 1.075 | 329.370 | 0.40% | 1.0803 |
| decode_2 | 715.852 | 914.377 | 712.098 | 712.165 | 0.52% | 1.284 | 708.238 | 1.06% | 1.2911 |
| decode_3 | 48.793 | 41.291 | 48.682 | 48.386 | 0.83% | 0.853 | 41.374 | 15.20% | 0.9980 |
| decode_4 | 52.640 | 50.420 | 53.858 | 54.066 | -2.71% | 0.933 | 47.600 | 9.58% | 1.0592 |
| mixed_1 | 1361.233 | 1371.783 | 1322.265 | 1316.726 | 3.27% | 1.042 | 1250.454 | 8.14% | 1.0970 |
| mixed_2 | 3587.749 | 2972.840 | 3536.840 | 3529.779 | 1.62% | 0.842 | 3292.170 | 8.24% | 0.9030 |

分组按 case 等权计算耗时比的几何平均。「Base C4」作分子时，比值大于 1 表示对应阶段比原始 Base 更快。「Base C8/最终 C4」使用历史 Base C8 与最终版 C4 的跨批次数据。

| 范围 | case 数 | Base/倒数复用 | Base/融合后 | Base C8/C4 | Base C8/融合 C4 | Base C4/最终 C4 | Base C8/最终 C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 12 | 1.0611 | 1.0663 | 0.9037 | 0.9636 | 1.1910 | 1.0764 |
| prefill | 6 | 1.1206 | 1.1302 | 0.8266 | 0.9342 | 1.3158 | 1.0877 |
| decode | 4 | 0.9964 | 0.9965 | 1.0274 | 1.0238 | 1.0726 | 1.1019 |
| mixed | 2 | 1.0219 | 1.0251 | 0.9138 | 0.9367 | 1.0892 | 0.9953 |

最终版 C4 相对历史 Base 的耗时比几何平均为 1.1910×，折合耗时减少 16.04%。`dual_prefill_1` 从 1544.096 μs 降至 1085.216 μs，减少 29.72%。对照历史 Base C8，最终 C4 在 12 例中有 9 例耗时更短，耗时比的几何平均为 1.0764×。

[Base 原始汇总](../../../operators/quant_flash_attn/bench16k_performance_summary_20261116/README.md) · [倒数复用阶段](../../../operators/quant_flash_attn/reciprocal_bench16k_20261117_card7_01/README.md) · [联合融合结果快照](../../../operators/quant_flash_attn/vf_compound_fusion_20261117_card7_01/remote-status-20261117.json) · [DN128/ND128 数据](../../../operators/quant_flash_attn/d256_results_20261118_card7_01/final-graph-results.csv)

#### PR2 三种 KV 布局的 bench16k 性能（2026-11-19）

使用 PR2 合并版本 `0e86482fd`，在 200 / `tp_a4c4_gqa_kernel` / NPU7 上对三种 KV 布局分别测量 C4 和 C8。12 个用例的源输入哈希在各布局间一致，共 36/36 个用例通过、72 路计时。C4 使用 schema 2。沿用上文的 Graph + Event 口径，预热 3 次，测量 5 轮，每轮 20 次调用，取中位数。输入准备、metadata 和布局转换在计时外。

下表单位为 μs，case 省略 `bench16k_` 前缀。

| case | PA_BBND C4 | PA_BBND C8 | PA_BNBD C4 | PA_BNBD C8 | PA_NZ C4 | PA_NZ C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 219.656 | 213.216 | 217.515 | 213.083 | 217.324 | 213.054 |
| single_prefill_2 | 832.201 | 938.431 | 828.179 | 938.285 | 829.692 | 936.724 |
| single_prefill_3 | 1166.990 | 1301.799 | 1147.919 | 1300.982 | 1153.668 | 1299.877 |
| dual_prefill_1 | 1086.719 | 1133.546 | 1087.749 | 1134.091 | 1085.387 | 1134.495 |
| dual_prefill_2 | 1121.293 | 1249.714 | 1121.891 | 1250.409 | 1122.541 | 1251.866 |
| dual_prefill_3 | 1824.575 | 2092.004 | 1813.151 | 2069.051 | 1819.610 | 2063.871 |
| decode_1 | 330.474 | 357.383 | 212.314 | 357.878 | 202.930 | 354.760 |
| decode_2 | 711.862 | 915.321 | 522.334 | 904.199 | 494.493 | 905.681 |
| decode_3 | 42.242 | 41.468 | 41.472 | 40.747 | 41.331 | 42.130 |
| decode_4 | 46.926 | 51.444 | 45.962 | 48.470 | 46.062 | 47.906 |
| mixed_1 | 1250.207 | 1378.764 | 1055.749 | 1281.337 | 1062.864 | 1190.715 |
| mixed_2 | 3300.406 | 2971.165 | 2837.765 | 2902.624 | 2880.708 | 2714.928 |

分组按 case 等权计算 C8 耗时 / C4 耗时的几何平均。比值大于 1 表示 C4 更快。

| KV 布局 | all（12） | prefill（6） | decode（4） | mixed（2） | C4 更快 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PA_BBND` | 1.0764 | 1.0846 | 1.1060 | 0.9964 | 9/12 |
| `PA_BNBD` | 1.1652 | 1.0890 | 1.3186 | 1.1142 | 10/12 |
| `PA_NZ` | 1.1596 | 1.0869 | 1.3573 | 1.0275 | 10/12 |

相对 `PA_BBND`，`PA_BNBD` 和 `PA_NZ` 的 C4 整体耗时分别降低 9.12% 和 9.66%，decode 耗时分别降低 17.95% 和 20.00%。

逐例 CSV、统计 JSON 和原始日志保存在本地工作区的 `operators/quant_flash_attn/pr2_bench16k_card7_20261119_zOKFAA`。

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

### D256 Chunk128 最终结果

#### 实现与精度验证

`fea334e` 将 D256 DN/ND 的 QK 子块由 64 增至 128，沿用 TaskN128、每组 16 个 task 和 20 个 L1 P 槽。QK 的 L0A/L0B/L0C 使用单槽，AIV 仍使用两个 score 接收槽。V2 消费 PV 结果后，输出转换和统计量复用该 PV 区域。普通输出和 Flash Decode（FD）的 MTE3 搬运完成后，AIV 才向 AIC 归还该缓冲区的写入权限。DN/ND 的每 AIV UB 占用分别为 242944/244736 字节，均在 248 KiB 内。D128 Chunk256 实验已撤回，D128 使用 Chunk128、双 QK L0 槽和独立输出暂存区。[Cube 实现](../op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h) · [Vector 实现](../op_kernel/arch35/quant_flash_attn_block_vec_mxfp4.h)

DN 候选与最终 DN/ND 版本均通过 schema 2 的 `document small` 204/204 和 `bench16k` 12/12。最终版额外通过 2 项 ND 冒烟测试，覆盖 FD 和非 FD 路径。测试复用经核验的 CPU 参考数据和原比较门限。111 项 D256 性能矩阵和 15228 项三 schema 全量精度待验收。[DN 验证记录](../../../operators/quant_flash_attn/scalar_chunk128_20261117_card7_01/RESULTS.md) · [ND 验证记录](../../../operators/quant_flash_attn/scalar_nd_chunk128_20261117_card7_01/RESULTS.md)

#### Graph 性能

两轮均在 200 机器的 `tp_a4c4_gqa_kernel` 容器内用物理卡 7 测试。每版预热 3 次，测量 5 轮，每轮 20 次回放。基线取前后两轮样本的合并中位数。逐例输入 SHA 和参数指纹一致，各轮 C8 对照波动在 ±3% 内。两列分别以各阶段的基线计算等权几何平均速度比。

| 范围 | 用例数 | DN64/ND64 → DN128/ND64 | DN128/ND64 → DN128/ND128 |
| --- | ---: | ---: | ---: |
| 全部 | 12 | 1.090799× | 1.021828× |
| Prefill | 6 | 1.162209× | 1.001360× |
| Decode | 4 | 1.004804× | 1.064361× |
| Mixed | 2 | 1.062808× | 1.000737× |

DN 阶段的 `dual_prefill_1` 从 1212.971 μs 降至 1086.339 μs，耗时减少 10.44%。随后 ND 阶段的收益主要来自两个短 decode。三组 AB/BA/AB 配对中，`decode_3` 的速度比为 1.144223×、1.148837×、1.146647×，`decode_4` 为 1.078997×、1.076679×、1.084043×。

| case | DN128/ND64 C4（μs） | 最终 C4（μs） | 同轮 C8（μs） | C8/最终 C4 |
| --- | ---: | ---: | ---: | ---: |
| single_prefill_1 | 220.471 | 219.154 | 212.908 | 0.9715 |
| single_prefill_2 | 831.180 | 831.003 | 938.075 | 1.1288 |
| single_prefill_3 | 1150.504 | 1149.859 | 1300.983 | 1.1314 |
| dual_prefill_1 | 1085.286 | 1085.216 | 1131.920 | 1.0430 |
| dual_prefill_2 | 1122.837 | 1122.707 | 1251.480 | 1.1147 |
| dual_prefill_3 | 1816.800 | 1814.609 | 2080.807 | 1.1467 |
| decode_1 | 331.537 | 329.370 | 356.051 | 1.0810 |
| decode_2 | 710.778 | 708.238 | 915.105 | 1.2921 |
| decode_3 | 47.579 | 41.374 | 40.965 | 0.9901 |
| decode_4 | 52.587 | 47.600 | 49.559 | 1.0412 |
| mixed_1 | 1250.993 | 1250.454 | 1374.312 | 1.0990 |
| mixed_2 | 3295.604 | 3292.170 | 2973.068 | 0.9031 |

最终 C4 快于同轮 C8 的用例为 9/12，达到 C8/C4 ≥ 1.3 的用例为 0/12。最终版输出有限，Graph 回放抽样与 eager 一致，释放后 allocated 为 0。[逐例结果 CSV](../../../operators/quant_flash_attn/d256_results_20261118_card7_01/final-graph-results.csv)

#### dual_prefill_1 的 msprof 与 msprof op（2026-11-18）

使用最终版安装包在同一张卡上采集 C4 和 C8。输入为 D256、B2、TND，Q 长度 `[4096, 3988]`，KV 长度 `[4096, 12056]`，16 个 Q 头、2 个 KV 头，page128，固定 31374 页，mask3，LSE 关闭，seed421。C4 使用 schema 2，C8 的 `p_scale` 为 448。四路输入 SHA 和 case 指纹均与 Graph 主例一致，msprof op 的 kernel 名匹配最终包的安装对象。

msprof / PipeUtilization 每侧采集 20 次应用预热及 5 次目标调用。25 个目标任务均已匹配调用顺序，下表取后 5 次的中位数。msprof op / Default 在 20 次应用预热后使用 10 次工具预热和 5 次抓取，工作频率为 1650 MHz。任务耗时取 5 个 `OpBasicInfo` 样本的中位数，流水指标取逐核均值，每侧含 160 个 AIC 样本和 320 个 AIV 样本。

| 采集口径 | 指标 | C4（μs） | C8（μs） |
| --- | --- | ---: | ---: |
| msprof / PipeUtilization | 任务耗时 | 1084.645 | 1133.513 |
| msprof / PipeUtilization | AIC Scalar | 1010.313 | 1052.702 |
| msprof / PipeUtilization | AIV Scalar | 816.704 | 711.113 |
| msprof / PipeUtilization | AIV Vector | 871.293 | 951.340 |
| msprof / PipeUtilization | AIC MAC | 710.423 | 1028.313 |
| msprof op / Default | 任务耗时 | 1089.663 | 1137.755 |
| msprof op / Default | AIC Scalar | 930.283 | 1031.460 |
| msprof op / Default | AIV Scalar | 817.189 | 711.302 |
| msprof op / Default | AIC Cube | 490.798 | 961.135 |
| msprof op / Default | AIV Vector | 872.245 | 951.801 |
| msprof op / Default | AIC MTE2 | 865.030 | 959.925 |
| msprof op / Default | AIV MTE2 | 5.741 | 5.988 |

同轮 C8 对比中，最终 C4 的 msprof 任务耗时低 4.31%。两个 profiler 均观测到 C4 的 AIC Scalar 和 AIV Vector 耗时更低、AIV Scalar 耗时更高。历史 DN64→DN128 配对测得 C4 的 msprof 任务耗时从 1214.181 μs 降至 1085.908 μs，AIC Scalar 降低 15.11%。同轮 msprof op 的逐核 AIC/AIV Scalar 均值分别降低 18.66% 和 8.10%。各流水存在重叠，任务耗时以各工具的总耗时字段为准。[同轮对照明细](c4a4-d256-results-20261118.md)

四路采集均为 `validated`，无数据丢失，输出有限且释放后 allocated 为 0。两路 Default 各完整抓取 5 次，每次含 7 类指标。原始采集包的 412 个成员已逐项核对 SHA256、大小和成员集合。[复算与原始指标](../../../operators/quant_flash_attn/d256_results_20261118_card7_01/results-audit.json) · [原始采集包](../../../operators/quant_flash_attn/d256_results_20261118_card7_01/collection-evidence.tar.gz)

### V 补列紧凑化：InitPaddingL0 只搬 16 通道补列（2026-11-20）

每个 PV 块原先无条件执行 `InitPaddingL0`，把整个 `N×(D+16)` 常数模板从 L1 搬进 L0 V 槽，而 `LoadV2L0` 随后只覆盖真实 D 列，补列之外的全部搬运是浪费。本轮把 L1 常数模板缩为 16 通道紧凑布局（`s2BaseSize×16`），`InitPaddingL0` 改用 C API 只搬补列分形：payload 走 `asc_copy_l12l0a`/`asc_copy_l12l0b` 的 fp4 重载直接写到 V 槽内 D/16 号分形，MX scale 走 `asc_copy_l12l0a_mx`/`asc_copy_l12l0b_mx` 独立寻址到补列 scale 分形。DN/ND 两路径均改；L1 常数区从 ~17–19 KiB 降到 1–2.25 KiB。

A/B 定义：A = 基线包 `cann_ops_transformer_qfa_base_20261119`（HEAD `90dbcef38`），B = 候选包 `cann_ops_transformer_qfa_compact_pad_20261119`（仅 `quant_flash_attn_block_cube_mxfp4.h` 不同）。测试在 184 机器 `tp_a4c4_gqa_kernel_ops` 容器，CANN 9.2.1，精度在物理卡 6、性能与 PMU 在物理卡 4 采集。A/B 两包的输入 SHA256 与 case 指纹逐例一致（12/12）。

#### 精度

按验收标准执行 document small + bench16k 两套 case-set × PA_BBND/PA_BNBD/PA_NZ 三种 layout，共 6 批次：

| 批次 | 结果 |
| --- | --- |
| document small × PA_BBND | 204 passed |
| bench16k × PA_BBND | 12 passed |
| document small × PA_BNBD | 204 passed |
| bench16k × PA_BNBD | 12 passed |
| document small × PA_NZ | 204 passed |
| bench16k × PA_NZ | 12 passed |

#### Graph 性能

12 case、graph 模式（预热 3、5 轮 × 20 回放取中位数），与总表口径一致。C4 全部 12 例几何平均 1.0217×。

| case | C4 A（μs） | C4 B（μs） | 变化 |
| --- | ---: | ---: | ---: |
| single_prefill_1 | 262.2 | 260.2 | -0.74% |
| single_prefill_2 | 966.1 | 968.2 | +0.21% |
| single_prefill_3 | 1348.1 | 1350.0 | +0.14% |
| dual_prefill_1 | 1247.8 | 1246.7 | -0.09% |
| dual_prefill_2 | 1382.0 | 1293.0 | -6.44% |
| dual_prefill_3 | 2490.3 | 2355.4 | -5.42% |
| decode_1 | 218.8 | 214.8 | -1.85% |
| decode_2 | 521.3 | 517.8 | -0.67% |
| decode_3 | 48.2 | 48.9 | +1.51% |
| decode_4 | 61.4 | 60.3 | -1.65% |
| mixed_1 | 1296.6 | 1239.5 | -4.40% |
| mixed_2 | 3607.1 | 3402.9 | -5.66% |

收益集中在 KV 体量大的 prefill/mixed 用例（-4.4% ~ -6.4%）；`dual_prefill_1` 与 decode 组在噪声带内。本轮噪声带用同包复测（A2）标定：均值 -0.54%、最大 |Δ| 5.34%、标准差 2.0（在不受本改动影响的 C8 路径上测得，跨包 B/A 分布与之不可区分）。解读 C4 单例时应对照该噪声带；重 prefill 4 例方向一致且幅度超出噪声带。

#### dual_prefill_1 的 MTE1 读写量复算

`msprof op --aic-metrics=Default` 经兼容适配器采集（该 CANN 9.2.1 环境下 MSTX 门控产出空数据，须去掉 `--mstx*` 并将内核名写为 `QuantFlashAttn_*` 通配），两侧各 5 次抓取、28 个 AIC 核、1650 MHz。输入为 bench16k_dual_prefill_1（B2、Q[4096,3988]、KV[4096,12056]、D256、page128、PA_NZ、schema2），A/B 的 kernel 对象与 tiling-key 一致。逐核均值：

| 指标 | A 基线 | B 紧凑补列 | 变化 |
| --- | ---: | ---: | ---: |
| MTE1 活跃带宽×时间（搬运量） | 190.41 MiB | 148.87 MiB | -21.8% |
| L1 读（MTE1 读侧） | 174.88 MiB | 137.02 MiB | -21.7% |
| L0A+L0B 写（MTE1 写侧） | 91.15 MiB | 65.86 MiB | -27.8% |
| MTE1 活跃时间 | 615.61 μs | 511.59 μs | -16.9% |
| MTE2 读量（对照，应不变） | 60.06 MiB | 60.05 MiB | -0.02% |
| MTE2 活跃时间（对照） | 682.12 μs | 688.00 μs | +0.86% |
| AIC 任务耗时 | 1152.16 μs | 1152.56 μs | +0.03% |

MTE1 读/写降幅与事前估算的 -23.2%/-29.6% 基本吻合（估算按 32 核摊分，实测该 shape 为 28 核，单核份额略高）。MTE2 读量与 AIC 任务耗时不变，证明改动只作用于 L1→L0 的 padding 搬运，不影响 GM→L1 流量与计算排布；`dual_prefill_1` 本身是 cube 受限用例，收益主要体现在重 prefill/mixed 用例上。

[逐例性能 JSON 与 PMU 复算](../../../operators/quant_flash_attn/compact_pad_20261119_card4_01/) · [PMU 原始采集包](../../../operators/quant_flash_attn/compact_pad_20261119_card4_01/pmu-evidence.tar.gz)

### 统一一拍 KV 预取后的主分支验证（2026-11-23）

主分支 `quant-flash-attn-c4a4-handwrite-softmaxfp32` 的提交 `8e0989914` 统一使用一拍 K/V 预取。按该提交重新编译 MXFP4 算子，在 `195.121.2.18` 的 `tp_a4c4_gqa_kernel_ops1` 容器验证，CANN 9.2.1。PA_NZ、PA_BNBD、PA_BBND 分别使用 4、5、6 号卡，Qwen3-8B 页长 256 的补充精度测试使用 7 号卡。

#### 精度

三种布局合计 **744/744 通过**。全部复用已有 CPU reference。Qwen3-8B 页长 256 的输入按逻辑 KV 顺序重新分页，逐块核对 payload 和 scale 后与原 CPU 输出比较。

| 测试集 | 每布局 | 三布局合计 |
| --- | --- | --- |
| document small | 204 | 612 |
| bench16k | 12 | 36 |
| Qwen3-8B，页长 128 | 16 | 48 |
| Qwen3-8B，页长 256 | 16 | 48 |
| 合计 | 248 | 744 |

#### bench16k Graph 性能

D256、页长 128、schema 2。12 个用例各测三种布局，同一组合的 C4 与 C8 使用相同输入和设备。预热 3 次，再测 5 轮，每轮 20 次 Graph replay。下表为每轮单次 attention 调用耗时的中位数，单位 µs，输入准备和布局转换在计时外。

逐例 C8/C4 为同一用例、同一布局下的 C8 耗时除以 C4 耗时，按未舍入的中位数计算。比值大于 1 表示 C4 更快。

| 用例 | PA_NZ C4 | PA_NZ C8 | PA_NZ C8/C4 | PA_BNBD C4 | PA_BNBD C8 | PA_BNBD C8/C4 | PA_BBND C4 | PA_BBND C8 | PA_BBND C8/C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_1 | 162.5 | 176.3 | 1.085× | 166.1 | 199.3 | 1.200× | 202.3 | 195.7 | 0.967× |
| decode_2 | 397.4 | 504.2 | 1.269× | 398.5 | 502.1 | 1.260× | 508.4 | 497.5 | 0.979× |
| decode_3 | 39.3 | 42.5 | 1.082× | 39.4 | 40.5 | 1.029× | 40.3 | 41.1 | 1.018× |
| decode_4 | 47.8 | 48.7 | 1.018× | 47.5 | 49.5 | 1.043× | 48.7 | 47.9 | 0.982× |
| dual_prefill_1 | 1235.8 | 1507.8 | 1.220× | 1235.7 | 1399.6 | 1.133× | 1236.9 | 1462.3 | 1.182× |
| dual_prefill_2 | 1273.1 | 1659.0 | 1.303× | 1269.5 | 1579.7 | 1.244× | 1293.4 | 1601.4 | 1.238× |
| dual_prefill_3 | 2323.1 | 2779.0 | 1.196× | 2164.6 | 2664.8 | 1.231× | 2371.2 | 2694.0 | 1.136× |
| mixed_1 | 1158.4 | 1334.9 | 1.152× | 1161.5 | 1248.1 | 1.075× | 1295.7 | 1338.2 | 1.033× |
| mixed_2 | 3247.4 | 3386.0 | 1.043× | 3102.3 | 3292.3 | 1.061× | 3619.8 | 3426.4 | 0.947× |
| single_prefill_1 | 251.5 | 242.3 | 0.963× | 250.7 | 241.8 | 0.965× | 251.8 | 242.0 | 0.961× |
| single_prefill_2 | 951.8 | 1164.1 | 1.223× | 951.4 | 1067.5 | 1.122× | 952.8 | 1066.3 | 1.119× |
| single_prefill_3 | 1344.4 | 1709.2 | 1.271× | 1301.0 | 1633.1 | 1.255× | 1324.0 | 1635.9 | 1.236× |

下表按布局汇总，C8/C4 取逐用例耗时比的等权几何平均。末列是当前 C4 相对四拍配对预取提交 `1b1764dfc` 的耗时变化，负值表示耗时下降。历史记录与本次测量逐项匹配设备号和输入 SHA256，差异包含两版间的全部改动及跨批次波动。

| 布局 | 组合数 | C8/C4 | C4 较快组合数 | C4 耗时相对历史四拍版本 |
| --- | --- | --- | --- | --- |
| PA_NZ | 12 | 1.147× | 11 | -7.67% |
| PA_BNBD | 12 | 1.131× | 11 | -7.83% |
| PA_BBND | 12 | 1.061× | 7 | -9.13% |
| 全部 | 36 | 1.112× | 29 | -8.21% |

36 组中，C4 有 10 组、C8 有 18 组的五轮极差超过本组中位数的 5%。其中 `single_prefill_1` 在三个布局上均为 C8 更快。

#### dual_prefill_1 的 msprof 与 msprof op Roofline

本例为 C4、D256 DN、B=2，Q 长度为 [4096, 3988]，KV 长度为 [4096, 12056]。采集与 Graph 使用相同的用例指纹及输入 SHA256。以下时间单位均为 µs，各流水可重叠。

**msprof PipeUtilization** 从 `op_summary` 取最后 3 次目标调用，各列分别取中位数。三个布局中均为 AIC Scalar 时间最长，其次在 PA_NZ、PA_BNBD 上为 AIV Vector，在 PA_BBND 上为 AIC MTE2。

| 布局 | 整算子 | AIC MAC | AIC Scalar | AIC MTE1 | AIC MTE2 | AIV Vector | AIV Scalar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA_NZ | 1238.8 | 815.3 | 1066.3 | 570.6 | 596.8 | 993.2 | 887.4 |
| PA_BNBD | 1235.5 | 814.0 | 1079.7 | 524.1 | 705.2 | 992.8 | 884.2 |
| PA_BBND | 1235.8 | 812.2 | 1162.1 | 542.1 | 1046.4 | 992.9 | 884.4 |

**msprof op Roofline** 使用 `--aic-metrics=Roofline`，按 `QuantFlashAttn_*` 过滤，工具预热 3 次后采 3 次。每次先对各流水的有效核取均值，再对三次采样取中位数。应用为 eager 模式，预热 20 次。关闭 MSTX 后，内核名过滤可能命中应用预热阶段的调用。

| 布局 | 整算子 | AIC MAC | AIC Scalar | AIC MTE1 | AIC MTE2 | AIV Vector | AIV Scalar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA_NZ | 1239.8 | 560.9 | 977.7 | 522.7 | 557.0 | 993.8 | 887.7 |
| PA_BNBD | 1238.1 | 560.9 | 992.4 | 483.2 | 653.8 | 993.1 | 884.3 |
| PA_BBND | 1241.0 | 560.9 | 1067.2 | 493.7 | 961.4 | 993.1 | 884.5 |

**三个布局的 9 次 Roofline 采样均判为 `latency bound:compute caused`，尚未达到 Compute Bound。** Cube 性能比取自 Roofline 的 Cube 曲线。下表的流水占比和 AIC MTE2 活跃带宽取自同次 msprof op 采集的 PipeUtilization 数据，带宽单位为 GB/s。

| 布局 | Roofline 分类 | Cube 性能比 | AIC MAC 占比 | AIC Scalar 占比 | AIV Vector 占比 | AIC MTE2 活跃带宽 |
| --- | --- | --- | --- | --- | --- | --- |
| PA_NZ | latency bound:compute caused | 45.5% | 49.3% | 86.0% | 80.3% | 105.3 |
| PA_BNBD | latency bound:compute caused | 45.6% | 49.3% | 87.2% | 80.4% | 89.7 |
| PA_BBND | latency bound:compute caused | 45.5% | 49.2% | 93.7% | 80.2% | 119.1 |

三种布局的 AIC Scalar 占比均最高，AIV Vector 约为 80%。按 100 GB/s 的 AIC MTE2 活跃带宽目标，PA_NZ 和 PA_BBND 已达标，PA_BNBD 还差 10.3 GB/s。

本次实测归档在工作区中与仓库同级的 `qfa_8e0989914_validation_20261123/` 目录。逐例 Graph 数据见 `bench16k.csv`，流水样本见 `dual_prefill_1_samples.csv`，两种 profiler 的原始产物见 `dual/`。
