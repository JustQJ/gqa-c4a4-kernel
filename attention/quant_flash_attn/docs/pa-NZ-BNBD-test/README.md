# C4A4 PA_NZ / PA_BNBD 精度测试报告

## 1. 总结

**全部通过。**

- 三 schema 设备精度测试：**216/216 PASS**，0 failure、0 error、0 skipped；
- 新增布局精度：PA_BNBD 72/72、PA_NZ 72/72；
- 原 PA_BBND 回归：72/72；
- 两种新布局与 BBND 的最大 Attention relative-L2：**0**；
- 两种新布局与 BBND 的最大 LSE relative-L2：**0**；
- C4A4 与 C8A8 对拍：**4/4 PASS**；
- 离线布局转换：60/60 组逐字节可逆一致。

本次只执行原 BBND `small` 精度矩阵及 D128/D256 对拍补充，不将其表述为 `full` 全量验收。设备测试使用现有 CPU kernel reference 和阈值，未启用可选的 Triton reference；C4/C8 对拍另外使用未量化 P 的 FP64 参考。

## 2. 环境与设备选择

| 项目 | 值 |
|---|---|
| 宿主机/容器 | `ascend-950pr-223-2-2` / `k831` |
| NPU | Ascend 950PR，NPU 7 |
| 选卡依据 | 预检时无运行进程且 Health=OK |
| CANN | 9.2.1 |
| Python | 3.12.16，x86_64 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0.post7 |
| 算子包 | `cann_ops_transformer_c4a4` |
| 基线提交 | `cafc1498c` |

完整 NPU 快照、算子包及关键源码 SHA256 见 [environment.json](./environment.json)。所有构建、安装、JIT 缓存和测试输出均位于容器的 `/home/k831` 下；未停止或修改其他容器、进程。

## 3. 原布局支持结论

修改前，公共 QFA/C8A8 已支持 PA_BNBD 和 PA_NZ，但 C4A4 arch35 内核实际只支持 PA_BBND：PA_NZ 只有部分公共校验预留，PA_BNBD 未进入 C4 支持链路。本次同时补齐主算子、独立 metadata 算子、tiling 模板和 GM→L1 搬运后，两种布局均可执行。

## 4. 测试方法

### 4.1 BBND 对齐精度矩阵

每个 `quant_schema` 执行以下两个入口：

- 原 `test_quant_flash_attn_debug.py`：24 个 BBND 回归用例；
- 新 `test_quant_flash_attn_pa_layout.py`：相同配置下 BNBD 24 个 + NZ 24 个。

三种场景为 `smoke`、`empty_and_q_padding`、`page_tail`，覆盖：

- TND / N2TGD Q scale；
- mask mode 0 / 3；
- LSE on / off；
- page 128 / 256 / 512；
- 空行、Q padding、页尾、随机未使用页及随机页尾填充。

新用例复用原输入生成器、CPU kernel golden、输出检查和阈值。报告生成器逐案例读取三种布局缓存，先验证原始 Q/K/V、scale、页表及长度张量完全一致，再计算直接误差。

### 4.2 精度门限

原 BBND golden 门限保持不变：

- BF16 Attention：`atol=1e-4`、`rtol=0.0078125`；
- FP32 LSE：`atol=2.5e-5`、`rtol=0.005`；
- isclose 比例、最大相对误差、shape、有限值及空行语义沿用原 runner。

新增 BBND parity 门限：

```text
Attention relative-L2 <= 0.01
LSE relative-L2       <= 0.01
```

### 4.3 C4/C8 对拍

4 个案例覆盖 D128/D256 × PA_BNBD/PA_NZ。将同一 FP4 E2M1 payload 精确展开为 FP8 E4M3，保证 C4/C8 的 Q/K/V 数值完全一致；scale、页表、mask、长度和物理布局也一致。C8 `p_scale=448`。

报告同时比较：

- C4 vs BBND；
- C4 vs C8；
- C4/C8 vs 原量化 CPU 参考；
- C4/C8 vs FP64 未量化 P 参考。

C4 与 C8 使用不同 P 量化格式，因此不把 C4/C8 的直接 Attention 差异强制限定为 1%；1% 门限用于新布局与 BBND 的等价性验证。

## 5. 结果

### 5.1 三 schema 结果

| quant_schema | pytest | BBND 回归 | PA_BNBD | PA_NZ | 最大 Attention L2 | 最大 LSE L2 |
|---|---:|---:|---:|---:|---:|---:|
| 0 / OCP | 72/72 | 24/24 | 24/24 | 24/24 | 0 | 0 |
| 1 / CX Legacy | 72/72 | 24/24 | 24/24 | 24/24 | 0 | 0 |
| 2 / CX Delta | 72/72 | 24/24 | 24/24 | 24/24 | 0 | 0 |
| **合计** | **216/216** | **72/72** | **72/72** | **72/72** | **0** | **0** |

pytest 时间分别为 28.67s、36.34s、21.46s。

逐 schema 文件：

- [schema-0 pytest 日志](./schema-0/pytest.log)、[JUnit](./schema-0/pytest-junit.xml)、[parity 报告](./schema-0/report/pa_bbnd_parity.md)、[逐案例 JSON](./schema-0/report/pa_bbnd_parity.json)；
- [schema-1 pytest 日志](./schema-1/pytest.log)、[JUnit](./schema-1/pytest-junit.xml)、[parity 报告](./schema-1/report/pa_bbnd_parity.md)、[逐案例 JSON](./schema-1/report/pa_bbnd_parity.json)；
- [schema-2 pytest 日志](./schema-2/pytest.log)、[JUnit](./schema-2/pytest-junit.xml)、[parity 报告](./schema-2/report/pa_bbnd_parity.md)、[逐案例 JSON](./schema-2/report/pa_bbnd_parity.json)。

### 5.2 C4/C8 结果

| D / 布局 | C4/BBND Attention L2 | C4/C8 Attention L2 | C4/C8 LSE L2 | C4/FP64 L2 | C8/FP64 L2 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| D128 / BNBD、NZ | 0 | 0.0850652 | 0.000979246 | 0.0817984 | 0.0226986 | PASS |
| D256 / BNBD、NZ | 0 | 0.0963764 | 0.00118005 | 0.0928129 | 0.0251967 | PASS |

补充核验：D128 和 D256 下，C4、C8 各自的 BNBD/NZ 输出均**逐位一致**。C4/C8 的 Attention 差异来自 P 量化路径，而非布局；C8 与 FP64 未量化 P 参考更接近。

详见 [C4/C8 汇总报告](./c8-comparison/summary.md)、[汇总 JSON](./c8-comparison/summary.json) 和该目录下 4 份逐案例 JSON；完整执行输出见 [c8-comparison.log](./c8-comparison.log)。

### 5.3 离线布局转换

对 page 64/128/256/512/1024、D128/D256、payload/K scale/V scale、BNBD/NZ 共 60 组输入执行 BBND→目标布局→BBND 转换，全部逐字节一致。该检查用于排除测试侧 permute/reshape 本身造成的数据重排错误。

## 6. 复现入口

测试源码：

- [PA 参数集](../../tests/pytest/qfa_mxfp4_test/quant_flash_attn_paramset_pa_layout.py)；
- [PA pytest 入口](../../tests/pytest/qfa_mxfp4_test/test_quant_flash_attn_pa_layout.py)；
- [布局转换与 NPU 调用](../../tests/pytest/qfa_mxfp4_test/common/quant_flash_attn_golden.py)；
- [C4/C8 对拍程序](../../tests/pytest/qfa_mxfp4_test/common/pa_layout_c8_compare.py)；
- [BBND parity 报告生成器](../../tests/pytest/qfa_mxfp4_test/common/pa_layout_report.py)。

核心命令（在 `attention/quant_flash_attn/tests/pytest` 下）：

```bash
python3 -m pytest \
  qfa_mxfp4_test/test_quant_flash_attn_debug.py \
  qfa_mxfp4_test/test_quant_flash_attn_pa_layout.py \
  --scope small --quant-schema 2 --golden-mode all \
  --reference-backend kernel --reference-threads 8 \
  --device-id 7 --package cann_ops_transformer_c4a4 \
  --cache-dir /home/k831/work/results/schema-2/cache \
  --junitxml /home/k831/work/results/schema-2/pytest-junit.xml -q --tb=short -x

python3 -m qfa_mxfp4_test.common.pa_layout_report \
  --scope small --quant-schema 2 --package cann_ops_transformer_c4a4 \
  --cache-dir /home/k831/work/results/schema-2/cache \
  --output-dir /home/k831/work/results/schema-2/report

python3 -m qfa_mxfp4_test.common.pa_layout_c8_compare \
  --device-id 7 --package cann_ops_transformer_c4a4 \
  --output-dir /home/k831/work/results/c8-comparison
```

将 `quant-schema` 和目录中的 `2` 分别替换为 `0`、`1` 可复现另外两组。

## 7. 开发文档

完整设计思想、布局定义、地址公式、源码修改清单和关键注意事项见 [C4A4 PA_NZ / PA_BNBD 开发文档](../c4a4-pa-nz-bnbd-development.md)。
