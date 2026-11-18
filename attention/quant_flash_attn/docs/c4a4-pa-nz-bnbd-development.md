# C4A4 PA_NZ / PA_BNBD 开发文档

## 1. 任务范围与结论

本次开发对象为 `attention/quant_flash_attn` 的 C4A4 MXFP4 算子，目标平台仅为昇腾 A950/Ascend 950PR，对应源码中的 `arch35`、`DAV_3510` 和 `quant_mode=5`（`A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP32`）。

开发目标是：

- 保持既有 PA_BBND、C8A8 和其他量化路径不变；
- 为 C4A4 增加 PA_BNBD、PA_NZ KV cache 物理布局；
- 复用原 BBND 精度用例、输入、golden 和阈值；
- 新布局与 PA_BBND 的 Attention/LSE relative-L2 误差均不超过 1%；
- 增加 C4A4 与 C8A8 在相同数值输入、相同布局下的精度对比。

详细实测结果和逐案例数据存放在 [`pa-NZ-BNBD-test`](./pa-NZ-BNBD-test/) 目录。

## 2. 修改前的布局支持现状

需要区分“公共框架声明支持”和“C4A4 内核实际可执行”：

| 布局 | 公共 QFA/C8A8 | 修改前 C4A4 arch35 实际情况 |
|---|---|---|
| PA_BBND | 支持 | 已支持，是现有 C4A4 精度基线 |
| PA_BNBD | 支持 | 不支持；C4 layout 表、descale shape 和 kernel 均未开放 |
| PA_NZ | 支持 | 公共 checker 有部分预留，但 C4 tiling 和 kernel 仅接受 BBND，不能执行 |

因此，修改前 C4A4 真正可用的 PA KV cache 只有 PA_BBND；本次把 C8A8 已有的 PA_BNBD/PA_NZ 布局语义补齐到 C4A4，而不是引入新的外部布局定义。

## 3. 总体设计思想

### 3.1 计算路径不变，只扩展 GM 到 L1 的布局适配

C4A4 原有 QK、PV、P 量化、FP32 Softmax、FA/FD 合并和输出逻辑均不改。三种 GM 布局在搬运后统一转换为原 PA_BBND 使用的 L1 格式：

- K/V payload：以 32B 为内径的 NZ 分形；
- K scale：`[ceil(S/16), D/64, 16, 2]`；
- V scale：`[Dv/16, S2/64, 16, 2]`。

这样从 L0 装载开始，三种布局复用完全相同的计算输入，从设计上消除由 kernel 数值路径分支引入的精度差异。

### 3.2 BBND/BNBD 转换，NZ 直拷

- PA_BBND：保留原 ND-to-NZ 搬运逻辑。
- PA_BNBD：调整 page/head/token 的寻址关系后，复用 ND-to-NZ 搬运。
- PA_NZ：GM 与 L1 都是 32B 内径分形，使用 `DataCopyParams` 跨页、跨 D1 直拷，避免重复格式转换。
- K/V scale：BBND/BNBD 沿用 DN/ND-to-NZ；NZ 按 16-token 或 64-token 分组直拷。

### 3.3 逻辑元素和物理字节严格分离

FP4 API shape/stride 使用逻辑元素，kernel 的 `GlobalTensor<uint8_t>` 使用打包字节：

- 2 个 FP4 逻辑元素占 1 字节；
- FP4 NZ 的 D0 为 64 个逻辑元素，对应 32 个物理字节；
- payload 的 page/head stride 从 host 传入时需要除以 2；
- E8M0 scale 在 kernel 中按一对 `uint16_t` 处理，因此 scale stride 也转换为 pair 单位。

这是本次开发最关键的边界条件。

## 4. 外部数据布局约定

以下 `D` 为逻辑 head dim，`Bs` 为 page/block size，`N` 为 KV head 数，`Bn` 为物理页数。FP4 payload 在 CPU 测试侧以 `uint8_t` 保存，最后一维是打包字节数；进入 NPU 前再 view 为 FP4 类型。

| 数据 | PA_BBND | PA_BNBD | PA_NZ（物理 uint8 视角） |
|---|---|---|---|
| K/V payload | `[Bn,Bs,N,D/2]` | `[Bn,N,Bs,D/2]` | `[Bn,N,D/64,Bs,32]` |
| K scale | `[Bn,Bs,N,D/64,2]` | `[Bn,N,Bs,D/64,2]` | `[Bn,N,Bs/16,D/64,16,2]` |
| V scale | `[Bn,Bs/64,N,Dv,2]` | `[Bn,N,Bs/64,Dv,2]` | `[Bn,N,Dv/16,Bs/64,16,2]` |

PA_NZ payload 在 view 为 FP4 后，最后一维由 32 字节解释为 64 个 FP4 逻辑元素，因此 host shape checker 使用 `[Bn,N,D/64,Bs,64]`。

## 5. 源码修改清单

### 5.1 Host 校验与 tiling

#### `op_host/checkers/quant_checker.cpp`

- C4A4 layout 白名单由 `{PA_BBND, PA_NZ}` 扩展为 `{PA_BBND, PA_BNBD, PA_NZ}`；
- K/V descale 维度校验改为：BBND/BNBD 为 5D，NZ 为 6D；
- 增加 PA_BNBD 的 K scale、V scale 期望 shape；
- 保留 PA_NZ FP4 `D0=64` 的既有正确处理。

#### `op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp`

- tiling 入口由“只允许 PA_BBND”改为允许三种 PA 布局；
- 既有 `KvLayoutType` 和 `paLayoutType` 映射保持不变：BBND=1、BNBD=2、NZ=3；
- 不改调度、workspace、FlashDecode 和其他量化模式。

#### `op_kernel/arch35/quant_flash_attn_template_tiling_key.h`

- MXFP4 模板注册增加 `KvLayoutType_PA_BNBD`、`KvLayoutType_PA_NZ`；
- 原 PA_BBND 和其他量化模板不变。

#### `../quant_flash_attn_metadata/op_host/quant_flash_attn_metadata_check.h`

- 独立 metadata 算子的 MxFP4 layout 白名单同步增加 PA_BNBD；
- 调度生成逻辑不变，BBND/BNBD/NZ 共用同一套 Q/KV 长度与任务划分语义。

### 5.2 Kernel 布局适配

#### `op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h`

主要修改以下函数：

- `CopyKvPayloadGm2L1`：增加 BNBD/NZ 寻址和 NZ 分形直拷，并新增 head stride 参数；
- `CopyKGm2L1` / `CopyVGm2L1`：同时传入 page stride 和 head stride；
- `CopyKScaleGm2L1`：支持 BBND/BNBD 的 DN-to-NZ 与 NZ 直接搬运；
- `CopyVScaleGm2L1`：支持 BBND/BNBD 的 ND-to-NZ 与 NZ 按 D1/S64 分形直接搬运；
- 模板静态约束由 `KvLayoutType == 1` 放宽为 `1..3`，仍强制 Q 为 TND。

PA_BNBD payload 的核心地址关系为：

```text
physical_page * page_byte_stride
+ kv_head * head_byte_stride
+ page_token * row_byte_count
```

PA_NZ payload 以 D1 为外层搬运：

```text
physical_page * page_byte_stride
+ kv_head * head_byte_stride
+ page_token * 32B
```

每个 D1 分形搬运 `pageTokenCount` 个 32B block，源步长为 `Bs-pageTokenCount`，目标步长为 `S2Base-pageTokenCount`。

K scale NZ 每次复制若干完整的 16-token 分形；V scale NZ 以 `Dv/16` 为 `blockCount`，每个 64-token 组对应一个 32B scale 分形。

### 5.3 测试与报告

#### `tests/pytest/qfa_mxfp4_test/quant_flash_attn_paramset_common.py`

- 参数校验开放三种 PA 布局；
- `build_paramset` 新增可选 `kv_layouts`，默认值仍为 `('PA_BBND',)`，所以原有用例名称、数量和执行方式不变。

#### `tests/pytest/qfa_mxfp4_test/common/quant_flash_attn_golden.py`

- 新增 `convert_bbnd_tensor`，仅在 NPU 输入准备阶段转换 payload/K scale/V scale；
- CPU golden 和 Triton golden 始终读取原始 BBND 数据，避免参考结果被转换逻辑污染；
- `layout_kv` 改为从当前用例参数透传。

#### `tests/pytest/qfa_mxfp4_test/common/mxfp4_schedule.py`

- metadata 的 `layout_kv` 从当前用例参数透传；
- 调度模型本身不变。

#### 新增 `quant_flash_attn_paramset_pa_layout.py`

- 直接复用 BBND debug 的三个场景、模型、Q scale、mask、LSE 和 page 配置；
- small 范围：BNBD 24 项 + NZ 24 项，共 48 项；原 BBND small 24 项用于回归和逐案例对拍。

#### 新增 `test_quant_flash_attn_pa_layout.py`

- 复用原 `test_runner.execute_test`、CPU kernel reference、输出校验和精度阈值；
- 不创建另一套 golden 或放宽阈值。

#### 新增 `common/pa_layout_c8_compare.py`

- 4 个 C4/C8 对拍案例：D128/D256 × PA_BNBD/PA_NZ；
- 将同一 FP4 E2M1 payload 精确展开成 FP8 E4M3，Q/K/V 数值完全一致；
- 两种后端使用相同 scale、页表、mask、长度和物理布局；
- 输出 C4-vs-BBND、C4-vs-C8、C4/C8-vs-quantized-CPU、C4/C8-vs-FP64-unquantized-P 指标；
- 保存逐案例 JSON、原始输出 `.pt` 和汇总 `summary.json`。

#### 新增 `common/pa_layout_report.py`

- 读取已完成的 BBND/BNBD/NZ 测试缓存；
- 逐案例验证原始输入张量完全一致；
- 计算 Attention/LSE 的 relative-L2、RMSE、最大绝对误差和 cosine；
- 检查原 golden 阈值与 1% BBND parity 门限，输出 JSON 和 Markdown 报告。

## 6. 构建与测试方法

### 6.1 构建裁剪

本次包限定 TND、无 V-tail，但不裁剪 `KvLayoutType`，使 BBND/BNBD/NZ 同时进入包中：

```bash
'-DKERNEL_TEMPLATE_INPUT=InOutLayoutType=2,HasVTail=false'
```

如果只设置 `KvLayoutType=1`，新布局会因缺少对应 kernel 二进制而无法运行。

### 6.2 精度测试

在 `attention/quant_flash_attn/tests/pytest` 下执行：

```bash
python3 -m pytest \
  qfa_mxfp4_test/test_quant_flash_attn_debug.py \
  qfa_mxfp4_test/test_quant_flash_attn_pa_layout.py \
  --scope small --quant-schema 2 --reference-backend kernel \
  --device-id 7 --package cann_ops_transformer_c4a4 \
  --cache-dir /home/k831/work/results/schema2/cache -q -s
```

`quant_schema` 分别执行 0、1、2。每次包含原 BBND 24 项回归和 BNBD/NZ 48 项精度测试。

生成 BBND parity 报告：

```bash
python3 -m qfa_mxfp4_test.common.pa_layout_report \
  --scope small --quant-schema 2 --package cann_ops_transformer_c4a4 \
  --cache-dir /home/k831/work/results/schema2/cache \
  --output-dir /home/k831/work/results/schema2/report
```

执行 C4/C8 对拍：

```bash
python3 -m qfa_mxfp4_test.common.pa_layout_c8_compare \
  --device-id 7 --package cann_ops_transformer_c4a4 \
  --output-dir /home/k831/work/results/c8-comparison
```

## 7. 精度判据

原 BBND golden 判据保持不变：

- BF16 Attention：`atol=1e-4`、`rtol=0.0078125`；
- FP32 LSE：`atol=2.5e-5`、`rtol=0.005`；
- isclose 比例、最大相对误差、shape、有限值和空行语义均由原 runner 校验。

新增布局与 BBND 的同输入直接对拍判据：

```text
Attention relative-L2 <= 0.01
LSE relative-L2       <= 0.01
```

C4/C8 对拍不将两者强制视为逐位一致，因为它们使用不同的 P 量化格式和量化路径；报告同时给出二者与 FP64 未量化 P 参考的差距，避免把格式差异误判为布局错误。

## 8. 关键注意事项与边界

1. **平台范围**：只修改 arch35 MXFP4，不声称支持其他芯片或 arch92 HiFP4。
2. **Q/输出布局**：C4A4 当前仍限定 TND；本次没有扩展 BNSD/BSND。
3. **Head dim**：保持原 C4A4 的 D=Dv=128/256 限制。
4. **页大小**：公共 C4 checker 支持 64/128/256/512/1024；精度矩阵沿用原 BBND 的 128/256/512 配置。
5. **V-tail**：不在本次范围，构建保持 `HasVTail=false`。
6. **stride 单位**：FP4 payload 是逻辑元素，E8M0 在 kernel 中按 pair；不可直接混用 host stride 与物理字节地址。
7. **页尾**：NZ scale 按完整分形读取，V payload 的无效尾行仍由原逻辑清零；有效 token 之外的随机填充不会参与结果。
8. **非连续输入**：公共 checker 允许 BNBD/NZ 的第 0/1 轴非连续，kernel 使用 page/head stride；其他轴仍必须连续。
9. **构建过滤**：新布局必须实际编入包，仅 host 校验通过不代表 kernel 可用。
10. **参考独立性**：布局转换只用于 NPU 输入，CPU/Triton/FP64 参考始终从原始 BBND 数据解码。

## 9. 兼容性与未改动范围

- PA_BBND 的 GM 地址公式、L1 结果、计算流水线和模板 key 保持不变；
- C8A8、FP8、HIF8、MXFP8 的 kernel 未修改；
- 未修改公共 C8 搬运代码，避免影响其他算子；
- 未修改算子 API 参数，只开放已有 `layout_kv` 字符串的 C4 使用范围；
- 原 BBND 测试默认参数和用例数量保持不变。

## 10. 本次实测结果

测试环境为 Ascend 950PR、CANN 9.2.1、Python 3.12.16、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post7。设备预检后选择当时无运行进程且 Health=OK 的 NPU 7。

### 10.1 离线转换与构建

- payload/K scale/V scale 在 5 种页大小、D128/D256、BNBD/NZ 下共 60 组转换均逐字节可逆一致；
- 三布局自定义算子包和 Python wheel 构建成功；
- 设备冒烟时补齐了独立 metadata 算子的 PA_BNBD 白名单，重新打包安装后通过；
- 原 C4 计算路径无需精度优化，新增布局即达到与 BBND 逐位一致。

### 10.2 三 schema 精度矩阵

每项包含 BBND 24 项回归、BNBD 24 项和 NZ 24 项；三个 schema 共 216 项，失败和跳过均为 0。

| quant_schema | BBND | PA_BNBD | PA_NZ | 最大 Attention relative-L2 | 最大 LSE relative-L2 |
|---|---:|---:|---:|---:|---:|
| 0 / OCP | 24/24 | 24/24 | 24/24 | 0 | 0 |
| 1 / CX Legacy | 24/24 | 24/24 | 24/24 | 0 | 0 |
| 2 / CX Delta | 24/24 | 24/24 | 24/24 | 0 | 0 |

报告生成器还逐案例核对了三种布局的原始输入张量完全一致，因此上述 0 误差是同输入、同 golden 下的直接对比结果，不是不同随机数据间的统计巧合。

### 10.3 C4/C8 对拍

4 个案例全部通过，且 C4、C8 各自的 BNBD/NZ 输出都逐位一致。

| D / 布局 | C4/C8 Attention relative-L2 | C4/C8 LSE relative-L2 | C4/FP64 Attention L2 | C8/FP64 Attention L2 |
|---|---:|---:|---:|---:|
| D128 / BNBD、NZ | 0.0850652 | 0.000979246 | 0.0817984 | 0.0226986 |
| D256 / BNBD、NZ | 0.0963764 | 0.00118005 | 0.0928129 | 0.0251967 |

C4 与 C8 的 Attention 差距来自 P 量化格式与路径差异；两种布局之间的精度完全一致。C8 与 FP64 未量化 P 参考更接近，符合其更宽 P 数据格式的预期。C4 与 BBND 的 1% 门限独立于上述 C4/C8 格式差异，实测为 0。

详细数据见 [`pa-NZ-BNBD-test`](./pa-NZ-BNBD-test/) 中的 JUnit、pytest 日志、逐案例 parity JSON/Markdown、C4/C8 JSON/Markdown 和环境清单。
