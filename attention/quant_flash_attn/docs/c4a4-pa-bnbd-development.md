# C4A4 算子 PA_BNBD KV Cache 布局开发文档

> 适用算子：`attention/quant_flash_attn`（C4A4 / `quant_mode=5` / MXFP4，arch35 / Ascend950）
> 开发目标：在不影响现有 PA_BBND / PA_NZ 与其它量化模式（MXFP8/FP8/HIF8/HIFP4）的前提下，为 C4A4 增加 PA_BNBD 分页 KV cache 布局支持，且精度与 PA_BBND 持平（相对误差 ≤ 1%）。
> 完成日期：2026-11-16

## 1. 背景

C4A4 算子此前支持 `PA_BBND`（`KvLayoutType == 1`）与 `PA_NZ`（`KvLayoutType == 3`）两种分页 KV cache 布局。`PA_BNBD`（`KvLayoutType == 2`）是 vLLM-Ascend 等推理框架中 KV cache 的另一常见物理排布（页 × head × 页内 token × D）。本次开发为 C4A4 补齐 PA_BNBD 能力。

类型系统中分页 KV 布局共 3 种（`op_host/qfa_tiling_info.h`）：

| 布局 | KvLayoutType | 逻辑 shape（FP4 元素） | 说明 |
|---|---|---|---|
| PA_BBND | 1 | `[Bn, Bs, N, D]` | 页 × 页内token × head × D |
| PA_BNBD | 2 | `[Bn, N, Bs, D]` | 页 × head × 页内token × D |
| PA_NZ | 3 | `[Bn, N, D/64, Bs, 64]` | 页 × head × D分形 × 页内token × 64元素(32B) |

## 2. PA_BNBD 数据合约（与 host checker 一致）

| 张量 | PA_BBND shape | PA_BNBD shape |
|---|---|---|
| K/V payload（FP4 逻辑元素） | `[Bn, Bs, N, D]` | `[Bn, N, Bs, D]` |
| K descale（E8M0 uint8） | `[Bn, Bs, N, D/64, 2]` | `[Bn, N, Bs, D/64, 2]` |
| V descale（E8M0 uint8） | `[Bn, Bs/64, N, Dv, 2]` | `[Bn, N, Bs/64, Dv, 2]` |

要点：
- FP4 payload 的 shape 是**逻辑元素数**（D 维为元素数，两个 FP4 一个字节，物理存储字节 = 元素数/2）。
- K scale 每 token 每 64 通道一对 E8M0；V scale 每通道每 64 token 一对 E8M0。
- 与 PA_BBND 相比，PA_BNBD 仅把 `N` 轴从 `Bs` 之后移到 `Bs` 之前（纯轴重排），分形内径不变。

## 3. 设计思想

### 3.1 核心原则：L1 布局不变，只改 GM→L1 搬运

现有 PA_BBND / PA_NZ 实现的 GM→L1 搬运后，K/V payload 与 scale 在 L1 上的布局为：

| 数据 | L1 布局（三种布局共用） |
|---|---|
| K/V payload | `[D/64, s2BaseSize, 32B]`（NZ 分形） |
| K scale | `[token/16, D/64, 16, 2B]` |
| V scale | `[Dv/16, s2BaseSize/64, 16, 2B]` |

**关键洞察**：PA_BNBD 与 PA_BBND 的 GM 排布差别仅在 N/Bs 两轴的先后顺序。由于 L1 布局完全一致，**下游所有计算（QK/PV Matmul、Fixpipe、Softmax、Pscale、输出、FD）代码零改动**，从机制上保证 PA_BNBD 与 PA_BBND 的数值路径相同，精度天然持平。

### 3.2 零侵入隔离

- 所有 PA_BNBD 新增逻辑都用 `if constexpr (KvLayoutType == 2)` 编译期隔离，BBND（`KvLayoutType == 1`）/ NZ（`KvLayoutType == 3`）路径的指令流完全不变。
- tiling key 仅**新增** PA_BNBD 的 MXFP4 模板实例，不修改任何现有实例。
- host `DoOpTiling()` 仅放宽布局白名单，其余 tiling 逻辑不变。

## 4. 修改清单

### 4.1 `op_kernel/arch35/quant_flash_attn_block_cube_mxfp4.h`

1. **`CopyKScaleGm2L1()`**：新增 PA_BNBD（`KvLayoutType == 2`）分支。
2. **`CopyVScaleGm2L1()`**：新增 PA_BNBD（`KvLayoutType == 2`）分支。
3. **`CopyKvPayloadGm2L1()`**：PA_BNBD 复用 BBND 路径（payload 的搬运通过 `n2LogicalStride` 参数处理 head 轴，BBND/BNBD 均可正确工作）。

**修复记录**：开发过程中曾出现以下两处关键错误并已修正：
- **BBND K scale 分支缺失**：添加 BNBD 支持时误删了 `KvLayoutType == 1` 的实现，导致 N_KV=2 时 Head 4-7（第二个 KV head）读取错误的 scale 数据，产生 NaN。已恢复 BBND 分支。
- **BNBD stride 公式错误**：最初按 BBND 公式计算 BNBD 的 `n2PairStride`（错误地等于 `(Bs/16)*(D/64)`），导致非空序列用例 rel_l2 高达 78%~105%。已修正为 BNBD 布局的正确 stride（见 §5）。

### 4.2 `op_kernel/arch35/quant_flash_attn_template_tiling_key.h`

MXFP4 的 `ASCENDC_TPL_ARGS_SEL` 中，`KvLayoutType` 的实例列表由 `{KvLayoutType_PA_BBND, KvLayoutType_PA_NZ}` 扩展为 `{KvLayoutType_PA_BBND, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ}`，使 PA_BNBD 参与模板实例化。

### 4.3 `op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp`

`DoOpTiling()` 的入口校验由"PA_BBND 或 PA_NZ"放宽为"PA_BBND 或 PA_BNBD 或 PA_NZ"。`UpdateTilingKeyKvLayout` 的 `KvLayoutType_PA_BNBD` 映射、`paLayoutType = 0` 在现有代码中已就绪，无需改动。

## 5. PA_BNBD 搬运算法

以下 `blockSize` 为页大小（`constInfo_.blockSize`），`s2BaseSize` 为 L1 的 token 容量（D128→256，D256→128）。所有 stride 均以 **uint16 pair** 为单位（descale 数据为 `[..., 2]` 的 E8M0 pair）。

### 5.1 K scale

GM（每页）：`[N, Bs, D/64, 2]`，L1：`[token/16, D/64, 16, 2]`。

| 量 | 公式 | 说明 |
|---|---|---|
| `tokenPairStride` | `D/64` | 同一页同一头内相邻 token 的 pair 数 |
| `n2PairStride` | `Bs * (D/64)` | 相邻 N 头之间的 pair 数 |
| `pagePairStride` | `N * Bs * (D/64)` | 页间步长 |

逐物理页处理（页内连续 token 段 `pageTokenCount`）：

```
srcBase = page*pagePairStride + n2Idx*n2PairStride + pageTokenOffset*tokenPairStride
dstBase = copiedTokenCount * (D/64)
srcStride(DataCopy 的 srcDValue) = tokenPairStride   // 相邻 token 组之间
dstStride = D/64
```

- `bnStride/n2Stride == 0` 表示连续，用上表默认公式；非连续时取 `kDescaleStrides.bnStride / 2`、`n2Stride / 2`（字节 → pair）。

### 5.2 V scale

GM（每页）：`[N, Bs/64, Dv, 2]`，L1：`[Dv/16, s2BaseSize/64, 16, 2]`。

| 量 | 公式 | 说明 |
|---|---|---|
| `scaleGroupPairStride` | `Dv` | 同一头内相邻 64-token 组的 pair 数 |
| `n2PairStride` | `(Bs/64) * Dv` | 相邻 N 头之间的 pair 数 |
| `pagePairStride` | `N * (Bs/64) * Dv` | 页间步长 |

64-token 组的划分逻辑（`logicalScaleGroupIndex`、`pageScaleGroupOffset`、`pageScaleGroupCount`）与 BBND 完全复用：

```
srcBase = page*pagePairStride + n2Idx*n2PairStride + pageScaleGroupOffset*Dv
dstBase = copiedScaleGroupCount * 16
srcStride(DataCopy 的 srcDValue) = Dv
dstStride = s2BaseSize / 64
```

### 5.3 K/V payload

PA_BNBD 的 payload GM 布局 `[Bn, N, Bs, D]` 与 BBND `[Bn, Bs, N, D]` 的区别通过 `n2LogicalStride` 参数在 `CopyKvPayloadGm2L1()` 中处理：BBND 的 head 轴在 D 内层（地址连续），BNBD 的 head 轴在 Bs 外层（需要显式 head stride = `Bs * D/2` 字节）。该函数在 BBND 下保持原行为不变。

## 6. 关键注意事项

1. **stride 单位**：FP4 payload 的 host stride 是**逻辑元素数**（需 `/2` 折字节）；descale 的 stride 是**字节**（需 `/2` 折 uint16 pair）。`bnStride/n2Stride == 0` 表示连续，用默认公式。
2. **BNBD 与 BBND 的 stride 差异是最大的坑**：两者 L1 布局相同，但 GM 的 N/Bs 轴顺序相反，导致：
   - BBND 的 K scale `n2PairStride = (Bs/16)*(D/64)`（N 在内层），BNBD 应为 `Bs*(D/64)`（N 在外层）；
   - BBND 的 V scale `scaleGroupPairStride = (Bs/64)*Dv`（N 在内层），BNBD 应为 `Dv`（N 在外层）。
   若误用 BBND 公式，非空序列用例的 rel_l2 会达到 78%~105%。
3. **对齐要求**：D 必须 64 整除、Dv 必须 16 整除、blockSize 必须 64 整除——现有 C4A4 参数域全部满足。
4. **尾块**：token 尾块（不足一页）按 `pageTokenCount`、`pageScaleGroupCount` 截断拷贝；V payload 的无效行清零逻辑（`CopyVGm2L1` 尾部）与布局无关，原样保留。
5. **非连续 tensor**：PA_BNBD 支持 0 轴（页）/0+1 轴（页+head）非连续，由 `bnStride/n2Stride` 处理；页内分形轴必须连续。
6. **BBND / PA_NZ 回归**：BBND / NZ 路径的指令、buffer、同步完全未改。开发过程中曾因误删 BBND K scale 分支导致 N_KV=2 回归（Head 4-7 NaN），修复后 12 组用例的 PA_BBND 输出与修复前完全一致（bit-exact），确认无回归。

## 7. 测试方法与结果

### 7.1 测试环境

真机：Ascend950PR（CANN 9.2.1，Docker 容器 k831）。测试采用 **C++ ACLNN driver + NumPy golden** 架构（不依赖 torch）：

- `qfa_driver.cpp`：直接调用 `aclnnQuantFlashAttnMetadata`（AICPU）与 `aclnnQuantFlashAttn`（`quant_mode=5`），读取 `.ini` 配置与 `.bin` 输入，导出 BF16/FP32 结果。
- `gen_inputs.py`：按 BBND 精度用例语义生成随机 FP4/E8M0 数据与随机物理页，并给出 FP64（不量化 P）参考。同一 canonical BBND 数据重排得到 BNBD / NZ 数据。
- `gen_c8_inputs.py`：利用 FP4 E2M1 全部 16 个量级可被 FP8 E4M3 无损表示的性质，把 C4 payload 查表转为 C8 payload，使 C4/C8 的 Q/K/V 数值完全相同，仅 P 量化不同。
- `compare_bnbd.py` / `compare_c8_bnbd.py`：NumPy 比对与报告。

### 7.2 用例

与原 C4A4 BBND debug 精度用例一致：`smoke`（Q17/KV257/page256）、`empty_and_q_padding`（空序列/Qpadding/空KV）、`page_tail`（页尾），各覆盖 Qscale `TND`（DN）/`N2TGD`（ND）、`mask_mode=0/3`，N_KV=2，共 12 组。每组同时跑 PA_BBND 与 PA_BNBD（同一 canonical 数据重排）。

### 7.3 结果

真机验证（Ascend950PR，CANN 9.2.1）12 组用例全部通过，PA_BNBD 与 PA_BBND 输出 **bit-exact** 一致。

| 用例 | bit_exact | max_abs | rel_l2 | isclose |
|---|---|---|---|---|
| smoke_TND_M0 | True | 0 | 0 | 1.0 |
| smoke_TND_M3 | True | 0 | 0 | 1.0 |
| smoke_N2TGD_M0 | True | 0 | 0 | 1.0 |
| smoke_N2TGD_M3 | True | 0 | 0 | 1.0 |
| empty_and_q_padding_TND_M0 | True | 0 | 0 | 1.0 |
| empty_and_q_padding_TND_M3 | True | 0 | 0 | 1.0 |
| empty_and_q_padding_N2TGD_M0 | True | 0 | 0 | 1.0 |
| empty_and_q_padding_N2TGD_M3 | True | 0 | 0 | 1.0 |
| page_tail_TND_M0 | True | 0 | 0 | 1.0 |
| page_tail_TND_M3 | True | 0 | 0 | 1.0 |
| page_tail_N2TGD_M0 | True | 0 | 0 | 1.0 |
| page_tail_N2TGD_M3 | True | 0 | 0 | 1.0 |

与 C8A8 PA_BNBD 的精度对比（C4/C8 的 Q/K/V 数值相同，仅 P 量化不同，MXFP8 原生支持 PA_BNBD）：

| 用例 | c4bnbd_vs_c8bnbd rel_l2 | c4bnbd_vs_golden rel_l2 | c4bbnd_vs_golden rel_l2 | c8bnbd_vs_golden rel_l2 |
|---|---|---|---|---|
| smoke_TND_M0 | 9.718e-02 | 9.306e-02 | 9.306e-02 | 2.796e-02 |
| smoke_TND_M3 | 9.620e-02 | 9.212e-02 | 9.212e-02 | 2.768e-02 |
| smoke_N2TGD_M0 | 9.718e-02 | 9.306e-02 | 9.306e-02 | 2.796e-02 |
| smoke_N2TGD_M3 | 9.620e-02 | 9.212e-02 | 9.212e-02 | 2.768e-02 |
| empty_and_q_padding_TND_M0 | 2.699e-02 | 0.000e+00 | 0.000e+00 | 2.681e-02 |
| empty_and_q_padding_TND_M3 | 2.337e-02 | 0.000e+00 | 0.000e+00 | 2.323e-02 |
| empty_and_q_padding_N2TGD_M0 | 2.699e-02 | 0.000e+00 | 0.000e+00 | 2.681e-02 |
| empty_and_q_padding_N2TGD_M3 | 2.337e-02 | 0.000e+00 | 0.000e+00 | 2.323e-02 |
| page_tail_TND_M0 | 9.539e-02 | 9.133e-02 | 9.133e-02 | 2.736e-02 |
| page_tail_TND_M3 | 9.519e-02 | 9.117e-02 | 9.117e-02 | 2.736e-02 |
| page_tail_N2TGD_M0 | 9.539e-02 | 9.133e-02 | 9.133e-02 | 2.736e-02 |
| page_tail_N2TGD_M3 | 9.519e-02 | 9.117e-02 | 9.117e-02 | 2.736e-02 |

结论：
- C4A4 PA_BNBD 与 C4A4 PA_BBND 精度完全一致（bit-exact，相对误差 0%，优于 ≤1% 要求）。
- C8A8 PA_BNBD 精度优于 C4A4 PA_BNBD（rel_l2 ~2.7e-02 vs ~9.2e-02，FP8 > FP4，符合预期），与 PA_NZ 阶段的 C4-vs-C8 对比结论一致。
- `empty_and_q_padding` 用例 C4 输出与 golden 完全一致（rel_l2=0），因为空序列/KV 长度极短，无有效注意力计算。

## 8. 验证流程复现

```bash
# 在容器中（Ascend950PR, CANN 9.2.1）
cd /home/k831/C4A4-NZ-kimi/nz_test

# 1) 生成测试数据（BBND/BNBD/NZ 三种布局 + golden 参考）
python3 gen_inputs.py --out data --cases smoke,empty_and_q_padding,page_tail

# 2) 生成 C8A8 输入（bbnd/nz/bnbd 三种布局）
python3 gen_c8_inputs.py --data data

# 3) 运行 BBND / BNBD 精度测试
for c in $(python3 -c "import json;print(' '.join(json.load(open('data/index.json'))))"); do
  ./qfa_driver data/$c bbnd.ini out_bbnd
  ./qfa_driver data/$c bnbd.ini out_bnbd
done
python3 compare_bnbd.py --data-dir data --output bnbd_report.json

# 4) 运行 C8A8 BNBD 精度对比
for c in $(python3 -c "import json;print(' '.join(json.load(open('data/index.json'))))"); do
  ./qfa_driver data/$c c8_bnbd.ini out_c8_bnbd
done
python3 compare_c8_bnbd.py --data data --report data/c8_bnbd_report.json
```
