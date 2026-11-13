# C4A4 首版算子对齐 CPU reference 的精度调测记录


## 1. 问题概览

| 问题 | 对精度的影响 | 当时采取的修改 |
|---|---|---|
| C4 FD 与非 FD 输出的 BF16 舍入规则不一致 | 相同 FP32 中点值可能转换成相邻的两个 BF16 值 | C4 FD 输出使用 `CAST_ROUND`，与非 FD 路径和 reference 对齐 |
| FD 对每个 split 先归一化，再加权合并 | 分段除法和后续乘法引入额外 FP32 舍入 | workspace 保存原始 FP32 PV、rowsum，合并时再处理归一化 |
| FD 通过自然对数参考计算合并倍率 | `R → R*ln2 → Exp` 的往返引入低位误差 | workspace 保存整数指数 R，在正规数范围直接构造 `2^(Ri-Rmax)` |
| 最终 FP32 除法的低位误差 | 商靠近 BF16 舍入中点时，1 个 FP32 ULP 的变化会改变最终 BF16 | 使用 `MXFP4_OUTPUT_DIV` 指定精确舍入除法 |
| 32 splits 时固定处理 8 行超过 FD 统计区容量 | 统计量可能越界并破坏相邻数据，影响输出及 LSE | 根据 split 数动态限制一次处理的行数，32 splits 时最多 6 行 |


## 2. BF16 输出舍入规则不一致

### 现象与原因

C4 的非 FD 输出及参考实现采用中点远离零的 BF16 转换规则。最初复用的 FD 输出分支对 BF16 使用 `CAST_RINT`，在中点处采用 ties-to-even。两条路径即使得到同一个 FP32 值，也可能产生不同的 BF16 输出。

例如，正数 `1.00390625` 正好位于 BF16 的 `1.0` 和 `1.0078125` 之间。不同中点规则会分别选择这两个值。此类差异应先检查输出转换，不能仅凭最终误差判断 QK、P 量化或 PV 计算有误。

### 两种舍入规则具体有什么区别

FP32 有 23 位显式尾数，BF16 有 7 位，转换时需要舍去低 16 位并决定是否进位。对于普通有限值，两种模式都选择最近的 BF16 值，区别在于恰好位于两个 BF16 值中间时的处理：

- `CAST_RINT`：ties-to-even，选择保留下来的尾数最低位为 0 的一侧。“偶数”指这一个二进制位，不是数值本身为偶整数。
- `CAST_ROUND`：ties-away-from-zero，选择绝对值较大的一侧。它不是对所有数都向远离零的方向舍入，非中点时仍选择最近值。

在 `[1, 2)` 区间内，相邻 BF16 的间距是 `2^-7 = 0.0078125`：

| 转换前的 FP32 值 | `CAST_RINT` | `CAST_ROUND` | 说明 |
|---|---|---|---|
| `1.0038` | `1.0` | `1.0` | 在中点下方，两种模式相同 |
| `1.00390625` | `1.0` | `1.0078125` | 恰好位于中点，规则不同 |
| `1.004` | `1.0078125` | `1.0078125` | 在中点上方，两种模式相同 |
| `1.01171875` | `1.015625` | `1.015625` | 虽然是中点，但偶数尾数一侧恰好也是远离零的一侧 |
| `-1.00390625` | `-1.0` | `-1.0078125` | 负数的 ties-away 选择绝对值更大的一侧 |

因此，并不是每个中点都会产生不同结果。对于 `1.00390625`，两种输出到原始 FP32 值的绝对误差均为 `0.00390625`；将 C4 统一为 `CAST_ROUND` 是为了统一输出规则，而不是因为它在中点处比 `CAST_RINT` 更准确。

### 修改与当前状态

在 `quant_flash_attn_block_vec_flashdecode.h::CopyFinalResOut` 中按输入类型区分：

- C4 输入为 FP4 时，输出转换使用 `CAST_ROUND`。
- 原有非 C4 的 BF16 分支继续使用 `CAST_RINT`。

当前代码保留了这项区分。CPU reference 的 `round_bf16_away`、Triton reference 的最终 BF16 转换也显式实现相应规则。

如果要检查最终 BF16 输出的逐位一致性，reference 也需要按相同规则舍入。当前 C4 CPU 整行参考及模拟 kernel 分块参考均先执行 `round_bf16_away(output)`，再转为 `torch.bfloat16`；Triton 分块参考在 `_combine_kernel` 中显式实现相同舍入，并写入 BF16 输出。

### C8 kernel 与 reference 当前采用什么方式

当前 arch35 中，C8 的两条 BF16 输出路径并没有统一为同一种中点规则：

| kernel | 非 FD 的 BF16 输出 | FD 的 BF16 输出 |
|---|---|---|
| C4 / MXFP4 | `CAST_ROUND` | `CAST_ROUND` |
| C8 / MXFP8 | `CAST_ROUND` | `CAST_RINT` |

C8 非 FD 在 `quant_flash_attn_block_vec_mxfp8.h::Bmm2ResCastAndCopyOut` 中使用 `CAST_ROUND`；FD 在共用的 `quant_flash_attn_block_vec_flashdecode.h::CopyFinalResOut` 中使用非 C4 的 BF16 分支。因此，“保留 C8 的 `CAST_RINT`”只指 FD 分支。

但 `qfa_mxfp8_test` 的默认 CPU reference **最终不转成 BF16**，所以不能将其最终输出归类为使用 `CAST_RINT` 或 `CAST_ROUND`：

- `common/quant_flash_attn_golden.py` 中，`_cpu_mxfp8_golden_sectioned` 与 `cpu_mxfp8_golden` 的最终 Attention 输出均保留 FP32；可选的高精参考保留 FP64。
- `common/test_runner.py` 对参考做布局转换后交给比较器，不在此处补做 BF16 舍入。
- `common/result_compare_method.py::check_result` 将实际 BF16 输出和参考输出都转成 FP32 后比较。实际 BF16 提升为 FP32 不会恢复已经舍去的低位。
- 默认 BF16 验收使用 `rtol=0.0078125`、`atol=0.0001`，并结合通过比例和异常误差门限；不能把其中一次 `torch.isclose` 判断当作完整验收逻辑。

这里描述的是最终 Attention 输出；参考内部 P 或 V-tail 的 BF16 转换是其他计算步骤。C4 与 C8 即使使用相同的比较器，也不代表参考输出具有相同的 dtype 和舍入过程。

### 如果 C4 reference 也保留 FP32，是否就没有这个问题

需要区分验收是否通过，以及 kernel 的两条路径是否输出相同的 BF16 值。

如果 C4 reference 保留未舍入的 FP32 结果，并采用上述 C8 的 BF16 容差，那么在正常有限值范围内，**仅由中点规则不同造成的误差通常不会导致验收失败**。例如以 `1.00390625` 为参考：

```text
RINT 输出  = 1.0        ，绝对误差 = 0.00390625
ROUND 输出 = 1.0078125  ，绝对误差 = 0.00390625
单元素容差 = atol + rtol * abs(reference)
           = 0.0001 + 0.0078125 * 1.00390625
           = 0.007943017578125
```

两种输出都在该容差内。如果参考先按 ties-away 舍入为 BF16，则参考是 `1.0078125`，RINT 输出与它相差 `0.0078125`，而 ROUND 输出与它完全一致。两种比较方式衡量的误差不同；是否通过仍由实际门限决定，不能仅凭 reference 的 dtype 判断。

只删除 reference 的 BF16 转换，却继续使用历史上严格的绝对误差 `<1e-5`，并不能解决问题：上述正常 BF16 转换产生的 `0.00390625` 误差仍会失败。保留 FP32 reference 时，需要将输出格式本身允许的舍入误差计入验收标准。

另外，FP32 reference 不会使两个 kernel 输出变成同一个 BF16 值，也不会修复 QK、P 量化、PV 或归一化中的计算错误。即使统一了最终 Cast，如果转换前的两个 FP32 结果分处 BF16 中点两侧，仍可能得到不同的 BF16 输出，见第 5 节。当前 C4 已统一 FD/非 FD 的 `CAST_ROUND`，仍有助于保持路径间的一致性。

本节补充比较方式及其含义；当前 C4 CPU/Triton reference 仍按上述规则输出 BF16，未改为 FP32。

## 3. FD 提前归一化造成额外舍入

### 原来的计算过程

设第 i 个 split 的原始 PV 为 `U_i`，rowsum 为 `s_i`，对齐到公共参考的倍率为 `a_i`，总分母为：

```text
S = sum(a_i * s_i)
```

最初沿用的合并方式先在各 split 内计算归一化输出，再合并：

```text
O_i = U_i / s_i
weight_i = a_i * s_i / S
O = sum(weight_i * O_i)
```

实数运算中 `s_i` 可以约掉，但 FP32 运算已经对 `U_i/s_i` 和后续乘法进行了舍入。分段计算不能通过代数消项恢复为没有这些中间舍入的结果，最终值在 BF16 中点附近时可能表现为输出不一致。

### 当时的修改

FA 写入 FD workspace 的内容改为：

- 未归一化的 FP32 PV；
- 原始 FP32 rowsum；
- 用于跨 split 对齐的整数指数 R。

对应 `quant_flash_attn_block_vec_mxfp4.h::WriteAttentionResult`：DN 使用不执行除法的 FP32 输出路径，ND 直接搬出原始 FP32 PV。两种布局都不在 split 内先除以 rowsum，也不提前转成 BF16。

当时新增的 FD 流程为：

```text
mergedPV = sum(a_i * U_i)
S = sum(a_i * s_i)
O = mergedPV / S
```

`PrepareFlashDecodeVf` 准备合并倍率和总分母，`NormalizeFlashDecodeVf` 在所有 split 合并后完成最终除法。

### 当前代码状态

后续按方案调整，`NormalizeFlashDecodeVf` 已删除。当前仍保存原始 PV，但在 `PrepareFlashDecodeVf` 中将总分母折入倍率：

```text
S = sum(a_i * s_i)
weight_i = a_i / S
O = sum(weight_i * U_i)
```

`ReduceFinalRes` 因而直接得到最终输出。这里乘的是原始 PV，倍率中不再乘 split 自身的 `s_i`。

“先合并再除”和“先求 `a_i/S` 再合并”数学上等价，FP32 运算顺序不同。当前 CPU/Triton 分块参考已按后一种顺序实现；历史的最终除法函数不能作为当前调用链描述。

## 4. FD 合并倍率中的指数往返误差

### 现象与原因

C4 的组参考本身来自整数二进制指数 R。若先将它转成自然对数参考 `c_i=R_i*ln(2)+C`，FD 再计算 `exp(c_i-c_max)`，会经历乘法、减法和指数运算的 FP32 舍入。

理论上的合并倍率是：

```text
a_i = 2^(R_i - R_max)
```

当 `R_i-R_max` 为整数且结果位于 FP32 正规数范围时，该倍率可以精确表示，无需通过上述自然对数往返生成。

### 修改与当前状态

- `PrepareAttentionStatisticsVf` 的 FD 分支输出整数指数 R，而非已换算的自然对数参考。workspace 仍使用 FP32 存储，只是数值表示整数指数。
- `PrepareFlashDecodeVf` 先求 `R_max`，再计算指数差。
- 对 `R_i-R_max >= -126` 的范围，通过 FP32 指数字段构造精确的二次幂。
- 对更小的指数差保留指数函数回退，没有把整个次正规数范围直接裁成零。
- 空 split 根据 rowsum 置零倍率，避免无效参考参与合并。

当前代码保留上述处理。自然对数转换在恢复 LSE 时按当前量化方案执行。这里的调整专指跨 split 的 FD 倍率，不代表算子中所有组间重缩放都改成了相同实现。

## 5. 最终 FP32 除法改变 BF16 舍入结果

### 现象与定位

修正输出 Cast 和 FD 合并方式后，严格比较仍可能在少量元素处不一致。进一步检查发现：归一化的 FP32 商接近 BF16 舍入中点，除法结果相差 1 个 FP32 ULP，就可能在转换后落入相邻 BF16 值。

因此，仅统一 BF16 Cast 规则还不够，Cast 输入的 FP32 商也需要检查。

### 修改与当前状态

在 `vf_softmax_fp32_mxfp4.h` 中定义：

```cpp
constexpr DivSpecificMode MXFP4_OUTPUT_DIV = {MaskMergeMode::ZEROING, true};
```

输出归一化调用显式使用该配置：

```cpp
Div<float, &MXFP4_OUTPUT_DIV>(...);
```

当时覆盖 DN/ND 的最终输出除法，以及 FD 合并后的最终除法。

当前对应关系为：

- 非 FD：`vf_output_update_fp32_mxfp4.h` 中 `processOutDnVF` / `processOutNdVF` 的输出除法。
- FD：`vf_flash_decode_mxfp4.h::PrepareFlashDecodeVf` 中计算 `a_i/S` 的除法。

其中 `true` 开启 `precisionMode`，SDK 内部使用差值补偿提高除法精度；它并不表示没有额外计算。这里没有增加手写的 FD 合并残差补偿，当前 FD 继续使用单个 FP32 `a_i/S` 参与合并。

### 关闭高精度除法后的现行门禁验证（2026-11-13）

**关闭高精度除法已通过当前小量精度门禁，但两种除法模式的最终输出并非逐位相同。** 本轮在 141.61.41.200 的 `tp_a4c4_gqa_kernel` 容器、6/7 号卡上，将隔离构建副本的配置改为：

```cpp
constexpr DivSpecificMode MXFP4_OUTPUT_DIV = {MaskMergeMode::ZEROING, false};
```

此配置同时作用于 DN/ND 非 FD 的输出归一化，以及 FD 的 `a_i/S` 计算。测试保留现有 BF16 CPU/Triton reference、最终 BF16 舍入规则和全部验收条件；Attention 仍使用 `rtol=0.0078125`、`atol=0.0001`，未放宽门限，也未改用不做 BF16 转换的 reference。

| 验证范围 | 通过数 | 失败/跳过数 |
|---|---:|---:|
| OCP，`quant_schema=0`，小量 | 228/228 | 0/0 |
| CX，`quant_schema=1`，小量 | 228/228 | 0/0 |
| CX Delta，`quant_schema=2`，小量 | 228/228 | 0/0 |
| D128 补充，三种方案合计 | 36/36 | 0/0 |
| 合计 | 720/720 | 0/0 |

CPU/Triton/kernel 的 Attention 与 LSE 比较均保留。全部 720 项中，Attention 对 CPU/Triton 的最大绝对差分别为 `0.00048828125` / `0.0001220703125`，LSE 对 CPU 的最大绝对差为 `1.9073486328125e-6`。4212 次有效比较的门限内元素匹配率均为 100%；这里的“匹配”指落在容差范围内。

另外，使用相同输入和分核计划，在同一机器上重新执行 24 个高精度对照用例，覆盖三种量化方案、D128/D256、DN/ND、FA/FD。高精度版本同样全部通过当前 CPU 门禁，但其中 **18/24 个用例的 BF16 Attention 输出与普通除法版本不同**，两个版本的最大绝对差为 `0.00048828125`；24 个用例的 LSE 均逐位相同。这些对照优先选择各组合中普通除法对 CPU 差异较大的用例，不能用 18/24 推断整个矩阵的变化比例。

因此，历史上为严格对齐 BF16 结果而开启高精度除法，并不意味着关闭后必然无法通过当前容差门禁。本轮确认的是“小量范围及 D128 补充通过”；**全量范围和性能收益尚未验证，正式工作区配置仍保留 `true`**。

## 6. 32 splits 时 FD 统计区容量不足

### 原因

FD 的每个统计量视图按每行重复 8 个 FP32 值组织，其单槽容量为 6144 字节。一次处理 M 行、合并 N 个 split 时，该视图需要：

```text
bytes = N * M * 8 * sizeof(float)
```

固定一次处理 8 行，在 32 splits 时需要：

```text
32 * 8 * 8 * 4 = 8192 bytes
```

超过 6144 字节槽容量。这属于缓冲区容量错误，需要先解决，不能将其导致的输出异常归因于普通浮点误差。

### 修改与当前状态

`quant_flash_attn_block_vec_flashdecode.h::FlashDecode` 根据实际 split 数计算行数上限：

```text
rowsPerIteration = min(8, floor(6144 / (splitCount * 8 * 4)))
```

32 splits 时上限为 6 行，对应 `32*6*8*4=6144` 字节。后续行数分块及尾块按这个值计算，同时增加 split 数合法性断言。当前 C4 分支保留该限制。

## 7. Triton reference 自身的除法差异

当时还出现过 kernel 与 CPU 输出一致、但与 Triton reference 相差一个 BF16 值的情况。例如历史 `exact-div-precision.json` 的 D256、H24、ND `long_prefill` 用例，对 CPU 的输出最大差为 0，对 Triton 的最大差为 `3.0517578125e-5`，只有一个元素超过当时门限。

这类情况需要同时比较 kernel/CPU、kernel/Triton、CPU/Triton，并查看转换前的 FP32 值。双参考之间不一致时，不能只依据其中一份参考改动 kernel。

历史调测对独立 Triton reference 的除法进行过验证和修正：使用拆分尾数求残差、比较相邻商的方式处理已定位的低位误差；并将归一化拆到独立 kernel，解决该参考实现的 D256 UB 资源问题。这是当时参考代码的处理，不应误写成生产 C4 kernel 增加了残差补偿。

当前本目录的 `triton_mxfp4_golden_kernel.py` 使用独立 `_combine_kernel`，通过 `libdevice.div_rn` 完成单 split 输出除法或多 split 的 `a_i/S`。本目录当前实现和历史独立参考的具体代码应分别描述。
