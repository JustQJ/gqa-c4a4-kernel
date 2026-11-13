# QFA MXFP4 性能测试与采集记录

本文介绍 qfa_mxfp4_test 目录内的性能测试入口、采集方法、计时口径和实测结果。精度测试及构建环境见 [README.md](README.md)。

| 目标 | 入口 | 产物与计时口径 |
| --- | --- | --- |
| 单个 C4/C8 kernel 耗时、PMU、核内流水 | [prof_quant_flash_attn.py](prof_quant_flash_attn.py)，调用 msprof op | OpBasicInfo、7组 PMU CSV 或 PipeTimeline |
| C4/C8/BF16 文档集合性能对比 | test_quant_flash_attn_perf_rdv.py --run-perf | graph_replay_us：NPUGraph + Event |
| MXFP8 迁入集合的 C4/C8 对比 | 同一 pytest 入口，--perf-case-set mxfp8 | page512、PA_BBND，205个非skip场景 |
| 批量成对 kernel 耗时 | 同一 pytest 入口加 --msprof | msprof_task_us：普通 msprof 的 Task Duration |

普通 msprof 的 PipeUtilization 是流水占用指标；核内时间线使用 msprof op 的 PipeTimeline。不同口径分别报告，不混算加速比。

## 1. 使用 msprof op 查看耗时和核内流水

### 1.1 单算子采集入口

[prof_quant_flash_attn.py](prof_quant_flash_attn.py) 直接启动 msprof op，随后校验采集结果。每次选择一个C4或C8场景，默认C4；批量性能对比的启动方法见第2节，实测结果见第3节。

先按[主 README 构建与运行说明](README.md#新机器编译安装精度测试与-python-调用2026-11-12)配置CANN、配套OPP/Python包和 QFA_PACKAGE，再进入包含 qfa_mxfp4_test 的pytest父目录。确认目标卡没有其他任务后运行；设备号只作示例。也支持从任意目录使用脚本绝对路径启动。

```bash
DEVICE=7
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PROF_ROOT="$QFA_WORK/msprof-op-$RUN_TAG"

# 工具预热3次，区间内1次主算子。
python3 -m qfa_mxfp4_test.prof_quant_flash_attn \
  --device "$DEVICE" --package "$QFA_PACKAGE" \
  --metrics BasicInfo --output "$PROF_ROOT/basic"

# 工具预热10次，区间内5次主算子。
python3 -m qfa_mxfp4_test.prof_quant_flash_attn \
  --device "$DEVICE" --package "$QFA_PACKAGE" \
  --metrics Default --output "$PROF_ROOT/metrics"

# 短输入验证核内流水；工具预热3次，区间内1次主算子。
python3 -m qfa_mxfp4_test.prof_quant_flash_attn \
  --device "$DEVICE" --package "$QFA_PACKAGE" \
  --q-lens 128 --kv-lens 512 \
  --metrics PipeTimeline --output "$PROF_ROOT/pipeline"

# C8核内流水：FP8 E4M3、mode1，KV固定BBND；这里使用page512、p_scale=1。
python3 -m qfa_mxfp4_test.prof_quant_flash_attn \
  --device "$DEVICE" --package "$QFA_PACKAGE" --backend c8 \
  --q-lens 128 --kv-lens 512 --page 512 --p-scale 1 \
  --metrics PipeTimeline --output "$PROF_ROOT/c8-pipeline"
```

C8同样支持 `--metrics BasicInfo`（耗时）和 `--metrics Default`（7组PMU指标），替换最后一条命令的metrics并使用新的output目录即可。C4/C8依次采集，分别保存结果；`--backend` 每次只选一路，不自动运行两路。

默认输入为：B1、Q1024、KV4096、D256、Q头8/KV头1、page128，Q scale为N2TGD，KV为PA_BBND，causal、schema2、LSE关闭、Eager。复用精度输入生成器，只准备输入和调用公开API，不计算CPU/Triton参考；默认metadata、QKV、mask及20次应用预热都在MSTX区间外。该单例输入与第2节的批量配对输入生成器不同，其CSV不直接混入批量加速比报告。

同形状、同seed时，两路复用相同FP4编码、随机E8M0 scale和页表。C4直接调用mode5；C8在采集前复用 [common/perf_convert.py](common/perf_convert.py) 将编码精确展开为FP8 E4M3，保留scale和页表，并让metadata与主算子都调用mode1。转换及首次Triton JIT不在MSTX区间内。C8的 `--p-scale` 默认448；需要与MXFP8集合中的某项对应时显式使用该case的值，例如1。`--quant-schema` 仅影响C4，C8忽略此项。

| 参数 | 作用 |
| --- | --- |
| --backend c4/c8 | 默认c4；选择mode5/FP4或mode1/FP8 |
| --p-scale VALUE | 仅C8可传，有限正数；C8默认448，设备端float32标量 |
| --q-lens 1 3 / --kv-lens 129 513 | 支持变长batch；两列表长度相同，各长度为正 |
| --d 128/256、--heads、--kv-heads | Q头数为KV头数的正整数倍 |
| --page 128/256/512 | page size；KV固定PA_BBND |
| --layout TND/N2TGD | Q scale布局，不是KV布局 |
| --mask 0/3、--lse、--quant-schema 0/1/2、--seed | mask、LSE、P量化方案和输入种子 |
| --warmup 20 | MSTX区间外的应用预热次数 |
| --repeat N | 区间内调用数；默认等于launch-count |
| --prof-warmup N / --launch-count N | 工具回放预热及采集上限，与应用预热分开 |
| --include-metadata | 每次重新生成metadata；kernel过滤仍只统计attention，不形成metadata+attention总耗时 |
| --mstx-backend cann/torch | 默认CANN host range；torch仅用于标记通路诊断 |
| --post-range-check | 区间外额外调用一次，验证区间过滤 |
| --mode eager/graph | 默认Eager；graph仅用于诊断，当前950环境曾出现空结果，不作为已验收能力 |
| --msprof-bin /path/to/msprof | 指定工具路径 |
| --dry-run | 打印参数和命令，不导入Torch、不打开NPU、不创建目录 |
| --worker | 只运行应用，供外部msprof包装；不启动工具、不检查报告 |

脚本默认通过 libms_tools_ext.so 调用 CANN mstxRangeStartA / mstxRangeEnd，使用host区间过滤。C4区间名为 `C4A4_PROFILE`，C8为 `C8A8_PROFILE`；自动入口会同步设置工具的 `--mstx-include`。结束前同步设备，ID0也成对关闭，异常时同样关闭区间。

需要手动控制工具参数时，可使用 --worker：

```bash
msprof op --aic-metrics=BasicInfo --kernel-name=QuantFlashAttn_ \
  --mstx=true --mstx-include=C4A4_PROFILE \
  --warm-up=3 --launch-count=1 --output="$PROF_ROOT/manual" \
  python3 -m qfa_mxfp4_test.prof_quant_flash_attn \
  --worker --device "$DEVICE" --package "$QFA_PACKAGE" \
  --repeat 1 --mstx-backend cann
```

手动包装C8时，应用增加 `--backend c8`，工具同步改为 `--mstx-include=C8A8_PROFILE`；两处必须一致，否则会采集为空。

### 1.2 产物与成功判定

支持单次采集的 OpBasicInfo.csv 和多次采集的 OpBasicInfo_时间戳.csv；指标按相同时间戳配对。自动入口要求 --output 为新目录，避免旧结果被当成本次采集成功。保存 invocation.json（参数/完整命令）、profile.log（工具和应用日志）；产物校验后生成 profile_summary.json，记录目标kernel耗时、各指标行数或流水事件数，以及 collection_status / data_loss_warnings。

两路主算子的名称均匹配 `QuantFlashAttn_`；使用invocation/summary中的backend、quant_mode、p_scale及MSTX区间区分。profile.log的 `OPERATOR_CONFIG` 还记录实际输入dtype和形状，可确认C8实际使用 `torch.float8_e4m3fn`。

- BasicInfo：要求 OpBasicInfo.csv 中有 QuantFlashAttn_ 主算子行，Task Duration为有限正数。
- Default：另要求 ArithmeticUtilization、L2Cache、Memory、MemoryL0、MemoryUB、PipeUtilization、ResourceConflictRatio 共7份指标CSV非空。
- PipeTimeline：另要求 trace.json 包含非空执行区间。可将 trace.json 或 visualize_data.bin 导入 MindStudio Insight。
- 工具失败、目标kernel缺失、无效耗时、缺失/空指标或流水均报错。日志出现 has data loss 时保留 summary 并标记 partial，入口返回失败；不能当作完整流水。使用 --worker 手动包装时需自行核对产物。工具退出0不能单独证明采集成功。

### 1.3 本目录采集验证（2026-11-13）

在200的 tp_a4c4_gqa_kernel 容器、7号卡，使用既有 single-api 包完成小量验证，Eager、CANN MSTX。以下为C4首次采集结果，schema2：

| 采集 | 输入 | 结果 |
| --- | --- | --- |
| BasicInfo | Q1024 / KV4096，其余为默认值 | 1条主算子；区间内外各1次、launch-count=2，最终只捕获区间内1次 |
| Default | 同上，repeat=5 | 5次主算子；每次7份PMU CSV，各96行 |
| PipeTimeline | Q128 / KV512，其余为默认值 | 713条流水事件，无数据丢失提示 |

Q1024/KV4096的首次PipeTimeline产生5954条事件，但日志报告 channel20数据丢失，保留为partial证据，不按完整流水通过。最终入口遇到此类日志会保存partial摘要并返回失败。建议先使用上面的短输入验证采集，再按目标场景扩大输入并检查日志。

本机与容器19项CPU契约检查通过，覆盖ID0关闭、异常同步、空/错误CSV、带时间戳的多次采集配对和data loss判定。31份最终源码/参数SHA及268份原始证据哈希一致，结束7号卡无残留进程。此处验证采集能力，没有重跑第3节记录的75/205场景对比，也没有验证图模式或InstrTimeline。

同日扩展C8后，在同一容器、7号卡依次验证以下四项，均为B1/Q128/KV512/D256、Q头8/KV头1、page512、PA_BBND、Q-scale=N2TGD、mask3、LSE关闭、seed421：

| 实现与采集 | p_scale | 结果 |
| --- | ---: | --- |
| C4 BasicInfo | 不传 | 1条主算子，保留默认C4路径 |
| C8 BasicInfo | 默认448 | 1条主算子；区间内外各1次、launch-count=2，仅捕获区间内1次 |
| C8 Default | 1 | repeat=1、launch-count=1；7份PMU CSV，各96行 |
| C8 PipeTimeline | 1 | 454条有效流水事件，无数据丢失提示 |

日志核实C8的Q/K/V均为FP8 E4M3、mode1，区间为C8A8_PROFILE；四项summary均为validated。本机和容器各21项CPU契约检查通过，新增C8传参/区间一致性及p_scale参数检查；30份当前源码/参数及75份原始证据SHA一致，结束7号卡无残留进程。该小量仅验证采集入口，不据此计算性能加速比。

## 2. 性能测试与对比

统一入口为 [test_quant_flash_attn_perf_rdv.py](test_quant_flash_attn_perf_rdv.py)，通过 `--run-perf` 启用。启动一次测试需要分别确定：**测哪个用例集合、测哪些实现、采用哪种计时模式、使用哪张卡和哪个分片**。这些参数相互独立，输出目录名只用于区分产物。

### 2.1 选择测试集合

`--perf-case-set` 决定用例集合；`--scope` 只对 `document` 集合生效。参数定义见 [quant_flash_attn_paramset_perf_rdv.py](quant_flash_attn_paramset_perf_rdv.py)。

| 集合与范围 | 用例数 | 主要配置 |
| --- | ---: | --- |
| `--perf-case-set document --scope small` | 75：Prefill15、Decode/MTP60 | Qwen3.6 TP2，D256、Q头8/KV头1；page256、causal、LSE关闭 |
| `--perf-case-set document --scope full` | 1395：Prefill135、Decode/MTP1260 | 三个模型各TP1/2/4；D128/256；page256、causal、LSE关闭 |
| `--perf-case-set mxfp8` | 211：6项skip，实际205 | page512、PA_BBND；逐序列变长、零Q、mask0/3、两种Q-scale布局及LSE组合 |

document集合的Prefill覆盖Q8192、B1/2/4、KV8192至262144；Decode/MTP覆盖Q1/2/4/6/8、KV32768/65536/131072/262144。small的Decode batch为1/2/16，full为1/2/4/8/16/32/64。默认 `--scope full`；小量测试需要显式写 `--scope small`。

MXFP8集合保存在本目录 [perf_cases_mxfp8.json](perf_cases_mxfp8.json)，包含来源SHA和source_case_id，运行时不依赖兄弟目录。211项中Prefill102、Decode109；跳过6项耗时过长用例后为Prefill98、Decode107。`--scope small` 不会缩小此集合。参数保留逐序列Q/KV长度、max长度提示、头数、D128/256、mask、Q-scale布局、LSE及C8的p_scale；194项常规case的p_scale=1且LSE关闭，17项D256红线case中7项开启LSE。

可用 `--perf-kind prefill` / `decode` / `all` 筛选场景类型，默认all；pytest `-k` 可进一步按case ID筛选。两张卡的分片按选定集合与类型拆分，skip用例仍保留其分片位置。

### 2.2 选择被测实现与输入方式

`--perf-backends` 指定被测实现及执行顺序：

| 参数 | 实际执行 |
| --- | --- |
| `--perf-backends c4,c8` | C4测完释放，再测C8 |
| `--perf-backends c4,c8,bf16` | C4、C8、BF16依次执行，每路完成后释放；这是默认值 |
| `--perf-backends c4` 或 `c8` | 只记录指定实现，无法形成C8/C4比值 |

同一case的各路始终使用同一张卡，双卡只拆分case。CPU保留一份随机FP4编码及页表；C8精确转成E4M3，BF16解码后乘相同scale（Q/K为1/16，V为1/4）。每条序列按实际KV长度分配独占物理页，页号随机打乱，页表未使用位置填-1；Q按cu_seqlens打包。数据转换、H2D/D2H、metadata、mask、首次JIT、检查及落盘均在计时外。

C4与C8使用 `--package` 指定的同一配套QFA包，分别为mode5和mode1。C4默认 `--quant-schema 2`；C8的p_scale在document集合为448，在mxfp8集合使用各case配置，C4不传此参数。缩放默认1/sqrt(D)，Q/输出为TND，C4/C8的KV为PA_BBND。

BF16使用torch_npu的FIA V2非量化分页接口，Q为TND、KV为BnBsH，使用同一页表和对应mask语义。三路的P算法不同，性能完成检查不要求三路输出逐位一致。

### 2.3 选择执行与计时模式

模式默认值在 [conftest.py](conftest.py) 中定义，执行逻辑见 [common/perf_runner.py](common/perf_runner.py)。

| 启动参数 | 执行方式 | 时间来源与范围 | 结果metric |
| --- | --- | --- | --- |
| `--perf-mode graph` | NPUGraph回放；默认模式 | NPU Event测整图执行跨度，包含区间内任务间隔 | `graph_replay_us` |
| `--perf-mode eager` | Python逐次调用算子 | NPU Event测一轮调用的执行跨度，可能包含提交等待间隙 | `eager_event_us` |
| `--msprof` | 启动普通msprof，子进程采用专用Eager路径 | CSV中QuantFlashAttn任务各自的Task Duration，不累计任务之间的间隔 | `msprof_task_us` |

默认采样参数为 `--perf-warmup 3 --perf-rounds 5 --perf-repeats 20`，三种模式的含义分别为：

- **Graph：**先3次Eager预热，将20次主算子调用捕获到一张图，再3次整图回放预热。正式计时5轮，每轮回放一次整图，用Event时长除以20，最终取5轮中位数。图捕获不计时；每轮一次提交整图，减少Python逐次提交的间隙。
- **Eager Event：**先3次预热，再测5轮，每轮逐次调用20次主算子，用Event时长除以20，最终取5轮中位数。
- **普通msprof：**每路执行3次预热和100次正式调用，解析时丢弃前3条目标任务；剩余任务每20条求平均，得到5轮样本，再取中位数。每路应有103条QuantFlashAttn任务。

`--msprof` 由 [common/perf_profiler.py](common/perf_profiler.py) 包装pytest，并在子进程追加 `--perf-mode=msprof`。直接指定 `--perf-mode msprof` 只会进入待解析的调用路径，不会启动采集工具；正常采集使用 `--msprof`。文档中的Graph和msprof命令分开执行、使用不同输出目录。

**当前普通msprof解析器仅统计QuantFlashAttn任务，采集时显式选择 `--perf-backends c4,c8` 或其中一路。** BF16调用的是FIA，不适用于此解析器；BF16性能使用graph或eager模式。第1节的 `msprof op` 用于单个C4或C8场景的PMU及核内流水，与本节批量普通msprof入口分开。

三种metric分别出表，不能混算加速比。Graph与msprof结果还涉及不同执行方式，不能把两者耗时差直接当作图调度开销。

### 2.4 启动参数速查

| 参数 | 默认值 | 控制内容 |
| --- | --- | --- |
| `--run-perf` | 关闭 | 启用性能case；未指定时跳过性能测试 |
| `--perf-case-set` | document | 选择document或mxfp8用例集合 |
| `--scope` | full | document的small/full范围；mxfp8忽略此选项 |
| `--perf-kind` | all | prefill、decode或all |
| `--perf-backends` | c4,c8,bf16 | 被测实现及逐路执行顺序；只比较C4/C8时显式写c4,c8 |
| `--perf-mode` | graph | graph/eager的Event计时；工具采集使用--msprof |
| `--perf-warmup` | 3 | Eager预热次数；graph模式另做同次数的整图预热 |
| `--perf-rounds` | 5 | 采样轮数，最终取各轮结果的中位数 |
| `--perf-repeats` | 20 | 每轮调用数；graph模式为图内捕获的调用数 |
| `--device-id` | 0 | 本进程使用的设备号 |
| `--perf-shard-index` / `--perf-shard-count` | 0 / 1 | 从0开始的分片编号及总分片数；不会自动启动其他进程 |
| `--package` | QFA_PACKAGE环境变量，未设置时cann_ops_transformer | 注册公开QFA算子的实际Python包 |
| `--quant-schema` | 解析后为2 | C4的P量化方案，可选0/1/2 |
| `--perf-output` | perf_output | 本次case JSON及采集产物目录，建议每次使用独立目录 |
| `--msprof` | 关闭 | 启动普通msprof并自动解析 |
| `--parse-prof` | 未设置 | 重新解析已有采集目录，需配套原perf-output中的case记录 |
| `--perf-baseline` / `--perf-threshold` | 未设置 / 8 | msprof流程的同实现历史基线检查及退化阈值百分比 |

pytest的 `-q` 控制精简输出，`-s` 显示测试打印；这两个选项不改变计时模式。

### 2.5 设备前置检查

设备预检入口为 [common/perf_device_check.py](common/perf_device_check.py)，用于验证性能输入、参考结果和执行通路。

| 预检的--case-set | 覆盖范围 |
| --- | --- |
| basic（默认） | 16种FP4编码转换；4组D128/256×Prefill/Decode；C4、BF16参考及三路图回放 |
| mxfp8 | 8组D128/256×双Q-scale布局×mask0/3；page512、变长/零Q、页表补-1、LSE；C4参考及C4/C8图回放 |
| all | 依次执行上述12组，编码转换检查执行一次 |

```bash
python3 -m qfa_mxfp4_test.common.perf_device_check \
  --package "$QFA_PACKAGE" --device-id 7 \
  --case-set all --output "$QFA_WORK/preflight-all"
```

这里的 `--case-set` 选择预检场景；正式pytest性能集合使用 `--perf-case-set`。预检输出preflight.json（case_set、fp4_codes、cases）及latency目录，正常性能运行不会自动执行预检。它不替代完整精度矩阵：LSE只检查存在且有限，图回放与Eager比较Attention输出前128个元素。

### 2.6 单卡启动命令

先按[主README构建与运行说明](README.md#新机器编译安装精度测试与-python-调用2026-11-12)配置CANN、隔离OPP/Python包及QFA_PACKAGE，进入包含qfa_mxfp4_test的pytest父目录。QFA_WORK为结果目录的父路径；设备号只是示例，使用前重查占用。下列每条命令都是独立测试，输出目录不共用。

**A. 文档小量集合，75个case，C4/C8/BF16，Graph计时**

```bash
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --run-perf --perf-case-set document --scope small --perf-kind all \
  --perf-backends c4,c8,bf16 --perf-mode graph \
  --package "$QFA_PACKAGE" --quant-schema 2 --device-id 7 \
  --perf-warmup 3 --perf-rounds 5 --perf-repeats 20 \
  --perf-output "$QFA_WORK/document-small-graph7" -q -s
```

改为 `--scope full` 即选择1395项；只比较C4/C8则改为 `--perf-backends c4,c8`。用例范围不会因为选择了不同backend而改变。

**B. MXFP8集合，205个可执行case，C4/C8，Graph计时**

```bash
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --run-perf --perf-case-set mxfp8 --perf-kind all \
  --perf-backends c4,c8 --perf-mode graph \
  --package "$QFA_PACKAGE" --quant-schema 2 --device-id 7 \
  --perf-warmup 3 --perf-rounds 5 --perf-repeats 20 \
  --perf-output "$QFA_WORK/mxfp8-graph-single7" -q -s
```

**C. 同一MXFP8集合与C4/C8，Eager Event计时**

```bash
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --run-perf --perf-case-set mxfp8 --perf-kind all \
  --perf-backends c4,c8 --perf-mode eager \
  --package "$QFA_PACKAGE" --quant-schema 2 --device-id 7 \
  --perf-warmup 3 --perf-rounds 5 --perf-repeats 20 \
  --perf-output "$QFA_WORK/mxfp8-eager-single7" -q -s
```

这条命令用Event计时；主算子任务级采集使用下面的命令D。第3节现有MXFP8结果表记录了Graph和msprof两种口径，没有Eager Event结果。

**D. 同一MXFP8集合与C4/C8，普通msprof任务计时**

```bash
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --run-perf --perf-case-set mxfp8 --perf-kind all \
  --perf-backends c4,c8 --msprof \
  --package "$QFA_PACKAGE" --quant-schema 2 --device-id 7 \
  --perf-warmup 3 --perf-rounds 5 --perf-repeats 20 \
  --perf-output "$QFA_WORK/mxfp8-msprof-single7" -q -s
```

此命令通过普通msprof运行专用Eager采集，结束后自动解析并生成该次运行的msprof_report。不要将上述Graph/Eager/msprof目录混在一个报告中。

### 2.7 两张卡分片启动

分片只拆分case，各case内部仍在同一卡上依次测试C4、C8。下面两条Graph命令在两个终端分别运行：

```bash
# 终端1：6号卡，分片0
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --run-perf --perf-case-set mxfp8 --perf-kind all \
  --perf-backends c4,c8 --perf-mode graph \
  --package "$QFA_PACKAGE" --quant-schema 2 --device-id 6 \
  --perf-shard-index 0 --perf-shard-count 2 \
  --perf-warmup 3 --perf-rounds 5 --perf-repeats 20 \
  --perf-output "$QFA_WORK/mxfp8-graph6" -q -s
```

```bash
# 终端2：7号卡，分片1
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --run-perf --perf-case-set mxfp8 --perf-kind all \
  --perf-backends c4,c8 --perf-mode graph \
  --package "$QFA_PACKAGE" --quant-schema 2 --device-id 7 \
  --perf-shard-index 1 --perf-shard-count 2 \
  --perf-warmup 3 --perf-rounds 5 --perf-repeats 20 \
  --perf-output "$QFA_WORK/mxfp8-graph7" -q -s
```

MXFP8完整集合在这两个分片上实际执行106和99项，另6项skip位于分片1。document small集合用同样的分片参数时为38和37项。

双卡msprof采集沿用上述两条命令的集合、backend、卡号、分片及采样参数，将 `--perf-mode graph` 替换为 `--msprof`，并将两个输出目录分别改为 `$QFA_WORK/mxfp8-msprof6` 和 `$QFA_WORK/mxfp8-msprof7`。每个分片会独立生成采集数据和分片报告。

### 2.8 结果文件、汇总与重解析

每个case保存独立JSON，包含分轮样本、中位数、波动、metadata头、输入SHA、包/算子库来源和逐路释放后的allocated显存。失败保留部分结果及traceback；单实现结果可以保存，但只能在两路都完成时计算相应比值。

普通msprof额外保存 `profile/` 原始数据和 `msprof_report/` 分片报告。case JSON包含每次任务耗时、预热耗时、任务行及CSV SHA。解析器只读取 `OP Type=QuantFlashAttn`，按case/backend实际执行顺序匹配，要求每次调用恰好一条目标任务；空CSV、缺失/额外任务、非正耗时或执行失败均报错，metadata和输入转换不参与该任务统计。

**汇总单卡document small的75项结果：**

```bash
python3 -m qfa_mxfp4_test.common.perf_report \
  "$QFA_WORK/document-small-graph7" \
  --case-set document --scope small --kind all --expected 75 \
  --output "$QFA_WORK/document-small-graph-report"
```

**汇总双卡MXFP8 Graph的205项结果：**

```bash
python3 -m qfa_mxfp4_test.common.perf_report \
  "$QFA_WORK/mxfp8-graph6" "$QFA_WORK/mxfp8-graph7" \
  --case-set mxfp8 --kind all --expected 205 \
  --output "$QFA_WORK/mxfp8-graph-report"
```

**汇总双卡MXFP8 msprof的205项结果：**

```bash
python3 -m qfa_mxfp4_test.common.perf_report \
  "$QFA_WORK/mxfp8-msprof6" "$QFA_WORK/mxfp8-msprof7" \
  --case-set mxfp8 --kind all --expected 205 \
  --output "$QFA_WORK/mxfp8-msprof-report"
```

传入每个分片的perf-output根目录，读取其中已解析的case JSON。单卡MXFP8只传一个对应目录，仍为 `--expected 205`；document full使用 `--scope full --expected 1395`。按类型筛选时，汇总的 `--kind` 需与测试的 `--perf-kind` 一致，例如MXFP8 Prefill为98项。汇总会核对完整case ID集合，`-k` 筛出的少量case不能当作完整205项报告。

汇总产物为comparison.csv和summary.json。逐case保存 `c4_us`、`c8_us`、可选的`bf16_us`及波动；比值 `c8_over_c4 = c8_us / c4_us`，大于1表示C4更快。BF16/C4、BF16/C8采用同样规则。all/prefill/decode分别计算逐case比值的几何平均、中位数、范围及更快case数；报告会拒绝混用计时口径、采样参数或算子库。

已有msprof原始数据可单独重解析，无需再次运行算子：

```bash
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py \
  --parse-prof "$QFA_WORK/mxfp8-msprof6/profile" \
  --perf-output "$QFA_WORK/mxfp8-msprof6"
```

在 `--msprof` 或 `--parse-prof` 流程中，可附加 `--perf-baseline /path/to/summary.json --perf-threshold 8`。基线按case指纹、schema、相同metric及backend匹配；任一路相对历史同实现退化超过8%则失败。基线文件应覆盖当前运行的case；两分片分别比较时可使用覆盖完整205项的同口径基线。C8/C4加速比本身不按这个阈值判失败。

## 3. 性能测试结果

第3.1、3.2节分别对应document small和MXFP8集合，采样均为3次预热、5轮、每轮20次调用。Graph另做3次整图预热。两组结果的用例配置和C8 p_scale不同，分别阅读。第3.3节为指定shape的msprof op采集，采样口径单独说明。

### 3.1 2026-11-13 小量结果

测试配置：`--perf-case-set document --scope small --perf-backends c4,c8,bf16 --perf-mode graph`，200机器6/7号卡分片执行。

200机器6/7号卡完成75/75个场景（38+37）、225路测量，均退出0。三路逐路测完释放，allocated显存均回到0。以下是每场景5轮中位数的比值再取几何平均，>1表示分母实现更快。

| 范围 | 场景数 | C8/C4 | BF16/C4 | BF16/C8 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 15 | 0.9409× | 2.5175× | 2.6757× |
| Decode/MTP | 60 | 1.1364× | 9.2308× | 8.1232× |
| 全部 | 75 | 1.0943× | 7.1185× | 6.5053× |

C4快于C8为40/75项，快于本次FIA BF16为75/75项。五轮极差/中位数>10%的路数分别为C4 33、C8 20、BF16 0；原样本全部保留，接近1×的单项差异应结合波动阅读。6号卡保留既有Alarm，其他卡0至3有vLLM任务。结果为本次单图20调用的Event口径，不代表最优BF16实现、纯kernel计时或整模型吞吐。

8项离线契约测试、双卡D128/D256前置检查、25份Python源码SHA、75项计划与JUnit及225路采样审计通过，130份证据下载逐SHA一致。未修改生产kernel、未跑full1395、未提交。

### 3.2 2026-11-13 MXFP8 集合实测结果

测试配置：`--perf-case-set mxfp8 --perf-backends c4,c8`，分别以 `--perf-mode graph` 和 `--msprof` 执行，200机器6/7号卡分片。

200指定6/7号卡完成两种口径各205/205项，6项skip未执行，JUnit均无错误和失败。卡6实际106项、卡7实际99项；只测C4/C8，不测BF16。两种口径各410路、41000次计时调用，均逐路释放且allocated回到0。表中为C8/C4比值的几何平均，大于1表示C4更快。

| 范围 | case数 | 图回放 C8/C4 | msprof主算子任务 C8/C4 |
| --- | ---: | ---: | ---: |
| Prefill | 98 | 0.8490× | 0.8411× |
| Decode | 107 | 1.1307× | 1.1245× |
| 全部 | 205 | 0.9860× | 0.9788× |

图口径C4更快79/205项，msprof口径71/205项。5轮极差/中位数超过10%的C4/C8路数分别为图27/21、msprof0/5；原样本全部保留。msprof实际42230条QuantFlashAttn任务，与3次预热+100次计时的410路计划完全一致，剔除1230条预热。205项跨口径输入SHA/库来源、28份源码及参数文件SHA、完整case ID/JUnit/样本/设备号审计通过。12项CPU契约及双卡各8组前置检查通过；C4 Attention与CPU参考最大差0，非完整205项精度矩阵验收。

6号卡保留既有Alarm；结束6/7无运行进程。未修改生产kernel、未提交。

### 3.3 2026-11-13 指定shape的Default与PipeTimeline

使用 [prof_quant_flash_attn.py](prof_quant_flash_attn.py)，在200的tp_a4c4_gqa_kernel容器、7号卡依次采集两种shape、C4/C8和Default/PipeTimeline，共8组。配置如下：

| 场景 | Q长度列表 | KV长度列表 | Q scale布局参数 |
| --- | --- | --- | --- |
| Prefill | [8192] × 4 | [32768] × 4 | `--layout TND` |
| Decode/MTP | [4] × 16 | [32768] × 16 | `--layout N2TGD` |

共同参数：D256、Q头8/KV头1、page512、KV=PA_BBND、mask3、LSE关闭、seed421、Eager；C4为schema2，C8显式p_scale=1。每组20次区间外应用预热；Default使用工具预热10次、repeat=5、launch-count=5；PipeTimeline使用工具预热3次、repeat=1、launch-count=1。

Default四组均完整通过，各5次主算子，每次7份PMU CSV、各96行。以下耗时为5次Task Duration中位数，单位μs：

| 场景 | C4 Default | C8 Default | C4 PipeTimeline | C8 PipeTimeline |
| --- | ---: | ---: | --- | --- |
| Prefill，TND | 10700.187500 | 10111.361328 | partial，1,147,578事件 | partial，1,283,177事件 |
| Decode/MTP，N2TGD | 134.686264 | 218.225433 | validated，11,070事件 | partial，26,233事件 |

8组均执行完成，5组完整通过、3组流水为partial。Prefill C4在channel17/20、Prefill C8在channel17/26报告has data loss；Decode C8也有多通道数据丢失提示。三份时间线保留，可查看已采到的片段，但不能作为完整核内流水；采集入口按预期返回非零并保存partial摘要。Decode C4流水无数据丢失提示。
