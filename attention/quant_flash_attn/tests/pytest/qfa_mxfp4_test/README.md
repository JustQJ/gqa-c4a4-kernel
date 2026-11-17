# QFA MXFP4 精度测试

沿用相邻 `qfa_mxfp8_test` 的参数集、pytest 入口、公共 runner、CPU golden、缓存、结果比较结构，调用公开的 `quant_flash_attn_metadata` 和 `quant_flash_attn`。本 README 介绍精度测试，不含 V-tail 测试；性能对比、msprof / msprof op 采集与历史结果统一见 [Pref_Readme.md](Pref_Readme.md)。

## 文件结构

```text
qfa_mxfp4_test/
├── conftest.py                           # 参数展开、执行阶段和范围选项
├── pytest.ini                           # debug / func_rdv / ci markers
├── quant_flash_attn_paramset_common.py   # 公共默认值、约束和展开
├── quant_flash_attn_paramset_debug.py    # 冒烟与接口边界
├── quant_flash_attn_paramset_func_rdv.py # 全量/小量功能矩阵
├── quant_flash_attn_paramset_bench16k.py # 12个固定16k bench精度/性能case及KV页池
├── quant_flash_attn_paramset_qwen3_8b.py # Q32/KV8/D128、TP1的16项Qwen3-8B bench
├── test_quant_flash_attn_debug.py        # debug 入口
├── test_quant_flash_attn_func_rdv.py     # 功能入口
├── prof_quant_flash_attn.py             # 按case统一msprof/op采集，支持C4/C8
├── Pref_Readme.md                       # 性能测试方法及实测记录
└── common/
    ├── quant_flash_attn_golden.py        # 数据生成、解码、CPU CX参考、公开API调用
    ├── cpu_mxfp4_golden_kernel.py        # metadata/16-task CPU 分块参考
    ├── cpu_mxfp4_golden_fp64.py          # 不量化 P 的 FP64 参考
    ├── triton_mxfp4_golden_kernel.py     # metadata/16-task Triton 分块参考
    ├── mxfp4_schedule.py                # metadata 解码与分段检查
    ├── reference_metrics.py             # FP64 差距指标
    ├── golden_cache.py                  # 输入、参考输出与 NPU 输出缓存
    ├── result_compare_method.py         # 原样复用MXFP8比较器
    └── test_runner.py                   # gen → cpu → npu → compare
```

使用包内相对导入，不依赖 MXFP8 的 `common` 模块或旧 `mxfp4_vector` 测试脚本。

## 输入与范围

固定 `quant_mode=5`、Q/输出为 TND、KV 为 PA_BBND、`p_scale=None`、连续张量。Qscale 覆盖 DN（TND）和 ND（N2TGD）；所有场景同时覆盖无 mask（0）和右下对齐 causal（3）。page size 仅使用 128、256、512。显式设置 `softmax_scale=1/sqrt(D)`；它不是公开 API 的默认值 1.0。

Q/K/V 使用独立随机 FP4 E2M1 数据；Q/K 的 E8M0 指数字节取 122～125，V 取 124～127，seed=421。物理页随机排列，未使用页及页尾也随机填充。缓存保存 uint8 原始位；NPU 阶段转成 `float4_e2m1fn_x2` / `float8_e8m0fnu`。V scale 沿 token 方向每 32 个值一组。输出为 BF16 `[T,H,D]`，启用 LSE 时为 FP32 `[H,T]`，关闭时为 FP32 空张量。

| 范围 | 模型及 TP | 功能精度 | debug | 总数 |
|---|---|---:|---:|---:|
| `full`（默认） | Qwen3.6-35B、Qwen3.8-27B、Qwen3-30B，各 TP=1/2/4 | 4860 | 216 | 5076 |
| `small` | Qwen3.6-35B，TP=2，D=256，Q头=8，KV头=1 | 204 | 24 | 228 |

模型未切 TP 时的 `(D,Q头,KV头)` 分别为 `(256,16,2)`、`(256,24,4)`、`(128,32,4)`；切分后 Q 头数除以 TP，KV 头数为 `max(1,KV头/TP)`。

功能矩阵与现有 C4 精度范围一致，全部启用 LSE：

- 短输入：Q=KV=1、3、4、5、11、22，分别使用 page=128、128、256、512、512、128。
- 混合 batch=4：Q=`[1,3,22,129]`，KV=`[11,32771,65,259]`，page=128。
- batch=36：Q=`[1,3,4,5,11,22,102,300,512,1024,3000,4096]`、KV=`[1,17,65,127,33,65,202,330,891,1999,3100,5196]`，各重复 3 次，page=128。
- 长 KV prefill：Q=129、KV=32771，page=512。
- 长 Q prefill：每条 Q=8192，KV=8192/32768/65536，batch=1/2，page=512。
- Decode/MTP：KV=32768/65536/131072，page=256。全量每条 Q=1～8、batch=1/2/4/8/16；小量每条 Q=1/2/3/4、batch=1/2/16。

debug 使用三个场景，均组合 DN/ND、双 mask 和 LSE 开/关：

- Q=17、KV=257、page=256。
- Q 存储长度 `[5,0,3]`、有效长度 `[3,0,3]`、KV=`[0,33,1]`、page=128，覆盖 Q padding、空序列、空 KV 和 causal 空行。
- Q=`[3,9]`、KV=`[129,513]`、page=512，覆盖页尾。

功能 DN 场景显式传入 `seqused_q`，ND 场景省略它；debug 的 Q padding 场景两种布局均传入有效长度。page 并非与所有长度做笛卡尔积，而是按上述场景设置。

## CPU 参考与通过条件

CPU 用 FP32 计算 `S=QK/sqrt(D)`，按 S32 做 P 量化。`--p-quant-scheme` 有两种选择，默认使用文档中 delta 同时参与指数选择的 `cx_delta`：

```text
k_g = ceil(max(S_g) / ln(2) - log2(7.25 / 1.5))
P4_g = FP4_E2M1(BF16(exp(S_g - k_g*ln(2) + ln(1.5))))
R = max(k_g)
Pscale_g = 2^(clamp(k_g - R + 124, 0, 254) - 127)
W = P4 * Pscale
O = (W @ V) / sum(W)
LSE = ln(sum(W)) + (R + 3)*ln(2) - ln(1.5)
```

`cx_delta` 的固定偏移为 `c=ceil(log2(7.25/1.5))=3`。`cx_legacy` 对应 `QuantSchema::CX_QUANT`，使用 `k_g=ceil(max(S_g)/ln(2)-log2(7.25))`，P4=`E2M1(BF16(exp(S-k_g*ln2)))`，LSE=`ln(sum(W))+(R+3)*ln2`。Pscale +124、E8M0 钳制和输出归一化保持。现在由 `--quant-schema` 同时选择公开API与reference；`--p-quant-scheme` 为命名别名，二者同时提供时必须一致。`quant_mode` 仍为5。

当前 AscendC kernel 默认使用枚举模板 `QuantSchema QUANT_SCHEMA=QuantSchema::CX_DELTA_QUANT`，选择 `cx_delta`；原两个量化bool模板参数已删除。同一新包通过 `--quant-schema 0/1/2` 分别测试OCP、无1.5的CX和CX Delta；选择值实际传给公开API。历史 228/228 通过记录属于原方案，不能继承为新默认方案的设备验收。

新旧模式分别通过 24 项 debug 的 `gen,cpu` 阶段；另有 40 组两套 CPU 参考对拍及 20 组独立 Triton 设备对拍。它们验证 reference 实现，不是新量化规则的公开 API kernel 验收。详见[reference 验证报告](../../../../../operators/quant_flash_attn/cx_delta_reference_20261111/README.md)。

忽略 masked group，空行输出 0、LSE 为 +inf。P 的 BF16/FP4 转换和最终 BF16 输出按中点远离零舍入。不加入 FD 残差补偿。

### 独立参考文件

| 文件 / 选择 | 模拟范围 |
|---|---|
| `common/quant_flash_attn_golden.py` / `--reference-backend global` | 保留原 CPU golden；按 32 个 Q 计算全有效 KV，不依赖设备 FA/FD metadata。全行参考结构保留；本次仅同步 `cx_legacy` 的 delta=1 语义。 |
| `common/cpu_mxfp4_golden_kernel.py` / `--reference-backend kernel`（默认） | 解析实际 metadata 的核与 section 分段；D128 每 task N256，D256 每 task N128；每 16 个 task 共用 reference，累积 C2 后更新 V2，最后按 `a_i/S` 合并原始 split PV。 |
| `common/triton_mxfp4_golden_kernel.py` / `--triton-reference` | 同一 metadata 和 16-task 语义的 NPU Triton 参考，独立计算分段 PV 并合并；解码后的 FP16 dot、FP32 累积。为限制 UB，分两遍计算组内 score。不是硬件指令逐位模拟。 |
| `common/cpu_mxfp4_golden_fp64.py` / `--fp64-reference` | Q/K/V 反量化后使用 FP64 普通 attention，不量化 P，不将输出舍入到 BF16；不模拟设备分块的舍入。 |

默认使用 `kernel` CPU 参考，pytest 命令行和直接调用 runner 的参数默认值保持一致；需要原全行参考时显式指定 `--reference-backend global`。`kernel` CPU 参考首次取得 metadata 时需要可用的 NPU 和指定算子包；缓存建立后可独立执行 CPU 阶段。metadata 解码失败直接报错，不退回 global golden。Triton 参考仍需显式添加 `--triton-reference`，在 `npu` 阶段生成，在 `compare` 阶段同时检查 kernel/CPU、kernel/Triton、CPU/Triton，全部通过才算用例通过。

FP64 仅在上述精度比较通过后运行。比较 JSON 记录 kernel、CPU 和 Triton 相对 FP64 的 max_abs、mean_abs、RMSE、relative L2 和 cosine；这衡量 QKV 反量化固定后的 P 量化及有限精度差距，不作为放宽量化参考验收门限的依据。

FP64 使用 Q64/KV2048 的稳定在线 softmax，仅保留一个 batch 的 FP64 KV。`--reference-threads` 大于 1 时，各 head 独立并行；每个 head 内的 KV 遍历与 FP64 累积顺序不变。`gen` 会重新生成 metadata 和 FP64 结果；使用相同输入的 compare-only 可以复用有效 FP64 缓存。

结果比较器与 `qfa_mxfp8_test/common/result_compare_method.py` 完全相同：

| 项目 | atol | rtol | 满足 `isclose` 的元素比例 | 超差元素最大 E |
|---|---:|---:|---:|---:|
| BF16 Attention | 0.0001 | 0.0078125 | ≥99.5% | <10 |
| FP32 LSE | 0.000025 | 0.005 | ≥99.5% | <2000 |

`isclose` 判断为 `abs(actual-reference) <= atol + rtol*abs(reference)`；额外的 `E=abs(actual-reference)/(max(abs(actual),abs(reference),2^-14/0.005)+1e-10)` 只统计超差元素。Attention 和启用时的 LSE 都必须通过。

runner 额外要求输出 shape/dtype 正确、Attention 全部有限、LSE 无 NaN/-inf，且 LSE 的 +inf 位置与参考完全一致。关闭 LSE 时仍校验 FP32 空张量接口契约。

这是沿用 MXFP8 标准的新测试套件；旧 `mxfp4_vector` 验收脚本的 `1e-5` 严格门限和用例未修改，两者的通过结果不能互相替代。

## 运行

在本目录运行。`--collect-only` 仅需要 pytest；`gen/cpu/compare` 需要 PyTorch，其中默认 `kernel` CPU 参考首次生成 metadata 时还需要可用 NPU 和指定算子包。`npu` 需要提供 FP4/E8M0 dtype 的 torch_npu、CANN 和已安装的 C4 算子包。仅使用原全行 CPU 参考时显式传入 `--reference-backend global`。`--package` 是负责注册 `torch.ops.cann_ops_transformer` 的 Python 包名，默认取环境变量 `QFA_PACKAGE`，否则为 `cann_ops_transformer`。

```bash
# 仅收集，不调用算子
python3 -m pytest --collect-only -q --scope small
python3 -m pytest --collect-only -q --scope full

# 小量 debug：全部阶段（包含真实NPU调用及精度判定）
python3 -m pytest -q -s -m debug --scope small --device-id 0 \
  --package cann_ops_transformer --cache-dir /tmp/qfa-c4-debug

# 小量功能精度；全量改成 --scope full
python3 -m pytest -q -s -m func_rdv --scope small --device-id 0 \
  --package cann_ops_transformer --cache-dir /tmp/qfa-c4-func

# 可单独筛选 case，例如：-k 'mtp1_kv32768_b1_TND_M0'

# 在同一个包中选择 CX（不乘1.5），参考也同步切换
python3 -m pytest -q -s -m debug --scope small --p-quant-scheme cx_legacy \
  --device-id 0 --package cann_ops_transformer --cache-dir /tmp/qfa-c4-legacy
```

新默认 kernel 的小量双参考验收与 FP64 差距统计：

```bash
python3 -m pytest -q -s --scope small --reference-backend kernel \
  --triton-reference --fp64-reference --p-quant-scheme cx_delta \
  --device-id 7 --package cann_ops_transformer --cache-dir /tmp/qfa-c4-cx-delta
```

也可以像 MXFP8 一样分阶段执行，必须使用相同范围、筛选条件和缓存目录：

```bash
python3 -m pytest -q -m debug --scope small --golden-mode gen,cpu --cache-dir /tmp/qfa-c4-debug
python3 -m pytest -q -m debug --scope small --golden-mode npu --cache-dir /tmp/qfa-c4-debug --package cann_ops_transformer
python3 -m pytest -q -m debug --scope small --golden-mode compare --cache-dir /tmp/qfa-c4-debug --package cann_ops_transformer
```

默认 `--golden-mode all` 按 `gen → cpu → npu → compare` 执行。CPU golden 默认单线程，可用 `--reference-threads` 调整。长 Q/KV 全量参考计算与缓存开销较大，可按 marker、case 名分批运行。

每条 case 保存输入、CPU/NPU 输出及比较 JSON。缓存校验参数指纹、参考版本和输入生成 ID；重新运行 `gen` 后必须重新计算 CPU/NPU 输出。compare-only 还核对 `--package`，但包名不能识别同名包的重新安装，重装后应重新执行 NPU 阶段。只执行 `gen/cpu/npu` 的 pytest 成功表示阶段执行完成，**不代表精度通过**；精度结论需要 `compare`。

`quant_schema`、`p_quant_scheme` 和 `reference_backend` 参与参数指纹，缓存版本为 `mxfp4-public-quant-schema-v5`。切换方案/参考或使用旧缓存时须重新生成；分阶段命令应传入相同选项，建议不同方案使用不同缓存目录。metadata 缓存也绑定输入生成 ID、参数和包名；同名包重装后使用新缓存目录。

## 2026-11-12 新默认方案设备验收

增加公开 `quant_schema` 前的 `cx_delta` 正式包在200机器完成小量228/228及补充16/16，均通过CPU分块、Triton分块三方比较；该次修改保留了原CPU golden。Attention与CPU分块完全一致，与Triton最大差0.0001220703125，按上述MXFP8门限通过；不等同于旧逐点1e-5标准。

不量化P的FP64参考对比，小量Attention最大绝对差0.7923678622、平均绝对差0.001608472643、RMSE 0.003279366704；最大差值来自仅两个可见KV的causal行中P量化为等权。完整范围、量化损失说明、日志和版本审计见[正式kernel验证报告](../../../../../operators/quant_flash_attn/cx_delta_kernel_20261112/README.md)。未性能测试或全量设备验收。

## 2026-11-12 UseCxQuant 去除1.5

`cx_legacy` 的名称保持，计算改为 MindIE-SD CX 的 delta=1 公式；原CPU、分块CPU与两套Triton参考同步更新。默认 `cx_delta` 不变。缓存升级v4，拒绝历史含1.5的旧CX缓存。上面的已完成小量记录和251项停止记录属于修改前快照，不是本次无1.5 CX 的设备验收；全量测试保持停止。


## quant_schema 三方案公开 API 验收（2026-11-12）

`--quant-schema 0/1/2` 同时控制kernel与所有reference，省略为2；`--p-quant-scheme ocp/cx_legacy/cx_delta` 是对应别名。两选项冲突会在收集阶段报错，避免kernel/reference错配。方案和缓存v5均进入缓存校验；每个方案使用独立缓存目录。

```bash
python -m pytest . --scope small --quant-schema 0 --reference-backend kernel --triton-reference --cache-dir /tmp/qfa-schema0
python -m pytest . --scope small --quant-schema 1 --reference-backend kernel --triton-reference --cache-dir /tmp/qfa-schema1
python -m pytest . --scope small --quant-schema 2 --reference-backend kernel --triton-reference --cache-dir /tmp/qfa-schema2
```

每个方案仍为debug24+功能204，共228项；三方案684项。全量输入范围和相同精度阈值保留。OCP参考使用floor(smax/ln2)、exp(s−(k−2)ln2)、Pscale编码k−R+125和LSE=ln(Z)+Rln2；CX两方案保持上文公式。当前模式的设备通过情况以本次报告为准，不继承历史记录。

本次公开API新包已在200机器完成三方案小量：OCP 228/228、CX（无1.5）228/228、CX Delta 228/228，共684项，0失败/跳过。kernel/CPU、kernel/Triton、CPU/Triton的Attention和LSE门限内匹配率均为100%，仍非逐位或旧1e-5结论。另16项D128/D256 API切换/默认值验证通过。额外D128分块36项因其他服务占卡未启动，不计入小量结果；未性能、full设备或本轮FP64差距统计。见[三方案公开API验收报告](../../../../../operators/quant_flash_attn/quant_schema_small_20261112/README.md)。


## 新机器编译、安装、精度测试与 Python 调用（2026-11-12）

本节将当前源码的构建和测试流程整理为可迁移命令。需要安装两部分：包含 ACLNN/Host/kernel 的算子 `.run` 包，以及负责 PyTorch 注册和 C++ 桥接的 Python wheel；同时构建 `quant_flash_attn` 和 `quant_flash_attn_metadata`。下面的构建参数与本轮单一 ACLNN 接口验收使用的参数一致，工作目录和 vendor 改为可配置值。

### 1. 基础环境与依赖

以下命令针对 Linux/aarch64、Ascend 950。先安装适配设备的驱动、固件、CANN，以及匹配的 PyTorch/torch_npu，并准备 GCC/G++、make、git 等构建工具。仓库源码本身不包含这些基础软件。2026-11-12 在 123 机器验证的环境如下；这是已验证版本记录，不是所有兼容版本的列表。

| 组件 | 本轮验证版本 |
|---|---|
| NPU | Ascend950DT_9574 |
| CANN | 9.2.0 |
| Python | 3.11.10 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0.post7 |
| Triton | 3.6.0 |
| Triton Ascend | 3.6.0+dev20261013220622 |

使用 `--triton-reference` 时需要 Triton Ascend。123 原环境的 `triton-ascend 3.2.2+dev20260729205041` 在参考实现 `_combine_kernel` 的编译阶段触发过 `TypeRange` 空类型断言；本轮使用独立的上述 3.6 环境后通过。请准备相应的 Triton/Triton Ascend wheel 或已经配置好的环境，仅安装普通 `triton` 包不等于安装了 Ascend 后端。只运行算子、不启用 Triton reference 时，不需要 Triton 参考实现。

进入上传后的仓库，在同一个 Bash 终端中依次执行下面各节。先按实际情况修改仓库和 CANN 路径，`python3` 应指向已配置好上述依赖的 Python 环境。

```bash
set -eo pipefail

cd /你的路径/ops-transformer
export QFA_REPO="$(pwd)"
export QFA_WORK="$(dirname "$QFA_REPO")/c4a4-build-test"
export QFA_VENDOR=c4a4

# 按实际 CANN 安装路径修改
source /usr/local/Ascend/cann/set_env.sh

mkdir -p "$QFA_WORK"/{tmp,logs,wheels,python}
export TMPDIR="$QFA_WORK/tmp"

python3 -m pip install -r "$QFA_REPO/requirements.txt"
python3 -m pip install \
  'cmake>=3.20,<4' build wheel ninja pybind11 'pytest>=8.3.2,<9'

# 联网下载 CMake 构建依赖
bash "$QFA_REPO/scripts/download_3rd_party.sh" \
  --cann_3rd_lib_path="$QFA_WORK/deps"

npu-smi info
python3 - <<'PY'
import importlib.metadata as metadata
import torch
import torch_npu
import triton
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("triton:", triton.__version__)
print("triton-ascend:", metadata.version("triton-ascend"))
PY
```

离线机器应提前上传下载好的 `deps` 目录和 Python 依赖 wheel（包括构建依赖），跳过联网下载步骤，并将 pip 安装命令改为使用本地 wheel 目录的 `--no-index --find-links` 形式。`TMPDIR` 指向有空间的工作目录，避免安装器在根分区或 `/root` 空间不足时解压失败。

### 2. 编译并安装算子包

使用新的构建目录，不复用其他机器的 CMake 产物。下面直接调用 CMake，并保留本轮验收的模板范围。

```bash
cmake -S "$QFA_REPO" -B "$QFA_WORK/build" \
  -DCANN_3RD_LIB_PATH="$QFA_WORK/deps" \
  -DBUILD_OPEN_PROJECT=ON \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  '-DASCEND_OP_NAME=quant_flash_attn;quant_flash_attn_metadata' \
  -DVENDOR_NAME="$QFA_VENDOR" \
  -DENABLE_BUILT_IN=OFF \
  -DENABLE_OPS_HOST=ON \
  -DENABLE_OPS_KERNEL=ON \
  -DENABLE_BUILD_PKG=ON \
  -DENABLE_TILING_SINK=OFF \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  '-DKERNEL_TEMPLATE_INPUT=InOutLayoutType=2,KvLayoutType=1,HasVTail=false'

cmake --build "$QFA_WORK/build" --target package -j16 \
  2>&1 | tee "$QFA_WORK/logs/build.log"

bash "$QFA_WORK/build/cann-ops-transformer-${QFA_VENDOR}_linux-$(uname -m).run" \
  --quiet \
  --install-path="$QFA_WORK/install"
```

该模板范围对应 TND 输入输出、PA_BBND 分页 KV、无 V tail，包含本测试需要的 C4 D128/D256、Qscale DN/ND、无 mask/causal 和 FA/FD 路径。Qscale 的 DN/ND 选择与这里的 Q/输出 layout 是不同概念。其他 Q/KV layout 不在这个裁剪包的范围内。

三种 `quant_schema` 都包含在同一个包中，在调用时选择，不需要分别编译。当前原名 ACLNN 接口已经直接接收 `quantSchema`，不保留 V2 或旧参数转发层；算子库与下面的 Python/C++ 扩展需要由同一份源码配套构建。

### 3. 构建并安装 Python 扩展

```bash
export TORCH_EXTENSION_OPS=quant_flash_attn,quant_flash_attn_metadata
export TORCH_EXTENSION_VENDOR="$QFA_VENDOR"

cd "$QFA_REPO/torch_extension"

python3 setup.py bdist_wheel \
  --bdist-dir "$QFA_WORK/wheel-build" \
  --dist-dir "$QFA_WORK/wheels"

python3 -m pip install --upgrade --no-deps \
  --target "$QFA_WORK/python" \
  "$QFA_WORK/wheels/cann_ops_transformer_${QFA_VENDOR}-"*.whl

export ASCEND_CUSTOM_OPP_PATH="$QFA_WORK/install/vendors/${QFA_VENDOR}_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QFA_WORK/python:${PYTHONPATH:-}"
export QFA_PACKAGE="cann_ops_transformer_${QFA_VENDOR}"

python3 - <<'PY'
import importlib
import os
package = importlib.import_module(os.environ["QFA_PACKAGE"])
print(package.__file__)
PY

cd "$QFA_REPO"
```

上面使用 `pip --target` 安装到工作目录，因此后续 Python 进程需要保留 `PYTHONPATH`。新开终端时需要重新加载 CANN 环境，并设置上述自定义 OPP、动态库和 Python 包路径。包中的 C++ 桥接在首次实际调用时进行 JIT 编译；仅导入成功还不能替代设备执行和精度比较。

### 4. 运行三方案小量精度测试

为本次执行创建新的结果和扩展缓存目录。通过 `npu-smi info` 确认所选卡没有其他进程，再设置 `QFA_DEVICE`；下面的 `0` 只是示例卡号。如果另行配置了设备可见性，传入的是当前进程看到的设备编号。

```bash
export QFA_RUN="$QFA_WORK/results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$QFA_RUN"

export TORCH_EXTENSIONS_DIR="$QFA_RUN/torch-extensions"
export TRITON_CACHE_DIR="$QFA_WORK/triton-cache"
export XDG_CACHE_HOME="$QFA_WORK/runtime-cache"
export ASCEND_PROCESS_LOG_PATH="$QFA_WORK/runtime-logs"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$ASCEND_PROCESS_LOG_PATH"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export QFA_DEVICE=0

for schema in 0 1 2; do
  python3 -m pytest \
    "$QFA_REPO/attention/quant_flash_attn/tests/pytest/qfa_mxfp4_test" \
    --scope small \
    --quant-schema "$schema" \
    --golden-mode all \
    --reference-backend kernel \
    --triton-reference \
    --reference-threads 8 \
    --device-id "$QFA_DEVICE" \
    --package "$QFA_PACKAGE" \
    --cache-dir "$QFA_RUN/cache-schema-$schema" \
    --junitxml "$QFA_RUN/schema-$schema.xml" \
    -q -s -x --tb=short \
    2>&1 | tee "$QFA_RUN/schema-$schema.log"
done
```

该循环执行 `test_quant_flash_attn_debug.py` 的24项和 `test_quant_flash_attn_func_rdv.py` 的204项，每种方案228项，三方案共684项。`set -eo pipefail` 配合 `-x`，使出现失败时停止当前测试和后续方案；只有三次 pytest 都完整通过，才能得到684项通过结论。

`--reference-backend kernel --triton-reference` 保留 CPU 分块 golden、Triton 分块 reference 与实际 kernel 的 Attention/LSE 三方比较，使用本 README 前述精度阈值。不同方案使用独立缓存目录，同名包重新编译安装后也应新建本次运行目录。

必须显式写 `--scope small`，因为默认范围是 `full`。需要全量时将其改为 `--scope full`，每种方案5076项，三方案15228项；这仅说明执行方式，不表示当前完成了该全量验收。上面的命令不包含性能测试，也不包含 `--fp64-reference` 差距统计。

2026-11-12 单一 ACLNN 新签名版本在123机器完成了三方案684项主矩阵，另有 D128补充36项、C4 API16项、C8 API2项通过。后面54项来自该轮辅助脚本，不属于本目录 pytest 默认收集的684项；辅助脚本中包含本轮包名和工作目录约定，不能原样当成通用安装入口。详细记录见[单一接口验收报告](../../../../../operators/quant_flash_attn/quant_schema_single_api_20261112/README.md)。

### 5. 在 Python 脚本中调用

以下示例沿用 `QFA_VENDOR=c4a4` 的包名。保存为 `demo_c4a4.py`，在保留上述运行环境变量的终端执行 `python3 demo_c4a4.py`。示例直接构造已知的 FP4/E8M0 编码，展示输入布局和 API 调用；实际业务需替换为已量化、正确排布的 Q/K/V 和 descale。

```python
import os
import torch
import torch_npu
import cann_ops_transformer_c4a4  # 导入后注册算子

# 沿用前面已确认可用的设备；未设置时示例使用0
DEVICE_ID = int(os.environ.get("QFA_DEVICE", "0"))
torch.npu.set_device(DEVICE_ID)
device = f"npu:{DEVICE_ID}"
ops = torch.ops.cann_ops_transformer

# B=1，Q长度4，KV长度8，Q头8，KV头1，D=128，page=128
T, KV_LEN = 4, 8
NQ, NKV, D = 8, 1, 128
PAGE_SIZE, NUM_PAGES = 128, 1

# FP4：0x00表示两个0，0x22表示两个1。
# E8M0：字节127表示scale=1。view只解释位模式，不执行量化。
def encoded_tensor(shape, byte_value, dtype):
    return torch.full(
        shape, byte_value, dtype=torch.uint8
    ).to(device).view(dtype)

fp4 = torch.float4_e2m1fn_x2
e8m0 = torch.float8_e8m0fnu

q = encoded_tensor((T, NQ, D // 2), 0x00, fp4)
k = encoded_tensor((NUM_PAGES, PAGE_SIZE, NKV, D // 2), 0x00, fp4)
v = encoded_tensor((NUM_PAGES, PAGE_SIZE, NKV, D // 2), 0x22, fp4)

# Qscale使用TND排布，对应DN路径；Q/K沿D每32个值共用scale。
q_descale = encoded_tensor((T, NQ, D // 64, 2), 127, e8m0)
k_descale = encoded_tensor(
    (NUM_PAGES, PAGE_SIZE, NKV, D // 64, 2), 127, e8m0
)
# V沿页内token每32个值共用scale。
v_descale = encoded_tensor(
    (NUM_PAGES, PAGE_SIZE // 64, NKV, D, 2), 127, e8m0
)

cu_q = torch.tensor([0, T], dtype=torch.int32, device=device)
used_kv = torch.tensor([KV_LEN], dtype=torch.int32, device=device)
block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

mask_mode = 0       # 0：无mask；3：右下对齐causal
quant_schema = 2   # 0：OCP；1：CX；2：CX Delta，默认2

common = dict(
    cu_seqlens_q=cu_q,
    seqused_kv=used_kv,
    max_seqlen_q=T,
    max_seqlen_kv=KV_LEN,
    mask_mode=mask_mode,
    layout_q="TND",
    layout_q_descale="TND",
    layout_kv="PA_BBND",
    layout_out="TND",
)

metadata = ops.quant_flash_attn_metadata(NQ, NKV, D, 5, **common)

attn_mask = None
if mask_mode == 3:
    attn_mask = torch.triu(
        torch.ones((2048, 2048), dtype=torch.int8), diagonal=1
    ).to(device)

out, lse = ops.quant_flash_attn(
    q, k, v,
    q_descale, k_descale, v_descale,
    5,  # quant_mode=5：C4A4 MXFP4、FP32 Softmax
    block_table=block_table,
    metadata=metadata,
    attn_mask=attn_mask,
    softmax_scale=D ** -0.5,
    return_softmax_lse=True,
    quant_schema=quant_schema,
    **common,
)

torch.npu.synchronize()
print(out.shape, out.dtype)  # [4, 8, 128]，bfloat16
print(lse.shape, lse.dtype)  # [8, 4]，float32
print(out[0, 0, :8].cpu())  # 示例V全为1，输出也应为1
```

`quant_schema` 只选择算子内部 P 的量化方案；Q/K/V 不会在入口自动从普通浮点数转换为 MXFP4。`softmax_scale` 的 API 默认值是1.0，需要使用常见 attention 缩放时应显式传入 `D ** -0.5`。设备、长度、头数、D、layout 或 mask 配置变化后，应重新生成匹配的 metadata。输入输出约定详见[API说明](../../../docs/c4a4-api-softmax-fp32.md)。

### 6. 包名、vendor 与 torch.ops 命名空间

指定算子构建 Python wheel 时，包名为 `cann_ops_transformer_<vendor>`。上述命令设置：

```bash
export TORCH_EXTENSION_OPS=quant_flash_attn,quant_flash_attn_metadata
export TORCH_EXTENSION_VENDOR=c4a4
```

因此导入名是 `cann_ops_transformer_c4a4`；如果 vendor 改为 `my_qfa`，导入名和 pytest 的 `--package` 就应改为 `cann_ops_transformer_my_qfa`。上面的构建命令对 CMake `VENDOR_NAME` 和 Python `TORCH_EXTENSION_VENDOR` 使用同一个变量，保证安装路径与包名配套。不指定算子、构建整个 Python 包时，名称仍为 `cann_ops_transformer`。

`torch.ops.cann_ops_transformer` 则是代码中固定的 PyTorch 算子命名空间，不随 vendor 改名。导入 Python 包会触发 schema、Meta 和设备实现注册；只有安装 wheel，或仅导入 `torch_npu`，并不等于完成当前自定义算子的注册。当前进程中若其他模块已完成该包导入，就不需要重复导入。

导入后也可以直接通过包的顶层入口访问同一个 dispatcher 算子，下面两组句柄等价：

```python
import torch
import cann_ops_transformer_c4a4 as qfa

metadata_op = qfa.quant_flash_attn_metadata
attention_op = qfa.quant_flash_attn

assert metadata_op is torch.ops.cann_ops_transformer.quant_flash_attn_metadata
assert attention_op is torch.ops.cann_ops_transformer.quant_flash_attn
```

## 性能测试与采集

原第7、8、9节已移至 [Pref_Readme.md](Pref_Readme.md)，包含独立 msprof op 采集、文档性能集合、MXFP8迁入集合及历史对比结果。

## 16k bench 精度集合

通过 `--accuracy-case-set bench16k` 选择用户指定的 12 个场景，替代原 336 个 FIA 合成 case。实现见 [quant_flash_attn_paramset_bench16k.py](quant_flash_attn_paramset_bench16k.py)。默认 document 和 debug 集合沿用原范围。

固定 Q 头数16、KV头数2、D256、page128、causal mask_mode=3、softmax_scale=1/16、LSE关闭、seed421；每个场景保留一份。Prefill和混合请求使用TND Q-scale，Decode使用N2TGD；Q及输出为TND，KV为PA_BBND。默认quant-schema=2，也可显式选择0/1。

所有场景的K、V各使用固定逻辑页池 `[15687*2,128,512] = [31374,128,512]`，512=2个KV头×D256。FP4实际打包形状为 `[31374,128,2,128]`，uint8存储；FP8为 `[31374,128,2,256]`。scale覆盖整个物理池，实际有效长度由下面的数组决定。精度生成器不再额外增加3页，性能生成器也不会缩小物理池。页表仅分配实际最大请求需要的列数，并从整个页池随机选择不共享的物理页。

| 场景 | Q长度 | KV长度 |
| --- | --- | --- |
| single_prefill_1 | [4096] | [4096] |
| single_prefill_2 | [3972] | [12072] |
| single_prefill_3 | [3980] | [16168] |
| dual_prefill_1 | [4096,3988] | [4096,12056] |
| dual_prefill_2 | [2048,2400] | [20264,7960] |
| dual_prefill_3 | [2083,2900] | [24252,16152] |
| decode_1 | [4]×32 | [8192+107i, i=0..15] + [16384+125i, i=0..15] |
| decode_2 | [4]×24 | [32768+32i, i=0..13] + [65536+444i, i=0..9] |
| decode_3 | [1]×32 | [2048+11i, i=0..15]+[256+25i, i=0..15] |
| decode_4 | [4]×33 | [2048+9i, i=0..31]+[16409] |
| mixed_1 | [2044]+[4]×26 | [2044]+[10240+117i, i=0..15]+[18432+125i, i=0..9] |
| mixed_2 | [4096]+[4]×22 | [8192]+[32768+137i, i=0..11]+[22528+100i, i=0..9] |

上表的“+”表示列表拼接，k按1024计算。bench16k是集合名称，实际KV长度可超过16k。`small`和`full`均选取全部12项（Prefill6、Decode4、Mixed2），不再扩展模型、TP或布局组合。cu_seqlens_q由逐请求Q累计得到，seqused_q无padding，seqused_kv保留各条实际长度。

在pytest父目录、已配置CANN和算子包的环境中执行；变量沿用前面的环境配置：

```bash
# 离线枚举，不导入Torch或访问NPU；--output可导出完整参数，拒绝覆盖已有文件。
python3 -m qfa_mxfp4_test.quant_flash_attn_paramset_bench16k --accuracy

# 每种schema使用独立缓存目录。
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_func_rdv.py --accuracy-case-set bench16k --quant-schema 2 --device-id "$DEVICE" --package "$QFA_PACKAGE" --triton-reference --cache-dir "$QFA_WORK/bench16k-accuracy-schema2"

# 离线契约与原有性能/采集回归检查。
python3 -m unittest qfa_mxfp4_test.common.bench16k_contract_test qfa_mxfp4_test.common.perf_contract_test qfa_mxfp4_test.common.prof_contract_test
```

沿用既有CPU/kernel/Triton参考、比较门限和golden-mode分阶段入口；默认kernel CPU参考首次获取metadata仍需要可用NPU。精度和性能共享长度、头数及池容量，分别使用既有随机数生成器；同一性能场景的C4/C8使用同一组FP4编码展开。固定池容量进入缓存及性能参数指纹，旧集合缓存和结果不能冒充新集合。

初始10项于2026-11-16在200容器卡7使用默认schema2，10/10精度和10/10 C4/C8 Graph性能均通过；真实固定池形状和20路显存释放已核对。详见[本次设备验证报告](../../../../../operators/quant_flash_attn/bench16k_npu_validation_20261116_card7_01/README.md)。性能执行和汇总见 [Pref_Readme.md](Pref_Readme.md) 的16k bench集合。

此前decode_3、decode_4在200容器卡7补测通过，见[历史新增Decode补测报告](../../../../../operators/quant_flash_attn/bench16k_decode_extra_20261116_card7_01/README.md)。decode_3现将后16条KV改为256+25i；旧decode_3耗时和缓存对应修改前的长度。

修订后的decode_3已在200容器卡7独立重测：三方精度1/1、C4/C8 Graph性能1/1通过；C4/C8耗时48.793/41.291μs，固定池形状与两路显存释放已核对。见[256起始短尾KV新版报告](../../../../../operators/quant_flash_attn/bench16k_decode3_shorttail_20261116_card7_01/README.md)。

统一采集入口支持按case选择普通msprof、msprof op或两者，使用 `--case decode_3 --profiler both --backend both`；命名case沿用本节固定池和性能输入生成器，使用说明见[Pref_Readme.md](Pref_Readme.md)第1节。

## Qwen3-8B 16项bench

通过 `--accuracy-case-set qwen3_8b` / `--perf-case-set qwen3_8b` 选择独立16项集合：Q32/KV8/D128、TP1，Prefill6/Decode6/Mixed4；当前KV500–16526，decode以Q1为主、decode_3为Q4，prefill Q500–2048、batch1–128。原有集合未精确覆盖此Q/KV头数和head_dim组合。完整case表、覆盖核对、固定页池与执行/导出命令见 [qwen3_8b_bench.md](qwen3_8b_bench.md)。
