# MXFP4 同页双 KV head 合并优化 · 完整验证报告

验证日期：2026-11-25
验收依据：[kernel-optimized.md](../attention/quant_flash_attn/docs/kernel-optimized.md) 开头「优化遵守标准」及其最近两节（`8e0989914`、`2e4ad3565`）的实际执行口径。

---

## 1. 结论摘要

| 项目 | 结果 |
| --- | --- |
| 精度 · document small × 3 布局 | **612/612 通过** |
| 精度 · bench16k × 3 布局 | **36/36 通过** |
| 精度 · Qwen3-8B 页长 128 × 3 布局 | **48/48 通过** |
| 精度 · Qwen3-8B 页长 256 × 3 布局 | **48/48 通过** |
| 精度 · 744 例矩阵合计 | **744/744 通过** |
| 精度 · document full（4860 例，PA_BBND） | **4860/4860 通过** |
| 性能 · bench16k 12 例 × 3 布局 × C4/C8 | **36/36 完成，72 路计时**；候选 C8/C4 几何均值 1.1941，C4 更快 31/36 |
| 性能 · PA_BBND decode C8/C4 | 基线 1.0875 → **1.3918** |
| 性能 · PA_BBND C4 耗时（对同轮基线） | **-8.02%** |
| 采集 · dual_prefill_1 双工具 × C4/C8 × 3 布局 | **12/12 validated，无 data loss** |
| **本轮发现并修复的缺陷** | **合并派发条件过宽：gSize 不是 ≥4 的 2 的幂时结果错误。已修复并复验（72/72）** |

本轮验证最重要的产出不是性能数字，而是**发现了一个此前所有测试集都覆盖不到的正确性缺陷**：合并路径的派发条件没有约束 GQA 分组大小，`N_q/N_kv` 不是 ≥4 的 2 的幂时算子输出错误。744 例矩阵里只有 4 个用例真正走到合并路径，因此历次全绿都没暴露它；只有 document full 的 270 个 `qwen38_27b_TP2` 用例能命中。详见第 4 章。

**性能结论**：收益完全集中在 PA_BBND 的 decode（decode_1 -35.13%、decode_2 -42.90%、decode_3 -14.25%），PA_BNBD / PA_NZ 基本不变，验证了改动的布局隔离性。decode_4 是唯一回退项。**PA_BBND decode 几何均值 1.3918，未达 1.4 目标，差 0.008。**

---

## 2. 验证对象

### 2.1 代码状态

| 项 | 值 |
| --- | --- |
| 分支 | `pref` |
| 优化提交 | `f1d13043b` perf(qfa): merge same-page KV heads for MXFP4 PA_BBND decode |
| 文档提交 | `35f85ba24` docs(qfa): record three-layout comparison and the failed merge tiering |
| 本轮修复 | 合并派发条件收紧（host tiling + AICPU metadata + golden schedule 三处同步） |
| 验证包 | `val2`（全新 build 目录，`BUILD_STATUS=0` / `INSTALL_STATUS=0`） |
| 同轮基线包 | `merge8`（合并前版本，metadata header `[1,*,128,128,28]`） |

构建前对容器 `candidate-src` 与本地工作区做了逐文件归一化 md5 比对（忽略 CRLF 差异），311 个源文件全部一致。

### 2.2 本轮修复内容

三处派发条件同步加入「gSize 为 ≥4 的 2 的幂」判据：

- [quant_flash_attn_tiling_mxfp4.cpp](../attention/quant_flash_attn/op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp)
- [quant_flash_attn_metadata_aicpu.cpp](../attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp)
- [mxfp4_schedule.py](../attention/quant_flash_attn/tests/pytest/qfa_mxfp4_test/common/mxfp4_schedule.py)

```cpp
const int64_t mergeGroupSize = qfaInfo_->n2Size > 0 ? qfaInfo_->n1Size / qfaInfo_->n2Size : 0;
const bool mergeGroupAligned = mergeGroupSize >= 4 && (mergeGroupSize & (mergeGroupSize - 1)) == 0;
```

不满足判据的 shape 回退基线路径，与合并前行为一致。

### 2.3 环境

| 项 | 值 |
| --- | --- |
| 宿主机 / 容器 | `ascend-950pr-223-2-2` / `k831` |
| CANN | 9.2.1 |
| Python / torch / torch_npu | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 |
| 芯片 | Ascend950PR_9579（28 cube core / 56 vector core，L2 128 MB，HBM 126112 MB） |
| 设备 | 性能与采集独占 NPU 7；精度并行使用 NPU 7 / 0 / 4，document full 改用 NPU 7 / 0 / 2 / 3（NPU4 运行期失效，见 5.3） |
| 算子包 | `cann_ops_transformer_qfa_bbnd_copy` |
| 量化 schema | 2（CX Delta） |

---

## 3. 验收范围与执行方法

### 3.1 精度矩阵

按 kernel-optimized.md 最近两节的实际口径，每布局 248 例、三布局合计 744 例：

| 测试集 | 每布局 | 三布局合计 |
| --- | ---: | ---: |
| document small | 204 | 612 |
| bench16k | 12 | 36 |
| Qwen3-8B，页长 128 | 16 | 48 |
| Qwen3-8B，页长 256 | 16 | 48 |
| 合计 | 248 | 744 |

另按「多次大的改动要进行一次 document full」执行 **document full 4860 例**（PA_BBND）。

参考口径：三布局均用 `--golden-mode all`，各自生成输入、CPU 参考与 NPU 输出。参考侧 `kv_cache_layout` 恒为 `PA_BBND`（布局转换在算子入口内完成），输入由同一 seed 生成，因此三布局比较的是同一套参考逻辑与同一份输入。Attention 与 LSE 都比对。

最初 BNBD/NZ 计划用 `--golden-mode npu,compare` 复用 BBND 的缓存，实测发现 Qwen3-8B 单例输入 9.13 GB，从磁盘 `torch.load` 极慢（跑 33 分钟未过半），改为 `all` 在内存重生成后同一集合只需约 7 分钟。每布局使用独立 cache 目录避免并发写冲突。

### 3.2 性能口径

```
test_quant_flash_attn_perf_rdv.py --perf-case-set bench16k --run-perf \
  --perf-backends c4,c8 --perf-mode graph --perf-warmup 3 --perf-rounds 5 --perf-repeats 20
```

12 个用例 × 3 布局，C4 与 C8 同用例、同输入、同设备。Graph 回放 + Event 计时，预热 3 次、测量 5 轮、每轮 20 次调用取中位数。输入准备、metadata 与布局转换在计时外。同轮另测基线包 `merge8` 作 A/B 对照。

### 3.3 采集口径

`prof_quant_flash_attn.py --case dual_prefill_1 --profiler both --backend both`，
普通 msprof 取 `PipeUtilization`，msprof op 取 `Default`；eager 模式，应用预热 20 次，工具预热 10 次、采 5 次。三布局 × C4/C8 共 12 组。

### 3.4 并行化

document full 的耗时 84% 以上在**单线程 CPU 参考**（`cpu_mxfp4_golden_kernel` 是 batch × head × m_tile × task × kv_head 的 Python 嵌套循环），NPU 占比不足 1%，因此多卡收益极小，真正的杠杆是跨用例的 CPU 并行。

实测拟合的成本模型：`t ≈ 1.34e-8 × QK工作量 + 3.0e-8 × 反量化元素数`（秒）。据此 document full 串行约 **12.68 小时**。

执行方案：按 model×TP 切 9 片，每片 540 例，每槽 16 个 `pytest-xdist` worker，按预估成本降序贪心填槽，**每片跑完立即删除该片缓存**，磁盘峰值约 610 GB 而非 909 GB。原计划 NPU 7 / 0 / 4 三槽（48 worker），NPU4 运行期失效后改为 NPU 2 / 3 两槽重跑受影响分片，实际用到 7 / 0 / 2 / 3 四张卡。

并行度受**单卡 HBM** 约束，必须按测试集的最大单例显存选择 worker 数：

| 测试集 | 最大单例输入 | 每卡 worker 数 |
| --- | ---: | ---: |
| document small | 0.57 GB | 16 |
| document full | 2.28 GB | 16 |
| bench16k | 2.18 GB | 16 |
| Qwen3-8B（两种页长） | **9.13 GB** | **6** |

首次用 16 worker 跑 Qwen3-8B 时有 4 例 `torch.OutOfMemoryError`（123.16 GiB 容量被 16×9.13 GB 占满），降到 6 worker 后重跑。该失败是并行资源配置问题，不是算子精度问题。

---

## 4. 本轮发现的缺陷：合并派发条件过宽

### 4.1 发现过程

对 document full 做 41 例系统抽样（每 121 例取 1）时出现 3 例失败，全部是 `qwen38_27b_TP2` + `N2TGD`（`N_q=12, N_kv=2, D=256, PA_BBND`）：

| 用例 | 候选（修复前） | 基线 `merge8` |
| --- | --- | --- |
| `qwen38_27b_TP2_short11_N2TGD_M0_LSE1` | 失败 | 通过 |
| `qwen38_27b_TP2_mtp2_kv32768_b16_N2TGD_M3_LSE1` | 失败 | 通过 |
| `qwen38_27b_TP2_mtp8_kv65536_b1_N2TGD_M0_LSE1` | 失败 | 通过 |

- kernel 参考侧报 `ValueError: C4 metadata traversal exceeded its domain`
- 换成与 metadata 无关的 `--reference-backend global` 后**仍然失败**：`AssertionError: LSE empty-row (+inf) positions do not match the CPU reference`
- 基线包在两种参考下都通过

即：这是**算子本身的正确性回归**，不是参考模型的描述能力问题。

### 4.2 支持域实测

用与 metadata 无关的全局 CPU 参考（`--reference-backend global`）对 `N_kv=2, D=256, PA_BBND, N2TGD` 扫描 `N_q`。每个 `N_q` 测 6 组：mask ∈ {0,3} × (q, kv) ∈ {(1,1024), (4,1024), (17,4096)}，共 72 组。修复前后用**同一套 shape**分别在 `val1`（修复前二进制）与 `val2` 上各跑一遍：

| N_q | gSize | 修复前通过 | 修复后通过 |
| ---: | ---: | ---: | ---: |
| 2 | 1 | 2/6 | 6/6 |
| 4 | 2 | 0/6 | 6/6 |
| 6 | 3 | 0/6 | 6/6 |
| **8** | **4** | **6/6** | 6/6 |
| 10 | 5 | 0/6 | 6/6 |
| 12 | 6 | 0/6 | 6/6 |
| **16** | **8** | **6/6** | 6/6 |
| 20 | 10 | 0/6 | 6/6 |
| 24 | 12 | 2/6 | 6/6 |
| **32** | **16** | **6/6** | 6/6 |
| 48 | 24 | 0/6 | 6/6 |
| **64** | **32** | **6/6** | 6/6 |
| 合计 | | **34/72** | **72/72** |

结论：**合并语义只在 gSize 为 ≥4 的 2 的幂时成立**。原先 golden 侧的 `real_group % 4 == 0` 也不正确——它会误放行 gSize=12。

注意 `N_q=2` 与 `N_q=24` 各有 2 组「通过」，都落在 `q=1, kv=1024` 的单 token decode 上；`q=4` 或 `q=17` 立刻失败。这解释了缺陷为何是**潜伏**的：只测单 token decode 的用例集可能完全看不到它。

机理：合并模式把 M64 tile 按 `mergePartRows = mBaseSize/2 = 32` 拆成两个 KV head part，head 边界落在每 `gSize/2` 行；只有 gSize 为 ≥4 的 2 的幂时该边界才与 Q 装载粒度和 VF 的行分组对齐。

### 4.3 覆盖盲区

修复前各测试集对合并路径的覆盖：

| 测试集 | 例数 | 命中合并派发 | 其中 gSize 不合法 |
| --- | ---: | ---: | ---: |
| document small | 204 | **0** | 0 |
| bench16k | 12 | 4 | 0 |
| Qwen3-8B 页长 128 | 16 | **0** | 0 |
| Qwen3-8B 页长 256 | 16 | **0** | 0 |
| document full | 4860 | 540 | **270** |

744 例矩阵里真正走到合并路径的只有 4 个用例（bench16k 的 4 个 decode，且只有 PA_BBND 实际合并）。**这解释了为什么此前多轮 744 例全绿却没有发现该缺陷**，也说明 document full 不可省略。

### 4.4 修复复验

在 `val2` 上复验，全部通过：

| 复验项 | 参考 | 结果 |
| --- | --- | --- |
| 原 3 个失败用例 | `--reference-backend global` | **3 passed**（68.40s） |
| 原 3 个失败用例 | kernel 参考（默认） | **3 passed**（24.39s） |
| gSize 扫描：N_q ∈ {2,4,6,8,10,12,16,20,24,32,48,64} × mask{0,3} × (q,kv){(1,1024),(4,1024),(17,4096)} | global | **72/72 PASS** |

修复前，4.2 表中所有非 2 的幂或小于 4 的 gSize 均失败；修复后同一扫描 72/72 通过：合法 gSize 仍走合并路径，非法 gSize 回退基线路径。

---

## 5. 精度结果

### 5.1 744 例矩阵

三布局合计 **744/744 通过**，Attention 与 LSE 均比对。

| 测试集 | 每布局 | PA_BBND | PA_BNBD | PA_NZ | 三布局合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| document small | 204 | 204 | 204 | 204 | 612 |
| bench16k | 12 | 12 | 12 | 12 | 36 |
| Qwen3-8B，页长 128 | 16 | 16 | 16 | 16 | 48 |
| Qwen3-8B，页长 256 | 16 | 16 | 16 | 16 | 48 |
| 合计 | 248 | **248** | **248** | **248** | **744** |

PA_NZ 的 Qwen3-8B 页长 256 首轮有 1 例驱动瞬时错误，单卡复跑 16/16 通过（见 5.2、5.4），表中计入复跑结果。

逐集合耗时（含并行度）：

| 测试集 | 布局 | 卡 | worker | 结果 | 耗时 |
| --- | --- | --- | ---: | --- | ---: |
| bench16k | PA_BBND | 7 | 16 | 12 passed | 51.85s |
| Qwen3-8B 页长128 | PA_BBND | 7 | 6 | 16 passed | 433.11s |
| Qwen3-8B 页长256 | PA_BBND | 7 | 6 | 16 passed | 702.28s |
| document small | PA_BBND | 7 | 16 | 204 passed | 599.27s |
| bench16k | PA_BNBD | 7 | 16 | 12 passed | 80.60s |
| Qwen3-8B 页长128 | PA_BNBD | 7 | 6 | 16 passed | 1670.24s |
| Qwen3-8B 页长256 | PA_BNBD | 7 | 6 | 16 passed | 1957.76s |
| document small | PA_BNBD | 4 | 16 | 204 passed | 596.56s |
| bench16k | PA_NZ | 4 | 16 | 12 passed | 101.54s |
| Qwen3-8B 页长128 | PA_NZ | 0 | 6 | 16 passed | 293.09s |
| Qwen3-8B 页长256 | PA_NZ | 0 | 6 | 15 passed + 1 驱动错误 | 433.35s |
| document small | PA_NZ | 4 | 16 | 204 passed | 618.27s |

### 5.2 两次非精度性失败的处置

本轮出现过两次失败，都不是算子精度问题，已定位并重跑：

1. **NPU OOM（并行配置问题）**：首次用 16 worker 跑 Qwen3-8B PA_BBND，4 例 `torch.OutOfMemoryError`。该测试集单例输入 9.13 GB，16×9.13 GB 超出单卡 123.16 GiB。降到 6 worker 后 16/16 通过。
2. **驱动瞬时错误**：Qwen3-8B 页长256 PA_NZ 的 `prefill_4` 报
   `RuntimeError: copy_between_host_and_device_opapi ... aclrtMemcpy, error code is 507899` /
   `ERR00100 PTA call acl api failed` / `rtMemcpy execution failed, reason=driver error:internal error`。
   这是三卡并发下的 host-device 拷贝驱动故障，不是数值比对失败。已在 NPU7 上以 4 worker 单独复跑该集合。

复跑结果见 5.4。

### 5.3 document full（4860 例）

**4860/4860 通过**（PA_BBND，schema 2，Attention + LSE）。按 model×TP 切 9 片，每片 540 例。

| 分片 | 例数 | 卡 | worker | 结果 | 耗时 |
| --- | ---: | --- | ---: | --- | ---: |
| qwen38_27b_TP1 | 540 | 7 | 16 | 540 passed | 2181.70s |
| qwen3_30b_TP1 | 540 | 0 | 16 | 540 passed | 2073.32s |
| qwen36_35b_TP1 | 540 | 2 | 16 | 540 passed | 2228.77s |
| **qwen38_27b_TP2** | **540** | **3** | **16** | **540 passed** | **1124.22s** |
| qwen3_30b_TP2 | 540 | 3 | 16 | 540 passed | 983.22s |
| qwen36_35b_TP2 | 540 | 3 | 16 | 540 passed | 725.02s |
| qwen38_27b_TP4 | 540 | 7 | 16 | 540 passed（复验） | 578.67s |
| qwen36_35b_TP4 | 540 | 3 | 16 | 540 passed | 430.97s |
| qwen3_30b_TP4 | 540 | 2 | 16 | 540 passed | 580.19s |
| 合计 | **4860** | | | **4860 passed** | |

**关键分片是 `qwen38_27b_TP2`**：它包含全部 270 个 gSize=6 的用例，也就是修复前必然失败的那一批。修复后 540/540 通过。

执行过程中遇到两次环境故障，均已定位并重跑：

1. **NPU4 运行期变为不可 retain**：首轮有 7 个分片被调度到 NPU4，全部报
   `TsdOpen failed. devId=4` / `rtSetDevice execution failed, reason=device retain error` /
   `Check param failed, dev can not be NULL!`。该卡在任务开始前的探测是可用的，运行期被其他用户占用后失效。
   改用当时可 retain 的 NPU2 / NPU3 重跑这 7 片，全部通过。
2. **NPU2 上一次驱动瞬时故障**：`qwen38_27b_TP4_mtp1_kv32768_b8_TND_M0_LSE1` 报
   `halMemAlloc failed, drvRetCode=7, ErrCode=507899, desc=[driver error:internal error]`，
   随后同样退化为 device retain 失败。在 NPU7 上整片复验 540/540 通过。

**耗时**：串行估算 12.68 小时；实际墙钟约 80 分钟（含上述卡失效导致的 7 片重跑与 1 片复验）。成功分片累计运行 10905s，在 2–4 张卡 × 16 worker 下并行完成。

### 5.4 复跑结果

| 集合 | 布局 | 卡 | worker | 结果 | 耗时 |
| --- | --- | --- | ---: | --- | ---: |
| Qwen3-8B 页长256 | PA_NZ | 7 | 4 | **16 passed** | 253.33s |

驱动瞬时错误确认为环境性质：同一集合在单卡低并行度下 16/16 通过。计入复跑后，**三布局 744/744 全部通过**。

---

## 6. bench16k Graph 性能

D256、页长 128、schema 2。12 用例 × 3 布局，C4/C8 同用例同输入同设备（NPU7）。预热 3 次、5 轮 × 20 次 Graph replay，取每轮单次 attention 调用耗时的中位数，单位 µs。输入准备、metadata 与布局转换在计时外。

同轮另测基线包 `merge8`（合并前）。**36 组逐例核对 `input_sha256` 与 `case_fingerprint`，全部一致**。

### 6.1 PA_BBND（本次优化的目标布局）

| 用例 | 基线 C4 | 候选 C4 | C4 变化 | 基线 C8 | 候选 C8 | 基线 C8/C4 耗时比 | 候选 C8/C4 耗时比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 251.201 | 250.304 | -0.36% | 242.228 | 241.911 | 0.9643 | 0.9665 |
| single_prefill_2 | 949.817 | 951.536 | +0.18% | 1069.201 | 1067.931 | 1.1257 | 1.1223 |
| single_prefill_3 | 1299.668 | 1301.738 | +0.16% | 1556.789 | 1552.984 | 1.1978 | 1.1930 |
| dual_prefill_1 | 1232.046 | 1235.575 | +0.29% | 1297.472 | 1319.137 | 1.0531 | 1.0676 |
| dual_prefill_2 | 1274.119 | 1271.801 | -0.18% | 1508.539 | 1497.003 | 1.1840 | 1.1771 |
| dual_prefill_3 | 2298.008 | 2301.913 | +0.17% | 2659.333 | 2689.060 | 1.1572 | 1.1682 |
| **decode_1** | **281.911** | **182.877** | **-35.13%** | 312.320 | 310.867 | 1.1079 | **1.6999** |
| **decode_2** | **790.869** | **451.556** | **-42.90%** | 793.276 | 793.626 | 1.0030 | **1.7575** |
| **decode_3** | **34.384** | **29.484** | **-14.25%** | 41.582 | 41.465 | 1.2093 | **1.4064** |
| **decode_4** | **46.386** | **54.138** | **+16.71%** | 48.271 | 48.353 | 1.0406 | **0.8931** |
| mixed_1 | 1335.973 | 1331.674 | -0.32% | 1347.941 | 1351.709 | 1.0090 | 1.0150 |
| mixed_2 | 3789.183 | 3753.304 | -0.95% | 3611.222 | 3574.779 | 0.9530 | 0.9524 |

### 6.2 PA_BNBD

| 用例 | 基线 C4 | 候选 C4 | C4 变化 | 基线 C8 | 候选 C8 | 基线 C8/C4 耗时比 | 候选 C8/C4 耗时比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 249.093 | 247.585 | -0.61% | 241.994 | 243.717 | 0.9715 | 0.9844 |
| single_prefill_2 | 949.120 | 948.522 | -0.06% | 1068.686 | 1069.244 | 1.1260 | 1.1273 |
| single_prefill_3 | 1298.206 | 1297.832 | -0.03% | 1569.249 | 1537.494 | 1.2088 | 1.1847 |
| dual_prefill_1 | 1227.151 | 1230.456 | +0.27% | 1316.128 | 1312.542 | 1.0725 | 1.0667 |
| dual_prefill_2 | 1267.286 | 1271.622 | +0.34% | 1498.091 | 1549.267 | 1.1821 | 1.2183 |
| dual_prefill_3 | 2178.506 | 2189.665 | +0.51% | 2664.394 | 2689.397 | 1.2230 | 1.2282 |
| decode_1 | 189.464 | 190.756 | +0.68% | 313.523 | 314.978 | 1.6548 | 1.6512 |
| decode_2 | 476.589 | 479.239 | +0.56% | 808.767 | 806.845 | 1.6970 | 1.6836 |
| decode_3 | 33.790 | 34.351 | +1.66% | 42.984 | 42.807 | 1.2721 | 1.2462 |
| decode_4 | 46.509 | 49.117 | +5.61% | 50.529 | 50.222 | 1.0864 | 1.0225 |
| mixed_1 | 1162.528 | 1164.027 | +0.13% | 1322.450 | 1321.433 | 1.1376 | 1.1352 |
| mixed_2 | 3117.068 | 3129.139 | +0.39% | 3461.479 | 3427.410 | 1.1105 | 1.0953 |

### 6.3 PA_NZ

| 用例 | 基线 C4 | 候选 C4 | C4 变化 | 基线 C8 | 候选 C8 | 基线 C8/C4 耗时比 | 候选 C8/C4 耗时比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_prefill_1 | 248.246 | 247.833 | -0.17% | 242.408 | 242.385 | 0.9765 | 0.9780 |
| single_prefill_2 | 951.431 | 949.869 | -0.16% | 1067.996 | 1067.983 | 1.1225 | 1.1243 |
| single_prefill_3 | 1299.339 | 1298.644 | -0.05% | 1585.096 | 1539.197 | 1.2199 | 1.1852 |
| dual_prefill_1 | 1228.980 | 1228.135 | -0.07% | 1290.594 | 1314.559 | 1.0501 | 1.0704 |
| dual_prefill_2 | 1269.994 | 1268.678 | -0.10% | 1480.983 | 1534.636 | 1.1661 | 1.2096 |
| dual_prefill_3 | 2203.214 | 2185.858 | -0.79% | 2656.881 | 2673.865 | 1.2059 | 1.2233 |
| decode_1 | 175.583 | 176.383 | +0.46% | 303.275 | 304.147 | 1.7272 | 1.7243 |
| decode_2 | 452.657 | 454.054 | +0.31% | 777.304 | 778.435 | 1.7172 | 1.7144 |
| decode_3 | 33.383 | 32.961 | -1.26% | 41.048 | 41.070 | 1.2296 | 1.2460 |
| decode_4 | 45.322 | 48.293 | +6.55% | 49.540 | 49.216 | 1.0931 | 1.0191 |
| mixed_1 | 1154.694 | 1152.010 | -0.23% | 1269.162 | 1270.686 | 1.0991 | 1.1030 |
| mixed_2 | 3109.658 | 3082.336 | -0.88% | 3381.580 | 3355.785 | 1.0874 | 1.0887 |

### 6.4 分组几何平均（按用例等权）

C8/C4 大于 1 表示 C4 更快。

| 布局 | 范围 | 基线 C8/C4 耗时比 | 候选 C8/C4 耗时比 | 候选 C4 更快 |
| --- | --- | ---: | ---: | ---: |
| PA_BBND | all（12） | 1.0802 | **1.1745** | 9/12 |
| PA_BBND | prefill（6） | 1.1106 | 1.1129 | |
| PA_BBND | **decode（4）** | **1.0875** | **1.3918** | |
| PA_BBND | mixed（2） | 0.9806 | 0.9832 | |
| PA_BNBD | all（12） | 1.2121 | 1.2036 | 11/12 |
| PA_BNBD | prefill（6） | 1.1271 | 1.1315 | |
| PA_BNBD | decode（4） | 1.4036 | 1.3719 | |
| PA_BNBD | mixed（2） | 1.1239 | 1.1151 | |
| PA_NZ | all（12） | 1.2054 | 1.2043 | 11/12 |
| PA_NZ | prefill（6） | 1.1201 | 1.1284 | |
| PA_NZ | decode（4） | 1.4130 | 1.3919 | |
| PA_NZ | mixed（2） | 1.0933 | 1.0958 | |
| 全部 36 组 | all | | **1.1941** | **31/36** |

候选 C4 相对同轮基线 C4 的耗时比几何平均：

| 布局 | C4 耗时比（基线/候选） | C4 耗时变化 |
| --- | ---: | ---: |
| **PA_BBND** | **1.0871** | **-8.02%** |
| PA_BNBD | 0.9923 | +0.78% |
| PA_NZ | 0.9972 | +0.28% |

### 6.5 解读

1. **收益完全集中在 PA_BBND 的 decode**：decode C8/C4 从 1.0875 提升到 1.3918，decode_1 耗时降 35.13%、decode_2 降 42.90%、decode_3 降 14.25%。这与设计一致——派发条件只覆盖 `N2TGD + PA_BBND + N_kv=2 + D256` 的 ND decode。
2. **PA_BNBD / PA_NZ 基本不变**（C4 耗时比 0.9923 / 0.9972），验证了改动的布局隔离性。这两个布局 decode 的高比值（1.37 / 1.39）**来自布局本身**，基线已经是 1.4036 / 1.4130，与本次优化无关。metadata header 也印证了派发隔离：`decode_1` 在 PA_BBND 上是 `[1,1,64,256,28,...]`（合并 Config 6），在 PA_BNBD / PA_NZ 上是 `[1,1,128,128,28,...]`（基线）。
3. **decode_4 是唯一回退项**：PA_BBND +16.71%，PA_BNBD +5.61%，PA_NZ +6.55%。BBND 上的回退源于每核约 24 个 task 刚跨过 `PRELOAD_N=20`，且 `kvHeadNum 2→1` 让负载均衡变粗；BNBD/NZ 上的小幅回退来自共享改动（`MXFP4_TASK_GROUP_N` 由 20 统一为 16、softmax VF 进位改写）。
4. **prefill / mixed 未受影响**（PA_BBND 上逐项变化都在 ±1% 内）。
5. **未达 1.4 目标**：PA_BBND decode 几何均值 1.3918，距 1.4 差 0.008。此前记录的 1.3939 与本轮 1.3918 属同一水平，差异在测量波动内。
6. 五轮极差超过组内中位数 5% 的有：PA_BBND 的 `single_prefill_3`（C4 12.92% / C8 11.36%）、`dual_prefill_1`（C8 13.78%）、`dual_prefill_2`（C4 9.96% / C8 11.93%）、`dual_prefill_3`（C4 7.59%）、`mixed_1`（C4 5.84% / C8 11.82%）、`mixed_2`（C4 5.40%）；BNBD/NZ 的长 prefill 也有类似抖动。decode 四例的极差都在 4% 以内，收益结论稳定。

---

## 7. dual_prefill_1 采集

本例 C4 为 D256 DN、B=2，Q 长度 [4096, 3988]，KV 长度 [4096, 12056]。采集与 Graph 使用相同用例指纹与输入 SHA256。eager 模式，应用预热 20 次；普通 msprof 取最后 3 次目标调用各列中位数（共捕获 25 次调用）；msprof op 用 `--aic-metrics=Default`，工具预热 10 次、采 5 次，每次先对有效核取均值再对 5 次取中位数。时间单位 µs，各流水可重叠。

**12 组采集（3 布局 × C4/C8 × 2 工具）全部 `collection_status=validated`，无 data loss。**

### 7.1 普通 msprof · PipeUtilization

| 布局 | 后端 | 整算子 | AIC MAC | AIC Scalar | AIC MTE1 | AIC MTE2 | AIC Fixpipe | AIV Vector | AIV Scalar | 最长流水 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PA_BBND | C4 | 1233.888 | 814.624 | 752.289 | 489.503 | **1042.154** | 701.878 | 992.407 | 584.942 | AIC MTE2 |
| PA_BBND | C8 | 1287.435 | 1169.731 | **1195.270** | 774.739 | 1096.076 | 877.971 | 1086.045 | 814.434 | AIC Scalar |
| PA_BNBD | C4 | 1229.083 | 811.855 | 725.242 | 466.581 | 703.895 | 697.380 | **992.422** | 585.971 | AIV Vector |
| PA_BNBD | C8 | 1288.080 | 1170.322 | **1192.092** | 773.541 | 1140.930 | 878.378 | 1086.232 | 810.271 | AIC Scalar |
| PA_NZ | C4 | 1231.050 | 814.265 | 748.098 | 518.247 | 596.622 | 698.363 | **992.465** | 588.785 | AIV Vector |
| PA_NZ | C8 | 1287.291 | 1169.715 | **1181.994** | 776.196 | 1017.670 | 877.909 | 1086.049 | 816.941 | AIC Scalar |

### 7.2 msprof op · Default

| 布局 | 后端 | 整算子 | AIC Cube | AIC Scalar | AIC MTE1 | AIC MTE2 | AIC Fixpipe | AIV Vector | AIV Scalar |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PA_BBND | C4 | 1238.315 | 560.912 | 691.001 | 449.422 | 959.972 | 642.581 | 992.672 | 584.522 |
| PA_BBND | C8 | 1292.269 | 1098.440 | 1176.116 | 762.183 | 1084.802 | 863.690 | 1086.414 | 814.513 |
| PA_BNBD | C4 | 1232.333 | 560.912 | 668.166 | 430.793 | 656.312 | 641.691 | 992.597 | 585.882 |
| PA_BNBD | C8 | 1291.401 | 1098.440 | 1172.831 | 760.219 | 1124.265 | 863.880 | 1086.409 | 810.347 |
| PA_NZ | C4 | 1234.687 | 560.912 | 686.657 | 476.114 | 556.919 | 641.251 | 992.923 | 589.000 |
| PA_NZ | C8 | 1293.760 | 1098.440 | 1162.991 | 763.149 | 1009.952 | 863.482 | 1086.567 | 817.232 |

### 7.3 流水占比与 MTE2 活跃带宽（msprof op 同次采集）

| 布局 | 后端 | AIC Cube 占比 | AIC Scalar 占比 | AIC MTE2 占比 | AIV Vector 占比 | AIC MTE2 活跃带宽（GB/s） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PA_BBND | C4 | 49.49% | 60.94% | 84.62% | 80.25% | **119.30** |
| PA_BBND | C8 | 86.53% | 92.62% | 85.42% | 84.24% | 107.10 |
| PA_BNBD | C4 | 49.50% | 58.94% | 57.90% | 80.66% | **89.36** |
| PA_BNBD | C8 | 86.53% | 92.38% | 88.52% | 84.29% | 101.73 |
| PA_NZ | C4 | 49.50% | 60.53% | 49.10% | 80.57% | **105.31** |
| PA_NZ | C8 | 86.47% | 91.56% | 79.51% | 84.16% | 113.25 |

### 7.4 解读

1. `dual_prefill_1` 是 DN prefill，**不走合并派发**，因此三布局的 C4 整算子耗时都在 1229–1234 µs，与 Graph 结果（1228–1236 µs）一致，也与基线持平。本例采集的作用是刻画当前流水瓶颈，不是展示合并收益。
2. C4 侧 AIV Vector 稳定在 992 µs（占比约 80%），是三布局共同的次长流水；PA_BBND 的最长流水是 AIC MTE2（1042 µs，占比 84.62%），显著高于 PA_BNBD 的 704 µs 与 PA_NZ 的 597 µs——这正是合并优化在 decode 上要解决的 GM 取回粒度问题在 prefill 上的体现。
3. 按 100 GB/s 的 MTE2 活跃带宽目标：C4 侧 PA_BBND 119.30、PA_NZ 105.31 达标，**PA_BNBD 89.36 仍差 10.64 GB/s**。
4. C8 侧 AIC Scalar 是最长流水（1163–1195 µs，占比 91.6–92.6%），Cube 占比 86.5%，与 C4 的 49.5% 形成鲜明对比。
5. 本轮使用 `--metrics Default`（kernel-optimized.md 开头「优化遵守标准」指定的口径），**未采集 Roofline**，因此不给出 Roofline 分类与 Cube 性能比曲线。

---

## 8. 证据与复现

### 8.1 归档位置

本轮全部原始证据在 [evidence/ev21](evidence/ev21)：

| 目录 | 内容 |
| --- | --- |
| `accuracy/` | 36 个日志：744 例矩阵逐集合日志、三条并行流日志、document full 9 个分片日志与复验日志、修复确认日志 |
| `perf/` | 6 组逐例 JSON（候选 3 布局 + 基线 3 布局，各 12 例）、6 个 pytest 日志、`bench16k.csv`（36 行汇总）、`summary.json`、`val2-perf-rows.json` |
| `prof/` | 12 组采集的 `profile_summary.json` / `invocation.json` / `worker_record.json` 与全部 CSV（msprof 的 `op_summary`，msprof op 的 8 类指标），共 270 个采集 CSV |
| `sweep/` | gSize 扫描修复前后对照日志（各 72 组） |
| `build/` | val2 的 configure / build / finish 日志 |

容器侧原始产物在 `/home/k831/perf-opt/`：`results/val2-*`、`prof/val2-dual_prefill_1-*`、`full-*.log`、`val2-*.log`。

### 8.2 复现命令

精度（单布局，串行）：

```bash
source /home/k831/perf-opt/env-val2.sh
cd /home/k831/perf-opt/candidate-src/attention/quant_flash_attn/tests/pytest
python -m pytest qfa_mxfp4_test -q -m func_rdv --device-id 7 \
  --package cann_ops_transformer_qfa_bbnd_copy --cache-dir <cache> \
  --accuracy-case-set document --scope small --kv-layout PA_BBND
```

精度（并行，加 `-n <workers>`；worker 数按 3.4 节的显存约束选）。document full 把 `--scope small` 换成 `--scope full`，并用 `-k <model>_TP<n>_` 分片。

性能：

```bash
bash run-perf.sh val2 <outdir> --perf-backends c4,c8 --kv-layout PA_BBND --perf-kind all
```

采集：

```bash
bash run-prof-both.sh PA_BBND val2
```

### 8.3 代码状态

本轮修复涉及 3 个文件、+13/-3 行，**位于工作区，尚未提交**：

| 文件 | 变更 |
| --- | --- |
| `attention/quant_flash_attn/op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp` | +6 / -1 |
| `attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp` | +4 / -1 |
| `attention/quant_flash_attn/tests/pytest/qfa_mxfp4_test/common/mxfp4_schedule.py` | +3 / -1 |

分支 `pref`，HEAD = `35f85ba24`。构建前已逐文件核对容器 `candidate-src` 与本地工作区一致（311 个源文件，忽略 CRLF 差异后 md5 全等）。

### 8.4 本轮未做的事

- **未采集 Roofline**：按 kernel-optimized.md 开头「优化遵守标准」使用 `msprof op --aic-metrics=Default`，因此不给出 Roofline 分类与 Cube 性能比曲线。文档最近两节用的是 Roofline，口径不同，不可直接对比。
- **document full 只跑 PA_BBND**：标准原文是「进行一次 document full 测试集」，本轮按一次执行。BNBD/NZ 的 full 未跑。
- **未继续追求 1.4**：PA_BBND decode 停在 1.3918。此前分析确认的分档方案需要跨算子接口改动（host 侧拿不到真实 `seqused_kv` 分布），已按方案 B 搁置。
- **未提交代码**：修复留在工作区，等待指示。

### 8.5 环境故障备忘

本轮共遇到 4 次环境性质的失败，全部与被测算子无关，均已重跑通过：

| 现象 | 卡 | 影响 | 处置 |
| --- | --- | --- | --- |
| `torch.OutOfMemoryError` | 7 | Qwen3-8B PA_BBND 4 例 | 16 worker → 6 worker |
| `aclrtMemcpy` 507899 驱动内部错误 | 0 | Qwen3-8B 页长256 PA_NZ 1 例 | NPU7 单卡 4 worker 复跑，16/16 |
| `TsdOpen failed` / device retain error | 4 | document full 7 个分片 | 改用 NPU2 / NPU3 重跑，全部通过 |
| `halMemAlloc failed` 507899 | 2 | document full 1 例 | NPU7 整片复验，540/540 |

共享机器上其他用户的任务会改变卡的可用性：**任务开始前的探测通过不代表运行期一直可用**。NPU4 在探测时无进程，运行期被占用后即不可 retain。后续长跑建议在分片失败时自动换卡重试，而不是固定卡列表。

---

## 9. 增量验证：运行时流水节奏下发（D4 / A9）

> 第 1–8 章记录的是合并本体（val2）的验证。本章是其后追加的 A9 改动的独立验证，
> 口径完全沿用第 3、4 章：同用例、同输入 SHA256、同卡、Graph 回放 + Event 计时、预热 3、5 轮 × 20 次取中位数。
> 8.4 中「未继续追求 1.4」「未提交代码」两条已被本章取代。

### 9.1 改动范围

9 个文件、+88/-10 行，详见 [optimization-design.md](optimization-design.md) 的 A9.3。核心是把 `GROUP_N` / `PRELOAD_N` 从编译期写死改为由 AICPU 经 metadata header 索引 6/7 下发，kernel 在 `Init` 时读回；缓冲区仍按编译期上界静态分配，运行时只取更小值。

### 9.2 精度：744 例矩阵

| 布局 | 卡 | bench16k | qwen3_8b | qwen3_8b_page256 | document small |
| --- | --- | ---: | ---: | ---: | ---: |
| PA_BBND | NPU7 | 12 passed | 16 passed | 16 passed | 204 passed |
| PA_BNBD | NPU0 | 12 passed | 16 passed | 16 passed | 204 passed |
| PA_NZ | NPU2 → **NPU0 复跑** | 12 passed | 16 passed | 16 passed | 首轮 76 failed / 128 passed → **204 passed** |

**PA_NZ 首轮 76 例失败的定性**：全部是 `kv32768` / `kv131072` 大用例，报错统一为设备侧超时，**没有一例是精度比对失败**：

```
[W1125 11:10:24] NPU warning, error code is 507014 [Error]: The aicore execution times out.
EE9999: rtDeviceSynchronizeWithTimeout execution failed, reason=aicore timeout
```

76 例的失败时间戳集中在同一秒（11:10:24），且 `npu-smi` 在无任何进程占用时仍报 NPU2 为 100% util / 494 W / 85 °C。同参数换 NPU0 复跑 204 passed（546.63 s）。判定为卡侧故障。

**教训**：`76 failed` 必须逐条看失败原因，不能直接当成精度回退；也不能仅凭「改动对该布局不生效」的推理就跳过复跑。

### 9.3 性能：12 例 × 三布局

val2（合并本体）→ d4a（叠加 A9），同卡 NPU7 串行采集：

| 用例 | val2 C4 | d4a C4 | C4 变化 | val2 C8/C4 | d4a C8/C4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode_1 | 182.877 | 181.861 | -0.56% | 1.6999 | 1.7127 |
| decode_2 | 451.556 | 452.414 | +0.19% | 1.7575 | 1.7540 |
| decode_3 | 29.484 | 26.109 | **-11.45%** | 1.4064 | **1.5760** |
| decode_4 | 54.138 | 51.243 | **-5.35%** | 0.8931 | **0.9606** |
| single_prefill_1 | 250.304 | 248.912 | -0.56% | 0.9665 | 0.9744 |
| single_prefill_2 | 951.536 | 948.132 | -0.36% | 1.1223 | 1.1280 |
| single_prefill_3 | 1301.738 | 1300.984 | -0.06% | 1.1930 | 1.1906 |
| dual_prefill_1 | 1235.575 | 1235.605 | +0.00% | 1.0676 | 1.0634 |
| dual_prefill_2 | 1271.801 | 1270.868 | -0.07% | 1.1771 | 1.1962 |
| dual_prefill_3 | 2301.913 | 2319.427 | +0.76% | 1.1682 | 1.1791 |
| mixed_1 | 1331.674 | 1334.519 | +0.21% | 1.0150 | 1.0090 |
| mixed_2 | 3753.304 | 3832.483 | +2.11% | 0.9524 | 0.9453 |

| 布局 | decode GM val2 → d4a | all GM val2 → d4a | C4 耗时几何比（val2/d4a） |
| --- | --- | --- | ---: |
| **PA_BBND** | 1.3918 → **1.4603** | 1.1745 → 1.1954 | **1.0134**（-1.32% 耗时） |
| PA_BNBD | 1.3719 → 1.3823 | 1.2036 → 1.2043 | 1.0024 |
| PA_NZ | 1.3919 → 1.3918 | 1.2043 → 1.2059 | 1.0002 |

**目标达成**：PA_BBND decode 几何均值 1.4603 > 1.4。

`mixed_2` +2.11% 是唯一超出 ±1% 的非 decode 项；该例在第 6 章记录的五轮极差为 5.40%，判为噪声，不作为回退结论。

### 9.4 布局隔离证据

逐例读 `metadata_header[6]/[7]`：

| 布局 | decode_1 | decode_2 | decode_3 | decode_4 | 其余 8 例 |
| --- | --- | --- | --- | --- | --- |
| PA_BBND | 0/0 | 0/0 | **4/7** | **8/11** | 0/0 |
| PA_BNBD | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| PA_NZ | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

`mBase` / `taskN` 也印证派发未变：BBND 四个 decode 仍是 `64 / 256`（合并 Config 6），其余全部 `128 / 128`。

### 9.5 分档阈值扫参

| 方案 | decode GM | 相对 val2 的 C4 耗时变化 |
| --- | ---: | ---: |
| 全部 4/7 | 1.4487 | -3.44% |
| 全部 8/11 | 1.4544 | — |
| 全部 2/5 | 1.4620 | — |
| **按 `maxCoreBlock` 分档（采纳）** | 1.4603 | **-4.06%** |

`2/5` 的 GM 最高但 C4 耗时更差，差异来自 C8 侧抖动。**按 C4 绝对耗时口径取分档方案**。

### 9.6 本章未做的事

- **未跑 document full**：A9 不改派发条件、不改形状与内存预算，只改两个流水节奏常量，且 kernel 侧对非法下发值有回退校验。按 kernel-optimized.md 的「小改动」档执行 `document small` + `bench16k` × 三布局。**但 9.2 的教训仍然成立：744 例矩阵里真正命中合并路径的只有 4 个用例**，若要覆盖更多 gSize / 形状组合，仍需补 full。
- **未采集 msprof**：A9 的收益机理（排空段缩短）由 task 数与 tick 模型直接解释，本轮未做流水级采集。
- **未验证 `maxCoreBlock ∈ [20,40)` 的真实业务形状**：阈值 40/20 只在 bench16k 四个 decode 上标定。

### 9.7 同期被证伪的一项改动（未提交）

尝试把 ND softmax VF 的 causal 判定从 `Duplicate + Cast + Compare` 改为向量-标量 `Compares`，每 4 行省 8 条向量指令。同轮同卡 A/B 结果：PA_BBND 12 例 C4 耗时几何比 **0.9953**（慢 0.47%）、PA_BNBD **0.9999**（持平）。

原因是 decode 不受 VF 吞吐限制，而真正 VF 受限的 DN prefill 走的是另一个函数。该改动**已回退**，详细过程与编译器约束见 optimization-design.md 的 B13。

---

## 10. 增量验证：DN 驻留快路径 8→16（已否决并回退）

> 目标：把 DN softmax 的重复装载减半，压低三布局共同的 AIV Vector 992 µs。
> 结论：**精度全绿，但性能回退 13~14%，方案否决**。代码已恢复到 `bfc88b514`。
> 完整推演见 [optimization-design.md](optimization-design.md) 的 B14。

### 10.1 先证伪了原目标

原计划直接改 `SoftmaxDnResidentEightVf` 的驻留行数。资源探针显示该函数**在 D256 上从未被实例化**——派发门槛 `TaskN == 256` 把它限制在 D128，而 12 个性能用例全是 D256。因此先放开该门槛，再提高驻留行数。

### 10.2 寄存器溢出验证（本轮的硬要求）

用 `bisheng --cce-res-usage` 对同一 tiling key（D256 DN prefill，25953538 / 93062402）逐档实测：

| 方案 | load / 32 行 | VReg number | PReg | **Stack size** | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| 通用路径 | 64 | 14 | 4 / 7 | 0 | 基线 |
| 放开门槛 + 驻留 8 | 56 | 15 | 3 | 0 | 无溢出 |
| **放开门槛 + 驻留 16** | **48** | **23** | **3** | **0** | **无溢出，采纳进入 A/B** |
| 放开门槛 + 驻留 32 | 32 | 32（顶格） | 3 | **2304 B** | **溢出，编译期即否决** |

驻留 32 的 `Stack size = 2304 B = 9 × 256 B`，即 9 个溢出槽。**由此实测出 devkit 未暴露的硬件上限：AIV 向量寄存器文件 = 32 个 256B 寄存器。** 驻留 32 理论峰值 38，超限 6 个。

探针方法（单 key + `dump_cce`，约 2 分钟一轮）已写入 B14.2，可作为今后改 VF 的常规关卡。

### 10.3 精度：744 例三布局全通过

| 布局 | 卡 | bench16k | qwen3_8b | page256 | document small |
| --- | --- | ---: | ---: | ---: | ---: |
| PA_BBND | NPU7 | 12 passed | 16 passed | 16 passed | 204 passed |
| PA_BNBD | NPU0 | 12 passed | 16 passed | 16 passed | 204 passed |
| PA_NZ | NPU3 | 12 passed | 16 passed | 16 passed | 204 passed |

### 10.4 性能：DN prefill 全线回退

同轮同卡 NPU7、Graph，`d4c`（基线）与 `d5a`（候选）交替采集：

| 用例 | BBND d4c → d5a | 变化 | BNBD d4c → d5a | 变化 |
| --- | --- | ---: | --- | ---: |
| decode_1 | 183.696 → 183.229 | -0.25% | 190.412 → 189.538 | -0.46% |
| decode_2 | 453.382 → 454.896 | +0.33% | 481.052 → 479.479 | -0.33% |
| decode_3 | 26.174 → 26.739 | +2.16% | 34.143 → 33.997 | -0.43% |
| decode_4 | 51.205 → 51.402 | +0.38% | 48.511 → 47.792 | -1.48% |
| single_prefill_1 | 250.763 → 296.566 | **+18.3%** | 247.293 → 296.601 | **+19.9%** |
| single_prefill_2 | 949.623 → 1199.157 | **+26.3%** | 949.605 → 1197.873 | **+26.1%** |
| single_prefill_3 | 1322.594 → 1650.310 | **+24.8%** | 1296.804 → 1649.858 | **+27.2%** |
| dual_prefill_1 | 1233.460 → 1509.855 | **+22.4%** | 1228.362 → 1507.386 | **+22.7%** |
| dual_prefill_2 | 1271.237 → 1595.229 | **+25.5%** | 1267.390 → 1595.436 | **+25.9%** |
| dual_prefill_3 | 2343.854 → 2756.866 | **+17.6%** | 2257.977 → 2641.830 | **+17.0%** |
| mixed_1 | 1334.518 → 1518.490 | +13.8% | 1159.894 → 1497.101 | **+29.1%** |
| mixed_2 | 3807.185 → 4201.651 | +10.4% | 3181.493 → （未取到） | — |

C4 耗时几何比：**PA_BBND 0.8849（慢 13.01%）、PA_BNBD 0.8745（慢 14.35%）**。

**内置对照排除了噪声解释**：decode 四例走 ND 路径、本次完全未改，在两布局上都稳定在 ±0.5%（decode_3 BBND +2.16% 属该例已知抖动）。若是卡上系统性膨胀，decode 会一起变慢。只有被改动的 DN prefill 全线劣化，且两布局幅度一致。

PA_NZ 未采集：两布局已互证同一结论，停止以节省机时。

### 10.5 机理（**已被第 11 章的对照推翻**）

> 本节结论错误，保留以记录推演过程。当时只有 d5a 一个变体，混淆了「通用→驻留」与「驻留 8→16」两个因素。第 11 章补测后证明回退来自后者。

当时的推断：load 少 25%，时间多 22%——驻留路径丢掉了通用路径的组间软件流水。通用路径把「组 i+1 求 max」与「组 i 打包」交错发射，两条独立指令流互相掩盖延迟；驻留路径变成逐组串行 `装 32 行 → 归约树 → 指数链 → 打包`，每组裸露约 11 个相关操作。

### 10.6 本轮未做的事

- **未分离归因**：D256 此前从未走驻留路径，本次 delta 同时含「通用→驻留（丢流水）」与「驻留 8→16」两个因素。机理解释指向前者，但没有单独测量「放开门槛 + 驻留 8」来分离。**→ 已在第 11 章补测，结论相反。**
- **未采集 msprof**：回退结论由 A/B 与内置对照直接支撑，未做流水级采集验证 ILP 假设。
- **未测 D128 性能**：D128 DN prefill 也从驻留 8 变成 16，精度已验证，但性能用例集里没有 D128，无法测量。

### 10.7 环境故障备忘（本轮新增）

**磁盘写满伪装成算子缺陷。** 首轮 BNBD qwen3_8b 2 failed、NZ page256 16 全 failed，报错是：

```
RuntimeError: basic_ios::clear: iostream error
RuntimeError: [enforce fail at inline_container.cc:668] . unexpected pos 4295993664 vs 4295993496
```

查 `df` 发现 `/home/k831` 已 **100% 满，仅剩 177 MB**。`--cache-dir` 的参考张量缓存单布局可达 320 GB，三布局并行近 1 TB。清理旧缓存与废弃构建目录后释放到 1.1 TB，并改为**每个测试集跑完立即删缓存**，同参数重跑全绿。

**教训**：`iostream error` / `inline_container` 报错是 torch 存档被截断，既不是精度不符也不是设备错误，先看磁盘。

---

## 11. 归因对照：驻留 8（d5b）——回退归因于驻留 16

> 第 10 章只有 d5a 一个变体，无法区分「通用→驻留（丢组间流水）」与「驻留 8→16」。
> 本章补测 d5b：**只放开 `TaskN` 门槛，驻留函数体逐字保持 8 行**（相对 HEAD 仅 +4/-1）。
> 结论：**驻留 8 性能中性，13~14% 的回退完全来自 16 行那一档**。第 10.5 节的机理推断据此作废。

### 11.1 对照关系

| 变体 | 门槛 | 驻留行数 | D256 DN 实际路径 |
| --- | --- | ---: | --- |
| HEAD（基线） | `TaskN == 256` | 8 | 通用路径（有组间流水） |
| **d5b** | 放开 | **8** | 驻留 8（无组间流水） |
| d5a | 放开 | 16 | 驻留 16（无组间流水） |

### 11.2 精度：744 例三布局全通过

| 布局 | 卡 | bench16k | qwen3_8b | page256 | document small |
| --- | --- | ---: | ---: | ---: | ---: |
| PA_BBND | NPU2 | 12 passed | 16 passed | 16 passed | 204 passed |
| PA_BNBD | NPU0 | 12 passed | 16 passed | 16 passed | 204 passed |
| PA_NZ | NPU3 | 12 passed | 16 passed | 16 passed（复跑） | 204 passed |

NZ page256 首轮 1 例 `aclrtMemcpy, error code is 507899`（host→device 拷贝的瞬时驱动错误，与 8.5 节同类），复跑 16 passed。

### 11.3 性能：prefill 中性

同轮同卡 NPU3、Graph，`d4d`（基线）与 `d5b` 交替采集，PA_BBND：

| 用例 | d4d C4 | d5b C4 | 变化 |
| --- | ---: | ---: | ---: |
| single_prefill_1 | 248.311 | 244.525 | -1.5% |
| single_prefill_2 | 949.994 | 923.799 | -2.8% |
| single_prefill_3 | 1329.339 | 1365.249 | +2.7% |
| dual_prefill_1 | 1230.529 | 1207.808 | -1.8% |
| dual_prefill_2 | 1299.896 | 1349.443 | +3.8% |
| dual_prefill_3 | 2356.359 | 2290.758 | -2.8% |
| mixed_1 | 1365.259 | 1340.453 | -1.8% |
| mixed_2 | 3979.894 | 3974.846 | -0.1% |

**prefill 八例 C4 耗时几何比 1.0057（快 0.57%），符号正负交替，判为中性。** 同口径下 d5a 是 +18~26%。

### 11.4 排除离群点：decode_3

d5b 全量轮里 decode_3 报 26.403 → 38.466（+45.7%），把 12 例整体几何比拉到 0.9715。该例走 ND 路径、本次未改，且只有 26 µs。补一轮**串行**decode-only 三方对照（NPU3，无并发）：

| 用例 | d4e 基线 | d5b 驻留 8 | d5a 驻留 16 |
| --- | ---: | ---: | ---: |
| decode_1 | 182.445 | 183.036 | 184.451 |
| decode_2 | 452.609 | 454.026 | 453.473 |
| **decode_3** | **25.938** | **26.437** | **26.467** |
| decode_4 | 51.226 | 51.602 | — |

decode_3 回到 26.4 µs，**38.466 确认为离群点**。decode 四例 C4 几何比 d4e→d5b 0.9919、d4e→d5a 0.9891，均在噪声内。

**故 d5b 整体判定为中性，而非 0.9715 暗示的「慢 2.9%」。** 教训：短用例（<30 µs）的单点异常足以扭曲 12 例几何均值，必须复测。

### 11.5 修正后的结论

1. **驻留结构本身不慢**——逐组串行的驻留 8 与带组间流水的通用路径等价，说明组内 ILP 已足够，组间交叠并非必需。第 10.5 节的推断错误。
2. **少 12.5% 的 load 换不出时间**，与 9.7 节（B13）一致：DN prefill 不是 load 条数受限。
3. **真正的代价是寄存器压力，且发生在 spill 阈值之下**：驻留 8 → VReg 15，驻留 16 → VReg 23，两者 `Stack size` 均为 0，但后者慢 13~14%。23 个活跃寄存器压缩了调度窗口，延迟掩盖不住。
   **`Stack size == 0` 只保证没 spill，不保证没有性能损失。** 已补入 A7.2 红线 10。

### 11.6 处置

d5b 性能中性、无可测收益，按与 9.7 节一致的口径**同样回退**，不提交。门槛 `TaskN == 256` 一并还原。

**方向关闭**：在这条 VF 上「多驻留几行来省 load」难有收益——省下的 load 换不出时间，多占的寄存器要付调度代价。性能拐点（VReg ~15→23 之间）远低于 spill 阈值 32。

### 11.7 环境故障备忘（本轮新增）

- **共享卡被其他容器占满会让新任务静默挂起**：NPU7 被另一容器的 6 个进程占去 35.7 GB HBM，我们的 pytest 起来后 CPU 0%、日志 0 字节、xdist worker 不生成，挂了 27 分钟。`npu-smi` 里的 PID 在本容器 `/proc` 下不存在即可判定是他人进程。处置：换到空闲卡（本轮改用 NPU2/0/3），并把 perf 脚本的硬编码 `--device-id 7` 参数化。
- **自己的后台任务之间也会争用**：NZ 精度复跑与 decode 性能采集一度同时跑在 NPU3 上，该轮性能数据作废。长跑脚本之间必须显式串行（等前一个的结束标记），不能假设「排队了就不会重叠」。