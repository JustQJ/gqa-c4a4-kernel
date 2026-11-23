# Qwen3-8B 16-case bench

参数定义：[quant_flash_attn_paramset_qwen3_8b.py](quant_flash_attn_paramset_qwen3_8b.py)。精度、性能和统一 profiler 的集合名均为 `qwen3_8b`，`small` / `full` 都选择这16项，可用 `--perf-kind` 筛选6项 prefill、6项 decode、4项 mixed。

用 `--accuracy-case-set qwen3_8b_page256`、`--perf-case-set qwen3_8b_page256` 或 profiler 的 `--case-set qwen3_8b_page256` 运行页长 256 的同一组场景。页池为 32768 页，Q/KV 长度、头数和总 token 容量与原集相同。

## 模型维度与原有覆盖

根据用户提供的 Qwen3-8B 配置，按 **TP=1** 使用 `N_q=32, N_kv=8, D=128`，GQA分组数为4，`softmax_scale=1/sqrt(128)`。逻辑Q为 `[sum(Q),32,128]`，每个token的K/V各为 `[8,128]`。这是单层attention的量化算子bench；模型配置的BF16权重、36层、RoPE等不属于本参数集的整网验证范围。

新增前离线枚举当前集合，精确匹配 `(N_q,N_kv,D)=(32,8,128)` 均为0：

| 原集合 | case数 | 精确匹配数 |
| --- | ---: | ---: |
| document full精度 | 4860 | 0 |
| debug full | 216 | 0 |
| document full性能 | 1395 | 0 |
| 迁移MXFP8性能 | 211（含6项skip） | 0 |
| bench16k | 12 | 0 |

原document的 `qwen3_30b` 为 Q32/KV4/D128，bench16k为 Q16/KV2/D256，均不能替代这份配置。

## 16项参数

下表case短名前统一加 `qwen3_8b_`。k按1024计算；Q/KV均为逐请求长度，KV包含当前Q对应的token。`R(n)` 精确定义为 `[1024 + floor(15360*i/(n-1)) for i in range(n)]`，包含1024、16384及不对齐的中间长度；“+”表示列表拼接。

| case短名 | 类型 | batch | Q长度 | KV长度 |
| --- | --- | ---: | --- | --- |
| prefill_1 | Prefill | 1 | [500] | [500] |
| prefill_2 | Prefill | 1 | [1024] | [4096] |
| prefill_3 | Prefill | 1 | [2048] | [16384] |
| prefill_4 | Prefill | 2 | [500,1024] | [1024,8192] |
| prefill_5 | Prefill | 2 | [1024,2048] | [4096,16384] |
| prefill_6 | Prefill | 4 | [500,1024,1536,2048] | [2048,4096,8192,16384] |
| decode_1 | Decode | 8 | [1]×8 | [1011,1022,1023,1024,2048,4099,4008,16526] |
| decode_2 | Decode | 16 | [1]×16 | [2048,4099,4008,2048]×4 |
| decode_3 | Decode | 32 | [4]×32 | [4096]×32 |
| decode_4 | Decode | 64 | [1]×64 | [4555,4900,8900,8192]×16 |
| decode_5 | Decode | 96 | [1]×96 | R(96) |
| decode_6 | Decode | 128 | [1]×128 | [16384]×128 |
| mixed_1 | Mixed | 16 | [500]+[1]×15 | [1024]+R(15) |
| mixed_2 | Mixed | 32 | [1024]+[1]×31 | [4096]+R(31) |
| mixed_3 | Mixed | 64 | [500,2048]+[1]×62 | [2048,16384]+R(62) |
| mixed_4 | Mixed | 128 | [1024,1536]+[1]×126 | [8192,16384]+R(126) |

全部使用 page128、PA_BBND、causal mask3、LSE关闭、seed421、默认quant-schema2和C8 p_scale448。Prefill/Mixed使用TND Q-scale，Decode使用N2TGD；`cu_seqlens_q`按实际Q累计，`seqused_q`无padding。

按用户修订，`decode_3`为Q4的Decode/MTP场景，其余decode及mixed中的decode仍为Q1；`prefill_1`为Q=KV=500的完整prefill。当前KV实际范围为500–16526，覆盖原1k–16k区间及指定的非对齐边界。

按用户要求，`KV_CACHE_NUM_BLOCKS = 16384 * 4 = 65536`，K/V各固定逻辑页池 `[65536,128,1024]`，末维 `1024=8*128`；最大decode实际需要16384页，物理池保持65536页。FP4打包各为 `[65536,128,8,64]`，C8各为 `[65536,128,8,128]`，对应K+V payload分别8GiB/16GiB，不包含Q、scale及workspace。精度与性能共享池容量和长度定义，各自沿用既有输入生成器。新集合使用独立case名称和参数指纹，报告会校验完整参数。

## 运行与导出

从 `attention/quant_flash_attn/tests/pytest` 目录运行。`$DEVICE`、`$QFA_PACKAGE`、`$QFA_WORK`沿用现有环境，输出和缓存目录应使用新路径。

```bash
# 离线查看摘要/导出全部16项性能参数（不覆盖已有文件）
python3 -m qfa_mxfp4_test.quant_flash_attn_paramset_qwen3_8b
python3 -m qfa_mxfp4_test.quant_flash_attn_paramset_qwen3_8b --output "$QFA_WORK/qwen3_8b_cases.json"

# 精度：沿用原CPU/kernel/Triton参考和比较门限
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_func_rdv.py --accuracy-case-set qwen3_8b --quant-schema 2 --device-id "$DEVICE" --package "$QFA_PACKAGE" --triton-reference --cache-dir "$QFA_WORK/qwen3_8b-accuracy-schema2"

# C4/C8 Graph性能及16项汇总
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py --run-perf --perf-case-set qwen3_8b --perf-backends c4,c8 --perf-mode graph --quant-schema 2 --device-id "$DEVICE" --package "$QFA_PACKAGE" --perf-output "$QFA_WORK/qwen3_8b-perf"
python3 -m qfa_mxfp4_test.common.perf_report "$QFA_WORK/qwen3_8b-perf" --case-set qwen3_8b --expected 16 --output "$QFA_WORK/qwen3_8b-perf/report"

# 普通msprof，沿用性能入口的输入和池容量
python3 -m pytest qfa_mxfp4_test/test_quant_flash_attn_perf_rdv.py --run-perf --perf-case-set qwen3_8b --perf-backends c4,c8 --msprof --device-id "$DEVICE" --package "$QFA_PACKAGE" --perf-output "$QFA_WORK/qwen3_8b-msprof"

# 统一采集入口：离线列出全部参数；选择单例采集
python3 -m qfa_mxfp4_test.prof_quant_flash_attn --case-set qwen3_8b --list-cases
python3 -m qfa_mxfp4_test.prof_quant_flash_attn --case-set qwen3_8b --case mixed_4 --device "$DEVICE" --package "$QFA_PACKAGE" --profiler both --backend both --metrics BasicInfo --output "$QFA_WORK/qwen3_8b-mixed4-prof"

# 无NPU的契约检查及原集合回归
python3 -m unittest qfa_mxfp4_test.common.qwen3_8b_contract_test qfa_mxfp4_test.common.bench16k_contract_test qfa_mxfp4_test.common.perf_contract_test qfa_mxfp4_test.common.prof_contract_test
```

## 设备验证（2026-11-17）

在200机器的 `tp_a4c4_gqa_kernel` 容器中，仅使用物理NPU7、quant-schema2及 `cann_ops_transformer_qfa_fuse_cmp_20261117` 包验证。全部11个阶段于16:21完成，退出码均为0。

| 验证 | Qwen3-8B | 原bench16k回归 |
| --- | --- | --- |
| kernel/Triton参考精度 | 16/16通过 | 12/12通过 |
| C4/C8 Graph性能入口 | 16/16通过 | 12/12通过 |
| 报告生成及参数指纹校验 | 通过 | 通过 |

43项原生离线契约检查通过，原生pytest分别收集32/24项；四份JUnit共56项，0失败、0错误、0跳过。28项Graph的56路输出有限、回放抽样一致，allocated在执行前和释放后均为0；计时为3次预热、5轮各20次调用。Qwen3-8B的65536页固定池及FP4/FP8实际输入形状均已核对。

统一profiler入口冒烟覆盖Qwen3-8B `decode_1` 的C4普通msprof和msprof op BasicInfo，以及不传 `--case-set` 的bench16k `decode_3` C4普通msprof。三路均validated、无data loss、释放后allocated为0；普通msprof均为20次预热加5次正式调用，BasicInfo为5次捕获。

本轮结论适用于上述两个固定集合、schema2和三个profiler冒烟组合，不包含document full、其他schema、BF16性能或InstrTimeline。
