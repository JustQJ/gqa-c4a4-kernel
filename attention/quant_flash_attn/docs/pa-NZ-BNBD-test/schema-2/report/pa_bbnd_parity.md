# C4A4 PA_BNBD / PA_NZ 与 PA_BBND 精度对比报告

- 测试范围：`small`；`quant_schema=2`
- 算子包：`cann_ops_transformer_c4a4`
- 总结：**PASS**（48/48）
- 判据：两种布局与 BBND 的 Attention、LSE relative-L2 均不超过 1%，且各自通过原 BBND golden 阈值。
- 输入：逐案例核对原始 Q/K/V、scale、页表和长度张量完全一致。

| 布局 | 通过/总数 | 最大 Attention relative-L2 | 最大 LSE relative-L2 |
|---|---:|---:|---:|
| PA_BNBD | 24/24 | 0 | 0 |
| PA_NZ | 24/24 | 0 | 0 |

## 逐案例结果

| 用例 | Golden | 输入一致 | Attention relative-L2 | LSE relative-L2 | 结果 |
|---|---|---|---:|---:|---|
| qwen36_35b_TP2_smoke_TND_M0_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M0_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M0_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M0_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M3_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M3_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M3_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_TND_M3_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M0_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M0_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M0_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M0_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M3_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M3_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M3_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_smoke_N2TGD_M3_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M0_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M0_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M0_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M0_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M3_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M3_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M3_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_TND_M3_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M0_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M0_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M0_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M0_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M3_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M3_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M3_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_empty_and_q_padding_N2TGD_M3_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M0_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M0_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M0_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M0_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M3_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M3_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M3_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_TND_M3_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M0_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M0_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M0_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M0_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M3_LSE0_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M3_LSE0_PA_NZ | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M3_LSE1_PA_BNBD | PASS | 是 | 0 | 0 | PASS |
| qwen36_35b_TP2_page_tail_N2TGD_M3_LSE1_PA_NZ | PASS | 是 | 0 | 0 | PASS |
