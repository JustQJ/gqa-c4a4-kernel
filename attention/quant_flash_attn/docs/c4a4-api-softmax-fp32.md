# C4A4 FP32 Softmax API 接口说明

本文以 2026-11-12 的 `f8034b9db` 为历史基线，KV 布局按 PR2 合并提交 `0e86482fd` 更新。本文只描述 `quant_mode=5`，即 `A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP32`。内部 Fixpipe 搬运开关不改变公开 API。

## 1. 当前实现范围

目标平台为 Ascend 950 / DAV_3510（arch35）。Metadata、Host tiling、正式 kernel 和 Python/ACLNN 调用链已经接通。输入 Q/K/V 为 MXFP4 E2M1，descale 为 E8M0，Softmax 和累积状态为 FP32，输出为 BF16。

| 配置 | 当前可执行范围 |
|---|---|
| `quant_mode` | `5` |
| Q / 输出 layout | `TND` / `TND` |
| KV layout | `PA_BBND`、`PA_BNBD`、`PA_NZ` |
| Q descale layout | `TND`（DN 路径）或 `N2TGD`（ND 路径） |
| head dimension | `D = Dv = 128` 或 `256` |
| page size | `64 / 128 / 256 / 512 / 1024` |
| 头数 | 正整数 `N1`、`N2`，且 `N1 % N2 == 0` |
| mask | `0` 无 mask、`3` 右下对齐 causal |
| P 量化 | 内部默认 CX Delta，`x=7.25`、`delta=1.5`；delta 同时参与指数选择，API 参数不变 |
| FD | 由 Metadata 分片和 Host 决定，不提供公开 `numSplits` |
| 反向、sinks、V tail、滑窗 | 当前 mode 5 不支持 |

三种 KV 布局均支持 DN 和 ND 路径。DN/ND 由 **Qscale 排布**决定，不能单凭 Q 长度或 prefill/decode 名称选择。

## 2. 维度、存储与量化布局

| 符号 | 含义 |
|---|---|
| `B` | batch 序列数 |
| `T` | Q 存储总 token 数，通常为 `cu_seqlens_q[-1]` |
| `N1 / N2 / G` | Q 头数 / KV 头数 / `N1 // N2` |
| `D / Dv` | QK / V 的逻辑维度，当前二者相等 |
| `Bn / Bs` | KV 物理页数 / 每页 token 数 |
| `Pmax` | 每个序列页表容量 |
| `C` | Metadata 每行容量，单位为 int32 元素 |

Python 的 `torch.float4_e2m1fn_x2` 一个存储元素包含两个 FP4 数值。桥接层 `TensorWrapper → ConvertType → CollectB4ShapeInfo → aclCreateTensor` 将描述符变为 `ACL_FLOAT4_E2M1` 的逻辑元素口径；连续末轴乘 2，外轴 stride 和 storage offset 也换算为半字节元素单位。此过程不量化、不解包、不复制 payload。

Q 的 Python 打包 shape 为 `[T,N1,D/2]`，ACLNN 逻辑 shape 为 `[T,N1,D]`。K/V 的 shape 由 `layout_kv` 决定。

| `layout_kv` | K 的 Python 打包 shape | V 的 Python 打包 shape | K 的 ACLNN 逻辑 shape | V 的 ACLNN 逻辑 shape |
|---|---|---|---|---|
| `PA_BBND` | `[Bn,Bs,N2,D/2]` | `[Bn,Bs,N2,Dv/2]` | `[Bn,Bs,N2,D]` | `[Bn,Bs,N2,Dv]` |
| `PA_BNBD` | `[Bn,N2,Bs,D/2]` | `[Bn,N2,Bs,Dv/2]` | `[Bn,N2,Bs,D]` | `[Bn,N2,Bs,Dv]` |
| `PA_NZ` | `[Bn,N2,D/64,Bs,32]` | `[Bn,N2,Dv/64,Bs,32]` | `[Bn,N2,D/64,Bs,64]` | `[Bn,N2,Dv/64,Bs,64]` |

`PA_NZ` 将 K/V 的 head dimension 切成每块 64 个逻辑 FP4 值，对应 Python 末轴的 32 个打包字节。以 D128 的 K 为例，`PA_BBND` 和 `PA_BNBD` 的 Python 末轴为 64，`PA_NZ` 在页内 token 轴前放置 2 个分块，末轴为 32。Host 的逻辑 `dSize` 均为 128。E8M0 scale 每元素一字节，Python 与 ACLNN 使用相同的 shape。

Q、K 沿 D 每 32 个值使用一个 scale。V 沿页内 token 每 32 个值使用一个 scale。

```text
Qscale DN: [T,N1,D/64,2]
Qscale ND: [N2,T,G,D/64,2]
```

| `layout_kv` | `k_descale` shape | `v_descale` shape |
|---|---|---|
| `PA_BBND` | `[Bn,Bs,N2,D/64,2]` | `[Bn,Bs/64,N2,Dv,2]` |
| `PA_BNBD` | `[Bn,N2,Bs,D/64,2]` | `[Bn,N2,Bs/64,Dv,2]` |
| `PA_NZ` | `[Bn,N2,Bs/16,D/64,16,2]` | `[Bn,N2,Dv/16,Bs/64,16,2]` |

设 `p` 为物理页编号，`u` 为页内 token 下标，`h` 为 KV 头下标，`d` 为该头内的通道下标。对应的 scale 下标如下。

```text
PA_BBND Kscale[p,u,h,d//64,(d%64)//32]
PA_BBND Vscale[p,u//64,h,d,(u%64)//32]
PA_BNBD Kscale[p,h,u,d//64,(d%64)//32]
PA_BNBD Vscale[p,h,u//64,d,(u%64)//32]
PA_NZ   Kscale[p,h,u//16,d//64,u%16,(d%64)//32]
PA_NZ   Vscale[p,h,d//16,u//64,d%16,(u%64)//32]
```

两种 Qscale 排布的等价转换如下；Q payload 保持 TND：

```python
q_descale_nd = (
    q_descale_dn.view(torch.uint8)
    .reshape(T, N2, G, D // 64, 2)
    .permute(1, 0, 2, 3, 4).contiguous()
    .view(torch.float8_e8m0fnu)
)
```

`view(dtype)` 只解释位模式，不是量化。Q/K/V 必须先完成正确的 FP4 编码；当前 Python 入口拒绝直接传 `uint8` Q/K/V。scale 推荐使用 `float8_e8m0fnu`；C++ 桥接也能把 `uint8` scale 位模式解释为 E8M0，但不会把普通整数转换成对应浮点 scale。

## 3. `quant_flash_attn_metadata`

### 3.1 Python 签名

下面为 `torch.ops` 的实际参数顺序和默认值。`*` 后为 keyword-only 参数。

```python
def quant_flash_attn_metadata(
    num_heads_q, num_heads_kv, head_dim, quant_mode, *,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    batch_size=None, max_seqlen_q=-1, max_seqlen_kv=-1,
    head_dim_v=None, mask_mode=0, win_left=-1, win_right=-1,
    layout_q="BSND", layout_q_descale="BSND",
    layout_kv="BSND", layout_out="BSND", is_grad_enabled=False,
) -> torch.Tensor:
    ...
```

### 3.2 输入

长度 Tensor 为同一 NPU 上的连续 `int32` Tensor。Metadata 不接收 Q/K/V、scale 或页表。

| 参数 | shape / 类型 | mode 5 约定 |
|---|---|---|
| `cu_seqlens_q` | `[B+1]` | 必须；从 0 开始非递减，表示 Q 存储段边界 |
| `cu_seqlens_kv` | — | `None`；PA 不传累计 KV 长度 |
| `seqused_q` | `[B]` | 可选；有效 Q 长度，不超过对应存储段；未传时使用累计边界之差 |
| `seqused_kv` | `[B]` | 必须；有效 KV token 数，非累计值、非页数 |
| `num_heads_q / num_heads_kv` | `int` | `N1 / N2` |
| `head_dim / head_dim_v` | `int / Optional[int]` | 逻辑 D；`head_dim_v=None` 时取 D |
| `quant_mode` | `int` | `5` |
| `batch_size` | `Optional[int]` | Python TND 调用必须 `None`，包装从长度 Tensor 推导 B |
| `max_seqlen_q / max_seqlen_kv` | `int` | 最大长度提示；`-1` 表示未提供，建议显式传实际最大长度 |
| `mask_mode` | `int` | 与 QFA 一致，`0` 或 `3` |
| `win_left / win_right` | `int` | 均为 `-1` |
| `layout_q / layout_out` | `str` | 均显式传 `"TND"` |
| `layout_q_descale` | `str` | `"TND"` 或 `"N2TGD"`，与实际 Qscale 一致 |
| `layout_kv` | `str` | 显式传 `"PA_BBND"`、`"PA_BNBD"` 或 `"PA_NZ"`，与主算子一致 |
| `is_grad_enabled` | `bool` | `False` |

Metadata 的公共检查接受的配置可能比 C4 kernel 广；生成成功不能替代 QFA 的设备支持检查。有效长度的内容和实际存储容量由调用者保持一致。

### 3.3 输出与复用

返回同设备 `int32` Tensor，shape 为 `[2,C]`。第 0 行是正向 header、FA/FD 调度数据，第 1 行为反向预留，mode 5 不使用。

```text
C = align_up(16 + 16 * (AIC核数 + AIV核数) * B * N1, 4096)
总字节 = 2 * C * 4
```

4096 的单位为元素。包装从当前设备查询核数。直接 ACLNN 调用可提供 `[C]` 正向空间或 `[2,C]`；所需正向容量由 section 数决定，AICPU 检查容量。未使用区域不保证为零。

Metadata 是调度描述，不能充当 workspace。改变设备、有效核数、长度、头数、D、layout 或 mask 配置后须重新生成；传给 QFA 的配置必须与生成时匹配。

## 4. `quant_flash_attn`

### 4.1 Python 签名

```python
attn_out, softmax_lse = torch.ops.cann_ops_transformer.quant_flash_attn(
    q, k, v, q_descale, k_descale, v_descale, quant_mode,
    block_table=None, p_scale=None,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    sinks=None, attn_mask=None, metadata=None,
    v_tail=None, block_table_tail=None, seqused_v_tail=None,
    softmax_scale=1.0, mask_mode=0, win_left=-1, win_right=-1,
    max_seqlen_q=-1, max_seqlen_kv=-1,
    layout_q="BSND", layout_q_descale="BSND",
    layout_kv="BSND", layout_out="BSND", return_softmax_lse=False, quant_schema=2,
)
```

`quant_schema` 是尾部可选整数参数，默认2，只选择 C4 的 P 量化方案：

| quant_schema | P 量化方案 | 模板选择 |
|---|---|---|
| 0 | 普通 MXFP4 / OCP | QUANT_SCHEMA=QuantSchema::OCP_QUANT |
| 1 | CX（cx_legacy），不乘1.5 | QUANT_SCHEMA=QuantSchema::CX_QUANT |
| 2（默认） | CX Delta（cx_delta），1.5纳入指数选择 | QUANT_SCHEMA=QuantSchema::CX_DELTA_QUANT |

可直接以 `quant_schema=0/1/2` 调用同一个安装包。省略参数等价于2，其他参数的位置和默认值不变。非法取值会报错；非 `quant_mode=5` 场景仅接受默认2，不改变这些场景原有算法。Metadata 接口不增加此参数，三种方案共享同一调度。

C 接口使用 `aclnnQuantFlashAttnGetWorkspaceSize` / `aclnnQuantFlashAttn`，第一阶段在 `returnSoftmaxLse` 后、输出 Tensor 前显式传入 `int64_t quantSchema`。Python 的 `quant_schema=2` 默认值保持。C++扩展与算子库须配套重新编译，直接C调用须按新签名传参；不保留旧二进制接口或V2兼容入口。

### 4.2 输入 Tensor

所有有效 Tensor 位于同一 NPU，ACL format 使用 `ND`。`PA_NZ` 通过 `layout_kv` 和第 2 节的物理排布指定。

| 参数 | Python dtype | Python shape | 用途 / 要求 |
|---|---|---|---|
| `q` | `float4_e2m1fn_x2` | `[T,N1,D/2]` | 已量化 Q |
| `k` | `float4_e2m1fn_x2` | 见第 2 节 | 按 `layout_kv` 排布的分页 K |
| `v` | `float4_e2m1fn_x2` | 见第 2 节 | 按 `layout_kv` 排布的分页 V |
| `q_descale`（DN） | `float8_e8m0fnu` | `[T,N1,D/64,2]` | 与 `layout_q_descale="TND"` 配套 |
| `q_descale`（ND） | `float8_e8m0fnu` | `[N2,T,G,D/64,2]` | 与 `layout_q_descale="N2TGD"` 配套；与上一项二选一 |
| `k_descale` | `float8_e8m0fnu` | 见第 2 节 | 按 `layout_kv` 排布，每 32 个通道共用一个 scale |
| `v_descale` | `float8_e8m0fnu` | 见第 2 节 | 按 `layout_kv` 排布，每 32 个页内 token 共用一个 scale |
| `block_table` | `int32` | `[B,Pmax]` | 必须；逻辑页到物理页编号 |
| `cu_seqlens_q` | `int32` | `[B+1]` | 必须；与 Metadata 相同 |
| `seqused_q` | `int32` | `[B]` | 可选；与 Metadata 相同 |
| `seqused_kv` | `int32` | `[B]` | 必须；与 Metadata 相同 |
| `metadata` | `int32` | `[2,C]`，或正向 `[C]` | 必须；签名默认 `None` 不代表 C4 可省略 |
| `attn_mask` | `int8` | `[2048,2048]` | mode 0 传 `None`；mode 3 按公共 Checker 必须提供 causal 模板 |
| `p_scale` | — | — | C4 调用传 `None`；公共 Checker 可接受 FP32 `[1]`，但当前 C4 kernel 不用它控制 P 量化 |
| `cu_seqlens_kv` | — | — | `None` |
| `sinks` | — | — | `None`，不支持 |
| `v_tail / block_table_tail / seqused_v_tail` | — | — | 均 `None`，不支持 |

`block_table[b,j]` 应位于 `[0,Bn)`，有效页槽不少于 `ceil(seqused_kv[b]/Bs)`；K/V/Kscale/Vscale 的同一物理页编号必须对应同一页。末页多余 token 由有效长度屏蔽。

Causal 模板取上三角为 1、其余为 0。C4 VF 实际根据 Q/KV 坐标计算右下对齐 mask：序列内 Q 行 `q` 可以访问 `k <= q + seq_kv - seq_q`。任意自定义 mask 内容不是当前 C4 功能；2048 模板大小不表示有效序列长度只能为 2048。

### 4.3 标量、连续性和输出

`quant_mode=5`；`softmax_scale` 为正数，API 默认 **1.0**，不会自动取 `1/sqrt(D)`。常用 attention 缩放必须显式传 `D ** -0.5`。QFA 没有独立的 B、头数、D 参数，这些值从输入解析。

`mask_mode`、`win_left/right`、`max_seqlen_q/kv` 和四个 layout 参数应与 Metadata 一致。公共默认 `BSND` 不适用于 C4。

建议使用连续的 K/V/Kscale/Vscale。Q/Qscale 等由 OpDef 标记为 AutoContiguous，K/V/Kscale/Vscale 标记为 IgnoreContiguous。`PA_BBND` 只允许物理页轴（轴 0）非连续，`PA_BNBD` 和 `PA_NZ` 只允许物理页轴、KV 头轴（轴 0、1）非连续。其余轴须连续。

传入非连续 KV 时，K 须作为 view 传入，以便 Host 读取这四个 Tensor 的 stride。Kscale/Vscale 的非连续页轴和头轴 stride 须为偶数，单位为 E8M0 元素。Python 拒绝这四个 Tensor 的零 stride。

| 输出 | dtype | shape | 含义 |
|---|---|---|---|
| `attn_out` | `bfloat16` | `[T,N1,Dv]` | 已恢复逻辑 Dv 的 attention 输出 |
| `softmax_lse`，开关为 True | `float32` | `[N1,T]` | 以标准 `logsumexp(s)` 为目标、已去除 CX 公共倍率的量化近似 |
| `softmax_lse`，开关为 False | `float32` | `[0]` | 仍返回空 Tensor，不是 `None` |

计算使用量化后的 P 和增强分母，不等同于对原始未量化 Q/K/V 的精确 Softmax。默认 CX Delta 使用 `k_g=ceil(max(s_g)/ln(2)-log2(x/delta))`，`c=ceil(log2(x/delta))`。对于非空行，令 Z 为实际量化 P 的增强分母，R 为合并后的整数参考，输出 `LSE=ln(Z)+(R+c)*ln(2)-ln(delta)`；固定 x=7.25、delta=1.5 时 c=3，即 `ln(Z)+(R+3)*ln(2)-ln(1.5)`。CX分支（quant_schema=1）采用 delta=1，使用 `ceil(log2(x))=3`，P量化前不乘1.5，LSE为 `ln(Z)+(R+3)*ln(2)`。忽略量化误差时恢复 `logsumexp(s)`，不保留量化配方的固定倍率。P 的量化链为 FP32 Exp → BF16 寄存器 → FP4；score、归约、指数和累积状态保持 FP32。空有效行输出零，启用 LSE 时当前实现用 `+inf` 表示空行。普通MXFP4/OCP（quant_schema=0）使用 `LSE=ln(Z)+R*ln(2)`。三种方案由公开 `quant_schema` 选择，公式详见 kernel 文档第10节。

## 5. Python 调用示例

先安装与源码匹配的 OPP/Python 包并导入扩展。以下函数接收已量化并按 `layout_kv` 排布的 K/V 及其 scale。将现有 `PA_BBND` 缓存转为其他布局时，需同时重排 K/V 和两个 scale。可参考[测试中的布局转换](../tests/pytest/qfa_mxfp4_test/common/quant_flash_attn_golden.py)。

```python
import torch
import torch_npu
import cann_ops_transformer

def c4a4_attention(q, k, v, q_descale, k_descale, v_descale,
                   block_table, cu_q, used_kv, max_q, max_kv,
                   qscale_layout="N2TGD", causal=False, quant_schema=2,
                   layout_kv="PA_BBND"):
    kv_head_axis = {"PA_BBND": 2, "PA_BNBD": 1, "PA_NZ": 1}[layout_kv]
    n1, n2, d = q.shape[1], k.shape[kv_head_axis], q.shape[-1] * 2
    mask_mode = 3 if causal else 0
    layouts = dict(layout_q="TND", layout_q_descale=qscale_layout,
                   layout_kv=layout_kv, layout_out="TND")
    metadata = torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
        n1, n2, d, 5, cu_seqlens_q=cu_q, seqused_kv=used_kv,
        max_seqlen_q=max_q, max_seqlen_kv=max_kv,
        mask_mode=mask_mode, **layouts,
    )
    mask = None
    if causal:
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(q.device)
    return torch.ops.cann_ops_transformer.quant_flash_attn(
        q, k, v, q_descale, k_descale, v_descale, 5,
        block_table=block_table, cu_seqlens_q=cu_q, seqused_kv=used_kv,
        metadata=metadata, attn_mask=mask, mask_mode=mask_mode,
        max_seqlen_q=max_q, max_seqlen_kv=max_kv,
        softmax_scale=d ** -0.5, return_softmax_lse=True,
        quant_schema=quant_schema, **layouts,
    )
```

`torch.ops` schema 的 mode 参数为整数；Python 包装函数中枚举/字符串解析能力不改变 dispatcher schema。本例使用整数避免歧义。

## 6. ACLNN 与实现依据

直接 ACLNN 调用分别使用 `aclnnQuantFlashAttnMetadataGetWorkspaceSize → aclnnQuantFlashAttnMetadata` 和 `aclnnQuantFlashAttnGetWorkspaceSize → aclnnQuantFlashAttn` 两阶段；主算子第一阶段显式传入 `quantSchema`，使用默认方案时传2。第二阶段传入第一阶段返回的 executor、所需 workspace 和执行 stream。调用者自行创建输出 Tensor，按照第 2 节的逻辑 FP4 shape/stride 创建输入描述符，并保证 Tensor/内存存活到执行完成。Metadata 的 `batchSize` 是整数，直接调用时可显式填 B；Python 的 `None` 是包装层约定。

| 依据 | 内容 |
|---|---|
| [Python 接口](../torch_extension/quant_flash_attn.py) | schema、默认值、dtype 检查、Meta、Metadata 容量 |
| [C++ 桥接](../torch_extension/csrc/quant_flash_attn.cpp) | FP4 wrapper、ACLNN 参数顺序、输出分配 |
| [公共 ACL 转换](../../../torch_extension/cann_ops_transformer/common/aclnn_common.h) | `CollectB4ShapeInfo`、shape/stride/offset 换算 |
| [Host checker](../op_host/checkers/quant_checker.cpp) | 量化 dtype、shape、layout、D128/D256 |
| [Mask checker](../op_host/checkers/mask_checker.cpp) | 模板 mask 和滑窗限制 |
| [Metadata checker](../../quant_flash_attn_metadata/op_host/quant_flash_attn_metadata_check.h) | mode 5 的 KV 布局和长度参数约束 |
| [C4 tiling](../op_host/arch35/quant_flash_attn_tiling_mxfp4.cpp) | 三种分页 KV 布局的模板配置 |
| [Kernel 实现说明](c4a4-kernel-imp-softmax-fp32.md) | 流水、存储、同步、数值精度 |

## 7. 验证边界与修订

历史基线 `f8034b9db` 有 436 次生产入口直调及 615 次公开 API 检查通过，随后测试曾暂停；这些历史计数不作为当前代码的验收结果。此前包含 CX Pscale、标准 LSE 补偿、默认双目标和 FD 残差补偿的版本在 200 机器完成小量公开 API 双参考验证（236/236 通过）。用户随后要求删除 FD 残差补偿，当前已恢复直接 `a_i/S` 乘加；该通过结论不适用于当前代码。直接乘加版本在初轮已有少量 BF16 输出超出原门限的记录，门限未放宽。历史范围与结果见[本轮精度验证报告](../../../operators/quant_flash_attn/cx_pscale_lse_small_20261111/README.md)。本轮使用显式 `1/sqrt(D)`；旧版额外大 scale 的 13/112 项失败未纳入本轮复验，不能声明已经修复。没有本轮性能结论。

2026-11-19 在 Ascend 950 上以 PR2 合并版本完成 `debug` 与 `functional document` 精度矩阵。三种 `quant_schema` 和三种 KV 布局的组合各有 5,076 个用例，共 45,684 项全部通过。`PA_BNBD` 和 `PA_NZ` 的 Attention/LSE 在同源输入下各有 15,228 组与 `PA_BBND` 逐位一致。

| 日期 | 修订者 | 内容 |
|---|---|---|
| 2026-11-11 | Codex | 按当前代码重写原 API 文档；收敛到 PA_BBND，明确已实现入口、打包 shape、保留参数和验证边界 |
| 2026-11-11 | Codex | CX Pscale 恢复默认偏置 124，FA/FD LSE 补偿 `+3ln(2)-ln(1.5)`，以标准 logsumexp 为忽略量化误差后的目标 |
| 2026-11-19 | Codex | 补充 `PA_BNBD` 和 `PA_NZ` 支持，更新 K/V 与 scale 的 shape、连续性要求和调用示例 |
