/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_common_def.h
 * \brief QuantFlashAttn算子tiling和kernel共用的相关定义
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

static constexpr uint32_t FIA_SYNC_MODE2 = 2;
static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
static constexpr float FLOAT_ZERO = 0;
static constexpr float FLOAT_MAX = 3.402823466e+38F;
static constexpr float FLOAT_INF = 3e+99;

static constexpr uint32_t QFA_HEAD_NEED_INIT_OUTPUT_INDEX = 15U;

// V尾块(高精窗口)边界: 与 v_descale 的 64 粒度保持一致, kernel与量化cache写入侧共享的单一事实来源
static constexpr uint32_t MXFP8_V_TAIL_BOUNDARY = 64U;

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

enum class FA_LAYOUT : uint32_t {
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class inferFaLayOutTypeEnum {
    None = 0,
    LAYOUT_BSND = 1,
    LAYOUT_SBND = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6
};

enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned72 = 72,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

// q_out layout → (inputLayout, outputLayout)
//   0=BSND → (BSND, BSND)
//   1=BNSD → (BNSD, BNSD)
//   2=TND  → (TND, TND)
//   3=NTD  → (NTD, TND)
static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[4][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BSND, inferFaLayOutTypeEnum::LAYOUT_BSND},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_TND},
};

// q_out layout
//   0: BSND
//   1: BNSD
//   2: TND
//   3: NTD
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
#define InOutLayoutType_NTD 3
// backward compat
#define InOutLayoutType_BSND_BSND InOutLayoutType_BSND
#define InOutLayoutType_BNSD_BNSD InOutLayoutType_BNSD
#define InOutLayoutType_TND_TND InOutLayoutType_TND
#define InOutLayoutType_NTD_TND InOutLayoutType_NTD

#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBND 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

static constexpr ConfigParams ConfigValue[] = {
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned72,
     inferDTemplateType::Aligned72},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256},
    // Config 6：M64/N256/D256，专用于同页双 KV head 块对角合并。
    // kvTokenBlock = 256/2 = 128 与基线每 head 的 token 块一致，故 task 数减半，
    // V2/Pscale 的总行数与基线相同；M 轴展开全部 N_q 个 Q head。
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256},
    {inferS1TemplateType::Aligned32, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256},
};

#define Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64 0
#define Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128 1
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 2
#define Config_S1Aligned128_S2Aligned256_DAligned256_DVAligned256 3
#define Config_S1Aligned128_S2Aligned512_DAligned72_DVAligned72 4
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 5
#define Config_MergeKvHeads_S1Aligned64_S2Aligned256_DAligned256_DVAligned256 6
#define Config_S1Aligned32_S2Aligned128_DAligned256_DVAligned256 7

#define QFA_MXFP8_FP32_PREFILL 1
#define QFA_MXFP8_FP32_DECODE 2
#define QFA_GQA_FP8_FULLQUANT 6
#define QFA_HIF8_FP32 0
#define QFA_MXFP4_FP32_PREFILL 3
#define QFA_MXFP4_FP32_DECODE 4

#define false 0
#define true 1

struct FDparamsX {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct RunInfoX {
    uint32_t loop = 0; // 当前 section 内的有效 task 编号，用于流水调度和缓冲槽轮转
    uint32_t mloop = 0; // M 块状态编号；同一 M 块的多个 S2 task 共用，用于选择 Softmax max/sum 状态槽。
    bool isValid = false; // 当前任务描述符是否有效
    bool isChangeBatch = false; // batch 切换标记；当前 MXFP8 CalcParams() 中设置为 false
    bool isFirstS2Loop = false; // 是否为当前 Q 块／本核分片的第一个 S2 task，需要初始化累积状态
    bool isLastS2Loop = false; // 是否为当前 Q 块／本核分片的最后一个 S2 task，需要结束累积并输出结果

    uint32_t bIdx = 0; // batch 编号
    uint32_t n2Idx = 0; // 原始 KV head 编号
    uint32_t gS1Idx = 0; // 当前 M 块在合并 Q 行轴上的起始行位置，等于调度 M 块编号乘 mBaseSize
    uint32_t gIdx = 0; // G 维组内编号的预留字段
    uint32_t s1Idx = 0; // 当前块起点对应的 Q token 位置，由 gS1Idx 按布局换算
    uint32_t s2Idx = 0; // 当前 task 的 KV token 起始位置，通常为 S2 块编号乘 s2BaseSize
    uint32_t realN2Idx = 0; // DN/ND 映射后的调度 head 编号。DN 下可包含原始 G 维的区分
    uint64_t actS1Size = 1; // 当前 batch 的实际 Q 序列长度
    uint64_t actS2Size = 1; // 当前 batch 的实际 KV 序列长度
    uint32_t actMSize = 0; // 当前 M 块的有效 Q 行数；尾块可能小于 mBaseSize
    uint32_t actMSizeAlign32; // actMSize 向上对齐到 32
    uint32_t actVecMSize; // 当前 AIV 实际负责的 Q 行数
    uint32_t vecMbaseIdx; // 当前 AIV 在这个 M 块内的起始行偏移

    uint32_t actSingleLoopS2Size = 0; // 当前 task 实际处理的 KV token 数；尾块可能不足一个完整 N 块
    uint32_t actSingleLoopS2SizeAlign; // 上述 KV 长度按当前实现要求对齐后的值
    bool isS2SplitCore = false; // 当前 Q 块是否只在本核计算部分 S2，需要将部分结果交给 FD 合并
    uint32_t faTmpOutWsPos = 0; // 当前部分结果写入 FD workspace 的槽位编号

    int64_t preTokensLeftUp = 0; // 换算到左上对齐坐标系后的左窗口范围
    int64_t nextTokensLeftUp = 0; // 换算到左上对齐坐标系后的右窗口范围

    uint64_t qPaddingBeginOffset = 0; // Q 左侧 padding 偏移，供 mask 计算使用
    uint64_t kvPaddingBeginOffset = 0; // KV 左侧 padding 偏移，供 mask 计算使用
};

struct RunInfoVTailX : RunInfoX {
    uint32_t kvTailLen = 0;
    bool isVtailTask = false;
    bool isLastDnTask = false;
};

struct StridesConstInfo {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

struct CommonConstInfo {
    uint32_t bSize; // batch 数量
    uint64_t t1Size; // Q 的 T 轴尺寸，供 TND、NTD 等布局寻址使用
    uint64_t t2Size; //  KV 的 T 轴尺寸，供 TND、NTD 等布局寻址使用
    uint32_t dSize; // Q/K 的逻辑 head dimension
    uint32_t dSizeV; // V 的逻辑 head dimension
    uint32_t dBasicBlock; // dSizeV 向上对齐
    uint32_t gSize; // 原始 GQA 分组大小，通常为 Q head 数除以 KV head 数
    uint32_t n2Size; // 原始 KV head 数
    uint32_t realGSize; // 实际用于调度及行轴合并的 G，不合轴相当是1，不等于gSize
    uint32_t realN2Size; // 实际用于调度的 head 数，不合轴相当于是q的head数量，合轴就是kv head的数量
    uint64_t s1Size; // tiling 提供的 Q/KV 序列尺寸，供默认长度等逻辑使用；不是每个 batch 的实际有效长度。
    uint64_t s2Size;
    uint64_t cuSeqLensQSize; // Q 累计长度输入的规模信息，就是数组长度
    uint64_t cuSeqLensKVSize; // KV 累计长度输入的对应规模信息
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;

    uint32_t sparseMode; // attention 的稀疏／mask 模式
    uint32_t attenMaskBatch; // mask 的 batch 维尺寸
    uint32_t attenMaskS1Size; // mask 的 Q/KV 方向尺寸
    uint32_t attenMaskS2Size;
    int64_t preTokens; // 原始左／右窗口配置
    int64_t nextTokens;
    float scaleValue; // attention score 的缩放系数，例如 1 / sqrt(D)

    uint32_t aicIdx; // 当前 AIC 编号；在 AIV 上表示其对应的 AIC 编号
    uint32_t aivIdx; // 当前 AIV 编号
    uint8_t subBlockIdx; // 当前 AIV 在所属 AIC 配对中的子核编号，0或1
    uint32_t coreNum;  // FA 路径参与计算的 AIC 数量

    uint32_t accumOutSize; // FD 部分输出区域的容量
    uint32_t logSumExpSize; // FD 统计量区域的容量

    FA_LAYOUT outputLayout; // 输出张量布局
    bool needInitOutput; // 是否需要初始化输出的标记
    bool enableFlashDecode; // 运行时是否启用 FD 的标记

    // 四个平面的 stride
    StridesConstInfo keyStrides;
    StridesConstInfo valueStrides;
    StridesConstInfo kDescaleStrides;
    StridesConstInfo vDescaleStrides;
};

struct PAConstInfo {
    uint32_t blockSize; // page的大小
    uint32_t maxBlockNumPerBatch; // block table 每个 batch 的条目数，用于定位对应行
    uint32_t paLayoutType; // 分页 KV 的布局编码
};

struct LseConstInfo {
    bool isSoftmaxLseEnable; // 是否输出 Softmax LSE
};

struct SinkConstInfo {
    bool learnableSinkFlag = false; // learnable sink 功能标记
};

struct ConstInfo_t : CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo {};

struct ConstInfoVTail_t : ConstInfo_t {
    uint32_t seqUsedKvTailSize = 0;
    uint32_t tailMaxBlockNum = 0;
    StridesConstInfo vTailStrides;
};

#endif // QUANT_FLASH_ATTN_COMMON_DEF_H_
