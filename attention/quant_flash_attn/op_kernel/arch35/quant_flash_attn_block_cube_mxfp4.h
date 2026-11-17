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
 * \file quant_flash_attn_block_cube_mxfp4.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP4_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP4_H_
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_mxfp4_common.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
        S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
        S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
        DTemplateType dTemplateType = DTemplateType::Aligned128,
        DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
        bool isDAligned = true, bool enableQf322F32Pre = false>
class QuantFlashAttnBlockCubeMxfp4 {
public:
    // 基本参数
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static_assert(dBaseSize == dVBaseSize, "MXFP4 requires Dv == D");
    static_assert(mBaseSize == 128 && (dBaseSize == 128 || dBaseSize == 256),
                  "MXFP4 supports M128 and D128/D256 only");
    static_assert(s2BaseSize == (dBaseSize == 128 ? 256 : 128), "Unsupported MXFP4 task size");
    static_assert(KvLayoutType == 1 && layout == LayOutTypeEnum::LAYOUT_TND,
                  "MXFP4 first version requires Q TND and KV PA_BBND");
    static_assert(isDAligned, "MXFP4 unaligned D is not implemented");
    static_assert(std::is_same<T, float>::value && std::is_same<INPUT_T, fp4x2_e2m1_t>::value,
                  "MXFP4 requires FP4 E2M1 input and FP32 accumulation");
    static constexpr uint32_t s2SplitSize = 128U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;
    static constexpr bool ENABLE_QF322F32_PRE = enableQf322F32Pre;
    static constexpr FixpipeConfig FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = T;

    using CrossBufferType = CVCrossBuffer<20, useDn, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType>;
    static constexpr uint32_t QK_L0_SLOT_COUNT = dBaseSize == 256 ? 1U : 2U;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_PARSER_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_PARSER_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<(!PAGE_ATTENTION && IS_TND_LAYOUT),
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;
    TPipe* tPipe_ = nullptr;

    /* =====================GM变量==================== */
    // 按搬运单位绑定：payload 为 packed FP4 字节，scale 为一对 E8M0；不做数值转换。
    GlobalTensor<uint8_t> queryGm_;
    GlobalTensor<uint8_t> keyGm_;
    GlobalTensor<uint8_t> valueGm_;
    GlobalTensor<uint16_t> queryScaleGm_;
    GlobalTensor<uint16_t> keyScaleGm_;
    GlobalTensor<uint16_t> valueScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqParserType* qSeqParserPtr_ = nullptr;
    KvSeqParserType* kvSeqParserPtr_ = nullptr;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::UB> ubBufferManager_;
    BufferManager<BufferType::L1> l1BufferManager_;
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;


    // L1 buffers
    CrossBufferType crossBuffers_;
    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers_;
    BuffersPolicy4buff<BufferType::L1> l1KVBuffers_; // FP4 K/V payload + E8M0 scale
    BuffersPolicySingleBuffer<BufferType::L1> l1VPaddingBuffers_; // rowsum需要的常数1

    // L0 buffers
    std::conditional_t<QK_L0_SLOT_COUNT == 1, BuffersPolicySingleBuffer<BufferType::L0A>,
        BuffersPolicyDB<BufferType::L0A>> mm1LeftBuffers_; // 在DN下，装K；在ND下，装Q
    BuffersPolicyDB<BufferType::L0A> mm2LeftBuffers_; // 在DN下，装V；在ND下，装P
    // 每次矩乘的左右槽成对轮转，共用左槽的 MTE1<->M 事件。
    // TPipe 在 C310 已保留 M_MTE1 的 0..2；四个成对槽只再申请 4 个事件。
    std::conditional_t<QK_L0_SLOT_COUNT == 1, BuffersPolicySingleBuffer<BufferType::L0B, SyncType::NO_SYNC>,
        BuffersPolicyDB<BufferType::L0B, SyncType::NO_SYNC>> mm1RightBuffers_; // DN:Q，ND:K
    BuffersPolicyDB<BufferType::L0B, SyncType::NO_SYNC> mm2RightBuffers_; // DN:P，ND:V
    std::conditional_t<QK_L0_SLOT_COUNT == 1, BuffersPolicySingleBuffer<BufferType::L0C>,
        BuffersPolicyDB<BufferType::L0C>> mm1ResBuffers_; // 两槽分别用 FIX_M/M_FIX 保护，允许 M 与 FIX 重叠。
    BuffersPolicySingleBuffer<BufferType::L0C> mm2ResBuffers_; // pv的结果使用单槽累积


    static constexpr uint32_t P_SLOT_COUNT = 20;
    uint64_t scoreSequence_ = 0;
    uint32_t pvGroupTasks_ = 0;
    uint64_t pvGroupSequence_ = 0; // Matches Vector nextGroupId_ across sections.
    bool qCached_ = false;
    uint32_t queryL0ReadyCount_ = 0; // Initial QK issues populate the dimension-specific dedicated L0 slots.
    const ConstInfo_t& constInfo_;

    /*=================================function============================================= */
    __aicore__ inline QuantFlashAttnBlockCubeMxfp4(ConstInfo_t& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe* pipe, __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                        __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                    __gm__ uint8_t* dequantScaleKey,
                                        __gm__ uint8_t* dequantScaleValue, QSeqParserType& qParser,
                                    KvSeqParserType& kvParser)
    {
        tPipe_ = pipe;
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        queryGm_.SetGlobalBuffer((__gm__ uint8_t*)query);
        keyGm_.SetGlobalBuffer((__gm__ uint8_t*)key);
        valueGm_.SetGlobalBuffer((__gm__ uint8_t*)value);
        queryScaleGm_.SetGlobalBuffer((__gm__ uint16_t*)dequantScaleQuery);
        keyScaleGm_.SetGlobalBuffer((__gm__ uint16_t*)dequantScaleKey);
        valueScaleGm_.SetGlobalBuffer((__gm__ uint16_t*)dequantScaleValue);
    }

    // process循环之前，进行buffer初始化
    __aicore__ inline void InitBuffers()
    {
        // L1 上和 UB 上跨核同步的buffer
        l1BufferManager_.Init(tPipe_, 512*1024); // L1 大小为 512KB;
        ubBufferManager_.Init(tPipe_, CrossBufferType::ubBmm1BufferBytes_ * CrossBufferType::SCORE_SLOT_COUNT + CrossBufferType::ubBmm2BufferBytes_);
        crossBuffers_.InitBuffers(l1BufferManager_, ubBufferManager_);

        // L1 上的其他buffer
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize / 2;
        constexpr uint32_t mm1QScaleSize = mBaseSize * dBaseSize / MXFP_GROUP_SIZE;
        constexpr uint32_t mmKVSize = dBaseSize * s2BaseSize / 2;
        constexpr uint32_t mmKVScaleSize = s2BaseSize * dBaseSize / MXFP_GROUP_SIZE;
        constexpr uint32_t mmVPaddngSize = s2BaseSize * (dBaseSize + 16) / 2;
        constexpr uint32_t mmVPaddngScaleSize = s2BaseSize * (dBaseSize + 16) / MXFP_GROUP_SIZE;
        static_assert(P_SLOT_COUNT * (CrossBufferType::l1PBufferBytes_ + CrossBufferType::l1PScaleBufferBytes_) +
                      mm1QSize + mm1QScaleSize + 4 * (mmKVSize + mmKVScaleSize) +
                      mmVPaddngSize + mmVPaddngScaleSize <= 512 * 1024, "MXFP4 L1 capacity exceeded");
        l1QBuffers_.Init(l1BufferManager_, mm1QSize + mm1QScaleSize);
        l1KVBuffers_.Init(l1BufferManager_, mmKVSize + mmKVScaleSize);
        l1VPaddingBuffers_.Init(l1BufferManager_, mmVPaddngSize + mmVPaddngScaleSize);

        // L0A B C 当前写死
        l0aBufferManager_.Init(tPipe_, 64 * 1024);  // 64KB
        l0bBufferManager_.Init(tPipe_, 64 * 1024);  // 64KB
        l0cBufferManager_.Init(tPipe_, 256 * 1024); // 256KB

        // L0上的buffer
        constexpr uint32_t mm1leftSize = USE_DN? s2SplitSize * dBaseSize / 2 : mBaseSize * dBaseSize / 2;
        constexpr uint32_t mm1RightSize = USE_DN? mBaseSize * dBaseSize / 2 : s2SplitSize * dBaseSize / 2;
        constexpr uint32_t mm2LeftSize = USE_DN? s2BaseSize * (dBaseSize + 16) / 2 : mBaseSize * s2BaseSize / 2;
        constexpr uint32_t mm2RightSize = USE_DN? mBaseSize * s2BaseSize / 2 : s2BaseSize * (dBaseSize + 16) / 2;
        constexpr uint32_t mm1ResSize = mBaseSize * s2SplitSize * sizeof(float);
        constexpr uint32_t mm2ResSize = mBaseSize * (dBaseSize + 16) * sizeof(float);
        static_assert(QK_L0_SLOT_COUNT * mm1leftSize + 2 * mm2LeftSize <= 64 * 1024, "MXFP4 L0A capacity exceeded");
        static_assert(QK_L0_SLOT_COUNT * mm1RightSize + 2 * mm2RightSize <= 64 * 1024, "MXFP4 L0B capacity exceeded");
        static_assert(QK_L0_SLOT_COUNT * mm1ResSize + mm2ResSize <= 256 * 1024, "MXFP4 L0C capacity exceeded");
        mm1LeftBuffers_.Init(l0aBufferManager_, mm1leftSize);
        mm1RightBuffers_.Init(l0bBufferManager_, mm1RightSize);
        mm2LeftBuffers_.Init(l0aBufferManager_, mm2LeftSize);
        mm2RightBuffers_.Init(l0bBufferManager_, mm2RightSize);
        mm1ResBuffers_.Init(l0cBufferManager_, mm1ResSize);
        mm2ResBuffers_.Init(l0cBufferManager_, mm2ResSize);

        InitPaddingL1();
        l1VPaddingBuffers_.Get().template Wait<HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void InitPaddingL1()
    {
        constexpr uint32_t payloadBytes =
            s2BaseSize * (dBaseSize + 16) / 2;
        constexpr uint32_t scaleBytes =
            s2BaseSize * (dBaseSize + 16) / MXFP_GROUP_SIZE;

        Buffer<BufferType::L1>& valuePaddingL1Buffer = l1VPaddingBuffers_.Get();
        LocalTensor<uint8_t> valuePaddingL1Bytes = valuePaddingL1Buffer.GetTensor<uint8_t>();
        LocalTensor<uint16_t> valuePaddingPayload = valuePaddingL1Bytes.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> valuePaddingScales = valuePaddingL1Bytes[payloadBytes].ReinterpretCast<uint16_t>();
        // HardEvent::A_B 表示A流水通知B流水
        // 这里是等待MTE1消费完，MET2进行填入
        valuePaddingL1Buffer.Wait<HardEvent::MTE1_MTE2>();
        // 0x6 的二进制编码在fp4下表示4，0x7D 在E8M0下表示 1/4 = 2^(125 - 127)
        InitConstValueParams<uint16_t> payloadFillParams(
            1, static_cast<uint16_t>(payloadBytes / 32), 0, 0x6666);
        Fill(valuePaddingPayload, payloadFillParams); // 填充的api接口， payloadBytes / 32 表示datablock的数量，一个datablock是32B
        InitConstValueParams<uint16_t> scaleFillParams(
            1, static_cast<uint16_t>(scaleBytes / 32), 0, 0x7D7D);
        Fill(valuePaddingScales, scaleFillParams);
        // 表示 MTE2 生产完，等待 MTE1 消费
        valuePaddingL1Buffer.Set<HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void InitPaddingL0(const LocalTensor<KV_T>& valueL0Tensor, uint32_t alignedKvTokenCount)
    {
        LoadData2DParamsV2 payloadLoadParams {};
        payloadLoadParams.mStep = (dVBaseSize + 16) / 16;
        payloadLoadParams.kStep = alignedKvTokenCount / 64;
        payloadLoadParams.srcStride = (dVBaseSize + 16) / 16;
        payloadLoadParams.dstStride = (dVBaseSize + 16) / 16;
        payloadLoadParams.ifTranspose = false;
        LoadData2DMxParams scaleLoadParams {};
        scaleLoadParams.xStep = (dVBaseSize + 16) / 16;
        scaleLoadParams.yStep = alignedKvTokenCount / 64;
        scaleLoadParams.srcStride = s2BaseSize / 64;
        scaleLoadParams.dstStride = alignedKvTokenCount / 64;
        LocalTensor<uint8_t> valuePaddingL1Bytes = l1VPaddingBuffers_.Get().template GetTensor<uint8_t>();
        LoadData(valueL0Tensor, valuePaddingL1Bytes.template ReinterpretCast<KV_T>(),
                 valuePaddingL1Bytes[s2BaseSize * (dVBaseSize + 16) / 2].template ReinterpretCast<SCALE_T>(),
                     payloadLoadParams, scaleLoadParams);
        PipeBarrier<PIPE_MTE1>();
    }

    __aicore__ inline void ComputeBmm1(Mxfp4RunInfo& runInfo)
    {
        if (runInfo.actMSize == 0 || runInfo.actSingleLoopS2Size == 0) {
            return;
        }
        Buffer<BufferType::L1>& queryL1Buffer = l1QBuffers_.Get();
        if (!qCached_) {
            queryL0ReadyCount_ = 0;
            queryL1Buffer.template Wait<HardEvent::MTE1_MTE2>();
            if constexpr (USE_DN) {
                CopyQGm2L1Dn(queryL1Buffer.template GetTensor<Q_T>(), runInfo);
            } else {
                CopyQGm2L1Nd(queryL1Buffer.template GetTensor<Q_T>(), runInfo);
            }
            CopyQScaleGm2L1(queryL1Buffer.template GetTensor<uint8_t>(mBaseSize * dBaseSize / 2)
                               .template ReinterpretCast<SCALE_T>(), runInfo);
            queryL1Buffer.template Set<HardEvent::MTE2_MTE1>();
            queryL1Buffer.template Wait<HardEvent::MTE2_MTE1>();
            qCached_ = true;
        }
        Buffer<BufferType::L1>& keyL1Buffer = l1KVBuffers_.Get();
        keyL1Buffer.template Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> keyL1Tensor = keyL1Buffer.template GetTensor<KV_T>();
        LocalTensor<SCALE_T> keyScaleL1Tensor = keyL1Buffer.template GetTensor<uint8_t>(s2BaseSize * dBaseSize / 2)
                          .template ReinterpretCast<SCALE_T>();
        CopyKGm2L1(keyL1Tensor, runInfo);
        CopyKScaleGm2L1(keyScaleL1Tensor, runInfo);
        keyL1Buffer.template Set<HardEvent::MTE2_MTE1>();
        keyL1Buffer.template Wait<HardEvent::MTE2_MTE1>();
        for (uint32_t subBlockS2Offset = 0; subBlockS2Offset < runInfo.actSingleLoopS2Size;
            subBlockS2Offset += s2SplitSize) {
            const uint32_t validKvTokenCount = AttentionCommon::Min(s2SplitSize,
                runInfo.actSingleLoopS2Size - subBlockS2Offset);
            // 三组 Get 同步轮转；只等待本槽的 FIX 完成，不排空另一槽。
            // 因而下一槽的 Mmad 可与本槽的 L0C -> UB Fixpipe 重叠。
            Buffer<BufferType::L0A>& qkLeftL0ABuffer = mm1LeftBuffers_.Get();
            Buffer<BufferType::L0B, SyncType::NO_SYNC>& qkRightL0BBuffer = mm1RightBuffers_.Get();
            Buffer<BufferType::L0C>& qkResultL0CBuffer = mm1ResBuffers_.Get();
            qkLeftL0ABuffer.template Wait<HardEvent::M_MTE1>();
            LocalTensor<Q_T> queryL1Tensor = queryL1Buffer.template GetTensor<Q_T>();
            LocalTensor<SCALE_T> queryScaleL1Tensor =
                queryL1Buffer.template GetTensor<uint8_t>(mBaseSize * dBaseSize / 2)
                              .template ReinterpretCast<SCALE_T>();
            if constexpr (USE_DN) {
                LoadK2L0(qkLeftL0ABuffer.template GetTensor<KV_T>(), keyL1Tensor, keyScaleL1Tensor,
                    subBlockS2Offset, validKvTokenCount);
                if (queryL0ReadyCount_ < QK_L0_SLOT_COUNT) {
                    LoadQ2L0(qkRightL0BBuffer.template GetTensor<Q_T>(), queryL1Tensor, queryScaleL1Tensor, runInfo);
                }
            } else {
                if (queryL0ReadyCount_ < QK_L0_SLOT_COUNT) {
                    LoadQ2L0(qkLeftL0ABuffer.template GetTensor<Q_T>(), queryL1Tensor, queryScaleL1Tensor, runInfo);
                }
                LoadK2L0(qkRightL0BBuffer.template GetTensor<KV_T>(), keyL1Tensor, keyScaleL1Tensor,
                    subBlockS2Offset, validKvTokenCount);
            }
            if (queryL0ReadyCount_ < QK_L0_SLOT_COUNT) {
                ++queryL0ReadyCount_;
            }
            qkLeftL0ABuffer.template Set<HardEvent::MTE1_M>();
            qkLeftL0ABuffer.template Wait<HardEvent::MTE1_M>();
            qkResultL0CBuffer.template Wait<HardEvent::FIX_M>();
            MatmulQK(qkResultL0CBuffer.template GetTensor<T>(), qkLeftL0ABuffer.template GetTensor<INPUT_T>(),
                     qkRightL0BBuffer.template GetTensor<INPUT_T>(), validKvTokenCount, runInfo.actMSize);
            qkLeftL0ABuffer.template Set<HardEvent::M_MTE1>();
            qkResultL0CBuffer.template Set<HardEvent::M_FIX>();
            qkResultL0CBuffer.template Wait<HardEvent::M_FIX>();
            const uint32_t scoreSlot = scoreSequence_++ % CrossBufferType::SCORE_SLOT_COUNT;
            crossBuffers_.waitBmm1Buffer(scoreSlot);
            FixpipeMm1(crossBuffers_.GetBmm1Slot(scoreSlot), qkResultL0CBuffer.template GetTensor<T>(), runInfo,
                validKvTokenCount);
            // Even an AIV with no valid query rows receives and acknowledges this event.
            crossBuffers_.setBmm1Buffer(scoreSlot);
            qkResultL0CBuffer.template Set<HardEvent::FIX_M>();
        }
        keyL1Buffer.template Set<HardEvent::MTE1_MTE2>();
        if (runInfo.isLastS2Loop) {
            queryL1Buffer.template Set<HardEvent::MTE1_MTE2>();
            qCached_ = false;
        }
    }

    __aicore__ inline void ComputeBmm2(Mxfp4RunInfo& runInfo)
    {
        if (runInfo.actMSize == 0 || runInfo.actSingleLoopS2Size == 0) {
            return;
        }
        ASSERT(pvGroupTasks_ < 16);
        const bool isGroupFirst = pvGroupTasks_ == 0;
        const bool isGroupLast = runInfo.isGroupLast || runInfo.isLastS2Loop;
        Buffer<BufferType::L0C>& pvResultL0CBuffer = mm2ResBuffers_.Get();
        if (isGroupFirst) {
            crossBuffers_.waitPscaleUpdate(pvGroupSequence_);
            pvResultL0CBuffer.template Wait<HardEvent::FIX_M>();
        }
        Buffer<BufferType::L1>& valueL1Buffer = l1KVBuffers_.Get();
        valueL1Buffer.template Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> valueL1Tensor = valueL1Buffer.template GetTensor<KV_T>();
        LocalTensor<SCALE_T> valueScaleL1Tensor = valueL1Buffer.template GetTensor<uint8_t>(s2BaseSize * dVBaseSize / 2)
                          .template ReinterpretCast<SCALE_T>();
        CopyVGm2L1(valueL1Tensor, runInfo);
        CopyVScaleGm2L1(valueScaleL1Tensor, runInfo);
        valueL1Buffer.template Set<HardEvent::MTE2_MTE1>();
        valueL1Buffer.template Wait<HardEvent::MTE2_MTE1>();
        const uint32_t alignedKvTokenCount = (runInfo.actSingleLoopS2Size + 63U) / 64U * 64U;
        Buffer<BufferType::L0A>& pvLeftL0ABuffer = mm2LeftBuffers_.Get();
        Buffer<BufferType::L0B, SyncType::NO_SYNC>& pvRightL0BBuffer = mm2RightBuffers_.Get();
        pvLeftL0ABuffer.template Wait<HardEvent::M_MTE1>();
        if constexpr (USE_DN) {
            InitPaddingL0(pvLeftL0ABuffer.template GetTensor<KV_T>(), alignedKvTokenCount);
            LoadV2L0(pvLeftL0ABuffer.template GetTensor<KV_T>(), valueL1Tensor, valueScaleL1Tensor,
                alignedKvTokenCount);
            LoadP2L0(pvRightL0BBuffer.template GetTensor<INPUT_T>(), runInfo.loop % P_SLOT_COUNT, alignedKvTokenCount, runInfo.actMSizeAlign32);
        } else {
            LoadP2L0(pvLeftL0ABuffer.template GetTensor<INPUT_T>(), runInfo.loop % P_SLOT_COUNT, alignedKvTokenCount, runInfo.actMSizeAlign32);
            InitPaddingL0(pvRightL0BBuffer.template GetTensor<KV_T>(), alignedKvTokenCount);
            LoadV2L0(pvRightL0BBuffer.template GetTensor<KV_T>(), valueL1Tensor, valueScaleL1Tensor,
                alignedKvTokenCount);
        }
        valueL1Buffer.template Set<HardEvent::MTE1_MTE2>();
        pvLeftL0ABuffer.template Set<HardEvent::MTE1_M>();
        pvLeftL0ABuffer.template Wait<HardEvent::MTE1_M>();
        if (!isGroupFirst) {
            PipeBarrier<PIPE_M>(); // Accumulation reuses the same L0C matrix.
        }
        MatmulPV(pvResultL0CBuffer.template GetTensor<T>(), pvLeftL0ABuffer.template GetTensor<INPUT_T>(),
                 pvRightL0BBuffer.template GetTensor<INPUT_T>(), alignedKvTokenCount, isGroupFirst, runInfo.actMSizeAlign32);
        pvLeftL0ABuffer.template Set<HardEvent::M_MTE1>();
        ++pvGroupTasks_;
        if (isGroupLast) {
            pvResultL0CBuffer.template Set<HardEvent::M_FIX>();
            pvResultL0CBuffer.template Wait<HardEvent::M_FIX>();
            crossBuffers_.watBmm2Buffer();
            FixpipeMm2(crossBuffers_.GetBmm2Slot(0), pvResultL0CBuffer.template GetTensor<T>(), runInfo);
            crossBuffers_.setBmm2Buffer();
            pvResultL0CBuffer.template Set<HardEvent::FIX_M>();
            pvGroupTasks_ = 0;
            ++pvGroupSequence_;
        }
    }

    __aicore__ inline void AllocEventID()
    {
        crossBuffers_.InitSync();
    }

    __aicore__ inline void FreeEventID()
    {
        ASSERT(pvGroupTasks_ == 0);
        crossBuffers_.waitBmm1Buffer(0);
        if constexpr (CrossBufferType::SCORE_SLOT_COUNT == 2) {
            crossBuffers_.waitBmm1Buffer(1);
        }
        crossBuffers_.watBmm2Buffer();
        if (qCached_) {
            l1QBuffers_.Get().template Set<HardEvent::MTE1_MTE2>();
            qCached_ = false;
        }
        l1VPaddingBuffers_.Get().template Set<HardEvent::MTE1_MTE2>();
        l1QBuffers_.Uninit(l1BufferManager_);
        l1KVBuffers_.Uninit(l1BufferManager_);
        l1VPaddingBuffers_.Uninit(l1BufferManager_);
        mm1LeftBuffers_.Uninit(l0aBufferManager_);
        mm2LeftBuffers_.Uninit(l0aBufferManager_);
        mm1RightBuffers_.Uninit(l0bBufferManager_);
        mm2RightBuffers_.Uninit(l0bBufferManager_);
        mm1ResBuffers_.Uninit(l0cBufferManager_);
        mm2ResBuffers_.Uninit(l0cBufferManager_);
    }

    __aicore__ inline void MatmulQK(const LocalTensor<T>& scoreL0CTensor, const LocalTensor<INPUT_T>& qkLeftL0ATensor,
                                   const LocalTensor<INPUT_T>& qkRightL0BTensor, uint32_t validKvTokenCount,
                                    uint32_t validQueryRowCount)
    {
        MmadParams qkMmadParams {};
        // ND dual destination splits M equally into the two Q16-aligned AIV
        // shares. Mmad and Fixpipe must agree on the padded NZ row stride.
        const uint32_t queryRows = (!USE_DN && !ENABLE_QF322F32_PRE) ?
            (validQueryRowCount + 31U) / 32U * 32U : validQueryRowCount;
        qkMmadParams.m = USE_DN ? validKvTokenCount : queryRows;
        qkMmadParams.n = USE_DN ? validQueryRowCount : validKvTokenCount;
        qkMmadParams.k = dBaseSize;
        qkMmadParams.cmatrixInitVal = true;
        qkMmadParams.cmatrixSource = false;
        qkMmadParams.disableGemv = true;
        Mmad(scoreL0CTensor, qkLeftL0ATensor, qkRightL0BTensor, qkMmadParams);
    }

    __aicore__ inline void MatmulPV(const LocalTensor<T>& pvResultL0CTensor,
        const LocalTensor<INPUT_T>& pvLeftL0ATensor,
                                   const LocalTensor<INPUT_T>& pvRightL0BTensor, uint32_t alignedKvTokenCount,
                                    bool isGroupFirst, uint32_t alignedQueryRows)
    {
        MmadParams pvMmadParams {};
        pvMmadParams.m = USE_DN ? dVBaseSize + 16 : alignedQueryRows;
        pvMmadParams.n = USE_DN ? mBaseSize : dVBaseSize + 16;
        pvMmadParams.k = alignedKvTokenCount;
        pvMmadParams.cmatrixInitVal = isGroupFirst;
        pvMmadParams.cmatrixSource = false;
        pvMmadParams.disableGemv = true;
        Mmad(pvResultL0CTensor, pvLeftL0ATensor, pvRightL0BTensor, pvMmadParams);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<float>& scoreUbTensor, const LocalTensor<T>& scoreL0CTensor,
                                     Mxfp4RunInfo& runInfo, uint32_t validKvTokenCount)
    {
        if constexpr (!ENABLE_QF322F32_PRE) {
            // Dual destination requires NoQuant. V1 applies the FP32 score
            // scale exactly once. DN keeps the fixed Q64 split even for tails;
            // padded Q columns stay inside this L0C slot and are not output.
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> scoreFixpipeParams {};
            scoreFixpipeParams.mSize = USE_DN ? validKvTokenCount : runInfo.actMSizeAlign32;
            scoreFixpipeParams.nSize = USE_DN ? mBaseSize : (validKvTokenCount + 7U) / 8U * 8U;
            scoreFixpipeParams.srcStride = USE_DN ?
                (validKvTokenCount + 15U) / 16U * 16U : runInfo.actMSizeAlign32;
            scoreFixpipeParams.dstStride = USE_DN ? mBaseSize / CV_RATIO : s2SplitSize;
            scoreFixpipeParams.dualDstCtl = USE_DN ? 2 : 1;
            scoreFixpipeParams.quantPre = QuantMode_t::NoQuant;
            scoreFixpipeParams.params.ndNum = 1;
            Fixpipe<float, T, FIXPIPE_CONFIG>(scoreUbTensor, scoreL0CTensor, scoreFixpipeParams);
        } else {
            const uint32_t l0cRowStride = ((USE_DN ? validKvTokenCount : runInfo.actMSize) + 15U) / 16U * 16U;
            for (uint32_t aivIndex = 0; aivIndex < CV_RATIO; ++aivIndex) {
                const uint32_t rowsPerAiv = USE_DN ? mBaseSize / CV_RATIO : runInfo.actMSizeAlign32 / CV_RATIO;
                const uint32_t firstQueryRow = aivIndex * rowsPerAiv;
                if (firstQueryRow >= runInfo.actMSize) {
                    continue;
                }
                const uint32_t validQueryRowCount = AttentionCommon::Min(rowsPerAiv, runInfo.actMSize - firstQueryRow);
                FixpipeParamsC310<CO2Layout::ROW_MAJOR> scoreFixpipeParams {};
                scoreFixpipeParams.mSize = USE_DN ? validKvTokenCount : validQueryRowCount;
                scoreFixpipeParams.nSize = ((USE_DN ? validQueryRowCount : validKvTokenCount) + 7U) / 8U * 8U;
                scoreFixpipeParams.srcStride = l0cRowStride;
                scoreFixpipeParams.dstStride = USE_DN ? mBaseSize / CV_RATIO : s2SplitSize;
                scoreFixpipeParams.dualDstCtl = 0;
                scoreFixpipeParams.subBlockId = aivIndex;
                scoreFixpipeParams.quantPre = QuantMode_t::NoQuant;
                if (CrossBufferType::CanScaleInFixpipe(constInfo_.scaleValue)) {
                    scoreFixpipeParams.quantPre = QuantMode_t::QF322F32_PRE;
                    float scale = constInfo_.scaleValue;
                    scoreFixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&scale));
                }

                scoreFixpipeParams.params.ndNum = 1;
                const uint32_t l0cQueryOffset = USE_DN ? firstQueryRow * l0cRowStride : firstQueryRow * 16U;
                Fixpipe<float, T, FIXPIPE_CONFIG>(scoreUbTensor, scoreL0CTensor[l0cQueryOffset], scoreFixpipeParams);
            }
        }
    }

    __aicore__ inline void FixpipeMm2(const LocalTensor<float>& pvResultUbTensor,
        const LocalTensor<T>& pvResultL0CTensor, const Mxfp4RunInfo& runInfo)
    {
        if constexpr (!USE_DN) {
          if (runInfo.actMSizeAlign32 < mBaseSize) {
            const uint32_t rowsPerAiv = runInfo.actMSizeAlign32 / CV_RATIO;
            for (uint32_t aivIndex = 0; aivIndex < CV_RATIO; ++aivIndex) {
                const uint32_t firstQueryRow = aivIndex * rowsPerAiv;
                if (firstQueryRow >= runInfo.actMSize) {
                    continue;
                }
                FixpipeParamsC310<CO2Layout::ROW_MAJOR> params {};
                params.mSize = AttentionCommon::Min(rowsPerAiv, runInfo.actMSize - firstQueryRow);
                params.nSize = dVBaseSize + 8;
                params.srcStride = runInfo.actMSizeAlign32;
                params.dstStride = dVBaseSize + 8;
                params.dualDstCtl = 0;
                params.subBlockId = aivIndex;
                params.quantPre = QuantMode_t::NoQuant;
                params.params.ndNum = 1;
                Fixpipe<float, T, FIXPIPE_CONFIG>(pvResultUbTensor,
                    pvResultL0CTensor[firstQueryRow * 16U], params);
            }
            return;
          }
        }
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> pvFixpipeParams {};
        pvFixpipeParams.mSize = USE_DN ? dVBaseSize + 1 : mBaseSize;
        pvFixpipeParams.nSize = USE_DN ? mBaseSize : dVBaseSize + 8;
        pvFixpipeParams.srcStride = USE_DN ? dVBaseSize + 16 : mBaseSize;
        pvFixpipeParams.dstStride = USE_DN ? mBaseSize / CV_RATIO : dVBaseSize + 8;
        pvFixpipeParams.dualDstCtl = USE_DN ? 2 : 1;
        pvFixpipeParams.quantPre = QuantMode_t::NoQuant;
        pvFixpipeParams.params.ndNum = 1;
        Fixpipe<float, T, FIXPIPE_CONFIG>(pvResultUbTensor, pvResultL0CTensor, pvFixpipeParams);
    }

    __aicore__ inline void LoadQ2L0(const LocalTensor<Q_T>& queryL0Tensor, const LocalTensor<Q_T>& queryL1Tensor,
                                   const LocalTensor<SCALE_T>& queryScaleL1Tensor, Mxfp4RunInfo& runInfo)
    {
        LoadData2DParamsV2 queryLoadParams {};
        queryLoadParams.mStep = (!USE_DN && !ENABLE_QF322F32_PRE) ?
            runInfo.actMSizeAlign32 / 16U : (runInfo.actMSize + 15U) / 16U;
        queryLoadParams.kStep = dBaseSize / 64;
        queryLoadParams.srcStride = runInfo.actMSizeAlign32 / 16;
        queryLoadParams.dstStride = queryLoadParams.mStep;
        queryLoadParams.ifTranspose = false;
        LoadData2DMxParams queryScaleLoadParams {};
        queryScaleLoadParams.xStep = queryLoadParams.mStep;
        queryScaleLoadParams.yStep = dBaseSize / 64;
        queryScaleLoadParams.srcStride = queryScaleLoadParams.yStep;
        queryScaleLoadParams.dstStride = queryScaleLoadParams.yStep;
        LoadData(queryL0Tensor, queryL1Tensor, queryScaleL1Tensor, queryLoadParams, queryScaleLoadParams);
    }

    __aicore__ inline void LoadK2L0(const LocalTensor<KV_T>& keyL0Tensor, const LocalTensor<KV_T>& keyL1Tensor,
                                   const LocalTensor<SCALE_T>& keyScaleL1Tensor, uint32_t subBlockS2Offset,
                                    uint32_t validKvTokenCount)
    {
        LoadData2DParamsV2 keyLoadParams {};
        keyLoadParams.mStartPosition = subBlockS2Offset / 16;
        keyLoadParams.mStep = (validKvTokenCount + 15U) / 16U;
        keyLoadParams.kStep = dBaseSize / 64;
        keyLoadParams.srcStride = s2BaseSize / 16;
        keyLoadParams.dstStride = keyLoadParams.mStep;
        keyLoadParams.ifTranspose = false;
        LoadData2DMxParams keyScaleLoadParams {};
        keyScaleLoadParams.xStartPosition = subBlockS2Offset / 16;
        keyScaleLoadParams.xStep = keyLoadParams.mStep;
        keyScaleLoadParams.yStep = dBaseSize / 64;
        keyScaleLoadParams.srcStride = keyScaleLoadParams.yStep;
        keyScaleLoadParams.dstStride = keyScaleLoadParams.yStep;
        LoadData(keyL0Tensor, keyL1Tensor, keyScaleL1Tensor, keyLoadParams, keyScaleLoadParams);
    }

    __aicore__ inline void LoadP2L0(const LocalTensor<INPUT_T>& probabilityL0Tensor, uint32_t probabilitySlot,
        uint32_t alignedKvTokenCount, uint32_t alignedQueryRows)
    {
        LoadData2DParamsV2 probabilityLoadParams {};
        probabilityLoadParams.mStep = USE_DN ? alignedKvTokenCount / 16 : alignedQueryRows / 16;
        probabilityLoadParams.kStep = USE_DN ? mBaseSize / 64 : alignedKvTokenCount / 64;
        probabilityLoadParams.srcStride = USE_DN ? s2BaseSize / 16 : mBaseSize / 16;
        probabilityLoadParams.dstStride = USE_DN ? mBaseSize / 16 : alignedQueryRows / 16;
        probabilityLoadParams.ifTranspose = USE_DN;
        LoadData2DMxParams probabilityScaleLoadParams {};
        probabilityScaleLoadParams.xStep = USE_DN ? mBaseSize / 16 : alignedQueryRows / 16;
        probabilityScaleLoadParams.yStep = alignedKvTokenCount / 64;
        probabilityScaleLoadParams.srcStride = s2BaseSize / 64 + 1;
        probabilityScaleLoadParams.dstStride = alignedKvTokenCount / 64;
        LoadData(probabilityL0Tensor, crossBuffers_.GetPSlot(probabilitySlot).template ReinterpretCast<INPUT_T>(),
                 crossBuffers_.GetPScaleSlot(probabilitySlot).template ReinterpretCast<SCALE_T>(),
                     probabilityLoadParams, probabilityScaleLoadParams);
    }

    __aicore__ inline void LoadV2L0(const LocalTensor<KV_T>& valueL0Tensor, const LocalTensor<KV_T>& valueL1Tensor,
                                   const LocalTensor<SCALE_T>& valueScaleL1Tensor, uint32_t alignedKvTokenCount)
    {
        LoadData2DParamsV2 valueLoadParams {};
        valueLoadParams.mStep = alignedKvTokenCount / 16;
        valueLoadParams.kStep = dVBaseSize / 64;
        valueLoadParams.srcStride = s2BaseSize / 16;
        valueLoadParams.dstStride = (dVBaseSize + 16) / 16;
        valueLoadParams.ifTranspose = true;
        LoadData2DMxParams valueScaleLoadParams {};
        valueScaleLoadParams.xStep = dVBaseSize / 16;
        valueScaleLoadParams.yStep = alignedKvTokenCount / 64;
        valueScaleLoadParams.srcStride = s2BaseSize / 64;
        valueScaleLoadParams.dstStride = alignedKvTokenCount / 64;
        LoadData(valueL0Tensor, valueL1Tensor, valueScaleL1Tensor, valueLoadParams, valueScaleLoadParams);
    }

    __aicore__ inline void CopyFp4RowsGm2L1(const LocalTensor<uint8_t>& payloadL1Tensor,
                                           const GlobalTensor<uint8_t>& payloadGmTensor, uint32_t rowCount,
                                           uint32_t rowByteCount, uint64_t sourceRowByteStride,
                                    uint32_t destinationRowStride)
    {
        if (rowCount == 0) {
            return;
        }
        Nd2NzParams payloadCopyParams {};
        payloadCopyParams.ndNum = 1;
        payloadCopyParams.nValue = rowCount;
        payloadCopyParams.dValue = rowByteCount;
        payloadCopyParams.dstNzC0Stride = destinationRowStride;
        payloadCopyParams.dstNzNStride = 1;
        // ND2NZ 的 srcDValue 只有 16 bit，大 head stride 退化为逐行搬运。
        if (sourceRowByteStride <= 65535U) {
            payloadCopyParams.srcDValue = static_cast<uint16_t>(sourceRowByteStride);
            DataCopy<uint8_t>(payloadL1Tensor, payloadGmTensor, payloadCopyParams);
        } else {
            payloadCopyParams.nValue = 1;
            payloadCopyParams.srcDValue = rowByteCount;
            for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
                DataCopy<uint8_t>(payloadL1Tensor[rowIndex * 32U], payloadGmTensor[rowIndex * sourceRowByteStride],
                    payloadCopyParams);
            }
        }
    }

    __aicore__ inline void CopyQGm2L1Dn(const LocalTensor<Q_T>& queryL1Tensor, Mxfp4RunInfo& runInfo)
    {
        static_assert(LAYOUT == LayOutTypeEnum::LAYOUT_TND, "MXFP4 Q copy currently supports TND only");
        if (runInfo.actMSize == 0) {
            return;
        }

        LocalTensor<uint8_t> queryL1Bytes = queryL1Tensor.template ReinterpretCast<uint8_t>();
        constexpr uint32_t queryRowBytes = dBaseSize / 2;
        const uint64_t queryHeadCount = static_cast<uint64_t>(constInfo_.realN2Size) * constInfo_.realGSize;
        const uint64_t batchTokenBase = qSeqParserPtr_->GetTBase(runInfo.bIdx);
        // DN 已展开 head，realGSize == 1，realN2Idx 是 Q head（不是 KV head）。
        const uint64_t queryGmByteOffset =
            ((batchTokenBase + runInfo.gS1Idx) * queryHeadCount + runInfo.realN2Idx) * queryRowBytes;
        CopyFp4RowsGm2L1(queryL1Bytes, queryGm_[queryGmByteOffset], runInfo.actMSize, queryRowBytes,
                         queryHeadCount * queryRowBytes, runInfo.actMSizeAlign32);
    }

    __aicore__ inline void CopyQGm2L1Nd(const LocalTensor<Q_T>& queryL1Tensor, Mxfp4RunInfo& runInfo)
    {
        static_assert(LAYOUT == LayOutTypeEnum::LAYOUT_TND, "MXFP4 Q copy currently supports TND only");
        if (runInfo.actMSize == 0) {
            return;
        }

        LocalTensor<uint8_t> queryL1Bytes = queryL1Tensor.template ReinterpretCast<uint8_t>();
        constexpr uint32_t queryRowBytes = dBaseSize / 2;
        const uint32_t groupSize = constInfo_.realGSize;
        const uint64_t tokenByteStride = static_cast<uint64_t>(constInfo_.realN2Size) * groupSize * queryRowBytes;
        const uint64_t batchTokenBase = qSeqParserPtr_->GetTBase(runInfo.bIdx);
        const uint32_t firstGroupHead = runInfo.gS1Idx % groupSize;
        uint64_t queryGmByteOffset = (batchTokenBase + runInfo.gS1Idx / groupSize) * tokenByteStride +
            (static_cast<uint64_t>(runInfo.realN2Idx) * groupSize + firstGroupHead) * queryRowBytes;
        uint32_t copiedRows = 0;
        uint32_t remainingQueryRows = runInfo.actMSize;
        // 只有一个 KV head 时，GM 中 T、G 本来就是连续的。
        if (constInfo_.realN2Size == 1) {
            CopyFp4RowsGm2L1(queryL1Bytes, queryGm_[queryGmByteOffset], remainingQueryRows, queryRowBytes,
                             queryRowBytes, runInfo.actMSizeAlign32);
            return;
        }
        // 不完整首 tokenIndex 单独搬；完整的首、尾 tokenIndex 均参与批量搬运。
        if (firstGroupHead != 0) {
            const uint32_t firstTokenRowCount = AttentionCommon::Min(remainingQueryRows, groupSize - firstGroupHead);
            CopyFp4RowsGm2L1(queryL1Bytes, queryGm_[queryGmByteOffset], firstTokenRowCount, queryRowBytes,
                             queryRowBytes, runInfo.actMSizeAlign32);
            copiedRows += firstTokenRowCount;
            remainingQueryRows -= firstTokenRowCount;
            queryGmByteOffset += tokenByteStride - firstGroupHead * queryRowBytes;
        }
        const uint32_t fullTokenCount = remainingQueryRows / groupSize;
        if (fullTokenCount != 0) {
            Nd2NzParams queryCopyParams {};
            queryCopyParams.ndNum = fullTokenCount;
            queryCopyParams.nValue = groupSize;
            queryCopyParams.dValue = queryRowBytes;
            queryCopyParams.srcDValue = queryRowBytes;
            queryCopyParams.dstNzC0Stride = runInfo.actMSizeAlign32;
            queryCopyParams.dstNzNStride = 1;
            // 多个 tokenIndex 拼入同一个 NZ 矩阵，目标行偏移以 32B 为单位。
            queryCopyParams.dstNzMatrixStride = groupSize * 32U;
            if (tokenByteStride <= 65535U) {
                queryCopyParams.srcNdMatrixStride = static_cast<uint16_t>(tokenByteStride);
                DataCopy<uint8_t>(queryL1Bytes[copiedRows * 32U], queryGm_[queryGmByteOffset], queryCopyParams);
            } else {
                queryCopyParams.ndNum = 1;
                for (uint32_t tokenIndex = 0; tokenIndex < fullTokenCount; ++tokenIndex) {
                    DataCopy<uint8_t>(queryL1Bytes[(copiedRows + tokenIndex * groupSize) * 32U],
                                             queryGm_[queryGmByteOffset + tokenIndex * tokenByteStride],
                                    queryCopyParams);
                }
            }
            const uint32_t fullTokenRowCount = fullTokenCount * groupSize;
            copiedRows += fullTokenRowCount;
            remainingQueryRows -= fullTokenRowCount;
            queryGmByteOffset += fullTokenCount * tokenByteStride;
        }
        if (remainingQueryRows != 0) {
            CopyFp4RowsGm2L1(queryL1Bytes[copiedRows * 32U], queryGm_[queryGmByteOffset], remainingQueryRows,
                queryRowBytes,
                             queryRowBytes, runInfo.actMSizeAlign32);
        }
    }

    __aicore__ inline void CopyQScaleGm2L1(const LocalTensor<SCALE_T>& queryScaleL1Tensor, Mxfp4RunInfo& runInfo)
    {
        static_assert(LAYOUT == LayOutTypeEnum::LAYOUT_TND, "MXFP4 Q scale copy currently supports TND only");
        if (runInfo.actMSize == 0) {
            return;
        }
        // uint16_t 原样搬运一对 E8M0，不做数值转换。
        // GM: DN [T,N1,D/64,2]；ND [N2,T,G,D/64,2]。
        // L1: [ceil(M/16),D/64,16,2]，scale 起点由完整 Q payload 预留决定。

        LocalTensor<uint16_t> queryScaleL1Pairs = queryScaleL1Tensor.template ReinterpretCast<uint16_t>();
        constexpr uint32_t scalePairsPerQueryRow = dBaseSize / MXFP_DIVISOR_SIZE;
        const uint32_t groupSize = constInfo_.realGSize;
        const uint64_t queryHeadCount = static_cast<uint64_t>(constInfo_.realN2Size) * groupSize;
        const uint64_t batchTokenBase = qSeqParserPtr_->GetTBase(runInfo.bIdx);
        uint64_t queryScaleGmPairOffset;
        if constexpr (USE_DN) {
            queryScaleGmPairOffset = ((batchTokenBase + runInfo.gS1Idx) * queryHeadCount +
                runInfo.realN2Idx) * scalePairsPerQueryRow;
        } else {
            queryScaleGmPairOffset =
                (static_cast<uint64_t>(runInfo.realN2Idx) * qSeqParserPtr_->GetTSize() * groupSize +
                        batchTokenBase * groupSize + runInfo.gS1Idx) * scalePairsPerQueryRow;
        }
        Dn2NzParams queryScaleCopyParams {};
        queryScaleCopyParams.dnNum = 1;
        queryScaleCopyParams.nValue = scalePairsPerQueryRow;
        queryScaleCopyParams.dValue = runInfo.actMSize;
        queryScaleCopyParams.srcDValue = USE_DN ? queryHeadCount * scalePairsPerQueryRow : scalePairsPerQueryRow;
        queryScaleCopyParams.dstNzC0Stride = scalePairsPerQueryRow;
        queryScaleCopyParams.dstNzNStride = 1;
        DataCopy<uint16_t>(queryScaleL1Pairs, queryScaleGm_[queryScaleGmPairOffset], queryScaleCopyParams);
    }

    __aicore__ inline void CopyKvPayloadGm2L1(const LocalTensor<uint8_t>& kvPayloadL1Tensor,
                                             const GlobalTensor<uint8_t>& kvPayloadGmTensor, uint64_t pageLogicalStride,
                                             uint32_t payloadRowBytes, Mxfp4RunInfo& runInfo)
    {
        static_assert(KvLayoutType == 1, "MXFP4 KV copy currently supports PA_BBND only");

        const uint32_t blockSize = constInfo_.blockSize;
        const uint64_t tokenByteStride = static_cast<uint64_t>(constInfo_.n2Size) * payloadRowBytes;
        // Host 的 FP4 stride 是逻辑元素数，需要折半；仅物理页轴允许不连续。
        // BBND 的 n2Stride 实际是 stride[1]，不能拿它当 head stride。
        const uint64_t pageByteStride = pageLogicalStride == 0 ? blockSize * tokenByteStride : pageLogicalStride / 2;
        uint32_t copiedTokenCount = 0;
        while (copiedTokenCount < runInfo.actSingleLoopS2Size) {
            const uint64_t logicalTokenIndex = static_cast<uint64_t>(runInfo.s2Idx) + copiedTokenCount;
            const uint32_t pageTokenOffset = logicalTokenIndex % blockSize;
            const uint32_t pageTokenCount =
                AttentionCommon::Min(static_cast<uint32_t>(runInfo.actSingleLoopS2Size - copiedTokenCount),
                                      blockSize - pageTokenOffset);
            const uint64_t physicalPageIndex = blockTableGm_.GetValue(
                static_cast<uint64_t>(runInfo.bIdx) * constInfo_.maxBlockNumPerBatch + logicalTokenIndex / blockSize);
            const uint64_t payloadGmByteOffset = physicalPageIndex * pageByteStride +
                pageTokenOffset * tokenByteStride +
                                    static_cast<uint64_t>(runInfo.n2Idx) * payloadRowBytes;
            CopyFp4RowsGm2L1(kvPayloadL1Tensor[copiedTokenCount * 32U], kvPayloadGmTensor[payloadGmByteOffset],
                pageTokenCount, payloadRowBytes, tokenByteStride, s2BaseSize);
            copiedTokenCount += pageTokenCount;
        }
    }

    // 调用者负责目标槽、payload/scale 区域及同步；搬运函数不分配或轮转 buffer。
    __aicore__ inline void CopyKGm2L1(const LocalTensor<KV_T>& keyL1Tensor, Mxfp4RunInfo& runInfo)
    {
        LocalTensor<uint8_t> keyL1Bytes = keyL1Tensor.template ReinterpretCast<uint8_t>();
        CopyKvPayloadGm2L1(keyL1Bytes, keyGm_, constInfo_.keyStrides.bnStride, dBaseSize / 2, runInfo);
    }

    __aicore__ inline void CopyKScaleGm2L1(const LocalTensor<SCALE_T>& keyScaleL1Tensor, Mxfp4RunInfo& runInfo)
    {
        static_assert(KvLayoutType == 1, "MXFP4 K scale copy currently supports PA_BBND only");
        // GM: [numBlocks,blockSize,N2,D/64,2]；L1: [ceil(logicalTokenIndex/16),D/64,16,2]。
        constexpr uint32_t scalePairsPerKeyRow = dBaseSize / MXFP_DIVISOR_SIZE;
        LocalTensor<uint16_t> keyScaleL1Pairs = keyScaleL1Tensor.template ReinterpretCast<uint16_t>();
        const uint32_t blockSize = constInfo_.blockSize;
        const uint64_t tokenPairStride = static_cast<uint64_t>(constInfo_.n2Size) * scalePairsPerKeyRow;
        const uint64_t pagePairStride = constInfo_.kDescaleStrides.bnStride == 0 ?
            blockSize * tokenPairStride : constInfo_.kDescaleStrides.bnStride / 2;
        Dn2NzParams keyScaleCopyParams {};
        keyScaleCopyParams.dnNum = 1;
        keyScaleCopyParams.nValue = scalePairsPerKeyRow;
        keyScaleCopyParams.srcDValue = tokenPairStride;
        keyScaleCopyParams.dstNzC0Stride = scalePairsPerKeyRow;
        keyScaleCopyParams.dstNzNStride = 1;
        uint32_t copiedTokenCount = 0;
        while (copiedTokenCount < runInfo.actSingleLoopS2Size) {
            const uint64_t logicalTokenIndex = static_cast<uint64_t>(runInfo.s2Idx) + copiedTokenCount;
            const uint32_t pageTokenOffset = logicalTokenIndex % blockSize;
            const uint32_t pageTokenCount =
                AttentionCommon::Min(static_cast<uint32_t>(runInfo.actSingleLoopS2Size - copiedTokenCount),
                                      blockSize - pageTokenOffset);
            const uint64_t physicalPageIndex = blockTableGm_.GetValue(
                static_cast<uint64_t>(runInfo.bIdx) * constInfo_.maxBlockNumPerBatch + logicalTokenIndex / blockSize);
            const uint64_t keyScaleGmPairOffset = physicalPageIndex * pagePairStride +
                pageTokenOffset * tokenPairStride +
                                    static_cast<uint64_t>(runInfo.n2Idx) * scalePairsPerKeyRow;
            keyScaleCopyParams.dValue = pageTokenCount;
            // 页边界为 64 token 的倍数，每个后续目标都是完整 16 行分块的起点。
            DataCopy<uint16_t>(keyScaleL1Pairs[copiedTokenCount * scalePairsPerKeyRow],
                keyScaleGm_[keyScaleGmPairOffset], keyScaleCopyParams);
            copiedTokenCount += pageTokenCount;
        }
    }

    __aicore__ inline void CopyVGm2L1(const LocalTensor<KV_T>& valueL1Tensor, Mxfp4RunInfo& runInfo)
    {
        LocalTensor<uint8_t> valueL1Bytes = valueL1Tensor.template ReinterpretCast<uint8_t>();
        CopyKvPayloadGm2L1(valueL1Bytes, valueGm_, constInfo_.valueStrides.bnStride, dVBaseSize / 2, runInfo);
        // PV 归约轴按 64 token 对齐；槽复用时清零无效 V 行。
        // 额外的 rowsum 常数列由 l1VPaddingBuffers_ 提供，不在这里修改。
        const uint32_t validKvTokenCount = runInfo.actSingleLoopS2Size;
        const uint32_t paddingTokenCount = (validKvTokenCount + 63U) / 64U * 64U - validKvTokenCount;
        if (paddingTokenCount != 0) {
            InitConstValueParams<uint16_t> paddingFillParams(1, static_cast<uint16_t>(paddingTokenCount), 0, 0);
            for (uint32_t valueChannelBlock = 0; valueChannelBlock < dVBaseSize / 64U; ++valueChannelBlock) {
                LocalTensor<uint16_t> valuePaddingL1Tensor = valueL1Bytes[(valueChannelBlock * s2BaseSize +
                    validKvTokenCount) * 32U].template ReinterpretCast<uint16_t>();
                Fill(valuePaddingL1Tensor, paddingFillParams);
            }
        }
    }

    __aicore__ inline void CopyVScaleGm2L1(const LocalTensor<SCALE_T>& valueScaleL1Tensor, Mxfp4RunInfo& runInfo)
    {
        static_assert(KvLayoutType == 1, "MXFP4 V scale copy currently supports PA_BBND only");
        // GM: [numBlocks,blockSize/64,N2,Dv,2]；L1: [Dv/16,s2BaseSize/64,16,2]。
        LocalTensor<uint16_t> valueScaleL1Pairs = valueScaleL1Tensor.template ReinterpretCast<uint16_t>();
        const uint32_t scaleGroupsPerPage = constInfo_.blockSize / MXFP_DIVISOR_SIZE;
        const uint32_t scaleGroupCount = (runInfo.actSingleLoopS2Size + 63U) / MXFP_DIVISOR_SIZE;
        const uint64_t scaleGroupPairStride = static_cast<uint64_t>(constInfo_.n2Size) * dVBaseSize;
        const uint64_t pagePairStride = constInfo_.vDescaleStrides.bnStride == 0 ?
            scaleGroupsPerPage * scaleGroupPairStride : constInfo_.vDescaleStrides.bnStride / 2;
        Nd2NzParams valueScaleCopyParams {};
        valueScaleCopyParams.ndNum = 1;
        valueScaleCopyParams.dValue = dVBaseSize;
        valueScaleCopyParams.dstNzC0Stride = s2BaseSize / MXFP_DIVISOR_SIZE;
        valueScaleCopyParams.dstNzNStride = 1;
        uint32_t copiedScaleGroupCount = 0;
        while (copiedScaleGroupCount < scaleGroupCount) {
            const uint64_t logicalScaleGroupIndex = static_cast<uint64_t>(runInfo.s2Idx) / MXFP_DIVISOR_SIZE +
                copiedScaleGroupCount;
            const uint32_t pageScaleGroupOffset = logicalScaleGroupIndex % scaleGroupsPerPage;
            const uint32_t pageScaleGroupCount = AttentionCommon::Min(scaleGroupCount - copiedScaleGroupCount,
                scaleGroupsPerPage - pageScaleGroupOffset);
            const uint64_t physicalPageIndex = blockTableGm_.GetValue(
                static_cast<uint64_t>(runInfo.bIdx) * constInfo_.maxBlockNumPerBatch +
                    logicalScaleGroupIndex / scaleGroupsPerPage);
            const uint64_t valueScaleGmPairOffset = physicalPageIndex * pagePairStride +
                pageScaleGroupOffset * scaleGroupPairStride +
                                    static_cast<uint64_t>(runInfo.n2Idx) * dVBaseSize;
            if (scaleGroupPairStride <= 65535U) {
                valueScaleCopyParams.nValue = pageScaleGroupCount;
                valueScaleCopyParams.srcDValue = static_cast<uint16_t>(scaleGroupPairStride);
                DataCopy<uint16_t>(valueScaleL1Pairs[copiedScaleGroupCount * 16U],
                    valueScaleGm_[valueScaleGmPairOffset], valueScaleCopyParams);
            } else {
                valueScaleCopyParams.nValue = 1;
                valueScaleCopyParams.srcDValue = dVBaseSize;
                for (uint32_t scaleGroupIndex = 0; scaleGroupIndex < pageScaleGroupCount; ++scaleGroupIndex) {
                    DataCopy<uint16_t>(valueScaleL1Pairs[(copiedScaleGroupCount + scaleGroupIndex) * 16U],
                        valueScaleGm_[valueScaleGmPairOffset + scaleGroupIndex * scaleGroupPairStride],
                        valueScaleCopyParams);
                }
            }
            copiedScaleGroupCount += pageScaleGroupCount;
        }
    }

};


template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool enableQf322F32Pre = false>
class QuantFlashAttnBlockCubeMxfp4Dummy {
public:
    static constexpr bool ENABLE_QF322F32_PRE = enableQf322F32Pre;
    static constexpr uint32_t mBaseSize = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t s2BaseSize = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t dBaseSize = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t dVBaseSize = static_cast<uint32_t>(dVTemplateType);
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = KvLayoutType > 0;
    static constexpr bool USE_DN = useDn;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using ConstInfoX = ConstInfo_t;
    using RunInfoX = Mxfp4RunInfo;
    __aicore__ inline QuantFlashAttnBlockCubeMxfp4Dummy(ConstInfoX&) {}
};

} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP4_H_
