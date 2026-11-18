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
 * \file quant_flash_attn_kernel_mxfp4.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_MXFP4_H_
#define QUANT_FLASH_ATTN_KERNEL_MXFP4_H_

#include <cstdint>
#include <type_traits>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_mxfp4_common.h"
#include "quant_flash_attn_block_cube_mxfp4.h"
#include "quant_flash_attn_block_vec_mxfp4.h"
#include "quant_flash_attn_block_vec_flashdecode.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_tiling_data.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memcopy/parser.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {

__aicore__ inline int64_t ClipMxfp4SInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class QuantFlashAttnKernelMxfp4 {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;
    static constexpr uint32_t s2SplitSize = 128U;

    static constexpr bool USE_DN = CubeBlockType::USE_DN; //DN 模式
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;
    static_assert(CubeBlockType::ENABLE_QF322F32_PRE == VecFaBlockType::ENABLE_QF322F32_PRE,
                  "MXFP4 Cube and Vector must agree on MM1 Fixpipe scaling");

    static constexpr uint32_t PRELOAD_N = 20; // C2落后C1 20 轮
    static constexpr bool GROUPED_PV = Mxfp4GroupedPvConfig::IsEnabled(dBaseSize, s2BaseSize);
    static constexpr uint32_t GROUP_N = GROUPED_PV ? Mxfp4GroupedPvConfig::TASKS : 16U;
    static constexpr uint32_t DELAY_P_SCALE_N = 3; // P scale 在每个group的16个task的C1完成后，再过3轮，对这16个task进行一次pscale的update
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1; // 存储任务信息的数组大小

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION; // 是否使用paged attn，true
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE; // 是否使用flash decoding
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT; // TND 为主
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT; // PA_BBND 和 PA_NZ
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>(); // TND的情况下，是累积的形式，即 len[i] = act[i+1]-act[i]
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>(); // paged attn的情况下，不是累积形式，即len[i] = act[i]

    using INPUT_T = typename CubeBlockType::Q_T; //Q 输入的数据类型
    using T = typename CubeBlockType::MM_T; // Cube block 定义的矩阵计算结果类型
    using OUT_T = typename VecFaBlockType::OUT_T; // 最终输出的数据类型

    static constexpr uint32_t FP4_NUM_PER_32B = 64; // 不是 32 / sizeof(INPUT_T)，因为INPUT_T是fp4x2_e2m1_t


    // 输入tensor的GM的地址
    GlobalTensor<int32_t> cuSeqLensGmQ_;
    GlobalTensor<int32_t> cuSeqLensGmKv_;
    GlobalTensor<int32_t> seqUsedGmQ_;
    GlobalTensor<int32_t> seqUsedGmKv_;
    GlobalTensor<uint32_t> faMetaDataGm_;
    GlobalTensor<uint32_t> fdMetaDataGm_;
    GlobalTensor<float> softmaxLseGm_;
    __gm__ uint8_t* keyPtr_ = nullptr;
    __gm__ uint8_t* valuePtr_ = nullptr;

    // constinfo 和 3个主要成员
    ConstInfo_t constInfo_ {};
    TPipe* pipe_ = nullptr;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    uint32_t createdTaskCount_ = 0U; // 当前 section 内已创建的有效任务数
    uint64_t groupSequence_ = 0; // PV 槽和 Pscale 信号在 section 间连续编号

    // tile调度的相关参数
    uint64_t actSeqLensKv_ = 0; // 当前 batch 实际参与计算的 KV token 数
    uint64_t actSeqLensQ_ = 0; // 当前 batch 实际参与计算的 Q token 数
    uint32_t curS2Start_ = 0; // 当前 M 块在本核需要处理的第一个 S2 块编号。结合 metadata 分片边界和 mask 范围计算
    uint32_t curS2End_ = 0; // 当前 M 块在本核需要处理的S2块的结束位置，不包括该位置，区间为 [curS2Start_, curS2End_)
    uint32_t prevBIdx_ = 0; // 上一个batch编号，用于判断是否到了一个新的batch，新的batch需要读取新的q和kv的长度
    uint32_t prevBN2Idx_ = 0; // 上一个BN2的合并的编号，用于判断是否到了一个新的head块，和prevGS1Idx_一起判断是否切到新的M
    uint32_t prevGS1Idx_ = 0; // 上一个M块的编号，需要切换到新的M块
    uint32_t mloop_ = 0; // 当前 section 内用于区分不同 M 块计算状态的编号。同一个 M 块遍历多个 S2 task 时保持相同；源码在 s2Cur == curS2Start_ 时递增。Vector 用它选择 Softmax 的 max/sum 状态槽
    bool headS2Split_ = 0; // 判断当前M开始的S2块是否是第一个S2块
    bool tailS2Split_ = 0; // 判断当前M结束的S2块是否是最后一个S2块，用于确定fd的写入

    // metadata 的信息
    uint32_t sectionNum_; // 总的section数
    uint32_t metadataAicNum_; // metadata 为每个 section 存放的 AIC 记录数，用于定位 FA 记录
    uint32_t metadataAivNum_; // metadata 为每个 section 存放的 AIV 记录数，用于定位 FD 记录
    // fa metadata，基本块的范围，表示起点和终点
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_; // 当前核第一个需要保存的 FA 分片结果在 workspace 中的槽位索引，不是字节地址。后续根据布局换算实际地址
    // fd metadata
    FDparamsX fdParams_ {}; //fd合并的相关参数

    // q和kv的长度的解析
    typename std::conditional<(LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                              ActualSeqLensParser<Q_MODE, int32_t>>::type qCuSeqLensParser_;
    typename std::conditional<(!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE, int32_t>>::type kvCuSeqLensParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser_;

    // ===========================================function========================================================================
    __aicore__ inline QuantFlashAttnKernelMxfp4(): cubeBlock_(constInfo_), vecFaBlock_(constInfo_), vecFdBlock_(constInfo_){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* cuSeqLensQ,
                                __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* pScale, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* attnOut, __gm__ uint8_t* workspace, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedKv,
                                const QuantFlashAttnTilingData& tiling, TPipe* tPipe)
    {
        this->pipe_ = tPipe;
        InitConstInfo(tiling);
        constInfo_.needInitOutput = ((__gm__ uint32_t*)metadata)[QFA_HEAD_NEED_INIT_OUTPUT_INDEX] != 0;
        constInfo_.enableFlashDecode = static_cast<bool>(((__gm__ uint32_t*)metadata)[1]);

        keyPtr_ = key;
        valuePtr_ = value;

        // 当前 LAYOUT_Q 只为 LayOutTypeEnum::LAYOUT_TND
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ_.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensQ, constInfo_.cuSeqLensQSize + 1);
            seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo_.seqUsedQSize);
        } else {
            seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo_.seqUsedQSize);
        }
        // 当前 LAYOUT_KV 只为 PA
        if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmKv_.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensKv, constInfo_.cuSeqLensKVSize + 1);
            seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo_.seqUsedKvSize);
        } else {
            seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo_.seqUsedKvSize);
        }

        sectionNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadataAicNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIC_NUM_INDEX];
        metadataAivNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIV_NUM_INDEX];

        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET),
                                      sectionNum_ * metadataAicNum_ * METADATA_STRIDE);
        InitQCuSeqLensParser(cuSeqLensQ, sequsedQ);
        InitKvCuSeqLensParser(cuSeqLensKv, sequsedKv);

        if ASCEND_IS_AIV {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vecFaBlock_.InitVecBlock(tPipe, cuSeqLensQ, cuSeqLensKv, pScale, attnMask, softmaxLse, attnOut,
                                         workspace);
                vecFaBlock_.SetCuSeqLensParser(qCuSeqLensParser_);
            } else {
                vecFaBlock_.InitVecBlock(tPipe, sequsedQ, sequsedKv, pScale, attnMask, softmaxLse, attnOut, workspace);
                vecFaBlock_.SetCuSeqLensParser(qSeqUsedParser_);
            }
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cubeBlock_.InitCubeBlock(tPipe, query, key, value, blockTable, dequantScaleQuery,
                                         dequantScaleKey, dequantScaleValue, qCuSeqLensParser_, kvCuSeqLensParser_);
            } else {
                cubeBlock_.InitCubeBlock(tPipe, query, key, value, blockTable, dequantScaleQuery,
                                         dequantScaleKey, dequantScaleValue, qSeqUsedParser_, kvSeqUsedParser_);
            }
        }
        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                if ASCEND_IS_AIV {
                    fdMetaDataGm_.SetGlobalBuffer(
                        (__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET +
                                           METADATA_STRIDE * metadataAicNum_ * sectionNum_ * sizeof(uint32_t)),
                        sectionNum_ * metadataAivNum_ * METADATA_STRIDE);
                    vecFdBlock_.InitParams();
                    vecFdBlock_.InitGlobalTensor(this->vecFaBlock_.softmaxFDMaxGm_, this->vecFaBlock_.softmaxFDSumGm_,
                                                 this->vecFaBlock_.accumOutGm_, this->vecFaBlock_.attentionOutGm_,
                                                 keyPtr_);
                    vecFdBlock_.InitPScaleValue(this->vecFaBlock_.pScaleValue_);
                    if (constInfo_.isSoftmaxLseEnable) {
                        softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
                        vecFdBlock_.InitSoftmaxLseGm(softmaxLseGm_);
                    }
                    if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                        vecFdBlock_.SetCuSeqLensParsers(qCuSeqLensParser_, kvCuSeqLensParser_);
                    } else {
                        vecFdBlock_.SetCuSeqLensParsers(qSeqUsedParser_, kvSeqUsedParser_);
                    }
                }
            }
        }

    }
    __aicore__ inline void InitConstInfo(const QuantFlashAttnTilingData& tiling)
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx(); // 对于aic来说获取的自己的编号
        } else {
            constInfo_.aivIdx = GetBlockIdx(); // 对于aiv来说也是获取的自己的编号
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum(); // aiv直接的编号除以每个aic中aiv的数量就是其对应的aic的编号
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }

        const QuantFlashAttnBaseParams& qfaBaseParams = tiling.baseTiling.quantFlashAttnBaseParams;
        const QuantFlashAttnAttenMaskParams& qfaAttenMaskParams = tiling.baseTiling.quantFlashAttnAttenMaskParams;
        const QuantFlashAttnPageAttentionParams& qfaPageAttentionParams = tiling.baseTiling.quantFlashAttnPageAttentionParams;
        const QuantFlashAttnWorkspaceParams& qfaWorkspaceParams = tiling.baseTiling.quantFlashAttnWorkspaceParams;

        constInfo_.bSize = qfaBaseParams.bSize;
        constInfo_.t1Size = qfaBaseParams.t1Size;
        constInfo_.t2Size = qfaBaseParams.t2Size;
        constInfo_.n2Size = qfaBaseParams.n2Size;
        constInfo_.gSize = qfaBaseParams.gSize;
        constInfo_.s1Size = qfaBaseParams.s1Size;
        constInfo_.s2Size = qfaBaseParams.s2Size;
        constInfo_.dSize = qfaBaseParams.dSize;
        constInfo_.dSizeV = qfaBaseParams.dSizeV;
        if constexpr (USE_DN) { // prefill不合轴
            constInfo_.realN2Size = constInfo_.n2Size * constInfo_.gSize;
            constInfo_.realGSize = 1;
        } else { // decode合轴
            constInfo_.realN2Size = constInfo_.n2Size;
            constInfo_.realGSize = constInfo_.gSize;
        }
        constInfo_.cuSeqLensQSize = qfaBaseParams.cuSeqLensQSize;
        constInfo_.cuSeqLensKVSize = qfaBaseParams.cuSeqLensKVSize;
        constInfo_.seqUsedQSize = qfaBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = qfaBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = static_cast<float>(qfaBaseParams.scaleValue);
        constInfo_.coreNum = qfaBaseParams.coreNum;
        constInfo_.outputLayout = static_cast<FA_LAYOUT>(qfaBaseParams.outputLayout);
        constInfo_.sparseMode =
            qfaAttenMaskParams.sparseMode;
        constInfo_.preTokens = qfaAttenMaskParams.winLefts;
        constInfo_.nextTokens = qfaAttenMaskParams.winRights;
        constInfo_.attenMaskBatch = qfaAttenMaskParams.attenMaskBatch;
        constInfo_.attenMaskS1Size = qfaAttenMaskParams.attenMaskS1Size;
        constInfo_.attenMaskS2Size = qfaAttenMaskParams.attenMaskS2Size;

        constInfo_.accumOutSize = qfaWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = qfaWorkspaceParams.logSumExpSize;

        // strides
        constInfo_.keyStrides.bnStride = qfaBaseParams.keyStrides.bnStride; // 在PA_BBND [blocknum, blocksize, headnum, D]下，bnStride = blocksize*headnum*D
        constInfo_.keyStrides.n2Stride = qfaBaseParams.keyStrides.n2Stride; // 在PA_BBND [blocknum, blocksize, headnum, D]下，n2Stride = headnum*D，其实是两个token之间的距离，不是head
        constInfo_.valueStrides.bnStride = qfaBaseParams.valueStrides.bnStride;
        constInfo_.valueStrides.n2Stride = qfaBaseParams.valueStrides.n2Stride;
        constInfo_.kDescaleStrides.bnStride = qfaBaseParams.kDescaleStrides.bnStride;
        constInfo_.kDescaleStrides.n2Stride = qfaBaseParams.kDescaleStrides.n2Stride;
        constInfo_.vDescaleStrides.bnStride = qfaBaseParams.vDescaleStrides.bnStride;
        constInfo_.vDescaleStrides.n2Stride = qfaBaseParams.vDescaleStrides.n2Stride;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo_.maxBlockNumPerBatch = qfaPageAttentionParams.maxBlockNumPerBatch;
            constInfo_.blockSize = qfaPageAttentionParams.blockSize;
            constInfo_.paLayoutType = qfaPageAttentionParams.paLayoutType;
        }

        // LSE
        constInfo_.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;

        constInfo_.dBasicBlock = Align64Func((uint16_t)constInfo_.dSizeV);

    }


    __aicore__ inline void InitQCuSeqLensParser(__gm__ uint8_t* cuSeqLensQPtr, __gm__ uint8_t* sequsedQPtr)
    {
        // 当前 LAYOUT_Q 只能为 LayOutTypeEnum::LAYOUT_TND
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser_.Init(cuSeqLensQPtr, constInfo_.cuSeqLensQSize + 1, sequsedQPtr, constInfo_.seqUsedQSize);
        } else {
            qSeqUsedParser_.Init(seqUsedGmQ_, constInfo_.seqUsedQSize, constInfo_.s1Size);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser(__gm__ uint8_t* cuSeqLensKvPtr, __gm__ uint8_t* sequsedKvPtr)
    {
        // 当前PAGE_ATTENTION为true，并且LAYOUT_KV一定不是LayOutTypeEnum::LAYOUT_TND，走else分支
        if constexpr (!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser_.Init(cuSeqLensKvPtr, constInfo_.cuSeqLensKVSize + 1, sequsedKvPtr,
                                    constInfo_.seqUsedKvSize);
        } else if constexpr (PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser_.Init(sequsedKvPtr, constInfo_.seqUsedKvSize, constInfo_.s2Size);
        } else {
            kvSeqUsedParser_.Init(seqUsedGmKv_, constInfo_.seqUsedKvSize, constInfo_.s2Size);
        }
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAicNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAivNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, QFA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    struct TaskCursor {
        uint32_t batchHead;
        uint32_t queryBlock;
        uint32_t kvBlock;
    };

    __aicore__ inline TaskCursor BeginFaSection(uint32_t sectionIdx)
    {
        GetFASectionInfo(sectionIdx);
        prevBN2Idx_ = bN2Start_;
        prevGS1Idx_ = gS1OStart_;
        headS2Split_ = false;
        tailS2Split_ = false;
        mloop_ = 0;
        createdTaskCount_ = 0;
        return {bN2Start_, gS1OStart_, s2OStart_};
    }

    __aicore__ inline bool ReadNextTask(TaskCursor& cursor, Mxfp4RunInfo& task)
    {
        while (ShouldDispatchTask(cursor.batchHead, cursor.queryBlock, cursor.kvBlock)) {
            const auto mode = GetTaskDealMode(cursor.batchHead, cursor.queryBlock, cursor.kvBlock);
            if (mode == TASK_DEAL_MODE::CREATE_TASK) {
                task = Mxfp4RunInfo {};
                CalcParams(createdTaskCount_++, cursor.batchHead, cursor.queryBlock, cursor.kvBlock, task);
                UpdateAxisInfo(mode, cursor.batchHead, cursor.queryBlock, cursor.kvBlock);
                return true;
            }
            UpdateAxisInfo(mode, cursor.batchHead, cursor.queryBlock, cursor.kvBlock);
        }
        return false;
    }

    static __aicore__ inline void SetGroupTaskSlots(Mxfp4RunInfo& task, uint32_t index, uint32_t sectionGroup,
        uint64_t groupId)
    {
        task.groupTaskIndex = index;
        // During startup both groups alternate scores in task order. In steady state
        // the first score uses the slot opposite to the PV of the group two back.
        task.scoreSlot = sectionGroup < Mxfp4GroupedPvConfig::GROUP_SLOTS ? task.loop % 2 :
            (1U - groupId % 2 + index) % 2;
    }

    __aicore__ inline uint32_t IssueGroupC1V1(TaskCursor& cursor, Mxfp4RunInfo& task, Mxfp4RunInfo* cubeTasks,
        uint32_t sectionGroup, uint64_t groupId, bool firstC1Issued)
    {
        uint32_t count = 0;
        do {
            SetGroupTaskSlots(task, count, sectionGroup, groupId);
            if ASCEND_IS_AIC {
                if (count != 0 || !firstC1Issued) {
                    cubeBlock_.ComputeBmm1(task);
                }
                cubeTasks[count] = task;
            } else {
                vecFaBlock_.ComputeVec1(task);
            }
            ++count;
            if (task.isGroupLast) {
                return count;
            }
        } while (count < GROUP_N && ReadNextTask(cursor, task));
        ASSERT(false); // Every valid Q block/KV split ends a group.
        return count;
    }

    __aicore__ inline void RunGroupedPvCube(TaskCursor& cursor)
    {
        constexpr uint32_t groupSlots = Mxfp4GroupedPvConfig::GROUP_SLOTS;
        Mxfp4RunInfo tasks[groupSlots][GROUP_N] {};
        uint32_t counts[groupSlots] {};
        Mxfp4RunInfo nextTask;
        for (uint32_t g = 0; g < groupSlots && ReadNextTask(cursor, nextTask); ++g) {
            counts[g] = IssueGroupC1V1(cursor, nextTask, tasks[g], g, groupSequence_ + g, false);
        }

        for (uint32_t old = 0; counts[old % groupSlots] != 0; ++old, ++groupSequence_) {
            const uint32_t slot = old % groupSlots;
            // PV can accumulate while AIV consumes scores already in UB. The
            // final PV Fixpipe waits for both AIVs to release its target slot.
            for (uint32_t t = 0; t < counts[slot]; ++t) {
                cubeBlock_.ComputeBmm2(tasks[slot][t]);
            }

            // The early task has its own descriptor. Cube manages L0C hazards:
            // D256 waits for the PV Fixpipe on both QK slots, then reuses one.
            const bool hasNext = ReadNextTask(cursor, nextTask);
            const uint32_t nextGroup = old + groupSlots;
            const uint64_t nextGroupId = groupSequence_ + groupSlots;
            if (hasNext) {
                SetGroupTaskSlots(nextTask, 0, nextGroup, nextGroupId);
                cubeBlock_.ComputeBmm1(nextTask);
            }
            cubeBlock_.WaitGroupRetired(groupSequence_);
            counts[slot] = 0;
            if (hasNext) {
                cubeBlock_.AllowGroupV1();
                counts[slot] = IssueGroupC1V1(cursor, nextTask, tasks[slot], nextGroup, nextGroupId, true);
            }
        }
    }

    __aicore__ inline void RunGroupedPvVector(TaskCursor& cursor)
    {
        constexpr uint32_t groupSlots = Mxfp4GroupedPvConfig::GROUP_SLOTS;
        Mxfp4RunInfo tails[groupSlots] {};
        Mxfp4RunInfo task;
        uint32_t group = 0;
        while (ReadNextTask(cursor, task)) {
            const uint32_t slot = group % groupSlots;
            if (group >= groupSlots) {
                vecFaBlock_.ComputeVec2(tails[slot]);
                vecFaBlock_.WaitGroupV1Allowed();
            }
            IssueGroupC1V1(cursor, task, nullptr, group, groupSequence_ + group, false);
            vecFaBlock_.UpdatePScale(task);
            tails[slot] = task;
            ++group;
        }

        for (uint32_t retired = group > groupSlots ? group - groupSlots : 0; retired < group; ++retired) {
            vecFaBlock_.ComputeVec2(tails[retired % groupSlots]);
        }
        groupSequence_ += group;
    }

    __aicore__ inline void FlashAttentionGroupedPv(uint32_t sectionIdx)
    {
        auto cursor = BeginFaSection(sectionIdx);
        if ASCEND_IS_AIC {
            RunGroupedPvCube(cursor);
        } else {
            RunGroupedPvVector(cursor);
        }
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        auto cursor = BeginFaSection(sectionIdx);
        Mxfp4RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] {};
        uint32_t& bN2Cur = cursor.batchHead;
        uint32_t& gS1Cur = cursor.queryBlock;
        uint32_t& s2Cur = cursor.kvBlock;
        uint64_t pipelineTick = 0;
        uint32_t pendingTasks = 0;
        while (pendingTasks != 0 || ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur)) {
            if (ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur)) {
                const TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode != TASK_DEAL_MODE::CREATE_TASK) {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
                Mxfp4RunInfo& currentTask = taskRunInfo[pipelineTick % PRELOAD_TASK_CACHE_SIZE];
                ASSERT(!currentTask.isValid);
                currentTask = Mxfp4RunInfo {};
                CalcParams(createdTaskCount_, bN2Cur, gS1Cur, s2Cur, currentTask);
                currentTask.isValid = true;
                ++createdTaskCount_;
                ++pendingTasks;
                UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
            }
            if (pendingTasks != 0) {
                const bool retiresTask = pipelineTick >= PRELOAD_N &&
                    taskRunInfo[(pipelineTick - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE].isValid;
                ExecuteTask(pipelineTick, taskRunInfo);
                ++pipelineTick;
                if (retiresTask) {
                    --pendingTasks;
                }
            }
            // Keep issuing across Q boundaries. Only the end of this section drains
            // the pipeline; skipped coordinates do not consume task or P slots.
        }
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, Mxfp4RunInfo& info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / (constInfo_.realN2Size); // batch idx
        info.n2Idx = (bN2Cur / (constInfo_.realN2Size / constInfo_.n2Size)) % constInfo_.n2Size; // 实际的kv head的idx
        info.realN2Idx = bN2Cur % constInfo_.realN2Size; // 实际的head idx
        info.gS1Idx = gS1Cur * mBaseSize; // M的行idx
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo_.realGSize; // query 的行
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ_;
        }
        info.s2Idx = s2Cur * s2BaseSize; // kv token的起点
        info.actS1Size = actSeqLensQ_;  // 实际的 query 的长度
        info.actS2Size = actSeqLensKv_; // 实际的 kv 的长度

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo_.realGSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) { // 实际的计算m行，去掉padding的行
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) { // // 实际的计算s2行，去掉padding的行
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }

        bool splitCurBlock = false;
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, FP4_NUM_PER_32B);

        info.isChangeBatch = false;

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = (loop == 0) || (s2Cur == curS2Start_);
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = s2Cur + 1 == curS2End_;
        info.isGroupLast = info.isLastS2Loop || ((s2Cur - curS2Start_ + 1U) % GROUP_N == 0U);

        info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
        // DN 固定 Q64；ND 按完整 Q16 scale 块均分给两个 AIV。
        info.vecMbaseIdx = 0;
        info.actVecMSize = info.actMSize;
        if ASCEND_IS_AIV {
            const uint32_t rowsPerAiv = USE_DN ? 64U : info.actMSizeAlign32 / CV_RATIO;
            info.vecMbaseIdx = constInfo_.subBlockIdx * rowsPerAiv;
            info.actVecMSize = info.actMSize > info.vecMbaseIdx ?
                AttentionCommon::Min(rowsPerAiv, info.actMSize - info.vecMbaseIdx) : 0U;
        }
        /*
        所有的切分只会发生在当前core处理的第一个head的第一个M行，和最后一个head的最后一个行，中间的任务都是完整的M块
        Q块A：            KV块4、5、6、7  ← 首块，只算后半段
        Q块B：KV块0、1、2、3、4、5、6、7  ← 中间块，完整计算
        Q块C：KV块0、1、2                 ← 尾块，只算前半段
        */
        if ((bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_)) {
            // 所有任务属于同一个S1G，因为当前的区间是左闭右开，所以当前Q行不会被当前core完整处理，因为
            // 如果完整处理，gS1OStart_ 不应该等于 gS1OEnd_
            // 同时，属于同一个行，只会写到一个faTmpOutWsPos中，不需要两个faTmpOutWsPos
            // 只有不同的Q块，才会需要两个faTmpOutWsPos
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, Mxfp4RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        Mxfp4RunInfo& runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务

        if (loop >= PRELOAD_N) { // 可以执行C2，然后在一组C2执行完成后，执行V2
            Mxfp4RunInfo& runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    cubeBlock_.ComputeBmm2(runInfoNegN); // 子循环和同步都在cube内部处理
                } else {
                   if (unlikely(runInfoNegN.isGroupLast)) { // 小组最后一个C2跑完，进行一次V2
                        vecFaBlock_.ComputeVec2(runInfoNegN);
                    }
                }
                // 对 Cube 来说，C2完成，这个任务就结束了
                // 对 vector 来说，如果不是一个组末的task，不需要执行V2，也结束了，只有最后一个V2需要执行
                runInfoNegN.isValid = false;
            }
        }

        if (loop >= DELAY_P_SCALE_N) { // 可以一组执行完V1的任务，进行一次P scale update
            Mxfp4RunInfo& runInfoNegScaleN = taskRunInfo[(loop - DELAY_P_SCALE_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfoNegScaleN.isValid && unlikely(runInfoNegScaleN.isGroupLast)) {
                if ASCEND_IS_AIV {
                    vecFaBlock_.UpdatePScale(runInfoNegScaleN);
                }
            }
        }

        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                cubeBlock_.ComputeBmm1(runInfo0); // 子循环和同步都在内部处理
            } else {
                vecFaBlock_.ComputeVec1(runInfo0); // 子循环和同步都在内部处理
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (bN2Cur != bN2End_) { // head 维度还没有到最后一个head，肯定有任务
            return bN2Cur < bN2End_;
        }
        if (gS1Cur != gS1OEnd_) { // 到最后一个head，这个head还没有到最后一个M块，肯定有任务
            return gS1Cur < gS1OEnd_;
        }
        return s2Cur < s2OEnd_; // 到最后一个head的最后一个M块，是否到了最后一个kv块

    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t& bN2Cur, uint32_t& gS1Cur,
                                          uint32_t& s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        // 尚未到达有效区间, 快进s2Cur到curS2Start_, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        }
        // 如果是不跳过剩余S2的情况，在s2没处理完时，继续下一个s2块
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前S2未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        // 当前BN2未处理完，更新M块行号
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完，跳到下一个head
        gS1Cur = 0;
        bN2Cur++;
    }


    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_); // 首个任务
        uint32_t bIdx = bN2Cur / constInfo_.realN2Size;
        if (isFirstTask || prevBIdx_ != bIdx) { // 换了一个batch，更新q和kv的长度
            prevBIdx_ = bIdx;
            if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensKv_ = kvCuSeqLensParser_.GetActualSeqLength(bIdx);
            } else {
                actSeqLensKv_ = kvSeqUsedParser_.GetActualSeqLength(bIdx);
            }
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensQ_ = qCuSeqLensParser_.GetActualSeqLength(bIdx);
            } else {
                actSeqLensQ_ = qSeqUsedParser_.GetActualSeqLength(bIdx);
            }
        }
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) { // batch 空的任务
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO; // 第一个空任务，需要清零输出
            }
            return TASK_DEAL_MODE::SKIP_ZERO; // 跳过
        }

        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算 S2 的起点和终点
        // 有mask和没有mask的计算方式不同
        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        // S2有效块区间为[curS2Start_, curS2End_), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start_继续计算, 不能跳行 (BAND等sparse模式curS2Start_常>0)
        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start_>=curS2End_)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start_) {
            mloop_++; // 推进一个新的m行
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        // 只有第一个开始的M行和最后一个结束的M行，s2的开始和结束才可能不是0和最后一个块
        // 中间的M行一定会处理一整行kv
        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split_ = s2OStart_ != 0U;
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split_ = s2OEnd_ != 0U;
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 计算左右的边界，对一个q token，其可见范围为 [q_idx-preTokenLeftUp, q_idx+nextTokenLeftUp]，q_idx的意思是其在当前输入的query token 中的下标
        // 对于我们常用的右下角causcal，nextTokenLeftUp = actSeqLensKv_ - actSeqLensQ_
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);

        // 计算当前M块的第一个token行和最后一个token行，包括合轴的行
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ_) * static_cast<int64_t>(constInfo_.realGSize)) -
            1;

        // 实际的 token 编号，去掉合轴的行
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo_.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo_.realGSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ_) ==
                s1GLastToken / static_cast<int64_t>(actSeqLensQ_)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ_);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ_);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
            }
        }

        // 3. 变成块编号
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp; // 是左右闭区间
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipMxfp4SInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipMxfp4SInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U; // +1 变成左右开区间

        // 4. 计算 curS2Start_, curS2End_
        curS2Start_ = s2StartWithSparse;
        curS2End_ = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            headS2Split_ = s2OStart_ > s2StartWithSparse ? true : false; // 如果不是起点，就是切分了
            curS2Start_ = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            tailS2Split_ = s2OEnd_ > 0U ? true : false; // 如果s2不是终点，s2就被切分了，当前需要存储到workspace
            curS2End_ = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t& preTokenLeftUp,
                                                 int64_t& nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode); // 更新preTokenLeftUp和nextTokenLeftUp

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL ||
            constInfo_.sparseMode == fa_base_vector::TREE) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        // FA has retired every PV consumer. Reuse PV/history O, preserving score
        // buffers because AIC can start the next section before AIV finishes FD.
        static_assert(VecFdBlockType::GetFdTotalUbSize() <= VecFaBlockType::GetFdReuseUbSize(),
                      "FD exceeds reusable PV/history UB region");
        if (!constInfo_.enableFlashDecode) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock_.template InitBuffers<VecFaBlockType::GetFdReuseUbOffset()>();
        AscendC::ICachePreLoad(2);
        SyncAll();
        vecFdBlock_.template FlashDecode<VecFaBlockType::QUANT_SCHEMA_VALUE>(fdParams_);
        SyncAll(); // 多section场景: 保证所有核FD读workspace完成后, 才能进入下一section FA覆写
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_START_INDEX, sectionIdx));
    }


    __aicore__ inline void Process()
    {
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.InitBuffers();
                vecFaBlock_.AllocEventID();
            } else {
                cubeBlock_.InitBuffers();
                cubeBlock_.AllocEventID();
            }
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                if constexpr (GROUPED_PV) {
                    FlashAttentionGroupedPv(sectionIdx);
                } else {
                    FlashAttention(sectionIdx);
                }
            }
            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(sectionIdx);
                }
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
            } else {
                cubeBlock_.FreeEventID();
            }
        }



    }




};

} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_KERNEL_MXFP4_H_
