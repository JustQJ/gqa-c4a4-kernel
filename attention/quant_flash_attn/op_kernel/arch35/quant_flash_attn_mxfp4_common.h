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
 * \file quant_flash_attn_mxfp4_common.h
 * \brief Task descriptors shared by the MXFP4 controller and its Cube/Vector blocks.
 */
#ifndef QUANT_FLASH_ATTN_MXFP4_COMMON_H_
#define QUANT_FLASH_ATTN_MXFP4_COMMON_H_

#include <cstdint>
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/buffers_policy.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/vector_common.h"

namespace BaseApi {
// Values match the public quant_schema attribute. Q/K/V encoding is unchanged.
enum class QuantSchema : uint8_t {
    OCP_QUANT = 0,
    CX_QUANT = 1,
    CX_DELTA_QUANT = 2,
};

using fa_base_matmul::Buffer;
using fa_base_matmul::BufferManager;
using fa_base_matmul::BufferType;
using fa_base_matmul::SyncType;
struct Mxfp4GroupedPvConfig {
    static constexpr uint32_t TASKS = 2;
    static constexpr uint32_t GROUP_SLOTS = 2;
    static constexpr uint32_t P_SLOTS = TASKS * GROUP_SLOTS;
    static constexpr uint32_t SCORE_N = 256;

    static constexpr bool IsEnabled(uint32_t headDim, uint32_t taskN)
    {
        return headDim == 256 && taskN == SCORE_N;
    }
};

struct Mxfp4RunInfo : RunInfoX {
    bool isGroupLast = false;
    uint8_t groupTaskIndex = 0;
    uint8_t scoreSlot = 0;
    // Assigned by AIV ComputeVec1 and retained in the controller's task ring.
    // Independent of loop/16: a Q block/split may end with a short group.
    // Cube does not consume this AIV-local field.
    uint64_t v1GroupId = ~uint64_t{0};
};

struct Mxfp4SubRunInfo {
    uint64_t sequence = 0;     // 连续的 QK 子块编号
    uint32_t subLoop = 0;      // 当前是 task 内的第几个子块
    uint32_t scoreSlot = 0;    // 使用哪个 score 缓冲槽
    uint32_t s2Offset = 0;     // 子块在完整 task 内的 KV 方向偏移
    uint32_t validN = 0;       // 子块实际有效的 KV token 数
    uint32_t alignedN = 0;     // 对齐后的 N 长度
    bool isFirst = false;     // 是否为 task 的首个／最后一个子块
    bool isLast = false;
};

// common class cv buffers
template <uint32_t PRELOAD_N = 20, bool useDn = false,
        S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
        S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
        DTemplateType dTemplateType = DTemplateType::Aligned128,
        DTemplateType dVTemplateType = DTemplateType::Aligned128>
class CVCrossBuffer {
public:

    // 核间同步事件
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {0, 1};
    static constexpr uint16_t CROSS_CORE_SYNC_C1_V1[2] = {2, 3};
    // Up to twenty single-task groups can be in flight across Q blocks. A
    // single counting flag can accumulate seventeen ready groups before C2;
    // alternate two channels so each counter stays below the hardware limit 15.
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2[2] = {4, 7};
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_GROUP_V1_ALLOWED = 9;

    static constexpr bool USE_DN = useDn;
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr bool GROUPED_PV = Mxfp4GroupedPvConfig::IsEnabled(dBaseSize, s2BaseSize);
    static constexpr uint32_t s2SplitSize = GROUPED_PV ? Mxfp4GroupedPvConfig::SCORE_N : 128U;

    static constexpr uint32_t SCORE_SLOT_COUNT = 2;

    static __aicore__ inline bool CanScaleInFixpipe(float scale)
    {
        const uint32_t bits = *reinterpret_cast<uint32_t*>(&scale);
        // QF322F32_PRE discards scale mantissa bits 12:0. Only use it when
        // those bits are already zero; arbitrary API scales remain in FP32 VF.
        return scale > 0.0F && bits < 0x7f800000U && (bits & 0x1fffU) == 0;
    }


    // L1 buffer
    Buffer<BufferType::L1, SyncType::NO_SYNC> l1PBuffers_; // 20 个 task的p
    Buffer<BufferType::L1, SyncType::NO_SYNC> l1PScaleBuffers_; // 20 个 task的p scale
    static constexpr uint32_t l1PBufferBytes_ = mBaseSize * s2BaseSize / 2;
    // 比 mBaseSize * s2BaseSize /32 多了 mBaseSize * 2，按 16 * 64 的小块，每个小块有 16 * 2 个scale，对应32字节。
    // 把16个query对应的所有一行s2BaseSize有s2BaseSize/64个块，作为一个组，在这个组后面补一个 16 * 2 个scale作为padding，那么16个query行就有 (s2BaseSize/64 + 1) 个小块。
    // 同时，一共有  (mBaseSize / 16) 个小组
    static constexpr uint32_t l1PScaleBufferBytes_ = (mBaseSize / 16) * (s2BaseSize/64 + 1) * 32;


    //UB buffer
    Buffer<BufferType::UB, SyncType::NO_SYNC> ubBmm1Buffers_; // qk
    Buffer<BufferType::UB, SyncType::NO_SYNC> ubBmm2Buffers_; // pv
    static constexpr uint32_t ubBmm1BufferBytes_ = mBaseSize / CV_RATIO * s2SplitSize * sizeof(float);
    // 对于pv结果，因为我们要使用cube计算rowsum，所以需要进行对齐
    // 对于 DN，我们计算的是(PV)^T = [D, M]，所以补齐一行为 [D+1, M]，而ND的结果是[M, D]，本来是[M, D+1]，但是为了让每行都32B对齐，所以增加7个float padding，变成 [M, D+8]
    static constexpr uint32_t ubBmm2BufferBytes_ = USE_DN? mBaseSize / CV_RATIO * (dVBaseSize + 1) * sizeof(float) : mBaseSize / CV_RATIO * (dVBaseSize + 8) * sizeof(float);


    static constexpr uint32_t GROUP_PV_SLOT_BYTES = mBaseSize / CV_RATIO * (dVBaseSize + 8) * sizeof(float);
    static constexpr uint32_t UB_SLOT_BYTES = GROUPED_PV && GROUP_PV_SLOT_BYTES > ubBmm1BufferBytes_ ?
        GROUP_PV_SLOT_BYTES : ubBmm1BufferBytes_;
    static constexpr uint32_t UB_BYTES = SCORE_SLOT_COUNT * UB_SLOT_BYTES + (GROUPED_PV ? 0 : ubBmm2BufferBytes_);

    __aicore__ inline CVCrossBuffer() {}

    __aicore__ inline void InitBuffers(BufferManager<BufferType::L1> &l1BufferManagerPtr_,
        BufferManager<BufferType::UB> &ubBufferManagerPtr_)
    {
        l1PBuffers_ = l1BufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(l1PBufferBytes_ * PRELOAD_N);
        l1PScaleBuffers_ = l1BufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(l1PScaleBufferBytes_ * PRELOAD_N);
        ubBmm1Buffers_ = ubBufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(UB_SLOT_BYTES * SCORE_SLOT_COUNT);
        if constexpr (!GROUPED_PV) {
            ubBmm2Buffers_ = ubBufferManagerPtr_.AllocBuffer<SyncType::NO_SYNC>(ubBmm2BufferBytes_);
        }
    }

    __aicore__ inline void InitSync()
    {
        if ASCEND_IS_AIV {
            if constexpr (GROUPED_PV) {
                return;
            }
            // 两个 QK 接收槽初始为空，允许 Cube 写入。
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(
                CROSS_CORE_SYNC_V1_C1[0]);
            if constexpr (SCORE_SLOT_COUNT == 2) {
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);
            }

            // PV 接收槽初始为空，允许 Cube 写入。
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(
                CROSS_CORE_SYNC_V2_C2);
        }
    }

    __aicore__ inline LocalTensor<uint8_t> GetPSlot(uint32_t slot) {
        LocalTensor<uint8_t> probabilityL1Tensor = l1PBuffers_.GetTensor<uint8_t>();
        LocalTensor<uint8_t> probabilitySlotTensor = probabilityL1Tensor[slot * l1PBufferBytes_];
        return probabilitySlotTensor;
    }

    __aicore__ inline LocalTensor<uint8_t> GetPScaleSlot(uint32_t slot) {
        LocalTensor<uint8_t> probabilityScaleL1Tensor = l1PScaleBuffers_.GetTensor<uint8_t>();
        LocalTensor<uint8_t> probabilityScaleSlotTensor = probabilityScaleL1Tensor[slot * l1PScaleBufferBytes_];
        return probabilityScaleSlotTensor;
    }

    __aicore__ inline LocalTensor<float> GetBmm1Slot(uint32_t slot) {
        LocalTensor<float> scoreUbTensor = ubBmm1Buffers_.GetTensor<float>();
        LocalTensor<float> scoreSlotTensor = scoreUbTensor[slot * (UB_SLOT_BYTES / sizeof(float))];
        return scoreSlotTensor;
    }

    __aicore__ inline LocalTensor<float> GetBmm2Slot(uint32_t slot) {
        if constexpr (GROUPED_PV) {
            return GetBmm1Slot(slot);
        }
        LocalTensor<float> pvResultUbTensor = ubBmm2Buffers_.GetTensor<float>();
        LocalTensor<float> pvResultSlotTensor = pvResultUbTensor[slot * (ubBmm2BufferBytes_ / sizeof(float))];
        return pvResultSlotTensor;
    }

    __aicore__ inline void setBmm1Buffer(uint32_t slot) {
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1[slot]);
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1[slot] + 16);
            if constexpr (GROUPED_PV) {
                ubInUse_[slot] = true;
            }
        } else {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[slot]);
        }
    }

    __aicore__ inline void waitBmm1Buffer(uint32_t slot) {
        if ASCEND_IS_AIC {
            if constexpr (GROUPED_PV) {
                if (!ubInUse_[slot]) {
                    return;
                }
                ubInUse_[slot] = false;
            }
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[slot]);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[slot] + 16);
        } else {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C1_V1[slot]);
        }
    }

    __aicore__ inline void setBmm2Buffer(uint32_t slot = 0) {
        if constexpr (GROUPED_PV) {
            setBmm1Buffer(slot);
            return;
        }
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
        } else {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
        }
    }

    __aicore__ inline void watBmm2Buffer(uint32_t slot = 0) {
        if constexpr (GROUPED_PV) {
            waitBmm1Buffer(slot);
            return;
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
        } else {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
        }
    }

    __aicore__ inline void setPscaleUpdate(uint64_t groupId) {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2[groupId % 2]);
        }
    }

    __aicore__ inline void waitPscaleUpdate(uint64_t groupId) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2[groupId % 2]);
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2[groupId % 2] + 16);
        }
    }

    __aicore__ inline void AllowGroupV1()
    {
        if ASCEND_IS_AIC {
            // Ordered on FIX after both returns of the retired group's shared UB slot.
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_GROUP_V1_ALLOWED);
            CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_GROUP_V1_ALLOWED + 16);
        }
    }

    __aicore__ inline void WaitGroupV1Allowed()
    {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_GROUP_V1_ALLOWED);
        }
    }

private:
    // Cube records which physical slots have a return pending on each AIV.
    bool ubInUse_[SCORE_SLOT_COUNT] {};
};



} // namespace BaseApi





#endif // QUANT_FLASH_ATTN_MXFP4_COMMON_H_
