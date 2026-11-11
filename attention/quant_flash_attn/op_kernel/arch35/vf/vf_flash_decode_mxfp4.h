/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QFA_VF_FLASH_DECODE_MXFP4_H_
#define QFA_VF_FLASH_DECODE_MXFP4_H_

#include "vf_softmax_fp32_mxfp4.h"

namespace Mxfp4Api {

// Every row is repeated eight times, matching the existing FD workspace ABI.
// C4 stores integer references R and raw PV; no per-split division is needed.
template <bool UseCxQuant>
__simd_vf__ inline void PrepareFlashDecodeVf(__ubuf__ float* splitReference,
    __ubuf__ float* splitSum, __ubuf__ float* splitMultiplier, __ubuf__ float* finalLse,
    __ubuf__ float* finalReference, __ubuf__ float* finalSum, uint32_t rowCount, uint32_t splitCount)
{
    uint32_t validElements = rowCount * 8;
    MaskReg rows = UpdateMask<float>(validElements);
    RegTensor<float> maximum, reference, sum, total, zero, one, invalid;
    Duplicate(zero, 0.0F);
    Duplicate(one, 1.0F);
    Duplicate(invalid, INVALID_GROUP_EXPONENT);
    Duplicate(maximum, INVALID_GROUP_EXPONENT);
    for (uint16_t split = 0; split < splitCount; ++split) {
        LoadAlign(reference, splitReference + split * rowCount * 8);
        Max(maximum, maximum, reference, rows);
    }
    Duplicate(total, 0.0F);
    for (uint16_t split = 0; split < splitCount; ++split) {
        RegTensor<float> difference, factor, fallback, biasedExponent;
        RegTensor<int32_t> exponentBits;
        MaskReg nonempty, normalPower;
        LoadAlign(reference, splitReference + split * rowCount * 8);
        LoadAlign(sum, splitSum + split * rowCount * 8);
        Compares<float, CMPMODE::GT>(nonempty, sum, 0.0F, rows);
        Sub(difference, reference, maximum, rows);
        Select(difference, difference, zero, nonempty);
        // Normal powers of two are represented exactly. Retain the existing
        // exponential behavior for the subnormal range instead of clamping it.
        Compares<float, CMPMODE::GE>(normalPower, difference, -126.0F, rows);
        Adds(biasedExponent, difference, 127.0F, rows);
        Cast<int32_t, float, MXFP4_CAST_ZERO>(exponentBits, biasedExponent, rows);
        ShiftLefts(exponentBits, exponentBits, static_cast<int16_t>(23), rows);
        Muls(fallback, difference, MXFP4_LN2, rows);
        Exp(fallback, fallback, rows);
        Select(factor, reinterpret_cast<RegTensor<float>&>(exponentBits), fallback, normalPower);
        Select(factor, factor, zero, nonempty);
        StoreAlign(splitMultiplier + split * rowCount * 8, factor, rows);
        Mul(sum, sum, factor, rows);
        Add(total, total, sum, rows);
    }
    StoreAlign(finalSum, total, rows);
    MaskReg nonemptyRows;
    RegTensor<float> safeSum, lse, naturalReference;
    RegTensor<uint32_t> infinityBits;
    Compares<float, CMPMODE::GT>(nonemptyRows, total, 0.0F, rows);
    Select(safeSum, total, one, nonemptyRows);
    Select(naturalReference, maximum, zero, nonemptyRows);
    Muls(naturalReference, naturalReference, MXFP4_LN2, rows);
    if constexpr (UseCxQuant) {
        Adds(naturalReference, naturalReference, MXFP4_NEG_LN_DELTA, rows);
    }
    Ln(lse, safeSum, rows);
    Add(lse, lse, naturalReference, rows);
    Duplicate(infinityBits, static_cast<uint32_t>(0x7f800000));
    Select(lse, lse, reinterpret_cast<RegTensor<float>&>(infinityBits), nonemptyRows);
    Select(naturalReference, naturalReference, invalid, nonemptyRows);
    StoreAlign(finalLse, lse, rows);
    StoreAlign(finalReference, naturalReference, rows);
}

__simd_vf__ inline void NormalizeFlashDecodeVf(__ubuf__ float* output,
    __ubuf__ float* rowSum, uint32_t rowCount, uint32_t dimension)
{
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero, one;
    Duplicate(zero, 0.0F);
    Duplicate(one, 1.0F);
    for (uint16_t row = 0; row < rowCount; ++row) {
        RegTensor<float> denominator;
        MaskReg nonempty;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(denominator, rowSum + row * 8);
        Compares<float, CMPMODE::GT>(nonempty, denominator, 0.0F, all);
        Select(denominator, denominator, one, nonempty);
        for (uint16_t channel = 0; channel < dimension; channel += 64) {
            RegTensor<float> value;
            LoadAlign(value, output + row * dimension + channel);
            Div<float, &MXFP4_OUTPUT_DIV>(value, value, denominator, all);
            Select(value, value, zero, nonempty);
            StoreAlign(output + row * dimension + channel, value, all);
        }
    }
}

} // namespace Mxfp4Api
#endif
