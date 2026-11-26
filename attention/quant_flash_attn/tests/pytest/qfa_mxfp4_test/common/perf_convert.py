#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact E2M1 expansion. Conversion runs before, never inside, measurement."""
import torch
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _expand(SRC, DST, COUNT: tl.constexpr, BF16: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        byte = tl.load(SRC+index//2, index < COUNT, 0).to(tl.uint32)
        code = (byte >> ((index % 2)*4)) & 15
        mag = code & 7
        if BF16:
            value = tl.where(mag == 0, 0., tl.where(mag == 1, .5, tl.where(mag == 2, 1.,
                    tl.where(mag == 3, 1.5, tl.where(mag == 4, 2., tl.where(mag == 5, 3.,
                    tl.where(mag == 6, 4., 6.)))))))
            value = tl.where((code & 8) != 0, -value, value)*SCALE
            tl.store(DST+index, value.to(tl.bfloat16), index < COUNT)
        else:
            encoded = tl.where(mag == 0, 0, tl.where(mag == 1, 48, tl.where(mag == 2, 56,
                      tl.where(mag == 3, 60, tl.where(mag == 4, 64, tl.where(mag == 5, 68,
                      tl.where(mag == 6, 72, 76)))))))
            tl.store(DST+index, (encoded | ((code & 8) << 4)).to(tl.uint8), index < COUNT)


def _convert_payload_torch(packed, backend, scale):
    """Bit-exact fallback for environments without a Triton Ascend wheel.

    Large fixed KV pools are expanded on the CPU to avoid oversized AICPU
    gather/shift operations. This is outside the measured attention region.
    """
    source = packed.cpu()
    codes = torch.stack((source & 15, (source >> 4) & 15), dim=-1).reshape(
        *source.shape[:-1], source.shape[-1]*2)
    magnitude = codes & 7
    negative = (codes & 8) != 0
    if backend == 'bf16':
        levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
        result = levels[magnitude.long()]*torch.where(negative, -1., 1.)*scale
        return result.to(torch.bfloat16).to(packed.device)
    encoded = torch.tensor([0, 48, 56, 60, 64, 68, 72, 76], dtype=torch.uint8)
    result = encoded[magnitude.long()] | ((codes & 8) << 4)
    return result.view(torch.float8_e4m3fn).to(packed.device)


def convert_payload(packed, backend, scale):
    if backend not in ('c8', 'bf16'):
        raise ValueError(backend)
    if triton is None:
        return _convert_payload_torch(packed, backend, scale)
    shape = (*packed.shape[:-1], packed.shape[-1]*2)
    out = torch.empty(shape, dtype=torch.bfloat16 if backend == 'bf16' else torch.uint8, device=packed.device)
    _expand[(triton.cdiv(out.numel(), 4096),)](packed, out, out.numel(), backend == 'bf16', scale, 4096)
    return out if backend == 'bf16' else out.view(torch.float8_e4m3fn)
