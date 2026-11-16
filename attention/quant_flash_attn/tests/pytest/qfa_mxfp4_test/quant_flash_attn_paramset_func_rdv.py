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

"""Accuracy-only full/small matrix; matches the established C4 input ranges."""
from .quant_flash_attn_paramset_common import build_paramset, expand_paramset_to_cases

CASE_SETS = ('document', 'bench16k')


def scenarios(scope):
    cases = [dict(name=f'short{q}', qlens=[q], klens=[q], page=page)
             for q, page in zip((1, 3, 4, 5, 11, 22), (128, 128, 256, 512, 512, 128))]
    cases += [
        dict(name='mixed', qlens=[1, 3, 22, 129], klens=[11, 32771, 65, 259], page=128),
        dict(name='large_batch', qlens=[1,3,4,5,11,22,102,300,512,1024,3000,4096]*3,
             klens=[1,17,65,127,33,65,202,330,891,1999,3100,5196]*3, page=128),
        dict(name='long_prefill', qlens=[129], klens=[32771], page=512),
    ]
    cases += [dict(name=f'prefill_q8192_kv{k}_b{b}', qlens=[8192]*b, klens=[k]*b, page=512)
              for k in (8192, 32768, 65536) for b in (1, 2)]
    lengths = range(1, 5) if scope == 'small' else range(1, 9)
    batches = (1, 2, 16) if scope == 'small' else (1, 2, 4, 8, 16)
    cases += [dict(name=f'mtp{q}_kv{k}_b{b}', qlens=[q]*b, klens=[k]*b, page=256)
              for q in lengths for k in (32768, 65536, 131072) for b in batches]
    return cases


TEST_PARAMS = build_paramset('full', scenarios('full'))
CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()


def get_cases(scope='full', case_set='document'):
    if case_set == 'bench16k':
        from .quant_flash_attn_paramset_bench16k import get_accuracy_cases
        return get_accuracy_cases(scope)
    if case_set != 'document':
        raise ValueError('Accuracy case-set must be document or bench16k')
    return CASES if scope == 'full' else expand_paramset_to_cases(build_paramset(scope, scenarios(scope)))
