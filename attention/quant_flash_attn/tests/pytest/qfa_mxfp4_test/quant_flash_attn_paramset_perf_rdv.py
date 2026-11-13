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

"""Dependency-free paired performance scope from c4a4-test-softmax-fp32.md."""
import hashlib
import itertools
import json
from pathlib import Path

from .quant_flash_attn_paramset_common import model_configs

BACKENDS = ('c4', 'c8', 'bf16')


def get_cases(scope='full', kind='all', case_set='document'):
    if kind not in ('all', 'prefill', 'decode'):
        raise ValueError('kind must be all, prefill or decode')
    if case_set == 'mxfp8':
        cases = json.loads(Path(__file__).with_name('perf_cases_mxfp8.json').read_text())['cases']
        return [c for c in cases if kind == 'all' or c['kind'] == kind]
    if case_set != 'document':
        raise ValueError('case-set must be document or mxfp8')
    prefill = [('prefill', b, 8192, k) for b, k in itertools.product(
        (1, 2, 4), (8192, 32768, 65536, 131072, 262144))]
    batches = (1, 2, 16) if scope == 'small' else (1, 2, 4, 8, 16, 32, 64)
    decode = [('decode', b, q, k) for b, q, k in itertools.product(
        batches, (1, 2, 4, 6, 8), (32768, 65536, 131072, 262144))]
    scenarios = prefill if kind == 'prefill' else decode if kind == 'decode' else prefill + decode
    return [dict(name=f'{model}_TP{tp}_{category}_B{b}_Q{q}_KV{k}',
                 model=model, tp=tp, kind=category, B=b, Q=q, KV=k,
                 D=d, N_q=h, N_kv=hk, page=256, mask_mode=3,
                 q_scale_layout='TND' if category == 'prefill' else 'N2TGD',
                 enable_lse=False, seed=20261111, quant_schema=2)
            for model, d, h, hk, tp in model_configs(scope)
            for category, b, q, k in scenarios]


def shard_cases(cases, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Require shard-count > 0 and 0 <= shard-index < shard-count')
    return cases[index::count]


def case_fingerprint(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def sequence_lengths(case):
    """Physical Q token offsets, optional effective Q lengths, and paged KV lengths."""
    cu = case.get('cu_seqlens_q', [i*case['Q'] for i in range(case['B']+1)])
    q = [b-a for a, b in zip(cu, cu[1:])]
    used_q = case.get('seqused_q', q if case['kind'] == 'prefill' else None)
    kv = case.get('seqused_kv', [case['KV']]*case['B'])
    if len(q) != case['B'] or len(kv) != case['B'] or any(n < 0 for n in q+kv):
        raise ValueError('Invalid per-sequence lengths')
    if used_q is not None and (len(used_q) != len(q) or any(n < 0 or n > size for n, size in zip(used_q, q))):
        raise ValueError('Invalid effective Q lengths')
    return cu, used_q, kv
