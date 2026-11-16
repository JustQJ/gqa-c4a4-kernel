#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit bench16k scenarios, shared by accuracy and performance tests."""
import argparse
import copy
import itertools
import json
from collections import Counter
from pathlib import Path

from .quant_flash_attn_paramset_common import normalize_params

KV_CACHE_NUM_BLOCKS = 15687 * 2
KV_CACHE_SHAPE = (KV_CACHE_NUM_BLOCKS, 128, 512)
_SPECS = (
    ('single_prefill_1', 'prefill', [4096], [4096]),
    ('single_prefill_2', 'prefill', [3972], [3988 + 4096 + 3988]),
    ('single_prefill_3', 'prefill', [3980], [3988 + 4096 + 3988 + 4096]),
    ('dual_prefill_1', 'prefill', [4096, 3988], [4096, 3988 + 4096 + 3972]),
    ('dual_prefill_2', 'prefill', [2048, 2400],
     [4096 + 4096 + 4096 + 3988 + 3988, 3988 + 3972]),
    ('dual_prefill_3', 'prefill', [2083, 2900],
     [4096 + 4096 + 4096 + 3988 + 3988 + 3988, 3988 + 3972 + 4096 + 4096]),
    ('decode_1', 'decode', [4] * 32,
     [8 * 1024 + 107 * i for i in range(16)] + [16 * 1024 + 125 * i for i in range(16)]),
    ('decode_2', 'decode', [4] * 24,
     [32 * 1024 + 32 * i for i in range(14)] + [64 * 1024 + 444 * i for i in range(10)]),
    ('decode_3', 'decode', [1] * 32,
     [2 * 1024 + 11 * i for i in range(16)] + [256 + 25 * i for i in range(16)]),
    ('decode_4', 'decode', [4] * 33,
     [2 * 1024 + 9 * i for i in range(32)] + [16 * 1024 + 25]),
    ('mixed_1', 'mixed', [2044] + [4] * 26,
     [2044] + [10 * 1024 + 117 * i for i in range(16)] + [18 * 1024 + 125 * i for i in range(10)]),
    ('mixed_2', 'mixed', [4096] + [4] * 22,
     [8192] + [32 * 1024 + 137 * i for i in range(12)] + [22 * 1024 + 100 * i for i in range(10)]),
)


def get_performance_cases(scope='full'):
    if scope not in ('small', 'full'):
        raise ValueError(f'bench16k scope must be small or full; both select the same {len(_SPECS)} cases')
    cases = []
    for name, kind, queries, keys in _SPECS:
        qlens, kvlens = list(queries), list(keys)
        cases.append(dict(
            name='bench16k_' + name, case_set='bench16k', model='bench16k', tp=None,
            kind=kind, B=len(qlens), Q=max(qlens), KV=max(kvlens), D=256, N_q=16, N_kv=2,
            page=128, layout_kv='PA_BBND', mask_mode=3,
            q_scale_layout='N2TGD' if kind == 'decode' else 'TND',
            enable_lse=False, seed=421, quant_schema=2, c8_p_scale=448., softmax_scale=1 / 16,
            cu_seqlens_q=[0] + list(itertools.accumulate(qlens)),
            seqused_q=qlens, seqused_kv=kvlens, max_seqlen_q=max(qlens), max_seqlen_kv=max(kvlens),
            kv_cache_num_blocks=KV_CACHE_NUM_BLOCKS, logical_kv_cache_shape=list(KV_CACHE_SHAPE),
        ))
    return cases


def get_accuracy_cases(scope='full'):
    result = []
    for case in get_performance_cases(scope):
        params = {key: copy.deepcopy(case[key]) for key in (
            'name', 'case_set', 'model', 'tp', 'B', 'N_q', 'N_kv', 'D',
            'cu_seqlens_q', 'seqused_q', 'seqused_kv', 'max_seqlen_q', 'max_seqlen_kv',
            'mask_mode', 'q_scale_layout', 'softmax_scale', 'enable_lse', 'seed', 'quant_schema',
            'kv_cache_num_blocks', 'logical_kv_cache_shape',
        )}
        params.update(block_size=case['page'])
        result.append(normalize_params(params))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('small', 'full'), default='full')
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = get_accuracy_cases(args.scope) if args.accuracy else get_performance_cases(args.scope)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(dict(case_set='bench16k', cases=cases), stream, indent=2, allow_nan=False)
            stream.write('\n')
    summary = dict(case_set='bench16k', count=len(cases), logical_kv_cache_shape=KV_CACHE_SHAPE)
    if not args.accuracy:
        summary['kinds'] = dict(Counter(case['kind'] for case in cases))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
