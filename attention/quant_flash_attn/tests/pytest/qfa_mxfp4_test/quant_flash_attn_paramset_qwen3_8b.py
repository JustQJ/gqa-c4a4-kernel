#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.

"""Sixteen Qwen3-8B TP1 attention cases shared by accuracy and performance."""
import argparse
import copy
import itertools
import json
from collections import Counter
from pathlib import Path

from .quant_flash_attn_paramset_common import normalize_params

CASE_SET = 'qwen3_8b'
# Keep four times the largest case's required pages: B128 * KV16384 / page128.
KV_CACHE_NUM_BLOCKS = 16384 * 4
KV_CACHE_SHAPE = (KV_CACHE_NUM_BLOCKS, 128, 8 * 128)
PAGE_SIZES = {CASE_SET: 128, 'qwen3_8b_page256': 256}


def _kv_range(batch):
    """Include both 1k/16k endpoints and non-page-aligned intermediate lengths."""
    return [1024 + (16384 - 1024) * i // (batch - 1) for i in range(batch)]


_SPECS = (
    ('prefill_1', 'prefill', [500], [500]),
    ('prefill_2', 'prefill', [1024], [4096]),
    ('prefill_3', 'prefill', [2048], [16384]),
    ('prefill_4', 'prefill', [500, 1024], [1024, 8192]),
    ('prefill_5', 'prefill', [1024, 2048], [4096, 16384]),
    ('prefill_6', 'prefill', [500, 1024, 1536, 2048], [2048, 4096, 8192, 16384]),
    ('decode_1', 'decode', [1] * 8, [1011, 1022, 1023, 1024, 2048, 4099, 4008, 16526]),
    ('decode_2', 'decode', [1] * 16, [2048, 4099, 4008, 2048] * 4),
    ('decode_3', 'decode', [4] * 32, [4096] * 32),
    ('decode_4', 'decode', [1] * 64, [4555, 4900, 8900, 8192] * 16),
    ('decode_5', 'decode', [1] * 96, _kv_range(96)),
    ('decode_6', 'decode', [1] * 128, [16384] * 128),
    ('mixed_1', 'mixed', [500] + [1] * 15, [1024] + _kv_range(15)),
    ('mixed_2', 'mixed', [1024] + [1] * 31, [4096] + _kv_range(31)),
    ('mixed_3', 'mixed', [500, 2048] + [1] * 62, [2048, 16384] + _kv_range(62)),
    ('mixed_4', 'mixed', [1024, 1536] + [1] * 126, [8192, 16384] + _kv_range(126)),
)


def get_performance_cases(scope='full', case_set=CASE_SET):
    if scope not in ('small', 'full'):
        raise ValueError('qwen3_8b scope must be small or full; both select the same 16 cases')
    page = PAGE_SIZES[case_set]
    num_blocks = KV_CACHE_NUM_BLOCKS * KV_CACHE_SHAPE[1] // page
    cases = []
    for name, kind, queries, keys in _SPECS:
        qlens, kvlens = list(queries), list(keys)
        cases.append(dict(
            name=case_set + '_' + name, case_set=case_set, model=CASE_SET, tp=1,
            kind=kind, B=len(qlens), Q=max(qlens), KV=max(kvlens), D=128, N_q=32, N_kv=8,
            page=page, layout_kv='PA_BBND', mask_mode=3,
            q_scale_layout='N2TGD' if kind == 'decode' else 'TND',
            enable_lse=False, seed=421, quant_schema=2, c8_p_scale=448., softmax_scale=128 ** -0.5,
            cu_seqlens_q=[0] + list(itertools.accumulate(qlens)),
            seqused_q=qlens, seqused_kv=kvlens, max_seqlen_q=max(qlens), max_seqlen_kv=max(kvlens),
            kv_cache_num_blocks=num_blocks, logical_kv_cache_shape=[num_blocks, page, KV_CACHE_SHAPE[2]],
        ))
    return cases


def get_accuracy_cases(scope='full', case_set=CASE_SET):
    result = []
    for case in get_performance_cases(scope, case_set):
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
    parser.add_argument('--case-set', choices=tuple(PAGE_SIZES), default=CASE_SET)
    parser.add_argument('--scope', choices=('small', 'full'), default='full')
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    loader = get_accuracy_cases if args.accuracy else get_performance_cases
    cases = loader(args.scope, args.case_set)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(dict(case_set=args.case_set, cases=cases), stream, indent=2, allow_nan=False)
            stream.write('\n')
    summary = dict(case_set=args.case_set, count=len(cases), logical_kv_cache_shape=cases[0]['logical_kv_cache_shape'])
    if not args.accuracy:
        summary['kinds'] = dict(Counter(case['kind'] for case in cases))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
