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

"""Dependency-free parameters for the supported C4 public interface."""
import copy
import itertools
import math

MODELS = [('qwen36_35b', 256, 16, 2), ('qwen38_27b', 256, 24, 4), ('qwen3_30b', 128, 32, 4)]
SCOPES = ('full', 'small')
P_QUANT_SCHEMES = ('ocp', 'cx_legacy', 'cx_delta')
PARAM_NAMES = [
    'B', 'N_q', 'N_kv', 'D', 'cu_seqlens_q', 'cu_seqlens_kv',
    'seqused_q', 'seqused_kv', 'max_seqlen_q', 'max_seqlen_kv',
    'enable_pa', 'kv_cache_layout', 'input_layout', 'block_size',
    'mask_mode', 'q_scale_layout', 'p_scale', 'softmax_scale',
    'enable_lse', 'enable_v_tail', 'quant_mode', 'is_contiguous', 'seed',
    'model', 'tp', 'quant_schema', 'p_quant_scheme', 'reference_backend',
]
TEST_PARAMS_DEFAULTS = {
    'cu_seqlens_kv': [None], 'seqused_q': [None],
    'max_seqlen_q': [-1], 'max_seqlen_kv': [-1],
    'enable_pa': [True], 'kv_cache_layout': ['PA_BBND'], 'input_layout': ['TND'],
    'p_scale': [None], 'softmax_scale': [None], 'enable_lse': [True],
    'enable_v_tail': [False], 'quant_mode': [5], 'is_contiguous': [True],
    'seed': [421], 'model': ['custom'], 'tp': [1], 'quant_schema': [None], 'p_quant_scheme': [None],
    'reference_backend': ['kernel'],
}


def resolve_quantization(quant_schema=None, p_quant_scheme=None):
    """One selection controls both the public operator and every reference."""
    if p_quant_scheme is not None and p_quant_scheme not in P_QUANT_SCHEMES:
        raise ValueError(f'Unknown P quantization scheme: {p_quant_scheme!r}')
    if quant_schema is None:
        quant_schema = 2 if p_quant_scheme is None else P_QUANT_SCHEMES.index(p_quant_scheme)
    if not isinstance(quant_schema, int) or quant_schema not in (0, 1, 2):
        raise ValueError('quant_schema must be 0, 1 or 2')
    scheme = P_QUANT_SCHEMES[quant_schema]
    if p_quant_scheme is not None and p_quant_scheme != scheme:
        raise ValueError('quant_schema and p_quant_scheme select different algorithms')
    return quant_schema, scheme


def paged_kv_pool_size(kv_lengths, page, requested=None, extra_pages=0):
    """Resolve physical capacity independently of the effective KV lengths."""
    required = sum((length + page - 1) // page for length in kv_lengths)
    if requested is None:
        return required + extra_pages
    if type(requested) is not int or requested <= 0 or requested < required:
        raise ValueError('kv_cache_num_blocks must be a positive integer covering all requested KV pages')
    return requested


def normalize_params(params):
    result = {key: copy.deepcopy(value[0]) for key, value in TEST_PARAMS_DEFAULTS.items()}
    result.update(copy.deepcopy(params))
    result['quant_schema'], result['p_quant_scheme'] = resolve_quantization(
        result['quant_schema'], result['p_quant_scheme'])
    missing = set(PARAM_NAMES) - set(result)
    if missing:
        raise ValueError(f'Missing C4 parameters: {sorted(missing)}')
    if result['quant_mode'] != 5 or result['D'] not in (128, 256):
        raise ValueError('C4 requires quant_mode=5 and D=128/256')
    if result['p_quant_scheme'] not in P_QUANT_SCHEMES:
        raise ValueError(f"Unknown reference P quantization scheme: {result['p_quant_scheme']!r}")
    if result['reference_backend'] not in ('global', 'kernel'):
        raise ValueError('reference_backend must be global or kernel')
    if result['N_q'] <= 0 or result['N_kv'] <= 0 or result['N_q'] % result['N_kv']:
        raise ValueError('Q heads must be a positive multiple of KV heads')
    if not result['enable_pa'] or result['kv_cache_layout'] != 'PA_BBND' or result['input_layout'] != 'TND':
        raise ValueError('C4 requires TND input/output and PA_BBND KV')
    if result['cu_seqlens_kv'] is not None:
        raise ValueError('Paged KV uses seqused_kv; cu_seqlens_kv must be None')
    if result['q_scale_layout'] not in ('TND', 'N2TGD') or result['mask_mode'] not in (0, 3):
        raise ValueError('Unsupported Qscale layout or mask mode')
    if result['block_size'] not in (128, 256, 512):
        raise ValueError('The accuracy scope uses page sizes 128/256/512')
    if result['enable_v_tail'] or result['p_scale'] is not None or not result['is_contiguous']:
        raise ValueError('This suite uses contiguous C4 inputs, no V-tail and p_scale=None')
    cu, kv = result['cu_seqlens_q'], result['seqused_kv']
    if result['B'] <= 0 or len(cu) != result['B'] + 1 or len(kv) != result['B']:
        raise ValueError('Length arrays do not match B')
    if cu[0] != 0 or any(a > b for a, b in zip(cu, cu[1:])) or cu[-1] <= 0:
        raise ValueError('Q cumulative storage lengths must start at 0 and be nondecreasing, with T>0')
    stored = [b-a for a, b in zip(cu, cu[1:])]
    used = result['seqused_q'] if result['seqused_q'] is not None else stored
    if len(used) != result['B'] or any(q < 0 or q > capacity for q, capacity in zip(used, stored)):
        raise ValueError('Effective Q lengths must fit their storage segments')
    if any(k < 0 for k in kv):
        raise ValueError('KV lengths must be nonnegative')
    paged_kv_pool_size(kv, result['block_size'], result.get('kv_cache_num_blocks'))
    logical_shape = result.get('logical_kv_cache_shape')
    if logical_shape is not None and logical_shape != [result.get('kv_cache_num_blocks'),
                                                       result['block_size'], result['N_kv'] * result['D']]:
        raise ValueError('logical_kv_cache_shape must match the page count, page size and KV heads')
    for name, lengths in [('max_seqlen_q', stored), ('max_seqlen_kv', kv)]:
        if result[name] != -1 and result[name] < max(lengths):
            raise ValueError(f'{name} is smaller than the represented lengths')
    scale = result['softmax_scale']
    result['softmax_scale'] = result['D'] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(result['softmax_scale']) or result['softmax_scale'] <= 0:
        raise ValueError('softmax_scale must be positive and finite')
    return result


def expand_paramset_to_cases(test_params):
    cases = []
    for name, params in test_params.items():
        expanded = dict(TEST_PARAMS_DEFAULTS, **params)
        unknown = set(expanded) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f'Unknown parameters: {sorted(unknown)}')
        products = list(itertools.product(*(expanded[key] for key in PARAM_NAMES)))
        for index, values in enumerate(products):
            case_name = name if len(products) == 1 else f'{name}__{index:03d}'
            cases.append(normalize_params(dict(zip(PARAM_NAMES, values), name=case_name)))
    return cases


def model_configs(scope):
    if scope not in SCOPES:
        raise ValueError(f'Unknown scope: {scope}')
    return [(name, d, heads//tp, max(1, kv//tp), tp)
            for index, (name, d, heads, kv) in enumerate(MODELS)
            for tp in (1, 2, 4) if scope == 'full' or (index == 0 and tp == 2)]


def build_paramset(scope, scenarios, lse_modes=(True,)):
    result = {}
    for model, dimension, heads, kv_heads, tp in model_configs(scope):
        for scenario in scenarios:
            stored, kv = scenario['qlens'], scenario['klens']
            cu = [0] + list(itertools.accumulate(stored))
            for qlayout, mask, lse in itertools.product(('TND', 'N2TGD'), (0, 3), lse_modes):
                name = f"{model}_TP{tp}_{scenario['name']}_{qlayout}_M{mask}_LSE{int(lse)}"
                used = scenario.get('used_q', stored if qlayout == 'TND' else None)
                result[name] = {
                    'B': [len(stored)], 'N_q': [heads], 'N_kv': [kv_heads], 'D': [dimension],
                    'cu_seqlens_q': [cu], 'seqused_q': [used], 'seqused_kv': [kv],
                    'max_seqlen_q': [max(stored)], 'max_seqlen_kv': [max(kv)],
                    'block_size': [scenario['page']], 'mask_mode': [mask],
                    'q_scale_layout': [qlayout], 'enable_lse': [lse], 'model': [model], 'tp': [tp],
                }
    return result
