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

"""Per-case input/CPU/NPU caches, bound to parameters and one input generation."""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import torch

CACHE_VERSION = 'mxfp4-public-quant-schema-v5'
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'golden_cache'


def fingerprint(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _path(name, suffix, cache_dir=None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError(f'Invalid cache case name: {name!r}')
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f'{name}_{suffix}.pt'


def _save(name, suffix, payload, params, input_id, cache_dir, producer=None):
    path = _path(name, suffix, cache_dir)
    envelope = dict(version=CACHE_VERSION, fingerprint=fingerprint(params), input_id=input_id,
                    payload=payload, producer=producer)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(envelope, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(name, suffix, params, input_id, cache_dir):
    path = _path(name, suffix, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f'Missing {suffix} cache: {path}; run the corresponding stage first')
    envelope = torch.load(path, map_location='cpu', weights_only=True)
    if envelope.get('version') != CACHE_VERSION or envelope.get('fingerprint') != fingerprint(params):
        raise ValueError(f'Stale cache parameters/reference version: {path}')
    if input_id is not None and envelope.get('input_id') != input_id:
        raise ValueError(f'Output belongs to another input generation: {path}')
    return envelope


def save_input(case_name, data_dict, params, cache_dir=None):
    input_id = uuid.uuid4().hex
    _save(case_name, 'input', data_dict, params, input_id, cache_dir)
    return input_id


def load_input(case_name, params, cache_dir=None):
    entry = _load(case_name, 'input', params, None, cache_dir)
    return entry['payload'], entry['input_id']


def save_cpu_output(case_name, cpu_out, cpu_lse, params, input_id, cache_dir=None):
    _save(case_name, 'cpu_output', (cpu_out, cpu_lse), params, input_id, cache_dir)


def load_cpu_output(case_name, params, input_id, cache_dir=None):
    return tuple(_load(case_name, 'cpu_output', params, input_id, cache_dir)['payload'])


def save_npu_output(case_name, out, lse, params, input_id, cache_dir=None, producer=None):
    _save(case_name, 'npu_output', (out, lse), params, input_id, cache_dir, producer)


def load_npu_output(case_name, params, input_id, cache_dir=None, package=None):
    entry = _load(case_name, 'npu_output', params, input_id, cache_dir)
    if package is not None and (entry.get('producer') or {}).get('package') != package:
        raise ValueError('NPU output cache was produced by another installed package')
    return tuple(entry['payload'])
