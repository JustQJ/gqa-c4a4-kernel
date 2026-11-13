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

"""One CPU packed input, materialized for only one NPU backend at a time."""
import hashlib
from ..quant_flash_attn_paramset_perf_rdv import sequence_lengths


def generate_inputs(case):
    import torch
    b, q, kv, d, h, hk, page = (case[k] for k in ('B', 'Q', 'KV', 'D', 'N_q', 'N_kv', 'page'))
    cu, _, kv_lengths = sequence_lengths(case)
    counts = [(length+page-1)//page for length in kv_lengths]
    pages_per_seq = max(counts)
    pages = sum(counts)
    generator = torch.Generator().manual_seed(case['seed'])
    shapes = ((cu[-1], h, d//2), (pages, page, hk, d//2), (pages, page, hk, d//2))
    raw = [torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator) for shape in shapes]
    order = torch.randperm(pages, generator=generator, dtype=torch.int64).to(torch.int32)
    table = torch.full((b, pages_per_seq), -1, dtype=torch.int32)
    offset = 0
    for index, count in enumerate(counts):
        table[index, :count] = order[offset:offset+count]
        offset += count
    digest = hashlib.sha256()
    for tensor in (*raw, table):
        digest.update(memoryview(tensor.numpy()).cast('B'))
    return dict(raw=raw, table=table, sha256=digest.hexdigest())


def prepare_inputs(source, case, backend, device):
    import torch
    from .perf_convert import convert_payload
    b, q, kv, d, h, hk, page = (case[k] for k in ('B', 'Q', 'KV', 'D', 'N_q', 'N_kv', 'page'))
    cu, used_q, kv_lengths = sequence_lengths(case)
    tensors = []
    for index, raw in enumerate(source['raw']):
        packed = raw.to(device)
        if backend == 'c4':
            value = packed.view(torch.float4_e2m1fn_x2)
        else:
            value = convert_payload(packed, backend, 0.25 if index == 2 else 0.0625)
        tensors.append(value)
        del packed, value
    result = dict(q=tensors[0], k=tensors[1], v=tensors[2],
                  table=source['table'].to(device),
                  cu=torch.tensor(cu, dtype=torch.int32, device=device),
                  used_kv=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
                  used_q=torch.tensor(used_q, dtype=torch.int32, device=device) if used_q is not None else None,
                  mask=torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(device) if case['mask_mode'] == 3 else None)
    if backend == 'bf16':
        result['k'] = result['k'].reshape(-1, page, hk*d)
        result['v'] = result['v'].reshape(-1, page, hk*d)
    else:
        qshape = (cu[-1], h, d//64, 2) if case['q_scale_layout'] == 'TND' else (hk, cu[-1], h//hk, d//64, 2)
        pages = source['raw'][1].shape[0]
        for name, shape, code in (
            ('qs', qshape, 123), ('ks', (pages, page, hk, d//64, 2), 123),
            ('vs', (pages, page//64, hk, d, 2), 125)):
            result[name] = torch.full(shape, code, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    return result
