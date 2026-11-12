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

"""Packed MXFP4 inputs, independent CPU CX reference and public NPU API call.

Only npu_mxfp4_fa imports torch_npu / the installed operator package.
CPU cache payloads always use portable uint8 storage for FP4 and E8M0.
"""
import importlib
import math
import torch
from ..quant_flash_attn_paramset_common import normalize_params

LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def unpack_fp4(payload):
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    return levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)


def decode_scale(payload):
    return torch.exp2(payload.float() - 127.)


def round_bf16_away(values):
    bits = values.float().contiguous().view(torch.int32)
    return ((bits + 0x8000) & -65536).view(torch.float32)


def quantize_probability(values):
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32)
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    rounded = round_bf16_away(values)
    return levels[torch.bucketize(rounded.contiguous(), boundaries, right=True)]


def generate_data(params):
    p = normalize_params(params)
    gen = torch.Generator().manual_seed(p['seed'])
    total, heads, kv_heads, dimension, page = (p['cu_seqlens_q'][-1], p['N_q'], p['N_kv'], p['D'], p['block_size'])
    counts = [(length + page - 1)//page for length in p['seqused_kv']]
    pages = sum(counts) + 3
    # Randomize unused pages and tails too: an erroneous padding read is observable.
    q = torch.randint(0, 256, (total, heads, dimension//2), dtype=torch.uint8, generator=gen)
    k = torch.randint(0, 256, (pages, page, kv_heads, dimension//2), dtype=torch.uint8, generator=gen)
    v = torch.randint(0, 256, k.shape, dtype=torch.uint8, generator=gen)
    qs = torch.randint(122, 126, (total, heads, dimension//64, 2), dtype=torch.uint8, generator=gen)
    ks = torch.randint(122, 126, (pages, page, kv_heads, dimension//64, 2), dtype=torch.uint8, generator=gen)
    vs = torch.randint(124, 128, (pages, page//64, kv_heads, dimension, 2), dtype=torch.uint8, generator=gen)
    if p['q_scale_layout'] == 'N2TGD':
        qs = qs.reshape(total, kv_heads, heads//kv_heads, dimension//64, 2).permute(1, 0, 2, 3, 4).contiguous()
    table = torch.zeros((p['B'], max(1, max(counts))), dtype=torch.int32)
    permutation = torch.randperm(pages, generator=gen)
    offset = 0
    for batch, count in enumerate(counts):
        table[batch, :count] = permutation[offset:offset+count].to(torch.int32)
        offset += count
    return dict(q_fp4=q, k_fp4=k, v_fp4=v, dequant_scale_q=qs,
                dequant_scale_k=ks, v_descale=vs, block_table_torch=table,
                cu_seqlens_q=torch.tensor(p['cu_seqlens_q'], dtype=torch.int32),
                seqused_q=None if p['seqused_q'] is None else torch.tensor(p['seqused_q'], dtype=torch.int32),
                seqused_kv=torch.tensor(p['seqused_kv'], dtype=torch.int32))


def decode_inputs(data, params):
    p = normalize_params(params)
    total, heads, kv_heads, dimension, page = (p['cu_seqlens_q'][-1], p['N_q'], p['N_kv'], p['D'], p['block_size'])
    qs = data['dequant_scale_q']
    if p['q_scale_layout'] == 'N2TGD':
        qs = qs.permute(1, 0, 2, 3, 4).contiguous()
    qs = qs.reshape(total, heads, dimension//32)
    query = unpack_fp4(data['q_fp4']) * decode_scale(qs).repeat_interleave(32, -1)
    keys, values = [], []
    for batch, length in enumerate(p['seqused_kv']):
        count = (length + page - 1)//page
        ids = data['block_table_torch'][batch, :count].long()
        k = unpack_fp4(data['k_fp4'][ids])
        ks = decode_scale(data['dequant_scale_k'][ids]).reshape(count, page, kv_heads, dimension//32)
        keys.append((k * ks.repeat_interleave(32, -1)).reshape(count*page, kv_heads, dimension)[:length])
        v = unpack_fp4(data['v_fp4'][ids])
        vs = decode_scale(data['v_descale'][ids]).permute(0, 1, 4, 2, 3)
        vs = vs.reshape(count, page//32, kv_heads, dimension).repeat_interleave(32, 1)
        values.append((v * vs).reshape(count*page, kv_heads, dimension)[:length])
    return query, keys, values


def cpu_mxfp4_golden(data, params):
    """FP32 score + selectable S32 P quantization; independent of task/split metadata."""
    p = normalize_params(params)
    log2_effective_x = math.log2(7.25 / 1.5) if p['p_quant_scheme'] == 'cx_delta' else math.log2(7.25)
    is_ocp = p['quant_schema'] == 0
    scale_shift = 2 if is_ocp else math.ceil(log2_effective_x)
    lse_shift = 0 if is_ocp else scale_shift
    ln_delta = math.log(1.5) if p['p_quant_scheme'] == 'cx_delta' else 0.0
    query, keys, values = decode_inputs(data, p)
    output = torch.zeros_like(query)
    lse = torch.full(query.shape[:2], float('inf'), dtype=torch.float32)
    used = p['seqused_q']
    if used is None:
        used = [b-a for a, b in zip(p['cu_seqlens_q'], p['cu_seqlens_q'][1:])]
    head_groups = p['N_q']//p['N_kv']
    for batch, (qlen, klen) in enumerate(zip(used, p['seqused_kv'])):
        start = p['cu_seqlens_q'][batch]
        if not qlen or not klen:
            continue
        padded = (klen + 31)//32*32
        key, value = keys[batch], values[batch]
        for head in range(p['N_q']):
            kv_head = head//head_groups
            for first in range(0, qlen, 32):
                count = min(32, qlen-first)
                score = torch.full((count, padded), -float('inf'), dtype=torch.float32)
                score[:, :klen] = (query[start+first:start+first+count, head] @ key[:, kv_head].T) * p['softmax_scale']
                if p['mask_mode'] == 3:
                    valid = torch.arange(klen)[None, :] <= torch.arange(first, first+count)[:, None] + klen-qlen
                    score[:, :klen].masked_fill_(~valid, -float('inf'))
                grouped = score.reshape(count, -1, 32)
                maxima = grouped.amax(-1)
                valid_group = torch.isfinite(maxima)
                scaled_max = torch.where(valid_group, maxima, 0.) * (1. / math.log(2.))
                exponents = torch.floor(scaled_max) if is_ocp else torch.ceil(scaled_max-log2_effective_x)
                reference = (exponents-2.)*math.log(2.) if is_ocp else exponents*math.log(2.)-ln_delta
                quantized = quantize_probability(torch.exp(grouped-reference[..., None]))
                quantized = torch.where(valid_group[..., None], quantized, 0.)
                row_reference = torch.where(valid_group, exponents, -1e30).amax(-1)
                row_reference = torch.where(valid_group.any(-1), row_reference, 0.)
                # OCP uses bias 125; both CX recipes use 124. Preserve the E8M0 clamp.
                encoded = (exponents-row_reference[:, None]+127.-scale_shift).clamp(0., 254.)
                weights = quantized * torch.exp2(encoded-127.)[..., None]
                weights = weights.reshape(count, padded)[:, :klen]
                denominator = weights.sum(-1)
                numerator = weights @ value[:, kv_head]
                result = torch.where(denominator[:, None] > 0, numerator/denominator[:, None].clamp_min(1e-30), 0.)
                output[start+first:start+first+count, head] = result
                lse[start+first:start+first+count, head] = torch.where(
                    denominator > 0, torch.log(denominator)+(row_reference+lse_shift)*math.log(2.)-ln_delta, float('inf'))
    out = round_bf16_away(output).to(torch.bfloat16)
    statistics = lse.T.contiguous() if p['enable_lse'] else torch.empty(0, dtype=torch.float32)
    return out, statistics


def prepare_npu_inputs(data, device):
    fp4, e8m0 = getattr(torch, 'float4_e2m1fn_x2', None), getattr(torch, 'float8_e8m0fnu', None)
    if fp4 is None or e8m0 is None:
        raise RuntimeError('Installed torch/torch_npu must provide FP4 E2M1 and E8M0 dtypes')
    converted = {name: tensor.to(device) if tensor is not None else None for name, tensor in data.items()}
    for name in ('q_fp4', 'k_fp4', 'v_fp4'):
        converted[name] = converted[name].view(fp4)
    for name in ('dequant_scale_q', 'dequant_scale_k', 'v_descale'):
        converted[name] = converted[name].view(e8m0)
    return converted


def invoke_operator(ops, data, params):
    p = normalize_params(params)
    common = dict(cu_seqlens_q=data['cu_seqlens_q'], seqused_q=data['seqused_q'],
                  seqused_kv=data['seqused_kv'], max_seqlen_q=p['max_seqlen_q'],
                  max_seqlen_kv=p['max_seqlen_kv'], mask_mode=p['mask_mode'],
                  layout_q='TND', layout_q_descale=p['q_scale_layout'], layout_kv='PA_BBND', layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(p['N_q'], p['N_kv'], p['D'], 5, **common)
    mask = None
    if p['mask_mode'] == 3:
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(data['q_fp4'].device)
    return ops.quant_flash_attn(
        *(data[name] for name in ('q_fp4', 'k_fp4', 'v_fp4', 'dequant_scale_q', 'dequant_scale_k', 'v_descale')),
        5, block_table=data['block_table_torch'], metadata=metadata, attn_mask=mask,
        p_scale=None, softmax_scale=p['softmax_scale'], return_softmax_lse=p['enable_lse'],
        quant_schema=p['quant_schema'], **common)


def npu_mxfp4_fa(data, params, package='cann_ops_transformer', device_id=0):
    import torch_npu  # noqa: F401 -- registers the NPU device and packed dtypes
    importlib.import_module(package)
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)
    prepared = prepare_npu_inputs(data, device)
    output, lse = invoke_operator(torch.ops.cann_ops_transformer, prepared, params)
    torch.npu.synchronize()
    if output.device != prepared['q_fp4'].device or lse.device != output.device:
        raise AssertionError('Both API outputs must be on the input NPU')
    return output.cpu(), lse.cpu()
