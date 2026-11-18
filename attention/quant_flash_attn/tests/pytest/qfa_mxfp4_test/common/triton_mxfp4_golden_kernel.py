"""C4 metadata split -> grouped PV -> raw-PV merge reference on NPU.

The split/workspace/combine structure follows vllm-qwen36-all's
oscar_pnq_decode_ops.py. CX exponent selection, actual metadata boundaries,
16-task common scales and a_i/S merge follow this repository's C4 kernel.
Q/K/V are decoded before launch; tl.dot uses FP16 operands and FP32 accumulators.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice
from .quant_flash_attn_golden import decode_inputs
from .mxfp4_schedule import decode_sections, effective_lengths, iter_tiles
from ..quant_flash_attn_paramset_common import normalize_params


@triton.jit
def _split_kernel(Q, K, V, DESC, PV, SUM, REF,
                  HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                  TASK_N: tl.constexpr, TASKS_PER_GROUP: tl.constexpr, SCALE: tl.constexpr, CAUSAL: tl.constexpr,
                  SCHEMA: tl.constexpr, BM: tl.constexpr = 16, BN: tl.constexpr = 32):
    pid = tl.program_id(0)
    qbase = tl.load(DESC+pid*9+0)
    kbase = tl.load(DESC+pid*9+1)
    qlen = tl.load(DESC+pid*9+2)
    klen = tl.load(DESC+pid*9+3)
    head = tl.load(DESC+pid*9+4)
    first_row = tl.load(DESC+pid*9+5)
    last_row = tl.load(DESC+pid*9+6)
    first_task = tl.load(DESC+pid*9+7)
    last_task = tl.load(DESC+pid*9+8)
    row = first_row+tl.arange(0, BM)
    token_q = row//G
    qhead = head*G+row%G
    if G == 1:
        khead = head//(HQ//HK)
    else:
        khead = head
    dims = tl.arange(0, D)
    cols = tl.arange(0, BN)
    q = tl.load(Q+((qbase+token_q[:, None])*HQ+qhead[:, None])*D+dims[None, :],
                row[:, None] < last_row, 0.).to(tl.float16)
    running = tl.full((BM,), -1e30, tl.float32)
    total = tl.zeros((BM,), tl.float32)
    history = tl.zeros((BM, D), tl.float32)
    for first in range(first_task, last_task, TASKS_PER_GROUP):
        stop = tl.minimum(first+TASKS_PER_GROUP, last_task)
        current = running
        # First pass selects one reference for the entire group.
        # Recompute scores in pass two to keep the independent reference's UB bounded.
        for lo in range(first*TASK_N, stop*TASK_N, BN):
            token = lo+cols
            key = tl.load(K+((kbase+token[None, :])*HK+khead)*D+dims[:, None],
                          token[None, :] < klen, 0.).to(tl.float16)
            score = tl.dot(q, key)*SCALE
            valid = (row[:, None] < last_row) & (token[None, :] < klen)
            if CAUSAL:
                valid = valid & (token[None, :] <= token_q[:, None]+klen-qlen)
            score = tl.where(valid, score, -float('inf'))
            maximum = tl.max(tl.reshape(score, (BM, BN//32, 32)), 2)
            if SCHEMA == 2:
                exponent = tl.ceil(maximum*1.4426950408889634-2.273018494406416)
            elif SCHEMA == 1:
                exponent = tl.ceil(maximum*1.4426950408889634-2.857980995127572)
            else:
                exponent = tl.floor(maximum*1.4426950408889634)
            exponent = tl.where(maximum != -float('inf'), exponent, -1e30)
            current = tl.maximum(current, tl.max(exponent, 1))
        alpha = tl.where(running > -1e30, tl.exp((running-current)*0.6931471805599453), 0.)
        group_pv = tl.zeros((BM, D), tl.float32)
        group_sum = tl.zeros((BM,), tl.float32)
        for lo in range(first*TASK_N, stop*TASK_N, BN):
            token = lo+cols
            key = tl.load(K+((kbase+token[None, :])*HK+khead)*D+dims[:, None],
                          token[None, :] < klen, 0.).to(tl.float16)
            score = tl.dot(q, key)*SCALE
            valid = (row[:, None] < last_row) & (token[None, :] < klen)
            if CAUSAL:
                valid = valid & (token[None, :] <= token_q[:, None]+klen-qlen)
            score = tl.where(valid, score, -float('inf'))
            grouped = tl.reshape(score, (BM, BN//32, 32))
            maximum = tl.max(grouped, 2)
            if SCHEMA == 2:
                exponent = tl.ceil(maximum*1.4426950408889634-2.273018494406416)
            elif SCHEMA == 1:
                exponent = tl.ceil(maximum*1.4426950408889634-2.857980995127572)
            else:
                exponent = tl.floor(maximum*1.4426950408889634)
            usable = maximum != -float('inf')
            exponent = tl.where(usable, exponent, 0.)
            if SCHEMA == 0:
                reference = (exponent-2.)*0.6931471805599453
            else:
                reference = exponent*0.6931471805599453
            if SCHEMA == 2:
                reference -= 0.4054651081081644
            probability = tl.exp(grouped-reference[:, :, None])
            probability = ((probability.to(tl.int32, bitcast=True)+0x8000)&-65536).to(tl.float32, bitcast=True)
            quant = tl.where(probability >= 5., 6., tl.where(probability >= 3.5, 4.,
                    tl.where(probability >= 2.5, 3., tl.where(probability >= 1.75, 2.,
                    tl.where(probability >= 1.25, 1.5, tl.where(probability >= .75, 1.,
                    tl.where(probability >= .25, .5, 0.)))))))
            quant = tl.where(usable[:, :, None], quant, 0.)
            code = tl.minimum(tl.maximum(exponent-current[:, None]+(125. if SCHEMA == 0 else 124.), 0.), 254.)
            weights = tl.reshape(quant*tl.exp2(code-127.)[:, :, None], (BM, BN))
            value = tl.load(V+((kbase+token[:, None])*HK+khead)*D+dims[None, :],
                            token[:, None] < klen, 0.).to(tl.float16)
            group_pv = tl.dot(weights.to(tl.float16), value, group_pv)
            group_sum += tl.sum(weights, 1)
        history = history*alpha[:, None]+group_pv
        total = total*alpha+group_sum
        running = current
    index = pid*BM+tl.arange(0, BM)
    tl.store(PV+index[:, None]*D+dims[None, :], history)
    tl.store(SUM+index, total)
    tl.store(REF+index, running)


@triton.jit
def _combine_kernel(PV, SUM, REF, MAP, COUNT, O, LSE,
                    ROWS: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr, SCHEMA: tl.constexpr):
    row = tl.program_id(0)
    dims = tl.arange(0, D)
    count = tl.load(COUNT+row)
    maximum = tl.full((), -1e30, tl.float32)
    for split in range(count):
        index = tl.load(MAP+row*SPLITS+split)
        maximum = tl.maximum(maximum, tl.load(REF+index))
    total = tl.full((), 0., tl.float32)
    for split in range(count):
        index = tl.load(MAP+row*SPLITS+split)
        value = tl.load(SUM+index)
        factor = tl.where(value > 0., tl.exp2(tl.load(REF+index)-maximum), 0.)
        total += value*factor
    safe = tl.where(total > 0., total, 1.)
    output = tl.zeros((D,), tl.float32)
    for split in range(count):
        index = tl.load(MAP+row*SPLITS+split)
        pv = tl.load(PV+index*D+dims)
        if count == 1:
            output = libdevice.div_rn(pv, safe)
        else:
            value = tl.load(SUM+index)
            factor = tl.where(value > 0., tl.exp2(tl.load(REF+index)-maximum), 0.)
            weight = libdevice.div_rn(factor, safe)
            output += pv*weight
    output = tl.where(total > 0., output, 0.)
    output = ((output.to(tl.int32, bitcast=True)+0x8000)&-65536).to(tl.float32, bitcast=True)
    natural_lse = tl.log(safe)+(maximum+(0. if SCHEMA == 0 else 3.))*0.6931471805599453
    if SCHEMA == 2:
        natural_lse -= 0.4054651081081644
    lse = tl.where(total > 0., natural_lse, float('inf'))
    tl.store(O+row*D+dims, output)
    tl.store(LSE+row, lse)


def triton_mxfp4_golden_kernel(data, params, metadata, device_id=0):
    p = normalize_params(params)
    schedule = decode_sections(metadata, p)
    q, keys, values = decode_inputs(data, p)
    group = schedule['group']
    k_offsets, offset = [], 0
    for key in keys:
        k_offsets.append(offset); offset += len(key)
    descriptors, slots = [], [[] for _ in range(q.shape[0]*p['N_q'])]
    qs = effective_lengths(p)
    for batch, head, first, last, parts in iter_tiles(p, schedule):
        for start, stop in parts:
            for row in range(first, last, 16):
                end = min(row+16, last)
                pid = len(descriptors)
                descriptors.append([p['cu_seqlens_q'][batch], k_offsets[batch], qs[batch],
                    p['seqused_kv'][batch], head, row, end, start, stop])
                for local in range(end-row):
                    merged = row+local
                    output_row = (p['cu_seqlens_q'][batch]+merged//group)*p['N_q']+head*group+merged%group
                    slots[output_row].append(pid*16+local)
    if not descriptors:
        return torch.zeros_like(q, dtype=torch.bfloat16), torch.full(
            (p['N_q'], q.shape[0]), float('inf')) if p['enable_lse'] else torch.empty(0)
    device = f'npu:{device_id}'
    q = q.to(device).contiguous()
    k, v = torch.cat(keys).to(device).contiguous(), torch.cat(values).to(device).contiguous()
    desc = torch.tensor(descriptors, dtype=torch.int64, device=device)
    pv = torch.empty((len(descriptors)*16, p['D']), dtype=torch.float32, device=device)
    sums = torch.empty(len(descriptors)*16, dtype=torch.float32, device=device)
    refs = torch.empty_like(sums)
    _split_kernel[(len(descriptors),)](q, k, v, desc, pv, sums, refs,
        p['N_q'], p['N_kv'], p['D'], group, schedule['task_n'], schedule['tasks_per_group'],
        p['softmax_scale'], p['mask_mode'] == 3, p['quant_schema'],
        BN=32 if p['D'] == 256 else 64, enable_fp_fusion=False)
    max_splits = max(map(len, slots))
    indices = torch.tensor([entry+[-1]*(max_splits-len(entry)) for entry in slots], dtype=torch.int64, device=device)
    counts = torch.tensor(list(map(len, slots)), dtype=torch.int32, device=device)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=device)
    _combine_kernel[(len(slots),)](pv, sums, refs, indices, counts, output, lse,
        len(slots), p['D'], max_splits, p['quant_schema'], enable_fp_fusion=False)
    torch.npu.synchronize()
    return output.cpu(), lse.T.contiguous().cpu() if p['enable_lse'] else torch.empty(0)
