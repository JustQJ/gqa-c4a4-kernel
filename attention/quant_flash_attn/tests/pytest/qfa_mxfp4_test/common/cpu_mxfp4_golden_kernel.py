"""C4 metadata/core chunks, grouped PV, V2 updates and raw-PV FD merge."""
import math
import torch
from .quant_flash_attn_golden import decode_inputs, quantize_probability, round_bf16_away
from .mxfp4_schedule import decode_sections, effective_lengths, iter_tiles
from ..quant_flash_attn_paramset_common import normalize_params


def cpu_mxfp4_golden_kernel(data, params, metadata):
    p = normalize_params(params)
    schedule = decode_sections(metadata, p)
    q, keys, values = decode_inputs(data, p)
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], float('inf'), dtype=torch.float32)
    task_n, group = schedule['task_n'], schedule['group']
    tasks_per_group = schedule['tasks_per_group']
    pv_chunk_n = 128 if p['D'] == 256 and task_n == 256 else task_n
    log_x = math.log2(7.25/1.5) if p['p_quant_scheme'] == 'cx_delta' else math.log2(7.25)
    qs, ln2 = effective_lengths(p), math.log(2.)
    ln_delta = math.log(1.5) if p['p_quant_scheme'] == 'cx_delta' else 0.0
    is_ocp = p['quant_schema'] == 0
    scale_bias, lse_shift = (125., 0.) if is_ocp else (124., 3.)
    for batch, head, first, last, parts in iter_tiles(p, schedule):
        rows = torch.arange(first, last)
        tokens = rows//group
        heads = head*group+rows%group
        kh = head if group > 1 else head//(p['N_q']//p['N_kv'])
        qi = q[p['cu_seqlens_q'][batch]+tokens, heads]
        key, value, klen = keys[batch][:, kh], values[batch][:, kh], p['seqused_kv'][batch]
        splits = []
        for task_start, task_stop in parts:
            pv = torch.zeros((len(rows), p['D']), dtype=torch.float32)
            total = torch.zeros(len(rows), dtype=torch.float32)
            running = torch.full_like(total, -1e30)
            # All P payloads in the group share the reference. C2 accumulates them before one V2.
            for first_task in range(task_start, task_stop, tasks_per_group):
                stop_task = min(first_task+tasks_per_group, task_stop)
                lo, hi = first_task*task_n, min(stop_task*task_n, klen)
                padded = (hi-lo+31)//32*32
                score = torch.full((len(rows), padded), -float('inf'))
                score[:, :hi-lo] = (qi @ key[lo:hi].T)*p['softmax_scale']
                if p['mask_mode'] == 3:
                    visible = torch.arange(lo, hi)[None, :] <= tokens[:, None]+klen-qs[batch]
                    score[:, :hi-lo].masked_fill_(~visible, -float('inf'))
                grouped = score.reshape(len(rows), -1, 32)
                maxima = grouped.amax(-1)
                valid = torch.isfinite(maxima)
                scaled_max = torch.where(valid, maxima, 0.)*(1./ln2)
                exponent = torch.floor(scaled_max) if is_ocp else torch.ceil(scaled_max-log_x)
                exp_reference = (exponent-2.)*ln2 if is_ocp else exponent*ln2-ln_delta
                quant = quantize_probability(torch.exp(grouped-exp_reference[..., None]))
                quant = torch.where(valid[..., None], quant, 0.)
                current = torch.maximum(running, torch.where(valid, exponent, -1e30).amax(-1))
                # Production UpdatePScale uses exp(delta_R * ln2), including
                # its FP32 rounding, instead of replacing it with exp2 here.
                alpha = torch.where(running > -1e30, torch.exp((running-current)*ln2), 0.)
                code = (exponent-current[:, None]+scale_bias).clamp(0., 254.)
                weights = (quant*torch.exp2(code-127.)[..., None]).reshape(len(rows), padded)[:, :hi-lo]
                group_pv = torch.zeros_like(pv)
                group_sum = torch.zeros_like(total)
                for a in range(0, hi-lo, pv_chunk_n):
                    b = min(a+pv_chunk_n, hi-lo)
                    w = weights[:, a:b]
                    group_pv += w @ value[lo+a:lo+b]
                    group_sum += w.sum(-1)
                pv = pv*alpha[:, None]+group_pv
                total = total*alpha+group_sum
                running = current
            splits.append((pv, total, running))
        reference = torch.stack([x[2] for x in splits]).amax(0)
        factors = [torch.where(s > 0, torch.exp2(r-reference), 0.) for _, s, r in splits]
        denominator = torch.zeros_like(reference)
        for (_, s, _), a in zip(splits, factors):
            denominator += s*a
        safe = torch.where(denominator > 0, denominator, 1.)
        result = torch.zeros_like(splits[0][0])
        if len(splits) == 1:
            result = splits[0][0]/safe[:, None]
        else:
            # Current C4 FD merges raw PV using a_i/S, not normalized split O.
            for (pv, _, _), a in zip(splits, factors):
                result += pv*(a/safe)[:, None]
        output[p['cu_seqlens_q'][batch]+tokens, heads] = result
        lse[p['cu_seqlens_q'][batch]+tokens, heads] = torch.where(
            denominator > 0, torch.log(safe)+(reference+lse_shift)*ln2-ln_delta, float('inf'))
    return round_bf16_away(output).to(torch.bfloat16), lse.T.contiguous() if p['enable_lse'] else torch.empty(0)
