"""FP64 attention on decoded MXFP4 Q/K/V; P is never quantized or cast down."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import torch
from .quant_flash_attn_golden import decode_inputs
from .mxfp4_schedule import effective_lengths
from ..quant_flash_attn_paramset_common import normalize_params


def cpu_mxfp4_golden_fp64(data, params):
    p = normalize_params(params)
    q, keys, values = decode_inputs(data, p)
    q = q.double()
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], float('inf'), dtype=torch.float64)
    groups = p['N_q']//p['N_kv']
    def compute_head(job):
        batch, head, qlen, klen, key, value = job
        start = p['cu_seqlens_q'][batch]
        # Heads write disjoint output slices. Parallelizing heads preserves
        # every row's KV traversal and FP64 accumulation order.
        kh = head//groups
        for first in range(0, qlen, 64):
            count = min(64, qlen-first)
            qi = q[start+first:start+first+count, head]
            maximum = torch.full((count,), -float('inf'), dtype=torch.float64)
            denominator = torch.zeros(count, dtype=torch.float64)
            numerator = torch.zeros((count, p['D']), dtype=torch.float64)
            for lo in range(0, klen, 2048):
                hi = min(lo+2048, klen)
                score = (qi @ key[lo:hi, kh].T)*p['softmax_scale']
                if p['mask_mode'] == 3:
                    visible = torch.arange(lo, hi)[None, :] <= torch.arange(first, first+count)[:, None]+klen-qlen
                    score.masked_fill_(~visible, -float('inf'))
                current = torch.maximum(maximum, score.amax(-1))
                safe = torch.where(torch.isfinite(current), current, 0.)
                alpha = torch.where(torch.isfinite(maximum), torch.exp(maximum-safe), 0.)
                probability = torch.exp(score-safe[:, None])
                numerator = numerator*alpha[:, None]+probability @ value[lo:hi, kh]
                denominator = denominator*alpha+probability.sum(-1)
                maximum = current
            safe_sum = torch.where(denominator > 0, denominator, 1.)
            output[start+first:start+first+count, head] = numerator/safe_sum[:, None]
            lse[start+first:start+first+count, head] = torch.where(
                denominator > 0, maximum+torch.log(safe_sum), float('inf'))
    workers = min(torch.get_num_threads(), p['N_q'])
    with (ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        for batch, (qlen, klen) in enumerate(zip(effective_lengths(p), p['seqused_kv'])):
            if not qlen or not klen:
                continue
            # Retain only one batch's FP64 KV to bound peak host memory.
            key, value = keys[batch].double(), values[batch].double()
            jobs = [(batch, head, qlen, klen, key, value) for head in range(p['N_q'])]
            if pool is not None:
                list(pool.map(compute_head, jobs))
            else:
                for job in jobs:
                    compute_head(job)
    return output, lse.T.contiguous() if p['enable_lse'] else torch.empty(0, dtype=torch.float64)
