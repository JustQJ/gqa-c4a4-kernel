"""Decode the actual C4 metadata. Never silently fall back to another schedule."""
import importlib
from ..quant_flash_attn_paramset_common import normalize_params


def effective_lengths(p):
    return p['seqused_q'] if p['seqused_q'] is not None else [
        b-a for a, b in zip(p['cu_seqlens_q'], p['cu_seqlens_q'][1:])]


def fetch_metadata(data, params, package, device_id):
    import torch
    import torch_npu  # noqa: F401
    importlib.import_module(package)
    torch.npu.set_device(device_id)
    p = normalize_params(params)
    def device(name):
        value = data[name]
        return None if value is None else value.to(f'npu:{device_id}')
    metadata = torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
        p['N_q'], p['N_kv'], p['D'], 5,
        cu_seqlens_q=device('cu_seqlens_q'), seqused_q=device('seqused_q'),
        seqused_kv=device('seqused_kv'), max_seqlen_q=p['max_seqlen_q'],
        max_seqlen_kv=p['max_seqlen_kv'], mask_mode=p['mask_mode'], layout_q='TND',
        layout_q_descale=p['q_scale_layout'], layout_kv='PA_BBND', layout_out='TND')
    torch.npu.synchronize()
    return metadata.cpu().reshape(-1).tolist()


def decode_sections(metadata, params):
    p = normalize_params(params)
    meta = [int(x) for x in metadata]
    if len(meta) < 16:
        raise ValueError('C4 metadata header is missing')
    sections, is_fd, m_base, task_n, cores = meta[:5]
    valid_task_sizes = (128, 256) if p['D'] == 256 else (256,)
    if sections < 1 or cores < 1 or m_base != 128 or task_n not in valid_task_sizes:
        raise ValueError(f'Unsupported C4 metadata header: {meta[:5]}')
    if len(meta) < 16 + sections*cores*16:
        raise ValueError('C4 metadata FA records are truncated')
    head_num = p['N_kv'] if p['q_scale_layout'] == 'N2TGD' else p['N_q']
    group = p['N_q']//head_num
    qs, ks = effective_lengths(p), p['seqused_kv']
    counts = [(q*group+m_base-1)//m_base for q in qs]
    ranges = []
    for q, k, count in zip(qs, ks, counts):
        per_batch = []
        for mi in range(count):
            last_q = (min((mi+1)*m_base, q*group)-1)//group
            end = k if p['mask_mode'] == 0 else min(k, max(0, last_q+k-q+1))
            per_batch.append((0, (end+task_n-1)//task_n))
        ranges.append(per_batch)
    segments = [[[] for _ in range(counts[bn//head_num])] for bn in range(p['B']*head_num)]
    def advance(bn, mi):
        mi += 1
        if mi >= counts[bn//head_num]:
            bn, mi = bn+1, 0
        return bn, mi
    for section in range(sections):
        for core in range(cores):
            offset = 16+(section*cores+core)*16
            bn, mi, start, bn_end, mi_end, task_end = meta[offset:offset+6]
            end_state = (bn_end, mi_end, task_end)
            steps = 0
            while (bn, mi, start) != end_state:
                steps += 1
                if bn >= len(segments) or steps > 10_000_000:
                    raise ValueError('C4 metadata traversal exceeded its domain')
                if mi >= counts[bn//head_num]:
                    bn, mi = advance(bn, mi); start = 0
                    continue
                lower, upper = ranges[bn//head_num][mi]
                if upper <= lower or start >= upper:
                    bn, mi = advance(bn, mi); start = 0
                    continue
                stop = min(task_end, upper) if (bn, mi) == (bn_end, mi_end) else upper
                if stop <= start:
                    raise ValueError('C4 metadata contains a reversed segment')
                segments[bn][mi].append((start, stop))
                if stop >= upper:
                    bn, mi = advance(bn, mi); start = 0
                else:
                    start = stop
    for bn, blocks in enumerate(segments):
        for mi, parts in enumerate(blocks):
            current, expected = ranges[bn//head_num][mi]
            for start, stop in parts:
                if start != current:
                    raise ValueError(f'Noncontiguous C4 segments: bn={bn}, mi={mi}')
                current = stop
            if current != expected:
                raise ValueError(f'Incomplete C4 segments: bn={bn}, mi={mi}, {current}!={expected}')
    tasks_per_group = 2 if p['D'] == 256 and task_n == 256 else 16
    return dict(m_base=m_base, task_n=task_n, tasks_per_group=tasks_per_group,
                head_num=head_num, group=group, segments=segments, is_fd=bool(is_fd))


def iter_tiles(params, schedule):
    p = normalize_params(params)
    qs = effective_lengths(p)
    for bn, blocks in enumerate(schedule['segments']):
        batch, head = divmod(bn, schedule['head_num'])
        for mi, parts in enumerate(blocks):
            if not parts:
                continue
            start = mi*schedule['m_base']
            stop = min(start+schedule['m_base'], qs[batch]*schedule['group'])
            yield batch, head, start, stop, parts
