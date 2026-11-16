#!/usr/bin/env python3
"""Generate C4A4 PA test inputs (BBND + PA_NZ) as raw .bin files + .ini configs.

The data semantics mirror qfa_mxfp4_test/common/quant_flash_attn_golden.py:
  - random uint8 FP4 payloads, E8M0 scales (Q/K: 122..125, V: 124..127)
  - random physical page permutation with 3 extra unused pages
  - identical canonical BBND tensors for both layouts; NZ tensors are repacked
    with convert_bbnd_to_pa_nz so the logical Q/K/V match exactly.

Output per case directory:
  <case>/{q,k,v,qs,ks,vs,block_table,cu_q,seqused_q,seqused_kv}.bin (BBND)
  <case>/{k_nz,v_nz,ks_nz,vs_nz}.bin                            (PA_NZ)
  <case>/bbnd.ini, <case>/nz.ini                                (driver configs)
  <case>/golden.npz                                             (CPU reference)
"""
import argparse
import json
import os
import numpy as np

LEVELS = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float64)


def unpack_fp4(payload):
    codes = np.stack((payload & 15, payload >> 4), axis=-1)
    codes = codes.reshape(*payload.shape[:-1], payload.shape[-1] * 2)
    return LEVELS[codes & 7] * np.where((codes & 8) != 0, -1., 1.)


def decode_scale(payload):
    return np.exp2(payload.astype(np.float64) - 127.)


def convert_bbnd_to_pa_nz(k, v, ks, vs):
    bn, bs, n, d2 = k.shape
    dv = vs.shape[3]
    return dict(
        k_nz=k.reshape(bn, bs, n, d2 // 32, 32).transpose(0, 2, 3, 1, 4).copy(),
        v_nz=v.reshape(bn, bs, n, d2 // 32, 32).transpose(0, 2, 3, 1, 4).copy(),
        ks_nz=ks.reshape(bn, bs // 16, 16, n, d2 // 32, 2).transpose(0, 3, 1, 4, 2, 5).copy(),
        vs_nz=vs.reshape(bn, bs // 64, n, dv // 16, 16, 2).transpose(0, 2, 3, 1, 4, 5).copy(),
    )


def convert_bbnd_to_pa_bnbd(k, v, ks, vs):
    """Convert PA_BBND layout [Bn, Bs, N, D] to PA_BNBD layout [Bn, N, Bs, D]."""
    bn, bs, n, d2 = k.shape
    return dict(
        k_bnbd=k.transpose(0, 2, 1, 3).copy(),
        v_bnbd=v.transpose(0, 2, 1, 3).copy(),
        ks_bnbd=ks.transpose(0, 2, 1, 3, 4).copy(),
        vs_bnbd=vs.transpose(0, 2, 1, 3, 4).copy(),
    )


def generate_case(scenario, n_q, n_kv, d, q_scale_layout, mask_mode, seed=421):
    qlens, klens, page = scenario['qlens'], scenario['klens'], scenario['page']
    used_q = scenario.get('used_q', qlens if q_scale_layout == 'TND' else None)
    rng = np.random.default_rng(seed)
    cu = [0] + list(np.cumsum(qlens))
    total = cu[-1]
    counts = [(k + page - 1) // page for k in klens]
    pages = sum(counts) + 3
    q = rng.integers(0, 256, (total, n_q, d // 2), dtype=np.uint8)
    k = rng.integers(0, 256, (pages, page, n_kv, d // 2), dtype=np.uint8)
    v = rng.integers(0, 256, (pages, page, n_kv, d // 2), dtype=np.uint8)
    qs = rng.integers(122, 126, (total, n_q, d // 64, 2), dtype=np.uint8)
    ks = rng.integers(122, 126, (pages, page, n_kv, d // 64, 2), dtype=np.uint8)
    vs = rng.integers(124, 128, (pages, page // 64, n_kv, d, 2), dtype=np.uint8)
    if q_scale_layout == 'N2TGD':
        qs = qs.reshape(total, n_kv, n_q // n_kv, d // 64, 2).transpose(1, 0, 2, 3, 4).copy()
    table = np.zeros((len(klens), max(1, max(counts))), dtype=np.int32)
    permutation = rng.permutation(pages)
    offset = 0
    for b, count in enumerate(counts):
        table[b, :count] = permutation[offset:offset + count].astype(np.int32)
        offset += count
    return dict(q=q, k=k, v=v, qs=qs, ks=ks, vs=vs, block_table=table,
                cu_q=np.array(cu, dtype=np.int32),
                seqused_q=None if used_q is None else np.array(used_q, dtype=np.int32),
                seqused_kv=np.array(klens, dtype=np.int32))


def decode_inputs(data, n_q, n_kv, d, page, q_scale_layout, klens):
    qs = data['qs']
    if q_scale_layout == 'N2TGD':
        qs = qs.transpose(1, 0, 2, 3, 4).copy()
    qs = qs.reshape(qs.shape[0], n_q, d // 32)
    query = unpack_fp4(data['q']) * np.repeat(decode_scale(qs), 32, axis=-1)
    keys, values = [], []
    for b, length in enumerate(klens):
        count = (length + page - 1) // page
        ids = data['block_table'][b, :count].astype(np.int64)
        k = unpack_fp4(data['k'][ids])
        ks = decode_scale(data['ks'][ids]).reshape(count, page, n_kv, d // 32)
        keys.append((k * np.repeat(ks, 32, axis=-1)).reshape(count * page, n_kv, d)[:length])
        v = unpack_fp4(data['v'][ids])
        vs = decode_scale(data['vs'][ids]).transpose(0, 1, 4, 2, 3)
        vs = vs.reshape(count, page // 32, n_kv, d).repeat(32, axis=1)
        values.append((v * vs).reshape(count * page, n_kv, d)[:length])
    return query, keys, values


def cpu_golden(data, n_q, n_kv, d, page, q_scale_layout, mask_mode, klens, cu, used_q, softmax_scale):
    """FP32 reference without P quantization (absolute accuracy yardstick)."""
    query, keys, values = decode_inputs(data, n_q, n_kv, d, page, q_scale_layout, klens)
    total, _, _ = query.shape
    output = np.zeros((total, n_q, d), dtype=np.float64)
    lse = np.full((total, n_q), np.inf, dtype=np.float64)
    used_q = used_q if used_q is not None else [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    groups = n_q // n_kv
    for b, (qlen, klen) in enumerate(zip(used_q, klens)):
        if not qlen or not klen:
            continue
        start = cu[b]
        key, value = keys[b], values[b]
        for head in range(n_q):
            kvh = head // groups
            q_h = query[start:start + qlen, head]  # [qlen, d]
            score = (q_h @ key[:, kvh].T) * softmax_scale  # [qlen, klen]
            if mask_mode == 3:
                valid = np.arange(klen)[None, :] <= np.arange(qlen)[:, None] + klen - qlen
                score = np.where(valid, score, -np.inf)
            m = score.max(axis=-1, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.)
            exp = np.exp(score - m)
            exp = np.where(np.isfinite(score), exp, 0.)
            denom = exp.sum(axis=-1, keepdims=True)
            safe = np.where(denom > 0, denom, 1.)
            output[start:start + qlen, head] = (exp @ value[:, kvh]) / safe
            lse[start:start + qlen, head] = np.where(denom[:, 0] > 0,
                                                     np.log(denom[:, 0]) + m[:, 0], np.inf)
    return output, lse.T.copy()  # lse -> [n_q, total]


def bf16_round(values):
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x8000) & 0xFFFF0000).view(np.float32)


def write_ini(path, cfg):
    with open(path, 'w') as f:
        for key, value in cfg.items():
            f.write(f'{key}={value}\n')


SCENARIOS = {
    'smoke': dict(qlens=[17], klens=[257], page=256),
    'empty_and_q_padding': dict(qlens=[5, 0, 3], used_q=[3, 0, 3], klens=[0, 33, 1], page=128),
    'page_tail': dict(qlens=[3, 9], klens=[129, 513], page=512),
    'short1': dict(qlens=[1], klens=[1], page=128),
    'short22': dict(qlens=[22], klens=[22], page=128),
    'mixed': dict(qlens=[1, 3, 22, 129], klens=[11, 32771, 65, 259], page=128),
    'long_prefill': dict(qlens=[129], klens=[32771], page=512),
    'mtp2_kv32768_b1': dict(qlens=[2], klens=[32768], page=256),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--cases', default='smoke,empty_and_q_padding,page_tail')
    parser.add_argument('--n-q', type=int, default=8)
    parser.add_argument('--n-kv', type=int, default=1)
    parser.add_argument('--d', type=int, default=256, choices=(128, 256))
    parser.add_argument('--seed', type=int, default=421)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = []
    for case_name in args.cases.split(','):
        scenario = SCENARIOS[case_name]
        for q_scale_layout in ('TND', 'N2TGD'):
            for mask_mode in (0, 3):
                name = f'{case_name}_{q_scale_layout}_M{mask_mode}'
                data = generate_case(scenario, args.n_q, args.n_kv, args.d, q_scale_layout, mask_mode, args.seed)
                cdir = os.path.join(args.out, name)
                os.makedirs(cdir, exist_ok=True)
                for tensor_name in ('q', 'k', 'v', 'qs', 'ks', 'vs', 'block_table', 'cu_q', 'seqused_kv'):
                    data[tensor_name].tofile(os.path.join(cdir, tensor_name + '.bin'))
                if data['seqused_q'] is not None:
                    data['seqused_q'].tofile(os.path.join(cdir, 'seqused_q.bin'))
                nz = convert_bbnd_to_pa_nz(data['k'], data['v'], data['ks'], data['vs'])
                for tensor_name, tensor in nz.items():
                    tensor.tofile(os.path.join(cdir, tensor_name + '.bin'))
                bnbd = convert_bbnd_to_pa_bnbd(data['k'], data['v'], data['ks'], data['vs'])
                for tensor_name, tensor in bnbd.items():
                    tensor.tofile(os.path.join(cdir, tensor_name + '.bin'))
                golden_out, golden_lse = cpu_golden(
                    data, args.n_q, args.n_kv, args.d, scenario['page'], q_scale_layout, mask_mode,
                    scenario['klens'], list(data['cu_q']),
                    None if data['seqused_q'] is None else list(data['seqused_q']), args.d ** -0.5)
                np.savez(os.path.join(cdir, 'golden.npz'), out=golden_out, lse=golden_lse)
                common = dict(
                    D=args.d, N_Q=args.n_q, N_KV=args.n_kv, PAGE=scenario['page'], MASK_MODE=mask_mode,
                    QUANT_MODE=5, QUANT_SCHEMA=2, SOFTMAX_SCALE=repr(args.d ** -0.5),
                    LAYOUT_Q='TND', LAYOUT_Q_DESCALE=q_scale_layout, LAYOUT_OUT='TND',
                    MAX_SEQLEN_Q=max(scenario['qlens']), MAX_SEQLEN_KV=max(scenario['klens']),
                    B=len(scenario['qlens']), ENABLE_LSE=1,
                    # FP4 tensors use LOGICAL element shapes (D, not packed bytes);
                    # the driver reads prod(shape)/2 bytes for FP4 payloads.
                    Q_SHAPE=','.join(map(str, (total := data['q'].shape[0], args.n_q, args.d))),
                    QS_SHAPE=','.join(map(str, data['qs'].shape)),
                    BLOCK_TABLE_SHAPE=','.join(map(str, data['block_table'].shape)),
                    CU_Q_SHAPE=str(data['cu_q'].shape[0]),
                    SEQUSED_KV_SHAPE=str(data['seqused_kv'].shape[0]),
                    SEQUSED_Q_SHAPE='0' if data['seqused_q'] is None else str(data['seqused_q'].shape[0]),
                    OUT_SHAPE=','.join(map(str, golden_out.shape)),
                    LSE_SHAPE=','.join(map(str, golden_lse.shape)),
                    Q_BIN='q.bin',
                )
                for layout, suffix in (('PA_BBND', 'bbnd'), ('PA_NZ', 'nz'), ('PA_BNBD', 'bnbd')):
                    cfg = dict(common)
                    cfg['LAYOUT_KV'] = layout
                    if layout == 'PA_BBND':
                        cfg.update(K_SHAPE=','.join(map(str, (data['k'].shape[0], scenario['page'], args.n_kv, args.d))),
                                   V_SHAPE=','.join(map(str, (data['v'].shape[0], scenario['page'], args.n_kv, args.d))),
                                   KS_SHAPE=','.join(map(str, data['ks'].shape)),
                                   VS_SHAPE=','.join(map(str, data['vs'].shape)),
                                   K_BIN='k.bin', V_BIN='v.bin', KS_BIN='ks.bin', VS_BIN='vs.bin')
                    elif layout == 'PA_NZ':
                        cfg.update(K_SHAPE=','.join(map(str, (data['k'].shape[0], args.n_kv, args.d // 64,
                                                              scenario['page'], 64))),
                                   V_SHAPE=','.join(map(str, (data['v'].shape[0], args.n_kv, args.d // 64,
                                                              scenario['page'], 64))),
                                   KS_SHAPE=','.join(map(str, nz['ks_nz'].shape)),
                                   VS_SHAPE=','.join(map(str, nz['vs_nz'].shape)),
                                   K_BIN='k_nz.bin', V_BIN='v_nz.bin', KS_BIN='ks_nz.bin', VS_BIN='vs_nz.bin')
                    else:  # PA_BNBD
                        cfg.update(K_SHAPE=','.join(map(str, bnbd['k_bnbd'].shape)),
                                   V_SHAPE=','.join(map(str, bnbd['v_bnbd'].shape)),
                                   KS_SHAPE=','.join(map(str, bnbd['ks_bnbd'].shape)),
                                   VS_SHAPE=','.join(map(str, bnbd['vs_bnbd'].shape)),
                                   K_BIN='k_bnbd.bin', V_BIN='v_bnbd.bin', KS_BIN='ks_bnbd.bin', VS_BIN='vs_bnbd.bin')
                    write_ini(os.path.join(cdir, suffix + '.ini'), cfg)
                index.append(name)
                print(f'generated {name}', flush=True)
    with open(os.path.join(args.out, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    print(f'ALL {len(index)} cases generated under {args.out}')


if __name__ == '__main__':
    main()
