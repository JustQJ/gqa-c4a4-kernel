#!/usr/bin/env python3
"""Generate C8A8 (quant_mode=1, MXFP8) PA_NZ inputs from existing C4A4 data.

Key property: every FP4 E2M1 level {0,.5,1,1.5,2,3,4,6} is exactly
representable in FP8 E4M3, and C4/C8 share the same per-32-group E8M0 scale
semantics. Therefore C8 payloads are produced by a LOSSLESS nibble->byte
lookup, keeping Q/K/V numerically identical to C4. Only the P quantization
differs (C4 P=MXFP4 vs C8 P=FP8_E4M3), which is exactly what the C4-vs-C8
precision comparison is meant to isolate.

Inputs : <case>/{q,qs,k,v,ks,vs,block_table,cu_q,seqused_q,seqused_kv}.bin + nz.ini
Outputs: <case>/{q_fp8,k_fp8,v_fp8}.bin (+ NZ variants), c8_nz.ini / c8_bbnd.ini
"""
import argparse
import json
import os
import numpy as np

# FP4 nibble (E2M1) -> FP8 E4M3 byte, exact for all 16 codes.
FP4_TO_FP8 = np.array([
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
], dtype=np.uint8)


def fp4_bin_to_fp8(payload):
    lo = FP4_TO_FP8[payload & 15]
    hi = FP4_TO_FP8[payload >> 4]
    return np.stack((lo, hi), axis=-1).reshape(*payload.shape[:-1], payload.shape[-1] * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    with open(os.path.join(args.data, 'index.json')) as f:
        cases = json.load(f)
    for name in cases:
        cdir = os.path.join(args.data, name)
        # C4 FP4 packed payloads (uint8) -> C8 FP8 payloads (uint8, 1 byte/elem)
        q = np.fromfile(os.path.join(cdir, 'q.bin'), dtype=np.uint8)
        k = np.fromfile(os.path.join(cdir, 'k.bin'), dtype=np.uint8)
        v = np.fromfile(os.path.join(cdir, 'v.bin'), dtype=np.uint8)
        q_fp8 = fp4_bin_to_fp8(q)
        k_fp8 = fp4_bin_to_fp8(k)
        v_fp8 = fp4_bin_to_fp8(v)
        q_fp8.tofile(os.path.join(cdir, 'q_fp8.bin'))
        k_fp8.tofile(os.path.join(cdir, 'k_fp8.bin'))
        v_fp8.tofile(os.path.join(cdir, 'v_fp8.bin'))
        # NZ payload: [Bn,Bs,N,D] fp8 -> [Bn,N,D/32,Bs,32]
        # BNBD payload: [Bn,Bs,N,D] fp8 -> [Bn,N,Bs,D] (pure axis transpose)
        bnbd = None
        for layout in ('bbnd', 'nz', 'bnbd'):
            cfg_path = os.path.join(cdir, layout + '.ini')
            cfg = {}
            with open(cfg_path) as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        cfg[key] = value
            n_kv, d, page = int(cfg['N_KV']), int(cfg['D']), int(cfg['PAGE'])
            bn = k_fp8.size // (page * n_kv * d)
            if layout == 'nz':
                k_nz = k_fp8.reshape(bn, page, n_kv, d // 32, 32).transpose(0, 2, 3, 1, 4).copy()
                v_nz = v_fp8.reshape(bn, page, n_kv, d // 32, 32).transpose(0, 2, 3, 1, 4).copy()
                k_nz.tofile(os.path.join(cdir, 'k_fp8_nz.bin'))
                v_nz.tofile(os.path.join(cdir, 'v_fp8_nz.bin'))
            elif layout == 'bnbd':
                # BNBD: [Bn, Bs, N, D] -> [Bn, N, Bs, D]
                k_bnbd = k_fp8.reshape(bn, page, n_kv, d).transpose(0, 2, 1, 3).copy()
                v_bnbd = v_fp8.reshape(bn, page, n_kv, d).transpose(0, 2, 1, 3).copy()
                k_bnbd.tofile(os.path.join(cdir, 'k_fp8_bnbd.bin'))
                v_bnbd.tofile(os.path.join(cdir, 'v_fp8_bnbd.bin'))
            # scales (identical values/layout as C4) just get relinked
            cfg['QUANT_MODE'] = '1'
            cfg['QUANT_SCHEMA'] = '2'
            cfg['Q_BIN'] = 'q_fp8.bin'
            cfg['Q_FP8_SHAPE'] = cfg['Q_SHAPE']
            if layout == 'bbnd':
                cfg['K_BIN'] = 'k_fp8.bin'
                cfg['V_BIN'] = 'v_fp8.bin'
            elif layout == 'nz':
                cfg['K_BIN'] = 'k_fp8_nz.bin'
                cfg['V_BIN'] = 'v_fp8_nz.bin'
                cfg['K_SHAPE'] = ','.join(map(str, (bn, n_kv, d // 32, page, 32)))
                cfg['V_SHAPE'] = ','.join(map(str, (bn, n_kv, d // 32, page, 32)))
            else:  # bnbd
                cfg['K_BIN'] = 'k_fp8_bnbd.bin'
                cfg['V_BIN'] = 'v_fp8_bnbd.bin'
                cfg['K_SHAPE'] = ','.join(map(str, (bn, n_kv, page, d)))
                cfg['V_SHAPE'] = ','.join(map(str, (bn, n_kv, page, d)))
                # scales for BNBD come from the bnbd variants (same values, layout [Bn,N,Bs,...])
                if os.path.exists(os.path.join(cdir, 'ks_bnbd.bin')):
                    cfg['KS_BIN'] = 'ks_bnbd.bin'
                    cfg['VS_BIN'] = 'vs_bnbd.bin'
            with open(os.path.join(cdir, f'c8_{layout}.ini'), 'w') as f:
                for key, value in cfg.items():
                    f.write(f'{key}={value}\n')
        print(f'c8 inputs generated for {name}', flush=True)
    print('ALL C8 INPUTS DONE')


if __name__ == '__main__':
    main()
