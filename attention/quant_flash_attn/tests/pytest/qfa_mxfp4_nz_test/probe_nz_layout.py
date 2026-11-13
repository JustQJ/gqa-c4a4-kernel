#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Minimal controlled experiment: same logical bytes, BBND vs PA_NZ launches.

For each probe only one (token, d1-block) cell of K/V is non-zero, so the
attention output directly reveals which logical element the kernel read from
the PA_NZ physical buffer. Compares against the BBND launch of the same data.
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qfa_mxfp4_test.common.quant_flash_attn_golden import (  # noqa: E402
    generate_data, prepare_npu_inputs, invoke_operator)
from qfa_mxfp4_nz_test.common.pa_nz_convert import convert_data_bbnd_to_nz  # noqa: E402


def run_layout(data, params, device_id, package, layout):
    import torch_npu  # noqa: F401
    importlib.import_module(package)
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)
    prepared = prepare_npu_inputs(data, device)
    if layout == 'PA_NZ':
        prepared = prepare_npu_inputs(convert_data_bbnd_to_nz(data), device)
    ops = torch.ops.cann_ops_transformer
    from qfa_mxfp4_test.quant_flash_attn_paramset_common import normalize_params
    p = normalize_params(params)
    common = dict(cu_seqlens_q=data['cu_seqlens_q'], seqused_q=data['seqused_q'],
                  seqused_kv=data['seqused_kv'], max_seqlen_q=p['max_seqlen_q'],
                  max_seqlen_kv=p['max_seqlen_kv'], mask_mode=p['mask_mode'],
                  layout_q='TND', layout_q_descale=p['q_scale_layout'],
                  layout_kv=layout, layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(p['N_q'], p['N_kv'], p['D'], 5, **common)
    out, lse = ops.quant_flash_attn(
        *(prepared[name] for name in ('q_fp4', 'k_fp4', 'v_fp4',
                                      'dequant_scale_q', 'dequant_scale_k', 'v_descale')),
        5, block_table=prepared['block_table_torch'], metadata=metadata, attn_mask=None,
        p_scale=None, softmax_scale=p['softmax_scale'], return_softmax_lse=True,
        quant_schema=2, **common)
    torch.npu.synchronize()
    return out.cpu(), lse.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--package', default='cann_ops_transformer')
    args = ap.parse_args()

    # Minimal case: B=1, Q=1, KV=128(one page), N_q=N_kv=1, D=128, page=128
    params = dict(name='probe', B=1, N_q=1, N_kv=1, D=128,
                  cu_seqlens_q=[0, 1], seqused_q=[1], seqused_kv=[128],
                  max_seqlen_q=1, max_seqlen_kv=128, block_size=128,
                  mask_mode=0, q_scale_layout='TND', enable_lse=True,
                  quant_schema=2)
    data = generate_data(params)

    # Overwrite K payload: only token 3 has non-zero codes in d1-block 0 (bytes 0..31)
    k = torch.zeros_like(data['k_fp4'])
    k[0, 3, 0, :32] = torch.randint(0, 256, (32,), dtype=torch.uint8)
    data['k_fp4'] = k
    # Overwrite V: all pages same known pattern: token t value = 1.0 everywhere
    v = torch.zeros_like(data['v_fp4'])
    v[0, :, 0, :] = 0x22  # FP4 codes of two 1.0 values
    data['v_fp4'] = v

    out_b, lse_b = run_layout(data, params, args.device, args.package, 'PA_BBND')
    out_n, lse_n = run_layout(data, params, args.device, args.package, 'PA_NZ')
    diff = (out_b.float() - out_n.float()).abs()
    print(f'BBND out[0,0,:8]: {out_b[0, 0, :8].tolist()}')
    print(f'NZ   out[0,0,:8]: {out_n[0, 0, :8].tolist()}')
    print(f'attention max abs diff: {diff.max().item():.6f}')
    print(f'LSE bbnd={lse_b.flatten().tolist()} nz={lse_n.flatten().tolist()}')
    print('LAYOUT MATCH' if diff.max().item() == 0.0 and torch.equal(lse_b, lse_n)
          else 'LAYOUT MISMATCH')


if __name__ == '__main__':
    main()
