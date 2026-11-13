#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Diagnose the PA_NZ launch against the BBND launch of the same data.

Uses exactly the same generation and invocation path as the test suites; the
only difference is the physical KV arrangement and layout_kv string. Reports
per-head / per-token error patterns to localize layout mismatches.
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qfa_mxfp4_test.common.quant_flash_attn_golden import (  # noqa: E402
    generate_data, prepare_npu_inputs)
from qfa_mxfp4_test.quant_flash_attn_paramset_debug import get_cases  # noqa: E402
from qfa_mxfp4_nz_test.common.pa_nz_convert import convert_data_bbnd_to_nz  # noqa: E402


def launch(data, p, device_id, package, layout):
    import torch_npu  # noqa: F401
    importlib.import_module(package)
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)
    prepared = prepare_npu_inputs(convert_data_bbnd_to_nz(data) if layout == 'PA_NZ' else data, device)
    ops = torch.ops.cann_ops_transformer
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
    ap.add_argument('--case', default='smoke')
    args = ap.parse_args()

    case = next(c for c in get_cases('small') if args.case in c['name'])
    data = generate_data(case)

    out_b, lse_b = launch(data, case, args.device, args.package, 'PA_BBND')
    out_n, lse_n = launch(data, case, args.device, args.package, 'PA_NZ')

    diff = (out_b.float() - out_n.float()).abs()
    print(f"case: {case['name']}")
    print(f"attention: shape={tuple(out_b.shape)} max_abs={diff.max().item():.6f} "
          f"mean_abs={diff.mean().item():.6f}")
    lse_diff = (lse_b - lse_n).abs()
    print(f"lse: max_abs={lse_diff.max().item():.6f}")

    # error profile by token
    per_token = diff.amax(dim=(1, 2))
    bad = (per_token > 1e-3).nonzero().flatten().tolist()
    print(f"tokens with error>1e-3: {bad[:20]}{'...' if len(bad) > 20 else ''} "
          f"({len(bad)}/{len(per_token)})")
    # error profile by head
    per_head = diff.amax(dim=(0, 2))
    print(f"per-head max err: {[f'{x:.4f}' for x in per_head.tolist()]}")
    # error profile by D channel
    per_d = diff.amax(dim=(0, 1))
    print(f"D channels with err>1e-3: {(per_d > 1e-3).nonzero().flatten().tolist()}")
    print('MATCH' if diff.max().item() == 0.0 else 'MISMATCH')


if __name__ == '__main__':
    main()
