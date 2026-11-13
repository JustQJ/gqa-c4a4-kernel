#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""C4A4 PA_NZ vs C8A8 (mode1, FP8 E4M3) accuracy comparison.

Same case matrix as qfa_mxfp4_test (scope small); for every case the same
logical FP4 payload/scales are launched twice:
  - C4 PA_NZ: quant_mode=5, layout_kv='PA_NZ' (exact packed FP4 path)
  - C8 PA_BBND: quant_mode=1, FP8 E4M3 payload converted exactly from the
    same FP4 codes via perf_convert.convert_payload, descale multiplied by
    the C8 p_scale (default 448 as in the perf suite)
Both outputs are compared with the BBND-suite thresholds, and the gap
between the two device paths is reported.
"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qfa_mxfp4_test.quant_flash_attn_paramset_debug import get_cases as debug_cases  # noqa: E402
from qfa_mxfp4_test.quant_flash_attn_paramset_func_rdv import get_cases as func_cases  # noqa: E402
from qfa_mxfp4_test.common.quant_flash_attn_golden import generate_data  # noqa: E402
from qfa_mxfp4_nz_test.common.pa_nz_convert import convert_data_bbnd_to_nz  # noqa: E402


def build_c8_inputs(data, p, device, p_scale=448.0):
    """Exact FP4->FP8 expansion for the C8 (mode1) PA_BBND launch.

    FP4 codes are expanded losslessly (each E2M1 level maps to an exactly
    representable E4M3 byte); the E8M0 descales absorb p_scale.
    """
    from qfa_mxfp4_test.common.perf_convert import convert_payload

    def to_dev(t):
        return None if t is None else t.to(device)

    q = to_dev(data['q_fp4']).view(torch.uint8)
    k = to_dev(data['k_fp4']).view(torch.uint8)
    v = to_dev(data['v_fp4']).view(torch.uint8)
    q8 = convert_payload(q, 'c8', 1.0).view(torch.float8_e4m3fn)
    k8 = convert_payload(k, 'c8', 1.0).view(torch.float8_e4m3fn)
    v8 = convert_payload(v, 'c8', 1.0).view(torch.float8_e4m3fn)
    # descale: E8M0 pair bytes -> one float per 32 values, then scale by p_scale
    def descale_f32(pairs):
        e8 = pairs.view(torch.uint8)
        # even byte of each pair is the E8M0 scale used by expand (see perf_convert)
        byte = e8[..., 0::2].reshape(*pairs.shape[:-1]).contiguous()
        return torch.exp2(byte.float() - 127.)

    qs = descale_f32(data['dequant_scale_q']) * p_scale
    ks = descale_f32(data['dequant_scale_k']) * p_scale
    vs = descale_f32(data['v_descale']) * p_scale
    # C8 mode1 shapes: q_descale [T,N,D/32], k_descale [P,Bs,N,D/32], v_descale [P,Bs/32,N,D]
    total, heads, dimension = data['q_fp4'].shape[0], p['N_q'], p['D']
    qs = qs.reshape(total, heads, dimension // 32).to(device)
    pages, page, kv_heads = data['k_fp4'].shape[:3]
    ks = ks.reshape(pages, page, kv_heads, dimension // 32).to(device)
    vs_d = data['v_descale']  # [P, page/64, N, D, 2]
    v_groups = vs_d.shape[1]
    vs = vs_d.reshape(pages, v_groups, kv_heads, dimension, 2)
    vs = vs[..., 0].repeat_interleave(32, dim=1).reshape(pages, page, kv_heads, dimension).to(device)
    return dict(q=q8, k=k8, v=v8, q_descale=qs, k_descale=ks, v_descale=vs)


def run_c8(ops, data, p, device, p_scale):
    c8 = build_c8_inputs(data, p, device, p_scale)
    common = dict(cu_seqlens_q=data['cu_seqlens_q'], seqused_q=data['seqused_q'],
                  seqused_kv=data['seqused_kv'], max_seqlen_q=p['max_seqlen_q'],
                  max_seqlen_kv=p['max_seqlen_kv'], mask_mode=p['mask_mode'],
                  layout_q='TND', layout_q_descale='TND', layout_kv='PA_BBND', layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(p['N_q'], p['N_kv'], p['D'], 1, **common)
    mask = None
    if p['mask_mode'] == 3:
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(device)
    return ops.quant_flash_attn(
        c8['q'], c8['k'], c8['v'], c8['q_descale'], c8['k_descale'], c8['v_descale'],
        1, block_table=data['block_table_torch'].to(device), metadata=metadata, attn_mask=mask,
        p_scale=p_scale, softmax_scale=p['softmax_scale'], return_softmax_lse=p['enable_lse'],
        **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--package', default='cann_ops_transformer')
    ap.add_argument('--scope', choices=('small', 'full'), default='small')
    ap.add_argument('--suite', choices=('debug', 'func'), default='func')
    ap.add_argument('--p-scale', type=float, default=448.0)
    ap.add_argument('--output', default='c4nz_c8_report.json')
    ap.add_argument('--limit', type=int, default=0, help='Only run the first N cases (0=all)')
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    importlib.import_module(args.package)
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    ops = torch.ops.cann_ops_transformer

    cases = (debug_cases if args.suite == 'debug' else func_cases)(args.scope)
    if args.limit:
        cases = cases[:args.limit]

    from qfa_mxfp4_nz_test.common.pa_nz_golden import npu_mxfp4_fa_nz

    records = []
    passed = 0
    for case in cases:
        p = dict(case)
        data = generate_data(p)
        out_nz, lse_nz = npu_mxfp4_fa_nz(data, p, args.package, args.device)
        out_c8, lse_c8 = run_c8(ops, data, p, device, args.p_scale)
        torch.npu.synchronize()
        out_c8, lse_c8 = out_c8.cpu(), lse_c8.cpu()
        # thresholds from the BBND suite: attention atol 1e-4 rtol 0.0078125
        gap = (out_nz.float() - out_c8.float()).abs()
        rel = gap / out_c8.float().abs().clamp_min(2.0 ** -14 / 0.005)
        close = gap <= 1e-4 + 0.0078125 * out_c8.float().abs()
        match = close.float().mean().item()
        lse_gap = None
        if p['enable_lse']:
            lse_gap = (lse_nz - lse_c8).abs()
            lse_close = lse_gap <= 2.5e-5 + 0.005 * lse_c8.abs()
            lse_match = lse_close.float().mean().item()
        else:
            lse_match = None
        ok = match >= 0.995 and (lse_match is None or lse_match >= 0.995)
        passed += ok
        records.append(dict(name=p['name'], c4nz_c8_attention_match=match,
                            c4nz_c8_attention_max_abs=gap.max().item(),
                            c4nz_c8_attention_mean_abs=gap.mean().item(),
                            c4nz_c8_lse_match=lse_match,
                            c4nz_c8_lse_max_abs=None if lse_gap is None else lse_gap.max().item(),
                            ok=ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {p['name']} attn_match={match:.6f}"
              + (f" lse_match={lse_match:.6f}" if lse_match is not None else ''), flush=True)

    summary = dict(scope=args.scope, suite=args.suite, total=len(records), passed=passed,
                   p_scale=args.p_scale, device=args.device, records=records)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"TOTAL {passed}/{len(records)}", flush=True)
    return 0 if passed == len(records) else 1


if __name__ == '__main__':
    sys.exit(main())
