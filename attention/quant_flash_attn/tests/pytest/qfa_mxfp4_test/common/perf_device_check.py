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

"""Device preflight for basic C4/C8/BF16 and migrated MXFP8 C4/C8 cases."""
import argparse
import gc
import importlib
import itertools
import json
from pathlib import Path

from ..quant_flash_attn_paramset_common import normalize_params
from ..quant_flash_attn_paramset_perf_rdv import get_cases, sequence_lengths


def get_check_cases(case_set='basic'):
    if case_set not in ('basic', 'mxfp8', 'all'):
        raise ValueError('case-set must be basic, mxfp8 or all')
    cases = []
    if case_set in ('basic', 'all'):
        base = get_cases('small')[0]
        for d, (kind, q) in itertools.product((128, 256), (('prefill', 17), ('decode', 4))):
            case = dict(base, name=f'preflight_D{d}_{kind}', D=d, B=2, Q=q, KV=257,
                        kind=kind, q_scale_layout='TND' if kind == 'prefill' else 'N2TGD')
            cases.append(('basic', case))
    if case_set in ('mxfp8', 'all'):
        base = get_cases(case_set='mxfp8')[0]
        for d, layout, mask in itertools.product((128, 256), ('TND', 'N2TGD'), (0, 3)):
            case = dict(base, name=f'preflight_D{d}_{layout}_mask{mask}',
                        B=3, D=d, N_q=20, N_kv=2, Q=17, KV=1025, cu_seqlens_q=[0,0,4,21],
                        seqused_q=[0,4,17], seqused_kv=[512,777,1025],
                        max_seqlen_q=17, max_seqlen_kv=1025, q_scale_layout=layout,
                        mask_mode=mask, enable_lse=True, c8_p_scale=1., skip_reason='')
            cases.append(('mxfp8', case))
    return cases


def reference_params(case):
    cu, used_q, used_kv = sequence_lengths(case)
    return normalize_params(dict(
        B=case['B'], N_q=case['N_q'], N_kv=case['N_kv'], D=case['D'],
        cu_seqlens_q=cu, seqused_q=used_q, seqused_kv=used_kv,
        block_size=case['page'], mask_mode=case['mask_mode'],
        q_scale_layout=case['q_scale_layout'], enable_lse=case['enable_lse']))


def check_fp4_codes(device):
    import torch
    from .perf_convert import convert_payload

    # Cover all 16 E2M1 codes across the packed nibble positions, including signed zero.
    codes = torch.arange(16, dtype=torch.uint8)
    raw = (codes[::2] | (codes[1::2] << 4)).to(device)
    expected8 = torch.tensor([0,48,56,60,64,68,72,76,128,176,184,188,192,196,200,204], dtype=torch.uint8)
    assert torch.equal(convert_payload(raw, 'c8', 1.).view(torch.uint8).cpu(), expected8)
    values = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
    for scale in (.0625, .25):
        assert torch.equal(convert_payload(raw, 'bf16', scale).cpu().float(), values*scale)


def check_references(group, case, device):
    import torch
    from .perf_inputs import generate_inputs, prepare_inputs
    from .perf_runner import prepare_call
    from .cpu_mxfp4_golden_kernel import cpu_mxfp4_golden_kernel
    from .quant_flash_attn_golden import decode_inputs

    source = generate_inputs(case)
    if group == 'mxfp8':
        assert source['raw'][0].shape[0] == 21
        assert source['raw'][1].shape[0] == 6
        assert source['table'].tolist()[0][1:] == [-1,-1]
    data = prepare_inputs(source, case, 'c4', device)
    call, metadata = prepare_call(data, case, 'c4')
    actual_c4 = call()[0].cpu()
    metadata_cpu = metadata.cpu().reshape(-1).tolist()
    cpu_data = dict(q_fp4=source['raw'][0], k_fp4=source['raw'][1], v_fp4=source['raw'][2],
        dequant_scale_q=data['qs'].view(torch.uint8).cpu(), dequant_scale_k=data['ks'].view(torch.uint8).cpu(),
        v_descale=data['vs'].view(torch.uint8).cpu(), block_table_torch=source['table'])
    params = reference_params(case)
    del call, metadata, data
    gc.collect()
    torch.npu.empty_cache()
    reference_c4, _ = cpu_mxfp4_golden_kernel(cpu_data, params, metadata_cpu)
    assert torch.allclose(actual_c4.float(), reference_c4.float(), atol=1e-4, rtol=.0078125), case['name']
    report = dict(case=case['name'], group=group, c4_reference=True,
                  c4_reference_max_abs=float((actual_c4.float()-reference_c4.float()).abs().max()))
    if group == 'basic':
        q, kv, d = case['Q'], case['KV'], case['D']
        query, keys, vals = decode_inputs(cpu_data, params)
        reference_bf = torch.empty_like(query)
        for bi in range(case['B']):
            for hi in range(case['N_q']):
                scores = query[bi*q:(bi+1)*q,hi] @ keys[bi][:,0].T * d**-.5
                scores.masked_fill_(torch.arange(kv)[None,:] > torch.arange(q)[:,None]+kv-q, -float('inf'))
                reference_bf[bi*q:(bi+1)*q,hi] = scores.softmax(-1) @ vals[bi][:,0]
        data = prepare_inputs(source, case, 'bf16', device)
        call, _ = prepare_call(data, case, 'bf16')
        actual_bf = call()[0].cpu().float()
        assert torch.allclose(actual_bf, reference_bf, atol=1e-3, rtol=.01), case['name']
        report['bf16_max_abs'] = float((actual_bf-reference_bf).abs().max())
        del call, data
    gc.collect()
    torch.npu.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--device-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-set', choices=('basic', 'mxfp8', 'all'), default='basic',
                        help='basic: 4 C4/C8/BF16 cases; mxfp8: 8 C4/C8 cases; all: both')
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from .perf_runner import execute_performance

    torch.set_num_threads(1)
    torch.npu.set_device(args.device_id)
    importlib.import_module(args.package)
    device = f'npu:{args.device_id}'
    check_codes = args.case_set in ('basic', 'all')
    if check_codes:
        check_fp4_codes(device)
    reports = []
    for group, case in get_check_cases(args.case_set):
        report = check_references(group, case, device)
        backends = ['c4', 'c8', 'bf16'] if group == 'basic' else ['c4', 'c8']
        options = dict(package=args.package, device_id=args.device_id, output_dir=str(args.output/'latency'),
                       backends=backends, mode='graph', warmup=3, rounds=2, repeats=3,
                       case_set='document' if group == 'basic' else 'mxfp8')
        execute_performance(case, options)
        report['graph_backends'] = backends
        if group == 'basic':
            report['graph_three_backends'] = True
        reports.append(report)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(case_set=args.case_set, fp4_codes=16 if check_codes else 0, cases=reports)
    (args.output/'preflight.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('PREFLIGHT_PASSED', json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
