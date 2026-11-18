#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.

"""Paired C4/C8 accuracy comparison for PA_BNBD and PA_NZ.

The same E2M1 payload is expanded exactly to E4M3, so Q/K/V values and all
E8M0 scales are identical across backends. The FP64 reference does not
quantize P. A JSON record and the raw outputs are saved for every case.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

from . import quant_flash_attn_golden as golden
from . import test_runner
from .cpu_mxfp4_golden_fp64 import cpu_mxfp4_golden_fp64
from .reference_metrics import reference_gap
from ..quant_flash_attn_paramset_common import expand_paramset_to_cases


def comparison_cases():
    params = {}
    for layout in ('PA_BNBD', 'PA_NZ'):
        for dimension, heads, kv_heads in ((128, 8, 2), (256, 8, 1)):
            name = f'c8_compare_D{dimension}_{layout}'
            params[name] = {
                'B': [1], 'N_q': [heads], 'N_kv': [kv_heads], 'D': [dimension],
                'cu_seqlens_q': [[0, 17]], 'seqused_q': [[17]], 'seqused_kv': [[257]],
                'max_seqlen_q': [17], 'max_seqlen_kv': [257], 'block_size': [128],
                'mask_mode': [3], 'q_scale_layout': ['TND'], 'enable_lse': [True],
                'kv_cache_layout': [layout], 'quant_schema': [2], 'seed': [421],
            }
    return expand_paramset_to_cases(params)


def _fp8_bytes(packed_fp4):
    values = golden.unpack_fp4(packed_fp4)
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def _invoke_c8(data, params, package, device_id):
    import torch_npu  # noqa: F401
    importlib.import_module(package)
    p = params
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)
    e8m0 = torch.float8_e8m0fnu
    fp8 = torch.float8_e4m3fn
    layout = p['kv_cache_layout']
    tensors = {
        'q': _fp8_bytes(data['q_fp4']),
        'k': golden.convert_bbnd_tensor(_fp8_bytes(data['k_fp4']), layout, 'payload'),
        'v': golden.convert_bbnd_tensor(_fp8_bytes(data['v_fp4']), layout, 'payload'),
        'qs': data['dequant_scale_q'],
        'ks': golden.convert_bbnd_tensor(data['dequant_scale_k'], layout, 'k_scale'),
        'vs': golden.convert_bbnd_tensor(data['v_descale'], layout, 'v_scale'),
    }
    tensors = {name: value.to(device) for name, value in tensors.items()}
    for name in ('q', 'k', 'v'):
        tensors[name] = tensors[name].view(fp8)
    for name in ('qs', 'ks', 'vs'):
        tensors[name] = tensors[name].view(e8m0)
    common = dict(
        cu_seqlens_q=data['cu_seqlens_q'].to(device),
        seqused_q=None if data['seqused_q'] is None else data['seqused_q'].to(device),
        seqused_kv=data['seqused_kv'].to(device), max_seqlen_q=p['max_seqlen_q'],
        max_seqlen_kv=p['max_seqlen_kv'], mask_mode=p['mask_mode'], layout_q='TND',
        layout_q_descale=p['q_scale_layout'], layout_kv=layout, layout_out='TND')
    ops = torch.ops.cann_ops_transformer
    metadata = ops.quant_flash_attn_metadata(p['N_q'], p['N_kv'], p['D'], 1, **common)
    mask = None
    if p['mask_mode'] == 3:
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(device)
    p_scale = torch.tensor([448.], dtype=torch.float32, device=device)
    output, lse = ops.quant_flash_attn(
        tensors['q'], tensors['k'], tensors['v'], tensors['qs'], tensors['ks'], tensors['vs'], 1,
        block_table=data['block_table_torch'].to(device), metadata=metadata, attn_mask=mask,
        p_scale=p_scale, softmax_scale=p['softmax_scale'], return_softmax_lse=p['enable_lse'], **common)
    torch.npu.synchronize()
    shapes = {name: list(value.shape) for name, value in tensors.items()}
    return (output.cpu(), lse.cpu()), shapes


def execute_case(params, package, device_id, output_dir):
    p = params
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = golden.generate_data(p)
    cpu_reference = golden.cpu_mxfp4_golden(data, p)
    fp64_reference = cpu_mxfp4_golden_fp64(data, p)
    c4 = golden.npu_mxfp4_fa(data, p, package, device_id)
    bbnd_params = dict(p, name=p['name'] + '_PA_BBND', kv_cache_layout='PA_BBND')
    bbnd = golden.npu_mxfp4_fa(data, bbnd_params, package, device_id)
    c8, c8_shapes = _invoke_c8(data, p, package, device_id)
    test_runner.validate_outputs(*c4, p)
    test_runner.validate_outputs(*bbnd, p)
    test_runner.validate_outputs(*c8, p)
    c4_check = test_runner.compare_outputs(cpu_reference, c4, p)
    bbnd_check = test_runner.compare_outputs(cpu_reference, bbnd, p)
    parity = reference_gap(c4, bbnd)
    parity_pass = all(parity[name]['relative_l2'] <= 0.01 for name in ('attention', 'lse'))
    golden_pass = all(result is None or result[0] == 'Pass' for result in (*c4_check, *bbnd_check))
    record = {
        'name': p['name'], 'params': p, 'package': package, 'device_id': device_id,
        'c8_p_scale': 448., 'c8_input_shapes': c8_shapes,
        'c4_vs_cpu_threshold': c4_check, 'bbnd_vs_cpu_threshold': bbnd_check,
        'c4_vs_bbnd': parity, 'c4_vs_c8': reference_gap(c4, c8),
        'c4_vs_quantized_cpu': reference_gap(c4, cpu_reference),
        'c8_vs_quantized_cpu': reference_gap(c8, cpu_reference),
        'c4_vs_fp64_unquantized_p': reference_gap(c4, fp64_reference),
        'c8_vs_fp64_unquantized_p': reference_gap(c8, fp64_reference),
        'parity_threshold': {'relative_l2_max': 0.01},
        'passed': bool(parity_pass and golden_pass),
    }
    stem = output_dir / p['name']
    stem.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    torch.save({'c4': c4, 'bbnd': bbnd, 'c8': c8, 'cpu': cpu_reference, 'fp64': fp64_reference},
               stem.with_suffix('.pt'))
    print(f"C8_COMPARE {p['name']}: {'PASS' if record['passed'] else 'FAIL'} "
          f"parity_l2={parity['attention']['relative_l2']:.6g}", flush=True)
    return record


def write_summary_markdown(summary, path):
    lines = [
        '# C4A4 与 C8A8 PA_BNBD / PA_NZ 精度对比报告', '',
        f"- 总结：**{'PASS' if summary['passed'] else 'FAIL'}**（{summary['case_count']} 个案例）",
        '- C4 与 BBND 的通过门限：Attention/LSE relative-L2 均不超过 1%。',
        '- C4 与 C8 的 Q/K/V 数值、scale、页表和布局完全一致；P 量化格式不同，因此二者不强制 1% 直接误差。',
        '- FP64 参考不量化 P，用于区分布局问题与 P 量化精度差异。', '',
        '| 用例 | C4/BBND Attention L2 | C4/C8 Attention L2 | C4/C8 LSE L2 | C4/FP64 L2 | C8/FP64 L2 | 结果 |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for case in summary['cases']:
        lines.append(
            f"| {case['name']} | {case['c4_vs_bbnd']['attention']['relative_l2']:.8g} | "
            f"{case['c4_vs_c8']['attention']['relative_l2']:.8g} | {case['c4_vs_c8']['lse']['relative_l2']:.8g} | "
            f"{case['c4_vs_fp64_unquantized_p']['attention']['relative_l2']:.8g} | "
            f"{case['c8_vs_fp64_unquantized_p']['attention']['relative_l2']:.8g} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='cann_ops_transformer_c4a4')
    parser.add_argument('--device-id', type=int, default=1)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--case', action='append', help='Case name substring; may be repeated')
    args = parser.parse_args(argv)
    cases = comparison_cases()
    if args.case:
        cases = [case for case in cases if any(token in case['name'] for token in args.case)]
    if not cases:
        parser.error('No comparison cases selected')
    records = [execute_case(case, args.package, args.device_id, args.output_dir) for case in cases]
    summary = {'passed': all(record['passed'] for record in records), 'case_count': len(records),
               'cases': [{key: record[key] for key in ('name', 'passed', 'c4_vs_bbnd', 'c4_vs_c8',
                         'c4_vs_fp64_unquantized_p', 'c8_vs_fp64_unquantized_p')} for record in records]}
    path = Path(args.output_dir) / 'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_summary_markdown(summary, path.with_suffix('.md'))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
