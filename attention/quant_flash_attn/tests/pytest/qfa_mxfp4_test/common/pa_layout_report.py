#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.

"""Build a detailed PA_BBND parity report from completed accuracy-test caches."""
import argparse
import json
from pathlib import Path

import torch

from . import golden_cache
from .reference_metrics import reference_gap
from .. import quant_flash_attn_paramset_pa_layout as pa_paramset
from ..quant_flash_attn_paramset_common import P_QUANT_SCHEMES


def _comparison(name, params, cache_dir):
    path = golden_cache._path(name, 'comparison', cache_dir).with_suffix('.json')
    return json.loads(path.read_text(encoding='utf-8'))


def _same_inputs(left, right):
    if left.keys() != right.keys():
        return False
    for name in left:
        a, b = left[name], right[name]
        if (a is None) != (b is None):
            return False
        if a is not None and not torch.equal(a, b):
            return False
    return True


def build_report(cache_dir, package, scope, quant_schema):
    records = []
    for original in pa_paramset.get_cases(scope):
        params = dict(original, quant_schema=quant_schema, p_quant_scheme=P_QUANT_SCHEMES[quant_schema])
        layout = params['kv_cache_layout']
        suffix = f'_{layout}'
        if not params['name'].endswith(suffix):
            raise ValueError(f"Unexpected PA layout case name: {params['name']}")
        base_name = params['name'][:-len(suffix)]
        base_params = dict(params, name=base_name, kv_cache_layout='PA_BBND')
        actual, actual_id = golden_cache.load_input(params['name'], params, cache_dir)
        baseline, baseline_id = golden_cache.load_input(base_name, base_params, cache_dir)
        inputs_equal = _same_inputs(actual, baseline)
        actual_output = golden_cache.load_npu_output(params['name'], params, actual_id, cache_dir, package)
        baseline_output = golden_cache.load_npu_output(base_name, base_params, baseline_id, cache_dir, package)
        parity = reference_gap(actual_output, baseline_output)
        actual_comparison = _comparison(params['name'], params, cache_dir)
        baseline_comparison = _comparison(base_name, base_params, cache_dir)
        parity_pass = all(parity[name]['relative_l2'] <= 0.01 for name in ('attention', 'lse'))
        passed = bool(inputs_equal and parity_pass and actual_comparison['passed'] and baseline_comparison['passed'])
        records.append({
            'name': params['name'], 'baseline_name': base_name, 'layout': layout,
            'quant_schema': params['quant_schema'], 'D': params['D'], 'block_size': params['block_size'],
            'q_scale_layout': params['q_scale_layout'], 'mask_mode': params['mask_mode'],
            'enable_lse': params['enable_lse'], 'inputs_equal': inputs_equal,
            'actual_comparison_passed': actual_comparison['passed'],
            'baseline_comparison_passed': baseline_comparison['passed'],
            'c4_vs_bbnd': parity, 'passed': passed,
        })
    grouped = {}
    for layout in pa_paramset.KV_LAYOUTS:
        selected = [record for record in records if record['layout'] == layout]
        grouped[layout] = {
            'cases': len(selected), 'passed': sum(record['passed'] for record in selected),
            'max_attention_relative_l2': max((record['c4_vs_bbnd']['attention']['relative_l2']
                                               for record in selected), default=0.),
            'max_lse_relative_l2': max((record['c4_vs_bbnd']['lse']['relative_l2']
                                         for record in selected), default=0.),
        }
    return {
        'scope': scope, 'quant_schema': quant_schema, 'package': package, 'cache_dir': str(cache_dir),
        'parity_threshold': {'attention_and_lse_relative_l2_max': 0.01},
        'case_count': len(records), 'passed_count': sum(record['passed'] for record in records),
        'passed': all(record['passed'] for record in records), 'by_layout': grouped, 'cases': records,
    }


def write_markdown(report, path):
    lines = [
        '# C4A4 PA_BNBD / PA_NZ 与 PA_BBND 精度对比报告', '',
        f"- 测试范围：`{report['scope']}`；`quant_schema={report['quant_schema']}`", f"- 算子包：`{report['package']}`",
        f"- 总结：**{'PASS' if report['passed'] else 'FAIL'}**（{report['passed_count']}/{report['case_count']}）",
        '- 判据：两种布局与 BBND 的 Attention、LSE relative-L2 均不超过 1%，且各自通过原 BBND golden 阈值。',
        '- 输入：逐案例核对原始 Q/K/V、scale、页表和长度张量完全一致。', '',
        '| 布局 | 通过/总数 | 最大 Attention relative-L2 | 最大 LSE relative-L2 |',
        '|---|---:|---:|---:|',
    ]
    for layout, group in report['by_layout'].items():
        lines.append(f"| {layout} | {group['passed']}/{group['cases']} | "
                     f"{group['max_attention_relative_l2']:.8g} | {group['max_lse_relative_l2']:.8g} |")
    lines += ['', '## 逐案例结果', '',
              '| 用例 | Golden | 输入一致 | Attention relative-L2 | LSE relative-L2 | 结果 |',
              '|---|---|---|---:|---:|---|']
    for record in report['cases']:
        golden_pass = record['actual_comparison_passed'] and record['baseline_comparison_passed']
        lines.append(f"| {record['name']} | {'PASS' if golden_pass else 'FAIL'} | "
                     f"{'是' if record['inputs_equal'] else '否'} | "
                     f"{record['c4_vs_bbnd']['attention']['relative_l2']:.8g} | "
                     f"{record['c4_vs_bbnd']['lse']['relative_l2']:.8g} | "
                     f"{'PASS' if record['passed'] else 'FAIL'} |")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--package', default='cann_ops_transformer_c4a4')
    parser.add_argument('--scope', choices=('small', 'full'), default='small')
    parser.add_argument('--quant-schema', type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    report = build_report(args.cache_dir, args.package, args.scope, args.quant_schema)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'pa_bbnd_parity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    write_markdown(report, output / 'pa_bbnd_parity.md')
    print(f"PA_PARITY: {'PASS' if report['passed'] else 'FAIL'} "
          f"{report['passed_count']}/{report['case_count']}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
