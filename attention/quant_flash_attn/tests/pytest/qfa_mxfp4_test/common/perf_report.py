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

"""Merge case records from devices; compare only within each measured case."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from ..quant_flash_attn_paramset_common import DEFAULT_KV_LAYOUT, KV_LAYOUTS


def summarize(records):
    rows, seen = [], set()
    contracts = set()
    for record in records:
        case, options = record['case'], record['options']
        layout = case.get('layout_kv', DEFAULT_KV_LAYOUT)
        key = (case['name'], case['quant_schema'])
        if key in seen:
            raise ValueError(f'Duplicate case: {key}')
        seen.add(key)
        contracts.add((record['metric'], record['measurement_scope'], options['warmup'],
                       options['rounds'], options['repeats'], case['quant_schema'],
                       tuple(options.get('backends', ('c4', 'c8', 'bf16'))),
                       options.get('case_set', 'document'), layout,
                       case.get('kv_cache_num_blocks'),
                       json.dumps(record['environment']['opapi'], sort_keys=True)))
        row = {k: case[k] for k in ('name', 'model', 'tp', 'kind', 'B', 'Q', 'KV', 'D', 'N_q', 'N_kv', 'q_scale_layout', 'quant_schema')}
        row.update(device=options['device_id'], status=record['status'], metric=record['metric'])
        row.update(page=case['page'], mask_mode=case['mask_mode'], enable_lse=case['enable_lse'],
                   c8_p_scale=case.get('c8_p_scale', 448.), source_case_id=case.get('source_case_id'),
                   case_fingerprint=record.get('case_fingerprint'), layout_kv=layout,
                   cu_seqlens_q=json.dumps(case.get('cu_seqlens_q')),
                   seqused_q=json.dumps(case.get('seqused_q')), seqused_kv=json.dumps(case.get('seqused_kv')))
        row.update(case_set=options.get('case_set', 'document'),
                   kv_cache_num_blocks=case.get('kv_cache_num_blocks'),
                   logical_kv_cache_shape=json.dumps(case.get('logical_kv_cache_shape')))
        required = options.get('backends', ('c4', 'c8', 'bf16'))
        for backend in ('c4', 'c8', 'bf16'):
            result = record['backends'].get(backend, {})
            if result:
                samples = result.get('samples_us', [])
                if len(samples) != options['rounds'] or not all(math.isfinite(v) and v > 0 for v in samples):
                    raise ValueError(f'Invalid samples for {key}/{backend}')
                if not math.isclose(statistics.median(samples), result['median_us'], rel_tol=1e-9):
                    raise ValueError(f'Median does not match samples for {key}/{backend}')
            row[backend+'_us'] = result.get('median_us')
            row[backend+'_spread_pct'] = result.get('spread_pct')
        valid = record['status'] == 'passed' and all(isinstance(row[b+'_us'], (int, float)) and
            math.isfinite(row[b+'_us']) and row[b+'_us'] > 0 for b in required)
        row['complete'] = valid
        for numerator, denominator in (('c8', 'c4'), ('bf16', 'c4'), ('bf16', 'c8')):
            row[numerator+'_over_'+denominator] = row[numerator+'_us']/row[denominator+'_us'] \
                if valid and numerator in required and denominator in required else None
        rows.append(row)
    if len(contracts) > 1:
        raise ValueError('Mixed timing contracts or operator libraries cannot share a comparison report')
    groups = {}
    for kind in ('all', 'prefill', 'decode', 'mixed'):
        selected = [r for r in rows if r['complete'] and (kind == 'all' or r['kind'] == kind)]
        if not selected:
            continue
        groups[kind] = dict(count=len(selected))
        for ratio in ('c8_over_c4', 'bf16_over_c4', 'bf16_over_c8'):
            values = [r[ratio] for r in selected if r[ratio] is not None]
            if not values:
                continue
            groups[kind][ratio] = dict(geomean=math.exp(statistics.mean(math.log(v) for v in values)),
                                       median=statistics.median(values), min=min(values), max=max(values),
                                       faster=sum(v > 1 for v in values))
    return dict(total=len(rows), complete=sum(r['complete'] for r in rows), groups=groups, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=int, required=True)
    parser.add_argument('--scope', choices=('small', 'full'), default='small')
    parser.add_argument('--case-set', choices=('document', 'mxfp8', 'bench16k', 'qwen3_8b', 'qwen3_8b_page256'), default='document')
    parser.add_argument('--kind', choices=('all', 'prefill', 'decode', 'mixed'), default='all')
    parser.add_argument('--kv-layout', '--kv_layout', choices=KV_LAYOUTS, default=DEFAULT_KV_LAYOUT)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for directory in args.directories for p in sorted(directory.glob('*.json'))]
    from ..quant_flash_attn_paramset_perf_rdv import get_cases, case_fingerprint, with_kv_layout
    expected_cases = {case['name']: case for case in get_cases(
        args.scope, args.kind, args.case_set) if not case.get('skip_reason')}
    expected_ids = set(expected_cases)
    if {record['case']['name'] for record in records} != expected_ids:
        raise SystemExit('Case IDs do not exactly cover the selected performance scope')
    if any(record['case'].get('layout_kv', DEFAULT_KV_LAYOUT) != args.kv_layout for record in records):
        raise SystemExit('Case KV layout does not match --kv-layout')
    if args.case_set in ('bench16k', 'qwen3_8b', 'qwen3_8b_page256'):
        for record in records:
            expected = with_kv_layout(dict(expected_cases[record['case']['name']],
                                           quant_schema=record['case']['quant_schema']), args.kv_layout)
            fingerprint = case_fingerprint(expected)
            if (record.get('case_fingerprint') != fingerprint or
                    case_fingerprint(record['case']) != fingerprint):
                raise SystemExit(f'{args.case_set} case parameters do not match the selected fixed scope')
    report = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if report['rows']:
        with (args.output/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]))
            writer.writeheader()
            writer.writerows(report['rows'])
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))
    if report['total'] != args.expected or report['complete'] != args.expected:
        raise SystemExit(f"Incomplete report: {report['complete']}/{args.expected}")


if __name__ == '__main__':
    main()
