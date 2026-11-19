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

"""CPU-only checks for scope, sharding and trustworthy performance aggregation."""
import copy
import unittest

from ..quant_flash_attn_paramset_perf_rdv import get_cases, shard_cases, case_fingerprint, sequence_lengths, with_kv_layout
from .perf_report import summarize
from .perf_profiler import attribute_tasks


class PerformanceContractTests(unittest.TestCase):
    def test_documented_scope(self):
        for scope, count, prefill in [('small', 75, 15), ('full', 1395, 135)]:
            cases = get_cases(scope)
            self.assertEqual(len(cases), count)
            self.assertEqual(len({c['name'] for c in cases}), count)
            self.assertEqual(sum(c['kind'] == 'prefill' for c in cases), prefill)
            self.assertTrue(all(c['page'] == 256 and c['mask_mode'] == 3 and not c['enable_lse'] for c in cases))
        self.assertTrue(all((c['D'], c['N_q'], c['N_kv'], c['tp']) == (256, 8, 1, 2) for c in get_cases('small')))

    def test_small_does_not_use_accuracy_mtp_scope(self):
        cases = get_cases('small', 'decode')
        self.assertEqual({c['Q'] for c in cases}, {1, 2, 4, 6, 8})
        self.assertEqual({c['B'] for c in cases}, {1, 2, 16})
        self.assertEqual({c['KV'] for c in cases}, {32768, 65536, 131072, 262144})
        self.assertTrue(all(c['q_scale_layout'] == 'N2TGD' for c in cases))

    def test_shards_are_disjoint_and_complete(self):
        cases = get_cases('small')
        first = {c['name'] for c in shard_cases(cases, 0, 2)}
        second = {c['name'] for c in shard_cases(cases, 1, 2)}
        self.assertFalse(first & second)
        self.assertEqual(first | second, {c['name'] for c in cases})
        self.assertEqual((len(first), len(second)), (38, 37))
        with self.assertRaises(ValueError):
            shard_cases(cases, 2, 2)

    def test_schema_changes_fingerprint(self):
        case = get_cases('small')[0]
        self.assertNotEqual(case_fingerprint(case), case_fingerprint(dict(case, quant_schema=0)))

    def test_kv_layout_changes_fingerprint_and_preserves_default(self):
        for case in (get_cases('small')[0], get_cases(case_set='bench16k')[0]):
            default = case_fingerprint(case)
            self.assertEqual(case_fingerprint(with_kv_layout(case, 'PA_BBND')), default)
            for layout in ('PA_BNBD', 'PA_NZ'):
                selected = with_kv_layout(case, layout)
                self.assertEqual(selected['layout_kv'], layout)
                self.assertNotEqual(case_fingerprint(selected), default)
                self.assertEqual(sequence_lengths(selected), sequence_lengths(case))

    def record(self):
        return dict(case=get_cases('small')[0], options=dict(device_id=6, warmup=3, rounds=5, repeats=20),
            environment=dict(opapi=[]), metric='graph_replay_us', measurement_scope='attention_only', status='passed',
            backends={b:dict(median_us=v, samples_us=[v]*5, spread_pct=0) for b,v in [('c4',10.),('c8',20.),('bf16',30.)]})

    def test_ratios(self):
        report = summarize([self.record()])
        self.assertEqual(report['complete'], 1)
        row = report['rows'][0]
        self.assertEqual((row['c8_over_c4'],row['bf16_over_c4'],row['bf16_over_c8']), (2.,3.,1.5))

    def test_report_keeps_the_measured_kv_layout(self):
        first, second = self.record(), self.record()
        second['case'] = with_kv_layout(get_cases('small')[1], 'PA_NZ')
        self.assertEqual(summarize([second])['rows'][0]['layout_kv'], 'PA_NZ')
        with self.assertRaisesRegex(ValueError, 'Mixed timing contracts'):
            summarize([first, second])

    def test_incomplete_not_counted(self):
        record = self.record()
        del record['backends']['bf16']
        self.assertEqual(summarize([record])['complete'], 0)

    def test_duplicate_and_inconsistent_results_rejected(self):
        record = self.record()
        with self.assertRaises(ValueError):
            summarize([record, copy.deepcopy(record)])
        record['backends']['c4']['samples_us'].pop()
        with self.assertRaises(ValueError):
            summarize([record])

    def test_mixed_timing_rejected(self):
        first, second = self.record(), self.record()
        second['case'] = get_cases('small')[1]
        second['metric'] = 'eager_event_us'
        with self.assertRaises(ValueError):
            summarize([first, second])

    def test_imported_scope_and_variable_lengths(self):
        cases = get_cases(case_set='mxfp8')
        self.assertEqual((len(cases), sum(bool(c['skip_reason']) for c in cases)), (211, 6))
        self.assertEqual(sum(c['kind'] == 'prefill' for c in cases), 102)
        self.assertTrue(all(c['page'] == 512 and c['layout_kv'] == 'PA_BBND' for c in cases))
        self.assertEqual({c['mask_mode'] for c in cases}, {0, 3})
        self.assertEqual(sum(c['enable_lse'] for c in cases), 7)
        ragged = [c for c in cases if 'KVS524288sum' in c['source_case_id']]
        self.assertTrue(ragged)
        self.assertTrue(all(sum(sequence_lengths(c)[2]) == 524288 for c in ragged))
        self.assertTrue(any(0 in [b-a for a,b in zip(c['cu_seqlens_q'],c['cu_seqlens_q'][1:])] for c in cases))

    def test_two_backend_report(self):
        record = self.record()
        record['options']['backends'] = ['c4','c8']
        del record['backends']['bf16']
        report = summarize([record])
        self.assertEqual(report['complete'], 1)
        self.assertEqual(report['groups']['all']['c8_over_c4']['geomean'], 2.)
        self.assertNotIn('bf16_over_c4', report['groups']['all'])

    def test_msprof_discards_warmup_and_keeps_backend_identity(self):
        record = self.record()
        record['options'].update(backends=['c4','c8'], warmup=1, rounds=2, repeats=2)
        record.update(metric='msprof_pending', started_unix=1)
        record['backends'] = {b:dict(profile_invocations=5) for b in ('c4','c8')}
        tasks = [dict(duration_us=v) for v in (999,10,12,14,16,999,20,24,28,32)]
        parsed = attribute_tasks([record], tasks)[0]
        self.assertEqual(parsed['backends']['c4']['samples_us'], [11,15])
        self.assertEqual(parsed['backends']['c8']['samples_us'], [22,30])
        self.assertEqual(summarize([parsed])['rows'][0]['c8_over_c4'], 2.)
        with self.assertRaises(ValueError):
            attribute_tasks([record], tasks[:-1])

    def test_msprof_refuses_other_timing_contract(self):
        record = self.record()
        record.update(started_unix=1)
        record['options'].update(backends=['c4'], warmup=1, rounds=1, repeats=1)
        with self.assertRaises(ValueError):
            attribute_tasks([record], [dict(duration_us=1)]*2)


if __name__ == '__main__':
    unittest.main()
