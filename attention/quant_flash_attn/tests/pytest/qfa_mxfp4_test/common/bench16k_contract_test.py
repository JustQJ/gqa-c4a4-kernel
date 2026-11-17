"""Offline checks for the requested lengths, fixed allocations and result identity."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .. import quant_flash_attn_paramset_bench16k as bench
from .. import quant_flash_attn_paramset_func_rdv as accuracy
from .. import quant_flash_attn_paramset_debug as debug
from ..quant_flash_attn_paramset_common import normalize_params, paged_kv_pool_size
from ..quant_flash_attn_paramset_perf_rdv import get_cases, case_fingerprint, sequence_lengths, shard_cases
from . import perf_inputs, perf_report

SUITE = Path(__file__).resolve().parents[1]


class AllocationTensor:
    """Record requested sizes without allocating the multi-GB benchmark tensors."""
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def to(self, *args):
        return self

    def numpy(self):
        return b'\0'

    def reshape(self, *shape):
        return AllocationTensor(shape)

    def permute(self, *order):
        return AllocationTensor([self.shape[index] for index in order])

    def contiguous(self):
        return self


class Bench16kContractTests(unittest.TestCase):
    def test_exact_requested_lengths_and_order(self):
        expected = [
            ([4096], [4096]), ([3972], [12072]), ([3980], [16168]),
            ([4096, 3988], [4096, 12056]), ([2048, 2400], [20264, 7960]),
            ([2083, 2900], [24252, 16152]),
            ([4] * 32, [8192 + 107*i for i in range(16)] + [16384 + 125*i for i in range(16)]),
            ([4] * 24, [32768 + 32*i for i in range(14)] + [65536 + 444*i for i in range(10)]),
            ([1] * 32, [2048 + 11*i for i in range(16)] + [256 + 25*i for i in range(16)]),
            ([4] * 33, [2048 + 9*i for i in range(32)] + [16409]),
            ([2044] + [4] * 26, [2044] + [10240 + 117*i for i in range(16)] + [18432 + 125*i for i in range(10)]),
            ([4096] + [4] * 22, [8192] + [32768 + 137*i for i in range(12)] + [22528 + 100*i for i in range(10)]),
        ]
        cases = get_cases(case_set='bench16k')
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case['name'] for case in cases}), 12)
        for case, (queries, keys) in zip(cases, expected):
            with self.subTest(case=case['name']):
                cu, used, kv = sequence_lengths(case)
                self.assertEqual(([b-a for a, b in zip(cu, cu[1:])], used, kv), (queries, queries, keys))
                self.assertEqual((cu[0], case['B'], case['Q'], case['KV']), (0, len(queries), max(queries), max(keys)))
                self.assertTrue(all(k >= q > 0 for q, k in zip(queries, keys)))

    def test_fixed_attributes_and_accuracy_performance_agreement(self):
        for perf, acc in zip(get_cases(case_set='bench16k'), accuracy.get_cases(case_set='bench16k')):
            for key in ('name', 'B', 'N_q', 'N_kv', 'D', 'cu_seqlens_q', 'seqused_q', 'seqused_kv',
                        'mask_mode', 'q_scale_layout', 'seed', 'softmax_scale', 'enable_lse',
                        'kv_cache_num_blocks', 'logical_kv_cache_shape'):
                self.assertEqual(perf[key], acc[key])
            self.assertEqual((perf['N_q'], perf['N_kv'], perf['D'], perf['page'], acc['block_size']), (16, 2, 256, 128, 128))
            self.assertEqual((perf['mask_mode'], perf['enable_lse'], perf['softmax_scale']), (3, False, 1/16))
            self.assertEqual(perf['logical_kv_cache_shape'], [31374, 128, 512])
            self.assertEqual(perf['q_scale_layout'], 'N2TGD' if perf['kind'] == 'decode' else 'TND')
            self.assertEqual(normalize_params(acc), acc)

    def test_selection_scopes_and_shards(self):
        cases = get_cases(case_set='bench16k')
        self.assertEqual(cases, get_cases('small', case_set='bench16k'))
        self.assertEqual([len(get_cases(kind=kind, case_set='bench16k')) for kind in ('prefill', 'decode', 'mixed')], [6, 4, 2])
        shards = [shard_cases(cases, index, 3) for index in range(3)]
        self.assertEqual(sum(map(len, shards)), 12)
        self.assertEqual(len({case['name'] for shard in shards for case in shard}), 12)
        for loader in (accuracy.get_cases, get_cases):
            with self.assertRaises(ValueError):
                loader(case_set='fia')

    def test_capacity_validation_and_compact_defaults(self):
        self.assertEqual(paged_kv_pool_size([129, 128], 128), 3)
        self.assertEqual(paged_kv_pool_size([129, 128], 128, extra_pages=3), 6)
        self.assertEqual(paged_kv_pool_size([129, 128], 128, 31374, extra_pages=3), 31374)
        params = bench.get_accuracy_cases()[0]
        for bad in (0, -1, True, 2.5, 31):
            with self.subTest(capacity=bad), self.assertRaises(ValueError):
                normalize_params(dict(params, kv_cache_num_blocks=bad))
        with self.assertRaises(ValueError):
            normalize_params(dict(params, logical_kv_cache_shape=[31374, 128, 256]))

    def test_both_input_generators_request_the_full_pool(self):
        fake = SimpleNamespace(
            uint8='uint8', int32='int32', int64='int64',
            Generator=lambda: SimpleNamespace(manual_seed=lambda seed: None),
            randint=lambda low, high, shape, **kwargs: AllocationTensor(shape),
            randperm=lambda count, **kwargs: AllocationTensor((count,)),
            full=lambda shape, value, **kwargs: AllocationTensor(shape),
            zeros=lambda shape, **kwargs: AllocationTensor(shape),
            tensor=lambda values, **kwargs: AllocationTensor((len(values),)),
        )
        spec = importlib.util.spec_from_file_location(__package__ + '._bench_golden', SUITE/'common/quant_flash_attn_golden.py')
        golden = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, torch=fake):
            spec.loader.exec_module(golden)
            for perf, acc in zip(bench.get_performance_cases(), bench.get_accuracy_cases()):
                with self.subTest(case=perf['name']):
                    source = perf_inputs.generate_inputs(perf)
                    data = golden.generate_data(acc)
                    self.assertEqual(source['raw'][1].shape, (31374, 128, 2, 128))
                    self.assertEqual(source['raw'][2].shape, (31374, 128, 2, 128))
                    self.assertEqual(data['k_fp4'].shape, (31374, 128, 2, 128))
                    self.assertEqual(data['v_fp4'].shape, (31374, 128, 2, 128))
                    self.assertEqual(data['dequant_scale_k'].shape[0], 31374)
                    self.assertEqual(data['v_descale'].shape[0], 31374)
            compact_perf = get_cases('small')[0]
            compact_acc = accuracy.get_cases('small')[0]
            self.assertEqual(perf_inputs.generate_inputs(compact_perf)['raw'][1].shape[0],
                             sum((n + 255)//256 for n in sequence_lengths(compact_perf)[2]))
            self.assertEqual(golden.generate_data(compact_acc)['k_fp4'].shape[0],
                             sum((n + compact_acc['block_size'] - 1)//compact_acc['block_size']
                                 for n in compact_acc['seqused_kv']) + 3)

    def record(self, case):
        return dict(case=case, case_fingerprint=case_fingerprint(case),
                    options=dict(device_id=7, warmup=3, rounds=5, repeats=20,
                                 backends=['c4', 'c8'], case_set='bench16k'),
                    environment=dict(opapi=[]), metric='graph_replay_us', measurement_scope='attention_only',
                    status='passed', backends={backend: dict(median_us=value, samples_us=[value]*5, spread_pct=0)
                                              for backend, value in [('c4', 10.), ('c8', 20.)]})

    def test_report_keeps_capacity_and_groups(self):
        report = perf_report.summarize([self.record(case) for case in bench.get_performance_cases()])
        self.assertEqual((report['total'], report['complete']), (12, 12))
        self.assertEqual([report['groups'][kind]['count'] for kind in ('prefill', 'decode', 'mixed')], [6, 4, 2])
        for row in report['rows']:
            self.assertEqual(row['kv_cache_num_blocks'], 31374)
            self.assertEqual(json.loads(row['logical_kv_cache_shape']), [31374, 128, 512])

    def test_report_cli_rejects_changed_lengths_and_capacity(self):
        cases = [dict(case, quant_schema=0) for case in bench.get_performance_cases()]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root/'records'
            inputs.mkdir()
            for index, case in enumerate(cases):
                (inputs/f'{index:03d}.json').write_text(json.dumps(self.record(case)))
            args = ['perf_report', str(inputs), '--output', str(root/'summary'), '--expected', '12', '--case-set', 'bench16k']
            with patch.object(sys, 'argv', args), redirect_stdout(io.StringIO()):
                perf_report.main()
            self.assertEqual(json.loads((root/'summary/summary.json').read_text())['complete'], 12)
            for key, value in [('seqused_kv', [4097]), ('kv_cache_num_blocks', 31375),
                               ('logical_kv_cache_shape', [31375, 128, 512])]:
                altered = dict(cases[0], **{key: value})
                (inputs/'000.json').write_text(json.dumps(self.record(altered)))
                with patch.object(sys, 'argv', args), redirect_stdout(io.StringIO()), \
                        self.assertRaisesRegex(SystemExit, 'bench16k case parameters'):
                    perf_report.main()

    def test_calls_return_independent_cases(self):
        first = bench.get_performance_cases()
        altered = bench.get_performance_cases()
        altered[0]['seqused_kv'][0] += 128
        altered[0]['logical_kv_cache_shape'][0] += 1
        self.assertNotEqual(case_fingerprint(first[0]), case_fingerprint(altered[0]))
        self.assertEqual(first, bench.get_performance_cases())

    def test_pytest_dispatch_and_options_without_hardware(self):
        fake = SimpleNamespace(
            fixture=lambda function=None, **kwargs: function if function is not None else lambda value: value,
            UsageError=ValueError, param=lambda value, **kwargs: SimpleNamespace(value=value, **kwargs),
            mark=SimpleNamespace(skip=lambda **kwargs: kwargs))
        spec = importlib.util.spec_from_file_location(__package__.rsplit('.', 1)[0] + '._bench_conftest', SUITE/'conftest.py')
        conftest = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, pytest=fake):
            spec.loader.exec_module(conftest)
        options, choices = {}, {}
        class Parser:
            def getgroup(self, name):
                return self
            def addoption(self, name, **kwargs):
                options[name] = kwargs.get('default', False)
                choices[name] = kwargs.get('choices')
        conftest.pytest_addoption(Parser())
        self.assertNotIn('--fia-shapes', options)
        self.assertEqual(choices['--accuracy-case-set'], ('document', 'bench16k', 'qwen3_8b'))
        options.update({'--scope': 'small', '--accuracy-case-set': 'bench16k', '--perf-case-set': 'bench16k',
                        '--perf-kind': 'mixed', '--run-perf': True, '--quant-schema': 0})
        config = SimpleNamespace(getoption=lambda key: options[key])
        def collect(fixtures, paramset):
            result = {}
            conftest.pytest_generate_tests(SimpleNamespace(fixturenames=fixtures, config=config,
                module=SimpleNamespace(paramset=paramset), parametrize=lambda key, values: result.update({key: values})))
            return result[fixtures[0]]
        functional = collect(['params'], accuracy)
        self.assertEqual(len(functional), 12)
        self.assertTrue(all(case.value['quant_schema'] == 0 and case.value['p_quant_scheme'] == 'ocp' for case in functional))
        self.assertEqual(len(collect(['perf_params'], None)), 2)
        self.assertEqual(len(collect(['params'], debug)), 24)
        self.assertEqual(conftest.perf_options(SimpleNamespace(config=config))['case_set'], 'bench16k')
        options.update({'--accuracy-case-set': 'qwen3_8b', '--perf-case-set': 'qwen3_8b'})
        functional = collect(['params'], accuracy)
        self.assertEqual(len(functional), 16)
        self.assertTrue(all((case.value['N_q'], case.value['N_kv'], case.value['D']) == (32, 8, 128)
                            and case.value['quant_schema'] == 0 for case in functional))
        self.assertEqual(len(collect(['perf_params'], None)), 4)
        self.assertEqual(conftest.perf_options(SimpleNamespace(config=config))['case_set'], 'qwen3_8b')


if __name__ == '__main__':
    unittest.main()
