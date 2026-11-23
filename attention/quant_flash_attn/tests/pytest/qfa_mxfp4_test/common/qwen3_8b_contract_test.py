"""Offline validation of the requested Qwen3-8B workload and its runner inputs."""
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

from .. import quant_flash_attn_paramset_qwen3_8b as bench
from ..quant_flash_attn_paramset_common import normalize_params, paged_kv_pool_size
from ..quant_flash_attn_paramset_func_rdv import get_cases as accuracy_cases
from ..quant_flash_attn_paramset_perf_rdv import get_cases, sequence_lengths
from ..prof_quant_flash_attn import arguments, build_command, main, named_cases
from . import bench16k_contract_test as bench_contract
from . import perf_inputs, perf_report


class Qwen3EightBContractTests(unittest.TestCase):
    def test_model_and_requested_workload(self):
        cases = get_cases(case_set='qwen3_8b')
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({case['name'] for case in cases}), 16)
        self.assertEqual([len(get_cases(kind=k, case_set='qwen3_8b')) for k in ('prefill', 'decode', 'mixed')], [6, 6, 4])
        self.assertEqual({case['B'] for case in cases}, {1, 2, 4, 8, 16, 32, 64, 96, 128})
        self.assertEqual((min(min(c['seqused_kv']) for c in cases), max(c['KV'] for c in cases)), (500, 16526))
        prefill_lengths = set()
        for case in cases:
            with self.subTest(case=case['name']):
                self.assertEqual((case['N_q'], case['N_kv'], case['D'], case['tp']), (32, 8, 128, 1))
                self.assertAlmostEqual(case['softmax_scale'], 1 / 128 ** 0.5)
                cu, used, kv = sequence_lengths(case)
                self.assertEqual([b-a for a, b in zip(cu, cu[1:])], used)
                self.assertTrue(all(500 <= k <= 16526 and 0 < q <= k for q, k in zip(used, kv)))
                prefills = [q for q in used if q >= 500]
                prefill_lengths.update(prefills)
                self.assertTrue(all(500 <= q <= 2048 for q in prefills))
                if case['kind'] == 'prefill':
                    self.assertLessEqual(case['B'], 4)
                    self.assertEqual(len(prefills), case['B'])
                elif case['kind'] == 'decode':
                    self.assertGreaterEqual(case['B'], 8)
                    expected_q = 4 if case['name'] == 'qwen3_8b_decode_3' else 1
                    self.assertEqual(used, [expected_q] * case['B'])
                else:
                    self.assertIn(len(prefills), (1, 2))
                    self.assertGreaterEqual(used.count(1), 15)
                    self.assertGreater(used.count(1), len(prefills))
        self.assertTrue({500, 1024, 1536, 2048} <= prefill_lengths)

    def test_accuracy_performance_capacity_and_selection(self):
        perf = get_cases(case_set='qwen3_8b')
        self.assertEqual(perf, get_cases('small', case_set='qwen3_8b'))
        for case, acc in zip(perf, accuracy_cases(case_set='qwen3_8b')):
            for key in set(case) & set(acc):
                self.assertEqual(case[key], acc[key])
            self.assertEqual(normalize_params(acc), acc)
            self.assertEqual((case['page'], acc['block_size']), (128, 128))
            self.assertEqual(case['logical_kv_cache_shape'], [65536, 128, 1024])
            self.assertEqual(paged_kv_pool_size(case['seqused_kv'], 128, 65536), 65536)
        largest = next(c for c in perf if c['kind'] == 'decode' and c['B'] == 128)
        self.assertEqual(paged_kv_pool_size(largest['seqused_kv'], 128), 16384)
        with self.assertRaises(ValueError):
            normalize_params(dict(accuracy_cases(case_set='qwen3_8b')[11], kv_cache_num_blocks=16383))
        for loader in (bench.get_performance_cases, bench.get_accuracy_cases):
            with self.assertRaises(ValueError):
                loader('invalid')

    def test_page256_preserves_workload_and_pool_capacity(self):
        suite = 'qwen3_8b_page256'
        original = get_cases(case_set='qwen3_8b')
        cases = get_cases(case_set=suite)
        accuracy = accuracy_cases(case_set=suite)
        self.assertEqual((len(original), len(cases), len(accuracy)), (16, 16, 16))
        for old, new, acc in zip(original, cases, accuracy):
            self.assertEqual(new['name'], old['name'].replace('qwen3_8b_', suite + '_', 1))
            self.assertEqual((new['page'], acc['block_size']), (256, 256))
            self.assertEqual(new['logical_kv_cache_shape'], [32768, 256, 1024])
            self.assertEqual(old['page'] * old['kv_cache_num_blocks'], new['page'] * new['kv_cache_num_blocks'])
            for key in ('cu_seqlens_q', 'seqused_q', 'seqused_kv', 'N_q', 'N_kv', 'D', 'q_scale_layout'):
                self.assertEqual(old[key], new[key])
                self.assertEqual(new[key], acc[key])

    def test_actual_generators_use_model_dimensions_and_fixed_pool(self):
        fake = SimpleNamespace(
            uint8='uint8', int32='int32', int64='int64',
            Generator=lambda: SimpleNamespace(manual_seed=lambda seed: None),
            randint=lambda low, high, shape, **kwargs: bench_contract.AllocationTensor(shape),
            randperm=lambda count, **kwargs: bench_contract.AllocationTensor((count,)),
            full=lambda shape, value, **kwargs: bench_contract.AllocationTensor(shape),
            zeros=lambda shape, **kwargs: bench_contract.AllocationTensor(shape),
            tensor=lambda values, **kwargs: bench_contract.AllocationTensor((len(values),)),
        )
        source_path = Path(__file__).with_name('quant_flash_attn_golden.py')
        spec = importlib.util.spec_from_file_location(__package__ + '._qwen_golden', source_path)
        golden = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, torch=fake):
            spec.loader.exec_module(golden)
            for case, acc in zip(bench.get_performance_cases(), bench.get_accuracy_cases()):
                source, data = perf_inputs.generate_inputs(case), golden.generate_data(acc)
                self.assertEqual(source['raw'][0].shape, (sum(case['seqused_q']), 32, 64))
                for tensor in (*source['raw'][1:], data['k_fp4'], data['v_fp4']):
                    self.assertEqual(tensor.shape, (65536, 128, 8, 64))

    def test_profiler_worker_roundtrip_and_listing(self):
        for profiler in ('msprof', 'op'):
            args = arguments(['--device', '7', '--case-set', 'qwen3_8b', '--case', 'mixed_4', '--profiler', profiler])
            command = build_command(args)
            worker = arguments(command[command.index('--worker'):])
            self.assertEqual(named_cases(args), named_cases(worker))
            self.assertEqual(named_cases(worker)[0], get_cases(case_set='qwen3_8b')[-1])
        with redirect_stdout(io.StringIO()) as output:
            main(['--case-set', 'qwen3_8b', '--list-cases'])
        self.assertEqual(json.loads(output.getvalue()), get_cases(case_set='qwen3_8b'))

    def test_report_requires_current_parameters(self):
        # Reuse the existing record fixture; these are synthetic timings, not measurements.
        records = [bench_contract.Bench16kContractTests().record(case) for case in get_cases(case_set='qwen3_8b')]
        for record in records:
            record['options']['case_set'] = 'qwen3_8b'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, record in enumerate(records):
                (root / f'{index:02d}.json').write_text(json.dumps(record))
            argv = ['perf_report', str(root), '--case-set', 'qwen3_8b', '--expected', '16', '--output', str(root / 'report')]
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                perf_report.main()
            report = json.loads((root / 'report/summary.json').read_text())
            self.assertEqual(report['complete'], 16)
            self.assertEqual([report['groups'][k]['count'] for k in ('prefill', 'decode', 'mixed')], [6, 6, 4])
            records[0]['case']['N_kv'] = 4
            (root / '00.json').write_text(json.dumps(records[0]))
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(SystemExit, 'qwen3_8b case parameters'):
                perf_report.main()


if __name__ == '__main__':
    unittest.main()
