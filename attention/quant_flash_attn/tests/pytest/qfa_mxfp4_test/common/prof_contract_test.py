"""CPU-only checks for profiling boundaries and false-success rejection."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ..prof_quant_flash_attn import (DEFAULT_METRICS, arguments, build_command, profile_range,
                                    validate_output, validate_application_output, named_cases, main)


class ProfilingContractTests(unittest.TestCase):
    def test_worker_preserves_case_and_excludes_launcher(self):
        args = arguments(['--device', '7', '--package', 'custom_qfa', '--metrics', 'Default',
                          '--q-lens', '1', '3', '--kv-lens', '129', '513', '--page', '512',
                          '--layout', 'TND', '--lse', '--include-metadata', '--post-range-check'])
        command = build_command(args)
        worker = arguments(command[command.index('--worker'):])
        for key in ('device', 'package', 'q_lens', 'kv_lens', 'page', 'layout', 'lse',
                    'include_metadata', 'post_range_check', 'warmup', 'repeat', 'quant_schema'):
            self.assertEqual(getattr(worker, key), getattr(args, key))
        self.assertTrue(worker.worker)
        self.assertIn('--kernel-name=QuantFlashAttn_', command)
        self.assertIn('--warm-up=10', command)
        self.assertEqual(worker.repeat, 5)
        self.assertEqual(worker.backend, 'c4')
        self.assertIsNone(worker.p_scale)
        self.assertIn('--mstx-include=C4A4_PROFILE', command)

    def test_c8_worker_and_marker_match_tool_filter(self):
        for extra, scale in (([], 448.), (['--p-scale', '1'], 1.)):
            args = arguments(['--device', '7', '--backend', 'c8', '--metrics', 'PipeTimeline', *extra])
            command = build_command(args)
            worker = arguments(command[command.index('--worker'):])
            self.assertEqual(worker.backend, 'c8')
            self.assertEqual(worker.p_scale, scale)
            name = next(item.split('=', 1)[1] for item in command if item.startswith('--mstx-include='))
            self.assertEqual(name, 'C8A8_PROFILE')
            library = SimpleNamespace(mstxRangeStartA=Mock(return_value=0), mstxRangeEnd=Mock())
            torch = SimpleNamespace(npu=SimpleNamespace(synchronize=Mock()))
            with patch('ctypes.CDLL', return_value=library):
                with profile_range(torch, None, worker.mstx_backend, name):
                    pass
            library.mstxRangeStartA.assert_called_once_with(b'C8A8_PROFILE', None)
            library.mstxRangeEnd.assert_called_once_with(0)

    def test_probability_scale_rejects_invalid_and_c4_arguments(self):
        with patch('sys.stderr'):
            for value in ('0', '-1', 'nan', 'inf'):
                with self.assertRaises(SystemExit):
                    arguments(['--device', '7', '--backend', 'c8', '--p-scale', value])
            with self.assertRaises(SystemExit):
                arguments(['--device', '7', '--p-scale', '1'])

    def test_cann_zero_range_is_closed_after_sync_even_on_error(self):
        events = []
        library = SimpleNamespace(mstxRangeStartA=Mock(return_value=0),
                                  mstxRangeEnd=Mock(side_effect=lambda token: events.append(('end', token))))
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append(('sync',))))
        with patch('ctypes.CDLL', return_value=library):
            with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                with profile_range(torch, None, 'cann'):
                    raise RuntimeError('kernel failed')
        self.assertEqual(events, [('sync',), ('end', 0)])

    @staticmethod
    def basic(root, name='QuantFlashAttn_test', duration='1.25'):
        with (root / 'OpBasicInfo.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Op Name', 'Task Duration(us)'])
            writer.writerow([name, duration])

    def test_empty_metadata_only_and_invalid_duration_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, 'No OpBasicInfo'):
                validate_output(root, 'BasicInfo')
            self.basic(root, 'QuantFlashAttnMetadata_test')
            with self.assertRaisesRegex(RuntimeError, 'No target'):
                validate_output(root, 'BasicInfo')
            for duration in ('0', '-1', 'nan', 'inf'):
                self.basic(root, duration=duration)
                with self.assertRaisesRegex(RuntimeError, 'Invalid target'):
                    validate_output(root, 'BasicInfo')
            self.basic(root)
            self.assertEqual(validate_output(root, 'BasicInfo')['reports'][0]['task_duration_us'], [1.25])

    def test_requested_metrics_must_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.basic(root)
            with self.assertRaisesRegex(RuntimeError, 'Missing or empty requested metric'):
                validate_output(root, 'Default')
            for metric in ('PipeTimeline', 'InstrTimeline'):
                with self.assertRaisesRegex(RuntimeError, 'Missing pipeline trace'):
                    validate_output(root, metric)
            (root / 'trace.json').write_text('{"traceEvents":[{"ph":"M","name":"process_name"}]}')
            for metric in ('PipeTimeline', 'InstrTimeline'):
                with self.assertRaisesRegex(RuntimeError, 'No pipeline spans'):
                    validate_output(root, metric)

    def test_timestamped_multilaunch_reports_pair_exact_metric_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(2):
                task = root / 'OPPROF_test' / 'QuantFlashAttn_test' / str(index)
                task.mkdir(parents=True)
                self.basic(task)
                suffix = f'_2026111314473832{index}.csv'
                (task / 'OpBasicInfo.csv').rename(task / ('OpBasicInfo' + suffix))
                for name in DEFAULT_METRICS:
                    (task / (name + suffix)).write_text('Core,Value\n0,1\n')
            summary = validate_output(root, 'Default')
            self.assertEqual(len(summary['reports']), 2)
            (task / ('Memory' + suffix)).rename(task / 'Memory.csv')
            with self.assertRaisesRegex(RuntimeError, 'Missing or empty requested metric'):
                validate_output(root, 'Default')

    def test_dry_run_never_launches_or_creates_output(self):
        from ..prof_quant_flash_attn import main
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            with patch('subprocess.run') as run:
                main(['--device', '7', '--dry-run', '--output', str(output)])
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_data_loss_is_partial_even_with_valid_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.basic(root)
            (root / 'profile.log').write_text('[WARN] InstrProf channel 20 has data loss\n')
            summary = validate_output(root, 'BasicInfo')
            self.assertEqual(summary['collection_status'], 'partial')
            self.assertEqual(len(summary['data_loss_warnings']), 1)

    def test_named_case_uses_latest_lengths_and_fixed_pool(self):
        args = arguments(['--device', '7', '--case', 'decode_3'])
        case = named_cases(args)[0]
        self.assertEqual(args.case_names, ['bench16k_decode_3'])
        self.assertEqual(case['seqused_q'], [1]*32)
        self.assertEqual(case['seqused_kv'], [2048+11*i for i in range(16)] + [256+25*i for i in range(16)])
        self.assertEqual((case['N_q'], case['N_kv'], case['D'], case['page']), (16, 2, 256, 128))
        self.assertEqual((case['mask_mode'], case['q_scale_layout'], case['kv_cache_num_blocks']), (3, 'N2TGD', 31374))
        self.assertEqual(case['logical_kv_cache_shape'], [31374, 128, 512])
        self.assertEqual(len(named_cases(arguments(['--device', '7', '--case', 'all']))), 12)

    def test_named_worker_roundtrip_for_both_tools(self):
        for profiler in ('msprof', 'op'):
            args = arguments(['--device', '7', '--profiler', profiler, '--case', 'decode_4',
                              '--backend', 'c8', '--p-scale', '80', '--seed', '123', '--quant-schema', '0'])
            command = build_command(args)
            worker = arguments(command[command.index('--worker'):])
            self.assertEqual(named_cases(args), named_cases(worker))
            self.assertEqual(worker.profiler, profiler)
            self.assertEqual(worker.p_scale, 80.)
            self.assertEqual(named_cases(worker)[0]['seed'], 123)
            self.assertEqual(named_cases(worker)[0]['quant_schema'], 0)
            self.assertEqual(named_cases(worker)[0]['c8_p_scale'], 80.)
            self.assertNotIn('--q-lens', command)
            if profiler == 'msprof':
                self.assertNotEqual(command[1], 'op')
                self.assertIn('--aic-metrics=PipeUtilization', command)
                self.assertFalse(any(value.startswith('--mstx=') or value.startswith('--mstx-include=') for value in command))

    def test_invalid_case_combinations_fail_before_collection(self):
        invalid = [
            ['--case', 'missing'], ['--case', 'decode_3', 'bench16k_decode_3'],
            ['--case', 'decode_3', '--heads', '8'], ['--case', 'decode_3', '--kv-cache-num-blocks', '32'],
            ['--profiler', 'msprof', '--mode', 'graph'], ['--worker', '--profiler', 'both'],
            ['--worker', '--backend', 'both'], ['--worker', '--case', 'all'],
            ['--kv-cache-num-blocks', '1'], ['--backend', 'both', '--p-scale', 'nan'],
        ]
        with patch('sys.stderr'):
            for extra in invalid:
                with self.subTest(args=extra), self.assertRaises(SystemExit):
                    arguments(['--device', '7', *extra])

    def test_explicit_no_mstx_is_preserved(self):
        args = arguments(['--device', '7', '--case', 'decode_3', '--no-mstx'])
        command = build_command(args)
        worker = arguments(command[command.index('--worker'):])
        self.assertTrue(worker.no_mstx)
        self.assertFalse(any(value.startswith('--mstx=') or value.startswith('--mstx-include=') for value in command))

    def test_manual_fixed_pool_forwarded(self):
        args = arguments(['--device', '7', '--kv-cache-num-blocks', '31374'])
        command = build_command(args)
        self.assertEqual(arguments(command[command.index('--worker'):]).kv_cache_num_blocks, 31374)

    def test_case_listing_and_matrix_dry_run_are_offline(self):
        with patch('subprocess.run') as run, redirect_stdout(io.StringIO()) as output:
            main(['--list-cases'])
        self.assertEqual(len(json.loads(output.getvalue())), 12)
        run.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'absent'
            with patch('subprocess.run') as run, redirect_stdout(io.StringIO()) as output:
                main(['--device', '7', '--case', 'decode_3', '--backend', 'both', '--profiler', 'both',
                      '--dry-run', '--output', str(target)])
            self.assertEqual(output.getvalue().count('MSPROF_COMMAND'), 4)
            self.assertFalse(target.exists())
            run.assert_not_called()

    def test_matrix_runs_each_selected_combination_once(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'matrix'
            leaves = []
            def profile(args):
                leaves.append((args.case_names[0], args.backend, args.profiler, args.p_scale, args.output))
                return dict(collection_status='validated')
            with patch('qfa_mxfp4_test.prof_quant_flash_attn.profile_one', side_effect=profile), redirect_stdout(io.StringIO()):
                main(['--device', '7', '--case', 'decode_3', 'decode_4', '--backend', 'both',
                      '--profiler', 'both', '--output', str(target)])
            self.assertEqual(len(leaves), 8)
            self.assertEqual(len({(name, backend, profiler) for name, backend, profiler, _, _ in leaves}), 8)
            self.assertTrue(all(scale == (448. if backend == 'c8' else None) for _, backend, _, scale, _ in leaves))
            self.assertEqual(len(json.loads((target / 'collection_summary.json').read_text())), 8)
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                main(['--device', '7', '--case', 'decode_3', '--profiler', 'both', '--output', str(target)])

    def test_application_samples_exclude_warmup_and_post_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (root / 'op_summary_0.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['OP Type', 'Task Duration(us)', 'Task Start Time(us)'])
                writer.writeheader()
                for index, value in enumerate([999., 999., 8., 10., 12., 999.]):
                    writer.writerow({'OP Type': 'QuantFlashAttn', 'Task Duration(us)': value, 'Task Start Time(us)': index})
                writer.writerow({'OP Type': 'QuantFlashAttnMetadata', 'Task Duration(us)': 1., 'Task Start Time(us)': 100})
            (root / 'profile.log').write_text('normal run\n')
            worker = dict(profile_invocations=6, warmup=2, repeat=3)
            summary = validate_application_output(root, worker)
            self.assertEqual((summary['target_task_count'], summary['samples_us'], summary['median_us']), (6, [8., 10., 12.], 10.))
            with self.assertRaisesRegex(RuntimeError, 'Target task count'):
                validate_application_output(root, dict(worker, profile_invocations=7))
            (root / 'profile.log').write_text('InstrProf has data loss\n')
            self.assertEqual(validate_application_output(root, worker)['collection_status'], 'partial')


if __name__ == '__main__':
    unittest.main()
