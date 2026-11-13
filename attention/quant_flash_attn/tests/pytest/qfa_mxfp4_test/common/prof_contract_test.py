"""CPU-only checks for profiling boundaries and false-success rejection."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ..prof_quant_flash_attn import DEFAULT_METRICS, arguments, build_command, profile_range, validate_output


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
            with self.assertRaisesRegex(RuntimeError, 'Missing pipeline trace'):
                validate_output(root, 'PipeTimeline')
            (root / 'trace.json').write_text('{"traceEvents":[{"ph":"M","name":"process_name"}]}')
            with self.assertRaisesRegex(RuntimeError, 'No pipeline spans'):
                validate_output(root, 'PipeTimeline')

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


if __name__ == '__main__':
    unittest.main()
