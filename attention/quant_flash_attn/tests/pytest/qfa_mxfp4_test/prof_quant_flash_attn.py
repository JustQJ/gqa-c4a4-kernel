#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.

"""Profile named or manual C4/C8 cases with msprof and msprof op; no references.

Run from the pytest parent: python -m qfa_mxfp4_test.prof_quant_flash_attn
--device 7 --metrics BasicInfo --output /path/to/new-output.
Use --worker only when wrapping the application in an external msprof command.
"""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import csv
import ctypes
from datetime import datetime
import importlib
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfa_mxfp4_test.quant_flash_attn_paramset_common import DEFAULT_KV_LAYOUT, KV_LAYOUTS
from qfa_mxfp4_test.quant_flash_attn_paramset_perf_rdv import with_kv_layout

RANGE_NAME = 'C4A4_PROFILE'
KERNEL_PREFIX = 'QuantFlashAttn_'
DEFAULT_METRICS = ('ArithmeticUtilization', 'L2Cache', 'Memory', 'MemoryL0',
                   'MemoryUB', 'PipeUtilization', 'ResourceConflictRatio')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', '--device-id', type=int)
    parser.add_argument('--profiler', choices=('msprof', 'op', 'both'), default='op',
                        help='Application profiling, operator profiling, or both sequentially')
    parser.add_argument('--case-set', choices=('bench16k', 'qwen3_8b', 'qwen3_8b_page256'), default='bench16k')
    parser.add_argument('--case', '--case-name', dest='case_names', nargs='+',
                        help='Names in the selected set, e.g. decode_3; use all for the full collection')
    parser.add_argument('--list-cases', action='store_true', help='List names and parameters without NPU access')
    parser.add_argument('--package', default=os.getenv('QFA_PACKAGE', 'cann_ops_transformer'))
    parser.add_argument('--backend', choices=('c4', 'c8', 'both'), default='c4')
    parser.add_argument('--p-scale', type=float, help='C8 probability scale; defaults to 448 (C8 only)')
    parser.add_argument('--q-lens', type=int, nargs='+')
    parser.add_argument('--kv-lens', type=int, nargs='+')
    parser.add_argument('--d', type=int, choices=(128, 256))
    parser.add_argument('--heads', type=int)
    parser.add_argument('--kv-heads', type=int)
    parser.add_argument('--page', type=int, choices=(128, 256, 512))
    parser.add_argument('--kv-cache-num-blocks', type=int, help='Optional fixed physical page count for manual input')
    parser.add_argument('--layout', choices=('TND', 'N2TGD'),
                        help='Q scale layout')
    parser.add_argument('--kv-layout', '--kv_layout', choices=KV_LAYOUTS, default=DEFAULT_KV_LAYOUT,
                        help='Paged KV layout used by the QFA kernel under test')
    parser.add_argument('--mask', type=int, choices=(0, 3))
    parser.add_argument('--quant-schema', type=int, choices=(0, 1, 2),
                        help='C4 probability quantization scheme; ignored by C8')
    parser.add_argument('--lse', action='store_true', default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--warmup', type=int, default=20, help='Application calls outside MSTX')
    parser.add_argument('--repeat', type=int, help='Measured calls (inside MSTX for op); defaults to launch-count')
    parser.add_argument('--mode', choices=('eager', 'graph'), default='eager',
                        help='graph is diagnostic: msprof op may produce empty reports on 950')
    parser.add_argument('--include-metadata', action='store_true',
                        help='Regenerate metadata per call; kernel filter still selects attention only')
    parser.add_argument('--mstx-backend', choices=('cann', 'torch'), default='cann',
                        help='Use CANN host range; torch stream marker is diagnostic only')
    parser.add_argument('--no-mstx', action='store_true',
                        help='op: kernel-only capture, possibly application warmup; no MSTX filtering')
    parser.add_argument('--post-range-check', action='store_true',
                        help='One extra attention call after MSTX to check range filtering')
    parser.add_argument('--metrics', choices=('BasicInfo', 'Default', 'PipeTimeline', 'InstrTimeline'), default='BasicInfo')
    parser.add_argument('--app-metrics', choices=DEFAULT_METRICS, default='PipeUtilization',
                        help='Ordinary msprof metric group; --metrics controls msprof op only')
    parser.add_argument('--prof-warmup', type=int, help='Tool replay warmup: Default=10, others=3')
    parser.add_argument('--launch-count', type=int, help='Tool capture limit: Default=5, others=1')
    parser.add_argument('--output', help='New output directory; existing directories are rejected')
    parser.add_argument('--msprof-bin', default='msprof')
    parser.add_argument('--worker', action='store_true', help='Run the application without launching msprof')
    parser.add_argument('--dry-run', action='store_true', help='Print configuration/command without Torch or NPU')
    args = parser.parse_args(argv)
    if args.list_cases:
        return args
    if args.device is None or args.device < 0:
        parser.error('device must be provided and nonnegative')
    if args.worker and (args.profiler == 'both' or args.backend == 'both'):
        parser.error('worker requires one profiler and one backend')
    if args.mode == 'graph' and args.profiler != 'op':
        parser.error('ordinary msprof requires eager mode for exact invocation association')
    if args.case_names:
        from qfa_mxfp4_test.quant_flash_attn_paramset_perf_rdv import get_cases
        names = [case['name'] for case in get_cases(case_set=args.case_set)]
        if args.case_names == ['all']:
            args.case_names = names
        else:
            prefix = args.case_set + '_'
            args.case_names = [name if name.startswith(prefix) else prefix + name for name in args.case_names]
        if len(set(args.case_names)) != len(args.case_names) or set(args.case_names) - set(names):
            parser.error(f'case names must be unique existing {args.case_set} names; use --list-cases')
        shape_options = ('q_lens', 'kv_lens', 'd', 'heads', 'kv_heads', 'page', 'layout', 'mask', 'lse', 'kv_cache_num_blocks')
        if any(getattr(args, name) is not None for name in shape_options):
            parser.error('named cases own their shapes, layout, mask, LSE and pool; do not combine manual shape options')
        if args.worker and len(args.case_names) != 1:
            parser.error('worker requires exactly one named case')
    else:
        defaults = dict(q_lens=[1024], kv_lens=[4096], d=256, heads=8, kv_heads=1, page=128,
                        layout='N2TGD', mask=3, lse=False, seed=421, quant_schema=2)
        for name, value in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        if len(args.q_lens) != len(args.kv_lens) or min(args.q_lens + args.kv_lens) <= 0:
            parser.error('Q/KV lists must have equal batch length and positive lengths')
        if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
            parser.error('heads must be a positive multiple of kv-heads')
        from qfa_mxfp4_test.quant_flash_attn_paramset_common import paged_kv_pool_size
        try:
            paged_kv_pool_size(args.kv_lens, args.page, args.kv_cache_num_blocks)
        except ValueError as error:
            parser.error(str(error))
    if args.backend == 'c8':
        if args.p_scale is None:
            args.p_scale = 448.
        if not math.isfinite(args.p_scale) or args.p_scale <= 0:
            parser.error('p-scale must be finite and positive')
    elif args.backend == 'c4' and args.p_scale is not None:
        parser.error('p-scale requires --backend c8')
    if args.p_scale is not None and (not math.isfinite(args.p_scale) or args.p_scale <= 0):
        parser.error('p-scale must be finite and positive')
    if args.seed is not None and args.seed < 0:
        parser.error('seed must be nonnegative')
    if args.prof_warmup is None:
        args.prof_warmup = 10 if args.metrics == 'Default' else 3
    if args.launch_count is None:
        args.launch_count = 5 if args.metrics == 'Default' else 1
    if args.repeat is None:
        args.repeat = args.launch_count
    if min(args.warmup, args.repeat, args.prof_warmup, args.launch_count) < 1:
        parser.error('warmup, repeat, prof-warmup and launch-count must be positive')
    if args.output is None:
        args.output = str(Path('prof-op') / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    return args


def range_name(backend):
    return RANGE_NAME if backend == 'c4' else 'C8A8_PROFILE'


def build_command(args):
    worker = [sys.executable, str(Path(__file__).resolve()), '--worker']
    names = ['device', 'package', 'profiler', 'backend', 'case_set', 'kv_layout', 'quant_schema', 'seed', 'warmup',
             'repeat', 'mode', 'mstx_backend', 'metrics', 'app_metrics', 'prof_warmup', 'launch_count', 'output']
    if args.case_names:
        worker.extend(['--case', *args.case_names])
    else:
        names += ['d', 'heads', 'kv_heads', 'page', 'layout', 'mask', 'kv_cache_num_blocks']
    for name in names:
        value = getattr(args, name)
        if value is not None:
            worker.extend(['--' + name.replace('_', '-'), str(value)])
    if args.p_scale is not None:
        worker.extend(['--p-scale', str(args.p_scale)])
    if not args.case_names:
        for name in ('q_lens', 'kv_lens'):
            worker.extend(['--' + name.replace('_', '-'), *map(str, getattr(args, name))])
    for name in ('lse', 'include_metadata', 'post_range_check', 'no_mstx'):
        if getattr(args, name):
            worker.append('--' + name.replace('_', '-'))
    if args.profiler == 'msprof':
        return [args.msprof_bin, '--output=' + str(Path(args.output).resolve()),
                '--aic-metrics=' + args.app_metrics, *worker]
    marker = [] if args.no_mstx else ['--mstx=true', '--mstx-include=' + range_name(args.backend)]
    return [args.msprof_bin, 'op', '--aic-metrics=' + args.metrics,
            '--kernel-name=' + KERNEL_PREFIX, *marker,
            '--warm-up=' + str(args.prof_warmup), '--launch-count=' + str(args.launch_count),
            '--output=' + str(Path(args.output).resolve()), *worker]


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def validate_output(output, metrics):
    """Reject exit-0 empty collections, unrelated kernels and missing requested products."""
    root = Path(output)
    reports = []
    for basic in sorted(root.rglob('OpBasicInfo*.csv')):
        if not re.fullmatch(r'OpBasicInfo(?:_\d+)?\.csv', basic.name):
            continue
        # Multiple captured launches add the same timestamp to all CSVs in a task directory.
        suffix = basic.name[len('OpBasicInfo'):-len('.csv')]
        rows = [row for row in csv_rows(basic) if row.get('Op Name', '').startswith(KERNEL_PREFIX)]
        if not rows:
            raise RuntimeError(f'No target kernel rows: {basic}')
        durations = [float(row['Task Duration(us)']) for row in rows]
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise RuntimeError(f'Invalid target kernel duration: {basic}')
        report = {'directory': str(basic.parent.relative_to(root)),
                  'basic_csv': str(basic.relative_to(root)), 'target_rows': len(rows),
                  'task_duration_us': durations, 'metric_rows': {}}
        if metrics == 'Default':
            for name in DEFAULT_METRICS:
                path = basic.with_name(name + suffix + '.csv')
                count = len(csv_rows(path)) if path.is_file() else 0
                if not count:
                    raise RuntimeError(f'Missing or empty requested metric: {path}')
                report['metric_rows'][name] = count
        if metrics in ('PipeTimeline', 'InstrTimeline'):
            trace = basic.with_name('trace' + suffix + '.json')
            if not trace.is_file():
                trace = basic.with_name('trace.json')
            if not trace.is_file():
                raise RuntimeError(f'Missing pipeline trace: {trace}')
            events = json.loads(trace.read_text(encoding='utf-8'))
            if isinstance(events, dict):
                events = events.get('traceEvents', [])
            spans = [event for event in events if event.get('ph') == 'X' and float(event.get('dur', 0)) > 0]
            if not spans:
                raise RuntimeError(f'No pipeline spans: {trace}')
            report['timeline_events'] = len(events)
            report['timeline_spans'] = len(spans)
            report['trace'] = str(trace.relative_to(root))
        reports.append(report)
    if not reports:
        raise RuntimeError(f'No OpBasicInfo CSV found under {root}; msprof exit 0 alone is not success')
    log = root / 'profile.log'
    warnings = ([line for line in log.read_text(encoding='utf-8', errors='replace').splitlines()
                 if 'has data loss' in line.lower()] if log.is_file() else [])
    return {'metrics': metrics, 'reports': reports, 'data_loss_warnings': warnings,
            'collection_status': 'partial' if warnings else 'validated'}


@contextmanager
def profile_range(torch, torch_npu, backend, name=RANGE_NAME):
    if backend == 'cann':
        mstx = ctypes.CDLL('libms_tools_ext.so')
        mstx.mstxRangeStartA.argtypes = (ctypes.c_char_p, ctypes.c_void_p)
        mstx.mstxRangeStartA.restype = ctypes.c_uint64
        mstx.mstxRangeEnd.argtypes = (ctypes.c_uint64,)
        mstx.mstxRangeEnd.restype = None
        token = mstx.mstxRangeStartA(name.encode(), None)
        end_range = mstx.mstxRangeEnd
    else:
        token = torch_npu.npu.mstx.range_start(name, torch.npu.current_stream())
        end_range = torch_npu.npu.mstx.range_end
    print('MSTX_RANGE_ID', token, 'backend=' + backend, flush=True)
    try:
        yield
    finally:
        try:
            torch.npu.synchronize()
        finally:
            # CANN profiler injection may return 0 for a valid range: always end it.
            if backend == 'cann' or token is not None:
                end_range(token)


def _prepare_manual(args):
    import torch
    import torch_npu

    # Also support direct-file invocation from any working directory.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qfa_mxfp4_test.common import quant_flash_attn_golden as golden
    from qfa_mxfp4_test.quant_flash_attn_paramset_common import normalize_params

    params = normalize_params(dict(
        B=len(args.q_lens), N_q=args.heads, N_kv=args.kv_heads, D=args.d,
        cu_seqlens_q=[0] + list(itertools.accumulate(args.q_lens)), seqused_q=args.q_lens,
        seqused_kv=args.kv_lens, max_seqlen_q=max(args.q_lens), max_seqlen_kv=max(args.kv_lens),
        block_size=args.page, mask_mode=args.mask, q_scale_layout=args.layout,
        enable_lse=args.lse, quant_schema=args.quant_schema, seed=args.seed,
        kv_cache_num_blocks=args.kv_cache_num_blocks))
    cpu_data = golden.generate_data(params)
    digest = hashlib.sha256()
    for key in sorted(cpu_data):
        if cpu_data[key] is not None:
            digest.update(memoryview(cpu_data[key].numpy()).cast('B'))
    data = golden.prepare_npu_inputs(cpu_data, f'npu:{args.device}')
    del cpu_data
    quant_mode = 5 if args.backend == 'c4' else 1
    q, k, v = (data.pop(key) for key in ('q_fp4', 'k_fp4', 'v_fp4'))
    if args.backend == 'c8':
        from qfa_mxfp4_test.common.perf_convert import convert_payload
        # Exact E2M1 -> E4M3 expansion; retain the same E8M0 scales and page table.
        # Conversion/JIT and p_scale allocation stay outside the profiled range.
        q, k, v = (convert_payload(tensor.view(torch.uint8), 'c8', 1.) for tensor in (q, k, v))
    k, v, ks, vs = golden.convert_paged_kv(k, v, data['dequant_scale_k'], data['v_descale'], args.kv_layout)
    payloads = (q, k, v)
    tensors = (*payloads, data['dequant_scale_q'], ks, vs)
    p_scale = (torch.tensor([args.p_scale], dtype=torch.float32, device=q.device)
               if args.backend == 'c8' else None)
    print('OPERATOR_CONFIG', json.dumps(dict(backend=args.backend, quant_mode=quant_mode,
          payload_dtypes=[str(t.dtype) for t in payloads], payload_shapes=[list(t.shape) for t in payloads],
          p_scale=args.p_scale, mstx_range=range_name(args.backend))), flush=True)
    common = dict(cu_seqlens_q=data['cu_seqlens_q'], seqused_q=data['seqused_q'],
                  seqused_kv=data['seqused_kv'], max_seqlen_q=params['max_seqlen_q'],
                  max_seqlen_kv=params['max_seqlen_kv'], mask_mode=args.mask, layout_q='TND',
                  layout_q_descale=args.layout, layout_kv=args.kv_layout, layout_out='TND')
    ops = torch.ops.cann_ops_transformer

    def get_metadata():
        return ops.quant_flash_attn_metadata(args.heads, args.kv_heads, args.d, quant_mode, **common)

    metadata = get_metadata()
    print('METADATA_HEADER', metadata.cpu().reshape(-1)[:5].tolist(), flush=True)
    mask = (torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(payloads[0].device)
            if args.mask else None)

    def invoke():
        current = get_metadata() if args.include_metadata else metadata
        return ops.quant_flash_attn(
            *tensors, quant_mode, block_table=data['block_table_torch'], metadata=current, attn_mask=mask,
            p_scale=p_scale, softmax_scale=params['softmax_scale'], return_softmax_lse=args.lse,
            quant_schema=args.quant_schema, **common)

    case = dict(name='manual', B=params['B'], N_q=args.heads, N_kv=args.kv_heads, D=args.d,
                page=args.page, mask_mode=args.mask, q_scale_layout=args.layout,
                cu_seqlens_q=params['cu_seqlens_q'], seqused_q=args.q_lens, seqused_kv=args.kv_lens,
                softmax_scale=params['softmax_scale'], enable_lse=args.lse, quant_schema=args.quant_schema,
                seed=args.seed, c8_p_scale=args.p_scale if args.backend == 'c8' else 448.,
                kv_cache_num_blocks=args.kv_cache_num_blocks)
    case = with_kv_layout(case, args.kv_layout)
    shapes = {name: list(tensor.shape) for name, tensor in zip(('q', 'k', 'v', 'qs', 'ks', 'vs'), tensors)}
    shapes['table'] = list(data['block_table_torch'].shape)
    return invoke, metadata, dict(case=case, input_sha256=digest.hexdigest(), input_shapes=shapes,
                                  payload_dtypes=[str(t.dtype) for t in payloads], input_policy='manual_accuracy_generator')


def named_cases(args):
    from qfa_mxfp4_test.quant_flash_attn_paramset_perf_rdv import get_cases
    indexed = {case['name']: case for case in get_cases(case_set=args.case_set)}
    result = []
    for name in args.case_names or []:
        case = copy.deepcopy(indexed[name])
        for key in ('quant_schema', 'seed'):
            value = getattr(args, key)
            if value is not None:
                case[key] = value
        if args.backend == 'c8' and args.p_scale is not None:
            case['c8_p_scale'] = args.p_scale
        result.append(with_kv_layout(case, args.kv_layout))
    return result


def _prepare_named(args):
    from qfa_mxfp4_test.common.perf_inputs import generate_inputs, prepare_inputs
    from qfa_mxfp4_test.common.perf_runner import prepare_call
    case = named_cases(args)[0]
    source = generate_inputs(case)
    sha = source['sha256']
    data = prepare_inputs(source, case, args.backend, f'npu:{args.device}')
    del source
    call, metadata = prepare_call(data, case, args.backend)
    if args.include_metadata:
        def call():
            return prepare_call(data, case, args.backend)[0]()
    shapes = {name: list(tensor.shape) for name, tensor in data.items() if tensor is not None}
    return call, metadata, dict(case=case, input_sha256=sha, input_shapes=shapes,
                                payload_dtypes=[str(data[key].dtype) for key in ('q', 'k', 'v')],
                                input_policy='paired_performance_generator')


def _execute_worker(args, torch, torch_npu):
    invoke, metadata, record = _prepare_named(args) if args.case_names else _prepare_manual(args)
    from qfa_mxfp4_test.quant_flash_attn_paramset_perf_rdv import case_fingerprint
    record['case_fingerprint'] = case_fingerprint(record['case'])
    record['metadata_header'] = metadata.cpu().reshape(-1).tolist()[:16]
    print('OPERATOR_CONFIG', json.dumps(record), flush=True)
    for _ in range(args.warmup):
        output = invoke()
    torch.npu.synchronize()
    graph = None
    try:
        if args.mode == 'graph':
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = invoke()
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            print('GRAPH_WARMUP_REPLAYS 3', flush=True)
        name = range_name(args.backend)
        print('PROFILE_RANGE_BEGIN', name, flush=True)
        use_range = args.profiler == 'op' and not args.no_mstx
        with profile_range(torch, torch_npu, args.mstx_backend, name) if use_range else nullcontext():
            for _ in range(args.repeat):
                if graph is None:
                    output = invoke()
                else:
                    graph.replay()
        print('PROFILE_RANGE_END', name, flush=True)
        if args.post_range_check:
            print('OUTSIDE_RANGE_CALL', flush=True)
            output = invoke()
            torch.npu.synchronize()
        case = record['case']
        expected = (case['cu_seqlens_q'][-1], case['N_q'], case['D'])
        if tuple(output[0].shape) != expected or output[0].dtype != torch.bfloat16 or not bool(torch.isfinite(output[0]).all()):
            raise RuntimeError('Invalid attention output shape/dtype or nonfinite output')
        if case['enable_lse'] and (output[1] is None or not bool(torch.isfinite(output[1]).all())):
            raise RuntimeError('Missing/nonfinite LSE output')
        record.update(finite=True, output_shape=list(expected), warmup=args.warmup, repeat=args.repeat,
                      profile_invocations=args.warmup + args.repeat + int(args.post_range_check) + (4 if graph is not None else 0))
        return record
    finally:
        if graph is not None:
            graph.reset()


def run_worker(args):
    import torch
    import torch_npu
    torch.set_num_threads(1 if args.case_names else 8)
    torch.npu.set_device(args.device)
    package = importlib.import_module(args.package)
    print('QFA_PACKAGE', getattr(package, '__file__', None), flush=True)
    before = torch.npu.memory_allocated()
    record = _execute_worker(args, torch, torch_npu)
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    from qfa_mxfp4_test.common.perf_runner import environment
    record.update(device=args.device, profiler=args.profiler, backend=args.backend,
                  allocated_before_bytes=before, allocated_after_release_bytes=torch.npu.memory_allocated(),
                  environment=environment(args.package), status='passed')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'worker_record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print('WORKER_DONE', record['case']['name'], args.profiler, args.backend,
          'released=', record['allocated_after_release_bytes'], flush=True)


def validate_application_output(output, worker):
    from qfa_mxfp4_test.common.perf_profiler import parse_tasks
    tasks, sources = parse_tasks(output)
    if len(tasks) != worker['profile_invocations']:
        raise RuntimeError(f"Target task count {len(tasks)} != worker calls {worker['profile_invocations']}")
    measured = tasks[worker['warmup']:worker['warmup'] + worker['repeat']]
    values = [row['duration_us'] for row in measured]
    if len(values) != worker['repeat']:
        raise RuntimeError('Missing measured target calls')
    log = Path(output) / 'profile.log'
    warnings = [line for line in log.read_text(errors='replace').splitlines() if 'has data loss' in line.lower()]
    return dict(metrics='application', collection_status='partial' if warnings else 'validated',
                measurement_scope='eager_single_case_diagnostic', target_task_count=len(tasks),
                measured_calls=len(values), median_us=statistics.median(values), samples_us=values,
                warmup_calls=worker['warmup'], csv_sha256=sources, profile_task_rows=tasks,
                data_loss_warnings=warnings)


def profile_one(args):
    command = build_command(args)
    print('MSPROF_COMMAND', shlex.join(command), flush=True)
    if args.dry_run:
        return None
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'invocation.json').write_text(json.dumps({'config': vars(args), 'command': command}, indent=2) + '\n')
    with (root / 'profile.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{args.profiler} exited {result.returncode}; see {root / "profile.log"}')
    worker = json.loads((root / 'worker_record.json').read_text())
    if worker['status'] != 'passed' or not worker['finite']:
        raise RuntimeError('Worker did not complete a finite-output check')
    if worker['allocated_after_release_bytes'] != worker['allocated_before_bytes']:
        raise RuntimeError('Worker did not release its NPU allocations')
    if args.profiler == 'msprof':
        summary = validate_application_output(root, worker)
    else:
        summary = validate_output(root, args.metrics)
        summary['capture_scope'] = 'kernel_filter_may_include_application_warmup' if args.no_mstx else 'mstx_range'
    summary.update(profiler=args.profiler, backend=args.backend, quant_mode=5 if args.backend == 'c4' else 1,
                   p_scale=args.p_scale, mstx_range=range_name(args.backend),
                   case=worker['case'], case_fingerprint=worker['case_fingerprint'],
                   input_sha256=worker['input_sha256'], input_shapes=worker['input_shapes'],
                   input_policy=worker['input_policy'], allocated_after_release_bytes=worker['allocated_after_release_bytes'])
    (root / 'profile_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if summary['data_loss_warnings']:
        raise RuntimeError(f'Partial profiling data (data loss); see {root / "profile_summary.json"}')
    print('PROFILE_OK', root / 'profile_summary.json', flush=True)
    return summary


def main(argv=None):
    args = arguments(argv)
    if args.list_cases:
        from qfa_mxfp4_test.quant_flash_attn_paramset_perf_rdv import get_cases
        cases = [with_kv_layout(case, args.kv_layout) for case in get_cases(case_set=args.case_set)]
        print(json.dumps(cases, indent=2))
        return
    print('PROFILE_CONFIG', json.dumps(vars(args)), flush=True)
    if args.worker:
        if not args.dry_run:
            run_worker(args)
        return
    profilers = ('msprof', 'op') if args.profiler == 'both' else (args.profiler,)
    backends = ('c4', 'c8') if args.backend == 'both' else (args.backend,)
    cases = args.case_names or [None]
    matrix = len(profilers) * len(backends) * len(cases) > 1
    root = Path(args.output).resolve()
    if matrix and not args.dry_run:
        root.mkdir(parents=True, exist_ok=False)
    results = []
    for name, backend, profiler in itertools.product(cases, backends, profilers):
        leaf = copy.deepcopy(args)
        leaf.case_names = [name] if name else None
        leaf.backend, leaf.profiler = backend, profiler
        leaf.p_scale = (args.p_scale or 448.) if backend == 'c8' else None
        leaf.output = str(root / (name or 'manual') / backend / profiler) if matrix else args.output
        try:
            summary = profile_one(leaf)
            results.append(dict(case=name or 'manual', backend=backend, profiler=profiler,
                                output=leaf.output, status='passed', summary=summary))
        except (RuntimeError, ValueError, OSError) as error:
            results.append(dict(case=name or 'manual', backend=backend, profiler=profiler,
                                output=leaf.output, status='failed', error=str(error)))
            if not matrix:
                raise
        if matrix and not args.dry_run:
            (root / 'collection_summary.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    if any(item['status'] == 'failed' for item in results):
        raise RuntimeError(f'Some profiles failed; see {root / "collection_summary.json"}')


if __name__ == '__main__':
    main()
