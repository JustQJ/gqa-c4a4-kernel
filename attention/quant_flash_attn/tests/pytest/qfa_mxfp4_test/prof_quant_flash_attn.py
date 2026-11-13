#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.

"""Profile one C4A4 or C8A8 public-API case with msprof op; no reference computation.

Run from the pytest parent: python -m qfa_mxfp4_test.prof_quant_flash_attn
--device 7 --metrics BasicInfo --output /path/to/new-output.
Use --worker only when wrapping the application in an external msprof command.
"""
import argparse
from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

RANGE_NAME = 'C4A4_PROFILE'
KERNEL_PREFIX = 'QuantFlashAttn_'
DEFAULT_METRICS = ('ArithmeticUtilization', 'L2Cache', 'Memory', 'MemoryL0',
                   'MemoryUB', 'PipeUtilization', 'ResourceConflictRatio')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', '--device-id', type=int, required=True)
    parser.add_argument('--package', default=os.getenv('QFA_PACKAGE', 'cann_ops_transformer'))
    parser.add_argument('--backend', choices=('c4', 'c8'), default='c4')
    parser.add_argument('--p-scale', type=float, help='C8 probability scale; defaults to 448 (C8 only)')
    parser.add_argument('--q-lens', type=int, nargs='+', default=[1024])
    parser.add_argument('--kv-lens', type=int, nargs='+', default=[4096])
    parser.add_argument('--d', type=int, choices=(128, 256), default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--kv-heads', type=int, default=1)
    parser.add_argument('--page', type=int, choices=(128, 256, 512), default=128)
    parser.add_argument('--layout', choices=('TND', 'N2TGD'), default='N2TGD',
                        help='Q scale layout; KV always uses PA_BBND')
    parser.add_argument('--mask', type=int, choices=(0, 3), default=3)
    parser.add_argument('--quant-schema', type=int, choices=(0, 1, 2), default=2,
                        help='C4 probability quantization scheme; ignored by C8')
    parser.add_argument('--lse', action='store_true')
    parser.add_argument('--seed', type=int, default=421)
    parser.add_argument('--warmup', type=int, default=20, help='Application calls outside MSTX')
    parser.add_argument('--repeat', type=int, help='Calls inside MSTX; defaults to launch-count')
    parser.add_argument('--mode', choices=('eager', 'graph'), default='eager',
                        help='graph is diagnostic: msprof op may produce empty reports on 950')
    parser.add_argument('--include-metadata', action='store_true',
                        help='Regenerate metadata per call; kernel filter still selects attention only')
    parser.add_argument('--mstx-backend', choices=('cann', 'torch'), default='cann',
                        help='Use CANN host range; torch stream marker is diagnostic only')
    parser.add_argument('--post-range-check', action='store_true',
                        help='One extra attention call after MSTX to check range filtering')
    parser.add_argument('--metrics', choices=('BasicInfo', 'Default', 'PipeTimeline'), default='BasicInfo')
    parser.add_argument('--prof-warmup', type=int, help='Tool replay warmup: Default=10, others=3')
    parser.add_argument('--launch-count', type=int, help='Tool capture limit: Default=5, others=1')
    parser.add_argument('--output', help='New output directory; existing directories are rejected')
    parser.add_argument('--msprof-bin', default='msprof')
    parser.add_argument('--worker', action='store_true', help='Run the application without launching msprof')
    parser.add_argument('--dry-run', action='store_true', help='Print configuration/command without Torch or NPU')
    args = parser.parse_args(argv)
    if args.backend == 'c8':
        if args.p_scale is None:
            args.p_scale = 448.
        if not math.isfinite(args.p_scale) or args.p_scale <= 0:
            parser.error('p-scale must be finite and positive')
    elif args.p_scale is not None:
        parser.error('p-scale requires --backend c8')
    if len(args.q_lens) != len(args.kv_lens) or min(args.q_lens + args.kv_lens) <= 0:
        parser.error('Q/KV lists must have equal batch length and positive lengths')
    if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
        parser.error('heads must be a positive multiple of kv-heads')
    if args.device < 0 or args.seed < 0:
        parser.error('device and seed must be nonnegative')
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
    for name in ('device', 'package', 'backend', 'd', 'heads', 'kv_heads', 'page', 'layout', 'mask',
                 'quant_schema', 'seed', 'warmup', 'repeat', 'mode', 'mstx_backend',
                 'metrics', 'prof_warmup', 'launch_count', 'output'):
        worker.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    if args.p_scale is not None:
        worker.extend(['--p-scale', str(args.p_scale)])
    for name in ('q_lens', 'kv_lens'):
        worker.extend(['--' + name.replace('_', '-'), *map(str, getattr(args, name))])
    for name in ('lse', 'include_metadata', 'post_range_check'):
        if getattr(args, name):
            worker.append('--' + name.replace('_', '-'))
    return [args.msprof_bin, 'op', '--aic-metrics=' + args.metrics,
            '--kernel-name=' + KERNEL_PREFIX, '--mstx=true', '--mstx-include=' + range_name(args.backend),
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
        if metrics == 'PipeTimeline':
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


def run_worker(args):
    import torch
    import torch_npu

    # Also support direct-file invocation from any working directory.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qfa_mxfp4_test.common import quant_flash_attn_golden as golden
    from qfa_mxfp4_test.quant_flash_attn_paramset_common import normalize_params

    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    package = importlib.import_module(args.package)
    print('QFA_PACKAGE', getattr(package, '__file__', None), flush=True)
    params = normalize_params(dict(
        B=len(args.q_lens), N_q=args.heads, N_kv=args.kv_heads, D=args.d,
        cu_seqlens_q=[0] + list(itertools.accumulate(args.q_lens)), seqused_q=args.q_lens,
        seqused_kv=args.kv_lens, max_seqlen_q=max(args.q_lens), max_seqlen_kv=max(args.kv_lens),
        block_size=args.page, mask_mode=args.mask, q_scale_layout=args.layout,
        enable_lse=args.lse, quant_schema=args.quant_schema, seed=args.seed))
    data = golden.prepare_npu_inputs(golden.generate_data(params), f'npu:{args.device}')
    quant_mode = 5 if args.backend == 'c4' else 1
    payloads = tuple(data.pop(key) for key in ('q_fp4', 'k_fp4', 'v_fp4'))
    if args.backend == 'c8':
        from qfa_mxfp4_test.common.perf_convert import convert_payload
        # Exact E2M1 -> E4M3 expansion; retain the same E8M0 scales and page table.
        # Conversion/JIT and p_scale allocation stay outside the profiled range.
        payloads = tuple(convert_payload(tensor.view(torch.uint8), 'c8', 1.) for tensor in payloads)
    tensors = payloads + tuple(data[key] for key in ('dequant_scale_q', 'dequant_scale_k', 'v_descale'))
    p_scale = (torch.tensor([args.p_scale], dtype=torch.float32, device=payloads[0].device)
               if args.backend == 'c8' else None)
    print('OPERATOR_CONFIG', json.dumps(dict(backend=args.backend, quant_mode=quant_mode,
          payload_dtypes=[str(t.dtype) for t in payloads], payload_shapes=[list(t.shape) for t in payloads],
          p_scale=args.p_scale, mstx_range=range_name(args.backend))), flush=True)
    common = dict(cu_seqlens_q=data['cu_seqlens_q'], seqused_q=data['seqused_q'],
                  seqused_kv=data['seqused_kv'], max_seqlen_q=params['max_seqlen_q'],
                  max_seqlen_kv=params['max_seqlen_kv'], mask_mode=args.mask, layout_q='TND',
                  layout_q_descale=args.layout, layout_kv='PA_BBND', layout_out='TND')
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
        with profile_range(torch, torch_npu, args.mstx_backend, name):
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
    finally:
        if graph is not None:
            graph.reset()


def main(argv=None):
    args = arguments(argv)
    print('PROFILE_CONFIG', json.dumps(vars(args)), flush=True)
    command = None if args.worker else build_command(args)
    if command:
        print('MSPROF_COMMAND', shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.worker:
        run_worker(args)
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'invocation.json').write_text(
        json.dumps({'config': vars(args), 'command': command}, indent=2) + '\n', encoding='utf-8')
    # Do not use a shell: package names, paths and user arguments remain distinct argv entries.
    with (root / 'profile.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'msprof op exited {result.returncode}; see {root / "profile.log"}')
    summary = validate_output(root, args.metrics)
    summary.update(backend=args.backend, quant_mode=5 if args.backend == 'c4' else 1,
                   p_scale=args.p_scale, mstx_range=range_name(args.backend))
    (root / 'profile_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if summary['data_loss_warnings']:
        raise RuntimeError(f'Partial profiling data (data loss); see {root / "profile_summary.json"}')
    print('PROFILE_OK', root / 'profile_summary.json', flush=True)


if __name__ == '__main__':
    main()
