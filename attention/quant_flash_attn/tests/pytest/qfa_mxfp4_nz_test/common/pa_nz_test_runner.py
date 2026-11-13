#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Same gen -> cpu -> npu -> compare stages, launching the operator with PA_NZ.

Reuses the BBND runner for reference stages; the only behavioural difference
is the npu stage, which converts the cached logical inputs to PA_NZ.
"""
import json
import sys
from pathlib import Path
import torch

_PYTEST_DIR = str(Path(__file__).resolve().parents[3])
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)
from qfa_mxfp4_test.common import golden_cache, test_runner as bbnd_runner  # noqa: E402
from qfa_mxfp4_test.common import result_compare_method  # noqa: E402
from qfa_mxfp4_test.quant_flash_attn_paramset_common import normalize_params  # noqa: E402
from . import pa_nz_golden  # noqa: E402


def apply_params(params):
    return normalize_params(params)


validate_outputs = bbnd_runner.validate_outputs
compare_outputs = bbnd_runner.compare_outputs


def execute_test(params, mode, cdir=None, *, package='cann_ops_transformer', device_id=0,
                 reference_threads=1, triton_reference=False, fp64_reference=False):
    p = apply_params(params)
    mode = set(mode)
    if 'all' in mode:
        mode = {'gen', 'cpu', 'npu', 'compare'}
    if not mode or mode - {'gen', 'cpu', 'npu', 'compare'}:
        raise ValueError('Unknown or empty execution mode')
    if reference_threads < 1:
        raise ValueError('reference_threads must be positive')
    torch.set_num_threads(reference_threads)
    name = p['name']
    if 'gen' in mode:
        data = pa_nz_golden.generate_data(p)
        input_id = golden_cache.save_input(name, data, p, cdir)
    else:
        data, input_id = golden_cache.load_input(name, p, cdir)
    reference = actual = None
    metadata = None

    def schedule():
        nonlocal metadata
        if metadata is None:
            try:
                if 'gen' in mode:
                    raise FileNotFoundError('Fresh input generation requires fresh metadata')
                cached = golden_cache._load(name, 'schedule', p, input_id, cdir)
                if (cached.get('producer') or {}).get('package') != package:
                    raise ValueError('Metadata cache belongs to a different operator package')
                metadata = cached['payload']
            except FileNotFoundError:
                from qfa_mxfp4_test.common.mxfp4_schedule import fetch_metadata
                metadata = fetch_metadata(data, p, package, device_id)
                golden_cache._save(name, 'schedule', metadata, p, input_id, cdir, producer={'package': package})
        return metadata

    if 'cpu' in mode:
        if p['reference_backend'] == 'kernel':
            from qfa_mxfp4_test.common.cpu_mxfp4_golden_kernel import cpu_mxfp4_golden_kernel
            reference = cpu_mxfp4_golden_kernel(data, p, schedule())
        else:
            reference = pa_nz_golden.cpu_mxfp4_golden(data, p)
        validate_outputs(*reference, p)
        golden_cache.save_cpu_output(name, *reference, p, input_id, cdir)
    if 'npu' in mode:
        actual = pa_nz_golden.npu_mxfp4_fa_nz(data, p, package, device_id)
        validate_outputs(*actual, p)
        golden_cache.save_npu_output(name, *actual, p, input_id, cdir,
                                     producer=dict(package=package, kv_layout='PA_NZ'))
        if triton_reference:
            from qfa_mxfp4_test.common.triton_mxfp4_golden_kernel import triton_mxfp4_golden_kernel
            tri = triton_mxfp4_golden_kernel(data, p, schedule(), device_id)
            validate_outputs(*tri, p)
            golden_cache._save(name, 'triton_output', tri, p, input_id, cdir)
    if 'compare' not in mode:
        print(f'STAGES_DONE {name}: {sorted(mode)}; precision was not compared', flush=True)
        return None, None
    if reference is None:
        reference = golden_cache.load_cpu_output(name, p, input_id, cdir)
    if actual is None:
        actual = golden_cache.load_npu_output(name, p, input_id, cdir, package=package)
    result, lse_result = compare_outputs(reference, actual, p)
    report = dict(name=name, params=p, input_id=input_id, package=package, attention=result, lse=lse_result,
                  passed=result[0] == 'Pass' and (lse_result is None or lse_result[0] == 'Pass'))
    tri = None
    if triton_reference:
        tri = tuple(golden_cache._load(name, 'triton_output', p, input_id, cdir)['payload'])
        tr, tlse = compare_outputs(tri, actual, p)
        cr, clse = compare_outputs(reference, tri, p)
        report.update(triton_attention=tr, triton_lse=tlse, cpu_triton_attention=cr, cpu_triton_lse=clse)
        report['passed'] &= all(x is None or x[0] == 'Pass' for x in (tr, tlse, cr, clse))
    if report['passed'] and fp64_reference:
        from qfa_mxfp4_test.common.cpu_mxfp4_golden_fp64 import cpu_mxfp4_golden_fp64
        from qfa_mxfp4_test.common.reference_metrics import reference_gap
        try:
            if 'gen' in mode:
                raise FileNotFoundError('Fresh input generation requires fresh FP64 output')
            fp64 = tuple(golden_cache._load(name, 'fp64_output', p, input_id, cdir)['payload'])
        except FileNotFoundError:
            fp64 = cpu_mxfp4_golden_fp64(data, p)
            golden_cache._save(name, 'fp64_output', fp64, p, input_id, cdir)
        report['fp64_gap'] = dict(kernel=reference_gap(actual, fp64), cpu=reference_gap(reference, fp64))
        if tri is not None:
            report['fp64_gap']['triton'] = reference_gap(tri, fp64)
    path = golden_cache._path(name, 'comparison', cdir).with_suffix('.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"COMPARE {name}: {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    if not report['passed'] and result[0] == 'Pass' and (lse_result is None or lse_result[0] == 'Pass'):
        result = ('Fail', 'Triton/reference cross-check failed; see comparison JSON')
    return result, lse_result


check_results = bbnd_runner.check_results
