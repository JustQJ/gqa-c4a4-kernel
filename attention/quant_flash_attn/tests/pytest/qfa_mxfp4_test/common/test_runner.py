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

"""Same gen -> cpu -> npu -> compare stages as the MXFP8 test runner."""
import json
import sys
import torch
from . import golden_cache, quant_flash_attn_golden as golden, result_compare_method
from ..quant_flash_attn_paramset_common import normalize_params


def apply_params(params):
    # Keep each invocation independent; no shared mutable golden configuration.
    return normalize_params(params)


def validate_outputs(output, lse, params):
    expected = (params['cu_seqlens_q'][-1], params['N_q'], params['D'])
    if tuple(output.shape) != expected or output.dtype != torch.bfloat16:
        raise AssertionError(f'Attention output must be BF16 {expected}, got {output.dtype} {tuple(output.shape)}')
    if not torch.isfinite(output.float()).all():
        raise AssertionError('Attention output contains NaN/Inf')
    expected_lse = (params['N_q'], expected[0]) if params['enable_lse'] else (0,)
    if tuple(lse.shape) != expected_lse or lse.dtype != torch.float32:
        raise AssertionError(f'LSE must be FP32 {expected_lse}, got {lse.dtype} {tuple(lse.shape)}')
    if torch.isnan(lse).any() or torch.isneginf(lse).any():
        raise AssertionError('LSE may contain +inf for empty rows, but no NaN/-inf')


def compare_outputs(reference, actual, params):
    cpu_out, cpu_lse = reference
    output, lse = actual
    validate_outputs(cpu_out, cpu_lse, params)
    validate_outputs(output, lse, params)
    # Explicitly guard shape and empty/nonempty row semantics before the shared
    # comparator's flattening and empty-tensor fast path.
    if params['enable_lse'] and not torch.equal(torch.isposinf(cpu_lse), torch.isposinf(lse)):
        raise AssertionError('LSE empty-row (+inf) positions do not match the CPU reference')
    result = result_compare_method.check_result(cpu_out, output)
    lse_result = result_compare_method.check_result(cpu_lse, lse, max_diff_hd=2000) if params['enable_lse'] else None
    return result, lse_result


def _provenance(package, device_id):
    module = sys.modules.get(package)
    return dict(package=package, package_file=getattr(module, '__file__', None), device_id=device_id,
                torch=str(torch.__version__), torch_npu=str(getattr(sys.modules.get('torch_npu'), '__version__', 'unknown')))


def execute_test(params, mode, cdir=None, *, package='cann_ops_transformer', device_id=0, reference_threads=1,
                 triton_reference=False, fp64_reference=False):
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
        data = golden.generate_data(p)
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
                from .mxfp4_schedule import fetch_metadata
                metadata = fetch_metadata(data, p, package, device_id)
                golden_cache._save(name, 'schedule', metadata, p, input_id, cdir, producer={'package': package})
        return metadata
    if 'cpu' in mode:
        if p['reference_backend'] == 'kernel':
            from .cpu_mxfp4_golden_kernel import cpu_mxfp4_golden_kernel
            reference = cpu_mxfp4_golden_kernel(data, p, schedule())
        else:
            reference = golden.cpu_mxfp4_golden(data, p)
        validate_outputs(*reference, p)
        golden_cache.save_cpu_output(name, *reference, p, input_id, cdir)
    if 'npu' in mode:
        actual = golden.npu_mxfp4_fa(data, p, package, device_id)
        validate_outputs(*actual, p)
        golden_cache.save_npu_output(name, *actual, p, input_id, cdir, producer=_provenance(package, device_id))
        if triton_reference:
            from .triton_mxfp4_golden_kernel import triton_mxfp4_golden_kernel
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
        from .cpu_mxfp4_golden_fp64 import cpu_mxfp4_golden_fp64
        from .reference_metrics import reference_gap
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
        # The original pytest entry consumes this pair; propagate a Triton-only
        # failure instead of reporting a successful CPU comparison as a pass.
        result = ('Fail', 'Triton/reference cross-check failed; see comparison JSON')
    return result, lse_result


def check_results(atten_result, lse_result):
    import pytest
    for label, result in [('attention', atten_result), ('LSE', lse_result)]:
        if result is not None and result[0] != 'Pass':
            pytest.fail(f'{label} compare failed: {result}')
