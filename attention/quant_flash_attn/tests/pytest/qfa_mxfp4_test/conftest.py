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

"""Accuracy options and dependency-free case collection."""
import os
import pytest
from .quant_flash_attn_paramset_common import P_QUANT_SCHEMES, resolve_quantization

_VALID_MODES = {'gen', 'cpu', 'npu', 'compare'}
# This is a runner module, not a pytest entry. Keep collection free of torch.
collect_ignore = ['common/test_runner.py']


def _parse_golden_mode(raw):
    parts = {part.strip() for part in raw.split(',') if part.strip()}
    if not parts or parts - (_VALID_MODES | {'all'}):
        raise pytest.UsageError('golden-mode must be all or a combination of gen,cpu,npu,compare')
    return set(_VALID_MODES) if 'all' in parts else parts


def pytest_addoption(parser):
    group = parser.getgroup('qfa_mxfp4 accuracy')
    group.addoption('--golden-mode', default='all', help='all or comma-separated gen,cpu,npu,compare')
    group.addoption('--cache-dir', default=None, help='Input and reference .pt cache directory')
    group.addoption('--scope', choices=('full', 'small'), default='full')
    group.addoption('--accuracy-case-set', choices=('document', 'bench16k'), default='document',
                    help='Functional accuracy cases: existing matrix or the explicit bench16k cases')
    group.addoption('--quant-schema', type=int, choices=(0, 1, 2), default=None,
                    help='Public API and references: 0=OCP, 1=CX, 2=CX Delta (default)')
    group.addoption('--p-quant-scheme', choices=P_QUANT_SCHEMES, default=None,
                    help='Named alias for quant-schema; must agree if both options are supplied')
    group.addoption('--reference-backend', choices=('global', 'kernel'), default='kernel',
                    help='Actual-metadata/16-task kernel model (default) or preserved global CPU golden')
    group.addoption('--triton-reference', action='store_true', help='Also compare against the metadata-based Triton reference')
    group.addoption('--fp64-reference', action='store_true', help='After precision passes, measure the gap to unquantized-P FP64')
    group.addoption('--device-id', type=int, default=0)
    group.addoption('--package', default=os.environ.get('QFA_PACKAGE', 'cann_ops_transformer'),
                    help='Installed package that registers torch.ops.cann_ops_transformer')
    group.addoption('--reference-threads', type=int, default=1)
    perf = parser.getgroup('qfa_mxfp4 performance')
    perf.addoption('--run-perf', action='store_true', help='Explicitly enable performance tests')
    perf.addoption('--perf-case-set', choices=('document', 'mxfp8', 'bench16k'), default='document',
                   help='Document small/full scope; MXFP8 and fixed bench16k cases ignore scope')
    perf.addoption('--perf-kind', choices=('all', 'prefill', 'decode', 'mixed'), default='all')
    perf.addoption('--perf-backends', default='c4,c8,bf16')
    perf.addoption('--perf-mode', choices=('graph', 'eager', 'msprof'), default='graph')
    perf.addoption('--perf-warmup', type=int, default=3)
    perf.addoption('--perf-rounds', type=int, default=5)
    perf.addoption('--perf-repeats', type=int, default=20)
    perf.addoption('--perf-output', default='perf_output')
    perf.addoption('--perf-shard-index', type=int, default=0)
    perf.addoption('--perf-shard-count', type=int, default=1)
    perf.addoption('--msprof', action='store_true', help='Collect paired eager kernel timings with msprof')
    perf.addoption('--parse-prof', default=None, help='Parse an existing msprof directory against perf-output records')
    perf.addoption('--perf-baseline', default=None, help='Compare against a compatible summary.json')
    perf.addoption('--perf-threshold', type=float, default=8.0)


def pytest_configure(config):
    if config.getoption('--msprof') or config.getoption('--parse-prof'):
        from .common.perf_profiler import run_or_parse
        try:
            code = run_or_parse(config)
        except (ValueError, RuntimeError) as error:
            raise pytest.UsageError(str(error)) from error
        pytest.exit('msprof collection and validation complete', returncode=code)


def pytest_generate_tests(metafunc):
    if 'perf_params' in metafunc.fixturenames:
        from .quant_flash_attn_paramset_perf_rdv import get_cases, shard_cases
        config = metafunc.config
        try:
            schema, _ = resolve_quantization(config.getoption('--quant-schema'), config.getoption('--p-quant-scheme'))
            cases = shard_cases(get_cases(config.getoption('--scope'), config.getoption('--perf-kind'),
                                         config.getoption('--perf-case-set')),
                                config.getoption('--perf-shard-index'), config.getoption('--perf-shard-count'))
        except (ValueError, OSError) as error:
            raise pytest.UsageError(str(error)) from error
        metafunc.parametrize('perf_params', [pytest.param(dict(case, quant_schema=schema), id=case['name'],
            marks=pytest.mark.skip(reason=case.get('skip_reason') or 'Requires --run-perf')
                if case.get('skip_reason') or not config.getoption('--run-perf') else ())
                                             for case in cases])
        return
    if 'params' not in metafunc.fixturenames:
        return
    try:
        schema, scheme = resolve_quantization(metafunc.config.getoption('--quant-schema'),
                                             metafunc.config.getoption('--p-quant-scheme'))
        paramset = metafunc.module.paramset
        # Debug retains its original matrix; the functional suite owns case-set selection.
        if hasattr(paramset, 'CASE_SETS'):
            selected = paramset.get_cases(
                metafunc.config.getoption('--scope'), metafunc.config.getoption('--accuracy-case-set'))
        else:
            selected = paramset.get_cases(metafunc.config.getoption('--scope'))
    except (ValueError, OSError) as error:
        raise pytest.UsageError(str(error)) from error
    cases = [dict(case, quant_schema=schema, p_quant_scheme=scheme,
                  reference_backend=metafunc.config.getoption('--reference-backend'))
             for case in selected]
    skipped = metafunc.module.paramset.SKIP_CASES
    metafunc.parametrize('params', [pytest.param(case, id=case['name'],
        marks=pytest.mark.skip(reason='in SKIP_CASES') if case['name'] in skipped else ()) for case in cases])


@pytest.fixture
def golden_mode(request):
    return _parse_golden_mode(request.config.getoption('--golden-mode'))


@pytest.fixture
def cache_dir(request):
    return request.config.getoption('--cache-dir')


@pytest.fixture
def execution_options(request):
    threads = request.config.getoption('--reference-threads')
    device = request.config.getoption('--device-id')
    if threads < 1 or device < 0:
        raise pytest.UsageError('reference-threads must be positive; device-id must be nonnegative')
    return dict(package=request.config.getoption('--package'), device_id=device, reference_threads=threads,
                triton_reference=request.config.getoption('--triton-reference'),
                fp64_reference=request.config.getoption('--fp64-reference'))


@pytest.fixture(scope='session')
def perf_options(request):
    from pathlib import Path
    from .quant_flash_attn_paramset_perf_rdv import BACKENDS
    config = request.config
    backends = config.getoption('--perf-backends').split(',')
    if not backends or len(set(backends)) != len(backends) or any(b not in BACKENDS for b in backends):
        raise pytest.UsageError('perf-backends must select unique c4,c8,bf16 backends')
    values = {name: config.getoption('--perf-'+name) for name in ('warmup', 'rounds', 'repeats')}
    if min(values.values()) <= 0 or config.getoption('--device-id') < 0:
        raise pytest.UsageError('Performance counts must be positive; device-id must be nonnegative')
    return dict(values, backends=backends, mode=config.getoption('--perf-mode'),
                case_set=config.getoption('--perf-case-set'),
                output_dir=str(Path(config.getoption('--perf-output')).resolve()),
                package=config.getoption('--package'), device_id=config.getoption('--device-id'))
