#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Accuracy options for the PA_NZ suite; same semantics as the BBND suite."""
import os
import sys
from pathlib import Path
import pytest

# Cross-suite imports: make qfa_mxfp4_test importable as a top-level package.
_PYTEST_DIR = str(Path(__file__).resolve().parents[1])
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)
from qfa_mxfp4_test.quant_flash_attn_paramset_common import P_QUANT_SCHEMES, resolve_quantization  # noqa: E402

_VALID_MODES = {'gen', 'cpu', 'npu', 'compare'}
collect_ignore = ['common/pa_nz_test_runner.py']


def _parse_golden_mode(raw):
    parts = {part.strip() for part in raw.split(',') if part.strip()}
    if not parts or parts - (_VALID_MODES | {'all'}):
        raise pytest.UsageError('golden-mode must be all or a combination of gen,cpu,npu,compare')
    return set(_VALID_MODES) if 'all' in parts else parts


def pytest_addoption(parser):
    group = parser.getgroup('qfa_mxfp4_nz accuracy')
    group.addoption('--golden-mode', default='all', help='all or comma-separated gen,cpu,npu,compare')
    group.addoption('--cache-dir', default=None, help='Input and reference .pt cache directory')
    group.addoption('--scope', choices=('full', 'small'), default='full')
    group.addoption('--quant-schema', type=int, choices=(0, 1, 2), default=None,
                    help='Public API and references: 0=OCP, 1=CX, 2=CX Delta (default)')
    group.addoption('--p-quant-scheme', choices=P_QUANT_SCHEMES, default=None,
                    help='Named alias for quant-schema; must agree if both options are supplied')
    group.addoption('--reference-backend', choices=('global', 'kernel'), default='kernel',
                    help='Actual-metadata/16-task kernel model (default) or preserved global CPU golden')
    group.addoption('--triton-reference', action='store_true')
    group.addoption('--fp64-reference', action='store_true')
    group.addoption('--device-id', type=int, default=0)
    group.addoption('--package', default=os.environ.get('QFA_PACKAGE', 'cann_ops_transformer'))
    group.addoption('--reference-threads', type=int, default=1)


def pytest_generate_tests(metafunc):
    if 'params' not in metafunc.fixturenames:
        return
    try:
        schema, scheme = resolve_quantization(metafunc.config.getoption('--quant-schema'),
                                             metafunc.config.getoption('--p-quant-scheme'))
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    cases = [dict(case, quant_schema=schema, p_quant_scheme=scheme,
                  reference_backend=metafunc.config.getoption('--reference-backend'))
             for case in metafunc.module.paramset.get_cases(metafunc.config.getoption('--scope'))]
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
