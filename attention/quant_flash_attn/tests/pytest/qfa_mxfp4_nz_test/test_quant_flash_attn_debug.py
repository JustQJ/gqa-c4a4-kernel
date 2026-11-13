#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""PA_NZ public C4 accuracy entry; same cases as the BBND debug suite."""
import concurrent.futures
import pytest
from qfa_mxfp4_test import quant_flash_attn_paramset_debug as paramset


@pytest.mark.debug
def test_quant_flash_attn_mxfp4_nz(params, golden_mode, cache_dir, execution_options):
    from .common import pa_nz_test_runner
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result, lse = executor.submit(pa_nz_test_runner.execute_test, params, golden_mode, cache_dir,
                                      **execution_options).result()
    pa_nz_test_runner.check_results(result, lse)
