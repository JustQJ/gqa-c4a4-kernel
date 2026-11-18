#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.

"""C4 PA_BNBD/PA_NZ accuracy with the unchanged BBND runner and thresholds."""
import concurrent.futures
import pytest
from . import quant_flash_attn_paramset_pa_layout as paramset


@pytest.mark.pa_layout
def test_quant_flash_attn_mxfp4_pa_layout(params, golden_mode, cache_dir, execution_options):
    from .common import test_runner
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result, lse = executor.submit(test_runner.execute_test, params, golden_mode, cache_dir,
                                      **execution_options).result()
    test_runner.check_results(result, lse)
