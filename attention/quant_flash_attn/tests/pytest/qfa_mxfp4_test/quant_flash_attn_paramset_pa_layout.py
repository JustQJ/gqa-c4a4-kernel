#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.

"""The established BBND debug matrix, changing only the KV physical layout."""
from .quant_flash_attn_paramset_common import build_paramset, expand_paramset_to_cases
from .quant_flash_attn_paramset_debug import SCENARIOS

KV_LAYOUTS = ('PA_BNBD', 'PA_NZ')
TEST_PARAMS = build_paramset('full', SCENARIOS, (False, True), kv_layouts=KV_LAYOUTS)
CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()


def get_cases(scope='full'):
    if scope == 'full':
        return CASES
    return expand_paramset_to_cases(build_paramset(scope, SCENARIOS, (False, True), kv_layouts=KV_LAYOUTS))
