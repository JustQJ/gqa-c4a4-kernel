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

"""Small shapes: both layouts/masks, LSE on/off, empty and padded Q rows."""
from .quant_flash_attn_paramset_common import build_paramset, expand_paramset_to_cases

SCENARIOS = [
    dict(name='smoke', qlens=[17], klens=[257], page=256),
    dict(name='empty_and_q_padding', qlens=[5, 0, 3], used_q=[3, 0, 3], klens=[0, 33, 1], page=128),
    dict(name='page_tail', qlens=[3, 9], klens=[129, 513], page=512),
]
TEST_PARAMS = build_paramset('full', SCENARIOS, (False, True))
CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()


def get_cases(scope='full'):
    return CASES if scope == 'full' else expand_paramset_to_cases(build_paramset(scope, SCENARIOS, (False, True)))
