#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""PA_NZ accuracy suite for the C4A4 MXFP4 operator.

Same case matrix, references and thresholds as qfa_mxfp4_test; only the
physical KV arrangement handed to the public operator changes to PA_NZ.
The logical KV bytes are generated once (BBND arrangement), then rewritten
into PA_NZ with common.pa_nz_convert; CPU/Triton references decode from the
logical BBND arrangement, which is bit-identical to the PA_NZ payload.
"""
