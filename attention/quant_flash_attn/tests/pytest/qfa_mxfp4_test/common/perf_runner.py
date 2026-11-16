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

"""Sequential C4/C8/BF16 latency measurement with metadata outside timing."""
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import traceback

from ..quant_flash_attn_paramset_perf_rdv import case_fingerprint, sequence_lengths
from .perf_inputs import generate_inputs, prepare_inputs


def prepare_call(data, case, backend):
    import torch
    import torch_npu
    b, q, kv, d, h, hk = (case[k] for k in ('B', 'Q', 'KV', 'D', 'N_q', 'N_kv'))
    cu, _, kv_lengths = sequence_lengths(case)
    scale = case.get('softmax_scale') or d**-.5
    if backend == 'bf16':
        def call():
            return torch_npu.npu_fused_infer_attention_score_v2(
                data['q'], data['k'], data['v'], atten_mask=data['mask'],
                actual_seq_qlen=cu[1:], actual_seq_kvlen=kv_lengths,
                block_table=data['table'], block_size=case['page'],
                num_query_heads=h, num_key_value_heads=hk, softmax_scale=scale,
                input_layout='TND', sparse_mode=case['mask_mode'], return_softmax_lse=case['enable_lse'])
        return call, None
    ops = torch.ops.cann_ops_transformer
    mode = 5 if backend == 'c4' else 1
    common = dict(cu_seqlens_q=data['cu'], seqused_q=data['used_q'], seqused_kv=data['used_kv'],
                  max_seqlen_q=case.get('max_seqlen_q', q), max_seqlen_kv=case.get('max_seqlen_kv', kv),
                  mask_mode=case['mask_mode'], layout_q='TND',
                  layout_q_descale=case['q_scale_layout'], layout_kv='PA_BBND', layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(h, hk, d, mode, **common)
    pscale = torch.tensor([case.get('c8_p_scale', 448.)], dtype=torch.float32, device=data['q'].device) if mode == 1 else None
    def call():
        return ops.quant_flash_attn(data['q'], data['k'], data['v'], data['qs'], data['ks'], data['vs'],
            mode, metadata=metadata, block_table=data['table'], attn_mask=data['mask'], p_scale=pscale,
            softmax_scale=scale, return_softmax_lse=case['enable_lse'], quant_schema=case['quant_schema'], **common)
    return call, metadata


def environment(package):
    import torch
    import torch_npu
    module = importlib.import_module(package)
    vendor = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    libs = []
    for root in vendor.split(':'):
        candidate = Path(root)/'op_api/lib/libcust_opapi.so'
        if root and candidate.is_file():
            libs.append(dict(path=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest()))
    versions = {}
    for name in ('triton', 'triton-ascend', 'pytest'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    suite = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(suite)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(suite.rglob('*')) if path.is_file() and path.suffix in ('.py', '.json') and '__pycache__' not in path.parts}
    return dict(hostname=platform.node(), python=platform.python_version(), torch=torch.__version__,
                torch_npu=torch_npu.__version__, package=package, package_file=module.__file__,
                cann=os.environ.get('ASCEND_HOME_PATH'), custom_opp=vendor, opapi=libs, versions=versions,
                test_source_sha256=sources)


def _measure_backend(source, case, backend, options):
    import torch
    device = f"npu:{options['device_id']}"
    data = prepare_inputs(source, case, backend, device)
    input_shapes = {name: list(tensor.shape) for name, tensor in data.items() if tensor is not None}
    call, metadata = prepare_call(data, case, backend)
    header = metadata.cpu().reshape(-1).tolist()[:16] if metadata is not None else None
    graph = None
    try:
        for _ in range(options['warmup']):
            output = call()
        torch.npu.synchronize()
        expected = (sequence_lengths(case)[0][-1], case['N_q'], case['D'])
        if tuple(output[0].shape) != expected or output[0].dtype != torch.bfloat16:
            raise AssertionError(f'{backend}: unexpected output {output[0].shape}/{output[0].dtype}')
        if not bool(torch.isfinite(output[0]).all()):
            raise AssertionError(f'{backend}: nonfinite output')
        if case['enable_lse'] and (output[1] is None or not bool(torch.isfinite(output[1]).all())):
            raise AssertionError(f'{backend}: missing/nonfinite LSE')
        eager_sample = output[0].reshape(-1)[:128].clone()
        if options['mode'] == 'graph':
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                # Submit a whole sampling round with one replay. Python loops
                # between individual replays otherwise dominate short kernels.
                for _ in range(options['repeats']):
                    output = call()
            invoke = graph.replay
            for _ in range(options['warmup']):
                invoke()
            torch.npu.synchronize()
            if not torch.equal(eager_sample, output[0].reshape(-1)[:128]):
                raise AssertionError(f'{backend}: graph replay differs from eager output sample')
        else:
            invoke = call
        if options['mode'] == 'msprof':
            for _ in range(options['rounds']*options['repeats']):
                output = invoke()
            torch.npu.synchronize()
            return dict(profile_invocations=options['warmup']+options['rounds']*options['repeats'], input_shapes=input_shapes,
                        metadata_header=header, finite=True, output_shape=list(expected),
                        peak_allocated_bytes=torch.npu.max_memory_allocated())
        start = torch.npu.Event(enable_timing=True)
        stop = torch.npu.Event(enable_timing=True)
        samples = []
        for _ in range(options['rounds']):
            start.record()
            for _ in range(1 if graph is not None else options['repeats']):
                invoke()
            stop.record()
            stop.synchronize()
            us = start.elapsed_time(stop)*1000/options['repeats']
            if not math.isfinite(us) or us <= 0:
                raise AssertionError(f'{backend}: invalid latency {us}')
            samples.append(us)
        median = statistics.median(samples)
        return dict(median_us=median, samples_us=samples, min_us=min(samples), max_us=max(samples), input_shapes=input_shapes,
                    spread_pct=(max(samples)-min(samples))/median*100,
                    metadata_header=header, finite=True, output_shape=list(expected),
                    captured_calls_per_graph=options['repeats'] if graph is not None else 0,
                    replay_sample_equal=options['mode'] == 'graph',
                    peak_allocated_bytes=torch.npu.max_memory_allocated())
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


def write_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def execute_performance(case, options):
    import torch
    import torch_npu  # noqa: F401
    torch.set_num_threads(1)
    torch.npu.set_device(options['device_id'])
    provenance = environment(options['package'])
    record = dict(case=case, case_fingerprint=case_fingerprint(case), options=options,
                  environment=provenance, measurement_scope='attention_only',
                  metric={'graph':'graph_replay_us', 'eager':'eager_event_us', 'msprof':'msprof_pending'}[options['mode']],
                  started_unix=time.time(), status='running', backends={})
    path = Path(options['output_dir'])/f"{case['name']}_schema{case['quant_schema']}.json"
    write_record(path, record)
    try:
        source = generate_inputs(case)
        record['input_sha256'] = source['sha256']
        for backend in options['backends']:
            gc.collect()
            torch.npu.empty_cache()
            before = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            result = _measure_backend(source, case, backend, options)
            gc.collect()
            torch.npu.empty_cache()
            after = torch.npu.memory_allocated()
            result.update(allocated_before_bytes=before, allocated_after_release_bytes=after)
            record['backends'][backend] = result
            write_record(path, record)
            value = f"{result['median_us']:.3f} us" if 'median_us' in result else 'awaiting msprof export'
            print(f"PERF {case['name']} {backend} {value} released={after}", flush=True)
        record['status'] = 'passed'
        if Path('/proc/self/maps').exists():
            record['loaded_operator_libraries'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if any(token in line for token in ('libcust_opapi', 'libopapi.so', 'libnnopbase', 'libtorch_npu'))})
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
        record['traceback'] = traceback.format_exc()
        raise
    finally:
        record['finished_unix'] = time.time()
        write_record(path, record)
    return record
