#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""PA_NZ launch wrapper around the BBND golden module.

gen stage: generate logical (BBND-arranged) inputs with the original golden,
           then derive the PA_NZ physical tensors for the NPU launch.
cpu stage: unchanged; references decode the logical BBND arrangement.
npu stage: convert cached BBND inputs to PA_NZ, call the public operator with
           layout_kv='PA_NZ'.
"""
import importlib
import sys
from pathlib import Path
import torch

_PYTEST_DIR = str(Path(__file__).resolve().parents[3])
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)
from qfa_mxfp4_test.common import quant_flash_attn_golden as bbnd_golden  # noqa: E402
from qfa_mxfp4_test.common import golden_cache  # noqa: E402
from .pa_nz_convert import convert_data_bbnd_to_nz  # noqa: E402


def prepare_npu_inputs_nz(data, device):
    prepared = bbnd_golden.prepare_npu_inputs(convert_data_bbnd_to_nz(data), device)
    return prepared


def invoke_operator_nz(ops, data, params):
    from qfa_mxfp4_test.quant_flash_attn_paramset_common import normalize_params
    p = normalize_params(params)
    common = dict(cu_seqlens_q=data['cu_seqlens_q'], seqused_q=data['seqused_q'],
                  seqused_kv=data['seqused_kv'], max_seqlen_q=p['max_seqlen_q'],
                  max_seqlen_kv=p['max_seqlen_kv'], mask_mode=p['mask_mode'],
                  layout_q='TND', layout_q_descale=p['q_scale_layout'],
                  layout_kv='PA_NZ', layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(p['N_q'], p['N_kv'], p['D'], 5, **common)
    mask = None
    if p['mask_mode'] == 3:
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(data['q_fp4'].device)
    return ops.quant_flash_attn(
        *(data[name] for name in ('q_fp4', 'k_fp4', 'v_fp4', 'dequant_scale_q', 'dequant_scale_k', 'v_descale')),
        5, block_table=data['block_table_torch'], metadata=metadata, attn_mask=mask,
        p_scale=None, softmax_scale=p['softmax_scale'], return_softmax_lse=p['enable_lse'],
        quant_schema=p['quant_schema'], **common)


def npu_mxfp4_fa_nz(data, params, package='cann_ops_transformer', device_id=0):
    import torch_npu  # noqa: F401 -- registers the NPU device and packed dtypes
    importlib.import_module(package)
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)
    prepared = prepare_npu_inputs_nz(data, device)
    output, lse = invoke_operator_nz(torch.ops.cann_ops_transformer, prepared, params)
    torch.npu.synchronize()
    if output.device != prepared['q_fp4'].device or lse.device != output.device:
        raise AssertionError('Both API outputs must be on the input NPU')
    return output.cpu(), lse.cpu()


# Reuse the original module surface for the shared runner contract.
generate_data = bbnd_golden.generate_data
decode_inputs = bbnd_golden.decode_inputs
cpu_mxfp4_golden = bbnd_golden.cpu_mxfp4_golden
save_input = golden_cache.save_input
