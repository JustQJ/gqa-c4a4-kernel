#!/usr/bin/env python3
"""Controlled probe: K/V only token X nonzero -> output reveals GM->L1 mapping.

Launches PA_BBND and PA_NZ in separate processes (avoid the same-process
crash); each worker writes its output to a file.
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qfa_mxfp4_test.common.quant_flash_attn_golden import (  # noqa: E402
    generate_data, prepare_npu_inputs)
from qfa_mxfp4_nz_test.common.pa_nz_convert import convert_data_bbnd_to_nz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--package', default='cann_ops_transformer')
    ap.add_argument('--layout', choices=('PA_BBND', 'PA_NZ'), required=True)
    ap.add_argument('--probe-token', type=int, default=3)
    ap.add_argument('--q-heads', type=int, default=1)
    ap.add_argument('--kv-heads', type=int, default=1)
    ap.add_argument('--q-tokens', type=int, default=1)
    ap.add_argument('--qscale-layout', choices=('TND', 'N2TGD'), default='TND')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    importlib.import_module(args.package)
    device = f'npu:{args.device}'
    torch.npu.set_device(args.device)

    params = dict(name='probe', B=1, N_q=args.q_heads, N_kv=args.kv_heads, D=128,
                  cu_seqlens_q=[0, args.q_tokens], seqused_q=None, seqused_kv=[257],
                  max_seqlen_q=args.q_tokens, max_seqlen_kv=257, block_size=256,
                  mask_mode=0, q_scale_layout=args.qscale_layout, enable_lse=True,
                  quant_schema=2)
    data = generate_data(params)
    # generate_data 会为 TND 布局生成 seqused_q=None；保持原值
    # Controlled: only probe-token K nonzero (fixed known codes), V = all 1.0 (all heads)
    k = torch.zeros_like(data['k_fp4'])
    # fp4 code 0x22 = two 1.0 values — all KV heads
    kvh = data['k_fp4'].shape[2]
    for h in range(kvh):
        k[0, args.probe_token, h, :] = 0x22
    data['k_fp4'] = k
    v = torch.zeros_like(data['v_fp4'])
    for h in range(kvh):
        v[0, :, h, :] = 0x22
    data['v_fp4'] = v
    # descale: 保留 generate_data 的随机值（真实场景），K/V payload 已受控
    # 如需均匀 descale 可取消下面两行注释
    # data['dequant_scale_k'] = torch.full_like(data['dequant_scale_k'], 127)
    # data['v_descale'] = torch.full_like(data['v_descale'], 127)

    prepared = prepare_npu_inputs(
        convert_data_bbnd_to_nz(data) if args.layout == 'PA_NZ' else data, device)
    ops = torch.ops.cann_ops_transformer
    seqused_q_dev = None if data['seqused_q'] is None else prepared['seqused_q']
    common = dict(cu_seqlens_q=prepared['cu_seqlens_q'], seqused_q=seqused_q_dev,
                  seqused_kv=prepared['seqused_kv'], max_seqlen_q=args.q_tokens,
                  max_seqlen_kv=257,
                  mask_mode=0, layout_q='TND', layout_q_descale=args.qscale_layout,
                  layout_kv=args.layout, layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(args.q_heads, args.kv_heads, 128, 5, **common)
    out, lse = ops.quant_flash_attn(
        prepared['q_fp4'], prepared['k_fp4'], prepared['v_fp4'],
        prepared['dequant_scale_q'], prepared['dequant_scale_k'], prepared['v_descale'],
        5, block_table=prepared['block_table_torch'], metadata=metadata, attn_mask=None,
        p_scale=None, softmax_scale=128 ** -0.5, return_softmax_lse=True,
        quant_schema=2, **common)
    torch.npu.synchronize()
    torch.save({'out': out.cpu(), 'lse': lse.cpu(), 'probe_token': args.probe_token},
               args.output)
    print(f"OK {args.layout}: out={out.cpu().flatten()[:4].tolist()} lse={lse.cpu().flatten().tolist()}")


if __name__ == '__main__':
    main()
