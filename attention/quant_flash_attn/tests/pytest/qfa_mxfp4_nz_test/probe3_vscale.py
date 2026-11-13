#!/usr/bin/env python3
"""V-descale byte-position probe.

V payload all-1.0, K zero except token 0 (score=0 -> weights uniform),
V descale all 127 except ONE selected byte; the output directly reveals
which (token-group, channel) the kernel actually read from that byte.

NZ GM layout: [Bn, N, Dv/16, Bs/64, 16, 2]; pick byte at (dv16=X, s64=0,
j16=Y, b2=Z). If the kernel reads it as (32token-group g, channel c) then
output[..., c] shows a factor of 2^(scale-127) applied to tokens 32g..32g+31.
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
    ap.add_argument('--nz-byte', type=int, default=-1,
                    help='NZ GM byte index within the page/head block to set to 130; -1 = all 127')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    importlib.import_module(args.package)
    device = f'npu:{args.device}'
    torch.npu.set_device(args.device)

    params = dict(name='probe3', B=1, N_q=1, N_kv=1, D=128,
                  cu_seqlens_q=[0, 1], seqused_q=None, seqused_kv=[128],
                  max_seqlen_q=1, max_seqlen_kv=128, block_size=128,
                  mask_mode=0, q_scale_layout='TND', enable_lse=True,
                  quant_schema=2)
    data = generate_data(params)
    kvlen, page, dimension = 128, 128, 128
    # K: only token 0 nonzero -> uniform-ish weights but scores differ by descale;
    # simpler: K all zero payload -> all scores 0 -> uniform weights 1/kvlen
    k = torch.zeros_like(data['k_fp4'])
    data['k_fp4'] = k
    data['dequant_scale_k'] = torch.full_like(data['dequant_scale_k'], 127)
    # V: all 1.0
    v = torch.zeros_like(data['v_fp4'])
    v[0, :, 0, :] = 0x22
    data['v_fp4'] = v
    # V descale: all 127 (=1.0) then optionally poke one byte in the NZ arrangement
    data['v_descale'] = torch.full_like(data['v_descale'], 127)
    if args.nz_byte >= 0 and args.layout == 'PA_NZ':
        # NZ v_descale GM: [P, N=1, Dv/16=8, Bs/64=1, 16, 2] -> byte index within page+head
        # convert BBND->NZ then poke: find flat index: dv16*32 + s64*32*? ... total per page/head = 8*1*16*2=256B
        # flat = dv16*32 + s64*16*2*? ... per (dv16,s64) block=32B -> flat = dv16*32 + s64*32 + j16*2 + b2
        nz = convert_data_bbnd_to_nz(data)['v_descale']  # [P, N, dv16, s64, 16, 2]
        flat = nz.reshape(-1)
        assert args.nz_byte < flat.numel(), f'byte index {args.nz_byte} >= {flat.numel()}'
        flat[args.nz_byte] = 130  # scale 8x
        nz2 = flat.reshape(nz.shape)
        # poke back through inverse: emulate by converting a marked BBND instead
        # simpler: keep NZ tensor and replace data['v_descale'] path manually below
        data['_nz_v_descale_override'] = nz2

    # For PA_NZ we must build the converted dict with the poked descale
    if args.layout == 'PA_NZ':
        converted = convert_data_bbnd_to_nz(data)
        if args.nz_byte >= 0:
            nz2 = data['_nz_v_descale_override']
            converted['v_descale'] = nz2
        prepared = prepare_npu_inputs(converted, device)
    else:
        # BBND probe: poke the equivalent BBND byte: [P, s64, N, D, 2]
        if args.nz_byte >= 0:
            vd = data['v_descale']
            # NZ flat -> (dv16, s64, j16, b2); BBND byte -> (s64, dv16*16+j16, b2)
            dv16, r = divmod(args.nz_byte, 32)
            s64, r2 = divmod(r, 32)
            j16, b2 = divmod(r2, 2)
            vd = vd.clone()
            vd[0, s64, 0, dv16 * 16 + j16, b2] = 130
            data['v_descale'] = vd
        prepared = prepare_npu_inputs(data, device)

    ops = torch.ops.cann_ops_transformer
    seqused_q_dev = None if data['seqused_q'] is None else prepared['seqused_q']
    common = dict(cu_seqlens_q=prepared['cu_seqlens_q'], seqused_q=seqused_q_dev,
                  seqused_kv=prepared['seqused_kv'], max_seqlen_q=1, max_seqlen_kv=128,
                  mask_mode=0, layout_q='TND', layout_q_descale='TND',
                  layout_kv=args.layout, layout_out='TND')
    metadata = ops.quant_flash_attn_metadata(1, 1, 128, 5, **common)
    out, lse = ops.quant_flash_attn(
        prepared['q_fp4'], prepared['k_fp4'], prepared['v_fp4'],
        prepared['dequant_scale_q'], prepared['dequant_scale_k'], prepared['v_descale'],
        5, block_table=prepared['block_table_torch'], metadata=metadata, attn_mask=None,
        p_scale=None, softmax_scale=128 ** -0.5, return_softmax_lse=True,
        quant_schema=2, **common)
    torch.npu.synchronize()
    torch.save({'out': out.cpu(), 'lse': lse.cpu()}, args.output)
    o = out.cpu().float().flatten()
    # uniform weights 1/64 * V(=1*8x on poked channel group) -> out value reveals channel/token mapping
    print(f"OK {args.layout} byte={args.nz_byte}: out={o[:8].tolist()} "
          f"nonzero_pos={(o != o.round()).nonzero().flatten()[:8].tolist() if False else [i for i, x in enumerate(o.tolist()) if abs(x - 1/64) > 1e-6][:8]}")


if __name__ == '__main__':
    main()
