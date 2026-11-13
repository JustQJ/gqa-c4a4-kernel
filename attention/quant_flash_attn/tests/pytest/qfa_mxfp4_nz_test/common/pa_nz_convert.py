#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""PA_BBND -> PA_NZ physical layout conversion for the C4A4 MXFP4 suite.

The golden/CPU/Triton references decode logical KV values from the BBND
arrangement, which is layout-agnostic. This module rewrites the same logical
bytes into the PA_NZ physical arrangement consumed by the public operator:

  payload    BBND [P, Bs, N, D/2]  ->  NZ [P, N, D/64, Bs, 32B]
  k_descale  BBND [P, Bs, N, D/64, 2] -> NZ [P, N, Bs/16, D/64, 16, 2]
  v_descale  BBND [P, Bs/64, N, D, 2] -> NZ [P, N, D/16, Bs/64, 16, 2]

Every conversion is a pure permutation of the original bytes: the decoded
logical values are bit-identical, so reference outputs computed on the BBND
arrangement remain valid for the PA_NZ launch.
"""
import torch


def bbnd_to_nz_payload(kv):
    """[P, Bs, N, D/2] -> [P, N, D/64, Bs, 32]. D/2 is a multiple of 32 for D=128/256."""
    pages, block, heads, row_bytes = kv.shape
    if row_bytes % 32:
        raise ValueError('PA_NZ payload requires row bytes to be 32B aligned')
    d1 = row_bytes // 32
    # [P, Bs, N, D1, 32] -> [P, N, D1, Bs, 32]
    return kv.reshape(pages, block, heads, d1, 32) \
              .permute(0, 2, 3, 1, 4).contiguous()


def bbnd_to_nz_k_scale(ks):
    """[P, Bs, N, D/64, 2] -> [P, N, Bs/16, D/64, 16, 2] (uint8 elements)."""
    pages, block, heads, pairs, two = ks.shape
    if block % 16 or two != 2:
        raise ValueError('PA_NZ k_descale requires block%16==0 and trailing dim 2')
    groups, pair_count = block // 16, pairs
    # [P, G, 16, N, PC, 2] -> [P, N, G, PC, 16, 2]
    return ks.reshape(pages, groups, 16, heads, pair_count, 2) \
             .permute(0, 3, 1, 4, 2, 5).contiguous()


def bbnd_to_nz_v_scale(vs):
    """[P, Bs/64, N, D, 2] -> [P, N, D/16, Bs/64, 16, 2] (uint8 elements).

    BBND semantics: vs[p, s64, n, d, b2] where (s64, b2) enumerates the
    per-32-token scale groups (s64*2+b2) inside each 64-token window.
    NZ rearrangement: contiguous 32B = 16 d channels x 2 group bytes for
    a fixed (page, head, channel-group dv16, token window s64).
    """
    pages, groups, heads, dims, two = vs.shape
    if two != 2 or dims % 16:
        raise ValueError('PA_NZ v_descale requires trailing dim 2 and D%16==0')
    dv_groups = dims // 16
    # [P, S64, N, DV16, 16, 2] -> [P, N, DV16, S64, 16, 2]
    return vs.reshape(pages, groups, heads, dv_groups, 16, 2) \
             .permute(0, 2, 3, 1, 4, 5).contiguous()


def convert_data_bbnd_to_nz(data):
    """Return a new dict whose KV tensors are rearranged into PA_NZ physical layout.

    Q tensors and the block table are shared: PA_NZ only changes the KV
    arrangement inside each physical page, not the page addressing itself.
    """
    return dict(
        data,
        k_fp4=bbnd_to_nz_payload(data['k_fp4']),
        v_fp4=bbnd_to_nz_payload(data['v_fp4']),
        dequant_scale_k=bbnd_to_nz_k_scale(data['dequant_scale_k']),
        v_descale=bbnd_to_nz_v_scale(data['v_descale']),
    )
