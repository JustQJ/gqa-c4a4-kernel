#!/usr/bin/env python3
"""C4A4 PA_BNBD vs C8A8 PA_BNBD precision comparison (identical Q/K/V values).

C8 inputs are a lossless FP4->FP8 conversion of the C4 payloads, so both
operators see numerically identical Q/K/V and only differ in P quantization
(C4 P=MXFP4 vs C8 P=FP8_E4M3) and the kernel's P handling.

Report per case:
  - c4_bnbd vs c8_bnbd direct output comparison
  - each vs the unquantized-P FP64 reference (absolute quantization loss)
  - c4_bbnd vs the same reference (the C4 baseline to beat)

Usage: python3 compare_c8_bnbd.py --data <data_dir> [--report c8_bnbd_report.json]
"""
import argparse
import json
import os
import numpy as np


def bf16_to_float(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def metrics(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-6)
    return dict(max_abs=float(diff.max()) if diff.size else 0.0,
                mean_abs=float(diff.mean()) if diff.size else 0.0,
                relative_l2=float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12)),
                max_rel=float((diff / denom).max()) if diff.size else 0.0,
                isclose_ratio=float(np.isclose(a, b, atol=1e-4, rtol=0.0078125).mean()) if diff.size else 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    with open(os.path.join(args.data, 'index.json')) as f:
        cases = json.load(f)
    records = []
    for name in cases:
        cdir = os.path.join(args.data, name)
        c4_bnbd = bf16_to_float(np.fromfile(os.path.join(cdir, 'out_bnbd_out.bin'), dtype=np.uint16))
        c4_bbnd = bf16_to_float(np.fromfile(os.path.join(cdir, 'out_bbnd_out.bin'), dtype=np.uint16))
        c8_bnbd = bf16_to_float(np.fromfile(os.path.join(cdir, 'out_c8_bnbd_out.bin'), dtype=np.uint16))
        golden = np.load(os.path.join(cdir, 'golden.npz'))['out'].reshape(-1)
        record = dict(name=name,
                      c4bnbd_vs_c8bnbd=metrics(c4_bnbd, c8_bnbd),
                      c4bnbd_vs_golden=metrics(c4_bnbd, golden),
                      c4bbnd_vs_golden=metrics(c4_bbnd, golden),
                      c8bnbd_vs_golden=metrics(c8_bnbd, golden))
        records.append(record)
        print(f"{name}: c4bnbd_vs_c8bnbd rel_l2={record['c4bnbd_vs_c8bnbd']['relative_l2']:.3e} | "
              f"vs golden rel_l2 c4bnbd={record['c4bnbd_vs_golden']['relative_l2']:.3e} "
              f"c4bbnd={record['c4bbnd_vs_golden']['relative_l2']:.3e} c8bnbd={record['c8bnbd_vs_golden']['relative_l2']:.3e}",
              flush=True)
    summary = dict(total=len(records))
    text = json.dumps(dict(summary=summary, cases=records), ensure_ascii=False, indent=2)
    if args.report:
        with open(args.report, 'w') as f:
            f.write(text + '\n')
    print(f"C8 BNBD COMPARISON DONE: {len(records)} cases")


if __name__ == '__main__':
    main()
