#!/usr/bin/env python3
"""Full cross-layout precision comparison for C4A4 PA_BBND / PA_NZ / PA_BNBD
and their C8A8 counterparts.

For each test case, compares:
  C4A4 layouts:
    - BBND vs NZ         (PA_NZ regression check)
    - BBND vs BNBD       (PA_BNBD regression check)
    - NZ vs BNBD         (cross-check between new layouts)
    - each vs golden     (absolute accuracy)
  C8A8 layouts:
    - c8_bbnd vs c8_nz   (C8 cross-layout)
    - c8_bbnd vs c8_bnbd (C8 cross-layout)
    - c8_nz vs c8_bnbd   (C8 cross-layout)
    - each vs golden
  C4 vs C8 (same layout):
    - c4_bbnd vs c8_bbnd  (P quantization isolation)
    - c4_nz vs c8_nz
    - c4_bnbd vs c8_bnbd

Usage: python3 compare_all_layouts.py --data <data_dir> [--report report.json]
"""
import argparse
import json
import os
import sys
import numpy as np


def bf16_to_float(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def metrics(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-6)
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12))
    close = np.isclose(a, b, atol=1e-4, rtol=0.0078125)
    return dict(
        max_abs=float(diff.max()) if diff.size else 0.0,
        mean_abs=float(diff.mean()) if diff.size else 0.0,
        relative_l2=rel_l2,
        max_rel=float((diff / denom).max()) if diff.size else 0.0,
        isclose_ratio=float(close.mean()) if diff.size else 1.0,
        bit_exact=bool(np.array_equal(actual, reference)),
        num_elements=int(diff.size),
    )


def safe_load(bin_path, dtype=np.uint16):
    if not os.path.exists(bin_path):
        return None
    return bf16_to_float(np.fromfile(bin_path, dtype=dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()

    with open(os.path.join(args.data, 'index.json')) as f:
        cases = json.load(f)

    # Define all output file prefixes and their display names
    c4_layouts = {
        'bbnd': 'out_bbnd_out.bin',
        'nz': 'out_nz_out.bin',
        'bnbd': 'out_bnbd_out.bin',
    }
    c8_layouts = {
        'c8_bbnd': 'out_c8_bbnd_out.bin',
        'c8_nz': 'out_c8_nz_out.bin',
        'c8_bnbd': 'out_c8_bnbd_out.bin',
    }

    records = []
    all_pass = True

    for name in cases:
        cdir = os.path.join(args.data, name)
        golden = np.load(os.path.join(cdir, 'golden.npz'))['out'].reshape(-1)

        # Load C4 outputs
        c4 = {}
        for layout, fname in c4_layouts.items():
            data = safe_load(os.path.join(cdir, fname))
            if data is not None:
                c4[layout] = data

        # Load C8 outputs
        c8 = {}
        for layout, fname in c8_layouts.items():
            data = safe_load(os.path.join(cdir, fname))
            if data is not None:
                c8[layout] = data

        record = dict(name=name)

        # --- C4 cross-layout comparisons (vs BBND baseline) ---
        if 'bbnd' in c4 and 'nz' in c4:
            m = metrics(c4['nz'], c4['bbnd'])
            record['c4_nz_vs_c4_bbnd'] = m
        if 'bbnd' in c4 and 'bnbd' in c4:
            m = metrics(c4['bnbd'], c4['bbnd'])
            record['c4_bnbd_vs_c4_bbnd'] = m
        if 'nz' in c4 and 'bnbd' in c4:
            m = metrics(c4['bnbd'], c4['nz'])
            record['c4_bnbd_vs_c4_nz'] = m

        # --- C4 vs golden ---
        for layout in c4:
            record[f'c4_{layout}_vs_golden'] = metrics(c4[layout], golden)

        # --- C8 cross-layout comparisons ---
        c8_keys = list(c8.keys())
        for i in range(len(c8_keys)):
            for j in range(i + 1, len(c8_keys)):
                m = metrics(c8[c8_keys[j]], c8[c8_keys[i]])
                record[f'{c8_keys[j]}_vs_{c8_keys[i]}'] = m

        # --- C8 vs golden ---
        for layout in c8:
            record[f'{layout}_vs_golden'] = metrics(c8[layout], golden)

        # --- C4 vs C8 (same layout) ---
        layout_pairs = [('bbnd', 'c8_bbnd'), ('nz', 'c8_nz'), ('bnbd', 'c8_bnbd')]
        for c4l, c8l in layout_pairs:
            if c4l in c4 and c8l in c8:
                m = metrics(c4[c4l], c8[c8l])
                record[f'c4_{c4l}_vs_{c8l}'] = m

        # --- Pass/fail determination ---
        # C4 layouts must be bit-exact or within 1% rel_l2 of BBND
        case_pass = True
        for key in ['c4_nz_vs_c4_bbnd', 'c4_bnbd_vs_c4_bbnd']:
            if key in record:
                m = record[key]
                if m['relative_l2'] > 0.01:
                    case_pass = False
                    all_pass = False

        record['passed'] = case_pass

        # Print summary line
        nz_m = record.get('c4_nz_vs_c4_bbnd', {})
        bnbd_m = record.get('c4_bnbd_vs_c4_bbnd', {})
        nz_be = nz_m.get('bit_exact', '?')
        bnbd_be = bnbd_m.get('bit_exact', '?')
        nz_rl2 = nz_m.get('relative_l2', -1)
        bnbd_rl2 = bnbd_m.get('relative_l2', -1)

        status = 'PASS' if case_pass else 'FAIL'
        print(f"{'✓' if case_pass else '✗'} {status} {name}: "
              f"NZ bit_exact={nz_be} rel_l2={nz_rl2:.3e} | "
              f"BNBD bit_exact={bnbd_be} rel_l2={bnbd_rl2:.3e}",
              flush=True)

        records.append(record)

    # Summary
    n_pass = sum(1 for r in records if r['passed'])
    n_total = len(records)
    summary = dict(total=n_total, passed=n_pass, failed=n_total - n_pass,
                   all_pass=all_pass)

    report = dict(summary=summary, cases=records)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        with open(args.report, 'w') as f:
            f.write(text + '\n')

    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_pass}/{n_total} passed, {n_total - n_pass} failed")
    print(f"{'='*70}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
