#!/usr/bin/env python3
"""Generate PA_BNBD test inputs from existing BBND data.

PA_BBND layout: [Bn, Bs, N, D]
PA_BNBD layout: [Bn, N, Bs, D]

This script converts BBND data to BNBD by transposing the N and Bs dimensions.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path


def parse_ini_config(ini_path):
    """Parse configuration from .ini file."""
    config = {}
    with open(ini_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key] = value
    return config


def parse_shape(shape_str):
    """Parse shape string like '5,256,1,256' into tuple."""
    return tuple(map(int, shape_str.split(',')))


def convert_bbnd_to_bnbd(k_bbnd, v_bbnd, ks_bbnd, vs_bbnd):
    """Convert PA_BBND layout to PA_BNBD layout.

    K/V payload: [Bn, Bs, N, D] -> [Bn, N, Bs, D]
    K scale: [Bn, Bs, N, D/64, 2] -> [Bn, N, Bs, D/64, 2]
    V scale: [Bn, Bs/64, N, D, 2] -> [Bn, N, Bs/64, D, 2]
    """
    # K/V payload: swap axes 1 (Bs) and 2 (N)
    k_bnbd = np.transpose(k_bbnd, (0, 2, 1, 3)).copy()
    v_bnbd = np.transpose(v_bbnd, (0, 2, 1, 3)).copy()

    # K scale: swap axes 1 (Bs) and 2 (N)
    # KS shape: [Bn, Bs, N, D/64, 2]
    ks_bnbd = np.transpose(ks_bbnd, (0, 2, 1, 3, 4)).copy()

    # V scale: swap axes 1 (Bs/64) and 2 (N)
    # VS shape: [Bn, Bs/64, N, D, 2]
    vs_bnbd = np.transpose(vs_bbnd, (0, 2, 1, 3, 4)).copy()

    return k_bnbd, v_bnbd, ks_bnbd, vs_bnbd


def process_test_case(case_dir):
    """Process a single test case directory."""
    case_path = Path(case_dir)

    # Read config from bbnd.ini
    bbnd_ini = case_path / 'bbnd.ini'
    if not bbnd_ini.exists():
        print(f"Warning: {bbnd_ini} not found, skipping")
        return False

    config = parse_ini_config(bbnd_ini)

    # Parse shapes (these are logical shapes, not packed)
    k_shape = parse_shape(config['K_SHAPE'])
    v_shape = parse_shape(config['V_SHAPE'])
    ks_shape = parse_shape(config['KS_SHAPE'])
    vs_shape = parse_shape(config['VS_SHAPE'])

    print(f"\nProcessing {case_path.name}:")
    print(f"  K: {k_shape} (logical), V: {v_shape} (logical)")
    print(f"  KS: {ks_shape}, VS: {vs_shape}")

    # K/V are FP4 packed: shape is [Bn, Bs, N, D] but stored as [Bn, Bs, N, D//2] bytes
    k_packed_shape = (*k_shape[:-1], k_shape[-1] // 2)
    v_packed_shape = (*v_shape[:-1], v_shape[-1] // 2)

    k_bbnd = np.fromfile(case_path / 'k.bin', dtype=np.uint8).reshape(k_packed_shape)
    v_bbnd = np.fromfile(case_path / 'v.bin', dtype=np.uint8).reshape(v_packed_shape)

    # Scales are E8M0 (1 byte per element)
    ks_bbnd = np.fromfile(case_path / 'ks.bin', dtype=np.uint8).reshape(ks_shape)
    vs_bbnd = np.fromfile(case_path / 'vs.bin', dtype=np.uint8).reshape(vs_shape)

    # Convert to BNBD
    k_bnbd, v_bnbd, ks_bnbd, vs_bnbd = convert_bbnd_to_bnbd(k_bbnd, v_bbnd, ks_bbnd, vs_bbnd)

    print(f"  Converted to BNBD:")
    print(f"    K: {k_bnbd.shape}, V: {v_bnbd.shape}")
    print(f"    KS: {ks_bnbd.shape}, VS: {vs_bnbd.shape}")

    # Save BNBD data
    k_bnbd.tofile(case_path / 'k_bnbd.bin')
    v_bnbd.tofile(case_path / 'v_bnbd.bin')
    ks_bnbd.tofile(case_path / 'ks_bnbd.bin')
    vs_bnbd.tofile(case_path / 'vs_bnbd.bin')

    # Create BNBD config file
    # Note: K_SHAPE and V_SHAPE should be LOGICAL shapes (with full D, not D//2)
    # The driver expects logical shapes, but the .bin files contain packed data
    k_logical_shape = (*k_bnbd.shape[:-1], k_bnbd.shape[-1] * 2)  # Convert D//2 back to D
    v_logical_shape = (*v_bnbd.shape[:-1], v_bnbd.shape[-1] * 2)  # Convert D//2 back to D

    bnbd_config = config.copy()
    bnbd_config['LAYOUT_KV'] = 'PA_BNBD'
    bnbd_config['K_SHAPE'] = ','.join(map(str, k_logical_shape))
    bnbd_config['V_SHAPE'] = ','.join(map(str, v_logical_shape))
    bnbd_config['KS_SHAPE'] = ','.join(map(str, ks_bnbd.shape))
    bnbd_config['VS_SHAPE'] = ','.join(map(str, vs_bnbd.shape))
    bnbd_config['K_BIN'] = 'k_bnbd.bin'
    bnbd_config['V_BIN'] = 'v_bnbd.bin'
    bnbd_config['KS_BIN'] = 'ks_bnbd.bin'
    bnbd_config['VS_BIN'] = 'vs_bnbd.bin'

    # Write bnbd.ini
    bnbd_ini = case_path / 'bnbd.ini'
    with open(bnbd_ini, 'w') as f:
        for key, value in bnbd_config.items():
            f.write(f"{key}={value}\n")

    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python gen_bnbd_inputs.py <data_dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)

    # Find all test case directories
    test_cases = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / 'bbnd.ini').exists():
            test_cases.append(item)

    if not test_cases:
        print(f"No test cases found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(test_cases)} test cases")
    print("=" * 60)

    # Process each test case
    success_count = 0
    for case_dir in sorted(test_cases):
        if process_test_case(case_dir):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Successfully processed {success_count}/{len(test_cases)} test cases")


if __name__ == '__main__':
    main()
