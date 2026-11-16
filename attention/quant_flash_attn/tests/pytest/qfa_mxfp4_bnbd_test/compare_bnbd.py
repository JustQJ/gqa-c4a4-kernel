#!/usr/bin/env python3
"""
Compare PA_BBND and PA_BNBD outputs for accuracy verification.
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def load_output(case_dir, layout, suffix="_out.bin"):
    """Load output binary file and convert to float32."""
    filename = f"out_{layout}{suffix}"
    filepath = case_dir / filename
    if not filepath.exists():
        return None

    # Read as uint16 (BF16 format)
    bf16_data = np.fromfile(filepath, dtype=np.uint16)

    # Convert BF16 to float32:
    # BF16 is stored as uint16, and represents the upper 16 bits of float32
    # So we need to shift left by 16 bits and reinterpret as float32
    float_data = (bf16_data.astype(np.uint32) << 16).view(np.float32)

    return float_data


def compare_arrays(bbnd_data, bnbd_data, rtol=1e-5, atol=1e-8):
    """Compare two arrays and return metrics."""
    if bbnd_data is None or bnbd_data is None:
        return None

    # Calculate differences
    abs_diff = np.abs(bbnd_data - bnbd_data)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Relative L2 error
    bbnd_norm = np.linalg.norm(bbnd_data)
    diff_norm = np.linalg.norm(abs_diff)
    relative_l2 = float(diff_norm / bbnd_norm) if bbnd_norm > 0 else 0.0

    # Check if close
    is_close = np.isclose(bbnd_data, bnbd_data, rtol=rtol, atol=atol)
    isclose_ratio = float(np.sum(is_close) / len(is_close))

    # Bit exact match
    bit_exact = np.array_equal(bbnd_data, bnbd_data)

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2": relative_l2,
        "isclose_ratio": isclose_ratio,
        "bit_exact": bit_exact,
        "num_elements": len(bbnd_data)
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PA_BBND vs PA_BNBD outputs")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing test case data")
    parser.add_argument("--output", type=str, default="bnbd_report.json", help="Output JSON report file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return 1

    # Find all test cases
    test_cases = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "bbnd.ini").exists():
            test_cases.append(item.name)

    test_cases.sort()
    print(f"Found {len(test_cases)} test cases")

    # Compare each test case
    results = []
    passed = 0
    failed = 0

    for case_name in test_cases:
        case_dir = data_dir / case_name
        print(f"\nComparing {case_name}...")

        # Load outputs
        bbnd_out = load_output(case_dir, "bbnd")
        bnbd_out = load_output(case_dir, "bnbd")

        if bbnd_out is None:
            print(f"  Warning: BBND output not found for {case_name}")
            continue

        if bnbd_out is None:
            print(f"  Warning: BNBD output not found for {case_name}")
            continue

        # Compare
        metrics = compare_arrays(bbnd_out, bnbd_out)

        if metrics is None:
            print(f"  Error: Could not compare {case_name}")
            continue

        # Determine pass/fail
        is_pass = metrics["bit_exact"] or (metrics["relative_l2"] < 1e-4 and metrics["isclose_ratio"] > 0.999)

        result = {
            "test_case": case_name,
            "passed": is_pass,
            "metrics": metrics
        }
        results.append(result)

        if is_pass:
            passed += 1
            print(f"  ✓ PASS - bit_exact={metrics['bit_exact']}, relative_l2={metrics['relative_l2']:.6e}, isclose={metrics['isclose_ratio']:.4f}")
        else:
            failed += 1
            print(f"  ✗ FAIL - bit_exact={metrics['bit_exact']}, relative_l2={metrics['relative_l2']:.6e}, isclose={metrics['isclose_ratio']:.4f}")

    # Generate report
    report = {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed}/{len(results)} ({100*passed/len(results):.1f}%)" if len(results) > 0 else "0/0 (0%)"
        },
        "results": results
    }

    # Write report
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Report saved to: {output_path}")
    print(f"Summary: {report['summary']['pass_rate']}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
