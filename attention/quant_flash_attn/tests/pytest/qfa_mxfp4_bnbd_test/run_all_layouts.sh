#!/bin/bash
# Full cross-layout verification: PA_BBND / PA_NZ / PA_BNBD
# for both C4A4 (MXFP4) and C8A8 (MXFP8)
# Total: 6 configs × 12 cases = 72 NPU executions
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/vendors/c4a4_transformer/vendors/c4a4_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH

cd /home/k831/C4A4-NZ-kimi/nz_test

CASES=(
    "smoke_TND_M0" "smoke_TND_M3" "smoke_N2TGD_M0" "smoke_N2TGD_M3"
    "empty_and_q_padding_TND_M0" "empty_and_q_padding_TND_M3" "empty_and_q_padding_N2TGD_M0" "empty_and_q_padding_N2TGD_M3"
    "page_tail_TND_M0" "page_tail_TND_M3" "page_tail_N2TGD_M0" "page_tail_N2TGD_M3"
)

# C4A4 configs: bbnd, nz, bnbd
# C8A8 configs: c8_bbnd, c8_nz, c8_bnbd
CONFIGS=(bbnd nz bnbd c8_bbnd c8_nz c8_bnbd)

echo "=========================================="
echo "Full Cross-Layout Verification"
echo "=========================================="
echo "Configs: ${CONFIGS[*]}"
echo "Cases: ${#CASES[@]}"
echo "Total executions: $((${#CASES[@]} * ${#CONFIGS[@]}))"
echo "=========================================="
echo ""

PASS=0
FAIL=0

for case in "${CASES[@]}"; do
    echo "=== $case ==="
    for cfg in "${CONFIGS[@]}"; do
        ini_file="data/$case/$cfg.ini"
        if [ ! -f "$ini_file" ]; then
            echo "  SKIP $cfg (no $cfg.ini)"
            continue
        fi
        result=$(./qfa_driver "data/$case" "$cfg.ini" "out_$cfg" 2>&1)
        if echo "$result" | grep -q "DRIVER_OK"; then
            echo "  ✓ $cfg OK"
            PASS=$((PASS+1))
        else
            echo "  ✗ $cfg FAIL: $(echo "$result" | tail -3)"
            FAIL=$((FAIL+1))
        fi
    done
    echo ""
done

echo "=========================================="
echo "Execution Summary: $PASS passed, $FAIL failed"
echo "=========================================="
echo ""

if [ $FAIL -gt 0 ]; then
    echo "ERROR: Some executions failed. Skipping comparison."
    exit 1
fi

echo "=== Running Cross-Layout Comparison ==="
python3 compare_all_layouts.py --data data --report data/cross_layout_report.json 2>&1

echo ""
echo "=========================================="
echo "Full Verification Complete"
echo "Report: data/cross_layout_report.json"
echo "=========================================="
