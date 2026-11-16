#!/bin/bash
# Test script for PA_BNBD layout support

set -e

echo "=========================================="
echo "Running PA_BNBD Test Suite"
echo "=========================================="

# 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置自定义算子路径
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/vendors/c4a4_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH

# 切换到测试目录
cd /home/k831/C4A4-NZ-kimi/nz_test

# 生成 BNBD 测试数据
echo ""
echo "[1/3] Generating PA_BNBD test data..."
python3 gen_bnbd_inputs.py data

# 检查测试驱动
echo ""
echo "[2/3] Checking test driver..."
if [ ! -f qfa_driver ]; then
    echo "Error: qfa_driver not found!"
    exit 1
fi
echo "qfa_driver found and ready to use"

# 运行测试
echo ""
echo "[3/3] Running accuracy tests..."
echo ""

# 定义测试用例
declare -a test_cases=(
    "empty_and_q_padding_N2TGD_M0"
    "empty_and_q_padding_N2TGD_M3"
    "empty_and_q_padding_TND_M0"
    "empty_and_q_padding_TND_M3"
    "page_tail_N2TGD_M0"
    "page_tail_N2TGD_M3"
    "page_tail_TND_M0"
    "page_tail_TND_M3"
    "smoke_N2TGD_M0"
    "smoke_N2TGD_M3"
    "smoke_TND_M0"
    "smoke_TND_M3"
)

# 运行每个测试用例
for case in "${test_cases[@]}"; do
    echo "----------------------------------------"
    echo "Testing: $case"
    echo "----------------------------------------"

    case_dir="data/$case"

    # 运行 BBND 测试（基准）
    echo "  Running BBND baseline..."
    ./qfa_driver "$case_dir" "bbnd.ini" "out_bbnd" 2>&1 | grep -E "(PASS|FAIL|Error)" || true

    # 运行 BNBD 测试
    echo "  Running BNBD test..."
    ./qfa_driver "$case_dir" "bnbd.ini" "out_bnbd" 2>&1 | grep -E "(PASS|FAIL|Error)" || true

    echo "  Done: $case"
    echo ""
done

echo "=========================================="
echo "PA_BNBD Test Suite Completed"
echo "=========================================="
echo ""
echo "Output files generated:"
ls -lh data/*/out_bnbd_*.bin 2>/dev/null | head -5
echo "..."
echo ""
echo "To compare results, run:"
echo "  python3 compare_bnbd.py --data-dir data --output bnbd_report.json"
