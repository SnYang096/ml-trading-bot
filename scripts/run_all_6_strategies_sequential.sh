#!/bin/bash
# 顺序运行6个策略的训练（确保每个策略的结果保存在不同目录）

set -e

SYMBOL=${1:-BTCUSDT}
TIMEFRAME=${2:-240T}
OUTPUT_ROOT=${3:-results/strategies_comparison_6_final}

echo "=========================================="
echo "训练6个策略"
echo "=========================================="
echo "Symbol: $SYMBOL"
echo "Timeframe: $TIMEFRAME"
echo "Output: $OUTPUT_ROOT"
echo ""

strategies=(
    "sr_reversal_long"
    "sr_reversal_long_sr_filter"
    "sr_reversal_long_weighted"
    "sr_reversal_rr_reg_long"
    "sr_reversal_rr_reg_long_sr_filter"
    "sr_reversal_rr_reg_long_weighted"
)

for strategy in "${strategies[@]}"; do
    echo "=========================================="
    echo "训练策略: $strategy"
    echo "=========================================="
    
    python3 scripts/train_strategy_pipeline.py \
        --config "config/strategies/$strategy" \
        --symbol "$SYMBOL" \
        --timeframe "$TIMEFRAME" \
        --output-root "$OUTPUT_ROOT" \
        2>&1 | tee "/tmp/train_${strategy}.log" | tail -20
    
    echo ""
    echo "✅ $strategy 训练完成"
    echo ""
done

echo "=========================================="
echo "所有策略训练完成"
echo "=========================================="

