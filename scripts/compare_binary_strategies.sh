#!/bin/bash
# 对比二分类反转策略：带权重 vs 无权重

set -e

SYMBOL=${1:-BTCUSDT}
TIMEFRAME=${2:-240T}
DATA_PATH=${3:-data/parquet_data}
OUTPUT_ROOT=${4:-results/strategies_comparison}

echo "=========================================="
echo "对比二分类反转策略性能"
echo "=========================================="
echo "交易符号: $SYMBOL"
echo "时间周期: $TIMEFRAME"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_ROOT"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 训练策略1：无权重版本
echo ""
echo "=========================================="
echo "策略1: sr_reversal_long (全量扫描，无权重)"
echo "=========================================="
python3 scripts/train_strategy_pipeline.py \
    --config config/strategies/sr_reversal_long \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --data-path "$DATA_PATH" \
    --output-root "$OUTPUT_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/sr_reversal_long_training.log"

# 训练策略2：带权重版本
echo ""
echo "=========================================="
echo "策略2: sr_reversal_long_weighted (全量扫描，带样本权重)"
echo "=========================================="
python3 scripts/train_strategy_pipeline.py \
    --config config/strategies/sr_reversal_long_weighted \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --data-path "$DATA_PATH" \
    --output-root "$OUTPUT_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/sr_reversal_long_weighted_training.log"

echo ""
echo "=========================================="
echo "对比完成！"
echo "=========================================="
echo "结果目录: $OUTPUT_ROOT"
echo ""
echo "查看结果:"
echo "  - 无权重版本: $OUTPUT_ROOT/sr_reversal_long/results.json"
echo "  - 带权重版本: $OUTPUT_ROOT/sr_reversal_long_weighted/results.json"
echo "  - 训练日志: $OUTPUT_ROOT/*_training.log"
echo ""

