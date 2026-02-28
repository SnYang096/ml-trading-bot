#!/bin/bash
# 一致性验证脚本: 特征一致性 + 事件回测
set -e
cd /home/yin/trading/ml_trading_bot

echo "=========================================="
echo "  Step 1: 特征一致性验证"
echo "=========================================="
python scripts/compare_same_data.py 2>&1

echo ""
echo "=========================================="
echo "  Step 2: 事件回测 (BPC+FER+ME, 180天)"
echo "=========================================="
python scripts/event_backtest.py --strategy bpc,fer,me --days 180 2>&1

echo ""
echo "CONSISTENCY_CHECK_DONE"
