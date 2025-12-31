#!/bin/bash
# 只输出关键状态，最小化token消耗
LOG="/workspaces/ml_trading_bot/results/pipeline_log.txt"
STATUS_FILE="/workspaces/ml_trading_bot/results/pipeline_status.txt"

{
echo "TIME=$(date '+%H:%M')"

# 检查进程是否还在运行
if pgrep -f "run_full_e2e_pipeline" > /dev/null; then
    echo "RUNNING=YES"
else
    echo "RUNNING=NO"
fi

# 检查最后完成的步骤
LAST_DONE=$(grep "DONE at" "$LOG" 2>/dev/null | tail -1 | grep -oP 'STEP \d/\d' || echo "NONE")
echo "LAST_DONE=$LAST_DONE"

# 检查是否有错误
if grep -q "ERROR\|Error\|Traceback\|Exception" "$LOG" 2>/dev/null; then
    echo "HAS_ERROR=YES"
    grep -A2 "ERROR\|Error\|Traceback" "$LOG" 2>/dev/null | tail -3 > /workspaces/ml_trading_bot/results/last_error.txt
else
    echo "HAS_ERROR=NO"
fi

# 检查是否完成
if grep -q "PIPELINE_STATUS=SUCCESS" "$LOG" 2>/dev/null; then
    echo "COMPLETED=YES"
else
    echo "COMPLETED=NO"
fi

# FeatureStore 进度
BTC_FS=$(ls /workspaces/ml_trading_bot/feature_store/features_*/BTCUSDT/240T/2024-*.parquet 2>/dev/null | wc -l)
ETH_FS=$(ls /workspaces/ml_trading_bot/feature_store/features_*/ETHUSDT/240T/2024-*.parquet 2>/dev/null | wc -l)
echo "FS_BTC_2024=$BTC_FS/12"
echo "FS_ETH_2024=$ETH_FS/12"

} > "$STATUS_FILE"

cat "$STATUS_FILE"

