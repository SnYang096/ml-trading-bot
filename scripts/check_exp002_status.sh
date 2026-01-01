#!/bin/bash
LOG="/workspaces/ml_trading_bot/results/exp002_log.txt"
echo "TIME=$(date '+%H:%M')"
if pgrep -f "run_exp002" > /dev/null; then echo "RUNNING=YES"; else echo "RUNNING=NO"; fi
grep -E "^\[.*\]|DONE|RESULTS|Sharpe|Error" "$LOG" 2>/dev/null | tail -10

