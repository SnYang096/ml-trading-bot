#!/usr/bin/env bash
# 分段 trading map：走 segment grid + trading_map: true（勿再全窗重跑一遍）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

GRID="config/experiments/20260610_tpc_macro_pullback_replace/tpc_macro_pullback_bull_maps_grid.yaml"
mkdir -p results/tpc/experiments/macro_pullback_add_20260610/logs

exec > >(tee -a results/tpc/experiments/macro_pullback_add_20260610/logs/bull_maps_grid.log) 2>&1
echo "bull segment maps grid start $(date -Iseconds)"
python -m scripts.event_backtest --variant-grid "$GRID" --quiet-signal-logs
echo "bull segment maps grid done $(date -Iseconds)"
echo "maps:"
find results/tpc/experiments/macro_pullback_replace_20260610 results/tpc/experiments/macro_pullback_add_20260610 \
  -path '*/bull_2023_2024/trading_map_tpc_event.html' 2>/dev/null | sort
