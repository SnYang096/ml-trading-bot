#!/usr/bin/env bash
# W7 vs E0 分段 trading map（segment grid + trading_map: true）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

GRID="config/experiments/20260616_t5_wall_entry_v2_validate/t5_wall_entry_w7_maps_grid.yaml"
LOG_DIR="results/tpc/experiments/t5_wall_entry_v2_20260616/logs"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/w7_maps_grid.log") 2>&1
echo "W7 trading maps start $(date -Iseconds)"
python -m scripts.event_backtest --variant-grid "$GRID" --quiet-signal-logs
echo "W7 trading maps done $(date -Iseconds)"
echo "maps:"
find results/tpc/experiments/t5_wall_entry_v2_20260616 \
  -path '*/trading_map_tpc_event.html' 2>/dev/null | sort
