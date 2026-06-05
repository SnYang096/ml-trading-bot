#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

echo "==> materialize config_experiments trees"
python scripts/research/prepare_tpc_entry_semantic_snapshots.py

mkdir -p results/tpc/maps/compare_entry_semantic_20260604/logs
exec > >(tee -a results/tpc/maps/compare_entry_semantic_20260604/logs/run_all.log) 2>&1

python -m scripts.event_backtest \
  --variant-grid config/experiments/20260604_tpc_entry_semantic_trading_maps/tpc_trading_maps_grid.yaml \
  --quiet-signal-logs

echo "done."
