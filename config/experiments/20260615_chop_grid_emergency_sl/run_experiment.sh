#!/usr/bin/env bash
# chop_grid Emergency SL backtest runner (variant grid)
# Usage: ./run_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GRID="config/experiments/20260615_chop_grid_emergency_sl/chop_grid_emergency_sl_grid.yaml"

echo "=== chop_grid Emergency SL Experiment ==="
echo "Grid: ${GRID}"
echo "Output root: results/chop_grid/experiments/emergency_sl_20260615/"
echo ""

PYTHONPATH=src:scripts python3 -m scripts.event_backtest \
  --variant-grid "${GRID}"

echo ""
echo "=== Experiment Complete ==="
echo "Index: results/chop_grid/experiments/EXPERIMENT_INDEX.json"
