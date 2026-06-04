#!/usr/bin/env bash
# Exp 1: G19 both sides + holdout τ (level)
# Exp 2: G20 both sides + EMA1200 position/slope side mask
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts
EXP=config/experiments/20260602_fast_scalp_tree_validate

echo "=== Step 0: holdout τ-scan (pred quantile) ==="
bash "$EXP/run_h3_tau_scan_both_sides.sh"

echo "=== Step 1: snapshots G19/G20 ==="
python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G19_h3_both_sides_strategies \
           fast_scalp_alpha_G20_h3_both_sides_ema_regime_strategies

echo "=== Step 2: event segment matrix (G3 ref + G19 + G20) ==="
python -m scripts.event_backtest --variant-grid "$EXP/segment_validate_both_sides_20260603.yaml"

echo "=== BOTH SIDES DONE ==="
