#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts
OUT=results/rd_loop/fast_scalp_tree_validate/track_a/tau_scan_h3_both
mkdir -p "$OUT"

echo "=== H=3 holdout τ scan (both sides, pred quantile grid) ==="
python scripts/research/tree_holdout_tau_rr_scan.py \
  --config config/strategies/tree_strategies/fast_scalp \
  --predictions results/rd_loop/fast_scalp_tree_validate/track_a/scores/h3_baseline_preds.parquet \
  --output-dir "$OUT" \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --quantile-grid "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50"

echo "=== Done: $OUT/tau_scan_holdout_rr.json ==="
