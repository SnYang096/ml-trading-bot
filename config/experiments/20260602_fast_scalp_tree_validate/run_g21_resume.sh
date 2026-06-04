#!/usr/bin/env bash
# Resume G21 after long/short train (export → τ → snapshot → OOS event)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts
EXP=config/experiments/20260602_fast_scalp_tree_validate
CFG=config/strategies/tree_strategies/fast_scalp
SYMS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
OUT_RD=results/rd_loop/fast_scalp_tree_validate/track_a/independent_sides
TRAIN_ROOT=results/train_final/fast_scalp

echo "=== 3) Export full-history scores ==="
mkdir -p "$OUT_RD/scores"
python scripts/research/export_tree_scores_from_artifact.py \
  --artifact-dir "$TRAIN_ROOT/train_long_win_h3/fast_scalp" \
  --config "$CFG" \
  --symbols "$SYMS" \
  --start-date 2022-01-01 --end-date 2026-04-01 \
  --output "$OUT_RD/scores/long_win_full_history.parquet" \
  --save-predictions "$OUT_RD/scores/long_win_preds.parquet"

python scripts/research/export_tree_scores_from_artifact.py \
  --artifact-dir "$TRAIN_ROOT/train_short_win_h3/fast_scalp" \
  --config "$CFG" \
  --symbols "$SYMS" \
  --start-date 2022-01-01 --end-date 2026-04-01 \
  --output "$OUT_RD/scores/short_win_full_history.parquet" \
  --save-predictions "$OUT_RD/scores/short_win_preds.parquet"

echo "=== 4) Merge + holdout τ scan ==="
python scripts/research/merge_independent_side_scores.py \
  --long-parquet "$OUT_RD/scores/long_win_preds.parquet" \
  --short-parquet "$OUT_RD/scores/short_win_preds.parquet" \
  --output "$OUT_RD/scores/independent_sides_preds.parquet"

python scripts/research/merge_independent_side_scores.py \
  --long-parquet "$OUT_RD/scores/long_win_full_history.parquet" \
  --short-parquet "$OUT_RD/scores/short_win_full_history.parquet" \
  --output "$OUT_RD/scores/independent_sides_event_scores.parquet"

python scripts/research/tree_holdout_tau_dual_prob_scan.py \
  --config "$CFG" \
  --predictions "$OUT_RD/scores/independent_sides_preds.parquet" \
  --output-dir "$OUT_RD/tau_scan"

echo "=== 5) Snapshot G21 + OOS event ==="
python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G21_independent_sides_strategies

python -m scripts.event_backtest \
  --variant-grid "$EXP/segment_validate_independent_sides_oos.yaml"

echo "=== G21 RESUME DONE ==="
