#!/usr/bin/env bash
# Vector RR backtest on canonical market segments for legacy tree_strategies slugs.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts

OUT_ROOT="results/rd_loop/tree_legacy_bull_bear_20260604"
SYMS="BTCUSDT,ETHUSDT"
TF="240T"
DATA="data/parquet_data"
TRAIN_START="2022-01-01"
TRAIN_END="2026-03-31"
HOLDOUT_START="2025-10-01"
HOLDOUT_END="2026-03-31"
FIXED_Q="0.10"

declare -A FS_LAYER=(
  [compression_breakout]=features_compression_breakout_240T_d6fa7cd035
  [sr_breakout]=features_sr_breakout_240T_58f85a9959
)

SEGMENTS=(
  "bear_2022:2022-01-01:2023-01-01"
  "bull_2023_2024:2023-01-01:2025-01-01"
  "recent_range_to_bear:2025-01-01:2026-04-01"
)

run_tau_segment() {
  local strat="$1" artifact="$2" layer="$3" seg="$4" start="$5" end="$6"
  local out="${OUT_ROOT}/${strat}/${seg}"
  mkdir -p "$out"
  echo "=== ${strat} @ ${seg} (${start} → ${end}) ==="
  python scripts/research/tree_holdout_tau_rr_scan.py \
    --config "config/strategies/tree_strategies/${strat}" \
    --artifact-dir "$artifact" \
    --feature-store-layer "$layer" \
    --output-dir "$out" \
    --start-date "$start" \
    --end-date "$end" \
    --symbols "$SYMS" \
    --timeframe "$TF" \
    --data-path "$DATA" \
    --segment-label "$seg" \
    --fixed-quantile "$FIXED_Q" \
    --no-filter-split
}

train_strategy() {
  local strat="$1"
  local out="results/train_final/${strat}/segment_bull_bear_20260604"
  echo "=== TRAIN ${strat} → ${out} ==="
  local extra=()
  if [[ -n "${FS_LAYER[$strat]:-}" ]]; then
    extra=(--feature-store-layer "${FS_LAYER[$strat]}")
  fi
  python scripts/train_strategy_pipeline.py \
    --config "config/strategies/tree_strategies/${strat}" \
    --symbol "$SYMS" \
    --timeframe "$TF" \
    --data-path "$DATA" \
    --start-date "$TRAIN_START" \
    --end-date "$TRAIN_END" \
    --holdout-start-date "$HOLDOUT_START" \
    --holdout-end-date "$HOLDOUT_END" \
    --output-root "$out" \
    --deterministic \
    "${extra[@]}"
}

resolve_artifact() {
  local strat="$1"
  local pinned="results/train_final/${strat}/train_final_20260530_124749_btceth/${strat}"
  if [[ -f "${pinned}/model.pkl" ]]; then
    echo "$pinned"
    return 0
  fi
  local fresh="results/train_final/${strat}/segment_bull_bear_20260604/${strat}"
  if [[ -f "${fresh}/model.pkl" ]]; then
    echo "$fresh"
    return 0
  fi
  return 1
}

resolve_layer() {
  local strat="$1"
  if [[ -n "${FS_LAYER[$strat]:-}" ]]; then
    echo "${FS_LAYER[$strat]}"
    return 0
  fi
  local found
  found="$(ls feature_store 2>/dev/null | rg "^features_${strat}_${TF}_" | tail -1 || true)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  return 1
}

for strat in compression_breakout sr_breakout trend_following sr_reversal_rr_reg_long; do
  artifact="$(resolve_artifact "$strat" || true)"
  if [[ -z "${artifact:-}" ]]; then
    train_strategy "$strat"
    artifact="$(resolve_artifact "$strat")"
  fi
  layer="$(resolve_layer "$strat" || true)"
  if [[ -z "${layer:-}" ]]; then
    echo "ERROR: no feature_store layer for ${strat}" >&2
    exit 1
  fi
  FS_LAYER[$strat]="$layer"
  for row in "${SEGMENTS[@]}"; do
    IFS=':' read -r seg start end <<<"$row"
    run_tau_segment "$strat" "$artifact" "$layer" "$seg" "$start" "$end"
  done
done

python scripts/research/summarize_tree_legacy_segment_matrix.py --root "$OUT_ROOT"
echo "=== DONE → ${OUT_ROOT}/segment_matrix_summary.md ==="
