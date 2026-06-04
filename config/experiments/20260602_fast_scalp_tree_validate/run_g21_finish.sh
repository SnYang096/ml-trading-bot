#!/usr/bin/env bash
# Finish G21: merge (if needed) → τ → snapshot → OOS (skip re-export)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts
EXP=config/experiments/20260602_fast_scalp_tree_validate
CFG=config/strategies/tree_strategies/fast_scalp
OUT_RD=results/rd_loop/fast_scalp_tree_validate/track_a/independent_sides

echo "=== Merge (preds + OHLC for τ-scan) ==="
python scripts/research/merge_independent_side_scores.py \
  --long-parquet "$OUT_RD/scores/long_win_preds.parquet" \
  --short-parquet "$OUT_RD/scores/short_win_preds.parquet" \
  --output "$OUT_RD/scores/independent_sides_preds.parquet" \
  --keep-ohlc

python scripts/research/merge_independent_side_scores.py \
  --long-parquet "$OUT_RD/scores/long_win_full_history.parquet" \
  --short-parquet "$OUT_RD/scores/short_win_full_history.parquet" \
  --output "$OUT_RD/scores/independent_sides_event_scores.parquet"

echo "=== τ scan ==="
python scripts/research/tree_holdout_tau_dual_prob_scan.py \
  --config "$CFG" \
  --predictions "$OUT_RD/scores/independent_sides_preds.parquet" \
  --output-dir "$OUT_RD/tau_scan"

python3 -c "
import json
from pathlib import Path
p = Path('$OUT_RD/tau_scan/tau_scan_holdout_dual_prob.json')
d = json.loads(p.read_text())
for side in ('long_scan', 'short_scan'):
    rows = d.get(side) or []
    n_sh = sum(1 for r in rows if r.get('sharpe') is not None)
    print(f'{side}: {n_sh}/{len(rows)} rows with sharpe')
    if n_sh == 0:
        raise SystemExit('tau scan produced no sharpe — check OHLC in merged preds')
print('recommended', d.get('recommended'))
"

echo "=== Snapshot + OOS (G21 only) ==="
python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G21_independent_sides_strategies

python -m scripts.event_backtest \
  --variant-grid "$EXP/segment_validate_g21_oos_only.yaml"

echo "=== G21 FINISH DONE ==="
