#!/usr/bin/env bash
# Dual-head reg + EMA-train variants + G3 adverse gate follow-up.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT="$PWD"
export PYTHONPATH=src:scripts
CFG=config/strategies/tree_strategies/fast_scalp
EXP=config/experiments/20260602_fast_scalp_tree_validate
OVR=$EXP/overrides
SYMS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
OUT_RD=results/rd_loop/fast_scalp_tree_validate

echo "=== 0. Ensure H3 full-history preds exist ==="
test -f $OUT_RD/track_a/scores/h3_baseline_preds.parquet || {
  python scripts/research/export_tree_scores_from_artifact.py \
    --artifact-dir results/train_final/fast_scalp/train_baseline_h3/fast_scalp \
    --config $CFG --symbols $SYMS \
    --start-date 2022-01-01 --end-date 2026-04-01 \
    --validate-short-entry 0.45 \
    --output $OUT_RD/track_a/scores/h3_baseline_full_history.parquet \
    --save-predictions $OUT_RD/track_a/scores/h3_baseline_preds.parquet
}

echo "=== 1. Prepare EMA1200 column for regime-conditioned dual-head train ==="
python scripts/train_strategy_pipeline.py \
  --config $CFG --features $OVR/features_ema1200_only.yaml \
  --symbol $SYMS --timeframe 120T \
  --start-date 2024-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/fast_scalp/prepare_ema1200_only \
  --prepare-only

echo "=== 2a. Dual head — profile=reg (regime ON at event, like G7) ==="
python scripts/research/train_tree_dual_head.py \
  --config $CFG \
  --predictions $OUT_RD/track_a/scores/h3_baseline_preds.parquet \
  --symbols $SYMS \
  --output-dir $OUT_RD/track_a/dual_head_reg \
  --train-end-date 2025-10-01 --score-start-date 2022-01-01 \
  --horizon 3 --rr-floor 0.30 --profile reg

echo "=== 2b. Dual head — profile=reg + EMA train split (long>=0.1, short<=-0.1) ==="
python scripts/research/train_tree_dual_head.py \
  --config $CFG \
  --predictions $OUT_RD/track_a/scores/h3_baseline_preds.parquet \
  --symbols $SYMS \
  --output-dir $OUT_RD/track_a/dual_head_reg_ema \
  --train-end-date 2025-10-01 --score-start-date 2022-01-01 \
  --horizon 3 --rr-floor 0.30 --profile reg \
  --ema-parquet results/train_final/fast_scalp/prepare_ema1200_only/fast_scalp/features_labeled.parquet \
  --long-ema-min 0.10 --short-ema-max -0.10

echo "=== 3. Gate wide prepare (if missing) ==="
test -f results/train_final/fast_scalp/gate_features_wide/fast_scalp/features_labeled.parquet || \
python scripts/train_strategy_pipeline.py \
  --config $CFG --features $OVR/features_gate_candidates.yaml \
  --symbol $SYMS --timeframe 120T \
  --start-date 2024-01-01 --end-date 2026-04-01 \
  --output-root results/train_final/fast_scalp/gate_features_wide --prepare-only

echo "=== 4. G3 adverse gate (H=3 entry τ, full-history preds) ==="
python scripts/research/train_tree_adverse_gate.py \
  --config $CFG \
  --predictions $OUT_RD/track_a/scores/h3_baseline_preds.parquet \
  --gate-features results/train_final/fast_scalp/gate_features_wide/fast_scalp/features_labeled.parquet \
  --features-gate-yaml $OVR/features_gate_candidates.yaml \
  --symbols $SYMS --start-date 2022-01-01 --end-date 2026-04-01 \
  --train-end-date 2025-10-01 \
  --long-entry 0.55 --short-entry 0.45 --entry-mode level \
  --min-abs-ic 0.03 --min-lift 0.05 --top-k 8 \
  --output-dir $OUT_RD/track_a/gate/g3_ic_prune_v2

echo "=== 5. Rebuild H3 inject with G3 gate feature cols ==="
python -c "
import yaml
from pathlib import Path
import pandas as pd
from scripts.research.export_tree_scores_for_event_backtest import export_scores
root = Path('results/rd_loop/fast_scalp_tree_validate/track_a')
tmp = root / 'scores/h3_baseline_preds.parquet'
out = root / 'scores/h3_baseline_full_history.parquet'
summary = yaml.safe_load((root / 'gate/g3_ic_prune_v2/train_summary.json').read_text())
sel = summary.get('selected_features') or []
defaults = ['trend_confidence','bpc_semantic_chop_ts_q','macro_tp_vwap_1200_position','atr','vol_accel','me_accel_5k']
df = pd.read_parquet(tmp)
extra = [c for c in dict.fromkeys(sel + defaults) if c in df.columns]
export_scores(tmp, out, symbols='BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT'.split(','),
            split=None, score_col='pred', start_date='2022-01-01', end_date='2026-04-01',
            extra_cols=extra)
print('rebuilt h3 inject, gate cols:', extra)
"

echo "=== 6. Snapshots G17 / G17 regimeoff / G18 ==="
python scripts/research/prepare_fast_scalp_alpha_snapshots.py \
  --only fast_scalp_alpha_G17_dual_head_reg_strategies \
           fast_scalp_alpha_G17_dual_head_reg_regimeoff_strategies \
           fast_scalp_alpha_G18_g3_h3_gate_strategies

echo "=== 7. Event segment matrix (6 variants x 4 segments) ==="
python -m scripts.event_backtest \
  --variant-grid $EXP/segment_validate_followup_20260603.yaml

echo "=== FOLLOWUP EXPERIMENTS DONE ==="
