# fast_scalp strategy config

| File | Role |
|------|------|
| **`ic_screen.yaml`** | **Holdout IC rules** (peak lag whitelist, thresholds, writeback) — read before ic-prune |
| `features.yaml` | Prepare pool + post-prune column singletons |
| `labels.yaml` | Signed forward RR @ H=3 (`target: label`) |
| `model.yaml` | LightGBM trainer |
| `backtest.yaml` | RR backtest + τ thresholds |
| `meta.yaml` | Slug metadata |

# fast_scalp archetypes

Tree channel layout (no prefilter / no entry_filters):

| File | Role |
|------|------|
| `model_features.yaml` | IC-pruned **model input columns** (rule-style `feature:` entries; regenerate via ic-prune) |
| `regime.yaml` | Thin EMA1200 macro filter |
| `direction.yaml` | Entry tree score τ (direction + timing) |
| `gate.yaml` | Rejection tree (Phase 2; disabled in Phase 1) |
| `execution.yaml` | Single-leg SL/trail/time stop |

## Provenance

| Field | Value |
|-------|-------|
| IC freeze date | 2026-05-31 |
| train_final run_id | `train_top20_cols_scoped_20260531` |
| entry τ plateau | holdout q=0.05 Sharpe 0.73；网格峰值 q=0.15 → 0.76（见 DECISION） |
| dual-period backtest | τ 双段 ✅ `dual_period_top20/`；event_backtest **0 trades**（树 score 待接线） |
