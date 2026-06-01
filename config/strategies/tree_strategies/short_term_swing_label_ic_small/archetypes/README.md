# short_term_swing strategy config

| File | Role |
|------|------|
| **`ic_screen.yaml`** | **Holdout IC rules** (peak lag ∈ {10,20}, min_ic, writeback) — read before ic-prune |
| `features.yaml` | Slow hypothesis prepare pool + post-prune column singletons |
| `labels.yaml` | Signed forward RR @ H=20 (`target: label`) |
| `model.yaml` | LightGBM trainer |
| `backtest.yaml` | RR backtest + τ thresholds |
| `meta.yaml` | Slug metadata |

# short_term_swing archetypes

| File | Role |
|------|------|
| `model_features.yaml` | IC-pruned model input columns (regenerate via ic-prune) |
