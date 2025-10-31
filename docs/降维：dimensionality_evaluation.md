## Dimensionality Reduction Effectiveness: How to Evaluate

This guide explains how to judge whether dimensionality reduction (Top‑K selection and/or Autoencoder compression) is actually beneficial to your production training.

### Recommended paths

- Quick, production-style comparison (preferred)
  - Runs paired training with identical data splits and outputs a single JSON summary with metrics.
  - Command:
  ```bash
  make dim-compare SYMBOL=BTCUSDT ENCODING_DIM=16 DIM_COMPARE_ARGS="--top-k 40"
  ```

- Manual two-run comparison
  1) Train full-feature model:
  ```bash
  make train SYMBOLS="BTCUSDT" START_DATE=2024-10-01 END_DATE=2024-12-31 OVERWRITE=1
  ```
  2) Produce Top‑K list and train Top‑K model:
  ```bash
  make dimensionality-real SYMBOL=BTCUSDT DATA_DIR=data/parquet_data
  make train-topk SYMBOLS="BTCUSDT" START_DATE=2024-10-01 END_DATE=2024-12-31 \
    TOP_FACTORS=results/.../top_factors_BTCUSDT.json OVERWRITE=1
  ```

### What to compare

- Core regression metrics
  - R²: higher is better
  - RMSE/MAE: lower is better
  - Compare both in-sample (train) and out-of-sample (validation/holdout)

- Stability and robustness
  - Rolling-window metrics variance: lower is better
  - Performance under drift periods (drawdown depth/duration)

- Backtest quality (optional but recommended)
  - Sharpe/Sortino; max drawdown; turnover; slippage sensitivity
  - Exposure stability and position concentration

- Model complexity and cost
  - Training time and memory footprint
  - Feature count reduction vs. accuracy trade-off

### Decision criteria

Consider dimensionality reduction “effective” if most of the following hold:

- Out-of-sample metrics improve or remain flat (R²↑ and/or RMSE/MAE↓)
- Rolling stability improves (lower variance of metrics across windows)
- Backtest risk-adjusted returns improve at similar or lower turnover
- Significant reduction in features (e.g., 50%+) with little/no accuracy loss
- Operational cost decreases (faster training/inference, less memory)

### Where to find outputs

- `make dim-compare` writes a JSON summary (e.g., `production_results.json`) containing side-by-side metrics for original vs compressed/Top‑K.
- `make dimensionality-real` writes `results/.../top_factors_<symbol>.json` and a research report with visual comparisons.
- `make train` / `make train-topk` write models under `models/` and logs/metrics under `results/`.

### Tips

- Keep the same date ranges and symbols when comparing.
- If using Autoencoder, tune `ENCODING_DIM` and AE epochs; too aggressive compression may lose predictive signal.
- Validate that features used by Top‑K are actually present in the production feature engineer; reconcile names if needed.


