# Cross-Sectional Multi-Factor Workflow

This guide explains how to build and evaluate cross-sectional (CS) models using
the utilities under `src/ml_trading/cross_sectional`.

## 0. Generate a Cross-Sectional Panel

```bash
PYTHONPATH=src python scripts/cross_sectional/generate_panel.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --timeframe 15T \
  --horizon 12 \
  --start-date 2024-11-01 \
  --end-date 2025-04-30 \
  --feature-type baseline \
  --output results/feature_exports/15T_baseline_12b.parquet
```

- Creates a parquet with MultiIndex `(timestamp, symbol)` and columns `close`, engineered factors, and `future_return_12`.
- Internally uses `PanelGenerationConfig` with either baseline or comprehensive feature engineer. Pass `--no-dropna` if you prefer to keep NaNs for later filtering.
- Expects raw agg-trade parquet/zip files under `data/parquet_data/`. If your order-flow archives are unavailable (or you only want OHLCV), pass `--no-orderflow`.

## 1. Assemble a Factor Panel

```python
from ml_trading.cross_sectional import FactorPanelBuilder, PanelConfig

config = PanelConfig(
    timestamp_col="timestamp",
    symbol_col="symbol",
    feature_cols=["sr_dist_high_s", "cvd_hurst", "taker_buy_ratio"],
    target_col="future_return_1h",
    forward_return_horizon=12,  # assuming 5 minute bars
    min_assets_per_ts=3,
    fill_method="ffill",
)

builder = FactorPanelBuilder(config)
panel = builder.from_concat_frame(features_df)
diagnostics = builder.describe_panel(panel)
```

- Input dataframe must include `timestamp` and `symbol`.
- When `forward_return_horizon` is provided and `target_col` is missing, the builder
  will compute forward returns using the `close` price.
- Use `fill_method=None` to disable filling (will drop any rows containing NaNs).

## 2. Cross-Sectional Processing

```python
from ml_trading.cross_sectional import (
    winsorize_by_sigma,
    cross_sectional_zscore,
    add_crypto_cross_sectional_factors,
)

factor_cols = ["sr_dist_high_s", "cvd_hurst", "taker_buy_ratio"]
panel = winsorize_by_sigma(panel, factor_cols, sigma=3.0)
panel = cross_sectional_zscore(panel, factor_cols, clip_sigma=4.0)
# Add crypto-specific panel factors (relative momentum, dominance, volume share)
panel = add_crypto_cross_sectional_factors(panel)
factor_cols = factor_cols + [
    col for col in panel.columns if col.startswith("cs_crypto_")
]
```

- Winsorization clips outliers within each timestamp slice.
- Z-score normalisation standardises exposures, making them comparable across assets.
- Use `cross_sectional_rank` when targeting rank-based alphas.

## 3. Fit a Cross-Sectional Model

```python
from ml_trading.cross_sectional import CrossSectionalRegressor

model = CrossSectionalRegressor(add_intercept=True, min_assets=4)
result = model.fit(panel, factor_cols=factor_cols, target_col="future_return_1h")

factor_summary = result.factor_summary()
ic_summary = result.ic_summary()
```

- The regressor runs Fama-MacBeth style regressions per timestamp.
- Diagnostics provide factor premia, information coefficients and residuals.
- Use `result.residuals` with the original panel index for residual screening.

## 4. Boosting-Based Cross-Sectional Model

```python
from ml_trading.cross_sectional import CrossSectionalBoostingModel

boost = CrossSectionalBoostingModel()
boost.fit(panel, feature_cols=factor_cols, target_col="future_return_1h")
pred_series = boost.predict(panel)
eval_result = boost.evaluate(panel, pred_series)
ic_stats = eval_result.ic_summary()
```

- Uses `sklearn`'s `HistGradientBoostingRegressor` by default; pass your own estimator if needed.
- `evaluate` returns IC / rank-IC per timestamp plus MSE diagnostics.
- Combine with ranking or neutralization steps before fitting to control exposure structure.
- Want automatic screening? Use the built-in IC/IR selector:

```bash
make cross-sectional-train \
  CS_TRAIN_AUTO_SELECT=1 \
  CS_TRAIN_SELECT_TOPK=50 \
  CS_TRAIN_IC_THRESHOLD=0.01 \
  CS_TRAIN_IR_THRESHOLD=0.5
```

This computes per-timestamp rank ICs, filters by the given thresholds, keeps the top-K by `ic_mean` (or `ic_ir` if `CS_TRAIN_SELECTION_STAT=ir`), and logs the selection metrics to `selection_metrics.json`.

## 5. Generate Expected Returns

```python
latest_slice = panel.xs(panel.index.get_level_values(0).max(), level=0)
expected_returns = model.predict(latest_slice)
```

Predicted returns can be consumed by portfolio construction modules or buckets.

## 6. Suggested Next Steps

- Plug panel construction into the existing `pipeline/training` flow to create CS-aware datasets.
- Add control variables (e.g. size, liquidity buckets) and use `neutralize_against` before regression.
- Evaluate the resulting spreads or IC time-series in `scripts/analysis` utilities.

### CLI Shortcuts

```bash
# 生成面板 + 报告 + 训练（自动推断年化频次）
make cross-sectional-workflow \
  CS_BUILD_SYMBOLS="BTCUSDT ETHUSDT" \
  CS_BUILD_TIMEFRAME=15T \
  CS_BUILD_HORIZON=12 \
  CS_BUILD_START=2024-11-01 \
  CS_BUILD_END=2025-04-30 \
  CS_PERIODS_PER_YEAR=auto \
  CS_TRAIN_AUTO_SELECT=1 \
  CS_TRAIN_SELECT_TOPK=50 \
  CS_TRAIN_IC_THRESHOLD=0.01 \
  CS_TRAIN_IR_THRESHOLD=0.5
```

- `CS_PERIODS_PER_YEAR=auto` 会根据索引间隔推断一年内的截面次数（例如 5 分钟约等于 17520）。
- 若 panel 中混入多个 timeframes，脚本会报错提示拆分；请确保每次输入仅含单周期数据。
- 若需要固定值，可自行覆盖（如 `CS_PERIODS_PER_YEAR=252`）。

```bash
# 从指定面板自动筛选因子（分类别→全局 Top-K）
make cross-sectional-select \
  CS_SELECT_INPUT="results/feature_exports/15T_baseline_12b.parquet" \
  CS_SELECT_MIN_ASSETS=8 \
  CS_SELECT_PER_CATEGORY_TOP=2 \
  CS_SELECT_GLOBAL_TOP=12 \
  CS_SELECT_IC_THRESHOLD=0.01 \
  CS_SELECT_IR_THRESHOLD=0.5
cat results/cross_sectional/selected_factors.txt
```

```bash
# 一键：生成面板 → 自动筛因子 → 报告 → 训练
make cross-sectional-auto \
  CS_BUILD_SYMBOLS="BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT DOTUSDT" \
  CS_BUILD_TIMEFRAME=15T \
  CS_BUILD_HORIZON=12 \
  CS_BUILD_START=2024-11-01 \
  CS_BUILD_END=2025-04-30 \
  CS_PERIODS_PER_YEAR=auto \
  CS_AUTO_PER_CATEGORY_TOP=2 \
  CS_AUTO_GLOBAL_TOP=12 \
  CS_AUTO_IC_THRESHOLD=0.01 \
  CS_AUTO_IR_THRESHOLD=0.5
```

- 自动流程会生成 `selected_factors.txt` 与 `selection_summary.json` 供复盘；后续可调整阈值/Top-K 或重新运行。
- 如果报告效果不佳，可调大 `CS_AUTO_PER_CATEGORY_TOP` 或 `CS_AUTO_GLOBAL_TOP`（前提是同截面的资产数足够），或通过 `CS_REPORT_EXTRA="--no-crypto-factors"` 等参数控制加入的特征族。

## Module Overview

| File | Purpose |
| ---- | ------- |
| `src/ml_trading/cross_sectional/panel.py` | Panel assembly and diagnostics |
| `src/ml_trading/cross_sectional/panel_generation.py` | Generate multi-asset panels from raw data |
| `src/ml_trading/cross_sectional/processing.py` | Cross-sectional preprocessing (winsorize, z-score, neutralize) |
| `src/ml_trading/cross_sectional/crypto_factors.py` | Crypto-specific cross-sectional factors (momentum dominance, liquidity, order flow) |
| `src/ml_trading/cross_sectional/factor_catalog.py` | Categorise factors into heuristic groups |
| `src/ml_trading/cross_sectional/factor_selection.py` | IC/IR scoring helpers and Top-K selection |
| `src/ml_trading/cross_sectional/model.py` | Fama-MacBeth style regression and prediction |
| `src/ml_trading/cross_sectional/boosting.py` | Gradient boosting wrapper for CS alphas |
| `scripts/cross_sectional/export_factor_catalog.py` | Export factor lists per category |
| `scripts/cross_sectional/auto_select_factors.py` | Fully automated factor selection CLI |


