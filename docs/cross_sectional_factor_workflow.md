# Cross-Sectional Multi-Factor Workflow

This guide explains how to build and evaluate cross-sectional (CS) models using
the utilities under `src/ml_trading/cross_sectional`.

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

## Module Overview

| File | Purpose |
| ---- | ------- |
| `src/ml_trading/cross_sectional/panel.py` | Panel assembly and diagnostics |
| `src/ml_trading/cross_sectional/processing.py` | Cross-sectional preprocessing (winsorize, z-score, neutralize) |
| `src/ml_trading/cross_sectional/crypto_factors.py` | Crypto-specific cross-sectional factors (momentum dominance, liquidity, order flow) |
| `src/ml_trading/cross_sectional/model.py` | Fama-MacBeth style regression and prediction |
| `src/ml_trading/cross_sectional/boosting.py` | Gradient boosting wrapper for CS alphas |


