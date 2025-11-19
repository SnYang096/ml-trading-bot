#!/usr/bin/env python3
"""Evaluate trading signal quality using Alphalens.

This script demonstrates how to:
1. Load model predictions
2. Create factor data from predictions
3. Get trade prices (next available prices to avoid look-ahead bias)
4. Evaluate signal quality using Alphalens tear sheets

Based on the workflow from "Machine Learning for Algorithmic Trading"
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns

# Alphalens imports
try:
    import alphalens as al
    from alphalens.utils import get_clean_factor_and_forward_returns
    from alphalens.tears import (
        create_summary_tear_sheet,
        create_full_tear_sheet,
        create_returns_tear_sheet,
        create_information_tear_sheet,
        create_turnover_tear_sheet,
    )
    from alphalens.performance import (
        mean_return_by_quantile,
        factor_information_coefficient,
        mean_information_coefficient,
    )
    from alphalens.plotting import (
        plot_quantile_returns_bar,
        plot_cumulative_returns_by_quantile,
        plot_ic_ts,
    )
except ImportError as e:
    print(f"❌ Failed to import alphalens: {e}")
    print("   Install with: pip install alphalens-reloaded")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")

print("=" * 80)
print("Alphalens Signal Quality Evaluation")
print("=" * 80)

# Setup output directory
output_dir = Path("results/alphalens_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Output directory: {output_dir.absolute()}")

# ============================================================================
# 1. Create Synthetic Predictions (Simulating Model Output)
# ============================================================================
print("\n[1/4] Creating synthetic model predictions...")
np.random.seed(42)

# Create date range: 180 days of daily data
start_date = datetime(2024, 1, 1)
dates = pd.date_range(start=start_date, periods=180, freq="D")
dates = pd.DatetimeIndex(dates.values, freq=None)

# Create 20 synthetic assets
assets = [f"STOCK_{i:03d}" for i in range(1, 21)]
n_assets = len(assets)
n_periods = len(dates)

# Simulate predictions from multiple models (e.g., 3 models)
# In real scenario, these would come from your trained models
predictions_data = []
for date in dates:
    for asset in assets:
        # Simulate predictions from 3 models
        pred_model1 = np.random.randn() * 0.02  # Model 1 prediction
        pred_model2 = np.random.randn() * 0.02  # Model 2 prediction
        pred_model3 = np.random.randn() * 0.02  # Model 3 prediction

        # Average predictions (common ensemble approach)
        avg_pred = (pred_model1 + pred_model2 + pred_model3) / 3

        predictions_data.append(
            {
                "date": date,
                "symbol": asset,
                "model_1": pred_model1,
                "model_2": pred_model2,
                "model_3": pred_model3,
                "prediction": avg_pred,
            }
        )

predictions = pd.DataFrame(predictions_data)
predictions = predictions.set_index(["date", "symbol"])

print(f"✅ Created predictions:")
print(f"   - Shape: {predictions.shape}")
print(f"   - Columns: {list(predictions.columns)}")
print(f"   - Date range: {dates[0]} to {dates[-1]}")

# Save predictions sample
predictions.head(20).to_csv(output_dir / "predictions_sample.csv")
print(f"✅ Predictions sample saved to: {output_dir / 'predictions_sample.csv'}")

# ============================================================================
# 2. Create Factor from Predictions
# ============================================================================
print("\n[2/4] Creating factor from predictions...")

# Use average of top N models (or all models)
# In this case, we use the average of all 3 models
factor = (
    predictions.iloc[:, :3]  # First 3 columns are model predictions
    .mean(axis=1)  # Average predictions
    .to_frame("factor")
)

# Ensure proper index structure: MultiIndex with [date, symbol]
if not isinstance(factor.index, pd.MultiIndex):
    factor = factor.reset_index().set_index(["date", "symbol"])

# Sort index
factor = factor.sort_index()

# Remove frequency from date index
if hasattr(factor.index, "levels") and len(factor.index.levels) > 0:
    date_level = factor.index.levels[0]
    if hasattr(date_level, "freq") and date_level.freq is not None:
        index_tuples = list(factor.index)
        dates_from_tuples = [t[0] for t in index_tuples]
        symbols_from_tuples = [t[1] for t in index_tuples]
        new_dates = pd.DatetimeIndex(dates_from_tuples, freq=None)
        new_index = pd.MultiIndex.from_arrays(
            [new_dates, symbols_from_tuples], names=factor.index.names
        )
        factor = pd.Series(factor["factor"].values, index=new_index, name="factor")

print(f"✅ Factor created:")
print(f"   - Shape: {factor.shape}")
print(f"   - Sample:")
print(factor.head(10))

# ============================================================================
# 3. Get Trade Prices (Next Available Prices)
# ============================================================================
print("\n[3/4] Creating trade prices (next available prices)...")

# Create synthetic price data
# In real scenario, this would come from your price database
price_data = {}
for asset in assets:
    # Create price series with random walk
    base_price = 100.0
    returns = np.random.randn(n_periods) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))
    price_data[asset] = prices

# Create prices DataFrame
prices_df = pd.DataFrame(price_data, index=dates)
prices_df.index.name = "date"
prices_df.index = pd.DatetimeIndex(prices_df.index.values, freq=None)

# IMPORTANT: Use next available prices (shift(-1)) to avoid look-ahead bias
# This ensures we use prices that would be available at the time of trading
trade_prices = prices_df.shift(-1).dropna()

print(f"✅ Trade prices created:")
print(f"   - Original prices shape: {prices_df.shape}")
print(f"   - Trade prices shape (after shift): {trade_prices.shape}")
print(f"   - Date range: {trade_prices.index[0]} to {trade_prices.index[-1]}")

# ============================================================================
# 4. Prepare Alphalens Data
# ============================================================================
print("\n[4/4] Preparing Alphalens data structure...")

# Align factor and prices
# Get common tickers
factor_tickers = factor.index.get_level_values("symbol").unique()
price_tickers = trade_prices.columns
common_tickers = factor_tickers.intersection(price_tickers)

print(f"   - Factor tickers: {len(factor_tickers)}")
print(f"   - Price tickers: {len(price_tickers)}")
print(f"   - Common tickers: {len(common_tickers)}")

# Filter to common tickers
factor_filtered = factor[factor.index.get_level_values("symbol").isin(common_tickers)]
trade_prices_filtered = trade_prices[common_tickers]

# Holding periods (in days)
periods = (1, 5, 10, 21)
quantiles = 5

# Monkey-patch to handle frequency issues
import alphalens.utils as al_utils

original_compute_forward_returns = al_utils.compute_forward_returns


def patched_compute_forward_returns(
    factor, prices, periods, filter_zscore=None, cumulative_returns=True
):
    """Patched version that bypasses frequency validation"""
    if hasattr(factor.index, "levels") and len(factor.index.levels) > 0:
        date_level = factor.index.levels[0]
        if hasattr(date_level, "freq") and (
            date_level.freq is not None or hasattr(date_level, "_freq")
        ):
            index_tuples = list(factor.index)
            dates = [t[0] for t in index_tuples]
            assets = [t[1] for t in index_tuples]
            new_dates = pd.DatetimeIndex(dates, freq=None)
            new_index = pd.MultiIndex.from_arrays(
                [new_dates, assets], names=factor.index.names
            )
            factor = pd.Series(factor.values, index=new_index, name=factor.name)

    if hasattr(prices.index, "freq") and prices.index.freq is not None:
        prices.index = pd.DatetimeIndex(prices.index.values, freq=None)

    try:
        from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin

        class NoFreqContext:
            def __init__(self):
                self.original_validate = None

            def __enter__(self):
                try:
                    self.original_validate = DatetimeLikeArrayMixin._validate_frequency

                    def noop_validate(self, value):
                        pass

                    DatetimeLikeArrayMixin._validate_frequency = noop_validate
                except:
                    pass
                return self

            def __exit__(self, *args):
                if self.original_validate:
                    try:
                        DatetimeLikeArrayMixin._validate_frequency = (
                            self.original_validate
                        )
                    except:
                        pass

        with NoFreqContext():
            result = original_compute_forward_returns(
                factor, prices, periods, filter_zscore, cumulative_returns
            )
        return result
    except Exception:
        return original_compute_forward_returns(
            factor, prices, periods, filter_zscore, cumulative_returns
        )


al_utils.compute_forward_returns = patched_compute_forward_returns

try:
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor_filtered,
        prices=trade_prices_filtered,
        quantiles=quantiles,
        periods=periods,
        bins=None,
        binning_by_group=False,
        max_loss=0.50,  # Allow up to 50% data loss
    )

    # Sort index
    factor_data = factor_data.sort_index()

    print(f"✅ Alphalens data created:")
    print(f"   - Shape: {factor_data.shape}")
    print(f"   - Columns: {list(factor_data.columns)}")
    print(f"   - Index levels: {factor_data.index.names}")
    print(f"   - Sample data:")
    print(factor_data.head(10))

except Exception as e:
    print(f"❌ Failed to create Alphalens data: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
finally:
    al_utils.compute_forward_returns = original_compute_forward_returns

# Save factor data sample
factor_data.reset_index().head(20).to_csv(
    output_dir / "factor_data_sample.csv", index=False
)
print(f"✅ Factor data sample saved")

# ============================================================================
# 5. Generate Tear Sheets and Analysis
# ============================================================================
print("\n[5/5] Generating Alphalens tear sheets and analysis...")


def save_figures(prefix="figure"):
    """Save all matplotlib figures"""
    saved_paths = []
    fig_nums = sorted(plt.get_fignums())

    if len(fig_nums) == 0:
        return saved_paths

    for idx, fig_num in enumerate(fig_nums):
        fig = plt.figure(fig_num)
        axes = fig.get_axes()

        if len(axes) == 0:
            plt.close(fig)
            continue

        fig_path = output_dir / f"{prefix}_fig_{idx+1}.png"
        try:
            fig.canvas.draw()
            fig.savefig(
                fig_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                format="png",
            )
            saved_paths.append(str(fig_path))
        except Exception as e:
            print(f"   ⚠️  Error saving figure {idx+1}: {e}")
        finally:
            plt.close(fig)

    return saved_paths


# Summary tear sheet
try:
    plt.close("all")
    print("   📊 Generating summary tear sheet...")
    create_summary_tear_sheet(factor_data)
    for fig_num in plt.get_fignums():
        plt.figure(fig_num).canvas.draw()
    saved = save_figures(prefix="summary_tear_sheet")
    print(f"   ✅ Summary tear sheet: {len(saved)} figures saved")
except Exception as e:
    print(f"   ⚠️  Summary tear sheet failed: {e}")

# Calculate and save IC statistics
try:
    print("   📊 Calculating IC statistics...")
    ic = factor_information_coefficient(factor_data)
    mean_ic = mean_information_coefficient(factor_data)

    ic.to_csv(output_dir / "ic_time_series.csv")
    mean_ic.to_csv(output_dir / "ic_summary.csv")
    print(f"   ✅ IC statistics saved")

    # Plot IC time series
    plt.close("all")
    plot_ic_ts(ic)
    plt.tight_layout()
    saved = save_figures(prefix="ic_timeseries")
    print(f"   ✅ IC time series chart: {len(saved)} figures saved")
except Exception as e:
    print(f"   ⚠️  IC analysis failed: {e}")

# Mean returns by quantile
try:
    print("   📊 Calculating mean returns by quantile...")
    mean_return_by_q, std_err = mean_return_by_quantile(factor_data)
    mean_return_by_q.to_csv(output_dir / "mean_return_by_quantile.csv")
    print(f"   ✅ Mean returns by quantile saved")

    # Plot returns bar chart
    plt.close("all")
    plot_quantile_returns_bar(mean_return_by_q)
    plt.tight_layout()
    saved = save_figures(prefix="returns_bar")
    print(f"   ✅ Returns bar chart: {len(saved)} figures saved")

    # Cumulative returns
    mean_return_by_q_daily, _ = mean_return_by_quantile(factor_data, by_date=True)
    plt.close("all")
    plot_cumulative_returns_by_quantile(
        mean_return_by_q_daily["5D"], period="5D", freq=None
    )
    plt.tight_layout()
    saved = save_figures(prefix="cumulative_returns")
    print(f"   ✅ Cumulative returns chart: {len(saved)} figures saved")
except Exception as e:
    print(f"   ⚠️  Returns analysis failed: {e}")

# Try full tear sheet
try:
    plt.close("all")
    print("   📊 Generating full tear sheet...")
    create_full_tear_sheet(factor_data, long_short=True, group_neutral=False)
    for fig_num in plt.get_fignums():
        plt.figure(fig_num).canvas.draw()
    saved = save_figures(prefix="full_tear_sheet")
    if len(saved) > 0:
        print(f"   ✅ Full tear sheet: {len(saved)} figures saved")
    else:
        print(f"   ℹ️  Full tear sheet: Statistics printed (no figures)")
except Exception as e:
    print(f"   ⚠️  Full tear sheet failed: {e}")

plt.close("all")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ Alphalens Signal Evaluation Complete!")
print(f"📁 All reports saved to: {output_dir.absolute()}")
print("=" * 80)

print("\n📊 Generated Files:")
print("   - CSV files: predictions_sample.csv, factor_data_sample.csv")
print("   - CSV files: ic_time_series.csv, ic_summary.csv")
print("   - CSV files: mean_return_by_quantile.csv")
print("   - PNG files: Various tear sheets and analysis charts")
print("\n🎉 Evaluation complete!")
