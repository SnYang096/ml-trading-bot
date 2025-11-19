#!/usr/bin/env python3
"""Complete Alphalens example script based on Quantopian tutorial.

This script demonstrates comprehensive factor analysis using alphalens:
- Creating forward returns and factor quantiles
- Summary tear sheet
- Returns analysis by quantiles
- Information Coefficient analysis
- Turnover analysis
- Various visualizations

Uses synthetic data for demonstration.
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
    from alphalens.performance import (
        mean_return_by_quantile,
        factor_information_coefficient,
        mean_information_coefficient,
        factor_returns,
    )
    from alphalens.plotting import (
        plot_quantile_returns_bar,
        plot_cumulative_returns_by_quantile,
        plot_quantile_returns_violin,
        plot_ic_ts,
    )
    from alphalens.tears import (
        create_summary_tear_sheet,
        create_returns_tear_sheet,
        create_information_tear_sheet,
        create_turnover_tear_sheet,
        create_full_tear_sheet,
    )
except ImportError as e:
    print(f"❌ Failed to import alphalens: {e}")
    print("   Install with: pip install alphalens-reloaded")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")

print("=" * 80)
print("Alphalens Complete Example - Factor Analysis")
print("=" * 80)

# Setup output directory
output_dir = Path("results/alphalens_example")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Output directory: {output_dir.absolute()}")

# ============================================================================
# 1. Create Synthetic Data
# ============================================================================
print("\n[1/6] Creating synthetic factor and price data...")
np.random.seed(42)

# Create date range: 180 days of daily data (enough for 42-day holding period)
start_date = datetime(2024, 1, 1)
dates = pd.date_range(start=start_date, periods=180, freq="D")
dates = pd.DatetimeIndex(dates.values, freq=None)  # Remove frequency

# Create 20 synthetic assets (simulating S&P 500 subset)
assets = [f"STOCK_{i:03d}" for i in range(1, 21)]
n_assets = len(assets)
n_periods = len(dates)

# Create synthetic factor values (mean reversion signal)
# Higher factor value = better mean reversion opportunity
factor_data = []
price_data = {}

for asset in assets:
    # Factor: mean reversion signal (negative correlation with recent returns)
    # Simulate price movements
    returns = np.random.randn(n_periods) * 0.02  # 2% daily volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100

    # Factor: negative of recent returns (mean reversion)
    recent_returns = pd.Series(returns).rolling(5).mean()
    factor_values = -recent_returns.fillna(0).values + np.random.randn(n_periods) * 0.1

    # Store factor data
    for i, date in enumerate(dates):
        factor_data.append({"date": date, "asset": asset, "factor": factor_values[i]})

    # Store price data
    price_data[asset] = prices

# Convert to DataFrames
factor_df = pd.DataFrame(factor_data)
factor_df = factor_df.set_index(["date", "asset"])
factor_series = factor_df["factor"]

prices_df = pd.DataFrame(price_data, index=dates)
prices_df.index.name = "date"
prices_df.index = pd.DatetimeIndex(prices_df.index.values, freq=None)

print(f"✅ Created synthetic data:")
print(f"   - Factor data: {len(factor_df)} rows, {len(assets)} assets")
print(f"   - Price data: {len(prices_df)} rows, {len(assets)} assets")
print(f"   - Date range: {dates[0]} to {dates[-1]}")

# ============================================================================
# 2. Create Alphalens Data Structure
# ============================================================================
print("\n[2/6] Creating Alphalens data structure...")

# Holding periods (in days)
HOLDING_PERIODS = (5, 10, 21, 42)
QUANTILES = 5

# Rebuild MultiIndex to remove frequency metadata
index_tuples = list(factor_series.index)
dates_from_tuples = [t[0] for t in index_tuples]
assets_from_tuples = [t[1] for t in index_tuples]
new_dates = pd.DatetimeIndex(dates_from_tuples, freq=None)
new_index = pd.MultiIndex.from_arrays(
    [new_dates, assets_from_tuples], names=factor_series.index.names
)
factor_series = pd.Series(
    factor_series.values, index=new_index, name=factor_series.name
)

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
    alphalens_data = get_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=prices_df,
        periods=HOLDING_PERIODS,
        quantiles=QUANTILES,
        bins=None,
        binning_by_group=False,
        max_loss=0.50,  # Allow up to 50% data loss for longer holding periods
    )
    print(f"✅ Alphalens data created:")
    print(f"   - Shape: {alphalens_data.shape}")
    print(f"   - Columns: {list(alphalens_data.columns)}")
    print(f"   - Sample data:")
    print(alphalens_data.head(10))
except Exception as e:
    print(f"❌ Failed to create Alphalens data: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
finally:
    al_utils.compute_forward_returns = original_compute_forward_returns

# Save factor data sample
alphalens_data.reset_index().head(20).to_csv(
    output_dir / "factor_data_sample.csv", index=False
)
print(f"✅ Sample factor data saved to: {output_dir / 'factor_data_sample.csv'}")

# ============================================================================
# 3. Summary Tear Sheet
# ============================================================================
print("\n[3/6] Generating summary tear sheet...")


def save_figures(prefix="figure"):
    """Save all matplotlib figures with content validation"""
    saved_paths = []
    fig_nums = sorted(plt.get_fignums())

    print(f"      Found {len(fig_nums)} figure(s) to save")

    if len(fig_nums) == 0:
        print("      ⚠️  No figures found to save")
        return saved_paths

    for idx, fig_num in enumerate(fig_nums):
        fig = plt.figure(fig_num)
        axes = fig.get_axes()

        # Check if figure has content
        has_content = False
        if len(axes) > 0:
            for ax in axes:
                # Check various plot elements
                has_data = (
                    ax.has_data()
                    or len(ax.patches) > 0
                    or len(ax.lines) > 0
                    or len(ax.texts) > 0
                    or len(ax.collections) > 0
                    or len(ax.images) > 0
                    or len(ax.artists) > 0
                    or len(ax.tables) > 0
                    or len(ax.get_children()) > 0
                )
                if has_data:
                    has_content = True
                    break

        # Always try to save, even if content check is inconclusive
        if not has_content and len(axes) > 0:
            print(
                f"      ℹ️  Figure {fig_num} content check inconclusive, will attempt to save anyway..."
            )
        elif len(axes) == 0:
            print(f"      ⚠️  Figure {fig_num} has no axes, skipping...")
            plt.close(fig)
            continue

        fig_path = output_dir / f"{prefix}_fig_{idx+1}.png"
        try:
            # Force a draw to ensure the figure is rendered
            fig.canvas.draw()
            # Save with white background
            fig.savefig(
                fig_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                format="png",
            )
            saved_paths.append(str(fig_path))
            print(f"      ✅ Figure {idx+1} saved to: {fig_path}")
        except Exception as e:
            print(f"      ⚠️  Error saving figure {idx+1}: {e}")
        finally:
            plt.close(fig)

    return saved_paths


try:
    plt.close("all")
    create_summary_tear_sheet(alphalens_data)
    for fig_num in plt.get_fignums():
        plt.figure(fig_num).canvas.draw()
    saved = save_figures(prefix="summary_tear_sheet")
    print(f"✅ Summary tear sheet: {len(saved)} figures saved")
except Exception as e:
    print(f"⚠️  Summary tear sheet failed: {e}")

# ============================================================================
# 4. Returns Analysis by Quantiles
# ============================================================================
print("\n[4/6] Analyzing returns by quantiles...")

try:
    # Mean return by quantile
    mean_return_by_q, std_err = mean_return_by_quantile(alphalens_data)

    # Save to CSV
    mean_return_by_q.to_csv(output_dir / "mean_return_by_quantile.csv")
    print(f"✅ Mean returns by quantile saved")

    # Plot: Mean Return by Holding Period and Quintile
    plt.close("all")
    plot_quantile_returns_bar(mean_return_by_q)
    plt.tight_layout()
    saved = save_figures(prefix="returns_bar")
    print(f"✅ Returns bar chart: {len(saved)} figures saved")

    # Mean return by quantile (daily)
    mean_return_by_q_daily, std_err = mean_return_by_quantile(
        alphalens_data, by_date=True
    )

    # Plot: Cumulative 5D Return
    plt.close("all")
    plot_cumulative_returns_by_quantile(
        mean_return_by_q_daily["5D"], period="5D", freq=None
    )
    plt.tight_layout()
    saved = save_figures(prefix="cumulative_returns")
    print(f"✅ Cumulative returns chart: {len(saved)} figures saved")

    # Plot: Return Distribution by Holding Period and Quintile
    plt.close("all")
    plot_quantile_returns_violin(mean_return_by_q_daily)
    plt.tight_layout()
    saved = save_figures(prefix="returns_violin")
    print(f"✅ Returns violin plot: {len(saved)} figures saved")

except Exception as e:
    print(f"⚠️  Returns analysis failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# 5. Information Coefficient Analysis
# ============================================================================
print("\n[5/6] Analyzing Information Coefficient...")

try:
    # Calculate IC
    ic = factor_information_coefficient(alphalens_data)
    mean_ic = mean_information_coefficient(alphalens_data)

    # Save to CSV
    ic.to_csv(output_dir / "ic_time_series.csv")
    mean_ic.to_csv(output_dir / "ic_summary.csv")
    print(f"✅ IC data saved")

    # Plot: 5D Information Coefficient (Rolling Average)
    plt.close("all")
    plot_ic_ts(ic[["5D"]])
    plt.tight_layout()
    saved = save_figures(prefix="ic_timeseries")
    print(f"✅ IC time series chart: {len(saved)} figures saved")

    # Plot: Information Coefficient by Holding Period (Annual)
    plt.close("all")
    ic_by_year = ic.resample("A").mean()
    ic_by_year.index = ic_by_year.index.year
    ic_by_year.plot.bar(figsize=(14, 6))
    plt.title("Mean IC by Year")
    plt.tight_layout()
    saved = save_figures(prefix="ic_by_year")
    print(f"✅ IC by year chart: {len(saved)} figures saved")

except Exception as e:
    print(f"⚠️  IC analysis failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# 6. Additional Tear Sheet Visualizations
# ============================================================================
print("\n[6/6] Generating additional tear sheet visualizations...")

# Note: Some tear sheet functions may only print statistics without creating plots
# We'll create additional visualizations using plotting functions directly

# IC Heatmap
try:
    plt.close("all")
    print("   📊 Generating IC heatmap...")
    from alphalens.plotting import plot_ic_heatmap

    ic = factor_information_coefficient(alphalens_data)
    plot_ic_heatmap(ic)
    plt.tight_layout()
    saved = save_figures(prefix="ic_heatmap")
    if len(saved) > 0:
        print(f"✅ IC heatmap: {len(saved)} figures saved")
except Exception as e:
    print(f"⚠️  IC heatmap failed: {e}")

# Factor-weighted returns
try:
    plt.close("all")
    print("   📊 Generating factor-weighted returns...")
    from alphalens.plotting import plot_factor_returns

    factor_ret = factor_returns(alphalens_data)
    plot_factor_returns(factor_ret)
    plt.tight_layout()
    saved = save_figures(prefix="factor_returns")
    if len(saved) > 0:
        print(f"✅ Factor returns: {len(saved)} figures saved")
except Exception as e:
    print(f"⚠️  Factor returns plot failed: {e}")

# Try tear sheets (they may only print statistics)
print("\n   ℹ️  Note: Some tear sheet functions may only print statistics.")
print("   ℹ️  The actual charts are generated by individual plotting functions above.")

# Try returns tear sheet (for statistics)
try:
    plt.close("all")
    print("   📊 Generating returns tear sheet (statistics)...")
    create_returns_tear_sheet(
        alphalens_data, long_short=True, group_neutral=False, set_context=True
    )
    # Check if any figures were created
    if len(plt.get_fignums()) > 0:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.canvas.draw()
        saved = save_figures(prefix="returns_tear_sheet")
        if len(saved) > 0:
            print(f"✅ Returns tear sheet: {len(saved)} figures saved")
    else:
        print(f"ℹ️  Returns tear sheet: Statistics printed (no figures created)")
except Exception as e:
    print(f"⚠️  Returns tear sheet failed: {e}")

# Try information tear sheet (for statistics)
try:
    plt.close("all")
    print("   📊 Generating information tear sheet (statistics)...")
    create_information_tear_sheet(
        alphalens_data, group_neutral=False, set_context=False
    )
    if len(plt.get_fignums()) > 0:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.canvas.draw()
        saved = save_figures(prefix="information_tear_sheet")
        if len(saved) > 0:
            print(f"✅ Information tear sheet: {len(saved)} figures saved")
    else:
        print(f"ℹ️  Information tear sheet: Statistics printed (no figures created)")
except Exception as e:
    print(f"⚠️  Information tear sheet failed: {e}")

# Try turnover tear sheet (for statistics)
try:
    plt.close("all")
    print("   📊 Generating turnover tear sheet (statistics)...")
    create_turnover_tear_sheet(alphalens_data, set_context=False)
    if len(plt.get_fignums()) > 0:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.canvas.draw()
        saved = save_figures(prefix="turnover_tear_sheet")
        if len(saved) > 0:
            print(f"✅ Turnover tear sheet: {len(saved)} figures saved")
    else:
        print(f"ℹ️  Turnover tear sheet: Statistics printed (no figures created)")
except Exception as e:
    print(f"⚠️  Turnover tear sheet failed: {e}")

plt.close("all")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ Alphalens Example Complete!")
print(f"📁 All reports saved to: {output_dir.absolute()}")
print("=" * 80)

print("\n📊 Generated Files:")
print("   - CSV files: factor_data_sample.csv, mean_return_by_quantile.csv")
print("   - CSV files: ic_time_series.csv, ic_summary.csv")
print("   - PNG files: Various tear sheets and analysis charts")
print("\n🎉 Analysis complete!")
