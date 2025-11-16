"""Factor effectiveness analysis using Alphalens.

Performs:
1. IC (Information Coefficient): rank correlation between factor and future_return
2. Quantile backtest: Top vs Bottom 10% PnL difference
3. Decay analysis: How long does factor predictive power last? (crypto typically <6 hours)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

try:
    import alphalens as al
except ImportError:
    raise ImportError(
        "Alphalens is required. Install with: pip install alphalens")

from data_tools.rolling_data import load_parquet_file
from data_tools.baseline_features import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factor effectiveness analysis using Alphalens")
    parser.add_argument("--data-dir",
                        type=str,
                        default=None,
                        help="Directory containing parquet files")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help=
        "Symbol(s) metadata. Can be comma-separated (e.g., BTCUSDT,ETHUSDT,SOLUSDT) for multi-asset analysis"
    )
    parser.add_argument("--freq",
                        type=str,
                        default="5T",
                        help="Bar timeframe (e.g., 5T, 15T)")
    parser.add_argument("--start",
                        type=str,
                        default=None,
                        help="Start YYYY-MM (inclusive)")
    parser.add_argument("--end",
                        type=str,
                        default=None,
                        help="End YYYY-MM (inclusive)")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="baseline",
        help=
        "baseline/default/enhanced/hurst/wavelet/hilbert/spectral/order_flow/dl_sequence/comprehensive"
    )
    parser.add_argument("--output-dir",
                        type=str,
                        default="results/factor_analysis",
                        help="Output directory for Alphalens tear sheets")
    parser.add_argument(
        "--periods",
        type=str,
        default="1,4,24",
        help=
        "Forward return periods in bars (e.g., 1,4,24 for 15min, 1h, 6h prediction)"
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=10,
        help="Number of quantiles for quantile analysis (default: 10)")
    parser.add_argument(
        "--factor-name",
        type=str,
        default=None,
        help="Specific factor name to analyze (if None, analyzes all factors)")
    return parser.parse_args()


def _collect_files(data_dir: Optional[str], start: Optional[str],
                   end: Optional[str], symbols: str) -> List[str]:
    """Collect parquet files matching criteria."""
    if not data_dir or not os.path.exists(data_dir):
        return []

    symbol_list = [s.strip() for s in symbols.split(",")]
    files: List[str] = []

    # Symbol mapping: BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD, SOLUSDT -> SOL-USD
    symbol_map = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
    }

    for symbol in symbol_list:
        # Map symbol to file naming convention
        file_symbol = symbol_map.get(symbol, symbol)

        # Try different naming patterns
        patterns = [
            f"{file_symbol}_*.parquet",  # BTC-USD_2024-11.parquet
            f"{symbol}-aggTrades-*.parquet",  # BTCUSDT-aggTrades-2024-10.parquet
            f"{symbol}_*.parquet",  # BTCUSDT_2024-11.parquet
            f"{symbol}-*.parquet",  # BTCUSDT-2024-11.parquet
        ]

        for pattern in patterns:
            for file in Path(data_dir).glob(pattern):
                if file.is_file():
                    file_str = str(file.name)

                    # Filter by date if specified
                    if start or end:
                        # Extract date from filename
                        # Formats: BTC-USD_2024-11.parquet, BTCUSDT-aggTrades-2024-10.parquet
                        file_date = None

                        # Try pattern: SYMBOL_YYYY-MM.parquet
                        if "_" in file_str:
                            parts = file_str.split("_")
                            if len(parts) >= 2:
                                date_part = parts[-1].replace(".parquet", "")
                                if "-" in date_part and len(
                                        date_part) == 7:  # YYYY-MM
                                    file_date = date_part

                        # Try pattern: SYMBOL-aggTrades-YYYY-MM.parquet
                        if file_date is None and "-" in file_str:
                            parts = file_str.split("-")
                            if len(parts) >= 3:
                                try:
                                    # Last two parts should be YYYY and MM.parquet
                                    year = parts[-2]
                                    month = parts[-1].replace(".parquet", "")
                                    if len(year) == 4 and len(month) == 2:
                                        file_date = f"{year}-{month}"
                                except Exception:
                                    pass

                        # Filter by date
                        if file_date:
                            if start and file_date < start:
                                continue
                            if end and file_date > end:
                                continue
                        # If we can't parse date but date filter is specified, skip
                        elif start or end:
                            continue

                    files.append(str(file))

    return sorted(list(set(files)))


def load_and_prepare_data(files: List[str], freq: str,
                          feature_type: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare data with features."""
    frames: List[pd.DataFrame] = []

    # Reverse symbol mapping: BTC-USD -> BTCUSDT, ETH-USD -> ETHUSDT, SOL-USD -> SOLUSDT
    reverse_symbol_map = {
        "BTC-USD": "BTCUSDT",
        "ETH-USD": "ETHUSDT",
        "SOL-USD": "SOLUSDT",
    }

    for f in files:
        df = load_parquet_file(f) if f.endswith(".parquet") else None
        if df is not None and len(df) > 0:
            # Resample if needed
            if freq:
                df = df.resample(freq).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()

            # Add symbol column if multi-asset
            if "symbol" not in df.columns:
                # Try to infer from filename
                fname = os.path.basename(f)
                symbol = None

                # Try to match file symbol format (BTC-USD, ETH-USD, SOL-USD)
                for file_symbol, symbol_name in reverse_symbol_map.items():
                    if file_symbol in fname:
                        symbol = symbol_name
                        break

                # Fallback to simple matching
                if symbol is None:
                    if "BTC" in fname.upper():
                        symbol = "BTCUSDT"
                    elif "ETH" in fname.upper():
                        symbol = "ETHUSDT"
                    elif "SOL" in fname.upper():
                        symbol = "SOLUSDT"
                    else:
                        symbol = "UNKNOWN"

                df["symbol"] = symbol

            frames.append(df)

    if not frames:
        raise ValueError("No data files found or loaded")

    # Combine all frames
    combined = pd.concat(frames, axis=0)
    combined = combined.sort_index()

    # Preserve symbol before engineering (some pipelines drop non-feature columns)
    if "symbol" in combined.columns:
        preserved_symbol = combined["symbol"].copy()
    else:
        preserved_symbol = None

    # Engineer features
    print(f"   🧪 Engineering {feature_type} features...")
    if feature_type == "baseline":
        combined_eng, _ = engineer_baseline_features(combined, None, fit=True)
        feature_cols = get_baseline_feature_columns(combined_eng)
    else:
        engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
        combined_eng = engineer.engineer_features(combined, fit=True)
        feature_cols = get_feature_columns_by_type(combined_eng, feature_type)

    # Reattach symbol column after engineering
    if preserved_symbol is not None:
        # Always ensure symbol column exists in the final DataFrame
        if "symbol" not in combined_eng.columns:
            try:
                # Try direct reindexing first
                combined_eng["symbol"] = preserved_symbol.reindex(
                    combined_eng.index, method="ffill").bfill()
            except Exception:
                # Fallback approach: merge on timestamp
                try:
                    sym_df = preserved_symbol.to_frame(name="symbol")
                    sym_df["timestamp"] = sym_df.index
                    tmp = combined_eng.copy()
                    tmp["timestamp"] = tmp.index
                    combined_eng = tmp.merge(
                        sym_df.drop_duplicates(subset=["timestamp"]),
                        on="timestamp",
                        how="left").set_index("timestamp")
                except Exception:
                    # Last resort: fill with UNKNOWN
                    combined_eng["symbol"] = "UNKNOWN"
        else:
            # If symbol column already exists but might have NaN values, fill them
            if combined_eng["symbol"].isna().any():
                combined_eng["symbol"] = combined_eng["symbol"].fillna(
                    "UNKNOWN")
    else:
        # If no symbol was preserved, add a default one
        if "symbol" not in combined_eng.columns:
            combined_eng["symbol"] = "UNKNOWN"

    # Ensure symbol column is of string type
    combined_eng["symbol"] = combined_eng["symbol"].astype(str)

    # Use engineered DataFrame going forward
    combined = combined_eng

    print(f"   ✅ Generated {len(feature_cols)} features")

    return combined_eng, feature_cols


def prepare_alphalens_data(df: pd.DataFrame, factor_col: str,
                           periods: List[int],
                           freq_str: Optional[str]) -> pd.DataFrame:
    """Prepare data for Alphalens: multi-index [symbol, timestamp]."""
    # Ensure we have symbol column
    if "symbol" not in df.columns:
        raise ValueError("DataFrame must have 'symbol' column")

    # Work directly with the DataFrame to avoid timezone issues
    # Extract factor and prices before any index manipulation
    if factor_col not in df.columns:
        raise ValueError(
            f"Factor column '{factor_col}' not found in DataFrame")

    # Rename columns to match Alphalens expectations
    df_renamed = df.rename(columns={"symbol": "asset"})
    # Ensure index name is 'date' for Alphalens compatibility
    df_renamed.index.name = 'date'

    # Get factor and prices series with original index
    factor_series = df_renamed[factor_col].copy()
    prices_series = df_renamed["close"].copy()
    assets = df_renamed["asset"].copy()

    # Use the DataFrame's index as dates
    dates = df_renamed.index

    # Ensure date is datetime and timezone-naive for Alphalens compatibility
    dates = pd.to_datetime(dates)
    # Remove timezone information if present for Alphalens compatibility
    if hasattr(dates, 'tz') and dates.tz is not None:
        dates = dates.tz_localize(None)
    # Drop explicit frequency metadata to keep Alphalens from forcing a
    # CustomBusinessDay calendar on intraday timestamps
    try:
        dates.freq = None  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        pass

    # Remove any NaN or inf values from factor and align all series
    # First, identify valid (non-NaN, non-inf) factor values
    valid_mask = np.isfinite(factor_series)

    # Check if we have any valid data
    if not valid_mask.any():
        raise ValueError(
            "No valid data points found after removing NaN/inf values")

    # Align all series to have the same valid entries
    factor_values = factor_series[valid_mask]
    prices_values = prices_series[valid_mask]
    assets_values = assets[valid_mask]
    dates_values = dates[valid_mask]

    # Create MultiIndex with correct names for Alphalens
    # Alphalens expects 'date' as first level and 'asset' as second level
    date_index = pd.DatetimeIndex(dates_values)
    if hasattr(date_index, 'tz') and date_index.tz is not None:
        date_index = date_index.tz_localize(None)
    try:
        date_index.freq = None  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        pass

    # Ensure the asset level is also a proper Index
    asset_index = pd.Index(assets_values)

    # Create MultiIndex with correct names: 'date' first, then 'asset'
    multi_index = pd.MultiIndex.from_arrays([date_index, asset_index],
                                            names=['date', 'asset'])

    # Parse frequency - but don't set it on MultiIndex for intraday data
    # Alphalens has issues with intraday frequencies, so we'll let it infer
    freq_offset = None
    is_intraday = False
    if freq_str:
        try:
            freq_offset = to_offset(freq_str)
            # Detect if this is intraday data (less than 1 day)
            # Common intraday patterns: 5T, 15T, 60T, 240T, etc.
            # Simple heuristic: if it contains 'T' (minutes) or 'H' (hours), it's likely intraday
            # Also check if the timedelta is less than 1 day
            freq_lower = freq_str.upper()
            if 'T' in freq_lower or 'H' in freq_lower:
                # Try to parse and compare with 1 day
                try:
                    test_timedelta = pd.Timedelta(freq_str)
                    one_day = pd.Timedelta(days=1)
                    is_intraday = test_timedelta < one_day
                except Exception:
                    # If parsing fails, assume intraday if it has 'T' or 'H'
                    is_intraday = True
            else:
                # For other formats (like 'D' for daily), check timedelta
                try:
                    test_timedelta = pd.Timedelta(freq_str)
                    one_day = pd.Timedelta(days=1)
                    is_intraday = test_timedelta < one_day
                except Exception:
                    # Default to False if we can't determine
                    is_intraday = False
        except (TypeError, ValueError):
            print(
                f"      ⚠️  Could not parse freq '{freq_str}' into pandas offset; defaulting to inference"
            )

    # Create factor Series with the new MultiIndex
    # Don't set freq on MultiIndex for intraday data - Alphalens will handle it
    factor = pd.Series(factor_values.values,
                       index=multi_index,
                       name=factor_col)

    # Create a DataFrame for prices with the proper MultiIndex structure
    # Alphalens expects prices to be a DataFrame with assets as columns and dates as index
    prices_df_pivot = pd.DataFrame({
        'date_col': dates_values,  # Use a different name to avoid ambiguity
        'asset': assets_values,
        'close': prices_values
    })

    # Pivot the DataFrame to have assets as columns (as Alphalens expects)
    prices = prices_df_pivot.pivot_table(values='close',
                                         index='date_col',
                                         columns='asset',
                                         aggfunc='first')

    # Rename the index to 'date' after pivot
    prices.index.name = 'date'

    # Ensure the index is timezone-naive DatetimeIndex
    prices.index = pd.DatetimeIndex(prices.index)
    if hasattr(prices.index, 'tz') and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    # For intraday data, Alphalens has issues with frequency validation
    # We'll remove freq attribute completely to avoid the error
    # Create a new index without freq
    prices.index = pd.DatetimeIndex(prices.index.values, freq=None)

    # Ensure prices DataFrame is properly sorted and filled
    prices = prices.sort_index().ffill().bfill()

    # Align factor and prices to have common assets
    common_assets = sorted(
        set(factor.index.get_level_values("asset")) & set(prices.columns))
    if not common_assets:
        raise ValueError("No common assets between factor and prices")

    prices = prices[common_assets]
    factor = factor.loc[factor.index.get_level_values("asset").isin(
        common_assets)]

    # Check if factor has enough unique values
    if factor.nunique() < 5:
        raise ValueError(
            f"Factor contains only {factor.nunique()} unique values, need at least 5 for quantile analysis"
        )

    # For intraday data, Alphalens has issues with frequency validation
    # We need to monkey-patch or work around the frequency setting in Alphalens
    # The issue is in alphalens/utils.py line 358 where it tries to set freq
    # Let's create a wrapper that prevents freq from being set

    # Prepare Alphalens data with workaround for frequency issues
    # Limit periods to avoid date overflow (Alphalens calculates future dates)
    max_period = max(periods) if periods else 1
    # Check if we have enough data for the maximum period
    if len(prices) < max_period + 10:
        raise ValueError(
            f"Not enough data: {len(prices)} rows, need at least {max_period + 10} for period {max_period}"
        )

    # For intraday data, Alphalens has known issues with frequency validation
    # Use custom workaround directly instead of trying Alphalens first
    if is_intraday:
        print(
            f"      ℹ️  Using custom forward returns calculation for intraday data (freq={freq_str})"
        )
        # Skip Alphalens attempt and go directly to workaround
        use_workaround = True
    else:
        # For daily or longer frequencies, try Alphalens first
        use_workaround = False
        # Try to prepare data - if it fails due to frequency, we'll use a workaround
        # Monkey-patch Alphalens to avoid frequency validation errors
        import alphalens.utils as al_utils
        original_compute_forward_returns = al_utils.compute_forward_returns

        def patched_compute_forward_returns(factor,
                                            prices,
                                            periods,
                                            filter_zscore=None,
                                            cumulative_returns=True):
            """Patched version that skips frequency validation for intraday data"""
            try:
                return original_compute_forward_returns(
                    factor, prices, periods, filter_zscore, cumulative_returns)
            except (ValueError, OverflowError) as e:
                if "frequency" in str(e).lower() or "overflow" in str(
                        e).lower():
                    # Remove freq from MultiIndex to avoid validation
                    if hasattr(factor.index, 'levels') and len(
                            factor.index.levels) > 0:
                        try:
                            # Create new index without freq
                            date_level = factor.index.levels[0]
                            if hasattr(date_level,
                                       'freq') and date_level.freq is not None:
                                new_date_level = pd.DatetimeIndex(
                                    date_level.values, freq=None)
                                new_index = pd.MultiIndex.from_arrays(
                                    [new_date_level, factor.index.levels[1]],
                                    names=factor.index.names)
                                factor = factor.reindex(new_index)
                        except Exception:
                            pass
                    # Retry
                    return original_compute_forward_returns(
                        factor, prices, periods, filter_zscore,
                        cumulative_returns)
                raise

        # Apply monkey-patch
        al_utils.compute_forward_returns = patched_compute_forward_returns

        try:
            factor_data = al.utils.get_clean_factor_and_forward_returns(
                factor=factor,
                prices=prices,
                periods=periods,
                quantiles=10,  # Use 10 quantiles for factor analysis
                bins=None,
                binning_by_group=False,
                max_loss=
                0.99,  # Allow up to 99% loss to avoid dropping too much data
            )
            use_workaround = False  # Success, no need for workaround
        except (ValueError, OverflowError, TypeError) as e:
            # Restore original function
            al_utils.compute_forward_returns = original_compute_forward_returns
            error_str = str(e).lower()
            if "frequency" in error_str or "overflow" in error_str or "cannot cast" in error_str:
                use_workaround = True
                print(
                    f"      ℹ️  Using custom forward returns calculation (Alphalens frequency issue detected)"
                )
            else:
                raise

    if use_workaround:
        # Manually compute forward returns without frequency validation
        from alphalens.utils import quantize_factor

        # quantize_factor expects a DataFrame with 'factor' column, not a Series
        # Convert factor Series to DataFrame
        # Ensure the index is properly structured as MultiIndex [date, asset]
        factor_df = factor.to_frame(name='factor')

        # Verify and fix index structure if needed
        if not isinstance(factor_df.index, pd.MultiIndex):
            raise ValueError(
                f"factor_df must have MultiIndex, got {type(factor_df.index)}")
        if len(factor_df.index.names) != 2:
            raise ValueError(
                f"factor_df index must have 2 levels, got {len(factor_df.index.names)}: {factor_df.index.names}"
            )
        # Ensure index names are correct
        if factor_df.index.names != ['date', 'asset']:
            factor_df.index.names = ['date', 'asset']

        # Check if factor has enough variation to be useful
        factor_values = factor_df['factor'].dropna()
        if len(factor_values) == 0:
            raise ValueError("Factor has no valid values")

        unique_ratio = factor_values.nunique() / len(factor_values)
        if unique_ratio < 0.001:  # Less than 0.1% unique values (very strict)
            # For factors with extremely low uniqueness, skip detailed analysis
            # but still allow basic statistics
            print(
                f"      ⚠️  Warning: Factor has very few unique values ({unique_ratio:.2%} unique)"
            )
            print(
                f"         This factor may have limited predictive power, but will attempt basic analysis"
            )
            # Don't raise error, just continue with warning
        elif unique_ratio < 0.01:  # Less than 1% unique values
            print(
                f"      ℹ️  Note: Factor has relatively few unique values ({unique_ratio:.2%} unique)"
            )
            print(
                f"         This is normal for some factors (e.g., time-based factors), continuing analysis..."
            )

        # Quantize factor first (check alphalens version for correct parameters)
        # If factor has too many duplicate values, try fewer quantiles or use bins
        quantized_factor = None
        last_error = None

        # Strategy 1: Try with fewer quantiles (5 instead of 10)
        strategies = [
            # (quantiles, bins, description)
            (5, None, "5 quantiles"),
            (3, None, "3 quantiles"),
            (None, 5, "5 bins"),
            (None, 3, "3 bins"),
        ]

        for quantiles_val, bins_val, desc in strategies:
            try:
                print(f"      Trying {desc}...")
                # Try with all parameters first
                try:
                    quantized_factor = quantize_factor(
                        factor_df,
                        quantiles=quantiles_val,
                        bins=bins_val,
                        binning_by_group=False,
                        max_loss=0.99,
                    )
                    print(f"      ✅ Success with {desc}")
                    break
                except TypeError:
                    # Try without binning_by_group
                    try:
                        quantized_factor = quantize_factor(
                            factor_df,
                            quantiles=quantiles_val,
                            bins=bins_val,
                            max_loss=0.99,
                        )
                        print(
                            f"      ✅ Success with {desc} (no binning_by_group)"
                        )
                        break
                    except TypeError:
                        # Try with minimal parameters
                        quantized_factor = quantize_factor(
                            factor_df,
                            quantiles=quantiles_val,
                            bins=bins_val,
                        )
                        print(f"      ✅ Success with {desc} (minimal params)")
                        break
            except (ValueError, TypeError) as e:
                last_error = e
                error_str = str(e).lower()
                if "bin edges" in error_str or "duplicate" in error_str:
                    # Continue to next strategy
                    continue
                elif isinstance(e, TypeError):
                    # Parameter error, continue
                    continue
                else:
                    # Unexpected error
                    raise

        if quantized_factor is None:
            raise ValueError(
                f"Could not quantize factor after trying all strategies. "
                f"Last error: {last_error}. "
                f"Factor may have too many duplicate values or insufficient variation."
            )

        # Manually compute forward returns for each period
        forward_returns_dict = {}
        for period in periods:
            # Calculate forward returns manually: (future_price / current_price) - 1
            # Shift prices forward by period, then calculate return
            future_prices = prices.shift(-period)
            forward_ret = (future_prices / prices) - 1
            forward_returns_dict[period] = forward_ret

        # Combine into MultiIndex DataFrame
        forward_returns_list = []
        for period, fwd_ret in forward_returns_dict.items():
            fwd_ret_stacked = fwd_ret.stack()
            # Ensure the stacked index has exactly 2 levels: [date, asset]
            if isinstance(fwd_ret_stacked.index, pd.MultiIndex):
                if len(fwd_ret_stacked.index.names) != 2:
                    # Reconstruct MultiIndex if it has wrong structure
                    dates = fwd_ret_stacked.index.get_level_values(0)
                    assets = fwd_ret_stacked.index.get_level_values(
                        -1)  # Get last level as asset
                    fwd_ret_stacked.index = pd.MultiIndex.from_arrays(
                        [dates, assets], names=['date', 'asset'])
                else:
                    fwd_ret_stacked.index.names = ['date', 'asset']
            else:
                raise ValueError(
                    f"Stacked forward returns should have MultiIndex, got {type(fwd_ret_stacked.index)}"
                )
            fwd_ret_stacked.name = period
            forward_returns_list.append(fwd_ret_stacked)

        forward_returns = pd.concat(forward_returns_list, axis=1)
        # Alphalens expects forward returns columns to be strings (or integers that can be converted)
        # Convert periods to strings to match Alphalens expectations
        forward_returns.columns = [str(p) for p in periods]

        # Ensure forward_returns has correct MultiIndex structure
        if not isinstance(forward_returns.index, pd.MultiIndex):
            raise ValueError(
                f"forward_returns must have MultiIndex, got {type(forward_returns.index)}"
            )
        if len(forward_returns.index.names) != 2:
            raise ValueError(
                f"forward_returns index must have 2 levels, got {len(forward_returns.index.names)}"
            )
        forward_returns.index.names = ['date', 'asset']

        # quantize_factor returns a DataFrame with 'factor_quantile' column (or Series)
        # Alphalens expects 'factor_quantile' column, not 'factor'
        # IMPORTANT: quantize_factor may return a different index structure
        # We need to ensure it matches the original factor index
        if isinstance(quantized_factor, pd.Series):
            # If Series, convert to DataFrame with correct column name
            quantized_factor_df = quantized_factor.to_frame(
                name='factor_quantile')
        elif isinstance(quantized_factor, pd.DataFrame):
            # Check if it has 'factor_quantile' column
            if 'factor_quantile' in quantized_factor.columns:
                quantized_factor_df = quantized_factor[['factor_quantile']]
            elif 'factor' in quantized_factor.columns:
                # Rename 'factor' to 'factor_quantile'
                quantized_factor_df = quantized_factor[[
                    'factor'
                ]].rename(columns={'factor': 'factor_quantile'})
            else:
                # Take first column and rename
                quantized_factor_df = quantized_factor.iloc[:, [0]].copy()
                quantized_factor_df.columns = ['factor_quantile']
        else:
            raise ValueError(
                f"Unexpected quantized_factor type: {type(quantized_factor)}")

        # CRITICAL: Ensure quantized_factor_df has the same MultiIndex structure as factor
        # quantize_factor should preserve the index, but we need to verify and fix if needed
        if not isinstance(quantized_factor_df.index, pd.MultiIndex):
            # If it's not a MultiIndex, this is unexpected - try to reconstruct from factor_df
            print(
                f"      ⚠️  quantize_factor returned non-MultiIndex, reconstructing from factor index"
            )
            quantized_factor_df = quantized_factor_df.reindex(factor_df.index)
        elif len(quantized_factor_df.index.names) != 2:
            # If it has wrong number of levels, this is a problem
            print(
                f"      ⚠️  quantize_factor returned {len(quantized_factor_df.index.names)} levels, expected 2. Reindexing..."
            )
            # Try to extract the first two levels if there are more
            if len(quantized_factor_df.index.names) > 2:
                # Take first level as date, last level as asset
                dates = quantized_factor_df.index.get_level_values(0)
                assets = quantized_factor_df.index.get_level_values(-1)
                new_index = pd.MultiIndex.from_arrays([dates, assets],
                                                      names=['date', 'asset'])
                quantized_factor_df.index = new_index
            else:
                # Reindex from factor_df
                quantized_factor_df = quantized_factor_df.reindex(
                    factor_df.index)
        else:
            # Ensure index names are correct
            quantized_factor_df.index.names = ['date', 'asset']

        # Align with factor_df index to ensure exact match
        # Only keep rows that exist in both
        common_idx = quantized_factor_df.index.intersection(factor_df.index)
        if len(common_idx) == 0:
            raise ValueError(
                "No common index between quantized_factor and original factor after processing"
            )
        quantized_factor_df = quantized_factor_df.loc[common_idx]

        # Align factor and forward returns on MultiIndex
        # Both should have [date, asset] MultiIndex with exactly 2 levels
        # Get common index, ensuring both have proper 2-level MultiIndex
        common_dates = quantized_factor_df.index.get_level_values(
            'date').intersection(
                forward_returns.index.get_level_values('date'))
        common_assets = quantized_factor_df.index.get_level_values(
            'asset').intersection(
                forward_returns.index.get_level_values('asset'))

        # Create proper MultiIndex from common dates and assets
        # Use only combinations that exist in both
        factor_index_set = set(quantized_factor_df.index)
        forward_index_set = set(forward_returns.index)
        common_index_set = factor_index_set & forward_index_set

        if len(common_index_set) == 0:
            raise ValueError(
                "No common index between factor and forward returns. "
                f"Factor index: {len(quantized_factor_df.index)}, "
                f"Forward returns index: {len(forward_returns.index)}")

        # Create MultiIndex from common tuples, ensuring proper structure
        common_tuples = list(common_index_set)
        # Verify all tuples have exactly 2 elements (date, asset)
        if any(len(t) != 2 for t in common_tuples):
            # Filter out invalid tuples
            common_tuples = [t for t in common_tuples if len(t) == 2]

        # Reconstruct MultiIndex properly
        dates_list = [t[0] for t in common_tuples]
        assets_list = [t[1] for t in common_tuples]
        common_index = pd.MultiIndex.from_arrays([dates_list, assets_list],
                                                 names=['date', 'asset'])
        common_index = common_index.sort_values()

        factor_aligned = quantized_factor_df.loc[common_index]
        forward_aligned = forward_returns.loc[common_index]

        # Combine into final DataFrame
        # Alphalens expects both 'factor' (original values) and 'factor_quantile' columns
        # We need to add the original factor values back
        factor_aligned_with_original = factor_aligned.copy()

        # Get original factor values for the common index
        # CRITICAL: Ensure factor has proper 2-level MultiIndex before indexing
        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError(
                f"factor must have MultiIndex, got {type(factor.index)}")
        if len(factor.index.names) != 2:
            raise ValueError(
                f"factor index must have 2 levels, got {len(factor.index.names)}: {factor.index.names}"
            )

        # Ensure factor index names are correct
        if factor.index.names != ['date', 'asset']:
            factor.index.names = ['date', 'asset']

        # Extract original factor values, ensuring alignment
        try:
            original_factor_series = factor.loc[common_index]
        except KeyError:
            # If direct indexing fails, try reindexing
            original_factor_series = factor.reindex(common_index)

        # CRITICAL: Verify original_factor_series has correct index structure
        if not isinstance(original_factor_series.index, pd.MultiIndex):
            raise ValueError(
                f"original_factor_series must have MultiIndex, got {type(original_factor_series.index)}"
            )
        if len(original_factor_series.index.names) != 2:
            raise ValueError(
                f"original_factor_series index must have 2 levels, got {len(original_factor_series.index.names)}"
            )

        # Ensure the index matches common_index exactly
        if not original_factor_series.index.equals(common_index):
            # Reindex to match common_index exactly
            original_factor_series = original_factor_series.reindex(
                common_index)

        # Ensure common_index is a proper MultiIndex with correct names
        if not isinstance(common_index, pd.MultiIndex):
            raise ValueError(
                f"common_index should be MultiIndex, got {type(common_index)}")
        if len(common_index.names) != 2:
            raise ValueError(
                f"common_index must have 2 levels, got {len(common_index.names)}"
            )

        # Ensure index names are correct
        if common_index.names != ['date', 'asset']:
            common_index.names = ['date', 'asset']

        # Create DataFrame with proper MultiIndex structure
        # Alphalens expects: 'factor' (original), 'factor_quantile', and period columns
        # CRITICAL: Use .values to avoid any index alignment issues
        factor_data_dict = {
            'factor': original_factor_series.values,
            'factor_quantile': factor_aligned['factor_quantile'].values
        }

        # Add forward returns columns (ensure they're strings)
        for period_col in forward_aligned.columns:
            factor_data_dict[str(
                period_col)] = forward_aligned[period_col].values

        # Create DataFrame with proper MultiIndex
        # CRITICAL: Reconstruct the index to ensure it's clean and has exactly 2 levels
        # Extract dates and assets from common_index to create a fresh MultiIndex
        dates_clean = common_index.get_level_values('date')
        assets_clean = common_index.get_level_values('asset')

        # Ensure dates are Timestamps (not tuples or other structures)
        if not isinstance(dates_clean, pd.DatetimeIndex):
            dates_clean = pd.to_datetime(dates_clean)

        # Create a fresh, clean MultiIndex
        clean_index = pd.MultiIndex.from_arrays([dates_clean, assets_clean],
                                                names=['date', 'asset'])

        # Create DataFrame with the clean index
        factor_data = pd.DataFrame(factor_data_dict, index=clean_index)

        # Final verification: Ensure index is properly formatted with exactly 2 levels
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError(
                f"factor_data must have MultiIndex, got {type(factor_data.index)}"
            )
        if len(factor_data.index.names) != 2:
            raise ValueError(
                f"factor_data index must have 2 levels, got {len(factor_data.index.names)}: {factor_data.index.names}"
            )
        factor_data.index.names = ['date', 'asset']

        # Verify all index tuples have exactly 2 elements
        sample_tuples = list(factor_data.index[:10])
        if any(len(t) != 2 for t in sample_tuples):
            # This is a critical error - we need to fix the index structure
            print(f"      ⚠️  Found invalid index tuples: {sample_tuples[:5]}")
            # Try to reconstruct the index from level values
            dates_fixed = factor_data.index.get_level_values(0)
            assets_fixed = factor_data.index.get_level_values(
                -1)  # Get last level
            # Ensure dates are proper Timestamps
            if not isinstance(dates_fixed, pd.DatetimeIndex):
                dates_fixed = pd.to_datetime(dates_fixed)
            factor_data.index = pd.MultiIndex.from_arrays(
                [dates_fixed, assets_fixed], names=['date', 'asset'])
            # Verify again
            sample_tuples = list(factor_data.index[:10])
            if any(len(t) != 2 for t in sample_tuples):
                raise ValueError(
                    f"Failed to fix index structure. Tuples still have wrong length: {sample_tuples[:5]}"
                )

        # Remove any rows with NaN
        factor_data = factor_data.dropna()

        if len(factor_data) == 0:
            raise ValueError("No valid data after alignment")

        # Verify structure - Alphalens expects both 'factor' and 'factor_quantile' columns
        if 'factor_quantile' not in factor_data.columns:
            raise ValueError(
                "'factor_quantile' column missing from final factor_data. "
                f"Available columns: {list(factor_data.columns)}")
        if 'factor' not in factor_data.columns:
            raise ValueError("'factor' column missing from final factor_data. "
                             f"Available columns: {list(factor_data.columns)}")

        # Final validation: Ensure index structure is correct before returning
        # Check that all index tuples are proper (date, asset) pairs
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError(
                f"factor_data.index must be MultiIndex, got {type(factor_data.index)}"
            )
        if len(factor_data.index.names) != 2:
            raise ValueError(
                f"factor_data.index must have 2 levels, got {len(factor_data.index.names)}"
            )
        if factor_data.index.names != ['date', 'asset']:
            factor_data.index.names = ['date', 'asset']

        # Verify level 0 (date) is DatetimeIndex
        date_level = factor_data.index.get_level_values(0)
        if not isinstance(date_level, pd.DatetimeIndex):
            # Try to convert
            dates_converted = pd.to_datetime(date_level)
            assets_level = factor_data.index.get_level_values(1)
            factor_data.index = pd.MultiIndex.from_arrays(
                [dates_converted, assets_level], names=['date', 'asset'])

        # Debug: Print index structure
        print(f"      Factor data index type: {type(factor_data.index)}")
        print(f"      Factor data index names: {factor_data.index.names}")
        print(
            f"      Factor data index levels: {len(factor_data.index.levels)}")
        sample_idx = factor_data.index[:3] if len(factor_data) > 0 else 'empty'
        print(f"      Factor data index sample: {sample_idx}")
        # Verify sample tuples
        if len(factor_data) > 0:
            sample_tuples_check = [tuple(idx) for idx in factor_data.index[:3]]
            print(
                f"      Factor data index tuples (first 3): {sample_tuples_check}"
            )
            if any(len(t) != 2 for t in sample_tuples_check):
                raise ValueError(
                    f"Index tuples must have 2 elements, found: {sample_tuples_check}"
                )

    return factor_data


def _fix_factor_data_index(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Fix factor_data index if it has wrong structure (e.g., 3 levels instead of 2)."""
    if not isinstance(factor_data.index, pd.MultiIndex):
        return factor_data

    # Check if index has wrong number of levels
    if factor_data.index.nlevels != 2:
        print(
            f"      ⚠️  WARNING: factor_data has {factor_data.index.nlevels} levels, expected 2"
        )
        print(
            f"         Attempting to fix by extracting first and last levels..."
        )

        # Extract first level (date) and last level (asset)
        dates = factor_data.index.get_level_values(0)
        assets = factor_data.index.get_level_values(-1)

        # Ensure dates are Timestamps
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)

        # Create new clean MultiIndex
        new_index = pd.MultiIndex.from_arrays([dates, assets],
                                              names=['date', 'asset'])
        factor_data = factor_data.copy()
        factor_data.index = new_index

        print(
            f"      ✅ Fixed index structure: now has {factor_data.index.nlevels} levels"
        )

    # Verify all tuples have exactly 2 elements
    sample_tuples = [tuple(idx) for idx in factor_data.index[:10]]
    if any(len(t) != 2 for t in sample_tuples):
        print(f"      ⚠️  WARNING: Some index tuples have wrong length")
        # Reconstruct index from level values
        dates = factor_data.index.get_level_values(0)
        assets = factor_data.index.get_level_values(-1)
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        new_index = pd.MultiIndex.from_arrays([dates, assets],
                                              names=['date', 'asset'])
        factor_data = factor_data.copy()
        factor_data.index = new_index
        print(f"      ✅ Reconstructed index from level values")

    return factor_data


def _compute_manual_statistics(factor_data: pd.DataFrame, factor_name: str,
                               output_dir: str) -> None:
    """Compute and print basic statistics manually when Alphalens fails."""
    print(f"\n      📊 Manual Statistics for {factor_name}:")
    print("      " + "=" * 60)

    # Basic factor statistics
    if 'factor' in factor_data.columns:
        factor_values = factor_data['factor'].dropna()
        print(f"      Factor Statistics:")
        print(f"        Count: {len(factor_values)}")
        print(f"        Mean: {factor_values.mean():.6f}")
        print(f"        Std: {factor_values.std():.6f}")
        print(f"        Min: {factor_values.min():.6f}")
        print(f"        Max: {factor_values.max():.6f}")
        print(
            f"        Unique values: {factor_values.nunique()} ({factor_values.nunique()/len(factor_values)*100:.2f}%)"
        )

    # Quantile statistics
    if 'factor_quantile' in factor_data.columns:
        quantile_counts = factor_data['factor_quantile'].value_counts(
        ).sort_index()
        print(f"\n      Quantile Distribution:")
        for q, count in quantile_counts.items():
            pct = count / len(factor_data) * 100
            print(f"        Quantile {q}: {count} ({pct:.2f}%)")

    # Forward returns statistics
    forward_cols = [
        col for col in factor_data.columns
        if col not in ['factor', 'factor_quantile']
    ]

    # DEBUG: Check forward columns
    if not forward_cols:
        print(
            f"      ⚠️  WARNING: No forward return columns found in factor_data"
        )
        print(f"         Available columns: {list(factor_data.columns)}")
        print(f"         This means IC/IR cannot be calculated")

    if forward_cols:
        print(f"\n      Forward Returns Statistics:")
        for col in forward_cols:
            returns = factor_data[col].dropna()
            if len(returns) > 0:
                print(f"        Period {col}:")
                print(f"          Mean: {returns.mean():.6f}")
                print(f"          Std: {returns.std():.6f}")
                print(f"          Min: {returns.min():.6f}")
                print(f"          Max: {returns.max():.6f}")

    # Simple IC calculation (Spearman correlation) and IR
    ic_results = {}
    if 'factor' in factor_data.columns and forward_cols:
        print(
            f"\n      🔍 DEBUG: Calculating IC/IR for {len(forward_cols)} periods..."
        )
        print(
            f"\n      Information Coefficient (IC) and Information Ratio (IR):"
        )
        print(
            f"        {'Period':<10} {'IC':<12} {'IC Mean':<12} {'IC Std':<12} {'IR':<12}"
        )
        print(f"        {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for col in forward_cols:
            try:
                # Calculate rank correlation between factor and forward returns
                factor_vals = factor_data['factor'].dropna()
                returns_vals = factor_data[col].dropna()
                common_idx = factor_vals.index.intersection(returns_vals.index)
                if len(common_idx) > 10:
                    factor_common = factor_data.loc[common_idx, 'factor']
                    returns_common = factor_data.loc[common_idx, col]

                    # Calculate IC (Spearman correlation)
                    ic = factor_common.corr(returns_common, method='spearman')

                    # Calculate IC over time (rolling IC for IR calculation)
                    # Group by date to get IC per period
                    ic_by_date = []
                    for date in factor_data.index.get_level_values(
                            'date').unique():
                        date_mask = factor_data.index.get_level_values(
                            'date') == date
                        date_data = factor_data.loc[date_mask]
                        if len(date_data) > 5:  # Need enough data points
                            try:
                                date_ic = date_data['factor'].corr(
                                    date_data[col], method='spearman')
                                if not np.isnan(date_ic):
                                    ic_by_date.append(date_ic)
                            except Exception:
                                pass

                    # Calculate IR (mean IC / std IC)
                    if len(ic_by_date) > 1:
                        ic_mean = np.mean(ic_by_date)
                        ic_std = np.std(ic_by_date)
                        ir = ic_mean / ic_std if ic_std > 0 else np.nan
                    else:
                        ic_mean = ic
                        ic_std = np.nan
                        ir = np.nan

                    ic_results[col] = {
                        'IC': ic,
                        'IC_Mean': ic_mean,
                        'IC_Std': ic_std,
                        'IR': ir
                    }

                    print(
                        f"        {col:<10} {ic:>12.6f} {ic_mean:>12.6f} {ic_std:>12.6f} {ir:>12.6f}"
                    )
            except Exception as e:
                print(f"        Period {col}: Error calculating IC/IR: {e}")
                import traceback
                traceback.print_exc()
                ic_results[col] = {
                    'IC': np.nan,
                    'IC_Mean': np.nan,
                    'IC_Std': np.nan,
                    'IR': np.nan
                }

    # DEBUG: Check if IC results were calculated
    if not ic_results:
        print(f"      ⚠️  WARNING: No IC/IR results calculated")
        print(f"         This may be because:")
        print(f"         1. No forward return columns found")
        print(f"         2. IC calculation failed for all periods")
        print(f"         3. Insufficient data for correlation calculation")
    else:
        print(f"      ✅ Calculated IC/IR for {len(ic_results)} periods")

    # Quantile return analysis
    if 'factor_quantile' in factor_data.columns and forward_cols:
        print(f"\n      Quantile Return Analysis:")
        for col in forward_cols:
            try:
                quantile_returns = factor_data.groupby(
                    'factor_quantile')[col].agg(['mean', 'std', 'count'])
                print(f"        Period {col}:")
                for q in quantile_returns.index:
                    mean_ret = quantile_returns.loc[q, 'mean']
                    std_ret = quantile_returns.loc[q, 'std']
                    count = quantile_returns.loc[q, 'count']
                    print(
                        f"          Q{q}: Mean={mean_ret:.6f}, Std={std_ret:.6f}, N={count}"
                    )
            except Exception as e:
                print(f"        Period {col}: Error in quantile analysis: {e}")

    print("      " + "=" * 60)

    # Save to text file
    try:
        factor_safe_name = factor_name.replace("/", "_").replace(" ", "_")
        stats_path = os.path.join(output_dir,
                                  f"{factor_safe_name}_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Statistics for {factor_name}\n")
            f.write("=" * 60 + "\n\n")
            if 'factor' in factor_data.columns:
                factor_values = factor_data['factor'].dropna()
                f.write(f"Factor Statistics:\n")
                f.write(f"  Count: {len(factor_values)}\n")
                f.write(f"  Mean: {factor_values.mean():.6f}\n")
                f.write(f"  Std: {factor_values.std():.6f}\n")
                f.write(f"  Min: {factor_values.min():.6f}\n")
                f.write(f"  Max: {factor_values.max():.6f}\n\n")

            # IC/IR Statistics
            if ic_results:
                f.write(
                    f"Information Coefficient (IC) and Information Ratio (IR):\n"
                )
                f.write(
                    f"  {'Period':<10} {'IC':<12} {'IC Mean':<12} {'IC Std':<12} {'IR':<12}\n"
                )
                f.write(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")
                for col, stats in ic_results.items():
                    ic_val = stats.get('IC', np.nan)
                    ic_mean = stats.get('IC_Mean', np.nan)
                    ic_std = stats.get('IC_Std', np.nan)
                    ir_val = stats.get('IR', np.nan)
                    f.write(
                        f"  {col:<10} {ic_val:>12.6f} {ic_mean:>12.6f} {ic_std:>12.6f} {ir_val:>12.6f}\n"
                    )
                f.write("\n")

            if 'factor_quantile' in factor_data.columns and forward_cols:
                for col in forward_cols:
                    try:
                        quantile_returns = factor_data.groupby(
                            'factor_quantile')[col].agg(
                                ['mean', 'std', 'count'])
                        f.write(f"Period {col} Quantile Returns:\n")
                        for q in quantile_returns.index:
                            mean_ret = quantile_returns.loc[q, 'mean']
                            std_ret = quantile_returns.loc[q, 'std']
                            count = quantile_returns.loc[q, 'count']
                            f.write(
                                f"  Q{q}: Mean={mean_ret:.6f}, Std={std_ret:.6f}, N={count}\n"
                            )
                        f.write("\n")
                    except Exception:
                        pass
        print(f"      ✅ Statistics saved to: {stats_path}")
    except Exception as e:
        print(f"      ⚠️  Error saving statistics file: {e}")


def analyze_factor(factor_data: pd.DataFrame, factor_name: str,
                   output_dir: str, quantiles: int):
    """Run Alphalens analysis and generate tear sheet."""
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    def _save_figures() -> List[str]:
        saved_paths: List[str] = []
        factor_safe_name = factor_name.replace("/", "_").replace(" ", "_")
        for idx, fig_num in enumerate(sorted(plt.get_fignums())):
            fig = plt.figure(fig_num)
            fig_path = os.path.join(output_dir,
                                    f"{factor_safe_name}_fig_{idx}.png")
            try:
                fig.savefig(fig_path,
                            dpi=150,
                            bbox_inches="tight",
                            facecolor="white")
                saved_paths.append(fig_path)
                print(f"      ✅ Figure {idx} saved to: {fig_path}")
            except Exception as fig_e:
                print(f"      ⚠️  Error saving figure {idx}: {fig_e}")
        return saved_paths

    def _generate_partial_tears() -> List[str]:
        print(
            "      ℹ️  Falling back to partial tear sheets (returns/information)..."
        )
        generated_any = False

        try:
            # DEBUG: Check data structure before calling Alphalens
            print(f"      🔍 DEBUG: Before create_returns_tear_sheet:")
            print(f"        factor_data.index type: {type(factor_data.index)}")
            print(
                f"        factor_data.index.nlevels: {factor_data.index.nlevels}"
            )
            if isinstance(factor_data.index, pd.MultiIndex):
                print(
                    f"        factor_data.index.levels: {len(factor_data.index.levels)}"
                )
                print(
                    f"        First index tuple: {factor_data.index[0]} (length: {len(factor_data.index[0])})"
                )

            al.tears.create_returns_tear_sheet(
                factor_data,
                long_short=True,
                group_neutral=False,
                set_context=True,
            )
            generated_any = True
        except Exception as ret_e:
            print(f"      ⚠️  create_returns_tear_sheet error: {ret_e}")
            import traceback
            print(f"      🔍 DEBUG: Full traceback:")
            traceback.print_exc()

        try:
            # DEBUG: Check data structure before calling Alphalens
            print(f"      🔍 DEBUG: Before create_information_tear_sheet:")
            print(f"        factor_data.index type: {type(factor_data.index)}")
            print(
                f"        factor_data.index.nlevels: {factor_data.index.nlevels}"
            )
            if isinstance(factor_data.index, pd.MultiIndex):
                print(
                    f"        factor_data.index.levels: {len(factor_data.index.levels)}"
                )
                print(
                    f"        First index tuple: {factor_data.index[0]} (length: {len(factor_data.index[0])})"
                )

            al.tears.create_information_tear_sheet(
                factor_data,
                group_neutral=False,
                set_context=False,
            )
            generated_any = True
        except Exception as info_e:
            print(f"      ⚠️  create_information_tear_sheet error: {info_e}")
            import traceback
            print(f"      🔍 DEBUG: Full traceback:")
            traceback.print_exc()

            # Try to diagnose the issue
            if "Cannot convert input" in str(info_e) and "tuple" in str(
                    info_e):
                print(
                    f"      🔍 DIAGNOSIS: Alphalens is receiving tuples with wrong structure"
                )
                print(
                    f"        This suggests the MultiIndex has more than 2 levels or is nested"
                )
                print(f"        Checking factor_data structure...")
                if isinstance(factor_data.index, pd.MultiIndex):
                    print(
                        f"        Index has {factor_data.index.nlevels} levels"
                    )
                    print(
                        f"        Index has {len(factor_data.index.levels)} level arrays"
                    )
                    for i, level in enumerate(factor_data.index.levels):
                        print(
                            f"          Level {i}: {type(level)}, length={len(level)}"
                        )
                        if len(level) > 0:
                            print(
                                f"            First value: {level[0]} (type: {type(level[0])})"
                            )

        try:
            al.tears.create_turnover_tear_sheet(
                factor_data,
                set_context=False,
            )
        except ValueError as turn_e:
            if "No objects to concatenate" in str(turn_e):
                print(
                    "      ⚠️  Turnover tear sheet skipped (insufficient factor variation across time periods)"
                )
                print(
                    "         This is normal for factors with low temporal variation or limited data range"
                )
            else:
                print(f"      ⚠️  create_turnover_tear_sheet error: {turn_e}")
        except Exception as turn_e:
            print(f"      ⚠️  create_turnover_tear_sheet error: {turn_e}")

        if generated_any:
            return _save_figures()
        return []

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"      Output directory created/verified: {output_dir}")
    except Exception as e:
        print(f"      ⚠️  Error creating output directory {output_dir}: {e}")
        # Try to use current directory as fallback
        output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

    print(f"   📊 Generating Alphalens tear sheet for {factor_name}...")
    print(f"      Output directory: {output_dir}")

    # Check if factor_data is valid
    if factor_data is None or len(factor_data) == 0:
        print(f"      ⚠️  No valid factor data for {factor_name}")
        return

    print(f"      Factor data shape: {factor_data.shape}")

    # CRITICAL: Fix index structure before passing to Alphalens
    factor_data = _fix_factor_data_index(factor_data)

    # DEBUG: Detailed index inspection before passing to Alphalens
    print(f"\n      🔍 DEBUG: Factor data structure inspection:")
    print(f"        Index type: {type(factor_data.index)}")
    print(f"        Index names: {factor_data.index.names}")
    print(f"        Index nlevels: {factor_data.index.nlevels}")
    if isinstance(factor_data.index, pd.MultiIndex):
        print(f"        Index levels: {len(factor_data.index.levels)}")
        for i, level in enumerate(factor_data.index.levels):
            print(
                f"          Level {i} ({factor_data.index.names[i]}): type={type(level)}, length={len(level)}"
            )
            if i == 0 and len(level) > 0:
                print(f"            Sample values: {level[:3]}")
            elif i == 1 and len(level) > 0:
                print(f"            Sample values: {level[:3]}")

    # Check first few index tuples
    sample_indices = list(factor_data.index[:5])
    print(f"        Sample index tuples (first 5):")
    for idx in sample_indices:
        print(
            f"          {idx} (type: {type(idx)}, length: {len(idx) if isinstance(idx, tuple) else 'N/A'})"
        )
        if isinstance(idx, tuple) and len(idx) != 2:
            print(
                f"            ⚠️  WARNING: Tuple has {len(idx)} elements, expected 2!"
            )

    print(f"        Columns: {list(factor_data.columns)}")
    print(f"        Data types:\n{factor_data.dtypes}")
    print(f"")

    # Create full tear sheet (this will display plots)
    try:
        fallback_needed = False
        try:
            al.tears.create_full_tear_sheet(
                factor_data,
                long_short=True,
                group_neutral=False,
            )
        except ValueError as full_e:
            if "No objects to concatenate" in str(full_e):
                print(
                    "      ⚠️  Full tear sheet turnover step failed (insufficient factor variation)."
                )
                print(
                    "         Falling back to partial tear sheets (returns/information only)"
                )
                fallback_needed = True
            else:
                raise

        saved_figures: List[str]
        if fallback_needed:
            plt.close("all")
            saved_figures = _generate_partial_tears()
            plt.close("all")
        else:
            saved_figures = _save_figures()
            plt.close("all")

        if saved_figures:
            print(
                f"      ✅ Total {len(saved_figures)} figures saved for {factor_name}"
            )
        else:
            print(f"      ⚠️  No figures were saved for {factor_name}")
            # If no figures were saved, at least compute and print basic statistics
            print(f"      ℹ️  Computing basic statistics as fallback...")
            try:
                _compute_manual_statistics(factor_data, factor_name,
                                           output_dir)
            except Exception as stats_e:
                print(
                    f"      ⚠️  Error computing fallback statistics: {stats_e}"
                )

        # Also compute and save IC statistics
        try:
            ic = al.performance.factor_information_coefficient(factor_data)
            ic_summary = al.performance.mean_information_coefficient(
                factor_data)

            # Save IC summary to CSV
            factor_safe_name = factor_name.replace("/", "_").replace(" ", "_")
            ic_csv_path = os.path.join(output_dir,
                                       f"{factor_safe_name}_ic_summary.csv")
            ic_summary.to_csv(ic_csv_path)
            print(f"      ✅ IC summary saved to: {ic_csv_path}")

            # Print IC statistics to console
            print(
                f"\n      📈 Information Coefficient (IC) Statistics for {factor_name}:"
            )
            print("      " + "=" * 60)
            if isinstance(ic_summary, pd.DataFrame):
                print(ic_summary.to_string())
            else:
                print(f"      Mean IC: {ic_summary}")
            print("      " + "=" * 60)
        except Exception as ic_e:
            print(f"      ⚠️  Error computing/saving IC statistics: {ic_e}")
            # Try manual IC calculation as fallback
            try:
                print(f"      🔄 Attempting manual IC calculation...")
                _compute_manual_statistics(factor_data, factor_name,
                                           output_dir)
            except Exception as manual_e:
                print(
                    f"      ⚠️  Manual statistics calculation also failed: {manual_e}"
                )

        print(f"   ✅ Tear sheet generated for {factor_name}")

    except Exception as e:
        print(f"      ⚠️  Error generating tear sheet: {e}")
        print(
            f"      factor_data shape: {getattr(factor_data, 'shape', 'N/A')}")
        print(
            f"      factor_data index names: {getattr(getattr(factor_data, 'index', None), 'names', 'N/A')}"
        )
        import traceback
        traceback.print_exc()
        plt.close("all")


def main() -> None:
    args = parse_args()

    # Parse periods
    periods = [int(p.strip()) for p in args.periods.split(",") if p.strip()]

    # Collect files
    files = _collect_files(args.data_dir, args.start, args.end, args.symbol)
    if not files:
        print(
            f"   ⚠️  No files found for symbols={args.symbol}, start={args.start}, end={args.end}"
        )
        return

    print(f"   📁 Found {len(files)} data files")

    # Load and prepare data
    df, feature_cols = load_and_prepare_data(files, args.freq,
                                             args.feature_type)

    # Filter by date if specified
    if args.start:
        start_dt = pd.to_datetime(args.start)
        df = df[df.index >= start_dt]
    if args.end:
        end_dt = pd.to_datetime(args.end)
        df = df[df.index <= end_dt]

    print(f"   📊 Data shape: {df.shape}")
    print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")

    # Get all numeric columns from DataFrame (excluding metadata columns)
    exclude_cols = {'symbol', 'timestamp', 'date', 'asset'}
    all_numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Determine which factors to analyze
    if args.factor_name:
        # Check if factor exists in DataFrame (not just in filtered feature_cols)
        if args.factor_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[args.factor_name]):
                factors_to_analyze = [args.factor_name]
            else:
                print(f"   ⚠️  Factor '{args.factor_name}' is not numeric")
                return
        else:
            print(f"   ⚠️  Factor '{args.factor_name}' not found in DataFrame")
            print(
                f"      Available factors (first 20): {all_numeric_cols[:20]}..."
            )
            print(f"      Total numeric columns: {len(all_numeric_cols)}")
            return
    else:
        # Analyze all features (or a subset for performance)
        factors_to_analyze = feature_cols[:
                                          20] if feature_cols else all_numeric_cols[:
                                                                                    20]
        print(
            f"   📊 Analyzing {len(factors_to_analyze)} factors (limited to first 20)"
        )

    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_tag = args.symbol.replace(",", "_")
    output_dir = os.path.join(
        args.output_dir,
        f"{timestamp}_{symbol_tag}_{args.feature_type}_{args.start or 'all'}_{args.end or 'all'}"
    )

    # Analyze each factor
    for factor_name in factors_to_analyze:
        try:
            print(f"\n   🔍 Analyzing factor: {factor_name}")

            # Prepare Alphalens data
            factor_data = prepare_alphalens_data(df, factor_name, periods,
                                                 args.freq)

            # Run analysis
            analyze_factor(factor_data, factor_name, output_dir,
                           args.quantiles)

        except ValueError as e:
            # For ValueError (like too many duplicates), print warning but continue
            error_msg = str(e)
            if "too many duplicate values" in error_msg.lower(
            ) or "no predictive power" in error_msg.lower():
                print(f"      ⚠️  Skipping {factor_name}: {error_msg}")
                print(
                    f"         This factor has insufficient variation for meaningful analysis"
                )
            else:
                print(f"      ⚠️  Error analyzing {factor_name}: {e}")
                import traceback
                traceback.print_exc()
            continue
        except Exception as e:
            print(f"      ⚠️  Error analyzing {factor_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n   ✅ Factor analysis complete!")
    print(f"      Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
