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
from data_tools.baseline_feature_engineering import (
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
    parser.add_argument(
        "--data-dir",
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/factor_analysis",
        help="Output directory for Alphalens tear sheets")
    parser.add_argument(
        "--periods",
        type=str,
        default="1,4,24",
        help="Forward return periods in bars (e.g., 1,4,24 for 15min, 1h, 6h prediction)"
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
                                if "-" in date_part and len(date_part) == 7:  # YYYY-MM
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
                    combined_eng = tmp.merge(sym_df.drop_duplicates(subset=["timestamp"]),
                                             on="timestamp",
                                             how="left").set_index("timestamp")
                except Exception:
                    # Last resort: fill with UNKNOWN
                    combined_eng["symbol"] = "UNKNOWN"
        else:
            # If symbol column already exists but might have NaN values, fill them
            if combined_eng["symbol"].isna().any():
                combined_eng["symbol"] = combined_eng["symbol"].fillna("UNKNOWN")
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
                            periods: List[int], freq_str: Optional[str]) -> pd.DataFrame:
    """Prepare data for Alphalens: multi-index [symbol, timestamp]."""
    # Ensure we have symbol column
    if "symbol" not in df.columns:
        raise ValueError("DataFrame must have 'symbol' column")

    # Work directly with the DataFrame to avoid timezone issues
    # Extract factor and prices before any index manipulation
    if factor_col not in df.columns:
        raise ValueError(f"Factor column '{factor_col}' not found in DataFrame")
    
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
        raise ValueError("No valid data points found after removing NaN/inf values")
    
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
    multi_index = pd.MultiIndex.from_arrays([date_index, asset_index], names=['date', 'asset'])
    
    # Parse frequency
    freq_offset = None
    if freq_str:
        try:
            freq_offset = to_offset(freq_str)
        except (TypeError, ValueError):
            print(f"      ⚠️  Could not parse freq '{freq_str}' into pandas offset; defaulting to inference")

    # Create factor Series with the new MultiIndex
    factor = pd.Series(factor_values.values, index=multi_index, name=factor_col)
    try:
        if freq_offset is not None:
            factor.index.levels[0].freq = freq_offset  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        pass
    
    # Create a DataFrame for prices with the proper MultiIndex structure
    # Alphalens expects prices to be a DataFrame with assets as columns and dates as index
    prices_df_pivot = pd.DataFrame({
        'date_col': dates_values,  # Use a different name to avoid ambiguity
        'asset': assets_values,
        'close': prices_values
    })
    
    # Pivot the DataFrame to have assets as columns (as Alphalens expects)
    prices = prices_df_pivot.pivot_table(values='close', index='date_col', columns='asset', aggfunc='first')
    
    # Rename the index to 'date' after pivot
    prices.index.name = 'date'
    
    # Ensure the index is timezone-naive DatetimeIndex
    prices.index = pd.DatetimeIndex(prices.index)
    if hasattr(prices.index, 'tz') and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    try:
        if freq_offset is not None:
            prices.index.freq = freq_offset  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        pass
    
    # Ensure prices DataFrame is properly sorted and filled
    prices = prices.sort_index().ffill().bfill()
    
    # Align factor and prices to have common assets
    common_assets = sorted(set(factor.index.get_level_values("asset")) & set(prices.columns))
    if not common_assets:
        raise ValueError("No common assets between factor and prices")
    
    prices = prices[common_assets]
    factor = factor.loc[factor.index.get_level_values("asset").isin(common_assets)]
    
    # Check if factor has enough unique values
    if factor.nunique() < 5:
        raise ValueError(f"Factor contains only {factor.nunique()} unique values, need at least 5 for quantile analysis")
    
    # Prepare Alphalens data
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        periods=periods,
        quantiles=10,  # Use 10 quantiles for factor analysis
        bins=None,
        binning_by_group=False,
        max_loss=0.99,  # Allow up to 99% loss to avoid dropping too much data
    )

    return factor_data


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
        print("      ℹ️  Falling back to partial tear sheets (returns/information)...")
        generated_any = False

        try:
            al.tears.create_returns_tear_sheet(
                factor_data,
                long_short=True,
                group_neutral=False,
                set_context=True,
            )
            generated_any = True
        except Exception as ret_e:
            print(f"      ⚠️  create_returns_tear_sheet error: {ret_e}")

        try:
            al.tears.create_information_tear_sheet(
                factor_data,
                group_neutral=False,
                set_context=False,
            )
            generated_any = True
        except Exception as info_e:
            print(f"      ⚠️  create_information_tear_sheet error: {info_e}")

        try:
            al.tears.create_turnover_tear_sheet(
                factor_data,
                set_context=False,
            )
        except ValueError as turn_e:
            if "No objects to concatenate" in str(turn_e):
                print("      ⚠️  Turnover tear sheet skipped (no data to concatenate)")
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
                    "      ⚠️  Full tear sheet turnover step failed (no objects to concatenate)."
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
            print(f"      ✅ Total {len(saved_figures)} figures saved for {factor_name}")
        else:
            print(f"      ⚠️  No figures were saved for {factor_name}")

        # Also compute and save IC statistics
        try:
            ic = al.performance.factor_information_coefficient(factor_data)
            ic_summary = al.performance.mean_information_coefficient(factor_data)

            # Save IC summary to CSV
            factor_safe_name = factor_name.replace("/", "_").replace(" ", "_")
            ic_csv_path = os.path.join(output_dir, f"{factor_safe_name}_ic_summary.csv")
            ic_summary.to_csv(ic_csv_path)
            print(f"      ✅ IC summary saved to: {ic_csv_path}")
        except Exception as ic_e:
            print(f"      ⚠️  Error computing/saving IC statistics: {ic_e}")

        print(f"   ✅ Tear sheet generated for {factor_name}")

    except Exception as e:
        print(f"      ⚠️  Error generating tear sheet: {e}")
        print(f"      factor_data shape: {getattr(factor_data, 'shape', 'N/A')}")
        print(f"      factor_data index names: {getattr(getattr(factor_data, 'index', None), 'names', 'N/A')}")
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
        print(f"   ⚠️  No files found for symbols={args.symbol}, start={args.start}, end={args.end}")
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

    # Determine which factors to analyze
    if args.factor_name:
        factors_to_analyze = [args.factor_name] if args.factor_name in feature_cols else []
        if not factors_to_analyze:
            print(f"   ⚠️  Factor '{args.factor_name}' not found in features")
            print(f"      Available factors: {feature_cols[:10]}...")
            return
    else:
        # Analyze all features (or a subset for performance)
        factors_to_analyze = feature_cols[:20]  # Limit to first 20 for performance
        print(f"   📊 Analyzing {len(factors_to_analyze)} factors (limited to first 20)")

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
            analyze_factor(factor_data, factor_name, output_dir, args.quantiles)

        except Exception as e:
            print(f"      ⚠️  Error analyzing {factor_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n   ✅ Factor analysis complete!")
    print(f"      Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

