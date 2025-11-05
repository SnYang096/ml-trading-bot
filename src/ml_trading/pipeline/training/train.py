from __future__ import annotations
"""
Single-run regression training:
- Predict future_return (mean), future_return quantiles (q10/q90) for uncertainty,
- Predict future_volatility (realized volatility) for risk-aware sizing.

Feature selection options: --feature-type, --use-top-factors, --topk, --topk-source
"""

# Copied from baseline.train_baseline with naming/docs adjusted
import os
import argparse
import json
from typing import List
import numpy as np
import pandas as pd

from ml_trading.data_tools.rolling_data import load_parquet_file
from ml_trading.data_tools.baseline_feature_engineering import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)
from ml_trading.models.lightgbm_model import LightGBMModel


def _load_many(files: List[str]) -> pd.DataFrame:
    """Load and merge multiple parquet files.
    
    For multi-asset training, all assets' data are merged together.
    All features are normalized (asset-agnostic), so the model can learn
    common patterns across different assets.
    """
    frames: List[pd.DataFrame] = []
    for f in files:
        df = load_parquet_file(f) if f.endswith(".parquet") else None
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid data files loaded")
    # Merge all dataframes (for multi-asset training, all assets are combined)
    # Note: Multi-asset data may not have 'symbol' column if it's already in the index
    # or if files contain data from different assets without explicit symbol column
    merged = pd.concat(frames, axis=0).sort_index()
    print(f"   Merged {len(frames)} data file(s), total {len(merged)} samples")

    # Ensure DatetimeIndex
    if not isinstance(merged.index, pd.DatetimeIndex):
        if "timestamp" in merged.columns:
            merged.set_index("timestamp", inplace=True)
        else:
            raise ValueError(
                "Merged data must have DatetimeIndex or 'timestamp' column")

    return merged


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to specified timeframe.
    
    CRITICAL: This function ensures different timeframes use different aggregated data.
    Without this, all timeframes would use the same raw data, leading to identical results.
    
    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        timeframe: Target timeframe (e.g., '5T', '15T', '45T', '240T')
    
    Returns:
        Resampled DataFrame with OHLCV data aggregated to the specified timeframe
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")

    # Handle multi-asset data: group by symbol if present
    if "symbol" in df.columns:
        # Multi-asset: resample each symbol separately, then combine
        resampled_frames = []
        for symbol in df["symbol"].unique():
            symbol_mask = df["symbol"] == symbol
            symbol_data = df[symbol_mask].copy()
            # Remove symbol column temporarily for resampling
            symbol_data = symbol_data.drop(columns=["symbol"])
            symbol_resampled = _resample_single_asset(symbol_data, timeframe)
            symbol_resampled["symbol"] = symbol
            resampled_frames.append(symbol_resampled)
        result = pd.concat(resampled_frames, axis=0).sort_index()
    else:
        # Single asset
        result = _resample_single_asset(df, timeframe)

    return result


def _resample_single_asset(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a single asset's OHLCV data."""
    # Ensure we have required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")

    # Resample OHLCV: open=first, high=max, low=min, close=last, volume=sum
    resampled_dict = {
        "open": df["open"].resample(timeframe).first(),
        "high": df["high"].resample(timeframe).max(),
        "low": df["low"].resample(timeframe).min(),
        "close": df["close"].resample(timeframe).last(),
        "volume": df["volume"].resample(timeframe).sum(),
    }

    # Handle optional columns
    if "buy_qty" in df.columns:
        resampled_dict["buy_qty"] = df["buy_qty"].resample(timeframe).sum()
    if "sell_qty" in df.columns:
        resampled_dict["sell_qty"] = df["sell_qty"].resample(timeframe).sum()
    if "taker_buy_ratio" in df.columns:
        resampled_dict["taker_buy_ratio"] = df["taker_buy_ratio"].resample(
            timeframe).mean()
    if "cvd" in df.columns:
        resampled_dict["cvd"] = df["cvd"].resample(timeframe).last()
    if "trade_count" in df.columns:
        resampled_dict["trade_count"] = df["trade_count"].resample(
            timeframe).sum()

    resampled = pd.DataFrame(resampled_dict)

    # Drop rows where OHLCV is NaN
    resampled = resampled.dropna(subset=required_cols)

    # Forward fill optional columns (handle deprecated fillna method)
    optional_cols = [
        "buy_qty", "sell_qty", "taker_buy_ratio", "cvd", "trade_count"
    ]
    for col in optional_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill()

    return resampled


def _collect_files(data: List[str],
                   data_dir: str | None,
                   start: str | None,
                   end: str | None,
                   symbols: str | None = None) -> List[str]:
    """Collect files for one or multiple symbols.
    
    Args:
        symbols: Single symbol or comma-separated symbols (e.g., "BTCUSDT" or "BTCUSDT,ETHUSDT,SOLUSDT")
    """
    files: List[str] = []
    files.extend(data)
    if data_dir and os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".parquet"):
                files.append(os.path.join(data_dir, name))
    files = [os.path.abspath(p) for p in files if os.path.exists(p)]

    if symbols:
        # Support multiple symbols (comma-separated)
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        mapping = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "BNBUSDT": "BNB-USD",
            "ADAUSDT": "ADA-USD",
            "SOLUSDT": "SOL-USD"
        }
        filtered = []
        for symbol in symbol_list:
            file_symbol = mapping.get(symbol, symbol.replace("USDT", "-USD"))
            for p in files:
                fn = os.path.basename(p).upper()
                if (fn.startswith(symbol.upper())
                        or fn.startswith(file_symbol.upper()) or fn.startswith(
                            file_symbol.replace("-", "_").upper())):
                    if p not in filtered:  # Avoid duplicates
                        filtered.append(p)
        files = filtered

    if start or end:
        import re

        def _ym(n: str) -> str | None:
            m = re.search(r"(20\d{2})[-_](\d{2})", os.path.basename(n))
            return f"{m.group(1)}-{m.group(2)}" if m else None

        filtered = []
        for p in files:
            ym = _ym(p)
            if ym is None:
                continue
            if start and ym < start:
                continue
            if end and ym > end:
                continue
            filtered.append(p)
        files = filtered
    if not files:
        raise FileNotFoundError("No parquet files found from inputs")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression training (returns + uncertainty + volatility)")
    parser.add_argument("--data",
                        type=str,
                        action="append",
                        default=[],
                        help="Parquet file(s) to use")
    parser.add_argument("--data-dir",
                        type=str,
                        default=None,
                        help="Directory containing parquet files")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help=
        "Symbol(s) metadata for report. Can be comma-separated (e.g., BTCUSDT,ETHUSDT,SOLUSDT) for multi-asset training"
    )
    parser.add_argument("--freq",
                        type=str,
                        default="5T",
                        help="Bar timeframe(s), comma-separated: 5T,15T")
    parser.add_argument("--start",
                        type=str,
                        default=None,
                        help="Start YYYY-MM (inclusive)")
    parser.add_argument("--end",
                        type=str,
                        default=None,
                        help="End YYYY-MM (inclusive)")
    parser.add_argument("--forward-bars",
                        type=str,
                        default="3",
                        help="Bars ahead (e.g., 1,5,10)")
    parser.add_argument("--cv-folds",
                        type=int,
                        default=0,
                        help="TimeSeries CV folds (0=disable)")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="baseline",
        help=
        "baseline/default/enhanced/hurst/wavelet/hilbert/spectral/order_flow/dl_sequence/comprehensive or combos (e.g., baseline,default,hurst)"
    )
    parser.add_argument("--oos-months",
                        type=int,
                        default=3,
                        help="OOS months after train end (0=disable)")
    parser.add_argument("--oos-start",
                        type=str,
                        default=None,
                        help="OOS start (YYYY-MM-DD)")
    parser.add_argument("--oos-end",
                        type=str,
                        default=None,
                        help="OOS end (YYYY-MM-DD)")
    parser.add_argument("--use-top-factors",
                        type=str,
                        default=None,
                        help="JSON of selected features to keep")
    parser.add_argument("--topk",
                        type=int,
                        default=0,
                        help="Keep only Top-K features (0=disabled)")
    parser.add_argument(
        "--topk-source",
        type=str,
        default=None,
        help="Ranking CSV(feature,score) or JSON list; fallback |IC|")
    parser.add_argument("--gpu",
                        action="store_true",
                        default=True,
                        help="Use GPU for LightGBM")
    args = parser.parse_args()

    freqs = [f.strip() for f in args.freq.split(",") if f.strip()]
    fbs = [int(x.strip()) for x in args.forward_bars.split(",") if x.strip()]

    files = _collect_files(args.data,
                           args.data_dir,
                           args.start,
                           args.end,
                           symbols=args.symbol)
    raw = _load_many(files)

    # Parse symbols for multi-asset training
    symbol_list = [s.strip() for s in args.symbol.split(",") if s.strip()]
    symbols_str = ",".join(symbol_list) if len(
        symbol_list) > 1 else symbol_list[0] if symbol_list else "UNKNOWN"
    print(f"📊 Training with symbol(s): {symbols_str}")
    if len(symbol_list) > 1:
        print(f"   Multi-asset training: {len(symbol_list)} assets")
        print(f"   Total samples: {len(raw)}")

    # Ensure raw data has DatetimeIndex for resampling
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "timestamp" in raw.columns:
            raw.set_index("timestamp", inplace=True)
        else:
            raise ValueError(
                "Data must have DatetimeIndex or 'timestamp' column for resampling"
            )

    # Create timestamped base directory for this training run to avoid mixing old data
    from datetime import datetime as _dt
    training_timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    # Format symbol for directory name (replace comma with underscore for multi-asset)
    symbol_dir = symbols_str.replace(",", "_")
    # Create base directory with timestamp, symbol, and feature_type
    # We'll finalize by appending train_start/train_end (YYYYMMDD) after first config is processed
    base_dir = f"{training_timestamp}_{symbol_dir}_{args.feature_type}"
    base_results_dir = os.path.join("results/training", base_dir)
    base_dir_finalized = False
    print(f"📁 Results will be saved to: {base_results_dir}")

    for freq in freqs:
        # CRITICAL FIX: Resample data to the specified timeframe BEFORE feature engineering
        # This ensures different timeframes use different aggregated data
        # Without this, all timeframes would use the same raw data, leading to identical results!
        print(f"\n{'='*60}")
        print(f"🔄 Resampling data to timeframe: {freq}")
        print(f"{'='*60}")
        try:
            resampled_data = _resample_ohlcv(raw, freq)
            print(
                f"   ✅ Original samples: {len(raw):,}, Resampled samples: {len(resampled_data):,}"
            )
            if len(resampled_data) == 0:
                print(
                    f"   ⚠️  Warning: Resampled data is empty for {freq}, skipping..."
                )
                continue
        except Exception as e:
            print(f"   ❌ Error resampling to {freq}: {e}")
            print(f"   ⚠️  Skipping timeframe {freq}...")
            continue

        for fb in fbs:
            print(
                f"\n⚙️  Training config: timeframe={freq}, forward_bars={fb}")
            print(f"   Samples in resampled data: {len(resampled_data):,}")
            feat_df = resampled_data.copy()
            # Multi-asset training: features are engineered on merged data
            # All features are normalized (asset-agnostic), so the model learns
            # common patterns across different assets
            if args.feature_type == "baseline":
                feat_df, base_eng = engineer_baseline_features(feat_df,
                                                               None,
                                                               fit=True)
                feature_engineer = base_eng
            else:
                feature_engineer = ComprehensiveFeatureEngineer(
                    feature_types=args.feature_type)
                feat_df = feature_engineer.engineer_all_features(feat_df,
                                                                 fit=True)

            # Calculate future return (simple return for fb bars ahead)
            # CRITICAL: Use close price, NOT high/low price, to avoid look-ahead bias
            # future_return[t] = (close[t+fb] / close[t]) - 1
            # This represents the return over the next fb bars, using closing prices
            feat_df["future_return"] = feat_df["close"].shift(
                -fb) / feat_df["close"] - 1

            # Verify label definition: log a warning if any suspicious pattern is detected
            # (e.g., if future_return seems to be calculated from high instead of close)
            # For now, we trust the calculation above, but we can add validation later if needed

            # CRITICAL FIX: Remove price continuity bias using AR(1) residual for ALL fb values
            # This addresses the issue where high IC/accuracy may be due to price continuity
            # (adjacent bars are highly correlated) rather than true predictive power
            # Reference: docs/特征：超高准确率的问题.md
            # ALL price-based features must use returns/differences, and ALL cases apply AR(1) residual
            ar1_phi = None
            ar1_autocorr = None
            ar1_autocorr_after = None

            print(
                f"   🔧 Removing price continuity bias using AR(1) residual (applied to all fb values)..."
            )
            # Calculate current period log returns for AR(1) estimation
            # Use log returns for consistency with AR(1) model and to handle compounding correctly
            current_returns = np.log(feat_df["close"] /
                                     feat_df["close"].shift(1))
            # Calculate lag-1 autocorrelation (AR(1) coefficient φ)
            valid_returns = current_returns.dropna()
            if len(valid_returns
                   ) > 100:  # Need enough data for stable estimation
                # Method 1: pandas autocorr (lag-1)
                ar1_autocorr = valid_returns.autocorr(lag=1)
                # Method 2: numpy correlation (alternative)
                if len(valid_returns) > 1:
                    r_t = valid_returns[:-1].values
                    r_t1 = valid_returns[1:].values
                    if len(r_t) > 0 and len(r_t1) > 0 and not np.isnan(
                            r_t).any() and not np.isnan(r_t1).any():
                        ar1_phi = np.corrcoef(
                            r_t, r_t1)[0, 1] if len(r_t) > 1 else None

                # Use pandas autocorr as primary, numpy as fallback
                if pd.notna(ar1_autocorr):
                    ar1_phi = ar1_autocorr
                elif ar1_phi is not None and not np.isnan(ar1_phi):
                    ar1_phi = ar1_phi
                else:
                    ar1_phi = 0.0

                if ar1_phi is not None and not np.isnan(ar1_phi):
                    print(
                        f"      📊 Lag-1 autocorrelation (AR(1) φ) before removal: {ar1_phi:.4f}"
                    )
                    if abs(ar1_phi) > 0.3:
                        print(
                            f"      ⚠️  High autocorrelation detected! This explains high IC/accuracy for fb={fb}."
                        )
                        if abs(ar1_phi) > 0.5:
                            print(
                                f"      🚨 Very high autocorrelation (>0.5)! This is a strong indicator of price continuity bias."
                            )

                    # Calculate AR(1) residual: target = return_{t+fb} - AR(1)_prediction
                    # For fb>1, we need to predict the cumulative return over fb bars
                    # Using AR(1) model: if r_t = φ * r_{t-1} + ε_t, then
                    # E[r_{t+1} | r_t] = φ * r_t
                    # E[r_{t+2} | r_t] = φ^2 * r_t
                    # ...
                    # E[r_{t+fb} | r_t] = φ^fb * r_t
                    # For cumulative log return over fb bars: E[Σ_{i=1}^{fb} r_{t+i} | r_t] ≈ φ * (1-φ^fb)/(1-φ) * r_t
                    # For simplicity and to avoid numerical issues, we use a simplified approach:
                    # For fb=1: subtract φ * r_t
                    # For fb>1: subtract φ * r_t (first-order approximation, as higher-order terms are small)

                    # Convert future_return to log return for consistency
                    future_return_log = np.log(1 + feat_df["future_return"])

                    # Calculate AR(1) prediction for cumulative log return
                    # For fb=1: AR(1) prediction = φ * r_t
                    # For fb>1: use simplified approximation φ * r_t (first-order effect)
                    # This removes the predictable component due to price continuity
                    ar1_prediction_log = ar1_phi * current_returns

                    # For fb>1, we could use a more sophisticated prediction, but for now
                    # we use the first-order approximation which is reasonable for small φ
                    original_future_return = feat_df["future_return"].copy()

                    # Subtract AR(1) prediction from log return, then convert back to simple return
                    future_return_log_residual = future_return_log - ar1_prediction_log
                    feat_df["future_return"] = np.exp(
                        future_return_log_residual) - 1

                    print(
                        f"      ✅ Applied AR(1) residual: target = log_return_{fb} - {ar1_phi:.4f} * log_return_t"
                    )
                    print(
                        f"      📈 Original future_return stats: mean={original_future_return.mean():.6f}, std={original_future_return.std():.6f}, min={original_future_return.min():.6f}, max={original_future_return.max():.6f}"
                    )
                    print(
                        f"      📈 Residual future_return stats: mean={feat_df['future_return'].mean():.6f}, std={feat_df['future_return'].std():.6f}, min={feat_df['future_return'].min():.6f}, max={feat_df['future_return'].max():.6f}"
                    )

                    # Check if residual returns are reasonable
                    if abs(feat_df['future_return'].max()) > 10.0 or abs(
                            feat_df['future_return'].min()) > 10.0:
                        print(
                            f"      ⚠️  Warning: Residual future_return has extreme values (>{10.0:.0%}), this may indicate an issue with AR(1) processing"
                        )

                    # Check autocorrelation AFTER removal to verify effectiveness
                    # Convert residual simple return back to log return for autocorrelation check
                    residual_log_returns = np.log(
                        1 + feat_df["future_return"]).dropna()
                    if len(residual_log_returns) > 100:
                        ar1_autocorr_after = residual_log_returns.autocorr(
                            lag=1)
                        if pd.notna(ar1_autocorr_after):
                            print(
                                f"      📊 Lag-1 autocorrelation AFTER removal: {ar1_autocorr_after:.4f}"
                            )
                            if abs(ar1_autocorr_after) < abs(ar1_autocorr):
                                reduction = abs(ar1_autocorr) - abs(
                                    ar1_autocorr_after)
                                print(
                                    f"      ✅ Autocorrelation reduced by {reduction:.4f} ({(reduction/abs(ar1_autocorr)*100):.1f}% reduction)"
                                )
                            else:
                                print(
                                    f"      ⚠️  Warning: Autocorrelation did not decrease significantly after AR(1) removal"
                                )
                else:
                    print(
                        f"      ⚠️  Could not estimate AR(1) coefficient, skipping continuity removal"
                    )
                    ar1_phi = None
            else:
                print(
                    f"      ⚠️  Insufficient data for AR(1) estimation (need >100 samples, got {len(valid_returns)})"
                )

            # FIXED: future_volatility should use future returns, not shifted current returns
            # Previous bug: one.shift(-1) was using future data (data leakage!)
            # Correct: Calculate volatility from actual future returns
            safe_window = max(2, fb)
            # Use future_return to calculate future volatility (realized volatility)
            # For fb=1, this is the volatility of the next bar's return
            future_returns = feat_df["future_return"]
            feat_df["future_volatility"] = future_returns.rolling(
                window=safe_window, min_periods=1).std(ddof=0)
            # Only drop rows where targets are NaN; allow feature NaNs (handled later)
            feat_df = feat_df.dropna(
                subset=["future_return", "future_volatility"]).copy()

            from dateutil.relativedelta import relativedelta
            train_end = feat_df.index.max() if not feat_df.empty else None
            oos_start_dt = None
            oos_end_dt = None
            oos_df = pd.DataFrame()
            if args.oos_months > 0 or args.oos_start is not None:
                if args.oos_start:
                    try:
                        oos_start_dt = pd.to_datetime(args.oos_start)
                    except Exception:
                        oos_start_dt = None
                if oos_start_dt is None and train_end is not None:
                    oos_start_dt = train_end + relativedelta(
                        months=args.oos_months)
                if args.oos_end:
                    try:
                        oos_end_dt = pd.to_datetime(args.oos_end)
                    except Exception:
                        oos_end_dt = None
                if oos_end_dt is None and oos_start_dt is not None:
                    oos_end_dt = oos_start_dt + relativedelta(months=3)
                if oos_start_dt is not None and oos_end_dt is not None:
                    oos_mask = (feat_df.index >= oos_start_dt) & (
                        feat_df.index <= oos_end_dt)
                    oos_df = feat_df[oos_mask].copy()

            train_df = feat_df if len(
                oos_df) == 0 or oos_start_dt is None else feat_df[
                    feat_df.index < oos_start_dt]

            if args.feature_type == "baseline":
                feature_cols = get_baseline_feature_columns(train_df)
            else:
                feature_cols = get_feature_columns_by_type(
                    train_df, args.feature_type)
            # optional top-factors
            if args.use_top_factors:
                try:
                    with open(args.use_top_factors, "r",
                              encoding="utf-8") as f:
                        top = json.load(f)
                    if isinstance(top, dict) and "features" in top:
                        top = top["features"]
                    if isinstance(top, list):
                        s = set(top)
                        feature_cols = [c for c in feature_cols if c in s]
                except Exception:
                    pass
            # numeric only
            feature_cols = [
                c for c in feature_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]
            # optional top-k
            if args.topk and args.topk > 0 and len(feature_cols) > args.topk:
                ranked = None
                if args.topk_source:
                    try:
                        if args.topk_source.lower().endswith(".csv"):
                            _df = pd.read_csv(args.topk_source)
                            if {"feature", "score"}.issubset(set(_df.columns)):
                                _df = _df.sort_values("score", ascending=False)
                                ranked = [
                                    f for f in _df["feature"].tolist()
                                    if f in feature_cols
                                ]
                        else:
                            lst = json.load(
                                open(args.topk_source, "r", encoding="utf-8"))
                            if isinstance(lst, dict) and "features" in lst:
                                lst = lst["features"]
                            if isinstance(lst, list):
                                ranked = [f for f in lst if f in feature_cols]
                    except Exception:
                        ranked = None
                if ranked is None:
                    try:
                        from scipy.stats import spearmanr
                        ic = []
                        for c in feature_cols:
                            try:
                                r, _ = spearmanr(
                                    train_df[c].values,
                                    train_df["future_return"].values,
                                    nan_policy="omit")
                                ic.append((c, abs(r) if pd.notna(r) else 0.0))
                            except Exception:
                                ic.append((c, 0.0))
                        ic.sort(key=lambda x: x[1], reverse=True)
                        ranked = [c for c, _ in ic]
                    except Exception:
                        ranked = feature_cols
                feature_cols = ranked[:args.topk]

            X_df = pd.DataFrame(train_df[feature_cols].values,
                                columns=feature_cols,
                                index=train_df.index)
            y_return = pd.Series(train_df["future_return"].values,
                                 index=train_df.index)
            y_vol = pd.Series(train_df["future_volatility"].values,
                              index=train_df.index)

            # DEBUG: Print actual value ranges to diagnose unit issues
            print(f"\n📊 Target Variable Ranges (fb={fb}):")
            print(
                f"   future_return: min={y_return.min():.6f}, max={y_return.max():.6f}, mean={y_return.mean():.6f}, std={y_return.std():.6f}"
            )
            print(
                f"   future_return percentiles: 1%={y_return.quantile(0.01):.6f}, 50%={y_return.quantile(0.5):.6f}, 99%={y_return.quantile(0.99):.6f}"
            )
            print(
                f"   future_volatility: min={y_vol.min():.6f}, max={y_vol.max():.6f}, mean={y_vol.mean():.6f}, std={y_vol.std():.6f}"
            )
            # Check if future_return values are reasonable
            # For fb=1, returns should typically be small (e.g., ±0.05 = ±5%)
            # For larger fb (e.g., fb=45), returns can exceed ±1.0 (100%) for volatile assets like crypto
            # Adjust threshold based on fb value
            max_reasonable_return = 0.1 if fb == 1 else min(
                0.5, 0.1 * fb)  # 10% for fb=1, or 10%*fb up to 50%
            if abs(y_return.max()) > max_reasonable_return or abs(
                    y_return.min()) < -max_reasonable_return:
                if fb == 1:
                    print(
                        f"   ⚠️ 警告: future_return超出±{max_reasonable_return:.1%}范围（fb={fb}），可能单位不是收益率比例！"
                    )
                    print(
                        f"      预期fb=1的收益率应在±5%范围内，当前范围: [{y_return.min():.2%}, {y_return.max():.2%}]"
                    )
                else:
                    print(
                        f"   ℹ️  提示: future_return超出±{max_reasonable_return:.1%}范围（fb={fb}），这是正常的"
                    )
                    print(
                        f"      对于fb={fb}，未来{fb}根K线的累积收益率可能较大，当前范围: [{y_return.min():.2%}, {y_return.max():.2%}]"
                    )
                    print(f"      如果数值过大（如>500%），请检查数据或计算逻辑")

            use_cv = args.cv_folds > 0
            n_splits = args.cv_folds if use_cv else 0

            # q50: median as primary point estimate (using new quantile API)
            model_q50 = LightGBMModel(model_type="quantile",
                                      quantile_alpha=0.5,
                                      use_gpu=args.gpu)
            # Use TimeSeries CV by default to avoid random split failures on edge cases
            q50_metrics = model_q50.train(X_df,
                                          y_return,
                                          n_splits=max(2, args.cv_folds or 2),
                                          use_time_series_cv=True)

            # q10: 10% quantile for uncertainty estimation
            model_q10 = LightGBMModel(model_type="quantile",
                                      quantile_alpha=0.1,
                                      use_gpu=args.gpu)
            q10_metrics = model_q10.train(X_df,
                                          y_return,
                                          n_splits=max(2, args.cv_folds or 2),
                                          use_time_series_cv=True)

            # q90: 90% quantile for uncertainty estimation
            model_q90 = LightGBMModel(model_type="quantile",
                                      quantile_alpha=0.9,
                                      use_gpu=args.gpu)
            q90_metrics = model_q90.train(X_df,
                                          y_return,
                                          n_splits=max(2, args.cv_folds or 2),
                                          use_time_series_cv=True)

            # volatility: regression model for volatility prediction
            model_vol = LightGBMModel(model_type="regression",
                                      use_gpu=args.gpu)
            vol_metrics = model_vol.train(X_df,
                                          y_vol,
                                          n_splits=n_splits,
                                          use_time_series_cv=use_cv)

            # Directional metrics (derived from q50 regression) and regression metrics containers
            oos_metrics = {}
            directional_metrics_train = {}
            if len(oos_df) > 0:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
                X_oos = oos_df[feature_cols].values
                y_ret_oos = oos_df["future_return"].values
                y_vol_oos = oos_df["future_volatility"].values
                y_pred_q50 = model_q50.model.predict(X_oos)
                oos_rmse = float(
                    np.sqrt(mean_squared_error(y_ret_oos, y_pred_q50)))
                oos_mae = float(mean_absolute_error(y_ret_oos, y_pred_q50))
                y_pred_q10 = model_q10.model.predict(X_oos)
                y_pred_q90 = model_q90.model.predict(X_oos)
                coverage = float(
                    np.mean((y_ret_oos >= y_pred_q10)
                            & (y_ret_oos <= y_pred_q90)))
                width = float(np.mean(np.maximum(0.0,
                                                 y_pred_q90 - y_pred_q10)))
                conf = float(
                    np.mean(
                        np.abs(y_pred_q50) /
                        (np.maximum(1e-8, y_pred_q90 - y_pred_q10))))
                y_pred_vol = model_vol.model.predict(X_oos)
                oos_vol_rmse = float(
                    np.sqrt(mean_squared_error(y_vol_oos, y_pred_vol)))
                oos_vol_mae = float(mean_absolute_error(y_vol_oos, y_pred_vol))
                # Derive directional metrics from q50 regression (direction prediction)
                from scipy.stats import spearmanr, pearsonr
                y_true_dir = (y_ret_oos > 0).astype(int)
                y_score = y_pred_q50
                y_pred_dir = (y_score > 0).astype(int)
                acc = float(accuracy_score(y_true_dir, y_pred_dir))

                # ⚠️ DATA LEAKAGE WARNING: Check for suspiciously high accuracy in OOS
                n_samples_oos = len(y_true_dir)
                suspicious_oos = False
                threshold_oos = 0.90 if fb == 1 else (0.85 if fb <= 5 else (
                    0.80 if fb <= 15 else 0.75))

                if acc > threshold_oos:
                    suspicious_oos = True
                if n_samples_oos < 200 and acc > 0.85:
                    suspicious_oos = True

                if suspicious_oos:
                    print("\n" + "=" * 70)
                    print("🚨 严重警告：样本外测试中检测到可能的数据泄露或异常表现！")
                    print("=" * 70)
                    print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                    print(f"   OOS样本数量: {n_samples_oos}")
                    print(
                        f"   方向准确率: {acc*100:.2f}% (阈值: {threshold_oos*100:.0f}%)"
                    )
                    print(
                        f"   即使在样本外测试中，预测未来 {fb} 根 {freq} K线方向准确率 {acc*100:.2f}% 也是极其罕见的！"
                    )
                    print(f"   这强烈暗示存在数据泄露、特征包含未来信息或市场处于极端单边行情！")
                    print(f"\n   建议检查：")
                    print(
                        f"   - 确认标签定义：future_return = close[t+fb] / close[t] - 1（使用收盘价，非最高/最低价）"
                    )
                    print(f"   - 确认特征工程：所有特征都使用 shift(1) 避免未来信息")
                    print(f"   - 检查数据resample是否正确（{freq}下应使用正确的聚合数据）")
                    print(f"   - 检查OOS时间段是否与训练期市场状态相似（如都是单边上涨）")
                    print(f"   - 如果样本数少（{n_samples_oos}条），考虑使用更长的时间跨度")
                    print("=" * 70 + "\n")

                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_dir, y_pred_dir, average="binary", zero_division=0)
                try:
                    auc = float(roc_auc_score(y_true_dir, y_score))
                except Exception:
                    auc = float("nan")
                try:
                    pr_auc = float(average_precision_score(
                        y_true_dir, y_score))
                except Exception:
                    pr_auc = float("nan")
                # Calculate IC (Information Coefficient) for OOS
                try:
                    ic_spearman_oos, _ = spearmanr(y_ret_oos,
                                                   y_score,
                                                   nan_policy="omit")
                    ic_spearman_oos = float(ic_spearman_oos) if not np.isnan(
                        ic_spearman_oos) else None
                except Exception:
                    ic_spearman_oos = None
                try:
                    ic_pearson_oos, _ = pearsonr(y_ret_oos, y_score)
                    ic_pearson_oos = float(ic_pearson_oos) if not np.isnan(
                        ic_pearson_oos) else None
                except Exception:
                    ic_pearson_oos = None
                oos_metrics = {
                    "directional_oos": {
                        "accuracy": acc,
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                        "auc": auc,
                        "pr_auc": pr_auc,
                        "ic_spearman": ic_spearman_oos,
                        "ic_pearson": ic_pearson_oos,
                        "samples": int(len(y_true_dir)),
                        "best_threshold": 0.0,
                        "quality_check": {
                            "passed":
                            bool(f1 >= 0.3
                                 or (not np.isnan(auc) and auc >= 0.6)),
                            "issues": []
                        },
                    },
                    "regression_return": {
                        "rmse": oos_rmse,
                        "mae": oos_mae,
                        "samples": len(oos_df)
                    },
                    "uncertainty": {
                        "coverage_10_90": coverage,
                        "avg_interval_width": width,
                        "avg_confidence": conf
                    },
                    "regression_volatility": {
                        "rmse": oos_vol_rmse,
                        "mae": oos_vol_mae,
                        "samples": len(oos_df)
                    },
                }
            else:
                # In-sample directional metrics (derived from q50 regression) for visibility when no OOS period
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
                X_all = train_df[feature_cols].values
                y_ret_all = train_df["future_return"].values
                y_score_all = model_q50.model.predict(X_all)
                y_true_dir_all = (y_ret_all > 0).astype(int)
                y_pred_dir_all = (y_score_all > 0).astype(int)
                acc = float(accuracy_score(y_true_dir_all, y_pred_dir_all))
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_dir_all,
                    y_pred_dir_all,
                    average="binary",
                    zero_division=0)
                try:
                    auc = float(roc_auc_score(y_true_dir_all, y_score_all))
                except Exception:
                    auc = float("nan")
                try:
                    pr_auc = float(
                        average_precision_score(y_true_dir_all, y_score_all))
                except Exception:
                    pr_auc = float("nan")
                directional_metrics_train = {
                    "accuracy": acc,
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "auc": auc,
                    "pr_auc": pr_auc,
                    "samples": int(len(y_true_dir_all)),
                }

            # Directional metrics (derived from q50 regression model) - CV metrics
            # Import metrics that may not be available from earlier imports
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            # Import scipy.stats functions (already imported above for OOS, but need here for CV)
            from scipy.stats import spearmanr as spearmanr_cv, pearsonr as pearsonr_cv
            y_score = model_q50.model.predict(X_df.values)
            y_true_dir = (y_return.values > 0).astype(int)
            y_pred_dir = (y_score > 0).astype(int)
            try:
                auc = float(roc_auc_score(y_true_dir, y_score))
            except Exception:
                auc = None
            try:
                pr_auc = float(average_precision_score(y_true_dir, y_score))
            except Exception:
                pr_auc = None
            cm = confusion_matrix(y_true_dir, y_pred_dir).tolist()
            # Calculate IC (Information Coefficient) - Spearman correlation
            try:
                ic_spearman, _ = spearmanr_cv(y_return.values,
                                              y_score,
                                              nan_policy="omit")
                ic_spearman = float(
                    ic_spearman) if not np.isnan(ic_spearman) else None
            except Exception:
                ic_spearman = None
            try:
                ic_pearson, _ = pearsonr_cv(y_return.values, y_score)
                ic_pearson = float(
                    ic_pearson) if not np.isnan(ic_pearson) else None
            except Exception:
                ic_pearson = None
            accuracy = float(accuracy_score(y_true_dir, y_pred_dir))

            # Get sample count for warning context
            n_samples = len(y_true_dir)

            # ⚠️ DATA LEAKAGE WARNING: Check for suspiciously high accuracy
            # For different fb values, use different thresholds
            suspicious = False
            threshold = 0.90  # Default threshold

            if fb == 1:
                threshold = 0.90  # fb=1: >90% is very suspicious
                if accuracy > threshold:
                    suspicious = True
            elif fb <= 5:
                threshold = 0.85  # fb=2-5: >85% is suspicious
                if accuracy > threshold:
                    suspicious = True
            elif fb <= 15:
                threshold = 0.80  # fb=6-15: >80% is suspicious
                if accuracy > threshold:
                    suspicious = True
            else:
                # For larger fb (e.g., fb=45), high accuracy is even more suspicious
                # Especially for longer timeframes (e.g., 240T) where sample size is small
                threshold = 0.75  # fb>15: >75% is suspicious
                if accuracy > threshold:
                    suspicious = True

            # Additional check: if sample size is small (<1000) and accuracy is very high, be extra cautious
            if n_samples < 1000 and accuracy > 0.85:
                suspicious = True
                print("\n" + "=" * 70)
                print("🚨 严重警告：小样本 + 高准确率组合异常！")
                print("=" * 70)
                print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                print(f"   样本数量: {n_samples}")
                print(f"   方向准确率: {accuracy*100:.2f}%")
                print(f"   小样本（<1000）时高准确率可能是：")
                print(f"   1. 过拟合特定市场阶段（如单边上涨/下跌）")
                print(f"   2. 样本时间跨度短，市场状态单一")
                print(f"   3. 数据泄露或标签定义问题")
                print("=" * 70 + "\n")

            if suspicious:
                print("\n" + "=" * 70)
                print("🚨 严重警告：检测到可能的数据泄露或异常表现！")
                print("=" * 70)
                print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                print(f"   样本数量: {n_samples}")
                print(
                    f"   方向准确率: {accuracy*100:.2f}% (阈值: {threshold*100:.0f}%)"
                )
                print(
                    f"   预测未来 {fb} 根 {freq} K线的方向，准确率 {accuracy*100:.2f}% 在真实市场中极其罕见！"
                )
                print(f"\n   可能的原因：")
                print(f"   1. 特征中包含了未来信息（look-ahead bias）")
                print(f"   2. 标签泄露（label leakage）- 确认使用收盘价而非最高/最低价")
                print(f"   3. 数据时间顺序错误")
                print(f"   4. 数据预处理错误（如未正确shift或resample）")
                print(f"   5. 样本太少（{n_samples}条）导致过拟合特定市场阶段")
                print(f"   6. 市场处于极端单边行情（如2025 Q1 BTC单边上涨）")
                print(f"\n   建议检查：")
                print(
                    f"   - 确认标签定义：future_return = close[t+fb] / close[t] - 1（使用收盘价）"
                )
                print(f"   - 确认特征工程：所有特征都使用 shift(1) 避免未来信息")
                print(f"   - 检查数据resample是否正确（{freq}下应使用正确的聚合数据）")
                print(f"   - 增加样本数量或使用更长的时间跨度")
                print("=" * 70 + "\n")

            directional_metrics_cv = {
                "accuracy":
                accuracy,
                "precision":
                float(precision_score(y_true_dir, y_pred_dir,
                                      zero_division=0)),
                "recall":
                float(recall_score(y_true_dir, y_pred_dir, zero_division=0)),
                "f1":
                float(f1_score(y_true_dir, y_pred_dir, zero_division=0)),
                "auc":
                auc,
                "pr_auc":
                pr_auc,
                "ic_spearman":
                ic_spearman,
                "ic_pearson":
                ic_pearson,
                "best_threshold":
                0.0,
                "samples":
                int(len(y_true_dir)),
                "confusion_matrix":
                cm,
            }

            # Save artifacts and report (neutral naming, no 'baseline')
            # Use timestamped base directory for this training run to avoid mixing old data
            combo_dir = base_results_dir
            if len(freqs) > 1 or len(fbs) > 1:
                # If multiple configs, create subdirectory for each config
                combo_dir = os.path.join(base_results_dir, f"fb{fb}_tf{freq}")
            os.makedirs(combo_dir, exist_ok=True)
            model_q50.model.save_model(
                os.path.join(combo_dir, "return_q50_model.txt"))
            model_q10.model.save_model(
                os.path.join(combo_dir, "return_q10_model.txt"))
            model_q90.model.save_model(
                os.path.join(combo_dir, "return_q90_model.txt"))
            model_vol.model.save_model(
                os.path.join(combo_dir, "volatility_model.txt"))

            scaler_path = os.path.join(combo_dir, "scalers.pkl")
            if args.feature_type == "baseline":
                if feature_engineer is not None:
                    feature_engineer.save_scalers(scaler_path)
            else:
                if feature_engineer is not None and hasattr(
                        feature_engineer, "save_scalers"):
                    feature_engineer.save_scalers(scaler_path)

            with open(os.path.join(combo_dir, "features.txt"), "w") as f:
                f.write("\n".join(feature_cols))

            # Defer writing training_info.json until after potential base dir finalization

            # Finalize base directory name with train_start/train_end (YYYYMMDD) on first config
            if not base_dir_finalized:
                try:
                    _ts = train_df.index.min()
                    _te = train_df.index.max()
                    if _ts is not None and _te is not None:
                        _ts_s = _ts.strftime("%Y%m%d")
                        _te_s = _te.strftime("%Y%m%d")
                        finalized_dir = f"{training_timestamp}_{symbol_dir}_{args.feature_type}_{_ts_s}_{_te_s}"
                        finalized_path = os.path.join("results/training",
                                                      finalized_dir)
                        if finalized_path != base_results_dir:
                            os.makedirs(os.path.dirname(finalized_path),
                                        exist_ok=True)
                            # Rename the base directory (moves all existing files/subdirs)
                            os.rename(base_results_dir, finalized_path)
                            base_results_dir = finalized_path
                            print(
                                f"📁 Renamed results directory to: {base_results_dir}"
                            )
                        base_dir_finalized = True
                        # Update combo_dir to finalized base for subsequent saves in this loop iteration
                        if len(freqs) > 1 or len(fbs) > 1:
                            combo_dir = os.path.join(base_results_dir,
                                                     f"fb{fb}_tf{freq}")
                except Exception as _e:
                    print(
                        f"Note: Could not finalize results directory name: {_e}"
                    )

            # Recompute output paths after potential rename and ensure directory exists
            if len(freqs) > 1 or len(fbs) > 1:
                combo_dir = os.path.join(base_results_dir, f"fb{fb}_tf{freq}")
            else:
                combo_dir = base_results_dir
            os.makedirs(combo_dir, exist_ok=True)

            # Compose model_info and write training_info.json with finalized paths
            info_path = os.path.join(combo_dir, "training_info.json")
            # Extract OOS time range if available
            oos_start_str = None
            oos_end_str = None
            if len(oos_df) > 0 and not oos_df.empty:
                oos_start_str = oos_df.index.min().isoformat(
                ) if oos_df.index.min() is not None else None
                oos_end_str = oos_df.index.max().isoformat(
                ) if oos_df.index.max() is not None else None
            elif oos_start_dt is not None and oos_end_dt is not None:
                oos_start_str = oos_start_dt.isoformat()
                oos_end_str = oos_end_dt.isoformat()

            model_info = {
                "model_path":
                os.path.join(combo_dir, "return_q50_model.txt"),
                "scaler_path":
                os.path.join(combo_dir, "scalers.pkl"),
                "training_date":
                _dt.now().isoformat(),
                "symbol":
                symbols_str,
                "actual_start":
                feat_df.index.min().isoformat() if not feat_df.empty else None,
                "actual_end":
                feat_df.index.max().isoformat() if not feat_df.empty else None,
                "train_start":
                train_df.index.min().isoformat()
                if not train_df.empty else None,
                "train_end":
                train_df.index.max().isoformat()
                if not train_df.empty else None,
                "oos_start":
                oos_start_str,
                "oos_end":
                oos_end_str,
                "total_bars":
                len(feat_df),
                "train_bars":
                len(train_df),
                "oos_bars":
                len(oos_df) if len(oos_df) > 0 else 0,
                "oos_months":
                args.oos_months if len(oos_df) > 0 else 0,
                "timeframes": {
                    freq: len(feat_df)
                },
                "price_range": [
                    float(feat_df["close"].min()) if not feat_df.empty else 0,
                    float(feat_df["close"].max()) if not feat_df.empty else 0,
                ],
                "metrics": {
                    "stage2": {
                        freq: q50_metrics
                    },
                    "q10": {
                        freq: q10_metrics
                    },
                    "q90": {
                        freq: q90_metrics
                    },
                    "volatility": {
                        freq: vol_metrics
                    },
                    "directional_train": {
                        freq: directional_metrics_train
                    } if directional_metrics_train else {},
                    "directional_cv": {
                        freq: directional_metrics_cv
                    } if directional_metrics_cv else {},
                },
                "feature_engineering":
                "BaselineFeatureEngineer" if args.feature_type == "baseline"
                else f"ComprehensiveFeatureEngineer({args.feature_type})",
                "feature_type":
                args.feature_type,
                "forward_bars":
                fb,
                "timeframe":
                freq,
                "data_files":
                files,
                "ar1_info": {
                    "ar1_phi":
                    float(ar1_phi)
                    if ar1_phi is not None and not np.isnan(ar1_phi) else None,
                    "ar1_autocorr_before":
                    float(ar1_autocorr) if ar1_autocorr is not None
                    and not np.isnan(ar1_autocorr) else None,
                    "ar1_autocorr_after":
                    float(ar1_autocorr_after) if ar1_autocorr_after is not None
                    and not np.isnan(ar1_autocorr_after) else None,
                    "continuity_bias_removed":
                    bool(ar1_phi is not None),
                } if ar1_phi is not None else None,
            }
            if oos_metrics:
                model_info["oos_metrics"] = oos_metrics
            with open(info_path, "w") as f:
                json.dump(model_info, f, indent=2, default=str)

            # Write a compact training HTML report (self-contained)
            report_path = os.path.join(combo_dir, "training_report.html")
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_json = json.load(f)

                # Extract AR(1) information if available (for fb=1)
                ar1_info = info_json.get("ar1_info")

                # Extract metrics for all 4 regression models
                q10_metrics = info_json.get("metrics", {}).get("q10", {})
                q50_metrics = info_json.get("metrics", {}).get("stage2", {})
                q90_metrics = info_json.get("metrics", {}).get("q90", {})
                vol_metrics = info_json.get("metrics",
                                            {}).get("volatility", {})

                # Quantile Loss section (q10, q50, q90)
                quantile_rows = []
                for tf in q50_metrics.keys():
                    q10_val = q10_metrics.get(tf,
                                              {}).get('cv_quantile_loss',
                                                      'N/A')
                    q50_val = q50_metrics.get(tf,
                                              {}).get('cv_quantile_loss',
                                                      'N/A')
                    q90_val = q90_metrics.get(tf,
                                              {}).get('cv_quantile_loss',
                                                      'N/A')

                    # Format values with 6 decimal places, but show scientific notation for very small values
                    def fmt_val(v):
                        if v == 'N/A' or v is None:
                            return 'N/A'
                        try:
                            fv = float(v)
                            # If value is very small (< 1e-6) or exactly 0, use scientific notation or show more precision
                            if abs(fv) < 1e-6 and fv != 0.0:
                                return f"{fv:.2e}"
                            elif fv == 0.0:
                                return "0.000000"
                            else:
                                return f"{fv:.6f}"
                        except (ValueError, TypeError):
                            return str(v)

                    quantile_rows.append(
                        f"<tr><td>{tf}</td><td>{fmt_val(q10_val)}</td><td>{fmt_val(q50_val)}</td><td>{fmt_val(q90_val)}</td></tr>"
                    )
                # AR(1) information section (for fb=1)
                ar1_section = ""
                if ar1_info:
                    ar1_phi = ar1_info.get("ar1_phi")
                    ar1_autocorr_before = ar1_info.get(
                        "ar1_autocorr_before") or ar1_info.get(
                            "ar1_autocorr")  # backward compatibility
                    ar1_autocorr_after = ar1_info.get("ar1_autocorr_after")
                    removed = ar1_info.get("continuity_bias_removed", False)

                    ar1_rows = []
                    if ar1_phi is not None:
                        ar1_rows.append(
                            f"<tr><td>AR(1) 系数 (φ)</td><td>{ar1_phi:.4f}</td></tr>"
                        )
                    if ar1_autocorr_before is not None:
                        ar1_rows.append(
                            f"<tr><td>Lag-1 自相关系数（移除前）</td><td>{ar1_autocorr_before:.4f}</td></tr>"
                        )
                    if ar1_autocorr_after is not None:
                        ar1_rows.append(
                            f"<tr><td>Lag-1 自相关系数（移除后）</td><td>{ar1_autocorr_after:.4f}</td></tr>"
                        )
                        if ar1_autocorr_before is not None:
                            reduction = abs(ar1_autocorr_before) - abs(
                                ar1_autocorr_after)
                            reduction_pct = (
                                reduction / abs(ar1_autocorr_before) *
                                100) if abs(ar1_autocorr_before) > 0 else 0
                            ar1_rows.append(
                                f"<tr><td>自相关性减少</td><td>{reduction:.4f} ({reduction_pct:.1f}%)</td></tr>"
                            )
                    ar1_rows.append(
                        f"<tr><td>连续性偏差已移除</td><td>{'✅ 是' if removed else '❌ 否'}</td></tr>"
                    )

                    if ar1_phi is not None and abs(ar1_phi) > 0.3:
                        if ar1_autocorr_after is not None and ar1_autocorr_before is not None:
                            if abs(ar1_autocorr_after) < abs(
                                    ar1_autocorr_before):
                                reduction = abs(ar1_autocorr_before) - abs(
                                    ar1_autocorr_after)
                                reduction_pct = (
                                    reduction / abs(ar1_autocorr_before) *
                                    100) if abs(ar1_autocorr_before) > 0 else 0
                                ar1_warning = f"<div style='background-color:#d4edda;border-left:4px solid #28a745;padding:10px;margin:10px 0;border-radius:4px;'><strong>✅ 已处理:</strong> 检测到高自相关性（|φ| = {ar1_phi:.4f}），已通过AR(1)残差移除。移除后自相关性从 {ar1_autocorr_before:.4f} 降至 {ar1_autocorr_after:.4f}（减少 {reduction_pct:.1f}%）。</div>"
                            else:
                                ar1_warning = f"<div style='background-color:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;border-radius:4px;'><strong>⚠️ 警告:</strong> 检测到高自相关性（|φ| = {ar1_phi:.4f}），这解释了为什么IC和准确率异常高。AR(1)残差已应用，但自相关性未显著降低（移除前: {ar1_autocorr_before:.4f}, 移除后: {ar1_autocorr_after:.4f}），可能存在其他数据泄露或需要更强的去相关处理。</div>"
                        else:
                            ar1_warning = f"<div style='background-color:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;border-radius:4px;'><strong>⚠️ 警告:</strong> 检测到高自相关性（|φ| = {ar1_phi:.4f}），这解释了为什么IC和准确率异常高。AR(1)残差已应用。</div>"
                    else:
                        ar1_warning = ""

                    ar1_explanation = "<p><em>AR(1) 信息说明：</em></p><ul style='margin:10px 0;padding-left:20px;'><li><strong>AR(1) 系数 (φ)</strong>：衡量价格连续性的强度，值越高表示相邻K线价格越相关</li><li><strong>Lag-1 自相关系数（移除前/后）</strong>：收益率序列的一阶自相关性，用于估计AR(1)模型参数。移除后自相关性应显著降低</li><li><strong>连续性偏差</strong>：高IC/准确率可能来自价格连续性而非真实预测能力。使用AR(1)残差可以移除这部分偏差</li><li><strong>自相关性减少</strong>：移除AR(1)成分后自相关性的减少幅度，用于验证去连续性处理的有效性</li></ul>"

                    # Get current fb value - it should be available in the current scope
                    # Since we're in a loop over fb values, use the fb variable directly
                    current_fb = fb if 'fb' in locals() else "N/A"
                    ar1_section = (
                        f"<h2>🔍 AR(1) 价格连续性分析 (fb={current_fb})</h2>" +
                        ar1_explanation + ar1_warning +
                        "<table><tr><th>指标</th><th>值</th></tr>" +
                        "".join(ar1_rows) + "</table>")

                quantile_section = ""
                if quantile_rows:
                    # Check for anomalies
                    warnings_list = []
                    for tf in q50_metrics.keys():
                        q10_val = q10_metrics.get(tf, {}).get(
                            'cv_quantile_loss', 'N/A')
                        q50_val = q50_metrics.get(tf, {}).get(
                            'cv_quantile_loss', 'N/A')
                        q90_val = q90_metrics.get(tf, {}).get(
                            'cv_quantile_loss', 'N/A')
                        try:
                            q10_f = float(
                                q10_val) if q10_val != 'N/A' else None
                            q50_f = float(
                                q50_val) if q50_val != 'N/A' else None
                            q90_f = float(
                                q90_val) if q90_val != 'N/A' else None
                            if q10_f is not None and q50_f is not None and q90_f is not None:
                                # Check for Q50 loss = 0 or suspiciously small (likely calculation error or data issue)
                                if q50_f == 0.0 or (q50_f < 1e-6
                                                    and q10_f > 1e-6):
                                    warnings_list.append(
                                        f"⚠️ {tf}: Q50 loss ({q50_f:.6f}) 异常小或为0！这可能是计算错误、数据问题或模型预测完全正确（不太可能）。请检查数据或计算逻辑。"
                                    )
                                # Check if Q50 loss violates quantile regression property (should be <= Q10 and Q90)
                                elif q50_f > q10_f or q50_f > q90_f:
                                    warnings_list.append(
                                        f"⚠️ {tf}: Q50 loss ({q50_f:.6f}) > Q10/Q90 loss（Q10={q10_f:.6f}, Q90={q90_f:.6f}），违反quantile regression性质！"
                                    )
                                # Check if Q50 loss is abnormally large
                                if q50_f > 1.0:
                                    warnings_list.append(
                                        f"⚠️ {tf}: Q50 loss ({q50_f:.6f})异常大！如果收益率是比例（±0.05），loss应在0.01-0.1量级，可能存在单位问题。"
                                    )
                        except (ValueError, TypeError):
                            pass

                    warning_html = ""
                    if warnings_list:
                        warning_html = "<div style='background-color:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;border-radius:4px;'><strong>⚠️ 警告:</strong><ul style='margin:5px 0;padding-left:20px;'>" + "".join(
                            [f"<li>{w}</li>"
                             for w in warnings_list]) + "</ul></div>"

                    quantile_section = (
                        "<h2>📊 Quantile Loss (CV)</h2>"
                        "<p><em>单位: Pinball Loss (与收益率比例单位相同，例如 0.01 表示 1% 的平均误差)</em></p>"
                        "<p><em>注意: 如果loss值>1.0，可能存在单位问题或数据异常。正常情况下，收益率在±5%范围内时，loss应在0.01-0.1量级。Q50 loss应≤Q10/Q90 loss。</em></p>"
                        + warning_html +
                        "<table><tr><th>Timeframe</th><th>Quantile Loss 0.1 (q10)</th><th>Quantile Loss 0.5 (q50)</th><th>Quantile Loss 0.9 (q90)</th></tr>"
                        + "".join(quantile_rows) + "</table>")

                # Volatility CV Metrics section
                vol_rows = []
                for tf, m in vol_metrics.items():
                    cv_rmse = m.get('cv_rmse', 'N/A')
                    cv_mse = m.get('cv_mse', 'N/A')

                    def fmt_val(v):
                        if v == 'N/A' or v is None:
                            return 'N/A'
                        try:
                            return f"{float(v):.6f}"
                        except (ValueError, TypeError):
                            return str(v)

                    vol_rows.append(
                        f"<tr><td>{tf}</td><td>{fmt_val(cv_rmse)}</td><td>{fmt_val(cv_mse)}</td></tr>"
                    )
                vol_section = ""
                if vol_rows:
                    # Check for anomalies
                    vol_warnings_list = []
                    for tf, m in vol_metrics.items():
                        cv_rmse = m.get('cv_rmse', 'N/A')
                        try:
                            rmse_f = float(
                                cv_rmse) if cv_rmse != 'N/A' else None
                            if rmse_f is not None and rmse_f > 1.0:
                                vol_warnings_list.append(
                                    f"⚠️ {tf}: CV RMSE ({rmse_f:.2f})异常大！如果波动率是比例（0.01=1%），RMSE应在0.01-0.1量级，可能存在单位问题。"
                                )
                        except (ValueError, TypeError):
                            pass

                    vol_warning_html = ""
                    if vol_warnings_list:
                        vol_warning_html = "<div style='background-color:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;border-radius:4px;'><strong>⚠️ 警告:</strong><ul style='margin:5px 0;padding-left:20px;'>" + "".join(
                            [f"<li>{w}</li>"
                             for w in vol_warnings_list]) + "</ul></div>"

                    vol_section = (
                        "<h2>📈 Volatility (Regression) CV Metrics</h2>"
                        "<p><em>单位: RMSE/MSE - 波动率比例 (与收益率比例单位相同，例如 0.01 表示 1%)</em></p>"
                        "<p><em>注意: 如果RMSE值>1.0，可能存在单位问题。正常情况下，波动率在1-5%范围内时，RMSE应在0.01-0.1量级。</em></p>"
                        + vol_warning_html +
                        "<table><tr><th>Timeframe</th><th>CV RMSE</th><th>CV MSE</th></tr>"
                        + "".join(vol_rows) + "</table>")

                # Directional and IC metrics (CV)
                directional_cv_metrics = info_json.get("metrics", {}).get(
                    "directional_cv", {})
                directional_section = ""
                if directional_cv_metrics:
                    dir_rows = []
                    for tf, dir_metrics in directional_cv_metrics.items():
                        if isinstance(dir_metrics, dict):

                            def fmt_pct(v):
                                if v == 'N/A' or v is None:
                                    return 'N/A'
                                try:
                                    return f"{float(v)*100:.2f}%"
                                except (ValueError, TypeError):
                                    return str(v)

                            def fmt_corr(v):
                                if v == 'N/A' or v is None:
                                    return 'N/A'
                                try:
                                    return f"{float(v):.4f}"
                                except (ValueError, TypeError):
                                    return str(v)

                            acc = fmt_pct(dir_metrics.get('accuracy'))
                            prec = fmt_pct(dir_metrics.get('precision'))
                            rec = fmt_pct(dir_metrics.get('recall'))
                            f1 = fmt_pct(dir_metrics.get('f1'))
                            auc_val = fmt_pct(
                                dir_metrics.get('auc')) if dir_metrics.get(
                                    'auc') is not None else 'N/A'
                            ic_spearman = fmt_corr(
                                dir_metrics.get('ic_spearman'))
                            ic_pearson = fmt_corr(
                                dir_metrics.get('ic_pearson'))
                            dir_rows.append(
                                f"<tr><td>{tf}</td><td>{acc}</td><td>{prec}</td><td>{rec}</td><td>{f1}</td><td>{auc_val}</td><td>{ic_spearman}</td><td>{ic_pearson}</td></tr>"
                            )
                    if dir_rows:
                        directional_section = (
                            "<h2>🎯 方向预测指标 (CV)</h2>"
                            "<p><em>方向准确率: 预测涨跌方向的准确率 | IC: 信息系数 (Information Coefficient)，预测值与实际值的相关性</em></p>"
                            "<table><tr><th>Timeframe</th><th>方向准确率</th><th>精确率</th><th>召回率</th><th>F1</th><th>AUC</th><th>IC (Spearman)</th><th>IC (Pearson)</th></tr>"
                            + "".join(dir_rows) + "</table>")

                # OOS section - regression metrics and directional metrics
                oos_section = ""
                if info_json.get("oos_metrics"):
                    oos_metrics = info_json["oos_metrics"]
                    oos_rows = []
                    oos_dir_rows = []

                    # Directional OOS metrics
                    if "directional_oos" in oos_metrics:
                        dir_oos = oos_metrics["directional_oos"]

                        def fmt_pct(v):
                            if v == 'N/A' or v is None:
                                return 'N/A'
                            try:
                                return f"{float(v)*100:.2f}%"
                            except (ValueError, TypeError):
                                return str(v)

                        def fmt_corr(v):
                            if v == 'N/A' or v is None:
                                return 'N/A'
                            try:
                                return f"{float(v):.4f}"
                            except (ValueError, TypeError):
                                return str(v)

                        oos_dir_rows.append(
                            f"<tr><td>方向准确率</td><td>{fmt_pct(dir_oos.get('accuracy'))}</td></tr>"
                        )
                        oos_dir_rows.append(
                            f"<tr><td>精确率</td><td>{fmt_pct(dir_oos.get('precision'))}</td></tr>"
                        )
                        oos_dir_rows.append(
                            f"<tr><td>召回率</td><td>{fmt_pct(dir_oos.get('recall'))}</td></tr>"
                        )
                        oos_dir_rows.append(
                            f"<tr><td>F1</td><td>{fmt_pct(dir_oos.get('f1'))}</td></tr>"
                        )
                        if dir_oos.get('auc') is not None:
                            oos_dir_rows.append(
                                f"<tr><td>AUC</td><td>{fmt_pct(dir_oos.get('auc'))}</td></tr>"
                            )
                        if dir_oos.get('ic_spearman') is not None:
                            oos_dir_rows.append(
                                f"<tr><td>IC (Spearman)</td><td>{fmt_corr(dir_oos.get('ic_spearman'))}</td></tr>"
                            )
                        if dir_oos.get('ic_pearson') is not None:
                            oos_dir_rows.append(
                                f"<tr><td>IC (Pearson)</td><td>{fmt_corr(dir_oos.get('ic_pearson'))}</td></tr>"
                            )

                    # Q50 OOS metrics (regression_return)
                    def fmt_val(v):
                        if v == 'N/A' or v is None:
                            return 'N/A'
                        try:
                            return f"{float(v):.6f}"
                        except (ValueError, TypeError):
                            return str(v)

                    if "regression_return" in oos_metrics:
                        reg_ret = oos_metrics["regression_return"]
                        oos_rows.append(
                            f"<tr><td>Q50 RMSE</td><td>{fmt_val(reg_ret.get('rmse'))} <em>(收益率比例)</em></td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Q50 MAE</td><td>{fmt_val(reg_ret.get('mae'))} <em>(收益率比例)</em></td></tr>"
                        )
                    # Also check for legacy q50 key
                    elif "q50" in oos_metrics:
                        q50_oos = oos_metrics["q50"]
                        oos_rows.append(
                            f"<tr><td>Q50 RMSE</td><td>{fmt_val(q50_oos.get('oos_rmse'))} <em>(收益率比例)</em></td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Q50 MAE</td><td>{fmt_val(q50_oos.get('oos_mae'))} <em>(收益率比例)</em></td></tr>"
                        )

                    # Q10 OOS metrics
                    if "q10" in oos_metrics:
                        q10_oos = oos_metrics["q10"]
                        oos_rows.append(
                            f"<tr><td>Q10 Quantile Loss</td><td>{fmt_val(q10_oos.get('oos_quantile_loss'))} <em>(收益率比例)</em></td></tr>"
                        )

                    # Q90 OOS metrics
                    if "q90" in oos_metrics:
                        q90_oos = oos_metrics["q90"]
                        oos_rows.append(
                            f"<tr><td>Q90 Quantile Loss</td><td>{fmt_val(q90_oos.get('oos_quantile_loss'))} <em>(收益率比例)</em></td></tr>"
                        )

                    # Volatility OOS metrics (regression_volatility)
                    if "regression_volatility" in oos_metrics:
                        vol_oos = oos_metrics["regression_volatility"]
                        oos_rows.append(
                            f"<tr><td>Volatility RMSE</td><td>{fmt_val(vol_oos.get('rmse'))} <em>(波动率比例)</em></td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Volatility MAE</td><td>{fmt_val(vol_oos.get('mae'))} <em>(波动率比例)</em></td></tr>"
                        )
                    # Also check for legacy volatility key
                    elif "volatility" in oos_metrics:
                        vol_oos = oos_metrics["volatility"]
                        oos_rows.append(
                            f"<tr><td>Volatility RMSE</td><td>{fmt_val(vol_oos.get('oos_rmse'))} <em>(波动率比例)</em></td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Volatility MAE</td><td>{fmt_val(vol_oos.get('oos_mae'))} <em>(波动率比例)</em></td></tr>"
                        )

                    # Uncertainty metrics
                    if "uncertainty" in oos_metrics:
                        unc_oos = oos_metrics["uncertainty"]

                        def fmt_pct(v):
                            if v == 'N/A' or v is None:
                                return 'N/A'
                            try:
                                return f"{float(v)*100:.2f}%"
                            except (ValueError, TypeError):
                                return str(v)

                        coverage = unc_oos.get(
                            'coverage_10_90') or unc_oos.get('coverage')
                        width = unc_oos.get('avg_interval_width'
                                            ) or unc_oos.get('interval_width')
                        conf = unc_oos.get('avg_confidence') or unc_oos.get(
                            'confidence')
                        oos_rows.append(
                            f"<tr><td>Coverage (q10-q90)</td><td>{fmt_pct(coverage)}</td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Interval Width</td><td>{fmt_val(width)} <em>(收益率比例)</em></td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Confidence</td><td>{fmt_val(conf)}</td></tr>"
                        )

                    # Signal strength
                    if "signal" in oos_metrics:
                        sig_oos = oos_metrics["signal"]
                        oos_rows.append(
                            f"<tr><td>Avg Signal Strength (|q50|/vol)</td><td>{fmt_val(sig_oos.get('avg_signal_strength'))}</td></tr>"
                        )

                    oos_sections = []
                    if oos_dir_rows:
                        oos_sections.append(
                            "<h2>🎯 OOS 方向预测指标</h2>"
                            "<table><tr><th>Metric</th><th>Value</th></tr>" +
                            "".join(oos_dir_rows) + "</table>")
                    if oos_rows:
                        oos_sections.append(
                            "<h2>📉 OOS 回归指标</h2>"
                            "<table><tr><th>Metric</th><th>Value</th></tr>" +
                            "".join(oos_rows) + "</table>")
                    if oos_sections:
                        oos_section = "\n".join(oos_sections)

                # Artifacts section - include all 4 models
                model_dir = os.path.dirname(info_json.get('model_path', ''))
                artifacts_list = [
                    f"<li>Q50 Model (median): {os.path.join(model_dir, 'return_q50_model.txt')}</li>",
                    f"<li>Q10 Model (10% quantile): {os.path.join(model_dir, 'return_q10_model.txt')}</li>",
                    f"<li>Q90 Model (90% quantile): {os.path.join(model_dir, 'return_q90_model.txt')}</li>",
                    f"<li>Volatility Model: {os.path.join(model_dir, 'volatility_model.txt')}</li>",
                    f"<li>Scalers: {info_json.get('scaler_path')}</li>"
                ]

                html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Training Report</title>
                <style>
                body{{font-family:Arial,sans-serif;margin:24px;color:#222;background:#f5f5f5}}
                .container{{max-width:1200px;margin:0 auto;background:white;padding:24px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
                h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
                h2{{color:#34495e;margin-top:30px;margin-bottom:15px;padding-left:10px;border-left:4px solid #3498db}}
                table{{border-collapse:collapse;width:100%;margin:15px 0;background:white}}
                th{{background:#3498db;color:#fff;padding:12px;text-align:left;font-weight:600}}
                td{{border:1px solid #ddd;padding:10px}}
                tr:nth-child(even){{background:#f9f9f9}}
                tr:hover{{background:#f0f8ff}}
                p{{line-height:1.6;margin:10px 0}}
                em{{color:#7f8c8d;font-size:0.9em}}
                ul{{line-height:1.8}}
                .info-section{{background:#ecf0f1;padding:15px;border-radius:5px;margin:15px 0}}
                </style>
                </head><body>
                <div class="container">
                <h1>📊 训练报告 (Training Report)</h1>
                <div class="info-section">
                <p><strong>Symbol:</strong> {info_json.get('symbol')} &nbsp; <strong>Period:</strong> {info_json.get('actual_start', 'N/A')} → {info_json.get('actual_end', 'N/A')}</p>
                <p><strong>Total Bars:</strong> {info_json.get('total_bars', 0)} &nbsp; <strong>Train Bars:</strong> {info_json.get('train_bars', 'N/A')}</p>
                <p><strong>Feature Type:</strong> {info_json.get('feature_type', 'N/A')} &nbsp; <strong>Forward Bars:</strong> {info_json.get('forward_bars', 'N/A')}</p>
                <p><strong>📝 预测输出说明:</strong></p>
                <ul>
                  <li><strong>Q50 (中位数预测):</strong> 预测未来{info_json.get('forward_bars', 'N/A')}个bar的收益率，单位：收益率比例（例如 0.01 表示 1%）</li>
                  <li><strong>Q10/Q90 (分位数预测):</strong> 用于构建不确定性区间，单位：收益率比例</li>
                  <li><strong>Volatility:</strong> 预测未来波动率，单位：波动率比例（例如 0.01 表示 1%）</li>
                </ul>
                </div>
                {ar1_section if 'ar1_section' in locals() and ar1_section else ""}
                {directional_section}
                {quantile_section}
                {vol_section}
                {oos_section}
                <h2>📁 模型文件 (Artifacts)</h2>
                <ul>
                  {''.join(artifacts_list)}
                </ul>
                </div>
                </body></html>"""
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"📝 HTML report written to: {report_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Note: Could not write compact training report: {exc}")

    # Generate training summary report for this training run
    # Generate report in the timestamped directory to avoid mixing with old data
    try:
        from ml_trading.pipeline.training.generate_summary_report import generate_summary_report
        # Generate report in the timestamped base directory
        generate_summary_report(base_results_dir, None)
        print(f"\n📊 Training summary report generated in: {base_results_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"Note: Could not generate training summary report: {exc}")


if __name__ == "__main__":
    main()
