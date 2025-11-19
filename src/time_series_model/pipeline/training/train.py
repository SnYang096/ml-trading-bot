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
import shutil
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from scipy.interpolate import interp1d

from data_tools.rolling_data import load_parquet_file
from data_tools.baseline_features import (
    engineer_baseline_features,
    get_baseline_feature_columns,
)
from data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer,
    get_feature_columns_by_type,
)
from time_series_model.models.lightgbm_model import LightGBMTrainer
from time_series_model.models.quant_trading_model import TradingModelPipeline
from time_series_model.pipeline.training.preprocessing import RobustWinsorizer
from time_series_model.pipeline.training.quantile_model_trainer import (
    QuantileModelTrainer,
)
from time_series_model.pipeline.training.classification_model_trainer import (
    ClassificationModelTrainer,
)
from .label_utils import rolling_rms_volatility, future_volatility_label
import joblib


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

    # 🔍 Multi-asset validation: Check data distribution and potential issues
    if "symbol" in merged.columns:
        symbol_counts = merged["symbol"].value_counts()
        print(f"\n   📊 Multi-asset data distribution:")
        for symbol, count in symbol_counts.items():
            pct = count / len(merged) * 100
            print(f"      {symbol}: {count:,} samples ({pct:.1f}%)")

        # Check for severe imbalance (>80% from one asset)
        max_pct = symbol_counts.max() / len(merged) * 100
        if max_pct > 80:
            print(f"\n   ⚠️  警告：数据分布严重不平衡！")
            print(f"      最大资产占比: {max_pct:.1f}%")
            print(f"      这可能导致模型偏向数据量更大的资产")
            print(f"      💡 建议：")
            print(f"         - 考虑使用样本权重平衡不同资产")
            print(f"         - 或分别训练每个资产的模型")
            print(f"         - 或对每个资产进行下采样以平衡数据量")

        # Check price level differences (if close price exists)
        if "close" in merged.columns:
            print(f"\n   📊 Price level check (for feature validation):")
            for symbol in symbol_counts.index:
                symbol_data = merged[merged["symbol"] == symbol]
                if len(symbol_data) > 0:
                    close_mean = float(symbol_data["close"].mean())
                    close_std = float(symbol_data["close"].std())
                    print(f"      {symbol}: mean={close_mean:.2f}, std={close_std:.2f}")

            # Check if features use returns (pct_change) or raw prices
            # This is a warning, not an error, as features should be asset-agnostic
            print(f"\n   💡 多资产合并训练注意事项：")
            print(f"      - 确保所有价格相关特征使用收益率（pct_change）或标准化")
            print(f"      - 技术指标应使用相对值（如 RSI、MACD）而非绝对值")
            print(f"      - 如果特征工程正确，价格水平差异不应影响模型")
            print(f"      - 但需要验证特征确实使用了收益率/标准化")

    # Ensure DatetimeIndex
    if not isinstance(merged.index, pd.DatetimeIndex):
        if "timestamp" in merged.columns:
            merged.set_index("timestamp", inplace=True)
        else:
            raise ValueError(
                "Merged data must have DatetimeIndex or 'timestamp' column"
            )

    return merged


def _compute_direction_threshold(
    y_score: np.ndarray, y_true_dir: np.ndarray, method: str = "f1_optimize"
) -> float:
    """
    计算用于方向预测的动态阈值。

    解决固定阈值0的问题：当模型预测值整体偏负时，固定阈值0会导致所有预测为跌，
    即使模型对涨跌的排序是正确的（AUC高但F1=0）。

    Args:
        y_score: 回归模型的预测值（连续值）
        y_true_dir: 真实方向标签（1=涨，0=跌）
        method: 阈值计算方法
            - "median": 使用预测值的中位数作为阈值
            - "f1_optimize": 在多个百分位点中寻找最大化F1分数的阈值（推荐）
            - "zero": 使用固定阈值0（原始方法，用于对比）

    Returns:
        计算得到的阈值
    """
    if method == "zero":
        return 0.0
    elif method == "median":
        return float(np.median(y_score))
    elif method == "f1_optimize":
        from sklearn.metrics import f1_score

        # 如果所有预测值相同或只有一个值，使用中位数
        if len(np.unique(y_score)) <= 1:
            return float(np.median(y_score))

        # 在多个百分位点中寻找最佳阈值
        percentiles = np.linspace(10, 90, 17)  # 10, 15, 20, ..., 90
        thresholds = np.percentile(y_score, percentiles)

        best_thresh = 0.0
        best_f1 = 0.0

        # 尝试固定阈值0
        f1_zero = f1_score(y_true_dir, (y_score > 0).astype(int), zero_division="warn")
        if f1_zero > best_f1:
            best_f1 = f1_zero
            best_thresh = 0.0

        # 尝试各个百分位点阈值
        for thresh in thresholds:
            y_pred = (y_score > thresh).astype(int)
            # 确保至少有一个正类和一个负类预测
            if len(np.unique(y_pred)) < 2:
                continue
            f1 = f1_score(y_true_dir, y_pred, zero_division="warn")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # 如果所有阈值都导致F1=0，回退到中位数
        if best_f1 == 0.0:
            return float(np.median(y_score))

        return float(best_thresh)
    else:
        raise ValueError(
            f"Unknown threshold method: {method}. Use 'median', 'f1_optimize', or 'zero'"
        )


def _extract_feature_importance_df(
    model: Optional[LightGBMTrainer], feature_cols: Optional[List[str]]
) -> Optional[pd.DataFrame]:
    """Return top gain-based feature importance for a trained LightGBM model."""
    if not model or not hasattr(model, "model") or model.model is None:
        return None
    try:
        booster = model.model
        gains = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        if gains is None or names is None:
            return None
        if len(gains) != len(names):
            if not feature_cols or len(gains) != len(feature_cols):
                return None
            names = feature_cols
        df_imp = pd.DataFrame({"feature": names, "importance": gains})
        df_imp = df_imp.groupby("feature", as_index=False)["importance"].sum()
        df_imp = df_imp.sort_values("importance", ascending=False).head(100)
        return df_imp
    except Exception:
        return None


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
        resampled_dict["taker_buy_ratio"] = (
            df["taker_buy_ratio"].resample(timeframe).mean()
        )
    if "cvd" in df.columns:
        resampled_dict["cvd"] = df["cvd"].resample(timeframe).last()
    if "trade_count" in df.columns:
        resampled_dict["trade_count"] = df["trade_count"].resample(timeframe).sum()

    resampled = pd.DataFrame(resampled_dict)

    # 🔧 CRITICAL FIX: 正确处理无交易窗口，避免样本选择偏差
    # 根据用户建议：价格类用 ffill()，事件类用 fillna(0)，比率类用 fillna(0) + has_trade 特征
    # 这样可以保留"静默时段"，让模型学习真实市场状态，避免幸存者偏差

    # 1. 价格类数据：使用前向填充（ffill），保持价格连续性
    # 无交易时，价格应该保持最后已知价格（这是合理的市场假设）
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].ffill()

    # 2. 事件类数据：使用 fillna(0)，表示"没有发生 = 0"
    # 无交易时，成交量、买卖量、交易次数等应该为 0
    event_cols = ["volume", "buy_qty", "sell_qty", "trade_count"]
    for col in event_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].fillna(0)

    # 3. CVD（累积成交量差）：使用 fillna(0)
    # 无交易时，累积差值为 0（或保持最后已知值，但 fillna(0) 更安全）
    if "cvd" in resampled.columns:
        # CVD 是累积值，应该保持最后已知值，但如果一开始就是 NaN，则设为 0
        resampled["cvd"] = resampled["cvd"].ffill().fillna(0)

    # 4. 比率类数据：使用 fillna(0) + 添加 has_trade 特征
    # 无交易时，比率无意义，设为 0，但添加 has_trade 标志帮助模型区分
    if "taker_buy_ratio" in resampled.columns:
        # 添加 has_trade 特征（强烈推荐，帮助模型区分"有交易"和"无交易"状态）
        resampled["has_trade"] = (resampled["volume"] > 0).astype(int)
        # 对于无交易的窗口，taker_buy_ratio 设为 0
        resampled["taker_buy_ratio"] = resampled["taker_buy_ratio"].fillna(0)

    # 5. 删除仍存在的 NaN（如开盘价一开始就缺失，无法 ffill）
    # 只对必要的列（close）进行 dropna，确保至少有一个价格数据
    before_dropna = len(resampled)
    resampled = resampled.dropna(subset=["close"]).copy()
    after_dropna = len(resampled)

    # 6. 验证价格数据的有效性（必须 > 0）
    # 但不再过滤 volume=0，因为这是真实的市场状态
    invalid_price_mask = (
        (resampled["close"] <= 0)
        | (resampled["open"] <= 0)
        | (resampled["high"] <= 0)
        | (resampled["low"] <= 0)
    )
    if invalid_price_mask.any():
        print(
            f"   ⚠️  Warning: 发现 {invalid_price_mask.sum()} 个无效价格（<=0），已删除"
        )
        resampled = resampled[~invalid_price_mask].copy()

    # 7. 诊断信息：统计无交易窗口的数量
    if "volume" in resampled.columns:
        zero_volume_count = (resampled["volume"] == 0).sum()
        if zero_volume_count > 0:
            zero_volume_ratio = zero_volume_count / len(resampled) * 100
            print(
                f"   📊 统计: {zero_volume_count} 个无交易窗口（{zero_volume_ratio:.1f}%），"
                f"已使用 fillna(0) 保留（避免样本选择偏差）"
            )

    if before_dropna > after_dropna:
        dropped_ratio = (before_dropna - after_dropna) / before_dropna * 100
        print(
            f"   📊 统计: 删除了 {before_dropna - after_dropna} 个完全缺失价格的窗口（{dropped_ratio:.1f}%）"
        )

    return resampled


def _collect_files(
    data: List[str],
    data_dir: str | None,
    start: str | None,
    end: str | None,
    symbols: str | None = None,
) -> List[str]:
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
        filtered = []
        for symbol in symbol_list:
            normalized = symbol.upper().replace("-", "").replace("/", "")
            if not normalized.endswith("USDT"):
                normalized = f"{normalized}USDT"
            legacy_symbol = normalized.replace("USDT", "-USD")
            prefixes = {
                normalized,
                legacy_symbol,
                legacy_symbol.replace("-", ""),
            }
            for p in files:
                fn = os.path.basename(p).upper()
                if any(
                    fn.startswith(prefix.upper())
                    or fn.startswith(prefix.replace("-", "_").upper())
                    for prefix in prefixes
                ):
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
    """
    DEPRECATED: This training script is deprecated.
    Use 'make rolling' instead for rolling training, which provides better evaluation
    through expanding window training and multiple model checkpoints.

    This script is kept for backward compatibility and utility functions only.
    """
    import warnings

    warnings.warn(
        "train.py main() is deprecated. Use 'make rolling' instead for production training.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="DEPRECATED: Regression training (returns + uncertainty + volatility). Use 'make rolling' instead."
    )
    parser.add_argument(
        "--data", type=str, action="append", default=[], help="Parquet file(s) to use"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Directory containing parquet files"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol(s) metadata for report. Can be comma-separated (e.g., BTCUSDT,ETHUSDT,SOLUSDT) for multi-asset training",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="5T",
        help="Bar timeframe(s), comma-separated: 5T,15T",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start YYYY-MM (inclusive)"
    )
    parser.add_argument("--end", type=str, default=None, help="End YYYY-MM (inclusive)")
    parser.add_argument(
        "--forward-bars", type=str, default="3", help="Bars ahead (e.g., 1,5,10)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=0, help="TimeSeries CV folds (0=disable)"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="baseline",
        help="baseline/default/enhanced/hurst/wavelet/hilbert/spectral/order_flow/dl_sequence/comprehensive or combos (e.g., baseline,default,hurst)",
    )
    parser.add_argument(
        "--oos-months",
        type=int,
        default=3,
        help="OOS months after train end (0=disable)",
    )
    parser.add_argument(
        "--oos-start", type=str, default=None, help="OOS start (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--oos-end", type=str, default=None, help="OOS end (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--use-top-factors",
        type=str,
        default=None,
        help="JSON of selected features to keep",
    )
    parser.add_argument(
        "--topk", type=int, default=0, help="Keep only Top-K features (0=disabled)"
    )
    parser.add_argument(
        "--topk-source",
        type=str,
        default=None,
        help="Ranking CSV(feature,score) or JSON list; fallback |IC|",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Use GPU for LightGBM"
    )
    parser.add_argument(
        "--direction-threshold",
        type=str,
        default="f1_optimize",
        choices=["zero", "median", "f1_optimize"],
        help="Method for computing direction prediction threshold: 'zero' (fixed 0), 'median' (median of predictions), 'f1_optimize' (optimize F1 score, default)",
    )
    parser.add_argument(
        "--auto-tune-params",
        action="store_true",
        default=False,
        help="Auto-tune hyperparameters (Q50-constraint-aware) before training",
    )
    parser.add_argument(
        "--winsorize-k",
        type=float,
        default=3.5,
        help="MAD-based winsorize multiplier for target Step 1 (<=0 disables)",
    )
    parser.add_argument(
        "--secondary-winsorize-k",
        type=float,
        default=3.5,
        help="MAD-based winsorize multiplier for target Step 2b (<=0 disables)",
    )
    parser.add_argument(
        "--feature-winsorize-k",
        type=float,
        default=4.0,
        help="MAD-based winsorize multiplier for feature cleaning (<=0 disables)",
    )
    parser.add_argument(
        "--disable-target-winsorize",
        action="store_true",
        default=False,
        help="Disable target winsorization (both Step 1 and Step 2b)",
    )
    parser.add_argument(
        "--disable-feature-winsorize",
        action="store_true",
        default=False,
        help="Disable feature winsorization during training",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter tuning (default: 20)",
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Path to JSON file containing pre-trained parameters (overrides auto-tune)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="quantile",
        choices=["quantile", "classification", "regression"],
        help="Model type: quantile (default), classification, or regression",
    )
    parser.add_argument(
        "--safe-multi-asset",
        action="store_true",
        default=False,
        help="Use safe multi-asset preprocessing: each symbol is processed independently (resample, features, labels) before merging. This prevents cross-asset data leakage but may reduce sample size.",
    )
    args = parser.parse_args()

    if args.model_type == "classification":
        # Classification workflow skips winsorization entirely
        if (
            args.disable_target_winsorize
            or args.disable_feature_winsorize
            or args.winsorize_k != 3.5
            or args.secondary_winsorize_k != 3.5
            or args.feature_winsorize_k != 4.0
        ):
            print("ℹ️  分类模型忽略 winsorize 相关参数，始终不进行裁剪")
        target_winsorize_k = 0.0
        secondary_winsorize_k = 0.0
        feature_winsorize_k = 0.0
    else:
        target_winsorize_k = (
            0.0 if args.disable_target_winsorize else max(args.winsorize_k, 0.0)
        )
        secondary_winsorize_k = (
            0.0
            if args.disable_target_winsorize
            else max(args.secondary_winsorize_k, 0.0)
        )
        feature_winsorize_k = (
            0.0
            if args.disable_feature_winsorize
            else max(args.feature_winsorize_k, 0.0)
        )

    freqs = [f.strip() for f in args.freq.split(",") if f.strip()]
    fbs = [int(x.strip()) for x in args.forward_bars.split(",") if x.strip()]

    # Calculate OOS date range to include OOS data files
    data_end = args.end
    if args.oos_months > 0 or args.oos_start:
        from dateutil.relativedelta import relativedelta

        if args.oos_start:
            try:
                oos_start_dt = pd.to_datetime(args.oos_start)
                oos_end_dt = (
                    pd.to_datetime(args.oos_end)
                    if args.oos_end
                    else oos_start_dt
                    + relativedelta(
                        months=args.oos_months if args.oos_months > 0 else 3
                    )
                )
                oos_end_ym = oos_end_dt.strftime("%Y-%m")
                if not data_end or oos_end_ym > data_end:
                    data_end = oos_end_ym
            except Exception:
                pass
        elif args.start and args.end:
            try:
                # Calculate OOS end date: train_end + oos_months + oos_months (for OOS period)
                train_end_dt = pd.to_datetime(args.end + "-28")
                oos_start_dt = train_end_dt + relativedelta(
                    days=1
                )  # Start from next day
                oos_end_dt = oos_start_dt + relativedelta(
                    months=args.oos_months if args.oos_months > 0 else 3
                )
                oos_end_ym = oos_end_dt.strftime("%Y-%m")
                if oos_end_ym > data_end:
                    data_end = oos_end_ym
            except Exception:
                pass

    files = _collect_files(
        args.data, args.data_dir, args.start, data_end, symbols=args.symbol
    )
    raw = _load_many(files)

    # Parse symbols for multi-asset training
    symbol_list = [s.strip() for s in args.symbol.split(",") if s.strip()]
    symbols_str = (
        ",".join(symbol_list)
        if len(symbol_list) > 1
        else symbol_list[0] if symbol_list else "UNKNOWN"
    )
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

    current_time = _dt.now()
    training_timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    # Format symbol for directory name (replace comma with underscore for multi-asset)
    symbol_dir = symbols_str.replace(",", "_")
    # Create base directory with timestamp, symbol, and feature_type
    # We'll finalize by appending train_start/train_end (YYYYMMDD) after first config is processed
    base_dir = f"{training_timestamp}_{symbol_dir}_{args.feature_type}"
    base_results_dir = os.path.join("results/training", base_dir)
    base_models_dir = os.path.join("models", base_dir)
    base_dir_finalized = False
    print(f"📁 Results will be saved to: {base_results_dir}")
    os.makedirs(base_models_dir, exist_ok=True)
    print(f"📁 Model artifacts will be saved to: {base_models_dir}")

    # 🔍 CRITICAL: Check if training data contains future data (data leakage)
    if isinstance(raw.index, pd.DatetimeIndex) and len(raw) > 0:
        data_max_date = raw.index.max()
        data_min_date = raw.index.min()
        print(f"\n{'='*70}")
        print(f"🔍 数据时间范围验证（Data Time Range Validation）")
        print(f"{'='*70}")
        print(f"   当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"   数据时间范围: {data_min_date.strftime('%Y-%m-%d')} 至 {data_max_date.strftime('%Y-%m-%d')}"
        )

        # Check if data contains future dates
        if data_max_date > current_time:
            print(f"\n   🚨 严重错误：数据包含未来日期！")
            print(f"      数据最大日期: {data_max_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      时间差: {(data_max_date - current_time).days} 天")
            print(f"\n   ⚠️  这是致命错误：不能用未来数据训练模型！")
            print(f"   💡 建议：")
            print(f"      - 检查数据源：确保数据时间范围 ≤ 当前时间")
            print(f"      - 如果使用历史数据，确保数据时间戳正确")
            print(f"      - 训练数据应该早于当前时间，例如：")
            print(
                f"        如果今天是 {current_time.strftime('%Y-%m-%d')}，训练数据最多到 {current_time.strftime('%Y-%m-%d')}"
            )
            print(f"      - 测试/验证集应在训练数据之后")
            print(f"\n   ❌ 训练将被终止，请修复数据时间范围问题！")
            raise ValueError(
                f"Data contains future dates! Max date: {data_max_date}, Current time: {current_time}. "
                "Training data must not contain future information. Please check your data source."
            )
        elif (current_time - data_max_date).days < 1:
            print(f"   ⚠️  警告：数据最大日期非常接近当前时间（<1天）")
            print(f"      这可能是正常的（使用最新数据），但请确保数据时间戳正确")
        else:
            print(
                f"   ✅ 数据时间范围正常（数据最大日期早于当前时间 {(current_time - data_max_date).days} 天）"
            )
        print(f"{'='*70}\n")

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
                print(f"   ⚠️  Warning: Resampled data is empty for {freq}, skipping...")
                continue
        except Exception as e:
            print(f"   ❌ Error resampling to {freq}: {e}")
            print(f"   ⚠️  Skipping timeframe {freq}...")
            continue

        for fb in fbs:
            print(f"\n⚙️  Training config: timeframe={freq}, forward_bars={fb}")

            # 🔒 Safe multi-asset preprocessing: each symbol processed independently
            # CRITICAL: 如果没有 symbol 信息，无法安全合并多标的
            # 必须能够识别每个样本属于哪个标的，否则会导致：
            # 1. 标签混淆（同一时间戳多个资产的标签混在一起）
            # 2. 评估失真（无法按标的分组评估）
            # 3. 模型学偏（模型不知道样本属于哪个资产）
            # 4. 推理时无法确定预测适用于哪个标的
            if args.safe_multi_asset:
                # 检查是否有多个标的（通过文件数量或数据中的 symbol 列）
                has_multiple_symbols = False
                if "symbol" in resampled_data.columns:
                    unique_symbols = resampled_data["symbol"].unique()
                    has_multiple_symbols = len(unique_symbols) > 1
                else:
                    # 如果没有 symbol 列，尝试从文件数量推断
                    # 如果只有一个文件，可能是单标的
                    if len(files) > 1:
                        print(
                            f"\n   ⚠️  警告: 检测到多个数据文件，但 resampled_data 中没有 symbol 列"
                        )
                        print(f"      这可能导致多标的合并不安全")
                        print(f"      💡 建议：")
                        print(f"         - 确保数据文件包含 symbol 列")
                        print(
                            f"         - 或使用 safe_multi_asset_preprocessing 自动从文件名推断 symbol"
                        )
                        print(f"         - 或使用单标的训练模式")
                        # 即使没有 symbol 列，也尝试使用 safe_multi_asset_preprocessing
                        # 因为它会从文件名推断 symbol
                        has_multiple_symbols = True

                # Always use safe_multi_asset_preprocessing when safe_multi_asset is enabled
                # It handles both single and multi-asset cases
                from time_series_model.pipeline.training.safe_multi_asset_preprocessing import (
                    safe_multi_asset_preprocessing,
                )

                print(f"   🔒 使用安全的多标的预处理（完全隔离）")
                feat_df, preprocessing_metadata = safe_multi_asset_preprocessing(
                    files=files,
                    feature_type=args.feature_type,
                    timeframe=freq,
                    forward_bars=fb,
                    feature_engineer=None,
                )
                # Extract feature_engineer from metadata if needed
                if args.feature_type == "baseline":
                    _, feature_engineer = engineer_baseline_features(
                        feat_df.head(100), None, fit=False
                    )
                else:
                    feature_engineer = ComprehensiveFeatureEngineer(
                        feature_types=args.feature_type
                    )
            else:
                # Original preprocessing (merged data)
                print(f"   Samples in resampled data: {len(resampled_data):,}")
                feat_df = resampled_data.copy()
                # Multi-asset training: features are engineered on merged data
                # All features are normalized (asset-agnostic), so the model learns
                # common patterns across different assets
                if args.feature_type == "baseline":
                    feat_df, base_eng = engineer_baseline_features(
                        feat_df, None, fit=True
                    )
                    feature_engineer = base_eng
                else:
                    feature_engineer = ComprehensiveFeatureEngineer(
                        feature_types=args.feature_type
                    )
                    feat_df = feature_engineer.engineer_all_features(feat_df, fit=True)

            # Calculate future return (simple return for fb bars ahead)
            # CRITICAL: Use close price, NOT high/low price, to avoid look-ahead bias
            # future_return[t] = (close[t+fb] / close[t]) - 1
            # This represents the return over the next fb bars, using closing prices
            # Note: If using safe_multi_asset preprocessing, labels are already calculated
            # Check if future_return already exists (from safe_multi_asset preprocessing)
            future_return_exists = (
                args.safe_multi_asset and "future_return" in feat_df.columns
            )
            if not future_return_exists:
                # 🔧 FIX: For multi-asset training, calculate future_return per symbol to avoid cross-asset leakage
                # 🔒 CRITICAL: 如果没有 symbol 信息，无法安全计算多标的的 future_return
                # 必须按标的分别计算，否则会导致跨标的数据泄露
                if "symbol" in feat_df.columns and len(feat_df["symbol"].unique()) > 1:
                    # Multi-asset: calculate future_return separately for each symbol
                    def calc_future_return(group):
                        group["future_return"] = (
                            group["close"].shift(-fb) / group["close"] - 1
                        )
                        return group

                    feat_df = feat_df.groupby("symbol", group_keys=False).apply(
                        calc_future_return
                    )
                elif (
                    "symbol" in feat_df.columns and len(feat_df["symbol"].unique()) == 1
                ):
                    # Single asset: calculate directly
                    feat_df["future_return"] = (
                        feat_df["close"].shift(-fb) / feat_df["close"] - 1
                    )
                else:
                    # 🔒 CRITICAL: 如果没有 symbol 信息，无法确定是单标的还是多标的
                    # 如果数据来自多个文件但没有 symbol 列，这是不安全的
                    if len(files) > 1:
                        raise ValueError(
                            "❌ 严重错误：检测到多个数据文件，但数据中没有 symbol 列！"
                            "没有 symbol 信息的多标的合并是不安全的，会导致："
                            "1. 标签混淆（同一时间戳多个资产的标签混在一起）"
                            "2. 评估失真（无法按标的分组评估）"
                            "3. 模型学偏（模型不知道样本属于哪个资产）"
                            "4. 推理时无法确定预测适用于哪个标的"
                            "\n💡 建议："
                            "   - 使用 --safe-multi-asset 选项（会自动从文件名推断 symbol）"
                            "   - 或确保数据文件包含 symbol 列"
                            "   - 或使用单标的训练模式"
                        )
                    else:
                        # 单文件单标的：直接计算
                        feat_df["future_return"] = (
                            feat_df["close"].shift(-fb) / feat_df["close"] - 1
                        )

            # Verify label definition: log a warning if any suspicious pattern is detected
            # (e.g., if future_return seems to be calculated from high instead of close)
            # For now, we trust the calculation above, but we can add validation later if needed

            # ⚠️ DEPRECATED: Old global preprocessing (STEP 1, STEP 2, STEP 2b) - DISABLED
            # This code caused lookahead bias by using global statistics (containing future data)
            # Preprocessing is now moved INSIDE CV loop (see code around line 967+)
            # Reference: User feedback on data leakage issue
            # All preprocessing statistics (median, mad, ar1_phi) must be computed ONLY from training data

            # DISABLED: Old global preprocessing code (STEP 1, STEP 2, STEP 2b) - causes lookahead bias
            # This code used global statistics (containing future data) to preprocess target variable
            # All preprocessing is now done INSIDE CV loop (see code around line 967+)
            # The old code is preserved in git history for reference but removed here to prevent accidental use
            # The old code included:
            # - STEP 1: Global Winsorize using global median/mad (causes lookahead bias)
            # - STEP 2: Global AR(1) residual using global ar1_phi (causes lookahead bias)
            # - STEP 2b: Secondary global cleaning (causes lookahead bias)
            # All preprocessing is now done in CV loop using ONLY training set statistics

            # FIXED: future_volatility should use future returns, not shifted current returns
            # Previous bug: one.shift(-1) was using future data (data leakage!)
            # 🔒 CRITICAL FIX: Cannot use rolling std for future_volatility as it introduces future information
            # future_volatility[t] = std(future_return[t:t+window]) requires future_return[t+1], ..., future_return[t+window-1]
            # But these values correspond to future returns (e.g., future_return[t+1] needs close[t+1+fb]), introducing future information
            # ✅ Correct approach: Compute future volatility label from future single-period returns
            # This computes vol[t] = RMS(r_{t+1}, ..., r_{t+horizon}) as the label
            # Note: If using safe_multi_asset preprocessing, labels are already calculated
            if not (args.safe_multi_asset and "future_volatility" in feat_df.columns):
                feat_df["future_volatility"] = future_volatility_label(
                    feat_df["close"],
                    horizon=fb,
                    min_periods=max(3, fb // 2),
                )
                # Only drop rows where targets are NaN; allow feature NaNs (handled later)
                feat_df = feat_df.dropna(
                    subset=["future_return", "future_volatility"]
                ).copy()

            # Calculate train_end from args.end, not from feat_df.index.max()
            # feat_df now includes OOS data, so we need to use args.end to determine training period
            train_end = None
            if args.end:
                try:
                    # Use last day of the training month
                    from calendar import monthrange

                    train_end_dt = pd.to_datetime(args.end + "-01")
                    last_day = monthrange(train_end_dt.year, train_end_dt.month)[1]
                    train_end = pd.to_datetime(args.end + f"-{last_day:02d} 23:59:59")
                except Exception:
                    # Fallback: use feat_df.index.max() but filter by args.end first
                    if not feat_df.empty:
                        train_end = (
                            feat_df[
                                feat_df.index <= pd.to_datetime(args.end + "-28")
                            ].index.max()
                            if args.end
                            else None
                        )
            else:
                # If no args.end, use feat_df.index.max() as fallback
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
                    # OOS starts from the day after training ends
                    oos_start_dt = train_end + pd.Timedelta(seconds=1)
                if args.oos_end:
                    try:
                        oos_end_dt = pd.to_datetime(args.oos_end)
                    except Exception:
                        oos_end_dt = None
                if oos_end_dt is None and oos_start_dt is not None:
                    oos_end_dt = oos_start_dt + relativedelta(
                        months=args.oos_months if args.oos_months > 0 else 3
                    )
                if oos_start_dt is not None and oos_end_dt is not None:
                    oos_mask = (feat_df.index >= oos_start_dt) & (
                        feat_df.index <= oos_end_dt
                    )
                    oos_df = feat_df[oos_mask].copy()
                    if len(oos_df) == 0:
                        print(f"\n   ⚠️  警告: OOS数据为空！")
                        print(f"      OOS开始时间: {oos_start_dt}")
                        print(f"      OOS结束时间: {oos_end_dt}")
                        print(
                            f"      数据时间范围: {feat_df.index.min()} 至 {feat_df.index.max()}"
                        )
                        print(f"      💡 建议: 检查数据文件是否包含OOS时间段的数据")
                    else:
                        print(f"\n   ✅ OOS数据加载成功: {len(oos_df)} 个样本")
                        print(
                            f"      OOS时间范围: {oos_df.index.min()} 至 {oos_df.index.max()}"
                        )

            # Split train and OOS data
            if len(oos_df) > 0 and oos_start_dt is not None:
                train_df = feat_df[feat_df.index < oos_start_dt].copy()
            else:
                train_df = feat_df.copy()

            if args.feature_type == "baseline":
                feature_cols = get_baseline_feature_columns(train_df)
            else:
                feature_cols = get_feature_columns_by_type(train_df, args.feature_type)
            # optional top-factors
            if args.use_top_factors:
                try:
                    with open(args.use_top_factors, "r", encoding="utf-8") as f:
                        top = json.load(f)
                    if isinstance(top, dict) and "features" in top:
                        top = top["features"]
                    if isinstance(top, list):
                        s = set(top)
                        feature_cols = [c for c in feature_cols if c in s]
                except Exception:
                    pass
            # numeric only, and exclude symbol and raw price columns
            # 🔒 CRITICAL: Symbol should only be used for data alignment and grouping, not as a model feature
            # Using symbol as a feature would cause the model to overfit to specific assets
            # and prevent it from learning cross-asset patterns
            # 🔒 CRITICAL: Raw price columns (open, high, low, close) should NOT be used as features
            # They have different scales across assets and would cause model bias
            # Only normalized/standardized features (returns, ratios, z-scores) should be used
            # Note: This is a double-check. get_feature_columns() should already exclude these,
            # but we add this as a safety measure to ensure no raw data leaks into features
            exclude_from_features = {
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "future_return",
                "future_volatility",
                "future_return_*",
                "future_volatility_*",
                "timestamp",
                "signal",
                "binary_signal",
                # Raw order flow data (normalized versions should be used instead)
                "buy_qty",
                "sell_qty",
                "trade_count",
                "cvd",  # Exclude raw CVD - use cvd_normalized, cvd_spectral_*, cvd_wpt_*, etc.
                # Note: taker_buy_ratio may have normalized versions, but raw version should also be excluded
                # if normalized versions exist (e.g., taker_buy_ratio_spectral_*, taker_buy_ratio_wpt_*)
                # get_feature_columns() should handle excluding raw versions when normalized exist
            }
            feature_cols = [
                c
                for c in feature_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
                and c not in exclude_from_features
            ]

            # Double-check: ensure excluded columns are NOT in feature_cols
            found_excluded = [c for c in feature_cols if c in exclude_from_features]
            if found_excluded:
                print(f"   ⚠️  警告: 以下列被包含在特征中，已自动排除: {found_excluded}")
                feature_cols = [
                    c for c in feature_cols if c not in exclude_from_features
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
                                    f
                                    for f in _df["feature"].tolist()
                                    if f in feature_cols
                                ]
                        else:
                            lst = json.load(
                                open(args.topk_source, "r", encoding="utf-8")
                            )
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
                                    nan_policy="omit",
                                )
                                ic.append((c, abs(r) if pd.notna(r) else 0.0))
                            except Exception:
                                ic.append((c, 0.0))
                        ic.sort(key=lambda x: x[1], reverse=True)
                        ranked = [c for c, _ in ic]
                    except Exception:
                        ranked = feature_cols
                feature_cols = ranked[: args.topk]

            X_df = pd.DataFrame(
                train_df[feature_cols].values,
                columns=feature_cols,
                index=train_df.index,
            )

            # 🔍 Multi-asset validation: Check feature distribution by asset
            if "symbol" in train_df.columns and len(train_df["symbol"].unique()) > 1:
                print(f"\n   🔍 多资产特征验证（Multi-asset Feature Validation）:")
                print(f"      资产数量: {len(train_df['symbol'].unique())}")
                for symbol in train_df["symbol"].unique():
                    symbol_mask = train_df["symbol"] == symbol
                    symbol_X = X_df[symbol_mask]
                    if len(symbol_X) > 0:
                        # Check if features have similar distributions across assets
                        # If features are properly normalized, means should be similar
                        feature_means = symbol_X.mean()
                        feature_stds = symbol_X.std()
                        print(f"      {symbol}: {len(symbol_X):,} samples")
                        print(
                            f"         特征均值范围: [{feature_means.min():.4f}, {feature_means.max():.4f}]"
                        )
                        print(
                            f"         特征标准差范围: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]"
                        )

                        # Check for extreme differences (potential issue)
                        # Identify problematic features (likely raw prices or unnormalized values)
                        problematic_features = []
                        for feat in symbol_X.columns:
                            feat_mean = abs(feature_means[feat])
                            feat_std = feature_stds[feat]
                            # Features with very large means or stds are likely problematic
                            if feat_mean > 1000 or feat_std > 1000:
                                problematic_features.append((feat, feat_mean, feat_std))

                        if problematic_features:
                            print(
                                f"         ⚠️  警告：发现 {len(problematic_features)} 个异常特征（可能使用了原始价格或未标准化值）"
                            )
                            # Show top 5 most problematic features
                            problematic_features.sort(
                                key=lambda x: max(x[1], x[2]), reverse=True
                            )
                            for feat, feat_mean, feat_std in problematic_features[:5]:
                                print(
                                    f"            - {feat}: mean={feat_mean:.2f}, std={feat_std:.2f}"
                                )
                            if len(problematic_features) > 5:
                                print(
                                    f"            ... 还有 {len(problematic_features) - 5} 个异常特征"
                                )
                            print(
                                f"            💡 建议：检查特征工程，确保所有特征都已标准化/归一化"
                            )

                        # Also check overall statistics
                        if abs(feature_means.mean()) > 10 or feature_stds.mean() > 100:
                            print(
                                f"         ⚠️  警告：特征值范围异常（整体均值={feature_means.mean():.2f}, 整体标准差={feature_stds.mean():.2f}）"
                            )
                            print(f"            这可能导致模型偏向价格更高的资产")

                # Check target variable distribution by asset
                print(f"\n      目标变量分布（按资产）:")
                for symbol in train_df["symbol"].unique():
                    symbol_mask = train_df["symbol"] == symbol
                    symbol_y = train_df.loc[symbol_mask, "future_return"]
                    if len(symbol_y) > 0:
                        y_mean = float(symbol_y.mean())
                        y_std = float(symbol_y.std())
                        y_abs_max = float(symbol_y.abs().max())
                        print(
                            f"      {symbol}: mean={y_mean:.6f}, std={y_std:.6f}, max_abs={y_abs_max:.6f}"
                        )

                print(f"\n      💡 多资产合并训练说明：")
                print(
                    f"         - 如果特征工程正确（使用收益率/标准化），价格水平差异不应影响模型"
                )
                print(f"         - 模型将学习跨资产的共同模式（如技术指标模式）")
                print(f"         - 加密货币之间通常有较高相关性，这是正常的")
                print(
                    f"         - 但如果数据量严重不平衡（>80%来自一个资产），模型可能偏向该资产"
                )

            # CRITICAL: Use RAW future_return (no global preprocessing)
            # Preprocessing will be applied INSIDE CV loop to prevent lookahead bias
            y_return = pd.Series(train_df["future_return"].values, index=train_df.index)
            y_vol = pd.Series(
                train_df["future_volatility"].values, index=train_df.index
            )

            # ✅ Feature cleaning moved to CV loop (prevents lookahead bias)
            # Feature cleaning is now done in LightGBMTrainer.train() within each CV fold
            # All statistics (median, mad) computed ONLY from training data per fold
            # Reference: docs/极端值：确保 Q50 loss ≤ Q10Q90 loss.md
            print(
                f"\n   ✅ Feature cleaning: Moved to CV loop (prevents lookahead bias)"
            )
            print(
                f"      Features will be cleaned per CV fold using training set statistics"
            )

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
            max_reasonable_return = (
                0.1 if fb == 1 else min(0.5, 0.1 * fb)
            )  # 10% for fb=1, or 10%*fb up to 50%
            if (
                abs(y_return.max()) > max_reasonable_return
                or abs(y_return.min()) < -max_reasonable_return
            ):
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
            # Load pre-trained parameters if provided, otherwise use defaults
            q50_params = None
            if args.params_file and os.path.exists(args.params_file):
                print(
                    f"\n   📂 Loading pre-trained parameters from: {args.params_file}"
                )
                with open(args.params_file, "r") as f:
                    loaded_params = json.load(f)
                from time_series_model.config.settings import DEFAULT_LGBM_PARAMS

                q50_params = DEFAULT_LGBM_PARAMS.copy()
                q50_params.update(loaded_params)
                print(f"   ✅ Loaded {len(loaded_params)} parameters")
                print(
                    f"      Key params: num_leaves={q50_params.get('num_leaves')}, "
                    f"learning_rate={q50_params.get('learning_rate')}, "
                    f"num_boost_round={q50_params.get('num_boost_round', 'default')}"
                )
            else:
                # Auto-adjust parameters for Q50 if we detect potential issues
                # Use more aggressive parameters to prevent underfitting
                # Check if y_return has extreme values that might cause Q50 issues
                if isinstance(y_return, pd.Series):
                    std_y = y_return.std()
                    mean_y = y_return.mean()
                    extreme_count = np.sum(np.abs(y_return - mean_y) > 3 * std_y)
                    if (
                        extreme_count > len(y_return) * 0.01
                    ):  # More than 1% extreme values
                        # Adjust parameters for better prediction of extremes
                        from time_series_model.config.settings import (
                            DEFAULT_LGBM_PARAMS,
                        )

                        q50_params = DEFAULT_LGBM_PARAMS.copy()
                        q50_params["num_leaves"] = 127  # Increase from default 31
                        q50_params["min_data_in_leaf"] = 10  # Decrease from default 20
                        q50_params["learning_rate"] = (
                            0.03  # Slightly lower for finer predictions
                        )
                        print(
                            f"   🔧 Auto-adjust: 检测到{extreme_count}个极端值（>1%），自动调整Q50模型参数:"
                        )
                        print(f"      num_leaves: 31 → 127")
                        print(f"      min_data_in_leaf: 20 → 10")
                        print(f"      learning_rate: 0.05 → 0.03")

            # 🔧 CRITICAL FIX: Move preprocessing INSIDE CV loop to prevent lookahead bias
            # All preprocessing statistics (median, mad, ar1_phi) must be computed ONLY from training data
            # Reference: User feedback on data leakage issue
            print(
                f"\n   🔧 CRITICAL: Preprocessing moved INSIDE CV loop (prevents lookahead bias)"
            )
            print(
                f"      All statistics (median, mad, ar1_phi) computed ONLY from training data"
            )

            # Prepare current_returns for AR(1) processing (needed for preprocessing)
            # 🔒 CRITICAL: Calculate log returns and handle extreme values
            # If close prices have gaps or errors, log returns can be extreme
            # This can cause AR(1) predictions to be huge, leading to extreme residuals
            # 🔒 CRITICAL: For multi-asset training, calculate current_returns per symbol
            # If we calculate across symbols, shift(1) may use price from a different asset
            # This would cause incorrect log returns (e.g., BTC close / ETH close)
            if "symbol" in feat_df.columns and len(feat_df["symbol"].unique()) > 1:
                # Multi-asset: calculate current_returns separately for each symbol
                def calc_current_returns(group):
                    group_returns = np.log(group["close"] / group["close"].shift(1))
                    return pd.Series(group_returns, index=group.index)

                current_returns = feat_df.groupby("symbol", group_keys=False).apply(
                    calc_current_returns
                )
                current_returns = current_returns.sort_index()
            else:
                # Single asset: calculate directly
                current_returns = np.log(feat_df["close"] / feat_df["close"].shift(1))
                current_returns = pd.Series(current_returns, index=feat_df.index)

            # 🔒 CRITICAL: Winsorize current_returns to prevent extreme AR(1) predictions
            # Clip to reasonable range (e.g., ±0.8 log return ≈ ±123% price change)
            # This prevents numerical issues in AR(1) residual transformation
            max_log_return = 0.8
            current_returns = current_returns.clip(-max_log_return, max_log_return)

            # Diagnostic: Check for extreme values (should be rare after clipping)
            # 🔍 DIAGNOSTIC: Print current_returns statistics to understand the distribution
            current_returns_valid = current_returns.dropna()
            if len(current_returns_valid) > 0:
                cr_mean = current_returns_valid.mean()
                cr_std = current_returns_valid.std()
                cr_median = current_returns_valid.median()
                cr_min = current_returns_valid.min()
                cr_max = current_returns_valid.max()
                cr_abs_max = current_returns_valid.abs().max()
                clip_hits = np.isclose(
                    current_returns_valid, max_log_return
                ) | np.isclose(current_returns_valid, -max_log_return)
                clip_ratio = (
                    (clip_hits.sum() / len(current_returns_valid))
                    if len(current_returns_valid) > 0
                    else 0.0
                )
                cr_p1 = current_returns_valid.quantile(0.01)
                cr_p99 = current_returns_valid.quantile(0.99)

                print(f"   🔍 current_returns 诊断（log return，clip 后）:")
                print(f"      范围: [{cr_min:.6f}, {cr_max:.6f}]")
                print(
                    f"      均值: {cr_mean:.6f}, 中位数: {cr_median:.6f}, 标准差: {cr_std:.6f}"
                )
                print(f"      绝对值最大值: {cr_abs_max:.6f} ({cr_abs_max*100:.2f}%)")
                print(f"      1%分位数: {cr_p1:.6f}, 99%分位数: {cr_p99:.6f}")
                if clip_ratio > 0:
                    print(
                        f"      ⚠️ clip比例: {clip_ratio:.2%} (命中 ±{max_log_return})"
                    )

                # Check for extreme values (>30% log return = >35% price change)
                # Note: After clipping to ±0.5, values > 0.3 are expected to be rare
                extreme_count = (current_returns_valid.abs() > 0.3).sum()
                if extreme_count > 0:
                    extreme_ratio = extreme_count / len(current_returns_valid) * 100
                    if extreme_ratio > 1.0:  # More than 1% extreme values
                        print(
                            f"      ⚠️  警告: 检测到 {extreme_count} 个极端 current_returns（>30% log return），占比 {extreme_ratio:.2f}%"
                        )
                        print(f"      已自动 clip 到 ±{max_log_return*100:.0f}% 范围")
                        # Additional diagnostic: check if this is due to clipping
                        # If most values are > 0.3, it means they were clipped from even larger values
                        if extreme_ratio > 50.0:  # More than 50% extreme values
                            print(
                                f"      🚨 严重警告: 极端值占比过高（{extreme_ratio:.2f}%），可能的原因："
                            )
                            print(
                                f"         1. 原始数据中有大量极端价格跳空（>35% 价格变化）"
                            )
                            print(
                                f"         2. 数据质量问题（价格数据错误、时间对齐问题）"
                            )
                            print(f"         3. 多标的合并时时间戳不一致导致计算错误")
                            print(
                                f"      💡 建议：检查原始价格数据，确认是否有异常价格跳空"
                            )

            # Create preprocessing function wrapper that has access to current_returns
            # This function will be called within each CV fold
            from time_series_model.pipeline.training.preprocessing import (
                preprocess_target_cv,
            )

            def create_preprocess_fn(
                current_returns_array,
                current_returns_index,
                forward_bars,
                k_winsorize=3.5,
                k_secondary=3.5,
                verbose=False,
            ):
                """Create a preprocessing function that can access current_returns by index

                Args:
                    current_returns_array: numpy array with current returns (aligned with y_return)
                    current_returns_index: pandas Index corresponding to current_returns_array
                """
                # Create index mapping for efficient lookup (avoid repeated reindex operations)
                # This maps from y_train/y_val indices to positions in current_returns_array
                # 🚀 OPTIMIZATION: Use pandas Index.get_indexer for faster and more memory-efficient lookup
                # This avoids creating a large dictionary (156k+ entries) which can use significant memory
                # Instead, we'll use pandas' built-in index alignment which is optimized
                # Store the index for direct use with pandas operations
                _current_returns_index = current_returns_index

                def preprocess_fn(y_train, y_val, **kwargs):
                    # 🚀 OPTIMIZATION: Use positional indexing instead of label-based indexing
                    # This avoids creating new Series objects and reduces memory usage

                    # Get positions for y_train and y_val indices
                    # 🚀 OPTIMIZATION: Use pandas Index.get_indexer for memory-efficient lookup
                    # This is faster and uses less memory than creating a large dictionary
                    # 🔒 CRITICAL: For multi-asset training, indices have duplicate timestamps
                    # get_indexer() doesn't work with duplicate indices (raises InvalidIndexError)
                    # .loc[] also doesn't work correctly because it returns all matches for duplicate indices
                    #
                    # 🚀 SOLUTION: Since current_returns and y_return have the same index structure,
                    # and y_train/y_val are positional subsets, we can use positional indexing directly
                    # We need to find the positions of y_train/y_val in the full dataset
                    #
                    # Since indices match exactly (same order), we can find positions by matching
                    # the first and last indices, then use positional slicing

                    # Find start and end positions in the full index
                    # For duplicate indices, we need to match by position, not by label
                    # Since y_train and y_val are subsets of y_return, and y_return has the same
                    # index as current_returns, we can use the fact that they're in the same order

                    # Get the position of the first element of y_train in the full index
                    # We'll use a different approach: since indices match exactly,
                    # we can create a mapping using enumerate
                    # But this is slow. Instead, let's assume indices match and use direct indexing

                    # 🚀 BEST APPROACH: Since y_train/y_val are created from y_return by positional
                    # indexing (train_idx, val_idx), and current_returns has the same index as y_return,
                    # we can directly use the same positional indices to extract from current_returns_array
                    # But we don't have train_idx/val_idx here. We need to find them.

                    # Alternative: Use a simple loop to match indices by position
                    # This is O(n) but should be fast enough for our use case
                    # Create a mapping: for each index in y_train/y_val, find its position in _current_returns_index
                    # Since indices match exactly, we can use a simple approach:
                    # - Iterate through _current_returns_index and y_train.index simultaneously
                    # - Match by position (assuming same order)

                    # Actually, the simplest solution: since indices should match exactly,
                    # we can iterate and match by position
                    # For each index in y_train.index, find its first occurrence in _current_returns_index
                    # and use that position. This works if indices are in the same order.

                    # Create position arrays by matching indices sequentially
                    # This assumes indices are in the same order (which they should be)
                    train_positions = []
                    val_positions = []

                    # Create a position counter for the full index
                    full_idx_pos = 0
                    full_idx_dict = {}  # Map (index, occurrence) to position

                    # Build mapping: for each unique index, track its occurrences
                    for idx in _current_returns_index:
                        if idx not in full_idx_dict:
                            full_idx_dict[idx] = []
                        full_idx_dict[idx].append(full_idx_pos)
                        full_idx_pos += 1

                    # Now match y_train and y_val indices
                    train_occurrence = {}
                    val_occurrence = {}

                    for idx in y_train.index:
                        if idx not in train_occurrence:
                            train_occurrence[idx] = 0
                        if idx in full_idx_dict and train_occurrence[idx] < len(
                            full_idx_dict[idx]
                        ):
                            train_positions.append(
                                full_idx_dict[idx][train_occurrence[idx]]
                            )
                            train_occurrence[idx] += 1
                        else:
                            train_positions.append(-1)  # Not found

                    for idx in y_val.index:
                        if idx not in val_occurrence:
                            val_occurrence[idx] = 0
                        if idx in full_idx_dict and val_occurrence[idx] < len(
                            full_idx_dict[idx]
                        ):
                            val_positions.append(
                                full_idx_dict[idx][val_occurrence[idx]]
                            )
                            val_occurrence[idx] += 1
                        else:
                            val_positions.append(-1)  # Not found

                    # Convert to numpy arrays
                    train_positions = np.array(train_positions, dtype=np.int32)
                    val_positions = np.array(val_positions, dtype=np.int32)

                    # Extract values using positions
                    train_mask = train_positions >= 0
                    val_mask = val_positions >= 0
                    current_returns_train_values = np.zeros(
                        len(train_positions), dtype=np.float32
                    )
                    current_returns_val_values = np.zeros(
                        len(val_positions), dtype=np.float32
                    )

                    if train_mask.any():
                        current_returns_train_values[train_mask] = (
                            current_returns_array[train_positions[train_mask]]
                        )
                    if val_mask.any():
                        current_returns_val_values[val_mask] = current_returns_array[
                            val_positions[val_mask]
                        ]

                    # Convert to Series for compatibility with preprocess_target_cv
                    current_returns_train = pd.Series(
                        current_returns_train_values,
                        index=y_train.index,
                        dtype=np.float32,
                        copy=False,
                    )
                    current_returns_val = pd.Series(
                        current_returns_val_values,
                        index=y_val.index,
                        dtype=np.float32,
                        copy=False,
                    )

                    # Get fold index for verbose logging (only first fold)
                    fold = kwargs.get("fold", 0)
                    verbose_fold = verbose and (fold == 0)

                    # Call the preprocessing function with enhanced options
                    y_train_proc, y_val_proc, stats = preprocess_target_cv(
                        y_train,
                        y_val,
                        current_returns_train,
                        current_returns_val,
                        forward_bars=forward_bars,
                        k_winsorize=k_winsorize,
                        k_secondary=k_secondary,
                        accurate_forward=True,  # Use accurate phi^fb for forward_bars > 1
                        use_symmetric_quantile=None,  # Disable percentile-based pre-clipping; rely on MAD winsorize
                        smooth_clip=False,  # Use pure MAD clipping to preserve tail structure
                        verbose=verbose_fold,
                    )
                    # Convert NamedTuple to dict for backward compatibility
                    stats_dict = {
                        "step1_winsorize": stats.winsorize,
                        "step2_ar1": stats.ar1,
                        "step2b_secondary": stats.secondary,
                    }
                    return y_train_proc, y_val_proc, stats_dict

                return preprocess_fn

            # Create preprocessing function (verbose only for first fold)
            # 🔒 CRITICAL: Filter current_returns to only include rows that exist in y_return
            # This ensures current_returns is aligned with the data that will be used for training
            # (y_return and X_df have the same index, and X_clean will be a subset of them)
            # We filter current_returns now to avoid index mismatches in preprocess_fn
            # 🚀 OPTIMIZATION: Since current_returns and y_return are created from the same feat_df,
            # their indices should match. Use direct indexing to avoid memory explosion.
            # For duplicate indices (multi-asset), we need to use positional indexing
            # Check if indices match (they should, since both come from feat_df)
            if len(current_returns) == len(y_return) and current_returns.index.equals(
                y_return.index
            ):
                # Indices match perfectly, use directly
                current_returns_filtered = current_returns.fillna(0.0)
            else:
                # Indices don't match, use a memory-efficient approach
                # Since both come from the same source, we can use positional matching
                # 🚀 OPTIMIZATION: Convert to arrays first to avoid Series overhead
                current_returns_arr = current_returns.values.astype(np.float32)
                y_return_arr = y_return.values

                # If lengths match, assume positional alignment
                if len(current_returns) == len(y_return):
                    # Use positional alignment (indices may differ but positions match)
                    current_returns_filtered = pd.Series(
                        current_returns_arr, index=y_return.index, dtype=np.float32
                    ).fillna(0.0)
                else:
                    # Lengths don't match - need to filter current_returns to match y_return
                    # This should be rare, but handle it by using index intersection
                    # Use a simple loop-based approach to avoid memory explosion
                    current_returns_dict = {}
                    for idx, val in zip(current_returns.index, current_returns_arr):
                        if idx in y_return.index:
                            current_returns_dict[idx] = val

                    # Create filtered series
                    current_returns_values = np.array(
                        [current_returns_dict.get(idx, 0.0) for idx in y_return.index],
                        dtype=np.float32,
                    )
                    current_returns_filtered = pd.Series(
                        current_returns_values, index=y_return.index, dtype=np.float32
                    )
            # 🚀 OPTIMIZATION: Convert to numpy array with index mapping for memory efficiency
            # This avoids creating multiple Series copies during CV loops
            current_returns_array = current_returns_filtered.values.astype(
                np.float32
            )  # Use float32 to save memory
            current_returns_index = current_returns_filtered.index

            preprocess_fn = create_preprocess_fn(
                current_returns_array,  # Pass numpy array instead of Series
                current_returns_index,  # Pass index separately for mapping
                fb,
                k_winsorize=target_winsorize_k,
                k_secondary=secondary_winsorize_k,
                verbose=True,
            )

            # Use TimeSeries CV by default to avoid random split failures on edge cases
            # Pass preprocessing function to be called within CV loop
            # 🔒 CRITICAL: For multi-asset training, pass groups (symbol) for GroupKFold
            # This ensures samples from the same symbol are not split across train/val
            groups = None
            if "symbol" in train_df.columns and len(train_df["symbol"].unique()) > 1:
                # Create groups array based on symbol (for GroupKFold)
                # Map symbol to integer group ID
                # 🔒 CRITICAL: groups must be aligned with y_return (not X_df)
                # because prepare_data will remove rows where y is NaN
                # So we need to create groups based on y_return's index
                symbol_to_group = {
                    symbol: idx
                    for idx, symbol in enumerate(train_df["symbol"].unique())
                }
                # Use y_return.index instead of X_df.index to ensure alignment
                # y_return and X_df should have the same index, but we use y_return to be safe
                # 🔒 CRITICAL: Ensure groups is created from train_df, not feat_df
                # train_df is a subset of feat_df (if OOS split exists), so we need to use train_df
                # Verify that y_return.index is a subset of train_df.index
                if not y_return.index.isin(train_df.index).all():
                    print(
                        f"   ⚠️  警告: y_return.index 中有 {sum(~y_return.index.isin(train_df.index))} 个索引不在 train_df.index 中"
                    )
                    # Filter y_return to only include indices that exist in train_df
                    y_return = y_return.loc[y_return.index.isin(train_df.index)]

                # Ensure X_df and y_return have the same index
                # If they don't match, align them
                if not X_df.index.equals(y_return.index):
                    print(
                        f"   ⚠️  警告: X_df.index 和 y_return.index 不匹配，正在对齐..."
                    )
                    # Find common indices
                    common_idx = X_df.index.intersection(y_return.index)
                    X_df = X_df.loc[common_idx]
                    y_return = y_return.loc[common_idx]
                    print(
                        f"      对齐后: X_df 长度={len(X_df)}, y_return 长度={len(y_return)}"
                    )

                # Ensure groups length matches y_return length
                # Check if y_return.index has duplicates
                if y_return.index.duplicated().any():
                    print(
                        f"   ⚠️  警告: y_return.index 中有 {y_return.index.duplicated().sum()} 个重复索引，正在处理..."
                    )
                    # If there are duplicates, we need to handle them
                    # Since y_return and train_df are both subsets of feat_df with the same order,
                    # we can use positional matching: y_return[i] should correspond to train_df[j]
                    # where j is the position of y_return.index[i] in train_df
                    # We'll use a counter to track which occurrence of each index we're on
                    from collections import defaultdict

                    index_counter = defaultdict(int)
                    groups = []

                    for idx in y_return.index:
                        # Find all positions of this index in train_df
                        positions = train_df.index.get_loc(idx)
                        if isinstance(positions, slice):
                            # If it's a slice, get all positions
                            all_positions = list(range(positions.start, positions.stop))
                        elif isinstance(positions, np.ndarray):
                            # If it's a boolean array, get all True positions
                            all_positions = np.where(positions)[0].tolist()
                        else:
                            # Single position
                            all_positions = [positions]

                        # Use the counter to get the correct occurrence
                        counter = index_counter[idx]
                        if counter < len(all_positions):
                            pos = all_positions[counter]
                            index_counter[idx] += 1
                        else:
                            # If we've used all occurrences, use the last one
                            pos = all_positions[-1]

                        symbol = train_df.iloc[pos]["symbol"]
                        groups.append(symbol_to_group.get(symbol, 0))

                    groups = np.array(groups)
                else:
                    # No duplicates, use index-based alignment
                    groups_series = train_df.loc[y_return.index, "symbol"].map(
                        symbol_to_group
                    )
                    # If lengths don't match, use reindex to align
                    if len(groups_series) != len(y_return):
                        # If lengths don't match, use reindex to align
                        groups_series = groups_series.reindex(y_return.index)
                    groups = groups_series.values
                print(
                    f"   🔒 使用 GroupKFold 交叉验证（按 symbol 分组，避免跨标的数据泄露）"
                )
                print(
                    f"      groups 长度: {len(groups)}, y_return 长度: {len(y_return)}, train_df 长度: {len(train_df)}, X_df 长度: {len(X_df)}"
                )
                # Debug: Check groups distribution
                unique_groups, counts = np.unique(groups, return_counts=True)
                print(f"      groups 分布: {dict(zip(unique_groups, counts))}")
                # Validate groups length matches X_df length
                if len(groups) != len(X_df):
                    raise ValueError(
                        f"groups length ({len(groups)}) does not match X_df length ({len(X_df)}). "
                        f"This may be due to duplicate indices in y_return.index. "
                        f"Please check data alignment."
                    )

            # Use pre-trained params if provided, otherwise allow auto-tuning
            use_auto_tune = args.auto_tune_params and not args.params_file

            # Use model trainers for different model types
            if args.model_type == "classification":
                trainer = ClassificationModelTrainer(
                    use_gpu=args.gpu,
                    auto_tune_params=use_auto_tune,
                    tune_trials=args.tune_trials,
                )
            else:
                trainer = QuantileModelTrainer(
                    use_gpu=args.gpu,
                    auto_tune_params=use_auto_tune,
                    tune_trials=args.tune_trials,
                )

            # Train models using the trainer
            models_dict, metrics_dict, preprocess_params_dict = trainer.train_models(
                X_df=X_df,
                y_return=y_return,
                y_vol=y_vol,
                train_df=train_df,
                n_splits=max(2, args.cv_folds or 2),
                groups=groups,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs={},
                q50_params=q50_params,
                feature_winsorize_k=feature_winsorize_k,
            )

            # Extract models and metrics based on model type
            if args.model_type == "classification":
                model_classification = models_dict.get("classification")
                model_return = models_dict.get("return")
                model_vol = models_dict.get("vol")
                classification_metrics = metrics_dict.get("classification")
                return_metrics = metrics_dict.get("return")
                vol_metrics = metrics_dict.get("vol")
                classification_preprocess_params = preprocess_params_dict.get(
                    "classification"
                )
                return_preprocess_params = preprocess_params_dict.get("return")
                vol_preprocess_params = preprocess_params_dict.get("vol")

                # Not used for classification
                model_q50 = None
                model_q10 = None
                model_q90 = None
                q50_metrics = None
                q10_metrics = None
                q90_metrics = None
                q50_preprocess_params = None
                q10_preprocess_params = None
                q90_preprocess_params = None
            else:
                model_q50 = models_dict.get("q50")
                model_q10 = models_dict.get("q10")
                model_q90 = models_dict.get("q90")
                model_vol = models_dict.get("vol")
                q50_metrics = metrics_dict.get("q50")
                q10_metrics = metrics_dict.get("q10")
                q90_metrics = metrics_dict.get("q90")
                vol_metrics = metrics_dict.get("vol")
                q50_preprocess_params = preprocess_params_dict.get("q50")
                q10_preprocess_params = preprocess_params_dict.get("q10")
                q90_preprocess_params = preprocess_params_dict.get("q90")
                vol_preprocess_params = preprocess_params_dict.get("vol")

                # Not used for quantile
                model_classification = None
                classification_metrics = None
                classification_preprocess_params = None

                # Diagnostic: Check for quantile loss anomalies (Q50 > Q10/Q90)
                # Reference: docs/极端值：确保 Q50 loss ≤ Q10Q90 loss.md
                print("\n" + "=" * 70)
                print("🔍 Quantile Model Diagnostics (检查Q50 loss异常)")
                print("=" * 70)
                q50_loss = q50_metrics.get("cv_quantile_loss", 0)
                q10_loss = q10_metrics.get("cv_quantile_loss", 0)
                q90_loss = q90_metrics.get("cv_quantile_loss", 0)

                # Model usability flag: Q50 loss should be <= Q10/Q90 loss
                # Threshold: Q50 loss ratio > 1.2 means model is unusable
                max_other_loss = (
                    max(q10_loss, q90_loss) if (q10_loss > 0 and q90_loss > 0) else 1.0
                )
                q50_loss_ratio = (
                    q50_loss / max_other_loss if max_other_loss > 0 else float("inf")
                )
                model_usable = q50_loss_ratio <= 1.2  # Allow 20% tolerance

                if q50_loss > q10_loss or q50_loss > q90_loss:
                    print(
                        f"⚠️  检测到Q50 loss异常: Q50={q50_loss:.6f}, Q10={q10_loss:.6f}, Q90={q90_loss:.6f}"
                    )
                    print(f"   正在检查预测值分布和异常值...")

                    # Use a subset of data for prediction diagnostics (avoid full prediction if too large)
                    n_diagnostic = min(10000, len(X_df))
                    X_diagnostic = X_df.iloc[:n_diagnostic]
                    y_diagnostic = (
                        y_return.iloc[:n_diagnostic]
                        if isinstance(y_return, pd.Series)
                        else y_return[:n_diagnostic]
                    )

                    # Get predictions from all three models
                    pred_q10 = model_q10.model.predict(X_diagnostic.values)
                    pred_q50 = model_q50.model.predict(X_diagnostic.values)
                    pred_q90 = model_q90.model.predict(X_diagnostic.values)

                    # Check if predictions satisfy Q10 <= Q50 <= Q90
                    violation_count = np.sum(
                        ~((pred_q10 <= pred_q50) & (pred_q50 <= pred_q90))
                    )
                    violation_pct = violation_count / len(pred_q10) * 100

                    print(f"\n   📊 预测值合理性检查 (样本数: {n_diagnostic}):")
                    print(
                        f"      Q10 <= Q50 <= Q90 违反次数: {violation_count} ({violation_pct:.1f}%)"
                    )
                    if violation_pct > 1.0:
                        print(
                            f"      ⚠️  警告: 超过1%的预测值违反quantile顺序，说明模型训练可能有问题"
                        )

                # Check statistics
                print(f"\n   📊 目标变量 (y_return) 统计:")
                y_diagnostic_series = (
                    pd.Series(y_diagnostic)
                    if not isinstance(y_diagnostic, pd.Series)
                    else y_diagnostic
                )
                print(f"      均值: {y_diagnostic_series.mean():.6f}")
                print(f"      标准差: {y_diagnostic_series.std():.6f}")
                print(f"      最小值: {y_diagnostic_series.min():.6f}")
                print(f"      最大值: {y_diagnostic_series.max():.6f}")
                print(f"      中位数: {y_diagnostic_series.median():.6f}")
                print(f"      1%分位数: {y_diagnostic_series.quantile(0.01):.6f}")
                print(f"      99%分位数: {y_diagnostic_series.quantile(0.99):.6f}")

                # Check for extreme values
                extreme_threshold = 0.1  # 10%
                n_extreme = np.sum(np.abs(y_diagnostic_series) > extreme_threshold)
                if n_extreme > 0:
                    print(
                        f"      ⚠️  极端值 (|y| > {extreme_threshold}): {n_extreme} 个 ({n_extreme/len(y_diagnostic_series)*100:.1f}%)"
                    )

                # Check for moderate outliers (important for quantile regression)
                outlier_threshold_high = y_diagnostic_series.quantile(0.99)
                outlier_threshold_low = y_diagnostic_series.quantile(0.01)
                iqr = y_diagnostic_series.quantile(0.75) - y_diagnostic_series.quantile(
                    0.25
                )
                outlier_high = outlier_threshold_high + 3 * iqr
                outlier_low = outlier_threshold_low - 3 * iqr
                n_outliers = np.sum(
                    (y_diagnostic_series > outlier_high)
                    | (y_diagnostic_series < outlier_low)
                )
                if n_outliers > 0:
                    print(
                        f"      ⚠️  异常值 (3*IQR规则): {n_outliers} 个 ({n_outliers/len(y_diagnostic_series)*100:.1f}%)"
                    )
                    outlier_values = y_diagnostic_series[
                        (y_diagnostic_series > outlier_high)
                        | (y_diagnostic_series < outlier_low)
                    ]
                    print(
                        f"         异常值范围: [{outlier_values.min():.6f}, {outlier_values.max():.6f}]"
                    )
                    print(
                        f"         这些异常值可能来自AR(1)处理或数据质量问题，会影响Q50的pinball loss"
                    )

                print(f"\n   📊 Q10 预测值统计:")
                print(f"      均值: {np.mean(pred_q10):.6f}")
                print(f"      标准差: {np.std(pred_q10):.6f}")
                print(f"      最小值: {np.min(pred_q10):.6f}")
                print(f"      最大值: {np.max(pred_q10):.6f}")
                print(f"      中位数: {np.median(pred_q10):.6f}")

                print(f"\n   📊 Q50 预测值统计:")
                print(f"      均值: {np.mean(pred_q50):.6f}")
                print(f"      标准差: {np.std(pred_q50):.6f}")
                print(f"      最小值: {np.min(pred_q50):.6f}")
                print(f"      最大值: {np.max(pred_q50):.6f}")
                print(f"      中位数: {np.median(pred_q50):.6f}")

                print(f"\n   📊 Q90 预测值统计:")
                print(f"      均值: {np.mean(pred_q90):.6f}")
                print(f"      标准差: {np.std(pred_q90):.6f}")
                print(f"      最小值: {np.min(pred_q90):.6f}")
                print(f"      最大值: {np.max(pred_q90):.6f}")
                print(f"      中位数: {np.median(pred_q90):.6f}")

                # Check prediction bias
                print(f"\n   📊 预测偏差检查:")
                bias_q10 = np.mean(pred_q10 - y_diagnostic_series)
                bias_q50 = np.mean(pred_q50 - y_diagnostic_series)
                bias_q90 = np.mean(pred_q90 - y_diagnostic_series)
                print(f"      Q10 平均偏差: {bias_q10:.6f}")
                print(f"      Q50 平均偏差: {bias_q50:.6f}")
                print(f"      Q90 平均偏差: {bias_q90:.6f}")

                # Check if Q50 has systematic bias
                if (
                    abs(bias_q50) > abs(bias_q10) * 2
                    and abs(bias_q50) > abs(bias_q90) * 2
                ):
                    print(f"      ⚠️  Q50预测有系统性偏差，可能是导致loss异常的原因")

                # Check prediction variance
                print(f"\n   📊 预测方差检查:")
                var_q10 = np.var(pred_q10)
                var_q50 = np.var(pred_q50)
                var_q90 = np.var(pred_q90)
                var_y = y_diagnostic_series.var()
                print(f"      y_return 方差: {var_y:.6f}")
                print(f"      Q10 预测方差: {var_q10:.6f} (比例: {var_q10/var_y:.2f}x)")
                print(f"      Q50 预测方差: {var_q50:.6f} (比例: {var_q50/var_y:.2f}x)")
                print(f"      Q90 预测方差: {var_q90:.6f} (比例: {var_q90/var_y:.2f}x)")

                if var_q50 < var_q10 * 0.5 or var_q50 < var_q90 * 0.5:
                    print(
                        f"      ⚠️  Q50预测方差过小（{var_q50/var_y:.2f}x y_return方差），可能模型预测过于保守（欠拟合）"
                    )
                    print(
                        f"         Q50预测范围: [{np.min(pred_q50):.6f}, {np.max(pred_q50):.6f}]"
                    )
                    print(
                        f"         y_return实际范围: [{y_diagnostic_series.min():.6f}, {y_diagnostic_series.max():.6f}]"
                    )
                    print(
                        f"         Q50预测范围/实际范围 = {np.max(pred_q50) - np.min(pred_q50):.6f} / {y_diagnostic_series.max() - y_diagnostic_series.min():.6f} = {(np.max(pred_q50) - np.min(pred_q50)) / (y_diagnostic_series.max() - y_diagnostic_series.min()):.2%}"
                    )
                    if (np.max(pred_q50) - np.min(pred_q50)) / (
                        y_diagnostic_series.max() - y_diagnostic_series.min()
                    ) < 0.5:
                        print(
                            f"         ⚠️  Q50预测范围仅为实际范围的{(np.max(pred_q50) - np.min(pred_q50)) / (y_diagnostic_series.max() - y_diagnostic_series.min()):.1%}，模型严重欠拟合！"
                        )
                        print(
                            f"         当y_return有极端值时，Q50的保守预测会导致pinball loss异常增大"
                        )
                elif var_q50 > var_q10 * 2 or var_q50 > var_q90 * 2:
                    print(
                        f"      ⚠️  Q50预测方差过大（{var_q50/var_y:.2f}x y_return方差），可能模型预测过于激进（过拟合）"
                    )

                # Additional analysis: why Q50 loss is higher
                print("\n   🔍 深入分析：为什么Q50 loss更大？")

                # Calculate pinball loss manually for each quantile on diagnostic set
                y_array = y_diagnostic_series.values

                # Pinball loss for Q10
                pinball_q10 = np.mean(
                    np.maximum(0.1 * (y_array - pred_q10), 0.9 * (pred_q10 - y_array))
                )
                # Pinball loss for Q50
                pinball_q50 = np.mean(
                    np.maximum(0.5 * (y_array - pred_q50), 0.5 * (pred_q50 - y_array))
                )
                # Pinball loss for Q90
                pinball_q90 = np.mean(
                    np.maximum(0.9 * (y_array - pred_q90), 0.1 * (pred_q90 - y_array))
                )

                print(f"      诊断集上的pinball loss:")
                print(f"        Q10: {pinball_q10:.6f}")
                print(f"        Q50: {pinball_q50:.6f}")
                print(f"        Q90: {pinball_q90:.6f}")

                if pinball_q50 > pinball_q10 and pinball_q50 > pinball_q90:
                    print(f"      ⚠️  诊断集上Q50 loss确实更大，验证了异常")

                    # Check contribution of extreme values
                    mask_extreme = np.abs(y_array) > 0.01  # 1% threshold
                    if np.sum(mask_extreme) > 0:
                        pinball_q50_extreme = np.mean(
                            np.maximum(
                                0.5 * (y_array[mask_extreme] - pred_q50[mask_extreme]),
                                0.5 * (pred_q50[mask_extreme] - y_array[mask_extreme]),
                            )
                        )
                        pinball_q50_normal = np.mean(
                            np.maximum(
                                0.5
                                * (y_array[~mask_extreme] - pred_q50[~mask_extreme]),
                                0.5
                                * (pred_q50[~mask_extreme] - y_array[~mask_extreme]),
                            )
                        )
                        print(f"\n      极端值贡献分析 (|y| > 1%):")
                        print(
                            f"        极端值样本数: {np.sum(mask_extreme)} ({np.sum(mask_extreme)/len(y_array)*100:.1f}%)"
                        )
                        print(f"        Q50 loss (极端值): {pinball_q50_extreme:.6f}")
                        print(f"        Q50 loss (正常值): {pinball_q50_normal:.6f}")
                        if pinball_q50_extreme > pinball_q50_normal * 2:
                            print(
                                f"        ⚠️  极端值的loss贡献是正常值的{pinball_q50_extreme/pinball_q50_normal:.1f}倍！"
                            )
                            print(f"         Q50模型对极端值预测不足，导致loss异常增大")

                print("\n   💡 建议:")
                if var_q50 < var_q10 * 0.5 or var_q50 < var_q90 * 0.5:
                    print(f"      🔧 Q50模型预测过于保守（欠拟合），建议:")
                    print(
                        f"         1. 增加模型复杂度: 增加num_leaves（如从31增加到127）"
                    )
                    print(
                        f"         2. 减少正则化: 降低min_data_in_leaf（如从20降到10）"
                    )
                    print(f"         3. 增加训练轮数: 增加num_boost_round")
                    print(
                        f"         4. 调整learning_rate: 降低learning_rate以获得更精细的预测"
                    )

                if n_extreme > 0 or (
                    n_outliers > 0 if "n_outliers" in locals() else False
                ):
                    print(f"      🔧 处理极端值/异常值:")
                    print(
                        f"         1. 检查AR(1)处理: AR(1)残差可能有数值不稳定，导致极端值"
                    )
                    print(f"         2. 数据清洗: 考虑clip极端值到合理范围（如±3σ）")
                    print(
                        f"         3. 使用稳健的损失函数: 考虑使用Huber loss或trimmed loss"
                    )
                    print(
                        f"         4. 检查数据源: 确认原始数据质量，是否有错误或异常K线"
                    )

                print(f"      🔧 其他建议:")
                print(
                    f"         1. 如果quantile顺序违反>1%，可能是LightGBM训练不稳定，尝试使用固定随机种子"
                )
                print(f"         2. 检查特征质量: 特征可能不足以预测极端值")
                print(f"         3. 考虑使用ensemble: 多个模型的平均可能更稳健")

                # 🔧 Comprehensive validation framework
                # Reference: docs suggestion - comprehensive validation beyond just loss ratio
                print("\n" + "=" * 70)
                print("🔍 Comprehensive Model Validation (综合模型验证)")
                print("=" * 70)

                # Use diagnostic subset for comprehensive validation
                n_validation = min(10000, len(X_df))
                X_validation = X_df.iloc[:n_validation]
                y_validation = (
                    y_return.iloc[:n_validation]
                    if isinstance(y_return, pd.Series)
                    else y_return[:n_validation]
                )

                pred_q10_val = model_q10.model.predict(X_validation.values)
                pred_q50_val = model_q50.model.predict(X_validation.values)
                pred_q90_val = model_q90.model.predict(X_validation.values)
                y_validation_series = (
                    pd.Series(y_validation)
                    if not isinstance(y_validation, pd.Series)
                    else y_validation
                )

                # Helper function for pinball loss
                def pinball_loss(y_true, y_pred, tau=0.5):
                    """Calculate pinball loss (quantile loss)"""
                    resid = y_true - y_pred
                    return np.mean(
                        np.where(resid >= 0, tau * resid, (1 - tau) * (-resid))
                    )

                # 1. Basic loss validation
                pinball_loss_q10 = pinball_loss(
                    y_validation_series.values, pred_q10_val, 0.1
                )
                pinball_loss_q50 = pinball_loss(
                    y_validation_series.values, pred_q50_val, 0.5
                )
                pinball_loss_q90 = pinball_loss(
                    y_validation_series.values, pred_q90_val, 0.9
                )

                q50_vs_q10_ratio = (
                    pinball_loss_q50 / pinball_loss_q10
                    if pinball_loss_q10 > 0
                    else float("inf")
                )
                q50_vs_q90_ratio = (
                    pinball_loss_q50 / pinball_loss_q90
                    if pinball_loss_q90 > 0
                    else float("inf")
                )

                # Outlier loss ratio: top 1% vs bottom 99% residual contribution
                q50_residuals_abs = np.abs(y_validation_series.values - pred_q50_val)
                thresh_99 = np.percentile(q50_residuals_abs, 99)
                top1_residuals = q50_residuals_abs[q50_residuals_abs >= thresh_99]
                bottom99_residuals = q50_residuals_abs[q50_residuals_abs < thresh_99]
                outlier_loss_ratio = (
                    top1_residuals.mean() / (bottom99_residuals.mean() + 1e-12)
                    if len(bottom99_residuals) > 0
                    else float("inf")
                )

                print(f"   1. Basic Loss Validation (验证集):")
                print(f"      Q50/Q10 loss ratio: {q50_vs_q10_ratio:.3f} (应 ≤ 1.0)")
                print(f"      Q50/Q90 loss ratio: {q50_vs_q90_ratio:.3f} (应 ≤ 1.0)")
                print(
                    f"      Outlier loss ratio (top1% vs bottom99%): {outlier_loss_ratio:.2f} (应 ≤ 10.0)"
                )

                if outlier_loss_ratio > 10.0:
                    print(
                        f"      ⚠️  极端值loss贡献过高（{outlier_loss_ratio:.1f}x），说明少数极端值主导了Q50 loss"
                    )

                # 2. Prediction range validation (using 1st-99th percentile as in doc template)
                pred_range_99 = np.percentile(pred_q50_val, 99) - np.percentile(
                    pred_q50_val, 1
                )
                true_range_99 = np.percentile(
                    y_validation_series.values, 99
                ) - np.percentile(y_validation_series.values, 1)
                range_ratio = (
                    pred_range_99 / true_range_99 if true_range_99 > 0 else 0.0
                )

                print(f"\n   2. Prediction Range Validation:")
                print(f"      Predicted range (99th-1st): {pred_range_99:.6f}")
                print(f"      True range (99th-1st): {true_range_99:.6f}")
                print(f"      Range ratio: {range_ratio:.3f} (应在 [0.5, 2.0] 之间)")

                if range_ratio < 0.5:
                    print(f"      ⚠️  预测范围过窄（< 0.5），模型预测过于保守")
                elif range_ratio > 2.0:
                    print(f"      ⚠️  预测范围过宽（> 2.0），模型预测过于激进")
                else:
                    print(f"      ✅ 预测范围合理")

                # 3. Conditional coverage validation
                in_interval = (y_validation_series.values >= pred_q10_val) & (
                    y_validation_series.values <= pred_q90_val
                )
                conditional_coverage = in_interval.mean()
                expected_coverage = 0.8  # 80% for 10th-90th quantile interval

                print(f"\n   3. Conditional Coverage Validation:")
                print(
                    f"      Coverage (10th-90th interval): {conditional_coverage:.3f} (期望: {expected_coverage:.3f})"
                )

                if conditional_coverage < expected_coverage - 0.1:
                    print(
                        f"      ⚠️  覆盖率过低（< {expected_coverage-0.1:.1f}），预测区间可能过窄"
                    )
                elif conditional_coverage > expected_coverage + 0.1:
                    print(
                        f"      ⚠️  覆盖率过高（> {expected_coverage+0.1:.1f}），预测区间可能过宽"
                    )
                else:
                    print(f"      ✅ 覆盖率合理")

                # 4. Direction prediction stability check (by volatility regime)
                if len(y_validation_series) > 20:
                    validation_vol = (
                        pd.Series(y_validation_series)
                        .rolling(window=20, min_periods=5)
                        .std()
                    )
                    vol_median = validation_vol.median()
                    high_vol_mask = validation_vol > vol_median
                    low_vol_mask = ~high_vol_mask

                    if high_vol_mask.sum() > 10 and low_vol_mask.sum() > 10:
                        y_true_dir = (y_validation_series.values > 0).astype(int)
                        # Use dynamic threshold instead of fixed 0
                        threshold_val = _compute_direction_threshold(
                            pred_q50_val, y_true_dir, method=args.direction_threshold
                        )
                        y_pred_dir = (pred_q50_val > threshold_val).astype(int)

                        acc_high_vol = np.mean(
                            y_pred_dir[high_vol_mask] == y_true_dir[high_vol_mask]
                        )
                        acc_low_vol = np.mean(
                            y_pred_dir[low_vol_mask] == y_true_dir[low_vol_mask]
                        )

                        print(
                            f"\n   4. Direction Prediction Stability (by volatility regime):"
                        )
                        print(f"      Accuracy (high volatility): {acc_high_vol:.3f}")
                        print(f"      Accuracy (low volatility): {acc_low_vol:.3f}")
                        print(
                            f"      Difference: {abs(acc_high_vol - acc_low_vol):.3f}"
                        )

                        if abs(acc_high_vol - acc_low_vol) > 0.15:
                            print(
                                f"      ⚠️  方向准确率在不同波动率regime下差异较大（> 0.15），模型可能不稳定"
                            )
                        else:
                            print(f"      ✅ 方向准确率在不同波动率regime下表现稳定")

                # 5. Model acceptance criteria (enhanced with outlier loss ratio)
                print(f"\n   5. Model Acceptance Criteria:")
                acceptance_conditions = [
                    (q50_vs_q10_ratio <= 1.0, "Q50/Q10 loss ratio ≤ 1.0"),
                    (q50_vs_q90_ratio <= 1.0, "Q50/Q90 loss ratio ≤ 1.0"),
                    (0.5 <= range_ratio <= 2.0, "Range ratio in [0.5, 2.0]"),
                    (conditional_coverage >= 0.7, "Conditional coverage ≥ 0.7"),
                    (outlier_loss_ratio <= 10.0, "Outlier loss ratio ≤ 10.0"),
                ]

                all_passed = True
                for condition, description in acceptance_conditions:
                    status = "✅" if condition else "❌"
                    print(f"      {status} {description}")
                    if not condition:
                        all_passed = False

                if all_passed:
                    print(f"\n   ✅ 所有验证标准通过，模型健康状况良好！")
                else:
                    print(f"\n   ⚠️  部分验证标准未通过，模型可能存在问题")

                print("=" * 70)

                if q50_loss > q10_loss or q50_loss > q90_loss:
                    pass  # Already printed above
                else:
                    print(
                        f"✅ Quantile loss正常: Q50={q50_loss:.6f}, Q10={q10_loss:.6f}, Q90={q90_loss:.6f}"
                    )
                    if model_usable:
                        print(
                            f"✅ 模型状态: ✅ 可用 (Q50 loss ratio: {q50_loss_ratio:.2f})"
                        )

                # Mark model as unusable if Q50 loss ratio > 1.2
                if q50_loss_ratio > 1.2:
                    model_usable = False
                    print("\n" + "=" * 70)
                    print("🚨 模型标记为不可用 (Model Marked as UNUSABLE)")
                    print("=" * 70)
                    print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                    print(f"   Q50 loss ratio: {q50_loss_ratio:.2f} (阈值: 1.2)")
                    print(f"   Q50 loss: {q50_loss:.6f}")
                    print(f"   Max(Q10, Q90) loss: {max_other_loss:.6f}")
                    print(
                        f"\n   ⚠️  根据文档要求（docs/极端值：确保 Q50 loss ≤ Q10Q90 loss.md）："
                    )
                    print(f"      Q50 loss必须 ≤ Q10/Q90 loss，否则模型不可用")
                    print(
                        f"      当前Q50 loss是Q10/Q90的{q50_loss_ratio:.1f}倍，违反了分位数回归基本性质"
                    )
                    print(f"\n   🔒 模型状态: ❌ 不可用")
                    print(f"   💡 建议: 不要使用此模型进行预测，重新训练或检查数据质量")
                    print("=" * 70 + "\n")

                print("=" * 70 + "\n")

                # Store model usability flag for later use (will be saved to training_info.json)
                quantile_model_usable = model_usable

                # Auto-remediation: If Q50 model is unusable, apply fixes and retrain (max 1 time)
                # Reference: docs/极端值：确保 Q50 loss ≤ Q10Q90 loss.md
                retrain_attempted = False
                calibration_params = {"enabled": False}  # Initialize calibration params
                if not model_usable and q50_loss_ratio > 1.2:
                    print("\n" + "=" * 70)
                    print("🔧 自动修复和重训 (Auto-Remediation and Retraining)")
                    print("=" * 70)
                    print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                    print(f"   检测到Q50 loss异常，开始自动修复和重训...")
                print("=" * 70)

                retrain_attempted = True

                # Step 1: More aggressive Winsorize on y_return and features X
                print("\n   步骤1: 更激进的Winsorize处理（y_return和特征X）")

                # Helper function for robust winsorize
                def robust_winsorize(data, k=3.0):
                    """Robust winsorize based on MAD"""
                    if isinstance(data, pd.Series):
                        data = data.values
                    median = np.median(data)
                    mad = np.median(np.abs(data - median))
                    if mad == 0:
                        return data
                    sigma = 1.4826 * mad  # Convert MAD to approximate std
                    lower = median - k * sigma
                    upper = median + k * sigma
                    return np.clip(data, lower, upper)

                # Enhanced Winsorize with adaptive parameters based on data characteristics
                def adaptive_winsorize(data, base_k=3.0, skew_threshold=1.0):
                    """Adaptive winsorize that adjusts k based on data skewness"""
                    if isinstance(data, pd.Series):
                        data = data.values

                    # Calculate skewness to determine if we need more aggressive clipping
                    skewness = scipy.stats.skew(data)
                    k = base_k

                    # Increase clipping aggressiveness for highly skewed data
                    if abs(skewness) > skew_threshold:
                        k = base_k * (1 + abs(skewness) / 2)
                        k = min(k, base_k * 2)  # Cap at 2x base_k

                    median = np.median(data)
                    mad = np.median(np.abs(data - median))
                    if mad == 0:
                        return data
                    sigma = 1.4826 * mad  # Convert MAD to approximate std
                    lower = median - k * sigma
                    upper = median + k * sigma
                    return np.clip(data, lower, upper)

                if target_winsorize_k > 0:
                    # Winsorize y_return with adaptive parameters
                    y_return_original = y_return.copy()
                    y_return = pd.Series(
                        adaptive_winsorize(y_return, base_k=2.5), index=y_return.index
                    )
                    n_clipped_y = np.sum(np.abs(y_return - y_return_original) > 1e-10)
                    if n_clipped_y > 0:
                        print(
                            f"      ✅ y_return: 已clip {n_clipped_y}个极端值（k=2.5，自适应）"
                        )
                else:
                    print("      ℹ️ 已禁用目标 winsorize，跳过 y_return 自适应剪裁")

                # Winsorize features X (only for return-based and derived features, not raw prices)
                # Apply winsorize to all numeric columns (features are already normalized/derived)
                print(
                    f"      正在对特征X进行Winsorize处理（{len(X_df.columns)}个特征）..."
                )
                X_df_original = X_df.copy()
                n_clipped_features = 0
                if feature_winsorize_k > 0:
                    for col in X_df.columns:
                        if X_df[col].dtype in [
                            np.float64,
                            np.float32,
                            np.int64,
                            np.int32,
                        ]:
                            X_df[col] = adaptive_winsorize(
                                X_df[col], base_k=4.0
                            )  # Use adaptive winsorize with base k=4.0
                            n_clipped = np.sum(
                                np.abs(X_df[col] - X_df_original[col]) > 1e-10
                            )
                            if n_clipped > 0:
                                n_clipped_features += 1
                    if n_clipped_features > 0:
                        print(
                            f"      ✅ 特征X: {n_clipped_features}个特征包含极端值并已clip（自适应参数）"
                        )
                    else:
                        print(f"      ✅ 特征X: 未发现极端值")
                else:
                    print("      ℹ️ 已禁用特征 winsorize，跳过特征自适应剪裁")

                # Step 2: Calculate sample weights to reduce extreme value influence
                print("\n   步骤2: 计算样本权重（降低极端值影响）")
                # Use initial Q50 predictions to identify outliers
                # Ensure y_return is a Series for alignment
                if not isinstance(y_return, pd.Series):
                    y_return = pd.Series(
                        y_return,
                        index=(
                            X_df.index
                            if hasattr(X_df, "index")
                            else range(len(y_return))
                        ),
                    )
                pred_q50_initial = model_q50.model.predict(X_df.values)
                residuals = y_return.values - pred_q50_initial

                # Enhanced sample weighting with multiple strategies
                def compute_sample_weights(residuals, method="huber"):
                    """
                    Compute sample weights using different strategies to reduce
                    the influence of extreme values.

                    Args:
                        residuals: Model residuals (y_true - y_pred)
                        method: Weighting method
                            - "huber": Huber-like weighting (current method)
                            - "tukey": Tukey's biweight function (more aggressive)
                            - "exponential": Exponential decay weighting
                            - "combined": Combination of Huber and Tukey
                    """
                    if method == "huber":
                        # Current Huber-like weighting
                        residual_median = np.median(np.abs(residuals))
                        delta = 2.0 * residual_median  # Huber threshold
                        # Weight: 1.0 for normal residuals, decreasing for extreme residuals
                        sample_weights = np.where(
                            np.abs(residuals) < delta, 1.0, delta / np.abs(residuals)
                        )
                    elif method == "tukey":
                        # Tukey's biweight function (more aggressive down-weighting)
                        residual_median = np.median(np.abs(residuals))
                        delta = 2.0 * residual_median
                        # Normalize residuals
                        u = np.abs(residuals) / delta
                        # Tukey's biweight: (1 - (u/delta)^2)^2 for |u| <= delta, 0 otherwise
                        sample_weights = np.where(u <= 1, (1 - u**2) ** 2, 0)
                    elif method == "exponential":
                        # Exponential decay weighting
                        residual_std = np.std(residuals)
                        # Exponential decay: exp(-|residual| / (k * std))
                        k = 2.0  # Decay rate parameter
                        sample_weights = np.exp(-np.abs(residuals) / (k * residual_std))
                    elif method == "combined":
                        # Combined approach: Huber for moderate outliers, Tukey for extreme
                        residual_median = np.median(np.abs(residuals))
                        delta = 2.0 * residual_median
                        u = np.abs(residuals) / delta

                        # For |u| <= 1: Huber-like weighting
                        # For |u| > 1: Tukey's biweight
                        sample_weights = np.where(
                            u <= 1,
                            np.where(u < 0.5, 1.0, delta / np.abs(residuals)),
                            (1 - np.minimum(u, 2) ** 2)
                            ** 2,  # Tukey for extreme values
                        )
                    else:
                        raise ValueError(f"Unknown weighting method: {method}")

                    # Ensure no zero weights (add small epsilon)
                    sample_weights = np.maximum(sample_weights, 1e-6)

                    # Normalize weights to have mean=1.0
                    sample_weights = sample_weights / np.mean(sample_weights)
                    return sample_weights

                # Try different weighting strategies and select the best one
                weighting_methods = ["huber", "tukey", "exponential", "combined"]
                best_weights = None
                best_weighting_method = "huber"

                print(f"      尝试不同的样本权重策略...")
                for method in weighting_methods:
                    try:
                        weights = compute_sample_weights(residuals, method=method)
                        # Evaluate weights by computing weighted loss
                        weighted_loss = np.mean(weights * np.abs(residuals))
                        print(f"        {method}: 加权平均损失 = {weighted_loss:.6f}")

                        # Select the method that gives the lowest weighted loss
                        if best_weights is None or weighted_loss < np.mean(
                            best_weights * np.abs(residuals)
                        ):
                            best_weights = weights
                            best_weighting_method = method
                    except Exception as e:
                        print(f"        {method}: 计算失败 ({str(e)})")

                sample_weights = best_weights
                print(f"      选择最佳权重策略: {best_weighting_method}")

                print(
                    f"      样本权重统计: min={np.min(sample_weights):.4f}, max={np.max(sample_weights):.4f}, mean={np.mean(sample_weights):.4f}"
                )
                n_low_weight = np.sum(sample_weights < 0.5)
                print(
                    f"      低权重样本数（权重<0.5）: {n_low_weight} ({n_low_weight/len(sample_weights)*100:.1f}%)"
                )

                # Step 3: Adjust Q50 model parameters for better robustness
                # Reference: docs/极端值：确保 Q50 loss ≤ Q10Q90 loss.md
                print("\n   步骤3: 调整Q50模型参数（增加正则化，防止过拟合噪声）")
                from time_series_model.config.settings import DEFAULT_LGBM_PARAMS

                q50_params_retrain = DEFAULT_LGBM_PARAMS.copy()

                # Determine appropriate min_data_in_leaf based on sample size
                # For high-frequency (5T), use 100-500 as recommended in doc
                n_samples = len(y_return)
                if n_samples < 1000:
                    min_data_in_leaf = 50
                elif n_samples < 5000:
                    min_data_in_leaf = 100
                elif n_samples < 20000:
                    min_data_in_leaf = 200
                else:
                    min_data_in_leaf = 300

                # Enhanced parameter adjustment based on frequency and sample size
                # More aggressive regularization for high-frequency data
                if freq in ["5T", "15T"]:
                    num_leaves = 31  # More conservative for high-frequency
                    learning_rate = 0.01  # Slower learning for stability
                    lambda_l1 = 10.0  # Stronger L1 regularization
                    lambda_l2 = 10.0  # Stronger L2 regularization
                    feature_fraction = 0.7  # Use fewer features to prevent overfitting
                else:
                    num_leaves = 63
                    learning_rate = 0.02
                    lambda_l1 = 5.0
                    lambda_l2 = 5.0
                    feature_fraction = 0.8

                q50_params_retrain.update(
                    {
                        "num_leaves": num_leaves,
                        "min_data_in_leaf": min_data_in_leaf,
                        "learning_rate": learning_rate,
                        "lambda_l1": lambda_l1,
                        "lambda_l2": lambda_l2,
                        "feature_fraction": feature_fraction,
                        "bagging_fraction": 0.8,  # Add bagging for additional robustness
                        "bagging_freq": 5,  # Bagging frequency
                        "min_gain_to_split": 0.01,  # Minimum gain to make a split
                        "max_depth": 8,  # Limit tree depth to prevent overfitting
                    }
                )
                print(f"      参数调整（针对样本数={n_samples}，频率={freq}）:")
                print(f"        num_leaves: 31 → {num_leaves}")
                print(
                    f"        min_data_in_leaf: 20 → {min_data_in_leaf}（根据样本数自适应）"
                )
                print(f"        learning_rate: 0.05 → {learning_rate}")
                print(f"        lambda_l1: 0 → {lambda_l1}")
                print(f"        lambda_l2: 0 → {lambda_l2}")
                print(f"        feature_fraction: 0.9 → {feature_fraction}")
                print(f"        bagging_fraction: 1.0 → 0.8")
                print(f"        max_depth: -1 → 8")
                # Step 4: Retrain Q50 model with fixed data and weights
                print("\n   步骤4: 使用修复后的数据和权重重训Q50模型")

                # Enhanced training with multiple strategies
                def enhanced_q50_training(
                    X_df, y_return, sample_weights, q50_params_retrain, args
                ):
                    """
                    Enhanced Q50 training with multiple strategies to improve model robustness.

                    Args:
                        X_df: Feature data
                        y_return: Target values
                        sample_weights: Sample weights
                        q50_params_retrain: Model parameters
                        args: Command line arguments

                    Returns:
                        Trained model and metrics
                    """
                    from time_series_model.models.lightgbm_model import LightGBMTrainer

                    # Strategy 1: Original training
                    model_q50_retrain = LightGBMTrainer(
                        model_type="quantile",
                        quantile_alpha=0.5,
                        params=q50_params_retrain,
                        use_gpu=args.gpu,
                    )

                    # Ensure sample_weights is numpy array
                    if not isinstance(sample_weights, np.ndarray):
                        sample_weights = np.array(sample_weights)

                    q50_metrics_retrain, q50_preprocess_params_retrain = (
                        model_q50_retrain.train(
                            X_df,
                            y_return,
                            n_splits=max(2, args.cv_folds or 2),
                            use_time_series_cv=True,
                            sample_weight=sample_weights,
                        )
                    )

                    # Strategy 2: Ensemble approach - train multiple models with different parameters
                    # This can help improve robustness by reducing variance
                    if freq in ["5T", "15T"]:
                        print(f"      🔄 训练集成模型以提高鲁棒性...")

                        # Create multiple models with slightly different parameters
                        ensemble_models = []
                        ensemble_metrics = []

                        # Base parameters
                        base_params = q50_params_retrain.copy()

                        # Train 3 models with different parameters
                        for i in range(3):
                            # Perturb parameters slightly
                            perturbed_params = base_params.copy()
                            perturbed_params["seed"] = 42 + i  # Different random seed
                            perturbed_params["feature_fraction"] = min(
                                1.0, base_params["feature_fraction"] + (i - 1) * 0.05
                            )
                            perturbed_params["bagging_fraction"] = min(
                                1.0,
                                base_params.get("bagging_fraction", 1.0)
                                + (i - 1) * 0.05,
                            )

                            model = LightGBMTrainer(
                                model_type="quantile",
                                quantile_alpha=0.5,
                                params=perturbed_params,
                                use_gpu=args.gpu,
                            )

                            metrics, preprocess_params = model.train(
                                X_df,
                                y_return,
                                n_splits=max(2, args.cv_folds or 2),
                                use_time_series_cv=True,
                                sample_weight=sample_weights,
                            )

                            ensemble_models.append(model)
                            ensemble_metrics.append(metrics)

                        # Select the best model based on CV loss
                        cv_losses = [
                            metrics.get("cv_quantile_loss", float("inf"))
                            for metrics in ensemble_metrics
                        ]
                        best_idx = np.argmin(cv_losses)

                        print(
                            f"      选择最佳集成模型 (CV loss: {cv_losses[best_idx]:.6f})"
                        )

                        # Use the best model
                        model_q50_retrain = ensemble_models[best_idx]
                        q50_metrics_retrain = ensemble_metrics[best_idx]
                        q50_preprocess_params_retrain = (
                            ensemble_models[best_idx].preprocess_params
                            if hasattr(ensemble_models[best_idx], "preprocess_params")
                            else {}
                        )

                    return (
                        model_q50_retrain,
                        q50_metrics_retrain,
                        q50_preprocess_params_retrain,
                    )

                # Use enhanced training
                (
                    model_q50_retrain,
                    q50_metrics_retrain,
                    q50_preprocess_params_retrain,
                ) = enhanced_q50_training(
                    X_df, y_return, sample_weights, q50_params_retrain, args
                )

                # Step 5: Re-check Q50 loss and prediction range coverage
                print("\n   步骤5: 重新检查Q50 loss和预测范围覆盖")
                # q50_metrics_retrain is a dict, extract cv_quantile_loss
                q50_loss_retrain = q50_metrics_retrain.get("cv_quantile_loss", 0)
                q50_loss_ratio_retrain = (
                    q50_loss_retrain / max_other_loss
                    if max_other_loss > 0
                    else float("inf")
                )

                print(
                    f"      重训前: Q50 loss = {q50_loss:.6f}, ratio = {q50_loss_ratio:.2f}"
                )
                print(
                    f"      重训后: Q50 loss = {q50_loss_retrain:.6f}, ratio = {q50_loss_ratio_retrain:.2f}"
                )

                # Check prediction range coverage
                # Use a subset for diagnostics (avoid full prediction if too large)
                n_diagnostic = min(10000, len(X_df))
                X_diagnostic = X_df.iloc[:n_diagnostic]
                y_diagnostic = (
                    y_return.iloc[:n_diagnostic]
                    if isinstance(y_return, pd.Series)
                    else y_return[:n_diagnostic]
                )

                pred_q50_retrain = model_q50_retrain.model.predict(X_diagnostic.values)
                pred_range = np.percentile(pred_q50_retrain, 99) - np.percentile(
                    pred_q50_retrain, 1
                )
                true_range = np.percentile(y_diagnostic, 99) - np.percentile(
                    y_diagnostic, 1
                )
                coverage = pred_range / true_range if true_range > 0 else 0.0

                print(
                    f"      预测范围覆盖: {coverage:.2%} (预测范围={pred_range:.6f}, 真实范围={true_range:.6f})"
                )

                if coverage < 0.3:
                    print(f"      ⚠️  警告: 预测范围覆盖 < 30%，预测范围过窄")

                # Step 6: Range Calibration (if coverage is too low)
                # Reference: docs template - use 1st-99th percentile for range calculation
                pred_q50_calibrated = None
                calibration_params = None
                if coverage < 0.5 and true_range > 0:
                    print(f"\n   步骤6: 范围校准（Range Calibration）")

                    # Use 1st-99th percentile range (as in doc template)
                    true_range_99 = np.percentile(
                        y_diagnostic.values, 99
                    ) - np.percentile(y_diagnostic.values, 1)
                    pred_range_99 = np.percentile(pred_q50_retrain, 99) - np.percentile(
                        pred_q50_retrain, 1
                    )

                    scale_factor = true_range_99 / (pred_range_99 + 1e-8)
                    # Limit scale factor to prevent over-amplification (clamp 1.0-3.0 as in doc)
                    scale_factor = min(max(scale_factor, 1.0), 3.0)
                    print(f"      缩放因子: {scale_factor:.2f} (限制在[1.0, 3.0]之间)")

                    # Precompute medians for parameter logging regardless of method
                    pred_median = float(np.median(pred_q50_retrain))
                    true_median = float(np.median(y_diagnostic.values))

                    # Enhanced calibration: Multiple calibration strategies
                    def enhanced_calibration(
                        pred_q50_retrain, y_diagnostic_values, method="shift_scale"
                    ):
                        """
                        Enhanced calibration with multiple strategies.

                        Args:
                            pred_q50_retrain: Predictions from retrained model
                            y_diagnostic_values: True values for diagnostics
                            method: Calibration method
                                - "shift_scale": Original method (shift to match median, then scale)
                                - "quantile_mapping": Quantile mapping calibration
                                - "isotonic": Isotonic regression calibration
                                - "combined": Combination of shift_scale and quantile_mapping
                        """
                        if method == "shift_scale":
                            # Original method: shift to match median, then scale
                            pred_median = np.median(pred_q50_retrain)
                            true_median = np.median(y_diagnostic_values)
                            calibrated = (
                                pred_q50_retrain - pred_median
                            ) * scale_factor + true_median
                        elif method == "quantile_mapping":
                            # Quantile mapping calibration
                            # Map prediction quantiles to true quantiles
                            pred_sorted = np.sort(pred_q50_retrain)
                            true_sorted = np.sort(y_diagnostic_values)

                            # Create quantile mapping function
                            pred_quantiles = np.linspace(0, 1, len(pred_sorted))
                            true_quantiles = np.linspace(0, 1, len(true_sorted))

                            # Interpolate to map prediction quantiles to true quantiles
                            quantile_map = interp1d(
                                pred_sorted,
                                np.percentile(
                                    y_diagnostic_values,
                                    np.linspace(0, 100, len(pred_sorted)),
                                ),
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            calibrated = quantile_map(pred_q50_retrain)
                        elif method == "isotonic":
                            # Isotonic regression calibration
                            iso_reg = IsotonicRegression(out_of_bounds="clip")
                            # Use a subset for fitting to avoid overfitting
                            n_fit = min(5000, len(pred_q50_retrain))
                            idx_fit = np.random.choice(
                                len(pred_q50_retrain), n_fit, replace=False
                            )
                            iso_reg.fit(
                                pred_q50_retrain[idx_fit], y_diagnostic_values[idx_fit]
                            )
                            calibrated = iso_reg.predict(pred_q50_retrain)
                        elif method == "combined":
                            # Combined approach: first shift_scale, then fine-tune with quantile mapping
                            # Step 1: Apply shift_scale calibration
                            pred_median = np.median(pred_q50_retrain)
                            true_median = np.median(y_diagnostic_values)
                            intermediate = (
                                pred_q50_retrain - pred_median
                            ) * scale_factor + true_median

                            # Step 2: Apply quantile mapping to fine-tune
                            from scipy.interpolate import interp1d

                            intermediate_sorted = np.sort(intermediate)
                            true_sorted = np.sort(y_diagnostic_values)

                            # Create quantile mapping function
                            quantile_map = interp1d(
                                intermediate_sorted,
                                np.percentile(
                                    y_diagnostic_values,
                                    np.linspace(0, 100, len(intermediate_sorted)),
                                ),
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            calibrated = quantile_map(intermediate)
                        else:
                            raise ValueError(f"Unknown calibration method: {method}")

                        return calibrated

                    # Try different calibration strategies and select the best one
                    calibration_methods = [
                        "shift_scale",
                        "quantile_mapping",
                        "combined",
                    ]
                    best_calibrated = None
                    best_calibration_method = "shift_scale"
                    best_coverage = coverage

                    print(f"      尝试不同的校准策略...")
                    for method in calibration_methods:
                        try:
                            calibrated = enhanced_calibration(
                                pred_q50_retrain, y_diagnostic.values, method=method
                            )

                            # Evaluate calibration by computing coverage after calibration
                            calibrated_range = np.percentile(
                                calibrated, 99
                            ) - np.percentile(calibrated, 1)
                            calibrated_coverage = (
                                calibrated_range / true_range_99
                                if true_range_99 > 0
                                else 0.0
                            )

                            print(
                                f"        {method}: 校准后覆盖 = {calibrated_coverage:.2%}"
                            )

                            # Select the method that gives the best coverage without overfitting
                            if (
                                calibrated_coverage > best_coverage
                                and calibrated_coverage <= 1.2
                            ):
                                best_calibrated = calibrated
                                best_calibration_method = method
                                best_coverage = calibrated_coverage
                        except Exception as e:
                            print(f"        {method}: 校准失败 ({str(e)})")

                    # Use the best calibration result
                    if best_calibrated is not None:
                        pred_q50_calibrated = best_calibrated
                        print(f"      选择最佳校准策略: {best_calibration_method}")
                    else:
                        # Fall back to original shift_scale method
                        pred_median = np.median(pred_q50_retrain)
                        true_median = np.median(y_diagnostic.values)
                        pred_q50_calibrated = (
                            pred_q50_retrain - pred_median
                        ) * scale_factor + true_median
                        print(f"      回退到原始校准策略: shift_scale")

                    # Recalculate coverage after calibration using 1st-99th percentile
                    pred_range_calibrated = np.percentile(
                        pred_q50_calibrated, 99
                    ) - np.percentile(pred_q50_calibrated, 1)
                    coverage_calibrated = (
                        pred_range_calibrated / true_range_99
                        if true_range_99 > 0
                        else 0.0
                    )
                    print(f"      校准前覆盖: {coverage:.2%}")
                    print(f"      校准后覆盖: {coverage_calibrated:.2%}")

                    # Re-diagnose with calibrated predictions
                    pinball_loss_q50_cal = np.mean(
                        np.maximum(
                            0.5 * (y_diagnostic.values - pred_q50_calibrated),
                            0.5 * (pred_q50_calibrated - y_diagnostic.values),
                        )
                    )
                    q50_loss_ratio_cal = (
                        pinball_loss_q50_cal / max_other_loss
                        if max_other_loss > 0
                        else float("inf")
                    )
                    print(
                        f"      校准后Q50 loss: {pinball_loss_q50_cal:.6f}, ratio: {q50_loss_ratio_cal:.2f}"
                    )

                    # Store calibration parameters for later use in OOS evaluation
                    calibration_params = {
                        "enabled": True,
                        "scale_factor": float(scale_factor),
                        "pred_median": float(pred_median),
                        "true_median": float(true_median),
                        "coverage_before": float(coverage),
                        "coverage_after": float(coverage_calibrated),
                        "q50_loss_ratio_calibrated": float(q50_loss_ratio_cal),
                        "method": best_calibration_method,
                    }

                    # If calibration improves the loss ratio, use it
                    if q50_loss_ratio_cal < q50_loss_ratio_retrain:
                        print(
                            f"      ✅ 校准后Q50 loss ratio改善（{q50_loss_ratio_retrain:.2f} → {q50_loss_ratio_cal:.2f}）"
                        )
                        calibration_applied = True
                    else:
                        print(
                            f"      ⚠️  校准后Q50 loss ratio未改善（{q50_loss_ratio_retrain:.2f} → {q50_loss_ratio_cal:.2f}），不应用校准"
                        )
                        calibration_applied = False
                        calibration_params = {"enabled": False}
                else:
                    calibration_params = {"enabled": False}
                    calibration_applied = False

                # Final decision: use retrained model if loss ratio is acceptable
                if q50_loss_ratio_retrain <= 1.2:
                    print(
                        f"\n      ✅ 重训成功！Q50 loss ratio降至{q50_loss_ratio_retrain:.2f}，模型现在可用"
                    )
                    if calibration_applied:
                        print(
                            f"      ✅ 已应用范围校准，预测范围覆盖从{coverage:.2%}提升至{coverage_calibrated:.2%}"
                        )
                    # Use retrained model
                    model_q50 = model_q50_retrain
                    q50_metrics = q50_metrics_retrain
                    q50_loss = q50_loss_retrain
                    q50_loss_ratio = q50_loss_ratio_retrain
                    quantile_model_usable = True
                    model_usable = True
                    remediation_status = "success"
                else:
                    print(
                        f"\n      ⚠️  重训后Q50 loss ratio仍为{q50_loss_ratio_retrain:.2f}，模型仍不可用"
                    )
                    if coverage < 0.3:
                        print(
                            f"      ⚠️  预测范围覆盖仅{coverage:.2%}，说明模型预测过于保守"
                        )
                    print(f"      这可能表示数据质量问题或需要更激进的修复措施")
                    print(
                        f"      🔒 根据文档建议：所有修复策略均失败，标记Q50为禁用（disabled_q50）"
                    )
                    print(
                        f"      ⚠️  建议：不要使用此模型的Q50信号进行实盘交易，或回退到上一次稳定模型"
                    )
                    # Keep original model but mark as unusable
                    quantile_model_usable = False
                    model_usable = False
                    remediation_status = "disabled_q50"

                print("=" * 70 + "\n")

            # volatility: regression model for volatility prediction
            model_vol = LightGBMTrainer(model_type="regression", use_gpu=args.gpu)
            vol_metrics, vol_preprocess_params = model_vol.train(
                X_df, y_vol, n_splits=n_splits, use_time_series_cv=use_cv
            )
            # Directional metrics (derived from q50 regression) and regression metrics containers
            oos_metrics = {}
            directional_metrics_train = {}
            if len(oos_df) > 0:
                from sklearn.metrics import (
                    mean_squared_error,
                    mean_absolute_error,
                    accuracy_score,
                    precision_recall_fscore_support,
                    roc_auc_score,
                    average_precision_score,
                )

                X_oos = oos_df[feature_cols].values
                y_ret_oos = oos_df["future_return"].values
                y_vol_oos = oos_df["future_volatility"].values

                # Use appropriate model based on model type
                if args.model_type == "classification":
                    # For classification, use return regression model for magnitude prediction
                    y_pred_return = model_return.model.predict(X_oos)
                    y_pred_q50 = (
                        y_pred_return  # Use return regression for OOS evaluation
                    )
                    y_pred_q10 = None
                    y_pred_q90 = None
                else:
                    # For quantile, use Q50 predictions
                    y_pred_q50 = model_q50.model.predict(X_oos)
                    y_pred_q10 = model_q10.model.predict(X_oos)
                    y_pred_q90 = model_q90.model.predict(X_oos)

                # Apply range calibration if available (from retraining) - only for quantile mode
                if (
                    args.model_type != "classification"
                    and "calibration_params" in locals()
                    and calibration_params
                    and calibration_params.get("enabled", False)
                ):
                    print(f"   🔧 Applying range calibration to OOS predictions")
                    scale_factor = calibration_params["scale_factor"]
                    pred_median = calibration_params["pred_median"]
                    true_median = calibration_params["true_median"]
                    y_pred_q50 = (y_pred_q50 - pred_median) * scale_factor + true_median
                    print(
                        f"      Calibration: scale={scale_factor:.2f}, pred_median={pred_median:.6f}, true_median={true_median:.6f}"
                    )

                # Calculate OOS regression metrics
                oos_rmse = float(np.sqrt(mean_squared_error(y_ret_oos, y_pred_q50)))
                oos_mae = float(mean_absolute_error(y_ret_oos, y_pred_q50))

                # Calculate R² for regression models
                from sklearn.metrics import r2_score

                oos_r2 = float(r2_score(y_ret_oos, y_pred_q50))

                # Quantile-specific metrics (only for quantile mode)
                if args.model_type != "classification":
                    coverage = float(
                        np.mean((y_ret_oos >= y_pred_q10) & (y_ret_oos <= y_pred_q90))
                    )
                    width = float(np.mean(np.maximum(0.0, y_pred_q90 - y_pred_q10)))
                    conf = float(
                        np.mean(
                            np.abs(y_pred_q50)
                            / (np.maximum(1e-8, y_pred_q90 - y_pred_q10))
                        )
                    )
                else:
                    coverage = None
                    width = None
                    conf = None

                y_pred_vol = model_vol.model.predict(X_oos)
                oos_vol_rmse = float(np.sqrt(mean_squared_error(y_vol_oos, y_pred_vol)))
                oos_vol_mae = float(mean_absolute_error(y_vol_oos, y_pred_vol))
                # Derive directional metrics from q50 regression (direction prediction)
                from scipy.stats import spearmanr, pearsonr

                y_true_dir = (y_ret_oos > 0).astype(int)
                y_score = y_pred_q50

                # 🔍 DIAGNOSTIC: Calculate prediction statistics first
                y_score_min = float(np.min(y_score))
                y_score_max = float(np.max(y_score))
                y_score_mean = float(np.mean(y_score))
                y_score_median = float(np.median(y_score))

                # Use dynamic threshold instead of fixed 0 (recommended solution)
                # This addresses the issue where all predictions <= 0 lead to F1=0 even with high AUC
                threshold_oos_opt = _compute_direction_threshold(
                    y_score, y_true_dir, method=args.direction_threshold
                )
                y_pred_dir = (y_score > threshold_oos_opt).astype(int)

                # Log threshold information
                if threshold_oos_opt != 0.0:
                    print(f"\n   📊 方向预测阈值优化（OOS）:")
                    print(f"      方法: {args.direction_threshold}")
                    print(f"      最佳阈值: {threshold_oos_opt:.6f} (固定阈值0: 0.0)")
                    print(f"      预测值范围: [{y_score_min:.6f}, {y_score_max:.6f}]")
                    print(f"      预测值中位数: {y_score_median:.6f}")

                # 🔍 DIAGNOSTIC: Check for F1=0 but high AUC anomaly
                # This occurs when all predictions are <= 0 (y_pred_dir all 0)
                # but AUC is high because predictions have good ranking (even if all negative)
                pred_positive_ratio = float(np.mean(y_pred_dir))
                true_positive_ratio = float(np.mean(y_true_dir))

                # Check for anomaly: F1=0 but potential high AUC
                if pred_positive_ratio == 0.0:
                    print("\n" + "=" * 70)
                    print("⚠️  警告：检测到方向预测异常（所有预测值 ≤ 0）！")
                    print("=" * 70)
                    print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                    print(f"   q50_pred 统计:")
                    print(f"     min: {y_score_min:.6f}, max: {y_score_max:.6f}")
                    print(
                        f"     mean: {y_score_mean:.6f}, median: {y_score_median:.6f}"
                    )
                    print(f"     预测为'涨'的比例: {pred_positive_ratio:.1%}")
                    print(f"     真实'涨'的比例: {true_positive_ratio:.1%}")
                    print(
                        f"   ⚠️  模型从未预测'涨'，F1 将为 0，但 AUC 可能仍很高（如果预测值排序正确）"
                    )
                    print(f"   💡 这通常意味着模型预测值整体偏负，可能需要检查：")
                    print(f"      - 模型是否过度保守（预测值偏负）")
                    print(
                        f"      - 预处理是否导致预测值偏移（如 AR(1) 残差的均值偏移）"
                    )
                    print(f"      - 是否需要调整预测阈值（使用分位数而非 0）")
                    print("=" * 70 + "\n")

                acc = float(accuracy_score(y_true_dir, y_pred_dir))

                # ⚠️ DATA LEAKAGE WARNING: Check for suspiciously high accuracy in OOS
                n_samples_oos = len(y_true_dir)
                suspicious_oos = False
                warning_level_oos = "high"

                # Use tiered thresholds for long-term predictions
                if fb == 1:
                    threshold_oos = 0.90
                    if acc > threshold_oos:
                        suspicious_oos = True
                        warning_level_oos = "high"
                elif fb <= 5:
                    threshold_oos = 0.85
                    if acc > threshold_oos:
                        suspicious_oos = True
                        warning_level_oos = "high"
                elif fb <= 15:
                    threshold_oos = 0.80
                    if acc > threshold_oos:
                        suspicious_oos = True
                        warning_level_oos = "high"
                else:
                    # For fb>15, use tiered approach
                    if acc > 0.85:
                        suspicious_oos = True
                        warning_level_oos = "high"
                        threshold_oos = 0.85
                    elif acc > 0.75:
                        suspicious_oos = True
                        warning_level_oos = "medium"
                        threshold_oos = 0.75
                    else:
                        suspicious_oos = False

                if n_samples_oos < 200 and acc > 0.85:
                    suspicious_oos = True
                    warning_level_oos = "high"

                if suspicious_oos:
                    if warning_level_oos == "high":
                        print("\n" + "=" * 70)
                        print(
                            "🚨 严重警告：样本外测试中检测到可能的数据泄露或异常表现！"
                        )
                        print("=" * 70)
                        print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                        print(f"   OOS样本数量: {n_samples_oos}")
                        print(
                            f"   方向准确率: {acc*100:.2f}% (阈值: {threshold_oos*100:.0f}%)"
                        )
                        print(
                            f"   即使在样本外测试中，预测未来 {fb} 根 {freq} K线方向准确率 {acc*100:.2f}% 也是极其罕见的！"
                        )
                        print(
                            f"   这强烈暗示存在数据泄露、特征包含未来信息或市场处于极端单边行情！"
                        )
                        print(f"\n   建议检查：")
                        print(
                            f"   - 确认标签定义：future_return = close[t+fb] / close[t] - 1（使用收盘价，非最高/最低价）"
                        )
                        print(
                            f"   - 确认特征工程：所有特征都使用 shift(1) 避免未来信息"
                        )
                        print(
                            f"   - 检查数据resample是否正确（{freq}下应使用正确的聚合数据）"
                        )
                        print(
                            f"   - 检查OOS时间段是否与训练期市场状态相似（如都是单边上涨）"
                        )
                        print(
                            f"   - 如果样本数少（{n_samples_oos}条），考虑使用更长的时间跨度"
                        )
                        print("=" * 70 + "\n")
                    elif warning_level_oos == "medium":
                        print("\n" + "=" * 70)
                        print("⚠️  提示：OOS测试中检测到较高的准确率（可能需要关注）")
                        print("=" * 70)
                        print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                        print(f"   OOS样本数量: {n_samples_oos}")
                        print(
                            f"   方向准确率: {acc*100:.2f}% (阈值: {threshold_oos*100:.0f}%)"
                        )
                        print(
                            f"   OOS测试中预测未来 {fb} 根 {freq} K线方向准确率 {acc*100:.2f}% 较高"
                        )
                        print(f"\n   💡 对于长期预测（fb={fb}），在趋势明显的市场中：")
                        print(f"      - 75-85%的准确率可能是合理的（趋势跟踪策略）")
                        print(f"      - 但如果OOS准确率显著高于训练期，需要警惕")
                        print(f"   ✅ 建议验证：")
                        print(
                            f"      - 检查OOS时间段是否与训练期市场状态相似（如都是单边上涨）"
                        )
                        print(
                            f"      - 如果OOS准确率显著下降，说明只是拟合了训练期趋势"
                        )
                        print("=" * 70 + "\n")

                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_dir, y_pred_dir, average="binary", zero_division=0
                )
                prec = float(prec)
                rec = float(rec)
                f1 = float(f1)

                # Check for label imbalance and model "giving up" behavior in OOS
                positive_ratio_oos = float(np.mean(y_true_dir))
                negative_ratio_oos = 1.0 - positive_ratio_oos

                # Check for extreme recall values
                if rec == 1.0 or rec == 0.0:
                    print("\n" + "=" * 70)
                    print('⚠️  警告：OOS测试中检测到模型可能"放弃预测"！')
                    print("=" * 70)
                    print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                    print(
                        f"   Recall: {rec:.4f} ({'完美' if rec == 1.0 else '完全失败'})"
                    )
                    print(f"   Precision: {prec:.4f}")
                    print(f"   Accuracy: {acc:.4f}")
                    print(
                        f"   OOS标签分布: 正类={positive_ratio_oos:.1%}, 负类={negative_ratio_oos:.1%}"
                    )
                    if (
                        rec == 1.0
                        and abs(prec - acc) < 0.01
                        and abs(positive_ratio_oos - acc) < 0.01
                    ):
                        print(f"   ⚠️  模型把所有OOS样本都预测为正类！")
                        print(
                            f"   💡 这强烈暗示模型在训练期内拟合了单边上涨行情，在OOS中失效"
                        )
                    print("=" * 70 + "\n")

                # Calculate AUC - but if all predictions are <= 0, AUC may be misleading
                # AUC can be high even if all predictions are negative (good ranking, but all wrong side)
                try:
                    # Check if we have both classes in true labels
                    if len(np.unique(y_true_dir)) < 2:
                        # Only one class (all 0 or all 1) - AUC undefined
                        auc = float("nan")
                    else:
                        auc = float(roc_auc_score(y_true_dir, y_score))
                        # ⚠️ If all predictions are <= 0 but AUC is high, this is suspicious
                        if pred_positive_ratio == 0.0 and auc > 0.7:
                            print(
                                f"   ⚠️  警告：所有预测 ≤ 0 但 AUC={auc:.4f} 很高，这可能是误导性的！"
                            )
                            print(
                                f"      AUC 基于预测值排序，即使所有预测为负，如果排序正确仍可能很高"
                            )
                            print(
                                f"      但实际预测方向全错（F1=0），模型无法用于交易！"
                            )
                except Exception:
                    auc = float("nan")

                # Calculate balanced accuracy for OOS
                from sklearn.metrics import (
                    balanced_accuracy_score as balanced_acc_score_oos,
                )

                balanced_acc_oos = float(balanced_acc_score_oos(y_true_dir, y_pred_dir))
                try:
                    pr_auc = float(average_precision_score(y_true_dir, y_score))
                except Exception:
                    pr_auc = float("nan")
                # Calculate IC (Information Coefficient) for OOS
                try:
                    ic_spearman_oos, _ = spearmanr(
                        y_ret_oos, y_score, nan_policy="omit"
                    )
                    ic_spearman_oos = (
                        float(ic_spearman_oos)
                        if not np.isnan(ic_spearman_oos)
                        else None
                    )
                except Exception:
                    ic_spearman_oos = None
                try:
                    ic_pearson_oos, _ = pearsonr(y_ret_oos, y_score)
                    ic_pearson_oos = (
                        float(ic_pearson_oos) if not np.isnan(ic_pearson_oos) else None
                    )
                except Exception:
                    ic_pearson_oos = None

                # 🔍 DIAGNOSTIC: Add prediction statistics for debugging F1=0 but high AUC
                pred_stats = {
                    "pred_positive_ratio": pred_positive_ratio,
                    "pred_min": y_score_min,
                    "pred_max": y_score_max,
                    "pred_mean": y_score_mean,
                    "pred_median": y_score_median,
                }

                # Quality check: F1=0 but high AUC is a critical issue
                # ⚠️ CRITICAL: F1=0 means model cannot be used for trading, even if AUC is high
                quality_issues = []
                if f1 == 0.0 and (auc is not None and not np.isnan(auc) and auc > 0.7):
                    quality_issues.append(
                        "F1=0 but AUC>0.7: 所有预测≤0但排序正确，模型无法用于交易"
                    )
                if pred_positive_ratio == 0.0:
                    quality_issues.append("所有预测值≤0: 模型过于保守，从未预测'涨'")

                # Quality check: F1=0 should fail even if AUC is high
                quality_passed_directional = bool(f1 > 0.0 and f1 >= 0.3) or (
                    auc is not None and not np.isnan(auc) and auc >= 0.6
                )
                # But if F1=0, quality should fail regardless of AUC
                if f1 == 0.0:
                    quality_passed_directional = False

                # 🔒 CRITICAL: Calculate per-symbol metrics for multi-asset training
                # This allows us to understand model performance on each asset separately
                per_symbol_metrics = {}
                if "symbol" in oos_df.columns and len(oos_df["symbol"].unique()) > 1:
                    print(f"\n   📊 按标的计算 OOS 指标（Per-Symbol OOS Metrics）:")
                    for symbol in oos_df["symbol"].unique():
                        symbol_mask = oos_df["symbol"] == symbol
                        symbol_y_true = y_ret_oos[symbol_mask]
                        symbol_y_pred = y_pred_q50[symbol_mask]
                        symbol_y_true_dir = y_true_dir[symbol_mask]
                        symbol_y_pred_dir = y_pred_dir[symbol_mask]

                        if len(symbol_y_true) > 0:
                            # Regression metrics
                            symbol_rmse = float(
                                np.sqrt(
                                    mean_squared_error(symbol_y_true, symbol_y_pred)
                                )
                            )
                            symbol_mae = float(
                                mean_absolute_error(symbol_y_true, symbol_y_pred)
                            )

                            # Directional metrics
                            symbol_acc = float(
                                accuracy_score(symbol_y_true_dir, symbol_y_pred_dir)
                            )
                            symbol_prec, symbol_rec, symbol_f1, _ = (
                                precision_recall_fscore_support(
                                    symbol_y_true_dir,
                                    symbol_y_pred_dir,
                                    average="binary",
                                    zero_division=0,
                                )
                            )

                            # IC metrics
                            try:
                                symbol_ic_spearman, _ = spearmanr(
                                    symbol_y_true, symbol_y_pred
                                )
                                symbol_ic_spearman = (
                                    float(symbol_ic_spearman)
                                    if not np.isnan(symbol_ic_spearman)
                                    else None
                                )
                            except Exception:
                                symbol_ic_spearman = None

                            try:
                                symbol_ic_pearson, _ = pearsonr(
                                    symbol_y_true, symbol_y_pred
                                )
                                symbol_ic_pearson = (
                                    float(symbol_ic_pearson)
                                    if not np.isnan(symbol_ic_pearson)
                                    else None
                                )
                            except Exception:
                                symbol_ic_pearson = None

                            # Calculate R² for regression models
                            from sklearn.metrics import r2_score

                            symbol_r2 = (
                                float(r2_score(symbol_y_true, symbol_y_pred))
                                if len(symbol_y_true) > 1
                                else None
                            )

                            # Simple Sharpe-like metric (using predicted returns)
                            # Note: This is a simplified metric, not a true Sharpe ratio
                            if len(symbol_y_pred) > 1 and np.std(symbol_y_pred) > 0:
                                symbol_sharpe_like = float(
                                    np.mean(symbol_y_pred)
                                    / (np.std(symbol_y_pred) + 1e-8)
                                )
                            else:
                                symbol_sharpe_like = None

                            per_symbol_metrics[symbol] = {
                                "rmse": symbol_rmse,
                                "mae": symbol_mae,
                                "r2": symbol_r2,
                                "accuracy": symbol_acc,
                                "precision": float(symbol_prec),
                                "recall": float(symbol_rec),
                                "f1": float(symbol_f1),
                                "ic_spearman": symbol_ic_spearman,
                                "ic_pearson": symbol_ic_pearson,
                                "sharpe_like": symbol_sharpe_like,
                                "samples": int(len(symbol_y_true)),
                            }

                            # Format values for printing
                            rmse_str = (
                                f"{symbol_rmse:.6f}"
                                if symbol_rmse is not None
                                else "N/A"
                            )
                            mae_str = (
                                f"{symbol_mae:.6f}" if symbol_mae is not None else "N/A"
                            )
                            r2_str = (
                                f"{symbol_r2:.4f}" if symbol_r2 is not None else "N/A"
                            )
                            ic_str = (
                                f"{symbol_ic_spearman:.4f}"
                                if symbol_ic_spearman is not None
                                else "N/A"
                            )

                            print(
                                f"      {symbol}: RMSE={rmse_str}, MAE={mae_str}, "
                                f"R²={r2_str}, "
                                f"Acc={symbol_acc:.4f}, F1={symbol_f1:.4f}, "
                                f"IC={ic_str}, "
                                f"样本数={len(symbol_y_true)}"
                            )

                oos_metrics = {
                    "directional_oos": {
                        "accuracy": acc,
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                        "auc": auc,
                        "pr_auc": pr_auc,
                        "balanced_accuracy": balanced_acc_oos,
                        "positive_ratio": positive_ratio_oos,
                        "negative_ratio": negative_ratio_oos,
                        "ic_spearman": ic_spearman_oos,
                        "ic_pearson": ic_pearson_oos,
                        "samples": int(len(y_true_dir)),
                        "best_threshold": float(threshold_oos_opt),
                        "threshold_method": args.direction_threshold,
                        "pred_stats": pred_stats,  # 🔍 Diagnostic info
                        "quality_check": {
                            "passed": quality_passed_directional,  # 🔍 Fixed: F1=0 should fail
                            "issues": quality_issues,  # 🔍 Quality issues
                        },
                    },
                    "regression_return": {
                        "rmse": oos_rmse,
                        "mae": oos_mae,
                        "r2": oos_r2,
                        "samples": len(oos_df),
                    },
                    "uncertainty": {
                        "coverage_10_90": coverage,
                        "avg_interval_width": width,
                        "avg_confidence": conf,
                    },
                    "regression_volatility": {
                        "rmse": oos_vol_rmse,
                        "mae": oos_vol_mae,
                        "samples": len(oos_df),
                    },
                }

                # Add per-symbol metrics if available
                if per_symbol_metrics:
                    oos_metrics["per_symbol"] = per_symbol_metrics
            else:
                # In-sample directional metrics for visibility when no OOS period
                from sklearn.metrics import (
                    accuracy_score,
                    precision_recall_fscore_support,
                    roc_auc_score,
                    average_precision_score,
                )

                X_all = train_df[feature_cols].values
                y_ret_all = train_df["future_return"].values

                # Use appropriate model based on model type
                if args.model_type == "classification":
                    # For classification, use probability predictions (predict returns probabilities for binary)
                    y_score_all = model_classification.model.predict(X_all)
                    y_true_dir_all = (y_ret_all > 0).astype(int)
                    # For classification, use fixed threshold of 0.5
                    threshold_train = 0.5
                    y_pred_dir_all = (y_score_all > threshold_train).astype(int)
                else:
                    # For quantile, use Q50 predictions
                    y_score_all = model_q50.model.predict(X_all)
                    y_true_dir_all = (y_ret_all > 0).astype(int)
                    # Use dynamic threshold instead of fixed 0
                    threshold_train = _compute_direction_threshold(
                        y_score_all, y_true_dir_all, method=args.direction_threshold
                    )
                    y_pred_dir_all = (y_score_all > threshold_train).astype(int)
                acc = float(accuracy_score(y_true_dir_all, y_pred_dir_all))
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_dir_all, y_pred_dir_all, average="binary", zero_division=0
                )
                try:
                    auc = float(roc_auc_score(y_true_dir_all, y_score_all))
                except Exception:
                    auc = float("nan")
                try:
                    pr_auc = float(average_precision_score(y_true_dir_all, y_score_all))
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
                    "best_threshold": float(threshold_train),
                    "threshold_method": args.direction_threshold,
                }

            # Directional metrics - CV metrics
            # Import metrics that may not be available from earlier imports
            from sklearn.metrics import (
                precision_score,
                recall_score,
                confusion_matrix,
            )

            # Import scipy.stats functions (already imported above for OOS, but need here for CV)
            from scipy.stats import spearmanr as spearmanr_cv, pearsonr as pearsonr_cv

            # Use appropriate model based on model type
            if args.model_type == "classification":
                # For classification, use probability predictions (predict returns probabilities for binary)
                y_score = model_classification.model.predict(X_df.values)
                y_true_dir = (y_return.values > 0).astype(int)
                # For classification, use fixed threshold of 0.5
                threshold_cv_opt = 0.5
                y_pred_dir = (y_score > threshold_cv_opt).astype(int)
            else:
                # For quantile, use Q50 predictions
                y_score = model_q50.model.predict(X_df.values)
                y_true_dir = (y_return.values > 0).astype(int)
                # Use dynamic threshold instead of fixed 0 (recommended solution)
                # This addresses the issue where all predictions <= 0 lead to F1=0 even with high AUC
                threshold_cv_opt = _compute_direction_threshold(
                    y_score, y_true_dir, method=args.direction_threshold
                )
                y_pred_dir = (y_score > threshold_cv_opt).astype(int)

            # 🔍 DIAGNOSTIC: Calculate prediction statistics first
            y_score_min_cv = float(np.min(y_score))
            y_score_max_cv = float(np.max(y_score))
            y_score_mean_cv = float(np.mean(y_score))
            y_score_median_cv = float(np.median(y_score))

            # Log threshold information
            if threshold_cv_opt != 0.0:
                print(f"\n   📊 方向预测阈值优化（CV）:")
                print(f"      方法: {args.direction_threshold}")
                print(f"      最佳阈值: {threshold_cv_opt:.6f} (固定阈值0: 0.0)")
                print(f"      预测值范围: [{y_score_min_cv:.6f}, {y_score_max_cv:.6f}]")
                print(f"      预测值中位数: {y_score_median_cv:.6f}")

            # 🔍 DIAGNOSTIC: Check for F1=0 but high AUC anomaly (CV metrics)
            pred_positive_ratio_cv = float(np.mean(y_pred_dir))
            true_positive_ratio_cv = float(np.mean(y_true_dir))

            # Check for anomaly: F1=0 but potential high AUC
            if pred_positive_ratio_cv == 0.0:
                print("\n" + "=" * 70)
                print("⚠️  警告（CV）：检测到方向预测异常（所有预测值 ≤ 0）！")
                print("=" * 70)
                print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                print(f"   q50_pred 统计:")
                print(f"     min: {y_score_min_cv:.6f}, max: {y_score_max_cv:.6f}")
                print(
                    f"     mean: {y_score_mean_cv:.6f}, median: {y_score_median_cv:.6f}"
                )
                print(f"     预测为'涨'的比例: {pred_positive_ratio_cv:.1%}")
                print(f"     真实'涨'的比例: {true_positive_ratio_cv:.1%}")
                print(
                    f"   ⚠️  模型从未预测'涨'，F1 将为 0，但 AUC 可能仍很高（如果预测值排序正确）"
                )
                print(f"   💡 这通常意味着模型预测值整体偏负，可能需要检查：")
                print(f"      - 预处理是否导致预测值偏移（如 AR(1) 残差的均值偏移）")
                print(f"      - 是否需要调整预测阈值（使用分位数而非 0）")
                print("=" * 70 + "\n")

            try:
                # Check if we have both classes in true labels
                if len(np.unique(y_true_dir)) < 2:
                    # Only one class (all 0 or all 1) - AUC undefined
                    auc = None
                else:
                    auc = float(roc_auc_score(y_true_dir, y_score))
                    # ⚠️ If all predictions are <= 0 but AUC is high, this is suspicious
                    if pred_positive_ratio_cv == 0.0 and auc > 0.7:
                        print(
                            f"   ⚠️  警告（CV）：所有预测 ≤ 0 但 AUC={auc:.4f} 很高，这可能是误导性的！"
                        )
                        print(
                            f"      AUC 基于预测值排序，即使所有预测为负，如果排序正确仍可能很高"
                        )
                        print(f"      但实际预测方向全错（F1=0），模型无法用于交易！")
            except Exception:
                auc = None
            try:
                pr_auc = float(average_precision_score(y_true_dir, y_score))
            except Exception:
                pr_auc = None
            cm = confusion_matrix(y_true_dir, y_pred_dir).tolist()

            # Calculate IC (Information Coefficient) - Spearman correlation
            try:
                ic_spearman, _ = spearmanr_cv(
                    y_return.values, y_score, nan_policy="omit"
                )
                ic_spearman = float(ic_spearman) if not np.isnan(ic_spearman) else None
            except Exception:
                ic_spearman = None
            try:
                ic_pearson, _ = pearsonr_cv(y_return.values, y_score)
                ic_pearson = float(ic_pearson) if not np.isnan(ic_pearson) else None
            except Exception:
                ic_pearson = None

            # Calculate IR (Information Ratio) = IC_mean / IC_std
            # For single fold, IR is undefined, but we can calculate it across multiple predictions
            # For now, we'll use IC as a proxy for IR (IR = IC / std(IC) over time)
            # In practice, IR is calculated over rolling windows of IC values
            ir_spearman = None
            ir_pearson = None
            if ic_spearman is not None:
                # For single evaluation, IR is approximately IC (assuming unit variance)
                # In production, IR should be calculated over rolling windows
                ir_spearman = ic_spearman  # Simplified: IR ≈ IC for single evaluation
            if ic_pearson is not None:
                ir_pearson = ic_pearson  # Simplified: IR ≈ IC for single evaluation

            # Calculate long/short separate metrics
            # Long: predict up (y_pred_dir == 1), Short: predict down (y_pred_dir == 0)
            long_mask = y_pred_dir == 1
            short_mask = y_pred_dir == 0
            long_metrics = {}
            short_metrics = {}

            if np.sum(long_mask) > 0:
                # Long position metrics
                y_true_long = y_true_dir[long_mask]
                y_pred_long = y_pred_dir[long_mask]
                y_score_long = y_score[long_mask]
                y_return_long = y_return.values[long_mask]

                long_accuracy = float(accuracy_score(y_true_long, y_pred_long))
                long_precision = float(
                    precision_score(y_true_long, y_pred_long, zero_division=0)
                )
                long_recall = float(
                    recall_score(y_true_long, y_pred_long, zero_division=0)
                )
                long_f1 = float(f1_score(y_true_long, y_pred_long, zero_division=0))

                try:
                    if len(np.unique(y_true_long)) >= 2:
                        long_auc = float(roc_auc_score(y_true_long, y_score_long))
                    else:
                        long_auc = None
                except Exception:
                    long_auc = None

                try:
                    long_pr_auc = float(
                        average_precision_score(y_true_long, y_score_long)
                    )
                except Exception:
                    long_pr_auc = None

                # IC for long positions
                try:
                    long_ic_spearman, _ = spearmanr_cv(
                        y_return_long, y_score_long, nan_policy="omit"
                    )
                    long_ic_spearman = (
                        float(long_ic_spearman)
                        if not np.isnan(long_ic_spearman)
                        else None
                    )
                except Exception:
                    long_ic_spearman = None

                try:
                    long_ic_pearson, _ = pearsonr_cv(y_return_long, y_score_long)
                    long_ic_pearson = (
                        float(long_ic_pearson)
                        if not np.isnan(long_ic_pearson)
                        else None
                    )
                except Exception:
                    long_ic_pearson = None

                long_metrics = {
                    "accuracy": long_accuracy,
                    "precision": long_precision,
                    "recall": long_recall,
                    "f1": long_f1,
                    "auc": long_auc,
                    "pr_auc": long_pr_auc,
                    "ic_spearman": long_ic_spearman,
                    "ic_pearson": long_ic_pearson,
                    "samples": int(np.sum(long_mask)),
                }

            if np.sum(short_mask) > 0:
                # Short position metrics (inverted: predict down, but we want to check if actual return < 0)
                # For short positions, we predict down (y_pred_dir == 0), and we want actual return < 0
                y_true_short = (y_return.values[short_mask] < 0).astype(
                    int
                )  # Actual down
                y_pred_short = (y_pred_dir[short_mask] == 0).astype(
                    int
                )  # Predicted down
                y_score_short = (
                    1.0 - y_score[short_mask]
                )  # Invert score for short (higher score = more confident in down)
                y_return_short = y_return.values[short_mask]

                short_accuracy = float(accuracy_score(y_true_short, y_pred_short))
                short_precision = float(
                    precision_score(y_true_short, y_pred_short, zero_division=0)
                )
                short_recall = float(
                    recall_score(y_true_short, y_pred_short, zero_division=0)
                )
                short_f1 = float(f1_score(y_true_short, y_pred_short, zero_division=0))

                try:
                    if len(np.unique(y_true_short)) >= 2:
                        short_auc = float(roc_auc_score(y_true_short, y_score_short))
                    else:
                        short_auc = None
                except Exception:
                    short_auc = None

                try:
                    short_pr_auc = float(
                        average_precision_score(y_true_short, y_score_short)
                    )
                except Exception:
                    short_pr_auc = None

                # IC for short positions (using negative returns)
                try:
                    short_ic_spearman, _ = spearmanr_cv(
                        -y_return_short, y_score_short, nan_policy="omit"
                    )
                    short_ic_spearman = (
                        float(short_ic_spearman)
                        if not np.isnan(short_ic_spearman)
                        else None
                    )
                except Exception:
                    short_ic_spearman = None

                try:
                    short_ic_pearson, _ = pearsonr_cv(-y_return_short, y_score_short)
                    short_ic_pearson = (
                        float(short_ic_pearson)
                        if not np.isnan(short_ic_pearson)
                        else None
                    )
                except Exception:
                    short_ic_pearson = None

                short_metrics = {
                    "accuracy": short_accuracy,
                    "precision": short_precision,
                    "recall": short_recall,
                    "f1": short_f1,
                    "auc": short_auc,
                    "pr_auc": short_pr_auc,
                    "ic_spearman": short_ic_spearman,
                    "ic_pearson": short_ic_pearson,
                    "samples": int(np.sum(short_mask)),
                }

            # Get feature importance for all trained models
            feature_importance = _extract_feature_importance_df(
                model_classification, feature_cols
            )
            return_feature_importance = _extract_feature_importance_df(
                model_return, feature_cols
            )
            vol_feature_importance = _extract_feature_importance_df(
                model_vol, feature_cols
            )

            if classification_metrics is not None and feature_importance is not None:
                classification_metrics["feature_importance"] = (
                    feature_importance.to_dict("records")
                )
            if return_metrics is not None and return_feature_importance is not None:
                return_metrics["feature_importance"] = (
                    return_feature_importance.to_dict("records")
                )
            if vol_metrics is not None and vol_feature_importance is not None:
                vol_metrics["feature_importance"] = vol_feature_importance.to_dict(
                    "records"
                )
            accuracy = float(accuracy_score(y_true_dir, y_pred_dir))

            # Get sample count for warning context
            n_samples = len(y_true_dir)

            # ⚠️ DATA LEAKAGE WARNING: Check for suspiciously high accuracy
            # For different fb values, use different thresholds
            # Note: Long-term predictions (fb>15) in trending markets can achieve higher accuracy
            # We need to balance between detecting data leakage and recognizing legitimate trend-following signals
            suspicious = False
            threshold = 0.90  # Default threshold
            warning_level = "high"  # "high", "medium", or None

            if fb == 1:
                threshold = 0.90  # fb=1: >90% is very suspicious
                if accuracy > threshold:
                    suspicious = True
                    warning_level = "high"
            elif fb <= 5:
                threshold = 0.85  # fb=2-5: >85% is suspicious
                if accuracy > threshold:
                    suspicious = True
                    warning_level = "high"
            elif fb <= 15:
                threshold = 0.80  # fb=6-15: >80% is suspicious
                if accuracy > threshold:
                    suspicious = True
                    warning_level = "high"
            else:
                # For larger fb (e.g., fb=45), high accuracy can be suspicious but also legitimate
                # In trending markets (e.g., 2025 Q1 BTC bull run), predicting long-term direction
                # can achieve 75-85% accuracy without data leakage
                if accuracy > 0.85:
                    suspicious = True
                    warning_level = "high"
                    threshold = 0.85
                elif accuracy > 0.75:
                    suspicious = True
                    warning_level = "medium"
                    threshold = 0.75
                else:
                    suspicious = False

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
                if warning_level == "high":
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
                    print(
                        f"   2. 标签泄露（label leakage）- 确认使用收盘价而非最高/最低价"
                    )
                    print(f"   3. 数据时间顺序错误")
                    print(f"   4. 数据预处理错误（如未正确shift或resample）")
                    print(f"   5. 样本太少（{n_samples}条）导致过拟合特定市场阶段")
                    print(f"   6. 市场处于极端单边行情（如2025 Q1 BTC单边上涨）")
                    print(f"\n   建议检查：")
                    print(
                        f"   - 确认标签定义：future_return = close[t+fb] / close[t] - 1（使用收盘价）"
                    )
                    print(f"   - 确认特征工程：所有特征都使用 shift(1) 避免未来信息")
                    print(
                        f"   - 检查数据resample是否正确（{freq}下应使用正确的聚合数据）"
                    )
                    print(f"   - 增加样本数量或使用更长的时间跨度")
                    print("=" * 70 + "\n")
                elif warning_level == "medium":
                    # For long-term predictions (fb>15), 75-85% accuracy can be legitimate in trending markets
                    # Check if this is likely due to market trend rather than data leakage
                    print("\n" + "=" * 70)
                    print("⚠️  提示：检测到较高的准确率（可能需要关注）")
                    print("=" * 70)
                    print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                    print(f"   样本数量: {n_samples}")
                    print(
                        f"   方向准确率: {accuracy*100:.2f}% (阈值: {threshold*100:.0f}%)"
                    )
                    print(
                        f"   预测未来 {fb} 根 {freq} K线的方向，准确率 {accuracy*100:.2f}% 较高"
                    )
                    print(f"\n   💡 对于长期预测（fb={fb}），在趋势明显的市场中：")
                    print(f"      - 75-85%的准确率可能是合理的（趋势跟踪策略）")
                    print(f"      - 但如果同时出现以下情况，需要警惕：")
                    print(f"        1. IC (Spearman) > 0.5 或接近1.0")
                    print(f"        2. 标签严重不平衡（正类占比 > 70%）")
                    print(f"        3. Recall = 1.0（模型全预测为正类）")
                    print(f"        4. 样本数过少（< 1000）且准确率 > 80%")
                    ic_info = (
                        f"IC (Spearman) = {ic_spearman:.4f}"
                        if ic_spearman is not None
                        else "IC (Spearman) = N/A"
                    )
                    print(f"\n   当前IC值: {ic_info}")
                    print(f"   ✅ 建议验证：")
                    print(
                        f"      - 检查IC值：如果IC < 0.3，说明预测能力有限，高准确率可能来自市场趋势"
                    )
                    print(
                        f"      - 检查OOS测试：如果OOS准确率显著下降，说明只是拟合了训练期趋势"
                    )
                    print(
                        f"      - 检查标签分布：如果正类占比接近准确率，模型可能只是预测多数类"
                    )
                    print("=" * 70 + "\n")

            precision = float(precision_score(y_true_dir, y_pred_dir, zero_division=0))
            recall = float(recall_score(y_true_dir, y_pred_dir, zero_division=0))
            f1 = float(f1_score(y_true_dir, y_pred_dir, zero_division=0))

            # Check for label imbalance and model "giving up" behavior
            # Calculate label distribution
            positive_ratio = float(np.mean(y_true_dir))
            negative_ratio = 1.0 - positive_ratio

            # Check for extreme recall values (Recall=1.0 or 0.0)
            model_gave_up = False
            if recall == 1.0 or recall == 0.0:
                model_gave_up = True
                print("\n" + "=" * 70)
                print('⚠️  警告：检测到模型可能"放弃预测"！')
                print("=" * 70)
                print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                print(
                    f"   Recall: {recall:.4f} ({'完美' if recall == 1.0 else '完全失败'})"
                )
                print(f"   Precision: {precision:.4f}")
                print(f"   Accuracy: {accuracy:.4f}")
                print(
                    f"   标签分布: 正类={positive_ratio:.1%}, 负类={negative_ratio:.1%}"
                )

                if recall == 1.0:
                    print(
                        f"\n   📊 分析：Recall=1.0 意味着所有真实为正类的样本都被预测为正类"
                    )
                    if (
                        abs(precision - accuracy) < 0.01
                        and abs(positive_ratio - accuracy) < 0.01
                    ):
                        print(
                            f"   ⚠️  发现：Precision ≈ Accuracy ≈ 正类比例，说明模型把所有样本都预测为正类！"
                        )
                        print(f'   💡 这意味着模型"放弃预测"，只是简单地预测"上涨"')
                        print(f"   🔍 可能原因：")
                        print(f"      1. 训练期内市场处于单边上涨（如2025 Q1 BTC牛市）")
                        print(
                            f"      2. 标签不平衡（正类占比过高：{positive_ratio:.1%}）"
                        )
                        print(
                            f"      3. 模型无法学习有效的预测信号，选择最安全的策略（全预测正类）"
                        )
                        print(f"   ⚠️  建议：")
                        print(f"      - 这不是模型能力，而是市场状态偏差")
                        print(f"      - 使用balanced accuracy或F1作为主要指标")
                        print(
                            f"      - 检查OOS测试，如果准确率跌至50%左右，说明只是拟合了牛市"
                        )
                        print(f"      - 考虑使用更长的时间跨度训练，包含熊市和震荡市")
                    else:
                        print(f"   ℹ️  这可能是因为正样本很少，模型容易全对")
                elif recall == 0.0:
                    print(
                        f"\n   📊 分析：Recall=0.0 意味着所有真实为正类的样本都被预测为负类"
                    )
                    print(f"   ⚠️  模型可能把所有样本都预测为负类（下跌）")
                    print(f"   💡 可能原因：市场处于单边下跌或模型完全失效")
                print("=" * 70 + "\n")

            # Check for severe label imbalance
            if positive_ratio > 0.7 or positive_ratio < 0.3:
                print("\n" + "=" * 70)
                print("⚠️  警告：检测到严重的标签不平衡！")
                print("=" * 70)
                print(f"   Timeframe: {freq}, Forward Bars: {fb}")
                print(
                    f"   标签分布: 正类={positive_ratio:.1%}, 负类={negative_ratio:.1%}"
                )
                print(f"   Accuracy: {accuracy:.4f}")
                if positive_ratio > 0.7:
                    print(f"   ⚠️  正类占比过高（>70%），模型可能倾向于预测正类")
                    print(f"   💡 建议：")
                    print(f"      - 使用balanced accuracy或F1作为主要指标")
                    print(f"      - 考虑使用class_weight='balanced'训练")
                    print(f"      - 检查是否是市场处于单边上涨导致的")
                elif positive_ratio < 0.3:
                    print(f"   ⚠️  负类占比过高（>70%），模型可能倾向于预测负类")
                    print(f"   💡 建议：")
                    print(f"      - 使用balanced accuracy或F1作为主要指标")
                    print(f"      - 考虑使用class_weight='balanced'训练")
                    print(f"      - 检查是否是市场处于单边下跌导致的")
                print("=" * 70 + "\n")

            # Calculate balanced accuracy
            from sklearn.metrics import balanced_accuracy_score as balanced_acc_score

            balanced_acc = float(balanced_acc_score(y_true_dir, y_pred_dir))

            # 🔍 DIAGNOSTIC: Add prediction statistics for debugging F1=0 but high AUC
            pred_stats_cv = {
                "pred_positive_ratio": pred_positive_ratio_cv,
                "pred_min": y_score_min_cv,
                "pred_max": y_score_max_cv,
                "pred_mean": y_score_mean_cv,
                "pred_median": y_score_median_cv,
            }

            # Quality check: F1=0 but high AUC is a critical issue
            quality_issues_cv = []
            if f1 == 0.0 and (auc is not None and auc > 0.7):
                quality_issues_cv.append(
                    "F1=0 but AUC>0.7: 所有预测≤0但排序正确，模型无法用于交易"
                )
            if pred_positive_ratio_cv == 0.0:
                quality_issues_cv.append("所有预测值≤0: 模型过于保守，从未预测'涨'")

            directional_metrics_cv = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "pr_auc": pr_auc,
                "ic_spearman": ic_spearman,
                "ic_pearson": ic_pearson,
                "ir_spearman": ir_spearman,
                "ir_pearson": ir_pearson,
                "pred_stats": pred_stats_cv,  # 🔍 Diagnostic info
                "balanced_accuracy": balanced_acc,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "best_threshold": float(threshold_cv_opt),
                "threshold_method": args.direction_threshold,
                "samples": int(len(y_true_dir)),
                "confusion_matrix": cm,
                "long_metrics": long_metrics if long_metrics else None,
                "short_metrics": short_metrics if short_metrics else None,
                "feature_importance": (
                    feature_importance.to_dict("records")
                    if feature_importance is not None
                    else None
                ),
            }

            # Save artifacts and report (neutral naming, no 'baseline')
            # Use timestamped base directory for this training run to avoid mixing old data
            combo_dir = base_results_dir
            models_combo_dir = base_models_dir
            if len(freqs) > 1 or len(fbs) > 1:
                # If multiple configs, create subdirectory for each config
                combo_dir = os.path.join(base_results_dir, f"fb{fb}_tf{freq}")
                models_combo_dir = os.path.join(base_models_dir, f"fb{fb}_tf{freq}")
            else:
                combo_dir = base_results_dir
                models_combo_dir = base_models_dir
            os.makedirs(combo_dir, exist_ok=True)
            os.makedirs(models_combo_dir, exist_ok=True)

            # ✅ 保存一体化 Pipeline（使用 joblib）
            # 参考文档：docs/工作流："预处理 + 模型 + 后处理"一体化保存与部署.md
            # 使用统一的 Pipeline 保存方式，简化代码逻辑

            # 保存分类模型 Pipeline
            if args.model_type == "classification":
                if classification_preprocess_params:
                    classification_pipeline = TradingModelPipeline(
                        model_type="classification",
                        forward_bars=fb,
                        feature_cols=feature_cols,
                        preprocess_params=classification_preprocess_params,
                        use_gpu=args.gpu,
                    )
                    classification_pipeline.model = model_classification.model
                    classification_pipeline.preprocessor = RobustWinsorizer.from_params(
                        classification_preprocess_params, forward_bars=fb
                    )
                    classification_pipeline.save(
                        os.path.join(combo_dir, "classification_pipeline.pkl")
                    )
                    print(
                        f"  ✅ Saved Classification pipeline to {os.path.join(combo_dir, 'classification_pipeline.pkl')}"
                    )

                # 保存 Return Regression 模型 Pipeline
                if return_preprocess_params:
                    return_pipeline = TradingModelPipeline(
                        model_type="regression",
                        forward_bars=fb,
                        feature_cols=feature_cols,
                        preprocess_params=return_preprocess_params,
                        use_gpu=args.gpu,
                        target_transform="log1p_abs",
                    )
                    return_pipeline.model = model_return.model
                    return_pipeline.preprocessor = RobustWinsorizer.from_params(
                        return_preprocess_params, forward_bars=fb
                    )
                    return_pipeline.save(os.path.join(combo_dir, "return_pipeline.pkl"))
                    print(
                        f"  ✅ Saved Return Regression pipeline to {os.path.join(combo_dir, 'return_pipeline.pkl')}"
                    )
                else:
                    # Return regression model without preprocessing
                    return_pipeline = TradingModelPipeline(
                        model_type="regression",
                        forward_bars=fb,
                        feature_cols=feature_cols,
                        preprocess_params=None,
                        use_gpu=args.gpu,
                        target_transform="log1p_abs",
                    )
                    return_pipeline.model = model_return.model
                    return_pipeline.save(os.path.join(combo_dir, "return_pipeline.pkl"))
                    print(
                        f"  ✅ Saved Return Regression pipeline to {os.path.join(combo_dir, 'return_pipeline.pkl')}"
                    )

            # 保存 Q50 模型 Pipeline
            if q50_preprocess_params:
                q50_pipeline = TradingModelPipeline(
                    model_type="quantile",
                    quantile_alpha=0.5,
                    forward_bars=fb,
                    feature_cols=feature_cols,
                    preprocess_params=q50_preprocess_params,
                    use_gpu=args.gpu,
                )
                q50_pipeline.model = model_q50.model
                q50_pipeline.preprocessor = RobustWinsorizer.from_params(
                    q50_preprocess_params, forward_bars=fb
                )
                q50_pipeline.save(os.path.join(combo_dir, "q50_pipeline.pkl"))
                print(
                    f"  ✅ Saved Q50 pipeline to {os.path.join(combo_dir, 'q50_pipeline.pkl')}"
                )

            # 保存 Q10 模型 Pipeline
            if q10_preprocess_params:
                q10_pipeline = TradingModelPipeline(
                    model_type="quantile",
                    quantile_alpha=0.1,
                    forward_bars=fb,
                    feature_cols=feature_cols,
                    preprocess_params=q10_preprocess_params,
                    use_gpu=args.gpu,
                )
                q10_pipeline.model = model_q10.model
                q10_pipeline.preprocessor = RobustWinsorizer.from_params(
                    q10_preprocess_params, forward_bars=fb
                )
                q10_pipeline.save(os.path.join(combo_dir, "q10_pipeline.pkl"))
                print(
                    f"  ✅ Saved Q10 pipeline to {os.path.join(combo_dir, 'q10_pipeline.pkl')}"
                )

            # 保存 Q90 模型 Pipeline
            if q90_preprocess_params:
                q90_pipeline = TradingModelPipeline(
                    model_type="quantile",
                    quantile_alpha=0.9,
                    forward_bars=fb,
                    feature_cols=feature_cols,
                    preprocess_params=q90_preprocess_params,
                    use_gpu=args.gpu,
                )
                q90_pipeline.model = model_q90.model
                q90_pipeline.preprocessor = RobustWinsorizer.from_params(
                    q90_preprocess_params, forward_bars=fb
                )
                q90_pipeline.save(os.path.join(combo_dir, "q90_pipeline.pkl"))
                print(
                    f"  ✅ Saved Q90 pipeline to {os.path.join(combo_dir, 'q90_pipeline.pkl')}"
                )

            # 保存 Volatility 模型 Pipeline（不需要预处理参数）
            vol_pipeline = TradingModelPipeline(
                model_type="regression",
                forward_bars=fb,
                feature_cols=feature_cols,
                preprocess_params=None,  # Volatility 模型不需要预处理
                use_gpu=args.gpu,
            )
            vol_pipeline.model = model_vol.model
            vol_pipeline.save(os.path.join(combo_dir, "vol_pipeline.pkl"))
            print(
                f"  ✅ Saved Volatility pipeline to {os.path.join(combo_dir, 'vol_pipeline.pkl')}"
            )

            scaler_path = os.path.join(combo_dir, "scalers.pkl")
            if args.feature_type == "baseline":
                if feature_engineer is not None:
                    feature_engineer.save_scalers(scaler_path)
            else:
                if feature_engineer is not None and hasattr(
                    feature_engineer, "save_scalers"
                ):
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
                        finalized_path = os.path.join("results/training", finalized_dir)
                        finalized_models_path = os.path.join("models", finalized_dir)
                        if finalized_path != base_results_dir:
                            os.makedirs(os.path.dirname(finalized_path), exist_ok=True)
                            # Rename the base directory (moves all existing files/subdirs)
                            os.rename(base_results_dir, finalized_path)
                            base_results_dir = finalized_path
                            print(
                                f"📁 Renamed results directory to: {base_results_dir}"
                            )
                        if finalized_models_path != base_models_dir:
                            os.makedirs(
                                os.path.dirname(finalized_models_path), exist_ok=True
                            )
                            os.rename(base_models_dir, finalized_models_path)
                            base_models_dir = finalized_models_path
                            print(f"📂 Renamed model directory to: {base_models_dir}")
                        base_dir_finalized = True
                        # Update combo_dir to finalized base for subsequent saves in this loop iteration
                        if len(freqs) > 1 or len(fbs) > 1:
                            combo_dir = os.path.join(
                                base_results_dir, f"fb{fb}_tf{freq}"
                            )
                            models_combo_dir = os.path.join(
                                base_models_dir, f"fb{fb}_tf{freq}"
                            )
                except Exception as _e:
                    print(f"Note: Could not finalize results directory name: {_e}")

            # Recompute output paths after potential rename and ensure directory exists
            if len(freqs) > 1 or len(fbs) > 1:
                combo_dir = os.path.join(base_results_dir, f"fb{fb}_tf{freq}")
                models_combo_dir = os.path.join(base_models_dir, f"fb{fb}_tf{freq}")
            else:
                combo_dir = base_results_dir
                models_combo_dir = base_models_dir
            os.makedirs(combo_dir, exist_ok=True)
            os.makedirs(models_combo_dir, exist_ok=True)

            # Compose model_info and write training_info.json with finalized paths
            info_path = os.path.join(combo_dir, "training_info.json")
            # Extract OOS time range if available
            oos_start_str = None
            oos_end_str = None
            if len(oos_df) > 0 and not oos_df.empty:
                oos_start_str = (
                    oos_df.index.min().isoformat()
                    if oos_df.index.min() is not None
                    else None
                )
                oos_end_str = (
                    oos_df.index.max().isoformat()
                    if oos_df.index.max() is not None
                    else None
                )
            elif oos_start_dt is not None and oos_end_dt is not None:
                oos_start_str = oos_start_dt.isoformat()
                oos_end_str = oos_end_dt.isoformat()

            # Model usability information
            model_usability_info = {
                "usable": (
                    quantile_model_usable
                    if "quantile_model_usable" in locals()
                    else True
                ),
                "q50_loss_ratio": (
                    q50_loss_ratio if "q50_loss_ratio" in locals() else 1.0
                ),
                "q50_loss": q50_loss if "q50_loss" in locals() else 0.0,
                "q10_loss": q10_loss if "q10_loss" in locals() else 0.0,
                "q90_loss": q90_loss if "q90_loss" in locals() else 0.0,
                "reason": (
                    "Q50 loss > Q10/Q90 loss (violates quantile regression property)"
                    if not (
                        quantile_model_usable
                        if "quantile_model_usable" in locals()
                        else True
                    )
                    else None
                ),
                "retrain_attempted": (
                    retrain_attempted if "retrain_attempted" in locals() else False
                ),
                "retrain_successful": (
                    retrain_attempted if "retrain_attempted" in locals() else False
                )
                and (
                    quantile_model_usable
                    if "quantile_model_usable" in locals()
                    else True
                ),
                "calibration_params": (
                    calibration_params
                    if "calibration_params" in locals()
                    else {"enabled": False}
                ),
                "remediation_status": (
                    remediation_status if "remediation_status" in locals() else None
                ),
            }

            model_info = {
                "model_paths": {
                    "q50": (
                        os.path.join(combo_dir, "q50_pipeline.pkl")
                        if model_q50
                        else None
                    ),
                    "q10": (
                        os.path.join(combo_dir, "q10_pipeline.pkl")
                        if model_q10
                        else None
                    ),
                    "q90": (
                        os.path.join(combo_dir, "q90_pipeline.pkl")
                        if model_q90
                        else None
                    ),
                    "classification": (
                        os.path.join(combo_dir, "classification_pipeline.pkl")
                        if model_classification
                        else None
                    ),
                    "volatility": os.path.join(combo_dir, "vol_pipeline.pkl"),
                },
                "model_type": args.model_type,
                "scaler_path": os.path.join(combo_dir, "scalers.pkl"),
                "training_date": _dt.now().isoformat(),
                "symbol": symbols_str,
                "actual_start": (
                    feat_df.index.min().isoformat() if not feat_df.empty else None
                ),
                "actual_end": (
                    feat_df.index.max().isoformat() if not feat_df.empty else None
                ),
                "train_start": (
                    train_df.index.min().isoformat() if not train_df.empty else None
                ),
                "train_end": (
                    train_df.index.max().isoformat() if not train_df.empty else None
                ),
                "oos_start": oos_start_str,
                "oos_end": oos_end_str,
                "total_bars": len(feat_df),
                "train_bars": len(train_df),
                "oos_bars": len(oos_df) if len(oos_df) > 0 else 0,
                "oos_months": args.oos_months if len(oos_df) > 0 else 0,
                "timeframes": {freq: len(feat_df)},
                "price_range": [
                    float(feat_df["close"].min()) if not feat_df.empty else 0,
                    float(feat_df["close"].max()) if not feat_df.empty else 0,
                ],
                "metrics": {
                    "stage2": {freq: q50_metrics} if q50_metrics else {},
                    "q10": {freq: q10_metrics} if q10_metrics else {},
                    "q90": {freq: q90_metrics} if q90_metrics else {},
                    "classification": (
                        {freq: classification_metrics} if classification_metrics else {}
                    ),
                    "return": (
                        {freq: return_metrics}
                        if (args.model_type == "classification" and return_metrics)
                        else {}
                    ),
                    "volatility": {freq: vol_metrics},
                    "directional_train": (
                        {freq: directional_metrics_train}
                        if directional_metrics_train
                        else {}
                    ),
                    "directional_cv": (
                        {freq: directional_metrics_cv} if directional_metrics_cv else {}
                    ),
                },
                "feature_engineering": (
                    "BaselineFeatureEngineer"
                    if args.feature_type == "baseline"
                    else f"ComprehensiveFeatureEngineer({args.feature_type})"
                ),
                "feature_type": args.feature_type,
                "forward_bars": fb,
                "timeframe": freq,
                "data_files": files,
                "ar1_info": {
                    # AR(1) statistics are now computed per CV fold, not globally
                    # We store first fold stats for deployment consistency
                    "note": "AR(1) preprocessing moved to CV loop - statistics computed per fold. Deployment params from first fold.",
                    "ar1_phi": (
                        q50_preprocess_params.get("ar1", {}).get("ar1_phi")
                        if q50_preprocess_params
                        else (
                            classification_preprocess_params.get("ar1", {}).get(
                                "ar1_phi"
                            )
                            if classification_preprocess_params
                            else None
                        )
                    ),
                    "ar1_autocorr_before": None,  # Not available in current implementation
                    "ar1_autocorr_after": (
                        q50_preprocess_params.get("ar1", {}).get("ar1_autocorr_after")
                        if q50_preprocess_params
                        else (
                            classification_preprocess_params.get("ar1", {}).get(
                                "ar1_autocorr_after"
                            )
                            if classification_preprocess_params
                            else None
                        )
                    ),
                    "continuity_bias_removed": True,  # Always applied in CV loop
                },
                # Preprocessing parameters for deployment (from first CV fold)
                # These parameters MUST be used for consistent preprocessing in production
                # See: RobustWinsorizer class in preprocessing.py for usage
                "preprocess_params": (
                    q50_preprocess_params
                    if q50_preprocess_params
                    else (
                        classification_preprocess_params
                        if classification_preprocess_params
                        else {
                            "note": "Preprocessing parameters not available (preprocessing not applied or failed)",
                        }
                    )
                ),
                "model_usability": model_usability_info,
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

                # Extract metrics based on model type
                model_type = info_json.get("model_type", "quantile")
                if model_type == "classification":
                    classification_metrics = info_json.get("metrics", {}).get(
                        "classification", {}
                    )
                    return_metrics = info_json.get("metrics", {}).get("return", {})
                    q10_metrics = {}
                    q50_metrics = {}
                    q90_metrics = {}
                else:
                    q10_metrics = info_json.get("metrics", {}).get("q10", {})
                    q50_metrics = info_json.get("metrics", {}).get("stage2", {})
                    q90_metrics = info_json.get("metrics", {}).get("q90", {})
                    classification_metrics = {}
                    return_metrics = {}
                vol_metrics = info_json.get("metrics", {}).get("volatility", {})

                def _format_feature_table(records, heading):
                    if not records or not isinstance(records, list):
                        return ""
                    total = 0.0
                    values = []
                    for rec in records:
                        if not isinstance(rec, dict):
                            continue
                        try:
                            val = float(rec.get("importance", 0.0))
                        except (TypeError, ValueError):
                            val = 0.0
                        values.append((rec.get("feature", "N/A"), val))
                        total += max(val, 0.0)
                    if not values:
                        return ""
                    total = total if total > 0 else 1.0
                    rows = []
                    for feat_name, val in values[:20]:
                        pct = val / total * 100.0
                        rows.append(
                            f"<tr><td>{feat_name}</td><td>{val:.2f}</td><td>{pct:.2f}%</td></tr>"
                        )
                    return "".join(
                        [
                            f"<h3>{heading}</h3>",
                            "<table><tr><th>特征名称</th><th>重要性 (Gain)</th><th>占比 (%)</th></tr>",
                            "".join(rows),
                            "</table>",
                        ]
                    )

                # Format values with 6 decimal places, but show scientific notation for very small values
                def fmt_val(v):
                    if v == "N/A" or v is None:
                        return "N/A"
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

                # Quantile Loss section (q10, q50, q90) - only for quantile models
                quantile_rows = []
                if q50_metrics and len(q50_metrics) > 0:
                    for tf in q50_metrics.keys():
                        q10_val = q10_metrics.get(tf, {}).get("cv_quantile_loss", "N/A")
                        q50_val = q50_metrics.get(tf, {}).get("cv_quantile_loss", "N/A")
                        q90_val = q90_metrics.get(tf, {}).get("cv_quantile_loss", "N/A")

                        quantile_rows.append(
                            f"<tr><td>{tf}</td><td>{fmt_val(q10_val)}</td><td>{fmt_val(q50_val)}</td><td>{fmt_val(q90_val)}</td></tr>"
                        )

                # AR(1) information section (for fb=1)
                # NOTE: AR(1) preprocessing is now done in CV loop, so global statistics are not available
                ar1_section = ""
                if ar1_info:
                    ar1_note = ar1_info.get("note")
                    ar1_phi = ar1_info.get("ar1_phi")
                    ar1_autocorr_before = ar1_info.get("ar1_autocorr_before")
                    ar1_autocorr_after = ar1_info.get("ar1_autocorr_after")
                    removed = ar1_info.get("continuity_bias_removed", False)

                    ar1_rows = []
                    # Display note about CV loop preprocessing
                    if ar1_note:
                        ar1_rows.append(
                            f"<tr><td colspan='2' style='background-color:#e7f3ff; padding:8px;'><strong>ℹ️ {ar1_note}</strong></td></tr>"
                        )
                    if ar1_phi is not None:
                        ar1_rows.append(
                            f"<tr><td>AR(1) 系数 (φ)</td><td>{ar1_phi:.4f}</td></tr>"
                        )
                    else:
                        ar1_rows.append(
                            f"<tr><td>AR(1) 系数 (φ)</td><td>N/A (computed per CV fold)</td></tr>"
                        )
                    if ar1_autocorr_before is not None:
                        ar1_rows.append(
                            f"<tr><td>Lag-1 自相关系数（移除前）</td><td>{ar1_autocorr_before:.4f}</td></tr>"
                        )
                    else:
                        ar1_rows.append(
                            f"<tr><td>Lag-1 自相关系数（移除前）</td><td>N/A (computed per CV fold)</td></tr>"
                        )
                    if ar1_autocorr_after is not None:
                        ar1_rows.append(
                            f"<tr><td>Lag-1 自相关系数（移除后）</td><td>{ar1_autocorr_after:.4f}</td></tr>"
                        )
                        if ar1_autocorr_before is not None:
                            reduction = abs(ar1_autocorr_before) - abs(
                                ar1_autocorr_after
                            )
                            reduction_pct = (
                                (reduction / abs(ar1_autocorr_before) * 100)
                                if abs(ar1_autocorr_before) > 0
                                else 0
                            )
                            ar1_rows.append(
                                f"<tr><td>自相关性减少</td><td>{reduction:.4f} ({reduction_pct:.1f}%)</td></tr>"
                            )
                    else:
                        ar1_rows.append(
                            f"<tr><td>Lag-1 自相关系数（移除后）</td><td>N/A (computed per CV fold)</td></tr>"
                        )
                    ar1_rows.append(
                        f"<tr><td>连续性偏差已移除</td><td>{'✅ 是' if removed else '❌ 否'}</td></tr>"
                    )

                    # Warning message (simplified since we don't have global stats)
                    if removed:
                        ar1_warning = f"<div style='background-color:#d4edda;border-left:4px solid #28a745;padding:10px;margin:10px 0;border-radius:4px;'><strong>✅ AR(1)预处理已应用:</strong> 价格连续性偏差已在CV循环中移除（每折独立计算AR(1)系数）。这确保了没有lookahead bias，所有统计量仅从训练集计算。</div>"
                    else:
                        ar1_warning = ""

                    ar1_explanation = "<p><em>AR(1) 信息说明：</em></p><ul style='margin:10px 0;padding-left:20px;'><li><strong>AR(1) 系数 (φ)</strong>：衡量价格连续性的强度，值越高表示相邻K线价格越相关</li><li><strong>Lag-1 自相关系数（移除前/后）</strong>：收益率序列的一阶自相关性，用于估计AR(1)模型参数。移除后自相关性应显著降低</li><li><strong>连续性偏差</strong>：高IC/准确率可能来自价格连续性而非真实预测能力。使用AR(1)残差可以移除这部分偏差</li><li><strong>自相关性减少</strong>：移除AR(1)成分后自相关性的减少幅度，用于验证去连续性处理的有效性</li></ul>"

                    # Get current fb value - it should be available in the current scope
                    # Since we're in a loop over fb values, use the fb variable directly
                    current_fb = fb if "fb" in locals() else "N/A"
                    ar1_section = (
                        f"<h2>🔍 AR(1) 价格连续性分析 (fb={current_fb})</h2>"
                        + ar1_explanation
                        + ar1_warning
                        + "<table><tr><th>指标</th><th>值</th></tr>"
                        + "".join(ar1_rows)
                        + "</table>"
                    )

                quantile_section = ""
                if quantile_rows:
                    # Check for anomalies
                    warnings_list = []
                    if q50_metrics and len(q50_metrics) > 0:
                        for tf in q50_metrics.keys():
                            q10_val = q10_metrics.get(tf, {}).get(
                                "cv_quantile_loss", "N/A"
                            )
                            q50_val = q50_metrics.get(tf, {}).get(
                                "cv_quantile_loss", "N/A"
                            )
                            q90_val = q90_metrics.get(tf, {}).get(
                                "cv_quantile_loss", "N/A"
                            )
                            try:
                                q10_f = float(q10_val) if q10_val != "N/A" else None
                                q50_f = float(q50_val) if q50_val != "N/A" else None
                                q90_f = float(q90_val) if q90_val != "N/A" else None
                                if (
                                    q10_f is not None
                                    and q50_f is not None
                                    and q90_f is not None
                                ):
                                    # Check for Q50 loss = 0 or suspiciously small (likely calculation error or data issue)
                                    if q50_f == 0.0 or (q50_f < 1e-6 and q10_f > 1e-6):
                                        warnings_list.append(
                                            f"⚠️ {tf}: Q50 loss ({q50_f:.6f}) 异常小或为0！这可能是计算错误、数据问题或模型预测完全正确（不太可能）。请检查数据或计算逻辑。"
                                        )
                                    # Check if Q50 loss violates quantile regression property (should be <= Q10 and Q90)
                                    elif q50_f > q10_f or q50_f > q90_f:
                                        ratio_q10 = (
                                            q50_f / q10_f if q10_f > 0 else float("inf")
                                        )
                                        ratio_q90 = (
                                            q50_f / q90_f if q90_f > 0 else float("inf")
                                        )
                                        warnings_list.append(
                                            f"⚠️ {tf}: Q50 loss ({q50_f:.6f}) > Q10/Q90 loss（Q10={q10_f:.6f}, Q90={q90_f:.6f}），违反quantile regression性质！"
                                            f" Q50是Q10的{ratio_q10:.1f}倍，是Q90的{ratio_q90:.1f}倍。"
                                            f" 可能原因：1) Q50模型训练质量差（过拟合/欠拟合）；2) 数据有异常值影响中位数预测；"
                                            f" 3) AR(1)处理引入了异常值；4) LightGBM quantile回归训练不稳定。"
                                            f" 建议：检查训练日志、Q50预测值分布、AR(1)处理后的数据异常值。"
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
                        warning_html = (
                            "<div style='background-color:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;border-radius:4px;'><strong>⚠️ 警告:</strong><ul style='margin:5px 0;padding-left:20px;'>"
                            + "".join([f"<li>{w}</li>" for w in warnings_list])
                            + "</ul></div>"
                        )

                    quantile_section = (
                        "<h2>📊 Quantile Loss (CV)</h2>"
                        "<p><em>单位: Pinball Loss (与收益率比例单位相同，例如 0.01 表示 1% 的平均误差)</em></p>"
                        "<p><em>注意: 如果loss值>1.0，可能存在单位问题或数据异常。正常情况下，收益率在±5%范围内时，loss应在0.01-0.1量级。Q50 loss应≤Q10/Q90 loss。</em></p>"
                        + warning_html
                        + "<table><tr><th>Timeframe</th><th>Quantile Loss 0.1 (q10)</th><th>Quantile Loss 0.5 (q50)</th><th>Quantile Loss 0.9 (q90)</th></tr>"
                        + "".join(quantile_rows)
                        + "</table>"
                    )

                # Return Regression CV Metrics section (for classification mode)
                return_rows = []
                return_section = ""
                if model_type == "classification" and return_metrics:
                    for tf, m in return_metrics.items():
                        cv_rmse = m.get("cv_rmse", "N/A")
                        cv_mse = m.get("cv_mse", "N/A")
                        cv_mae = m.get("cv_mae", "N/A")
                        cv_r2 = m.get("cv_r2", "N/A")

                        def fmt_r2(v):
                            if v == "N/A" or v is None:
                                return "N/A"
                            try:
                                return f"{float(v):.4f}"
                            except (ValueError, TypeError):
                                return str(v)

                        return_rows.append(
                            f"<tr><td>{tf}</td><td>{fmt_val(cv_rmse)}</td><td>{fmt_val(cv_mse)}</td><td>{fmt_val(cv_mae)}</td><td>{fmt_r2(cv_r2)}</td></tr>"
                        )

                    if return_rows:
                        return_section = (
                            "<h2>📊 Return Regression CV Metrics</h2>"
                            "<p><em>单位: RMSE/MSE/MAE - 收益率比例 (例如 0.01 表示 1%)，R² - 决定系数 (范围 [0, 1]，越高越好)</em></p>"
                            "<p><em>使用说明：</em></p>"
                            "<ul style='margin:10px 0;padding-left:20px;'>"
                            "<li><strong>RMSE (Root Mean Squared Error)</strong>：均方根误差，衡量预测收益率的平均误差大小。值越小越好，通常应在0.01-0.1量级（对应1%-10%的收益率误差）</li>"
                            "<li><strong>MSE (Mean Squared Error)</strong>：均方误差，RMSE的平方。对大误差更敏感，用于评估模型对极端值的预测能力</li>"
                            "<li><strong>MAE (Mean Absolute Error)</strong>：平均绝对误差，衡量预测收益率的平均绝对偏差。相比RMSE，MAE对大误差的惩罚较小</li>"
                            "<li><strong>R² (R-squared)</strong>：决定系数，衡量模型解释目标变量变异的比例。R²=1表示完美预测，R²=0表示模型不优于均值预测，R²<0表示模型预测比均值更差。在金融预测中，R²>0.01通常表示模型有一定预测能力</li>"
                            "<li><strong>使用场景</strong>：收益回归模型用于：1) 预测收益幅度（与分类模型配合，分类模型预测方向，回归模型预测幅度）；2) 计算信号强度（signal_strength = return_pred / vol_pred）；3) 仓位管理（根据预测收益幅度调整仓位大小）</li>"
                            "</ul>"
                            "<p><em>注意: 如果RMSE值>1.0，可能存在单位问题。正常情况下，收益率在±5%范围内时，RMSE应在0.01-0.1量级。</em></p>"
                            "<table><tr><th>Timeframe</th><th>CV RMSE</th><th>CV MSE</th><th>CV MAE</th><th>CV R²</th></tr>"
                            + "".join(return_rows)
                            + "</table>"
                        )

                # Volatility CV Metrics section
                vol_rows = []
                for tf, m in vol_metrics.items():
                    cv_rmse = m.get("cv_rmse", "N/A")
                    cv_mse = m.get("cv_mse", "N/A")

                    def fmt_val(v):
                        if v == "N/A" or v is None:
                            return "N/A"
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
                        cv_rmse = m.get("cv_rmse", "N/A")
                        try:
                            rmse_f = float(cv_rmse) if cv_rmse != "N/A" else None
                            if rmse_f is not None and rmse_f > 1.0:
                                vol_warnings_list.append(
                                    f"⚠️ {tf}: CV RMSE ({rmse_f:.2f})异常大！如果波动率是比例（0.01=1%），RMSE应在0.01-0.1量级，可能存在单位问题。"
                                )
                        except (ValueError, TypeError):
                            pass

                    vol_warning_html = ""
                    if vol_warnings_list:
                        vol_warning_html = (
                            "<div style='background-color:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;border-radius:4px;'><strong>⚠️ 警告:</strong><ul style='margin:5px 0;padding-left:20px;'>"
                            + "".join([f"<li>{w}</li>" for w in vol_warnings_list])
                            + "</ul></div>"
                        )

                    vol_section = (
                        "<h2>📈 Volatility (Regression) CV Metrics</h2>"
                        "<p><em>单位: RMSE/MSE - 波动率比例 (与收益率比例单位相同，例如 0.01 表示 1%)</em></p>"
                        "<p><em>使用说明：</em></p>"
                        "<ul style='margin:10px 0;padding-left:20px;'>"
                        "<li><strong>RMSE (Root Mean Squared Error)</strong>：均方根误差，衡量预测波动率的平均误差大小。值越小越好，通常应在0.01-0.1量级（对应1%-10%的波动率误差）</li>"
                        "<li><strong>MSE (Mean Squared Error)</strong>：均方误差，RMSE的平方。对大误差更敏感，用于评估模型对极端波动率的预测能力</li>"
                        "<li><strong>使用场景</strong>：波动率模型用于：1) 风险调整收益（signal_strength = return / vol）；2) 仓位管理（根据预测波动率调整仓位大小）；3) 止损设置（高波动率时设置更宽的止损）</li>"
                        "</ul>"
                        "<p><em>注意: 如果RMSE值>1.0，可能存在单位问题。正常情况下，波动率在1-5%范围内时，RMSE应在0.01-0.1量级。</em></p>"
                        + vol_warning_html
                        + "<table><tr><th>Timeframe</th><th>CV RMSE</th><th>CV MSE</th></tr>"
                        + "".join(vol_rows)
                        + "</table>"
                    )

                # Directional and IC metrics (CV)
                directional_cv_metrics = info_json.get("metrics", {}).get(
                    "directional_cv", {}
                )
                directional_section = ""
                if directional_cv_metrics:
                    dir_rows = []
                    long_short_rows_all = []
                    feature_importance_all = None

                    for tf, dir_metrics in directional_cv_metrics.items():
                        if isinstance(dir_metrics, dict):

                            def fmt_pct(v):
                                if v == "N/A" or v is None:
                                    return "N/A"
                                try:
                                    return f"{float(v)*100:.2f}%"
                                except (ValueError, TypeError):
                                    return str(v)

                            def fmt_corr(v):
                                if v == "N/A" or v is None:
                                    return "N/A"
                                try:
                                    return f"{float(v):.4f}"
                                except (ValueError, TypeError):
                                    return str(v)

                            acc = fmt_pct(dir_metrics.get("accuracy"))
                            prec = fmt_pct(dir_metrics.get("precision"))
                            rec = fmt_pct(dir_metrics.get("recall"))
                            f1 = fmt_pct(dir_metrics.get("f1"))
                            auc_val = (
                                fmt_pct(dir_metrics.get("auc"))
                                if dir_metrics.get("auc") is not None
                                else "N/A"
                            )
                            pr_auc_val = (
                                fmt_pct(dir_metrics.get("pr_auc"))
                                if dir_metrics.get("pr_auc") is not None
                                else "N/A"
                            )
                            ic_spearman = fmt_corr(dir_metrics.get("ic_spearman"))
                            ic_pearson = fmt_corr(dir_metrics.get("ic_pearson"))
                            ir_spearman = (
                                fmt_corr(dir_metrics.get("ir_spearman"))
                                if dir_metrics.get("ir_spearman") is not None
                                else "N/A"
                            )
                            ir_pearson = (
                                fmt_corr(dir_metrics.get("ir_pearson"))
                                if dir_metrics.get("ir_pearson") is not None
                                else "N/A"
                            )
                            dir_rows.append(
                                f"<tr><td>{tf}</td><td>{acc}</td><td>{prec}</td><td>{rec}</td><td>{f1}</td><td>{auc_val}</td><td>{pr_auc_val}</td><td>{ic_spearman}</td><td>{ic_pearson}</td><td>{ir_spearman}</td><td>{ir_pearson}</td></tr>"
                            )

                            # Collect long/short separate metrics (use first timeframe's data)
                            if not long_short_rows_all:
                                long_metrics = dir_metrics.get("long_metrics")
                                short_metrics = dir_metrics.get("short_metrics")
                                if long_metrics or short_metrics:
                                    if long_metrics:
                                        long_acc = fmt_pct(long_metrics.get("accuracy"))
                                        long_prec = fmt_pct(
                                            long_metrics.get("precision")
                                        )
                                        long_rec = fmt_pct(long_metrics.get("recall"))
                                        long_f1 = fmt_pct(long_metrics.get("f1"))
                                        long_auc = (
                                            fmt_pct(long_metrics.get("auc"))
                                            if long_metrics.get("auc") is not None
                                            else "N/A"
                                        )
                                        long_pr_auc = (
                                            fmt_pct(long_metrics.get("pr_auc"))
                                            if long_metrics.get("pr_auc") is not None
                                            else "N/A"
                                        )
                                        long_ic_spearman = (
                                            fmt_corr(long_metrics.get("ic_spearman"))
                                            if long_metrics.get("ic_spearman")
                                            is not None
                                            else "N/A"
                                        )
                                        long_ic_pearson = (
                                            fmt_corr(long_metrics.get("ic_pearson"))
                                            if long_metrics.get("ic_pearson")
                                            is not None
                                            else "N/A"
                                        )
                                        long_samples = long_metrics.get(
                                            "samples", "N/A"
                                        )
                                        long_short_rows_all.append(
                                            f"<tr><td>做多 (Long)</td><td>{long_acc}</td><td>{long_prec}</td><td>{long_rec}</td><td>{long_f1}</td><td>{long_auc}</td><td>{long_pr_auc}</td><td>{long_ic_spearman}</td><td>{long_ic_pearson}</td><td>{long_samples}</td></tr>"
                                        )
                                    if short_metrics:
                                        short_acc = fmt_pct(
                                            short_metrics.get("accuracy")
                                        )
                                        short_prec = fmt_pct(
                                            short_metrics.get("precision")
                                        )
                                        short_rec = fmt_pct(short_metrics.get("recall"))
                                        short_f1 = fmt_pct(short_metrics.get("f1"))
                                        short_auc = (
                                            fmt_pct(short_metrics.get("auc"))
                                            if short_metrics.get("auc") is not None
                                            else "N/A"
                                        )
                                        short_pr_auc = (
                                            fmt_pct(short_metrics.get("pr_auc"))
                                            if short_metrics.get("pr_auc") is not None
                                            else "N/A"
                                        )
                                        short_ic_spearman = (
                                            fmt_corr(short_metrics.get("ic_spearman"))
                                            if short_metrics.get("ic_spearman")
                                            is not None
                                            else "N/A"
                                        )
                                        short_ic_pearson = (
                                            fmt_corr(short_metrics.get("ic_pearson"))
                                            if short_metrics.get("ic_pearson")
                                            is not None
                                            else "N/A"
                                        )
                                        short_samples = short_metrics.get(
                                            "samples", "N/A"
                                        )
                                        long_short_rows_all.append(
                                            f"<tr><td>做空 (Short)</td><td>{short_acc}</td><td>{short_prec}</td><td>{short_rec}</td><td>{short_f1}</td><td>{short_auc}</td><td>{short_pr_auc}</td><td>{short_ic_spearman}</td><td>{short_ic_pearson}</td><td>{short_samples}</td></tr>"
                                        )

                            # Collect feature importance (use first timeframe's data)
                            if feature_importance_all is None:
                                feature_importance_all = dir_metrics.get(
                                    "feature_importance"
                                )

                    if dir_rows:
                        directional_section = "".join(
                            [
                                "<h2>🎯 方向预测指标 (CV)</h2>",
                                "<p><em>方向准确率: 预测涨跌方向的准确率 | IC: 信息系数 (Information Coefficient)，预测值与实际值的相关性</em></p>",
                                "<p><em>使用说明：</em></p>",
                                "<ul style='margin:10px 0;padding-left:20px;'>",
                                "<li><strong>方向准确率 (Accuracy)</strong>：预测涨跌方向的正确率。在金融预测中，准确率>50%表示模型有预测能力，>55%表示较强的预测能力</li>",
                                "<li><strong>精确率 (Precision)</strong>：预测为'涨'的样本中，实际为'涨'的比例。高精确率表示模型预测'涨'时可信度高</li>",
                                "<li><strong>召回率 (Recall)</strong>：实际为'涨'的样本中，被正确预测为'涨'的比例。高召回率表示模型能捕捉到大部分上涨机会</li>",
                                "<li><strong>F1 Score</strong>：精确率和召回率的调和平均数，平衡两者。F1>0.5表示模型有一定预测能力</li>",
                                "<li><strong>AUC (Area Under ROC Curve)</strong>：ROC曲线下面积，衡量模型区分涨跌的能力。AUC>0.5表示模型优于随机，AUC>0.6表示较强的预测能力</li>",
                                "<li><strong>IC (Spearman/Pearson)</strong>：信息系数，预测值与实际值的秩相关/线性相关。IC>0.05表示模型有预测能力，IC>0.1表示较强的预测能力</li>",
                                "<li><strong>IR (Information Ratio)</strong>：信息比率，IC的均值除以标准差。IR>0.5表示模型有稳定的预测能力，IR>1.0表示较强的预测能力</li>",
                                "<li><strong>PR-AUC (Precision-Recall AUC)</strong>：精确率-召回率曲线下面积，衡量模型在不平衡数据上的表现。PR-AUC>0.5表示模型有预测能力，PR-AUC>0.7表示较强的预测能力</li>",
                                "</ul>",
                                "<table><tr><th>Timeframe</th><th>方向准确率</th><th>精确率</th><th>召回率</th><th>F1</th><th>AUC</th><th>PR-AUC</th><th>IC (Spearman)</th><th>IC (Pearson)</th><th>IR (Spearman)</th><th>IR (Pearson)</th></tr>",
                                "".join(dir_rows),
                                "</table>",
                            ]
                        )

                        # Add long/short separate metrics section
                        if long_short_rows_all:
                            directional_section += "".join(
                                [
                                    "<h3>📈 做多/做空分开指标 (Long/Short Separate Metrics)</h3>",
                                    "<p><em>做多：预测上涨时的性能指标 | 做空：预测下跌时的性能指标</em></p>",
                                    "<table><tr><th>方向</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1</th><th>AUC</th><th>PR-AUC</th><th>IC (Spearman)</th><th>IC (Pearson)</th><th>样本数</th></tr>",
                                    "".join(long_short_rows_all),
                                    "</table>",
                                ]
                            )

                        # Add feature importance section
                        if (
                            feature_importance_all
                            and isinstance(feature_importance_all, list)
                            and len(feature_importance_all) > 0
                        ):
                            # Get top 20 features
                            top_features = feature_importance_all[:20]
                            feature_rows = []
                            for feat in top_features:
                                feat_name = feat.get("feature", "N/A")
                                feat_importance = feat.get("importance", 0)
                                # Normalize importance to percentage
                                total_importance = sum(
                                    f.get("importance", 0)
                                    for f in feature_importance_all
                                )
                                feat_pct = (
                                    (feat_importance / total_importance * 100)
                                    if total_importance > 0
                                    else 0
                                )
                                feature_rows.append(
                                    f"<tr><td>{feat_name}</td><td>{feat_importance:.2f}</td><td>{feat_pct:.2f}%</td></tr>"
                                )

                            if feature_rows:
                                directional_section += "".join(
                                    [
                                        "<h3>🔍 特征重要性 (Top 20 Features)</h3>",
                                        "<p><em>特征重要性：LightGBM gain-based importance，衡量特征对模型预测的贡献</em></p>",
                                        "<table><tr><th>特征名称</th><th>重要性 (Gain)</th><th>占比 (%)</th></tr>",
                                        "".join(feature_rows),
                                        "</table>",
                                    ]
                                )
                # OOS section - regression metrics and directional metrics
                oos_section = ""
                model_usable = True
                model_issues = []
                if info_json.get("oos_metrics"):
                    oos_metrics = info_json["oos_metrics"]
                    oos_rows = []
                    oos_dir_rows = []

                    # Helper function to format with color based on threshold
                    def fmt_pct_with_color(
                        v, good_threshold=0.5, excellent_threshold=0.55
                    ):
                        if v == "N/A" or v is None:
                            return "<td>N/A</td>"
                        try:
                            val = float(v)
                            color = (
                                "green"
                                if val >= excellent_threshold
                                else ("#90EE90" if val >= good_threshold else "red")
                            )
                            return f'<td style="color: {color}; font-weight: bold;">{val*100:.2f}%</td>'
                        except (ValueError, TypeError):
                            return f"<td>{str(v)}</td>"

                    def fmt_corr_with_color(
                        v, good_threshold=0.05, excellent_threshold=0.1
                    ):
                        if v == "N/A" or v is None:
                            return "<td>N/A</td>"
                        try:
                            val = float(v)
                            color = (
                                "green"
                                if abs(val) >= excellent_threshold
                                else (
                                    "#90EE90" if abs(val) >= good_threshold else "red"
                                )
                            )
                            return f'<td style="color: {color}; font-weight: bold;">{val:.4f}</td>'
                        except (ValueError, TypeError):
                            return f"<td>{str(v)}</td>"

                    def fmt_val_with_color(
                        v,
                        is_lower_better=True,
                        good_threshold=None,
                        excellent_threshold=None,
                    ):
                        if v == "N/A" or v is None:
                            return "<td>N/A</td>"
                        try:
                            val = float(v)
                            if good_threshold is None or excellent_threshold is None:
                                return f"<td>{val:.6f}</td>"
                            if is_lower_better:
                                color = (
                                    "green"
                                    if val <= excellent_threshold
                                    else ("#90EE90" if val <= good_threshold else "red")
                                )
                            else:
                                color = (
                                    "green"
                                    if val >= excellent_threshold
                                    else ("#90EE90" if val >= good_threshold else "red")
                                )
                            return f'<td style="color: {color}; font-weight: bold;">{val:.6f}</td>'
                        except (ValueError, TypeError):
                            return f"<td>{str(v)}</td>"

                    # Directional OOS metrics
                    if "directional_oos" in oos_metrics:
                        dir_oos = oos_metrics["directional_oos"]

                        acc_val = dir_oos.get("accuracy")
                        f1_val = dir_oos.get("f1")
                        auc_val = dir_oos.get("auc")
                        ic_spearman_val = dir_oos.get("ic_spearman")

                        # Check model usability based on OOS metrics
                        if acc_val is not None and acc_val < 0.5:
                            model_usable = False
                            model_issues.append(
                                f"OOS准确率 {acc_val*100:.2f}% < 50%（低于随机）"
                            )
                        if f1_val is not None and f1_val < 0.5:
                            model_issues.append(f"OOS F1 {f1_val*100:.2f}% < 50%")
                        if auc_val is not None and auc_val < 0.5:
                            model_issues.append(
                                f"OOS AUC {auc_val*100:.2f}% < 50%（低于随机）"
                            )
                        if ic_spearman_val is not None and abs(ic_spearman_val) < 0.05:
                            model_issues.append(
                                f"OOS IC (Spearman) {ic_spearman_val:.4f} < 0.05（预测能力弱）"
                            )

                        oos_dir_rows.append(
                            f"<tr><td>方向准确率</td>{fmt_pct_with_color(acc_val)}</tr>"
                        )
                        oos_dir_rows.append(
                            f"<tr><td>精确率</td>{fmt_pct_with_color(dir_oos.get('precision'))}</tr>"
                        )
                        oos_dir_rows.append(
                            f"<tr><td>召回率</td>{fmt_pct_with_color(dir_oos.get('recall'))}</tr>"
                        )
                        oos_dir_rows.append(
                            f"<tr><td>F1</td>{fmt_pct_with_color(f1_val)}</tr>"
                        )
                        if dir_oos.get("auc") is not None:
                            oos_dir_rows.append(
                                f"<tr><td>AUC</td>{fmt_pct_with_color(auc_val)}</tr>"
                            )
                        if dir_oos.get("ic_spearman") is not None:
                            oos_dir_rows.append(
                                f"<tr><td>IC (Spearman)</td>{fmt_corr_with_color(ic_spearman_val)}</tr>"
                            )
                        if dir_oos.get("ic_pearson") is not None:
                            oos_dir_rows.append(
                                f"<tr><td>IC (Pearson)</td>{fmt_corr_with_color(dir_oos.get('ic_pearson'))}</tr>"
                            )

                    # Return Regression OOS metrics (for classification mode)
                    def fmt_val(v):
                        if v == "N/A" or v is None:
                            return "N/A"
                        try:
                            return f"{float(v):.6f}"
                        except (ValueError, TypeError):
                            return str(v)

                    # Check for return regression OOS metrics (classification mode)
                    if "regression_return" in oos_metrics:
                        reg_ret = oos_metrics["regression_return"]
                        oos_rows.append(
                            f"<tr><td>Return Regression RMSE</td><td>{fmt_val(reg_ret.get('rmse'))} <em>(收益率比例)</em></td></tr>"
                        )
                        oos_rows.append(
                            f"<tr><td>Return Regression MAE</td><td>{fmt_val(reg_ret.get('mae'))} <em>(收益率比例)</em></td></tr>"
                        )
                        if reg_ret.get("r2") is not None:
                            oos_rows.append(
                                f"<tr><td>Return Regression R²</td><td>{fmt_val(reg_ret.get('r2'))} <em>(决定系数，越高越好)</em></td></tr>"
                            )
                    # Q50 OOS metrics (quantile mode)
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
                            if v == "N/A" or v is None:
                                return "N/A"
                            try:
                                return f"{float(v)*100:.2f}%"
                            except (ValueError, TypeError):
                                return str(v)

                        coverage = unc_oos.get("coverage_10_90") or unc_oos.get(
                            "coverage"
                        )
                        width = unc_oos.get("avg_interval_width") or unc_oos.get(
                            "interval_width"
                        )
                        conf = unc_oos.get("avg_confidence") or unc_oos.get(
                            "confidence"
                        )
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

                    # Per-symbol OOS metrics (for multi-asset training)
                    per_symbol_section = ""
                    if "per_symbol" in oos_metrics:
                        per_symbol_metrics = oos_metrics["per_symbol"]
                        per_symbol_rows = []

                        def fmt_val(v):
                            if v == "N/A" or v is None:
                                return "N/A"
                            try:
                                return f"{float(v):.6f}"
                            except (ValueError, TypeError):
                                return str(v)

                        def fmt_pct(v):
                            if v == "N/A" or v is None:
                                return "N/A"
                            try:
                                return f"{float(v)*100:.2f}%"
                            except (ValueError, TypeError):
                                return str(v)

                        def fmt_corr(v):
                            if v == "N/A" or v is None:
                                return "N/A"
                            try:
                                return f"{float(v):.4f}"
                            except (ValueError, TypeError):
                                return str(v)

                        for symbol, metrics in per_symbol_metrics.items():
                            per_symbol_rows.append(
                                f"<tr><td rowspan='2' style='vertical-align:middle;'><strong>{symbol}</strong></td>"
                                f"<td>RMSE</td><td>{fmt_val(metrics.get('rmse'))}</td>"
                                f"<td>MAE</td><td>{fmt_val(metrics.get('mae'))}</td>"
                                f"<td>R²</td><td>{fmt_val(metrics.get('r2'))}</td></tr>"
                                f"<tr><td>Accuracy</td><td>{fmt_pct(metrics.get('accuracy'))}</td>"
                                f"<td>F1</td><td>{fmt_pct(metrics.get('f1'))}</td>"
                                f"<td>IC (Spearman)</td><td>{fmt_corr(metrics.get('ic_spearman'))}</td></tr>"
                            )

                        if per_symbol_rows:
                            per_symbol_section = (
                                "<h2>📊 按标的 OOS 指标 (Per-Symbol OOS Metrics)</h2>"
                                "<p><em>使用说明：</em></p>"
                                "<ul style='margin:10px 0;padding-left:20px;'>"
                                "<li><strong>按标的分开显示</strong>：多资产训练时，每个标的的模型表现可能不同。此表格按标的分别显示回归指标（RMSE、MAE、R²）和方向预测指标（Accuracy、F1、IC）</li>"
                                "<li><strong>回归指标</strong>：RMSE、MAE、R² 衡量模型预测收益率的准确性。R²>0.01 表示模型有一定预测能力</li>"
                                "<li><strong>方向预测指标</strong>：Accuracy、F1、IC 衡量模型预测涨跌方向的能力。Accuracy>50% 表示模型有预测能力</li>"
                                "<li><strong>使用场景</strong>：如果某个标的的指标明显低于其他标的，可能需要：1) 检查该标的的数据质量；2) 考虑单独训练该标的的模型；3) 检查该标的的市场特征是否与其他标的不同</li>"
                                "<li><strong>如何解读</strong>：比较不同标的的指标，如果某个标的的R²<0或Accuracy<50%，说明该标的的模型预测能力较弱，可能需要单独优化或排除该标的</li>"
                                "</ul>"
                                "<table><tr><th>Symbol</th><th>回归指标</th><th>Value</th><th>回归指标</th><th>Value</th><th>回归指标</th><th>Value</th></tr>"
                                "<tr><th></th><th>方向指标</th><th>Value</th><th>方向指标</th><th>Value</th><th>方向指标</th><th>Value</th></tr>"
                                + "".join(per_symbol_rows)
                                + "</table>"
                            )

                    oos_sections = []
                    if oos_dir_rows:
                        oos_sections.append(
                            "<h2>🎯 OOS 方向预测指标</h2>"
                            "<table><tr><th>Metric</th><th>Value</th></tr>"
                            + "".join(oos_dir_rows)
                            + "</table>"
                        )
                    if oos_rows:
                        oos_sections.append(
                            "<h2>📉 OOS 回归指标</h2>"
                            "<table><tr><th>Metric</th><th>Value</th></tr>"
                            + "".join(oos_rows)
                            + "</table>"
                        )
                    if per_symbol_section:
                        oos_sections.append(per_symbol_section)

                    # Add model usability conclusion
                    if model_issues:
                        status_color = "red" if not model_usable else "orange"
                        status_text = (
                            "❌ 不可用" if not model_usable else "⚠️ 可用但有问题"
                        )
                        # Mark issues in red
                        issues_html = "".join(
                            [
                                f"<li style='color: red; font-weight: bold;'>{issue}</li>"
                                for issue in model_issues
                            ]
                        )
                        oos_sections.append(
                            f"<h2 style='color: {status_color};'>🔍 模型可用性结论 (Model Usability)</h2>"
                            f"<div style='background-color: {'#ffebee' if not model_usable else '#fff3e0'}; padding: 15px; border-left: 4px solid {status_color}; border-radius: 5px; margin: 15px 0;'>"
                            f"<p style='font-size: 18px; font-weight: bold; color: {status_color};'>{status_text}</p>"
                            f"<p><strong>问题列表：</strong></p>"
                            f"<ul style='margin: 10px 0; padding-left: 20px;'>{issues_html}</ul>"
                            f"<p><strong>建议：</strong></p>"
                            f"<ul style='margin: 10px 0; padding-left: 20px;'>"
                            f"<li>如果模型不可用，建议重新训练或检查数据质量</li>"
                            f"<li>如果模型可用但有问题，建议优化特征或调整模型参数</li>"
                            f"<li>检查训练数据是否包含足够的样本和多样性</li>"
                            f"<li>考虑使用更长的时间跨度训练，包含不同的市场状态</li>"
                            f"</ul>"
                            f"</div>"
                        )
                    elif oos_sections:
                        oos_sections.append(
                            "<h2 style='color: green;'>✅ 模型可用性结论 (Model Usability)</h2>"
                            "<div style='background-color: #e8f5e9; padding: 15px; border-left: 4px solid green; border-radius: 5px; margin: 15px 0;'>"
                            "<p style='font-size: 18px; font-weight: bold; color: green;'>✅ 模型可用</p>"
                            "<p>所有OOS指标均达到可接受水平，模型可以用于实盘交易。</p>"
                            "<p><strong>建议：</strong></p>"
                            "<ul style='margin: 10px 0; padding-left: 20px;'>"
                            "<li>继续监控模型在实盘中的表现</li>"
                            "<li>定期重新训练模型以保持预测能力</li>"
                            "<li>关注市场环境变化，必要时调整模型参数</li>"
                            "</ul>"
                            "</div>"
                        )

                    if oos_sections:
                        oos_section = "\n".join(oos_sections)

                # Artifacts section - include all models based on model type
                # Get model directory from model_paths or fallback to old model_path
                model_paths = info_json.get("model_paths", {})
                if model_paths:
                    if model_type == "classification":
                        model_dir = os.path.dirname(
                            model_paths.get("classification", "")
                            or model_paths.get("volatility", "")
                        )
                        artifacts_list = [
                            f"<li><strong>Classification Model Pipeline</strong> ({model_paths.get('classification', 'N/A')}): "
                            f"<em>分类模型，用于预测未来涨跌方向的概率（0-1之间，0.5为阈值）。输出概率可用于计算信号强度。</em></li>",
                            f"<li><strong>Return Regression Model Pipeline</strong> ({model_paths.get('return', 'N/A')}): "
                            f"<em>收益回归模型，用于预测未来收益率的幅度。与分类模型配合使用，分类模型预测方向，回归模型预测幅度。</em></li>",
                            f"<li><strong>Volatility Model Pipeline</strong> ({model_paths.get('volatility', 'N/A')}): "
                            f"<em>波动率模型，用于预测未来波动率。用于风险调整收益计算（signal_strength = return / vol）和仓位管理。</em></li>",
                            f"<li><strong>Scalers</strong> ({info_json.get('scaler_path', 'N/A')}): "
                            f"<em>数据标准化器，用于特征预处理。包含RobustWinsorizer等预处理参数，确保推理时使用与训练时相同的预处理方式。</em></li>",
                        ]
                    else:
                        model_dir = os.path.dirname(model_paths.get("q50", ""))
                        artifacts_list = [
                            f"<li><strong>Q50 Model Pipeline</strong> ({model_paths.get('q50', 'N/A')}): "
                            f"<em>中位数预测模型，用于预测未来收益率的中位数。这是主要的预测模型，输出期望收益率。</em></li>",
                            f"<li><strong>Q10 Model Pipeline</strong> ({model_paths.get('q10', 'N/A')}): "
                            f"<em>10%分位数预测模型，用于构建不确定性区间的下界。与Q90配合使用，构建预测区间。</em></li>",
                            f"<li><strong>Q90 Model Pipeline</strong> ({model_paths.get('q90', 'N/A')}): "
                            f"<em>90%分位数预测模型，用于构建不确定性区间的上界。与Q10配合使用，构建预测区间。</em></li>",
                            f"<li><strong>Volatility Model Pipeline</strong> ({model_paths.get('volatility', 'N/A')}): "
                            f"<em>波动率模型，用于预测未来波动率。用于风险调整收益计算和仓位管理。</em></li>",
                            f"<li><strong>Scalers</strong> ({info_json.get('scaler_path', 'N/A')}): "
                            f"<em>数据标准化器，用于特征预处理。包含RobustWinsorizer等预处理参数，确保推理时使用与训练时相同的预处理方式。</em></li>",
                        ]
                else:
                    # Fallback for old format
                    model_dir = os.path.dirname(info_json.get("model_path", ""))
                    artifacts_list = [
                        f"<li><strong>Models</strong> ({model_dir}): "
                        f"<em>模型文件目录，包含所有训练好的模型文件。</em></li>",
                        f"<li><strong>Scalers</strong> ({info_json.get('scaler_path', 'N/A')}): "
                        f"<em>数据标准化器，用于特征预处理。包含RobustWinsorizer等预处理参数，确保推理时使用与训练时相同的预处理方式。</em></li>",
                    ]

                current_timeframe = info_json.get("timeframe")
                return_feature_section = ""
                if (
                    model_type == "classification"
                    and current_timeframe
                    and isinstance(return_metrics, dict)
                ):
                    return_tf_metrics = return_metrics.get(current_timeframe, {})
                    if isinstance(return_tf_metrics, dict):
                        return_feature_section = _format_feature_table(
                            return_tf_metrics.get("feature_importance"),
                            "🔍 Return Regression Feature Importance (Top 20)",
                        )

                vol_feature_section = ""
                if current_timeframe and isinstance(vol_metrics, dict):
                    vol_tf_metrics = vol_metrics.get(current_timeframe, {})
                    if isinstance(vol_tf_metrics, dict):
                        vol_feature_section = _format_feature_table(
                            vol_tf_metrics.get("feature_importance"),
                            "🔍 Volatility Feature Importance (Top 20)",
                        )

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
                  <li><strong>Classification (分类预测):</strong> 预测未来涨跌方向的概率，范围 [0, 1]，0.5为阈值</li>
                  <li><strong>Return Regression (收益回归):</strong> 预测未来收益率的幅度，单位：收益率比例（例如 0.01 表示 1%）</li>
                  <li><strong>Volatility:</strong> 预测未来波动率，单位：波动率比例（例如 0.01 表示 1%）</li>
                </ul>
                </div>
                {ar1_section if 'ar1_section' in locals() and ar1_section else ""}
                {directional_section}
                {quantile_section}
                {return_section if 'return_section' in locals() and return_section else ""}
                {return_feature_section}
                {vol_section}
                {vol_feature_section}
                {oos_section}
                <h2>📁 模型文件 (Artifacts)</h2>
                <p><em>所有训练好的模型文件，可以直接用于实盘推理。每个Pipeline包含完整的预处理+模型+后处理流程。</em></p>
                <p><strong>使用说明：</strong></p>
                <ul style='margin:10px 0;padding-left:20px;'>
                  <li><strong>Pipeline文件（.pkl）</strong>：使用joblib保存的完整Pipeline，包含预处理、模型和后处理。可以直接加载使用：<code>pipeline = joblib.load('xxx_pipeline.pkl')</code>，然后调用<code>pipeline.predict(X)</code>进行预测。
                    <ul style='margin:5px 0;padding-left:20px;'>
                      <li>✅ <strong>已包含</strong>：模型本身（LightGBM）、目标变量预处理参数（RobustWinsorizer）、特征列列表、模型配置</li>
                      <li>✅ <strong>可以直接使用</strong>：加载Pipeline后，传入特征数据即可直接预测，无需额外配置</li>
                    </ul>
                  </li>
                  <li><strong>Scalers文件（scalers.pkl）</strong>：包含特征工程阶段的标准化参数，用于特征预处理。
                    <ul style='margin:5px 0;padding-left:20px;'>
                      <li>📌 <strong>用途</strong>：在特征工程阶段标准化特征（如StandardScaler、ATR分位数、波动率分位数等）</li>
                      <li>📌 <strong>何时需要</strong>：如果使用特征工程器（FeatureEngineer）生成特征，需要<strong>手动加载</strong>scalers.pkl来标准化特征，确保推理时使用与训练时相同的预处理方式</li>
                      <li>📌 <strong>使用方法</strong>：
                        <pre style='background:#f5f5f5;padding:10px;border-radius:5px;margin:5px 0;'><code>from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer

# 初始化特征工程器
fe = ComprehensiveFeatureEngineer()

# 手动加载scalers（重要！）
fe.load_scalers('scalers.pkl')

# 生成特征（会自动使用加载的scalers）
features = fe.engineer_all_features(df, fit=False)</code></pre>
                      </li>
                      <li>📌 <strong>注意</strong>：Pipeline文件中的预处理（RobustWinsorizer）是针对目标变量的，而scalers.pkl是针对特征变量的，两者作用不同。如果特征已经准备好（已标准化），则不需要scalers.pkl</li>
                    </ul>
                  </li>
                  <li><strong>部署建议</strong>：在生产环境中，建议定期重新训练模型以保持预测能力。同时监控模型在实盘中的表现，如果指标下降，及时更新模型。</li>
                </ul>
                <ul>
                  {''.join(artifacts_list)}
                </ul>
                </div>
                </body></html>"""
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"📝 HTML report written to: {report_path}")

                artifacts_to_copy = [
                    "classification_pipeline.pkl",
                    "return_pipeline.pkl",
                    "vol_pipeline.pkl",
                    "q10_pipeline.pkl",
                    "q50_pipeline.pkl",
                    "q90_pipeline.pkl",
                    "scalers.pkl",
                    "features.txt",
                    "training_info.json",
                    "training_report.html",
                ]
                for artifact_name in artifacts_to_copy:
                    src_path = os.path.join(combo_dir, artifact_name)
                    if os.path.isfile(src_path):
                        dst_path = os.path.join(models_combo_dir, artifact_name)
                        shutil.copy2(src_path, dst_path)

            except Exception as exc:  # noqa: BLE001
                print(f"Note: Could not write compact training report: {exc}")

    # Generate training summary report for this training run
    # Generate report in the timestamped directory to avoid mixing with old data
    try:
        from time_series_model.pipeline.training.generate_summary_report import (
            generate_summary_report,
        )

        # Generate report in the timestamped base directory
        generate_summary_report(base_results_dir, None)
        print(f"\n📊 Training summary report generated in: {base_results_dir}")
        summary_path = os.path.join(base_results_dir, "summary_report.html")
        if os.path.exists(summary_path):
            shutil.copy2(
                summary_path, os.path.join(base_models_dir, "summary_report.html")
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Note: Could not generate training summary report: {exc}")


if __name__ == "__main__":
    main()
