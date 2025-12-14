"""Data loading and preprocessing module."""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

TIMEFRAME_CACHE_DIR = Path("cache/timeframes")
TIMEFRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MarketDataLoader:
    """Handles loading of tick-level parquet and resamples to requested timeframe."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None

    def load_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "15T",
    ) -> pd.DataFrame:
        """
        Load raw market data.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Resampling timeframe (e.g., "15T", "60T", "240T")

        Returns:
            DataFrame with OHLCV data
        """
        if not self.data_path:
            raise ValueError("data_path is required for MarketDataLoader.")

        if not symbol:
            raise ValueError("Symbol must be provided to load tick parquet files.")

        cache_file = self._cache_file_path(symbol, timeframe)
        if cache_file.exists():
            df_cached = pd.read_parquet(cache_file)
        else:
            df_cached = self._build_timeframe_cache(symbol, timeframe)
            df_cached.to_parquet(cache_file)

        df_cached.index = pd.to_datetime(df_cached.index)
        start_ts = pd.to_datetime(start_date) if start_date else df_cached.index.min()
        end_ts = pd.to_datetime(end_date) if end_date else df_cached.index.max()
        df_subset = df_cached.loc[
            (df_cached.index >= start_ts) & (df_cached.index <= end_ts)
        ].copy()
        df_subset["_symbol"] = symbol
        self.raw_data = df_subset
        print(f"Loaded {len(df_subset)} bars")
        return self.raw_data

    def _cache_file_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return TIMEFRAME_CACHE_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def _build_timeframe_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        data_root = Path(self.data_path)
        pattern = f"{symbol}_*.parquet"
        files = sorted(data_root.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for symbol {symbol} under {data_root}"
            )

        frames = []
        for file_path in files:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue
            if {"open", "high", "low", "close"}.issubset(df.columns):
                frames.append(self._resample_from_ohlc(df, timeframe))
            else:
                frames.append(self._resample_from_ticks(df, timeframe))

        if not frames:
            raise ValueError(f"No usable data after reading {len(files)} files.")

        result = pd.concat(frames).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        return result

    def _resample_from_ohlc(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        df = df.sort_index()

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        optional_cols = [
            "buy_qty",
            "sell_qty",
            "taker_buy_ratio",
            "cvd",
            "cvd_short",
            "cvd_medium",
            "cvd_long",
            "cvd_change_1",
            "cvd_change_5",
            "cvd_change_20",
            "cvd_normalized",
        ]
        for col in optional_cols:
            if col in df.columns:
                agg_dict[col] = "sum"

        resampled = df.resample(timeframe).agg(agg_dict).dropna()
        resampled["trade_count"] = df["close"].resample(timeframe).size()
        return resampled

    def _resample_from_ticks(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" not in df.columns:
            raise ValueError("Tick parquet must contain 'timestamp' column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["timestamp", "price", "volume"])
        df = df.sort_values("timestamp").set_index("timestamp")

        price_ohlc = df["price"].resample(timeframe).ohlc().dropna()
        volume = df["volume"].resample(timeframe).sum()
        result = price_ohlc.rename(
            columns={"open": "open", "high": "high", "low": "low", "close": "close"}
        )
        result["volume"] = volume
        result["trade_count"] = df["price"].resample(timeframe).size()

        if "side" in df.columns:
            buy_series = np.where(df["side"] == 1, df["volume"], 0.0)
            sell_series = np.where(df["side"] == -1, df["volume"], 0.0)
            order_flow = pd.DataFrame(
                {"buy_qty": buy_series, "sell_qty": sell_series}, index=df.index
            )
            flow_resampled = order_flow.resample(timeframe).sum()
            result["buy_qty"] = flow_resampled["buy_qty"]
            result["sell_qty"] = flow_resampled["sell_qty"]
            total_flow = result["buy_qty"] + result["sell_qty"]
            result["taker_buy_ratio"] = (
                result["buy_qty"] / total_flow.replace(0, np.nan)
            ).fillna(0.5)

            delta = result["buy_qty"].fillna(0) - result["sell_qty"].fillna(0)
            result["cvd_change_1"] = delta
            result["cvd_change_5"] = delta.rolling(window=5, min_periods=1).sum()
            result["cvd_change_20"] = delta.rolling(window=20, min_periods=1).sum()
            result["cvd_short"] = delta.rolling(window=20, min_periods=1).sum()
            result["cvd_medium"] = delta.rolling(window=60, min_periods=1).sum()
            result["cvd_long"] = delta.rolling(window=288, min_periods=1).sum()
            result["cvd"] = delta.cumsum()
            result["cvd_normalized"] = delta / total_flow.replace(0, np.nan)
            result["cvd_normalized"] = result["cvd_normalized"].fillna(0)
        else:
            for col in [
                "buy_qty",
                "sell_qty",
                "taker_buy_ratio",
                "cvd",
                "cvd_short",
                "cvd_medium",
                "cvd_long",
                "cvd_change_1",
                "cvd_change_5",
                "cvd_change_20",
                "cvd_normalized",
            ]:
                result[col] = 0.0

        return result

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """
        Resample data to specified timeframe.

        Args:
            timeframe: Target timeframe (e.g., '5min', '15min', '45min')

        Returns:
            Resampled DataFrame
        """
        if self.raw_data is None:
            self.load_data()

        # Ensure raw_data is not None before using it
        if self.raw_data is None:
            raise ValueError("Raw data is None. Call load_data() first.")

        # Convert timeframe format: pandas resample accepts 'T' (minutes) directly
        # Keep 'T' format as pandas prefers it (e.g., '5T', '15T', '240T')
        # Handle different input formats:
        # 1. Pure number (e.g., "15") -> "15T" (15 minutes)
        # 2. "min" format (e.g., "15min") -> "15T"
        # 3. Already in "T" format (e.g., "15T") -> keep as is
        if timeframe.isdigit():
            # Pure number: assume minutes
            timeframe = f"{timeframe}T"
        elif timeframe.endswith("min") and not timeframe.endswith("T"):
            # "min" format: convert to "T"
            timeframe = timeframe.replace("min", "T")
        # If already ends with "T", keep as is

        # Using separate operations for each column to avoid type issues
        resampled_open = self.raw_data["open"].resample(timeframe).first()
        resampled_high = self.raw_data["high"].resample(timeframe).max()
        resampled_low = self.raw_data["low"].resample(timeframe).min()
        resampled_close = self.raw_data["close"].resample(timeframe).last()
        resampled_volume = self.raw_data["volume"].resample(timeframe).sum()

        # Optional microstructure columns propagated if present in raw_data
        have_buy = "buy_qty" in self.raw_data.columns
        have_sell = "sell_qty" in self.raw_data.columns
        have_ratio = "taker_buy_ratio" in self.raw_data.columns
        have_cvd = "cvd" in self.raw_data.columns

        data_dict = {
            "open": resampled_open,
            "high": resampled_high,
            "low": resampled_low,
            "close": resampled_close,
            "volume": resampled_volume,
        }
        if have_buy:
            data_dict["buy_qty"] = self.raw_data["buy_qty"].resample(timeframe).sum()
        if have_sell:
            data_dict["sell_qty"] = self.raw_data["sell_qty"].resample(timeframe).sum()
        if have_ratio:
            # ratio is averaged over the window
            data_dict["taker_buy_ratio"] = (
                self.raw_data["taker_buy_ratio"].resample(timeframe).mean()
            )
        if have_cvd:
            # cvd is cumulative; use last value in window
            data_dict["cvd"] = self.raw_data["cvd"].resample(timeframe).last()

        # Combine into a single DataFrame
        resampled = pd.DataFrame(data_dict).dropna()

        return resampled
