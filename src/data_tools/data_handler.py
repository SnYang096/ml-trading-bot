"""Data loading and preprocessing module.

This module provides:
- MarketDataLoader: Low-level loader for tick/OHLC parquet files with caching
- DataHandler: Unified high-level interface for OHLCV and tick data loading
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

from src.data_tools.tick_loader import load_tick_data
from src.data_tools.processors import (
    Processor,
    ProcessorChain,
    get_default_processor_chain,
)

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


class DataHandler:
    """
    Unified data handler for market data loading.

    This class provides a consistent interface for loading OHLCV and tick data,
    ensuring that all entry points (training, backtesting, feature store, live)
    use the same data preprocessing logic.

    Attributes:
        data_path: Path to OHLCV parquet data directory
        tick_data_path: Path to tick data directory (defaults to data_path)
    """

    # Standard base columns that should be present in all OHLCV DataFrames
    BASE_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "_symbol",
    ]

    # Orderflow base columns (computed from ticks or present in raw data)
    ORDERFLOW_BASE_COLUMNS = [
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
        "delta",
    ]

    def __init__(
        self,
        data_path: str,
        tick_data_path: Optional[str] = None,
        processors: Optional[List[Processor]] = None,
        use_default_processors: bool = False,
    ):
        """
        Initialize DataHandler.

        Args:
            data_path: Path to OHLCV parquet data directory
            tick_data_path: Path to tick data directory (defaults to data_path)
            processors: Optional list of processors to apply after loading
            use_default_processors: If True, use default processor chain
        """
        self.data_path = Path(data_path)
        self.tick_data_path = Path(tick_data_path) if tick_data_path else self.data_path
        self._market_loader = MarketDataLoader(str(self.data_path))

        # Setup processor chain
        if processors:
            self._processor_chain = ProcessorChain(processors)
        elif use_default_processors:
            self._processor_chain = get_default_processor_chain()
        else:
            self._processor_chain = None

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data with base columns.

        This method loads and resamples market data, ensuring a consistent
        base schema across all entry points.

        Args:
            symbol: Trading symbol(s), comma-separated for multi-asset
            timeframe: Resampling timeframe (e.g., "15T", "1H", "240T")
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional

        Returns:
            DataFrame with OHLCV + base columns, indexed by datetime

        Raises:
            ValueError: If no data found for symbol(s)
        """
        symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
        all_dfs = []

        for sym in symbol_list:
            df_single = self._market_loader.load_data(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )

            if df_single is not None and not df_single.empty:
                # Ensure datetime index
                if not isinstance(df_single.index, pd.DatetimeIndex):
                    for col in ("datetime", "timestamp", "date"):
                        if col in df_single.columns:
                            df_single.index = pd.to_datetime(df_single[col])
                            break

                # Resample to ensure consistent aggregation rules
                if isinstance(df_single.index, pd.DatetimeIndex):
                    agg_dict = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }

                    # Add orderflow columns if they exist
                    for col in self.ORDERFLOW_BASE_COLUMNS:
                        if col in df_single.columns:
                            if col in ["buy_qty", "sell_qty", "delta"]:
                                agg_dict[col] = "sum"
                            else:
                                agg_dict[col] = "last"

                    # Add other numeric columns (use last as default)
                    for col in df_single.columns:
                        if (
                            col not in agg_dict
                            and pd.api.types.is_numeric_dtype(df_single[col])
                            and col != "_symbol"
                        ):
                            agg_dict[col] = "last"

                    df_single = df_single.resample(timeframe).agg(agg_dict).dropna()

                # Ensure _symbol column
                if "_symbol" not in df_single.columns:
                    df_single["_symbol"] = sym

                df_single = df_single.sort_index()
                all_dfs.append(df_single)

        if not all_dfs:
            raise ValueError(f"No data found for symbol(s): {symbol}")

        df = pd.concat(all_dfs, axis=0).sort_index()

        # Remove duplicate indices (keep last)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]

        # Apply processor chain if configured
        if self._processor_chain:
            df = self._processor_chain.process(df)

        return df

    def load_ticks(
        self,
        symbol: str,
        start_ts: str,
        end_ts: str,
        df_bars: Optional[pd.DataFrame] = None,
        lookback_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Load tick data aligned to bar timeframe.

        Args:
            symbol: Trading symbol
            start_ts: Start timestamp (YYYY-MM-DD HH:MM:SS or ISO format)
            end_ts: End timestamp (YYYY-MM-DD HH:MM:SS or ISO format)
            df_bars: Optional DataFrame with bar index to align ticks
            lookback_minutes: Additional lookback minutes for tick loading

        Returns:
            DataFrame with tick data (columns: price, volume, side)
        """
        if df_bars is not None and not df_bars.empty:
            if isinstance(df_bars.index, pd.DatetimeIndex):
                start_ts = df_bars.index.min() - pd.Timedelta(minutes=lookback_minutes)
                end_ts = df_bars.index.max() + pd.Timedelta(minutes=lookback_minutes)
                start_ts = start_ts.strftime("%Y-%m-%d %H:%M:%S")
                end_ts = end_ts.strftime("%Y-%m-%d %H:%M:%S")

        return load_tick_data(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            ticks_dir=str(self.tick_data_path),
            lookback_minutes=lookback_minutes,
        )

    def get_base_schema(self) -> List[str]:
        """Get the list of base columns that should be present in all OHLCV DataFrames."""
        return self.BASE_COLUMNS.copy()

    def get_orderflow_base_schema(self) -> List[str]:
        """Get the list of orderflow base columns."""
        return self.ORDERFLOW_BASE_COLUMNS.copy()

    def ensure_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that base columns are present in the DataFrame.

        Missing columns are filled with appropriate defaults.
        """
        df = df.copy()

        if "_symbol" not in df.columns:
            df["_symbol"] = "UNKNOWN"

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the processor chain to a DataFrame.

        Useful for processing data that was loaded separately or
        for applying processors to feature data.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        if self._processor_chain:
            return self._processor_chain.process(df)
        return df

    def set_processors(self, processors: List[Processor]) -> None:
        """
        Set a new processor chain.

        Args:
            processors: List of processors to apply
        """
        self._processor_chain = ProcessorChain(processors)
