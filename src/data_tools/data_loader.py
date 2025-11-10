"""Data loading and preprocessing module."""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from time_series_model.config.settings import TIMEFRAMES, TECHNICAL_INDICATORS


class MarketDataLoader:
    """Handles loading and preprocessing of market data.
    data_loader.py 的 MarketDataLoader 负责“路径解析 + 目录扫描 + 多格式读取（parquet/csv/zip）+ agg-trade→1s OHLCV 转换 + 按日期裁剪”。这是底层通用数据层。
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to market data CSV file
        """
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None

    def load_data(self,
                  symbol: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw market data.

        Returns:
            DataFrame with OHLCV data
        """
        if self.data_path:
            path = Path(self.data_path)
            print(f"Loading data from {path}")

            def _normalize_symbol(s: str) -> str:
                return s.replace("_", "-").upper()

            def _symbol_tokens(sym: Optional[str]) -> set:
                if not sym:
                    return set()
                s = sym.upper()
                tokens = {s, s.replace("_", "-"), s.replace("-", "_")}
                # Common mapping: BTCUSDT -> BTC-USD / BTC_USD
                if s.endswith("USDT") and len(s) > 4:
                    base = s[:-4]
                    tokens.add(f"{base}-USD")
                    tokens.add(f"{base}_USD")
                return {t.upper() for t in tokens}

            def _read_and_standardize_frame(p: Path) -> pd.DataFrame:
                if p.suffix.lower() == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)

                # Try aggregate trade schema -> resample to OHLCV
                if {"transact_time", "price", "quantity"}.issubset(df.columns):
                    df["timestamp"] = pd.to_datetime(df["transact_time"],
                                                     unit="ms")
                    df.set_index("timestamp", inplace=True)
                    df["price"] = pd.to_numeric(df["price"], errors="coerce")
                    df["quantity"] = pd.to_numeric(df["quantity"],
                                                   errors="coerce")
                    raw_ohlc = df.groupby(pd.Grouper(freq="1s")).agg({
                        "price":
                        "ohlc",
                        "quantity":
                        "sum"
                    })
                    raw_ohlc.columns = [
                        "open", "high", "low", "close", "volume"
                    ]
                    return raw_ohlc

                # Try OHLCV parquet/csv with timestamp column
                if "timestamp" in df.columns and {
                        "open", "high", "low", "close", "volume"
                }.issubset(df.columns):
                    df["timestamp"] = pd.to_datetime(df["timestamp"],
                                                     errors="coerce")
                    df = df.dropna(subset=["timestamp"]).set_index(
                        "timestamp").sort_index()
                    # Preserve auxiliary columns (e.g., order-flow)
                    return df

                # Try already indexed by timestamp
                if {"open", "high", "low", "close",
                        "volume"}.issubset(df.columns):
                    # Preserve auxiliary columns
                    return df

                raise ValueError(f"Unrecognized data schema in file: {p}")

            frames: List[pd.DataFrame] = []
            if path.is_dir():
                # Collect candidate files
                exts = (".parquet", ".csv")
                candidates = [
                    p for p in path.rglob("*") if p.suffix.lower() in exts
                ]
                if symbol:
                    sym_tokens = _symbol_tokens(symbol)

                    def _match(name: str) -> bool:
                        up = name.upper().replace("_", "-")
                        return any(tok in up for tok in sym_tokens)

                    candidates = [p for p in candidates if _match(p.name)]
                if not candidates:
                    raise FileNotFoundError(
                        f"No data files found under directory: {path}")
                for file_path in sorted(candidates):
                    try:
                        frames.append(_read_and_standardize_frame(file_path))
                    except Exception:
                        # Skip unreadable files silently to allow mixed folders
                        continue
                if not frames:
                    raise FileNotFoundError(
                        f"No readable OHLCV/agg-trade files in {path}")
                df_all = pd.concat(frames).sort_index()
                # Deduplicate on index if overlapping months
                df_all = df_all[~df_all.index.duplicated(keep="last")]
                self.raw_data = df_all.ffill().dropna()
            else:
                # Single file path
                p = path
                self.raw_data = _read_and_standardize_frame(p).ffill().dropna()

            # Optional date filtering on index
            if start_date or end_date:
                idx = self.raw_data.index
                if not isinstance(idx, pd.DatetimeIndex):
                    self.raw_data = self.raw_data.copy()
                    self.raw_data.index = pd.to_datetime(self.raw_data.index,
                                                         errors="coerce")
                start_ts = pd.to_datetime(
                    start_date) if start_date else self.raw_data.index.min()
                end_ts = pd.to_datetime(
                    end_date) if end_date else self.raw_data.index.max()
                self.raw_data = self.raw_data.loc[
                    (self.raw_data.index >= start_ts)
                    & (self.raw_data.index <= end_ts)]

            print(f"Loaded {len(self.raw_data)} bars")
        else:
            # Generate sample data for demonstration
            print("Generating sample data for demonstration")
            dates = pd.date_range("2020-01-01", periods=10000, freq="1min")
            prices = 100 + np.cumsum(np.random.randn(10000) * 0.1)
            volume = np.random.randint(1000, 10000, size=10000)

            self.raw_data = pd.DataFrame({
                "timestamp":
                dates,
                "open":
                prices,
                "high":
                prices + np.abs(np.random.randn(10000) * 0.05),
                "low":
                prices - np.abs(np.random.randn(10000) * 0.05),
                "close":
                prices + np.random.randn(10000) * 0.02,
                "volume":
                volume,
            })

            self.raw_data["timestamp"] = pd.to_datetime(
                self.raw_data["timestamp"])
            self.raw_data.set_index("timestamp", inplace=True)

        return self.raw_data

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

        # Replace deprecated 'T' with 'min'
        timeframe = timeframe.replace("T", "min")

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
            data_dict["buy_qty"] = self.raw_data["buy_qty"].resample(
                timeframe).sum()
        if have_sell:
            data_dict["sell_qty"] = self.raw_data["sell_qty"].resample(
                timeframe).sum()
        if have_ratio:
            # ratio is averaged over the window
            data_dict["taker_buy_ratio"] = (
                self.raw_data["taker_buy_ratio"].resample(timeframe).mean())
        if have_cvd:
            # cvd is cumulative; use last value in window
            data_dict["cvd"] = self.raw_data["cvd"].resample(timeframe).last()

        # Combine into a single DataFrame
        resampled = pd.DataFrame(data_dict).dropna()

        return resampled

    def get_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for all configured timeframes.

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        multi_tf_data = {}
        for tf in TIMEFRAMES:
            multi_tf_data[tf] = self.resample_data(tf)
        return multi_tf_data
