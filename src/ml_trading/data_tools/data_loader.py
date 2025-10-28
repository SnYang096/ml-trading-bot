"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from ml_trading.config.settings import TIMEFRAMES, TECHNICAL_INDICATORS


class MarketDataLoader:
    """Handles loading and preprocessing of market data."""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to market data CSV file
        """
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load raw market data.

        Returns:
            DataFrame with OHLCV data
        """
        if self.data_path:
            # Load real data from CSV file
            print(f"Loading data from {self.data_path}")
            # Load the aggregate trade data
            agg_trades = pd.read_csv(self.data_path)

            # Convert timestamp to datetime
            agg_trades["timestamp"] = pd.to_datetime(
                agg_trades["transact_time"], unit="ms"
            )
            agg_trades.set_index("timestamp", inplace=True)

            # Convert price and quantity to numeric
            agg_trades["price"] = pd.to_numeric(agg_trades["price"], errors="coerce")
            agg_trades["quantity"] = pd.to_numeric(
                agg_trades["quantity"], errors="coerce"
            )

            # Resample to 1-second bars to create OHLCV data
            ohlc_dict = {"price": "ohlc", "quantity": "sum"}

            # Group by 1-second intervals and create OHLCV
            raw_ohlc = agg_trades.groupby(pd.Grouper(freq="1s")).agg(
                ohlc_dict
            )  # Changed '1S' to '1s'
            raw_ohlc.columns = ["open", "high", "low", "close", "volume"]

            # Forward fill any missing values
            raw_ohlc = raw_ohlc.ffill()

            self.raw_data = raw_ohlc.dropna()
            print(
                f"Loaded {len(self.raw_data)} 1-second bars from aggregate trade data"
            )
        else:
            # Generate sample data for demonstration
            print("Generating sample data for demonstration")
            dates = pd.date_range("2020-01-01", periods=10000, freq="1min")
            prices = 100 + np.cumsum(np.random.randn(10000) * 0.1)
            volume = np.random.randint(1000, 10000, size=10000)

            self.raw_data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": prices,
                    "high": prices + np.abs(np.random.randn(10000) * 0.05),
                    "low": prices - np.abs(np.random.randn(10000) * 0.05),
                    "close": prices + np.random.randn(10000) * 0.02,
                    "volume": volume,
                }
            )

            self.raw_data["timestamp"] = pd.to_datetime(self.raw_data["timestamp"])
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
