"""
数据缺失补全器

检测实时流中的数据缺失，并从交易所 API 下载补全。
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ ccxt not installed. Install with: pip install ccxt")


class DataGapFiller:
    """
    数据缺失补全器

    功能：
    1. 检测数据缺失（通过时间戳连续性）
    2. 从交易所 API 下载缺失数据
    3. 验证和清洗下载的数据
    4. 返回补全的数据
    """

    def __init__(self, exchange: Any):
        """
        Args:
            exchange: ccxt Exchange 实例
        """
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required for DataGapFiller")

        self.exchange = exchange

    def detect_missing_bars(
        self,
        df: pd.DataFrame,
        timeframe: str,
        tolerance: Optional[pd.Timedelta] = None,
    ) -> List[pd.Timestamp]:
        """
        检测缺失的 K线数据

        Args:
            df: 已有的 K线数据（按时间排序）
            timeframe: 时间框架（如 "15T"）
            tolerance: 允许的时间误差（默认 10%）

        Returns:
            缺失的时间戳列表
        """
        if len(df) < 2:
            return []

        if "timestamp" not in df.columns:
            return []

        # 计算期望的间隔
        expected_interval = pd.Timedelta(timeframe)
        if tolerance is None:
            tolerance = expected_interval * 0.1

        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        missing_timestamps = []

        for i in range(len(df_sorted) - 1):
            current_time = pd.Timestamp(df_sorted.iloc[i]["timestamp"])
            next_time = pd.Timestamp(df_sorted.iloc[i + 1]["timestamp"])

            # 计算实际间隔
            gap = next_time - current_time

            # 如果间隔大于期望间隔 + 容差，认为有缺失
            if gap > expected_interval + tolerance:
                # 计算缺失的时间戳数量
                missing_count = int((gap - tolerance) / expected_interval)

                for j in range(1, missing_count + 1):
                    missing_time = current_time + expected_interval * j
                    # 确保缺失时间在容差范围内
                    if missing_time < next_time - tolerance:
                        missing_timestamps.append(missing_time)

        return missing_timestamps

    def download_missing_bars(
        self,
        symbol: str,
        missing_timestamps: List[pd.Timestamp],
        timeframe: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        从交易所下载缺失的 K线数据

        Args:
            symbol: 交易对符号（ccxt 格式，如 "BTC/USDT:USDT"）
            missing_timestamps: 缺失的时间戳列表
            timeframe: 时间框架（ccxt 格式，如 "15m"）
            max_retries: 最大重试次数

        Returns:
            下载的 K线数据 DataFrame
        """
        if not missing_timestamps:
            return pd.DataFrame()

        # 转换时间框架格式
        ccxt_timeframe = self._convert_timeframe(timeframe)

        # 找到时间范围
        start_time = min(missing_timestamps)
        end_time = max(missing_timestamps)

        # 计算需要下载的数据量
        expected_count = len(missing_timestamps)

        # 多下载一些，避免边界问题
        limit = expected_count + 100

        for attempt in range(max_retries):
            try:
                # 转换为毫秒时间戳
                since = int(start_time.timestamp() * 1000)

                # 下载数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=ccxt_timeframe,
                    since=since,
                    limit=limit,
                )

                if not ohlcv:
                    print(f"⚠️ 下载返回空数据")
                    return pd.DataFrame()

                # 转换为 DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # 过滤出缺失的时间戳（允许小的时间误差）
                tolerance = pd.Timedelta(timeframe) * 0.1
                matched_bars = []

                for missing_ts in missing_timestamps:
                    # 找到最接近的时间戳
                    time_diffs = abs(df["timestamp"] - missing_ts)
                    min_diff_idx = time_diffs.idxmin()

                    if time_diffs.iloc[min_diff_idx] <= tolerance:
                        matched_bars.append(min_diff_idx)

                if matched_bars:
                    df_matched = df.iloc[matched_bars].copy()
                    df_matched = df_matched.drop_duplicates(subset=["timestamp"])
                    print(
                        f"✅ 下载了 {len(df_matched)} 条缺失数据（期望 {expected_count} 条）"
                    )
                    return df_matched
                else:
                    print(f"⚠️ 下载的数据中没有匹配的时间戳")
                    return pd.DataFrame()

            except Exception as e:
                print(f"⚠️ 下载缺失数据失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 指数退避
                else:
                    print(f"❌ 下载失败，已重试 {max_retries} 次")
                    return pd.DataFrame()

        return pd.DataFrame()

    def _convert_timeframe(self, timeframe: str) -> str:
        """
        转换时间框架格式

        Args:
            timeframe: 时间框架（如 "15T" 或 "15m"）

        Returns:
            ccxt 格式的时间框架（如 "15m"）
        """
        if "T" in timeframe:
            return timeframe.replace("T", "m")
        return timeframe

    def validate_downloaded_data(
        self,
        df: pd.DataFrame,
        expected_timestamps: List[pd.Timestamp],
        timeframe: str,
    ) -> pd.DataFrame:
        """
        验证下载的数据质量

        Args:
            df: 下载的数据
            expected_timestamps: 期望的时间戳列表
            timeframe: 时间框架

        Returns:
            验证通过的数据
        """
        if len(df) == 0:
            return df

        tolerance = pd.Timedelta(timeframe) * 0.1
        validated = []

        for expected_ts in expected_timestamps:
            # 找到最接近的数据
            time_diffs = abs(df["timestamp"] - expected_ts)
            min_diff_idx = time_diffs.idxmin()

            if time_diffs.iloc[min_diff_idx] <= tolerance:
                row = df.iloc[min_diff_idx].copy()

                # 验证数据合理性
                if self._validate_bar(row):
                    validated.append(row)

        if validated:
            return pd.DataFrame(validated).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def _validate_bar(self, bar: pd.Series) -> bool:
        """
        验证单条 K线数据的合理性

        Args:
            bar: K线数据 Series

        Returns:
            是否通过验证
        """
        try:
            # 检查基本字段
            required_fields = ["open", "high", "low", "close", "volume"]
            for field in required_fields:
                if field not in bar or pd.isna(bar[field]):
                    return False

            # 检查价格合理性
            if not (bar["low"] <= bar["open"] <= bar["high"]):
                return False
            if not (bar["low"] <= bar["close"] <= bar["high"]):
                return False
            if not (bar["low"] <= bar["high"]):
                return False

            # 检查数值合理性
            if bar["volume"] < 0:
                return False
            if bar["open"] <= 0 or bar["close"] <= 0:
                return False

            return True

        except Exception:
            return False


# 使用示例
if __name__ == "__main__":
    if not CCXT_AVAILABLE:
        print("❌ 请先安装 ccxt: pip install ccxt")
        exit(1)

    # 创建交易所实例
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    # 创建数据补全器
    gap_filler = DataGapFiller(exchange)

    # 模拟已有数据（有缺失）
    df_existing = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01 10:00:00"),
                pd.Timestamp("2024-01-01 10:15:00"),
                # 缺失 10:30, 10:45
                pd.Timestamp("2024-01-01 11:00:00"),
            ],
            "open": [50000, 50100, 50200],
            "high": [50100, 50200, 50300],
            "low": [49900, 50000, 50100],
            "close": [50050, 50150, 50250],
            "volume": [100, 110, 120],
        }
    )

    # 检测缺失
    missing = gap_filler.detect_missing_bars(df_existing, timeframe="15T")
    print(f"检测到缺失数据: {missing}")

    # 下载缺失数据
    if missing:
        df_filled = gap_filler.download_missing_bars(
            symbol="BTC/USDT:USDT",
            missing_timestamps=missing,
            timeframe="15T",
        )
        print(f"下载的数据: {len(df_filled)} 条")
