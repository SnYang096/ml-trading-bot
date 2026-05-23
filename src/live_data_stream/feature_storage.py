"""
特征存储模块

实现三种存储路径：
1. features_4h/{symbol}/{YYYY-MM-DD}.parquet - 4小时特征（每4小时保存）
2. features_15min/{symbol}/{YYYY-MM-DD}.parquet - 15分钟特征（每15分钟保存）
3. ticks/{symbol}/{YYYY-MM-DD}.parquet - 1分钟聚合tick数据（实时保存，包括未完成的bar）

用于：
- Warmup启动时加载历史数据
- 恢复特征计算状态
- 补数据时知道从哪里开始
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from src.live_data_stream.parquet_io import (
    atomic_write_parquet,
    is_unreadable_parquet,
    quarantine_corrupt_parquet,
    read_parquet_safe,
)

_logger = logging.getLogger(__name__)

_TICK_EMPTY = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def sanitize_dated_parquet_for_symbols(
    storage: "StorageManager",
    symbols: Iterable[str],
    *,
    lookback_days: int = 3,
) -> int:
    """Quarantine corrupt tick/bar day files for recent dates before warmup."""
    removed = 0
    cutoff = datetime.now() - timedelta(days=max(1, int(lookback_days)))
    roots = (storage.ticks.root, storage.bar_1min.root)
    sym_set = {str(s).upper() for s in symbols}
    for root in roots:
        if not root.exists():
            continue
        for symbol_dir in root.iterdir():
            if not symbol_dir.is_dir():
                continue
            if sym_set and symbol_dir.name.upper() not in sym_set:
                continue
            for path in symbol_dir.glob("*.parquet"):
                try:
                    file_date = datetime.strptime(path.stem, "%Y-%m-%d")
                except ValueError:
                    continue
                if file_date < cutoff:
                    continue
                if is_unreadable_parquet(path):
                    if quarantine_corrupt_parquet(path, logger=_logger):
                        removed += 1
    if removed:
        _logger.warning(
            "sanitize: quarantined %d corrupt tick/bar parquet file(s)", removed
        )
    return removed


def _cleanup_dated_parquet(root: Path, days: int) -> int:
    """通用: 清理 root/{symbol}/{YYYY-MM-DD}.parquet 中超过 days 天的文件"""
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    if not root.exists():
        return 0
    for symbol_dir in root.iterdir():
        if not symbol_dir.is_dir():
            continue
        for f in symbol_dir.glob("*.parquet"):
            # 文件名格式: YYYY-MM-DD.parquet
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    if removed > 0:
        _logger.info(
            "cleanup: 清理 %s 下 %d 个 > %d 天的旧文件", root.name, removed, days
        )
    return removed


@dataclass
class Feature4HStorage:
    """
    4小时特征存储

    每4小时保存一次，用于warmup启动。
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, trading_date: str) -> Path:
        """获取文件路径"""
        symbol_dir = self.root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{trading_date}.parquet"

    def save(self, symbol: str, trading_date: str, features: pd.DataFrame) -> Path:
        """
        保存4小时特征

        Args:
            symbol: 交易对符号
            trading_date: 交易日期 (YYYY-MM-DD)
            features: 特征DataFrame，必须包含timestamp列

        Returns:
            保存的文件路径
        """
        target = self._path(symbol, trading_date)

        if target.exists():
            existing = read_parquet_safe(target, logger=_logger)
            combined = pd.concat([existing, features])
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = features.sort_values("timestamp").reset_index(drop=True)
        atomic_write_parquet(combined, target)
        return target

    def load(self, symbol: str, trading_date: str) -> pd.DataFrame:
        """加载4小时特征"""
        target = self._path(symbol, trading_date)
        return _normalize_timestamp_column(
            read_parquet_safe(target, logger=_logger)
        )

    def load_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载日期范围内的4小时特征"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        dfs = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            df = self.load(symbol, date_str)
            if len(df) > 0:
                dfs.append(df)
            current += timedelta(days=1)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def cleanup_old_files(self, days: int = 30) -> int:
        """删除超过 days 天的旧 parquet 文件

        Returns:
            删除的文件数
        """
        return _cleanup_dated_parquet(self.root, days)


@dataclass
class Feature15MinStorage:
    """
    15分钟特征存储

    每15分钟保存一次，用于warmup和恢复特征计算状态。
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, trading_date: str) -> Path:
        """获取文件路径"""
        symbol_dir = self.root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{trading_date}.parquet"

    def save(self, symbol: str, trading_date: str, features: pd.DataFrame) -> Path:
        """
        保存15分钟特征

        Args:
            symbol: 交易对符号
            trading_date: 交易日期 (YYYY-MM-DD)
            features: 特征DataFrame，必须包含timestamp列

        Returns:
            保存的文件路径
        """
        target = self._path(symbol, trading_date)

        if target.exists():
            existing = read_parquet_safe(target, logger=_logger)
            combined = pd.concat([existing, features])
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = features.sort_values("timestamp").reset_index(drop=True)
        atomic_write_parquet(combined, target)
        return target

    def load(self, symbol: str, trading_date: str) -> pd.DataFrame:
        """加载15分钟特征"""
        target = self._path(symbol, trading_date)
        return _normalize_timestamp_column(
            read_parquet_safe(target, logger=_logger)
        )

    def load_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载日期范围内的15分钟特征"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        dfs = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            df = self.load(symbol, date_str)
            if len(df) > 0:
                dfs.append(df)
            current += timedelta(days=1)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def get_latest_timestamp(
        self, symbol: str, trading_date: str
    ) -> Optional[pd.Timestamp]:
        """获取最新的特征时间戳（用于恢复计算状态）"""
        df = self.load(symbol, trading_date)
        if len(df) == 0:
            return None
        return pd.Timestamp(df["timestamp"].max())

    def cleanup_old_files(self, days: int = 30) -> int:
        """删除超过 days 天的旧 parquet 文件

        Returns:
            删除的文件数
        """
        return _cleanup_dated_parquet(self.root, days)


@dataclass
class TickStorage:
    """
    1分钟聚合tick数据存储（按买卖分离，与研究pipeline格式一致）

    数据格式：[timestamp, price, volume, side]
    - timestamp: pd.Timestamp
    - price: float (VWAP)
    - volume: float
    - side: int (1=buy, -1=sell)

    每1分钟生成2条记录（买方和卖方分开）
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, trading_date: str) -> Path:
        """获取文件路径"""
        symbol_dir = self.root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{trading_date}.parquet"

    def append(self, symbol: str, trading_date: str, ticks: pd.DataFrame) -> Path:
        """
        追加1分钟聚合tick数据

        Args:
            symbol: 交易对符号
            trading_date: 交易日期 (YYYY-MM-DD)
            ticks: tick的DataFrame，必须包含: timestamp, price, volume, side

        Returns:
            保存的文件路径
        """
        # 验证必要的列
        required_cols = ["timestamp", "price", "volume", "side"]
        missing_cols = [col for col in required_cols if col not in ticks.columns]
        if missing_cols:
            raise ValueError(
                f"Tick data must contain columns: {required_cols}. Missing: {missing_cols}"
            )

        target = self._path(symbol, trading_date)

        if target.exists():
            existing = read_parquet_safe(target, empty=_TICK_EMPTY, logger=_logger)
            combined = pd.concat([existing, ticks])
            combined = combined.drop_duplicates(
                subset=["timestamp", "side"], keep="last"
            )
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = ticks.sort_values("timestamp").reset_index(drop=True)
        atomic_write_parquet(combined, target)
        return target

    def load(self, symbol: str, trading_date: str) -> pd.DataFrame:
        """加载1分钟聚合tick数据"""
        target = self._path(symbol, trading_date)
        return _normalize_timestamp_column(
            read_parquet_safe(target, empty=_TICK_EMPTY.copy(), logger=_logger)
        )

    def load_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加製日期范围内的1分钟聚合tick数据"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_ticks = []
        current = start
        while current <= end:
            trading_date = current.strftime("%Y-%m-%d")
            daily_ticks = self.load(symbol, trading_date)
            if len(daily_ticks) > 0:
                all_ticks.append(daily_ticks)
            current += timedelta(days=1)

        if not all_ticks:
            return pd.DataFrame(columns=["timestamp", "price", "volume", "side"])

        combined = pd.concat(all_ticks, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def get_latest_timestamp(
        self, symbol: str, trading_date: str
    ) -> Optional[pd.Timestamp]:
        """获取最后一条tick的时间戳"""
        ticks = self.load(symbol, trading_date)
        if len(ticks) == 0:
            return None
        return pd.Timestamp(ticks["timestamp"].max())


@dataclass
class Tick1MinStorage:
    """
    1分钟聚合tick数据存储

    实时保存，包括未完成的bar，用于补数据时知道从哪里开始。
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, trading_date: str) -> Path:
        """获取文件路径"""
        symbol_dir = self.root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{trading_date}.parquet"

    def append(
        self,
        symbol: str,
        trading_date: str,
        bars: pd.DataFrame,
        include_incomplete: bool = True,
    ) -> Path:
        """
        追加1分钟聚合bar数据

        Args:
            symbol: 交易对符号
            trading_date: 交易日期 (YYYY-MM-DD)
            bars: 1分钟聚合bar的DataFrame，必须包含timestamp列
            include_incomplete: 是否包含未完成的bar（用于补数据时知道从哪里开始）

        Returns:
            保存的文件路径
        """
        target = self._path(symbol, trading_date)

        if target.exists():
            existing = read_parquet_safe(target, logger=_logger)
            if include_incomplete and len(existing) > 0:
                last_existing = existing.iloc[-1]
                last_new = bars.iloc[-1] if len(bars) > 0 else None
                if (
                    last_new is not None
                    and last_existing["timestamp"] == last_new["timestamp"]
                ):
                    existing = existing.iloc[:-1]
            combined = pd.concat([existing, bars])
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = bars.sort_values("timestamp").reset_index(drop=True)
        atomic_write_parquet(combined, target)
        return target

    def load(self, symbol: str, trading_date: str) -> pd.DataFrame:
        """加载1分钟聚合bar数据"""
        target = self._path(symbol, trading_date)
        return _normalize_timestamp_column(
            read_parquet_safe(target, logger=_logger)
        )

    def load_range(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载日期范围内的1分钟聚合tick数据"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        dfs = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            df = self.load(symbol, date_str)
            if len(df) > 0:
                dfs.append(df)
            current += timedelta(days=1)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def get_latest_timestamp(
        self, symbol: str, trading_date: str
    ) -> Optional[pd.Timestamp]:
        """获取最新的bar时间戳（用于补数据时知道从哪里开始）"""
        df = self.load(symbol, trading_date)
        if len(df) == 0:
            return None
        return pd.Timestamp(df["timestamp"].max())

    def get_incomplete_bar(self, symbol: str, trading_date: str) -> Optional[pd.Series]:
        """获取未完成的bar（最后一条，可能还在更新中）"""
        df = self.load(symbol, trading_date)
        if len(df) == 0:
            return None

        # 返回最后一条（可能是未完成的）
        return df.iloc[-1]


class StorageManager:
    """
    存储管理器

    统一管理三种存储，提供便捷的保存和加载接口。
    """

    def __init__(
        self,
        base_path: str | Path = "data/live_storage",
    ):
        """
        Args:
            base_path: 存储根目录
        """
        base_path = Path(base_path)
        self.base_path = base_path

        # 初始化四种存储
        self.feature_4h = Feature4HStorage(base_path / "features_4h")
        self.feature_15min = Feature15MinStorage(base_path / "features_15min")
        self.ticks = TickStorage(base_path / "ticks")  # 新增：tick级数据
        self.bar_1min = Tick1MinStorage(base_path / "bars")  # 重命名：bar级数据

    def get_trading_date(self, timestamp: pd.Timestamp | datetime | str) -> str:
        """获取交易日期字符串 (YYYY-MM-DD)"""
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp)

        # 使用UTC时间，转换为日期字符串
        return timestamp.strftime("%Y-%m-%d")

    def save_4h_features(
        self,
        symbol: str,
        features: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Path:
        """保存4小时特征"""
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz="UTC")
        trading_date = self.get_trading_date(timestamp)
        return self.feature_4h.save(symbol, trading_date, features)

    def save_15min_features(
        self,
        symbol: str,
        features: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Path:
        """保存15分钟特征"""
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz="UTC")
        trading_date = self.get_trading_date(timestamp)
        return self.feature_15min.save(symbol, trading_date, features)

    def save_ticks(
        self, symbol: str, ticks: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> Path:
        """保存1分钟聚合tick数据（按买卖分离）"""
        if timestamp is None:
            # 使用ticks中的第一个时间戳
            if len(ticks) > 0 and "timestamp" in ticks.columns:
                timestamp = pd.Timestamp(ticks["timestamp"].iloc[0])
            else:
                timestamp = pd.Timestamp.now(tz="UTC")
        trading_date = self.get_trading_date(timestamp)
        return self.ticks.append(symbol, trading_date, ticks)

    def save_1min_ticks(
        self,
        symbol: str,
        bars: pd.DataFrame,
        include_incomplete: bool = True,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Path:
        """保存1分钟聚合bar数据（重命名：实际保存的是OHLCV bar）"""
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz="UTC")
        trading_date = self.get_trading_date(timestamp)
        return self.bar_1min.append(symbol, trading_date, bars, include_incomplete)

    def warmup_load(
        self, symbol: str, days: int = 30, end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        加载warmup数据

        Args:
            symbol: 交易对符号
            days: 加载最近N天的数据
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天

        Returns:
            包含四种数据的字典：
            - features_4h: 4小时特征
            - features_15min: 15分钟特征
            - bars_1min: 1分钟OHLCV bar数据
            - ticks_1min: 1分钟聚合tick数据（按买卖分离）
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=days)
        start_date = start.strftime("%Y-%m-%d")

        return {
            "features_4h": self.feature_4h.load_range(symbol, start_date, end_date),
            "features_15min": self.feature_15min.load_range(
                symbol, start_date, end_date
            ),
            "bars_1min": self.bar_1min.load_range(symbol, start_date, end_date),
            "ticks_1min": self.ticks.load_range(symbol, start_date, end_date),  # 新增
        }

    def get_recovery_state(
        self, symbol: str, trading_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取恢复状态信息（用于从断线中恢复）

        Returns:
            包含恢复状态的字典：
            - latest_15min_timestamp: 最新的15分钟特征时间戳
            - latest_1min_bar_timestamp: 最新的1分钟bar时间戳
            - latest_tick_timestamp: 最新的tick时间戳
            - incomplete_bar: 未完成的1分钟bar（如果有）
        """
        if trading_date is None:
            trading_date = datetime.now().strftime("%Y-%m-%d")

        latest_15min = self.feature_15min.get_latest_timestamp(symbol, trading_date)
        latest_1min_bar = self.bar_1min.get_latest_timestamp(symbol, trading_date)
        latest_tick = self.ticks.get_latest_timestamp(symbol, trading_date)  # 新增
        incomplete_bar = self.bar_1min.get_incomplete_bar(symbol, trading_date)

        return {
            "latest_15min_timestamp": latest_15min,
            "latest_1min_bar_timestamp": latest_1min_bar,
            "latest_tick_timestamp": latest_tick,  # 新增
            "incomplete_bar": (
                incomplete_bar.to_dict() if incomplete_bar is not None else None
            ),
        }
