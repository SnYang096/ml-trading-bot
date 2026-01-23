"""
数据下载辅助模块

封装BinanceMultiSymbolDownloader用于测试，支持下载指定日期范围的tick数据
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    from src.data_tools.download_training_data import BinanceMultiSymbolDownloader

    DOWNLOADER_AVAILABLE = True
except ImportError:
    DOWNLOADER_AVAILABLE = False
    BinanceMultiSymbolDownloader = None


class TestDataDownloader:
    """
    测试数据下载器

    封装BinanceMultiSymbolDownloader用于测试
    """

    def __init__(
        self,
        data_dir: str = "data/test_raw",
        parquet_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir: 原始数据下载目录
            parquet_dir: Parquet输出目录（如果为None，不转换）
        """
        if not DOWNLOADER_AVAILABLE:
            raise ImportError("BinanceMultiSymbolDownloader不可用")

        self.downloader = BinanceMultiSymbolDownloader(
            data_dir=data_dir,
            parquet_dir=parquet_dir,
        )
        self.data_dir = Path(data_dir)
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None

    def download_date_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> bool:
        """
        下载指定日期范围的tick数据

        Args:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            是否成功
        """
        # 生成月份列表
        months = self._get_month_list(start_date, end_date)

        # 下载每个月的文件
        success_count = 0
        for year, month in months:
            try:
                result = self.downloader.download_symbol_data(
                    symbol=symbol,
                    months=[(year, month)],
                )
                if result.get("success", False):
                    success_count += 1
                    print(f"✅ 下载成功: {symbol} {year}-{month:02d}")
                else:
                    print(f"⚠️ 下载失败: {symbol} {year}-{month:02d}")
            except Exception as e:
                print(f"❌ 下载异常: {symbol} {year}-{month:02d}: {e}")

        return success_count > 0

    def download_days(
        self,
        symbol: str,
        days: int,
        end_date: Optional[datetime] = None,
    ) -> bool:
        """
        下载最近N天的数据

        Args:
            symbol: 交易对符号
            days: 天数
            end_date: 结束日期（如果为None，使用今天）

        Returns:
            是否成功
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days)

        return self.download_date_range(symbol, start_date, end_date)

    def _get_month_list(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[int, int]]:
        """
        生成月份列表

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            [(year, month), ...] 列表
        """
        months = []
        current = start_date.replace(day=1)

        while current <= end_date:
            months.append((current.year, current.month))
            # 下一个月
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return months

    def check_data_exists(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> bool:
        """
        检查数据文件是否存在

        Args:
            symbol: 交易对符号
            year: 年份
            month: 月份

        Returns:
            是否存在
        """
        # 检查ZIP文件
        zip_path = self.data_dir / f"{symbol}-aggTrades-{year}-{month:02d}.zip"
        if zip_path.exists():
            return True

        # 检查Parquet文件
        if self.parquet_dir:
            parquet_path = self.parquet_dir / f"{symbol}_{year}-{month:02d}.parquet"
            if parquet_path.exists():
                return True

        return False

    def cleanup_test_data(
        self,
        symbol: Optional[str] = None,
        keep_recent_days: int = 0,
    ) -> None:
        """
        清理测试数据

        Args:
            symbol: 交易对符号（如果为None，清理所有）
            keep_recent_days: 保留最近N天的数据（0表示全部清理）
        """
        if keep_recent_days > 0:
            cutoff_date = datetime.now() - timedelta(days=keep_recent_days)
        else:
            cutoff_date = None

        # 清理ZIP文件
        for zip_file in self.data_dir.glob("*.zip"):
            if symbol and symbol not in zip_file.name:
                continue

            if cutoff_date:
                # 从文件名提取日期
                # 格式: {symbol}-aggTrades-{year}-{month}.zip
                parts = zip_file.stem.split("-")
                if len(parts) >= 4:
                    try:
                        year = int(parts[-2])
                        month = int(parts[-1])
                        file_date = datetime(year, month, 1)
                        if file_date >= cutoff_date:
                            continue
                    except ValueError:
                        pass

            zip_file.unlink()
            print(f"🗑️  删除: {zip_file}")

        # 清理Parquet文件
        if self.parquet_dir:
            for parquet_file in self.parquet_dir.glob("*.parquet"):
                if symbol and symbol not in parquet_file.name:
                    continue

                if cutoff_date:
                    # 从文件名提取日期
                    # 格式: {symbol}_{year}-{month}.parquet
                    parts = parquet_file.stem.split("_")
                    if len(parts) >= 2:
                        try:
                            date_part = parts[-1]
                            year, month = map(int, date_part.split("-"))
                            file_date = datetime(year, month, 1)
                            if file_date >= cutoff_date:
                                continue
                        except ValueError:
                            pass

                parquet_file.unlink()
                print(f"🗑️  删除: {parquet_file}")
