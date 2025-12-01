"""
币安 Tick 数据下载器

从币安官方数据下载页面 (https://data.binance.vision/) 自动下载缺失的 tick 数据。

注意：
- 币安提供 WebSocket 实时 tick 数据流（aggTrades stream）
- 但历史 tick 数据只能从官方数据下载页面获取
- 本工具用于补全历史数据或实时流中断后的数据缺失
"""

from __future__ import annotations

import os
import re
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import pandas as pd

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not installed. Install with: pip install requests")

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    # BeautifulSoup 目前未使用，但保留以备将来使用


class BinanceTickDataDownloader:
    """
    币安 Tick 数据下载器

    从 https://data.binance.vision/ 下载 aggTrades 数据
    """

    BASE_URL = "https://data.binance.vision"

    def __init__(
        self,
        symbol: str,
        market_type: str = "futures",  # "spot" 或 "futures"
        contract_type: str = "um",  # "um" (USDT-M) 或 "cm" (COIN-M)
        download_dir: str = "data/downloads",
    ):
        """
        Args:
            symbol: 交易对符号（如 "BTCUSDT"）
            market_type: 市场类型（"spot" 或 "futures"）
            contract_type: 合约类型（"um" 或 "cm"），仅用于 futures
            download_dir: 下载目录
        """
        self.symbol = symbol.upper()
        self.market_type = market_type
        self.contract_type = contract_type
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def get_data_url(self, year: int, month: int) -> str:
        """
        构建数据下载 URL

        Args:
            year: 年份
            month: 月份（1-12）

        Returns:
            数据文件 URL
        """
        if self.market_type == "spot":
            path = f"data/spot/daily/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{year}-{month:02d}-01.zip"
        else:  # futures
            path = f"data/futures/{self.contract_type}/daily/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{year}-{month:02d}-01.zip"

        return urljoin(self.BASE_URL, path)

    def list_available_files(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Tuple[int, int, str]]:
        """
        列出可用的数据文件

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            [(year, month, url), ...] 列表
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        files = []
        current = start_date.replace(day=1)

        while current <= end_date:
            year = current.year
            month = current.month
            url = self.get_data_url(year, month)
            files.append((year, month, url))

            # 下一个月
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        return files

    def download_file(
        self,
        year: int,
        month: int,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> Optional[Path]:
        """
        下载指定月份的数据文件

        Args:
            year: 年份
            month: 月份
            max_retries: 最大重试次数
            timeout: 超时时间（秒）

        Returns:
            下载的文件路径，如果失败返回 None
        """
        url = self.get_data_url(year, month)
        filename = f"{self.symbol}-aggTrades-{year}-{month:02d}-01.zip"
        filepath = self.download_dir / filename

        # 如果文件已存在，直接返回
        if filepath.exists():
            print(f"✅ 文件已存在: {filepath}")
            return filepath

        if not REQUESTS_AVAILABLE:
            print("❌ requests 未安装，无法下载数据")
            return None

        for attempt in range(max_retries):
            try:
                print(f"📥 下载 {filename} (尝试 {attempt + 1}/{max_retries})...")

                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                # 检查文件大小
                content_length = response.headers.get("Content-Length")
                if content_length:
                    file_size = int(content_length)
                    if file_size < 1000:  # 文件太小，可能是错误页面
                        print(f"⚠️ 文件大小异常: {file_size} bytes")
                        return None

                # 下载文件
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # 验证 ZIP 文件
                try:
                    with zipfile.ZipFile(filepath, "r") as zf:
                        if len(zf.namelist()) == 0:
                            print(f"⚠️ ZIP 文件为空")
                            filepath.unlink()
                            return None
                except zipfile.BadZipFile:
                    print(f"⚠️ 无效的 ZIP 文件")
                    filepath.unlink()
                    return None

                print(f"✅ 下载完成: {filepath}")
                return filepath

            except requests.exceptions.RequestException as e:
                print(f"⚠️ 下载失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 指数退避
                else:
                    print(f"❌ 下载失败，已重试 {max_retries} 次")
                    return None

        return None

    def download_missing_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        existing_files: Optional[List[Path]] = None,
    ) -> List[Path]:
        """
        下载缺失时间段的数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            existing_files: 已存在的文件列表（用于跳过已下载的文件）

        Returns:
            下载的文件路径列表
        """
        if existing_files is None:
            existing_files = []

        # 列出需要下载的文件
        required_files = self.list_available_files(start_date, end_date)

        downloaded = []
        for year, month, url in required_files:
            filename = f"{self.symbol}-aggTrades-{year}-{month:02d}-01.zip"
            filepath = self.download_dir / filename

            # 检查是否已存在
            if filepath in existing_files or filepath.exists():
                print(f"⏭️  跳过已存在的文件: {filename}")
                continue

            # 下载文件
            downloaded_file = self.download_file(year, month)
            if downloaded_file:
                downloaded.append(downloaded_file)

            # 避免请求过快
            time.sleep(1)

        return downloaded

    def extract_and_convert_to_parquet(
        self,
        zip_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        解压 ZIP 文件并转换为 Parquet 格式

        Args:
            zip_path: ZIP 文件路径
            output_dir: 输出目录（如果为 None，使用 download_dir）

        Returns:
            Parquet 文件路径
        """
        if output_dir is None:
            output_dir = self.download_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # 找到 CSV 文件
                csv_name = None
                for name in zf.namelist():
                    if name.endswith(".csv"):
                        csv_name = name
                        break

                if csv_name is None:
                    print(f"⚠️ ZIP 文件中没有找到 CSV 文件: {zip_path}")
                    return None

                # 读取 CSV
                with zf.open(csv_name) as handle:
                    first_line = handle.readline().decode("utf-8", errors="ignore")

                read_params = {"low_memory": False}
                if first_line.strip().split(",")[0].replace(".", "").isdigit():
                    read_params.update(
                        {
                            "header": None,
                            "names": [
                                "agg_trade_id",
                                "price",
                                "quantity",
                                "first_trade_id",
                                "last_trade_id",
                                "transact_time",
                                "is_buyer_maker",
                            ],
                        }
                    )

                with zf.open(csv_name) as handle:
                    df = pd.read_csv(handle, **read_params)

                # 转换为标准格式
                ticks_df = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(df["transact_time"], unit="ms"),
                        "price": pd.to_numeric(df["price"], errors="coerce"),
                        "volume": pd.to_numeric(
                            df.get("quantity", df.get("volume")), errors="coerce"
                        ),
                    }
                ).dropna()

                # 添加 side 字段
                if "is_buyer_maker" in df.columns:
                    sides = pd.Series(
                        [-1 if x else 1 for x in df["is_buyer_maker"].astype(bool)]
                    )
                else:
                    sides = pd.Series([1] * len(ticks_df))

                ticks_df["side"] = sides.values[: len(ticks_df)]

                # 保存为 Parquet
                parquet_path = (
                    output_dir
                    / f"{self.symbol}_{zip_path.stem.replace('-01', '')}.parquet"
                )
                ticks_df.to_parquet(parquet_path, index=False)

                print(f"✅ 转换为 Parquet: {parquet_path} ({len(ticks_df)} 条)")
                return parquet_path

        except Exception as e:
            print(f"❌ 解压和转换失败: {e}")
            return None


class TickDataGapFiller:
    """
    Tick 数据缺失补全器

    专门用于补全 tick 数据的缺失部分
    """

    def __init__(
        self,
        symbol: str,
        market_type: str = "futures",
        contract_type: str = "um",
        download_dir: str = "data/downloads",
        parquet_dir: str = "data/parquet_data",
    ):
        """
        Args:
            symbol: 交易对符号
            market_type: 市场类型
            contract_type: 合约类型
            download_dir: 下载目录
            parquet_dir: Parquet 数据目录
        """
        self.symbol = symbol
        self.downloader = BinanceTickDataDownloader(
            symbol=symbol,
            market_type=market_type,
            contract_type=contract_type,
            download_dir=download_dir,
        )
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    def detect_missing_periods(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[int, int]]:
        """
        检测缺失的时间段

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            [(year, month), ...] 缺失的月份列表
        """
        # 检查已存在的 Parquet 文件
        existing_files = set()
        for parquet_file in self.parquet_dir.glob(f"{self.symbol}_*.parquet"):
            # 从文件名提取年月：BTCUSDT_2024-01.parquet
            match = re.search(r"(\d{4})-(\d{2})", parquet_file.stem)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                existing_files.add((year, month))

        # 列出需要的文件
        required_files = self.downloader.list_available_files(start_date, end_date)

        # 找出缺失的
        missing = []
        for year, month, _ in required_files:
            if (year, month) not in existing_files:
                missing.append((year, month))

        return missing

    def fill_missing_data(
        self,
        start_date: datetime,
        end_date: datetime,
        auto_convert: bool = True,
    ) -> List[Path]:
        """
        补全缺失的数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            auto_convert: 是否自动转换为 Parquet

        Returns:
            下载/转换的文件路径列表
        """
        # 检测缺失的月份
        missing_periods = self.detect_missing_periods(start_date, end_date)

        if not missing_periods:
            print("✅ 没有缺失的数据")
            return []

        print(f"📥 发现 {len(missing_periods)} 个月份的数据缺失，开始下载...")

        downloaded_files = []
        for year, month in missing_periods:
            # 下载 ZIP 文件
            zip_path = self.downloader.download_file(year, month)
            if zip_path:
                downloaded_files.append(zip_path)

                # 转换为 Parquet
                if auto_convert:
                    parquet_path = self.downloader.extract_and_convert_to_parquet(
                        zip_path,
                        output_dir=self.parquet_dir,
                    )
                    if parquet_path:
                        downloaded_files.append(parquet_path)

            # 避免请求过快
            time.sleep(1)

        return downloaded_files


# 使用示例
if __name__ == "__main__":
    # 创建下载器
    downloader = BinanceTickDataDownloader(
        symbol="BTCUSDT",
        market_type="futures",
        contract_type="um",
    )

    # 下载最近一个月的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # 下载缺失的数据
    tick_filler = TickDataGapFiller(
        symbol="BTCUSDT",
        market_type="futures",
        contract_type="um",
    )

    files = tick_filler.fill_missing_data(start_date, end_date)
    print(f"✅ 完成，共处理 {len(files)} 个文件")
